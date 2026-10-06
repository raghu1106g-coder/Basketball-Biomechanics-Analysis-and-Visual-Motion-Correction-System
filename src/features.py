"""
Module 3 — Feature Extraction
===============================
Compute per-shot biomechanical features from landmark data.

Features:
    • Elbow / knee angle time-series
    • Wrist angle at release
    • Shoulder abduction at release
    • Forearm release angle (relative to vertical)
    • Elbow-knee timing offset
    • Follow-through duration
    • Lateral elbow deviation
    • Joint angular velocity & acceleration

CLI usage:
    python -m src.features --landmarks path/to/landmarks.csv
    python -m src.features --batch metadata.csv --output-dir data/processed/landmarks
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from src.utils import (
    load_config, load_landmarks, angle_series, angular_velocity,
    angular_acceleration, normalize_timeseries,
    SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP,
    NUM_LANDMARKS,
)
from src.segment_phases import segment_phases


def compute_features(landmarks, fps=30.0, config=None):
    """
    Compute all biomechanical features for a single shot.

    Parameters
    ----------
    landmarks : ndarray (T, 33, 4)
    fps       : float
    config    : dict or None

    Returns
    -------
    dict of scalar features + nested time-series
    """
    if config is None:
        config = load_config()

    T = landmarks.shape[0]
    vis_thresh = config["extraction"]["visibility_threshold"]

    # ── Angle time-series ──
    elbow_ts = angle_series(landmarks, SHOULDER, ELBOW, WRIST)
    knee_ts  = angle_series(landmarks, HIP, KNEE, ANKLE)

    # Wrist angle: forearm-hand angle (elbow-wrist-index_finger_tip≈wrist+hand)
    # Approximate using landmarks 16 (wrist), 14 (elbow), 20 (index finger)
    wrist_ts = angle_series(landmarks, ELBOW, WRIST, 20)  # 20 = right index

    # Shoulder abduction: angle in frontal plane (hip-shoulder-elbow)
    shoulder_abd_ts = angle_series(landmarks, HIP, SHOULDER, ELBOW)

    # ── Velocities ──
    elbow_vel = angular_velocity(elbow_ts, fps)
    elbow_acc = angular_acceleration(elbow_ts, fps)
    knee_vel  = angular_velocity(knee_ts, fps)

    # ── Phase segmentation ──
    phases = segment_phases(landmarks, fps, config)
    load_frame = phases["load"]
    release_frame = phases["release"]
    end_frame = phases["end"]

    # ── Forearm release angle (relative to vertical) ──
    # Vector from elbow to wrist at release frame
    elbow_pos = landmarks[release_frame, ELBOW, :2]
    wrist_pos = landmarks[release_frame, WRIST, :2]
    forearm_vec = wrist_pos - elbow_pos
    # Angle from vertical (0,1) in normalised coords (y-down)
    vertical = np.array([0.0, 1.0])
    cos_a = np.dot(forearm_vec, vertical) / (np.linalg.norm(forearm_vec) + 1e-8)
    release_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    # ── Elbow-knee timing offset ──
    elbow_min_frame = int(np.argmin(elbow_ts))
    knee_min_frame  = int(np.argmin(knee_ts))
    timing_offset   = elbow_min_frame - knee_min_frame

    # ── Follow-through duration ──
    followthrough_frames = max(0, end_frame - release_frame)

    # ── Lateral elbow deviation (x-distance from shoulder-wrist line) ──
    s_x = landmarks[release_frame, SHOULDER, 0]
    w_x = landmarks[release_frame, WRIST, 0]
    e_x = landmarks[release_frame, ELBOW, 0]
    # Deviation = elbow x minus midpoint of shoulder-wrist x
    lateral_dev = float(e_x - (s_x + w_x) / 2.0)

    # ── Reliability ──
    knee_reliable  = float((landmarks[:, KNEE, 3] > vis_thresh).mean() * 100)
    elbow_reliable = float((landmarks[:, ELBOW, 3] > vis_thresh).mean() * 100)

    return {
        # Scalar features
        "elbow_angle_load":         round(float(elbow_ts[load_frame]), 2),
        "elbow_angle_release":      round(float(elbow_ts[release_frame]), 2),
        "knee_flex_min":            round(float(knee_ts[load_frame]), 2),
        "knee_angle_release":       round(float(knee_ts[release_frame]), 2),
        "wrist_angle_release":      round(float(wrist_ts[release_frame]), 2),
        "shoulder_abduction_release": round(float(shoulder_abd_ts[release_frame]), 2),
        "release_angle":            round(release_angle, 2),
        "elbow_knee_timing_offset": int(timing_offset),
        "followthrough_frames":     int(followthrough_frames),
        "lateral_elbow_deviation":  round(lateral_dev, 4),
        "elbow_max_angular_vel":    round(float(np.max(np.abs(elbow_vel))), 2),
        "elbow_max_angular_acc":    round(float(np.max(np.abs(elbow_acc))), 2),
        "knee_max_angular_vel":     round(float(np.max(np.abs(knee_vel))), 2),
        "knee_reliable_pct":        round(knee_reliable, 1),
        "elbow_reliable_pct":       round(elbow_reliable, 1),
        # Phase boundaries
        "load_frame":               load_frame,
        "release_frame":            release_frame,
        # Time-series (for template building)
        "elbow_angle_ts":           elbow_ts.tolist(),
        "knee_flexion_ts":          knee_ts.tolist(),
        "wrist_angle_ts":           wrist_ts.tolist(),
        "shoulder_abd_ts":          shoulder_abd_ts.tolist(),
    }


def features_to_scalar_row(features, shooter_id="", shot_num=0):
    """Extract only scalar features suitable for a CSV row."""
    scalar_keys = [
        "elbow_angle_load", "elbow_angle_release", "knee_flex_min",
        "knee_angle_release", "wrist_angle_release", "shoulder_abduction_release",
        "release_angle", "elbow_knee_timing_offset", "followthrough_frames",
        "lateral_elbow_deviation", "elbow_max_angular_vel",
        "elbow_max_angular_acc", "knee_max_angular_vel",
        "knee_reliable_pct", "elbow_reliable_pct",
        "load_frame", "release_frame",
    ]
    row = {k: features[k] for k in scalar_keys}
    row["shooter_id"] = shooter_id
    row["shot_num"] = shot_num
    return row


# ── Batch processing ───────────────────────────────────────────────────────

def process_batch(metadata_path, landmark_dir="data/processed/landmarks",
                  output_csv="data/features.csv", config=None):
    """
    Compute features for all shots referenced in metadata.
    Falls back to legacy .npy files if CSV landmarks aren't found.
    """
    if config is None:
        config = load_config()

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.dropna(subset=["shooter_id", "view", "shot_num", "filepath"])

    # Load existing features if present
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_keys = set(zip(existing["shooter_id"], existing["shot_num"].astype(int)))
    else:
        existing = pd.DataFrame()
        done_keys = set()

    new_rows = []
    for _, row in metadata.iterrows():
        shooter_id = str(row["shooter_id"]).strip()
        shot_num = int(float(row["shot_num"]))
        view = str(row["view"]).strip()

        if (shooter_id, shot_num) in done_keys:
            continue

        # Try CSV first, then legacy .npy
        csv_path = os.path.join(landmark_dir, "{}_{}_{}.csv".format(shooter_id, view, shot_num))
        npy_path = os.path.join("processed", "{}_{}_{}.npy".format(shooter_id, view, shot_num))

        if os.path.exists(csv_path):
            lm_path = csv_path
        elif os.path.exists(npy_path):
            lm_path = npy_path
        else:
            print("Skipping {}/{}: no landmark file found".format(shooter_id, shot_num))
            continue

        landmarks = load_landmarks(lm_path)
        features = compute_features(landmarks, config=config)
        scalar_row = features_to_scalar_row(features, shooter_id, shot_num)
        new_rows.append(scalar_row)

        # Save time-series as JSON sidecar
        ts_dir = os.path.join(os.path.dirname(output_csv), "timeseries")
        os.makedirs(ts_dir, exist_ok=True)
        ts_path = os.path.join(ts_dir, "{}_{}_{}_ts.json".format(shooter_id, view, shot_num))
        ts_data = {
            "elbow_angle_ts": features["elbow_angle_ts"],
            "knee_flexion_ts": features["knee_flexion_ts"],
            "wrist_angle_ts": features["wrist_angle_ts"],
            "shoulder_abd_ts": features["shoulder_abd_ts"],
        }
        with open(ts_path, "w") as f:
            json.dump(ts_data, f)

        print("{}/{}: elbow_release={}, knee_min={}".format(
            shooter_id, shot_num,
            scalar_row["elbow_angle_release"],
            scalar_row["knee_flex_min"],
        ))

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        combined.to_csv(output_csv, index=False)
        print("Added {} rows -> {} now has {} total".format(
            len(new_rows), output_csv, len(combined)))
    else:
        print("No new rows to add.")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract biomechanical features.")
    parser.add_argument("--landmarks", type=str, default=None,
                        help="Single landmark file (CSV or NPY).")
    parser.add_argument("--batch", type=str, default=None,
                        help="Metadata CSV for batch mode.")
    parser.add_argument("--landmark-dir", type=str, default="data/processed/landmarks")
    parser.add_argument("--output", type=str, default="data/features.csv")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    if args.landmarks:
        landmarks = load_landmarks(args.landmarks)
        features = compute_features(landmarks, config=config)
        # Print scalar features
        for k, v in features.items():
            if not isinstance(v, list):
                print("  {:30s}  {}".format(k, v))
    elif args.batch:
        process_batch(args.batch, args.landmark_dir, args.output, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
