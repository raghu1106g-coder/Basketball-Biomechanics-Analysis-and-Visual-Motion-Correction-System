"""
Module 2 — Phase Segmentation
===============================
Segment a basketball shot into four phases:

    Stance → Load → Release → Follow-through

Based on knee-angle minima and wrist/elbow kinematics.

CLI usage:
    python -m src.segment_phases --landmarks path/to/landmarks.csv
"""

import argparse
import json
import sys

import numpy as np
from scipy.signal import find_peaks

from src.utils import (
    load_config, load_landmarks,
    angle_series, angular_velocity,
    SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE,
)


def segment_phases(landmarks, fps=30.0, config=None):
    """
    Identify the four phase boundaries in a shooting motion.

    Parameters
    ----------
    landmarks : ndarray (T, 33, 4)
    fps       : float
    config    : dict or None (loads from rules.yaml)

    Returns
    -------
    dict with keys:
        stance       – start frame index
        load         – frame of deepest knee bend
        release      – frame of ball release
        followthrough – start of follow-through (= release + 1)
        end          – last frame
        phases       – list of (phase_name, start_frame, end_frame)
    """
    if config is None:
        config = load_config()
    phase_cfg = config["phases"]
    T = landmarks.shape[0]

    # ── Knee angle series (hip-knee-ankle) ──
    knee_angles = angle_series(landmarks, HIP, KNEE, ANKLE)

    # ── Elbow angle series (shoulder-elbow-wrist) ──
    elbow_angles = angle_series(landmarks, SHOULDER, ELBOW, WRIST)
    elbow_vel = angular_velocity(elbow_angles, fps)

    # ── 1. Stance: first frame with valid detection ──
    stance_frame = 0
    for i in range(T):
        if not np.isnan(landmarks[i, KNEE, 0]):
            stance_frame = i
            break

    # ── 2. Load: frame of minimum knee angle (deepest bend) ──
    min_frame = int(phase_cfg.get("min_load_frame", 3))
    search_start = max(stance_frame, min_frame)
    if search_start >= T:
        search_start = stance_frame
    load_frame = search_start + int(np.argmin(knee_angles[search_start:]))

    # ── 3. Release: post-load frame where wrist rises above shoulder
    #       AND elbow extension velocity peaks ──
    release_frame = T - 1  # default: last frame

    wrist_y = landmarks[:, WRIST, 1]
    shoulder_y = landmarks[:, SHOULDER, 1]
    margin = phase_cfg.get("wrist_above_shoulder_margin", 0.02)

    # Find peak elbow extension velocity after load
    post_load_vel = elbow_vel[load_frame:]
    vel_peaks, _ = find_peaks(
        post_load_vel,
        prominence=phase_cfg.get("knee_min_prominence", 10),
        distance=phase_cfg.get("knee_min_distance", 5),
    )

    # Candidate: wrist above shoulder after load
    for i in range(load_frame + 1, T):
        wrist_above = wrist_y[i] < (shoulder_y[i] - margin)
        if wrist_above:
            release_frame = i
            break

    # Refine: if elbow velocity peak is near the wrist-above frame, prefer it
    if len(vel_peaks) > 0:
        abs_peaks = vel_peaks + load_frame
        # Pick the velocity peak closest to the wrist-based release
        best_peak = abs_peaks[np.argmin(np.abs(abs_peaks - release_frame))]
        # Use it if within ±5 frames
        if abs(best_peak - release_frame) <= 5:
            release_frame = int(best_peak)

    # Clamp
    release_frame = max(load_frame + 1, min(release_frame, T - 1))

    # ── 4. Follow-through: release to end ──
    followthrough_frame = min(release_frame + 1, T - 1)

    phases = [
        ("stance",        stance_frame,        load_frame - 1),
        ("load",          load_frame,          release_frame - 1),
        ("release",       release_frame,       release_frame),
        ("followthrough", followthrough_frame, T - 1),
    ]

    return {
        "stance": stance_frame,
        "load": load_frame,
        "release": release_frame,
        "followthrough": followthrough_frame,
        "end": T - 1,
        "phases": phases,
        "knee_angle_at_load": float(knee_angles[load_frame]),
        "elbow_angle_at_release": float(elbow_angles[release_frame]),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segment a basketball shot into phases."
    )
    parser.add_argument("--landmarks", type=str, required=True,
                        help="Path to landmark CSV or NPY file.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    landmarks = load_landmarks(args.landmarks)
    result = segment_phases(landmarks, config=config)

    print(json.dumps({
        "stance": result["stance"],
        "load": result["load"],
        "release": result["release"],
        "followthrough": result["followthrough"],
        "end": result["end"],
        "knee_angle_at_load": round(result["knee_angle_at_load"], 2),
        "elbow_angle_at_release": round(result["elbow_angle_at_release"], 2),
    }, indent=2))

    for name, start, end in result["phases"]:
        print("  {:15s}  frames {:4d} – {:4d}".format(name, start, end))


if __name__ == "__main__":
    main()
