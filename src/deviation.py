"""
Module 5 — Deviation Analysis
===============================
Compare a test shot against the expert template using:
    1. DTW distance per joint
    2. Rule-based threshold checks from config/rules.yaml

Output: per-joint deviation scores + detected faults with phase, frame range,
        severity.

CLI usage:
    python -m src.deviation --landmarks shot.csv --template data/expert_template.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from src.utils import (
    load_config, load_landmarks, angle_series, normalize_timeseries,
    SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE,
)
from src.segment_phases import segment_phases
from src.build_template import TEMPLATE_JOINTS

# Use fastdtw (pure-python, no compilation needed) — falls back to scipy
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean as _euc
    def dtw_distance(a, b, radius=10):
        dist, _ = fastdtw(a.reshape(-1, 1), b.reshape(-1, 1), radius=radius, dist=_euc)
        return dist
except ImportError:
    # Fallback: simple sum-of-squared-differences (not true DTW but functional)
    def dtw_distance(a, b, radius=10):
        n = min(len(a), len(b))
        return float(np.sqrt(np.sum((a[:n] - b[:n]) ** 2)))


def load_template(template_path):
    """
    Load the expert template CSV.

    Returns
    -------
    dict  {joint_name: {"mean": ndarray, "std": ndarray}}
    """
    df = pd.read_csv(template_path)
    template = {}
    for joint_name in TEMPLATE_JOINTS:
        mean_col = "{}_mean".format(joint_name)
        std_col  = "{}_std".format(joint_name)
        if mean_col in df.columns and std_col in df.columns:
            template[joint_name] = {
                "mean": df[mean_col].values,
                "std":  df[std_col].values,
            }
    return template


def compute_deviation(landmarks, template, fps=30.0, config=None):
    """
    Compute per-joint DTW deviation and rule-based faults.

    Parameters
    ----------
    landmarks : ndarray (T, 33, 4)
    template  : dict from load_template()
    fps       : float
    config    : dict or None

    Returns
    -------
    dict with keys:
        per_joint_dtw     : {joint: float}
        per_joint_zscore  : {joint: float}
        faults            : list of fault dicts
    """
    if config is None:
        config = load_config()

    dtw_cfg   = config["dtw"]
    thresh    = config["thresholds"]
    severity  = config["severity"]
    norm_len  = dtw_cfg["normalize_length"]
    radius    = dtw_cfg["window_size"]

    # Phase segmentation
    phases = segment_phases(landmarks, fps, config)

    # ── Per-joint DTW ──
    dtw_scores = {}
    z_scores = {}
    for joint_name, (a, b, c) in TEMPLATE_JOINTS.items():
        ts = angle_series(landmarks, a, b, c)
        normed = normalize_timeseries(ts, norm_len)

        if joint_name in template:
            tmpl_mean = template[joint_name]["mean"]
            tmpl_std  = template[joint_name]["std"]

            dist = dtw_distance(normed, tmpl_mean, radius)
            dtw_scores[joint_name] = round(dist, 4)

            # Z-score: mean absolute deviation normalised by template std
            mean_std = np.mean(tmpl_std) if np.mean(tmpl_std) > 0 else 1.0
            z = dist / (norm_len * mean_std + 1e-8)
            z_scores[joint_name] = round(z, 4)

    # ── Rule-based threshold checks ──
    faults = []

    def _severity_label(z):
        if z >= severity["severe"]:
            return "severe"
        elif z >= severity["moderate"]:
            return "moderate"
        elif z >= severity["mild"]:
            return "mild"
        return None

    # Elbow at set-point (load phase)
    elbow_ts = angle_series(landmarks, SHOULDER, ELBOW, WRIST)
    elbow_at_load = float(elbow_ts[phases["load"]])
    elbow_range = thresh["elbow_set_point"]
    if elbow_at_load < elbow_range[0] or elbow_at_load > elbow_range[1]:
        faults.append({
            "joint": "elbow",
            "phase": "load",
            "frame_range": [phases["stance"], phases["load"]],
            "description": "Elbow angle {:.1f} deg outside target [{}, {}] at set-point".format(
                elbow_at_load, elbow_range[0], elbow_range[1]),
            "severity": _severity_label(z_scores.get("elbow", 0)) or "moderate",
            "actual_value": round(elbow_at_load, 2),
            "expected_range": elbow_range,
        })

    # Knee flexion at load
    knee_ts = angle_series(landmarks, HIP, KNEE, ANKLE)
    knee_at_load = float(knee_ts[phases["load"]])
    knee_range = thresh["knee_flexion_load"]
    if knee_at_load < knee_range[0] or knee_at_load > knee_range[1]:
        faults.append({
            "joint": "knee",
            "phase": "load",
            "frame_range": [phases["stance"], phases["load"]],
            "description": "Knee flexion {:.1f} deg outside target [{}, {}] at load".format(
                knee_at_load, knee_range[0], knee_range[1]),
            "severity": _severity_label(z_scores.get("knee", 0)) or "moderate",
            "actual_value": round(knee_at_load, 2),
            "expected_range": knee_range,
        })

    # Release angle
    elbow_pos = landmarks[phases["release"], ELBOW, :2]
    wrist_pos = landmarks[phases["release"], WRIST, :2]
    forearm = wrist_pos - elbow_pos
    vertical = np.array([0.0, 1.0])
    cos_a = np.dot(forearm, vertical) / (np.linalg.norm(forearm) + 1e-8)
    release_angle_val = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    release_range = thresh["release_angle"]
    if release_angle_val < release_range[0] or release_angle_val > release_range[1]:
        faults.append({
            "joint": "forearm",
            "phase": "release",
            "frame_range": [phases["release"], phases["release"]],
            "description": "Release angle {:.1f} deg outside target [{}, {}]".format(
                release_angle_val, release_range[0], release_range[1]),
            "severity": "mild",
            "actual_value": round(release_angle_val, 2),
            "expected_range": release_range,
        })

    # Follow-through duration
    ft_frames = phases["end"] - phases["release"]
    ft_min = thresh["followthrough_min_frames"]
    if ft_frames < ft_min:
        faults.append({
            "joint": "wrist",
            "phase": "followthrough",
            "frame_range": [phases["release"], phases["end"]],
            "description": "Follow-through only {} frames (min {})".format(ft_frames, ft_min),
            "severity": "mild",
            "actual_value": ft_frames,
            "expected_range": [ft_min, None],
        })

    # Shoulder abduction
    shoulder_abd = angle_series(landmarks, HIP, SHOULDER, ELBOW)
    abd_at_release = float(shoulder_abd[phases["release"]])
    abd_max = thresh["shoulder_abduction_max"]
    if abd_at_release > abd_max:
        faults.append({
            "joint": "shoulder",
            "phase": "release",
            "frame_range": [phases["load"], phases["release"]],
            "description": "Shoulder abduction {:.1f} deg exceeds max {}".format(
                abd_at_release, abd_max),
            "severity": "moderate",
            "actual_value": round(abd_at_release, 2),
            "expected_range": [0, abd_max],
        })

    return {
        "per_joint_dtw": dtw_scores,
        "per_joint_zscore": z_scores,
        "faults": faults,
        "phases": {
            "stance": phases["stance"],
            "load": phases["load"],
            "release": phases["release"],
            "followthrough": phases["followthrough"],
            "end": phases["end"],
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute deviation from expert template.")
    parser.add_argument("--landmarks", type=str, required=True,
                        help="Path to landmark file (CSV or NPY).")
    parser.add_argument("--template", type=str, default="data/expert_template.csv",
                        help="Path to expert template CSV.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for deviation results.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    landmarks = load_landmarks(args.landmarks)
    template = load_template(args.template)
    result = compute_deviation(landmarks, template, config=config)

    print("\n=== Per-Joint DTW Distances ===")
    for joint, dist in result["per_joint_dtw"].items():
        z = result["per_joint_zscore"].get(joint, "?")
        print("  {:15s}  DTW={:>10.4f}   z={:>8.4f}".format(joint, dist, z))

    print("\n=== Detected Faults ({}) ===".format(len(result["faults"])))
    for f in result["faults"]:
        print("  [{:>8s}] {:10s} @ {:12s}  frames {}-{}".format(
            f["severity"], f["joint"], f["phase"],
            f["frame_range"][0], f["frame_range"][1]))
        print("            {}".format(f["description"]))

    # Save to JSON
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fp:
            json.dump(result, fp, indent=2)
        print("\nSaved to {}".format(args.output))
    else:
        print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
