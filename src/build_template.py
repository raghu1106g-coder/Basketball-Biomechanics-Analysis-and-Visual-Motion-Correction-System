"""
Module 4 — Expert Template Builder
====================================
Build a normalised (time-warped to N points) mean joint-angle trajectory
with per-joint standard-deviation bands from reference/expert shots.

Output: data/expert_template.csv with columns:
    norm_frame, {joint}_mean, {joint}_std  for each tracked joint

CLI usage:
    python -m src.build_template --shots shooter01_side_3 shooter01_side_4
    python -m src.build_template --auto --metadata metadata.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from src.utils import (
    load_config, load_landmarks, angle_series, normalize_timeseries,
    SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE,
)

# Joints to include in the template
TEMPLATE_JOINTS = {
    "elbow":        (SHOULDER, ELBOW, WRIST),
    "knee":         (HIP, KNEE, ANKLE),
    "wrist":        (ELBOW, WRIST, 20),       # 20 = right index finger
    "shoulder_abd": (HIP, SHOULDER, ELBOW),
}


def build_template(landmark_paths, config=None, output_path="data/expert_template.csv"):
    """
    Build a normalised template from a list of reference landmark files.

    Parameters
    ----------
    landmark_paths : list of str — paths to CSV or NPY files
    config : dict or None
    output_path : str

    Returns
    -------
    DataFrame with norm_frame + per-joint mean/std columns
    """
    if config is None:
        config = load_config()

    norm_len = config["dtw"]["normalize_length"]

    # Collect per-joint time-series from all reference shots
    all_series = {joint: [] for joint in TEMPLATE_JOINTS}

    for path in landmark_paths:
        landmarks = load_landmarks(path)
        for joint_name, (a, b, c) in TEMPLATE_JOINTS.items():
            ts = angle_series(landmarks, a, b, c)
            normed = normalize_timeseries(ts, norm_len)
            all_series[joint_name].append(normed)

    if not landmark_paths:
        print("WARNING: No reference shots provided. Template will be empty.")
        return pd.DataFrame()

    # Build the template dataframe
    records = []
    for i in range(norm_len):
        row = {"norm_frame": i}
        for joint_name in TEMPLATE_JOINTS:
            values = np.array([s[i] for s in all_series[joint_name]])
            row["{}_mean".format(joint_name)] = round(float(np.mean(values)), 4)
            row["{}_std".format(joint_name)]  = round(float(np.std(values)), 4)
        records.append(row)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Template saved to {} ({} reference shots, {} points)".format(
        output_path, len(landmark_paths), norm_len))
    return df


def auto_select_reference_shots(metadata_path, landmark_dir, config=None):
    """
    Auto-select reference shots from metadata based on quality criteria.
    Uses made shots with good arc quality as template candidates.

    Returns list of landmark file paths.
    """
    if config is None:
        config = load_config()

    template_cfg = config.get("template", {}).get("auto_select_criteria", {})
    require_made = template_cfg.get("made_shot", True)
    min_quality = template_cfg.get("min_arc_quality", "good")

    quality_rank = {"good": 3, "moderate": 2, "low": 1, "bad": 0, "high": 2}

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.dropna(subset=["shooter_id", "view", "shot_num", "filepath"])

    selected = []
    for _, row in metadata.iterrows():
        # Filter by made_shot
        if require_made and str(row.get("made_shot", "")).strip().lower() != "made":
            continue

        # Filter by arc quality
        arc = str(row.get("arc_quality", "")).strip().lower()
        if quality_rank.get(arc, 0) < quality_rank.get(min_quality, 3):
            continue

        shooter_id = str(row["shooter_id"]).strip()
        view = str(row["view"]).strip()
        shot_num = str(int(float(row["shot_num"])))

        # Find landmark file (CSV or NPY)
        csv_path = os.path.join(landmark_dir, "{}_{}_{}.csv".format(shooter_id, view, shot_num))
        npy_path = os.path.join("processed", "{}_{}_{}.npy".format(shooter_id, view, shot_num))

        if os.path.exists(csv_path):
            selected.append(csv_path)
        elif os.path.exists(npy_path):
            selected.append(npy_path)
        else:
            print("Reference shot {}/{} — no landmark file found".format(shooter_id, shot_num))

    print("Auto-selected {} reference shots".format(len(selected)))
    for s in selected:
        print("  {}".format(s))

    # Fallback: if nothing matches strict criteria, use all available shots
    if not selected:
        print("WARNING: No shots matched criteria. Falling back to ALL available shots.")
        for _, row in metadata.iterrows():
            shooter_id = str(row["shooter_id"]).strip()
            view = str(row["view"]).strip()
            shot_num = str(int(float(row["shot_num"])))
            csv_path = os.path.join(landmark_dir, "{}_{}_{}.csv".format(shooter_id, view, shot_num))
            npy_path = os.path.join("processed", "{}_{}_{}.npy".format(shooter_id, view, shot_num))
            if os.path.exists(csv_path):
                selected.append(csv_path)
            elif os.path.exists(npy_path):
                selected.append(npy_path)

    return selected


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build expert template from reference shots.")
    parser.add_argument("--shots", nargs="+", default=None,
                        help="Explicit list of landmark file paths to use as reference.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select from metadata quality columns.")
    parser.add_argument("--metadata", type=str, default="metadata.csv",
                        help="Metadata CSV (used with --auto).")
    parser.add_argument("--landmark-dir", type=str, default="data/processed/landmarks",
                        help="Directory containing landmark CSVs.")
    parser.add_argument("--output", type=str, default="data/expert_template.csv")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    if args.shots:
        paths = args.shots
    elif args.auto:
        paths = auto_select_reference_shots(args.metadata, args.landmark_dir, config)
    else:
        parser.print_help()
        sys.exit(1)

    if not paths:
        print("ERROR: No reference shots found.")
        sys.exit(1)

    df = build_template(paths, config, args.output)
    print("\nTemplate preview (first 5 rows):")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
