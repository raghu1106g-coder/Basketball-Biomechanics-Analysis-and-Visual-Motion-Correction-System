#!/usr/bin/env python3
"""Feedback-logic demo on SYNTHETIC data (not real accuracy evidence).

Picks one synthetic shot and prints plain-language corrections from:
  (a) reference_ranges.csv           -> only when a value is OUTSIDE the functional range
  (b) personal_best_templates.csv    -> only when the shot drifts from the shooter's own best form
Inside range / close to own best -> no correction printed.

Usage: python feedback_demo.py --dir synthetic/ [--shooter shooter103 --shot 4] [--seed 7]
"""
import argparse
import os

import numpy as np
import pandas as pd

# feature -> (message if BELOW range, message if ABOVE range, unit, joint used for reliability)
RANGE_MSG = {
    "elbow_angle_load":     ("Elbow load is too tight", "Elbow load is too open", "deg", "elbow"),
    "elbow_angle_release":  ("Elbow is under-extended at release", "Elbow is over-extended at release", "deg", "elbow"),
    "knee_flex_min":        ("Knee bend is too deep", "Knee bend is shallow", "deg", "knee"),
    "knee_angle_release":   ("Legs are not fully extended at release", "Legs are over-extended at release", "deg", "knee"),
    "followthrough_frames": ("Follow-through is too short", "Follow-through is held too long", "frames", "elbow"),
}
# feature -> (message if shot is LOWER than best, message if HIGHER than best); {d} = size of gap
BEST_MSG = {
    "elbow_angle_load":     ("Elbow load is {d} deg tighter than in your best shots",
                             "Elbow load is {d} deg more open than in your best shots"),
    "elbow_angle_release":  ("Elbow dropped {d} deg compared to your best shots",
                             "Elbow extended {d} deg more at release than in your best shots"),
    "knee_flex_min":        ("Knee bend is {d} deg deeper than in your best shots",
                             "Knee bend is {d} deg shallower than in your best shots"),
    "knee_angle_release":   ("Legs finish {d} deg less extended than in your best shots",
                             "Legs finish {d} deg more extended than in your best shots"),
    "followthrough_frames": ("Follow-through is {d} frames shorter than in your best shots",
                             "Follow-through is {d} frames longer than in your best shots"),
}
TOL = {"elbow_angle_load": 8, "elbow_angle_release": 6, "knee_flex_min": 6,
       "knee_angle_release": 5, "followthrough_frames": 3}   # drift tolerated vs own best
MIN_RELIABLE = 70   # below this tracking %, don't trust that joint's numbers


def resolve_paths(data_dir="."):
    data_dir = os.path.normpath(data_dir)
    is_synthetic = (
        os.path.exists(os.path.join(data_dir, "synthetic_features.csv"))
        or os.path.basename(data_dir).lower() == "synthetic"
    )
    if is_synthetic:
        features_path = os.path.join(data_dir, "synthetic_features.csv")
    else:
        features_path = os.path.join(data_dir, "features.csv")

    ranges_path = os.path.join(data_dir, "reference_ranges.csv")
    if not os.path.exists(ranges_path):
        ranges_path = "reference_ranges.csv"

    best_path = os.path.join(data_dir, "personal_best_templates.csv")

    return {
        "dir": data_dir,
        "is_synthetic": is_synthetic,
        "features": features_path,
        "ranges": ranges_path,
        "best": best_path,
    }


def main():
    ap = argparse.ArgumentParser(description="Basketball biomechanics feedback demo")
    ap.add_argument("data_dir", nargs="?", default=".", help="Data directory (. or synthetic/)")
    ap.add_argument("--dir", dest="dir_flag", default=None, help="Alternative flag for data directory")
    ap.add_argument("--shooter")
    ap.add_argument("--shot", type=int)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    data_dir = a.dir_flag if a.dir_flag else a.data_dir
    paths = resolve_paths(data_dir)

    if not os.path.exists(paths["features"]):
        raise FileNotFoundError(f"Features file not found: {paths['features']}")
    if not os.path.exists(paths["ranges"]):
        raise FileNotFoundError(f"Reference ranges not found: {paths['ranges']}")

    feats = pd.read_csv(paths["features"])
    ref = pd.read_csv(paths["ranges"]).set_index("feature")
    best = None
    if os.path.exists(paths["best"]):
        best = pd.read_csv(paths["best"]).set_index("shooter_id")

    if a.shooter and a.shot:
        matched = feats[(feats.shooter_id == a.shooter) & (feats.shot_num == a.shot)]
        if matched.empty:
            raise ValueError(f"No shot found for {a.shooter} shot {a.shot}")
        row = matched.iloc[0]
    else:
        row = feats.iloc[int(np.random.default_rng(a.seed).integers(len(feats)))]
    sid = row.shooter_id
    label_prefix = "[SYNTHETIC DATA]" if paths["is_synthetic"] else "[REAL DATA]"
    print(f"{label_prefix} {sid}, shot {int(row.shot_num)}")

    # joints with unreliable tracking: skip their corrections instead of coaching on noise
    bad = {j for j in ("elbow", "knee") if row[f"{j}_reliable_pct"] < MIN_RELIABLE}
    for j in sorted(bad):
        print(f"  Note: {j} tracking was only {row[f'{j}_reliable_pct']:.0f}% reliable - skipping {j}-based feedback.")

    print("\n(a) Compared with reference ranges (assumed, not literature):")
    n = 0
    for f, (low_msg, high_msg, unit, joint) in RANGE_MSG.items():
        if joint in bad or f not in ref.index:
            continue
        v, lo, hi = row[f], ref.loc[f, "functional_min"], ref.loc[f, "functional_max"]
        if v < lo or v > hi:
            msg = low_msg if v < lo else high_msg
            print(f"  - {msg} ({v:.0f} {unit}; target {lo:g}-{hi:g})")
            n += 1
    if n == 0:
        print("  No correction needed - all measured values are inside the functional ranges.")

    if best is not None and sid in best.index:
        print("\n(b) Compared with your own best shots:")
        n = 0
        tmpl = best.loc[sid]
        for f, (lower_msg, higher_msg) in BEST_MSG.items():
            if RANGE_MSG[f][3] in bad or f not in tmpl:
                continue
            d = row[f] - tmpl[f]
            if abs(d) > TOL[f]:
                print("  - " + (lower_msg if d < 0 else higher_msg).format(d=f"{abs(d):.0f}"))
                n += 1
        if n == 0:
            print("  No correction needed - this shot is close to your best form.")
        print(f"  (template: {int(tmpl.n_template_shots)} shot(s), basis = {tmpl.template_basis})")
    else:
        print("\n(b) Personal best template comparison skipped (no template found for shooter).")


if __name__ == "__main__":
    main()
