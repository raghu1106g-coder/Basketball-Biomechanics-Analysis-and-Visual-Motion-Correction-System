"""Range Check Module: Functional range-based fault detection.

Reads features and reference_ranges.csv, flagging values outside functional bands
as faults. Style variations inside ranges are not flagged.
Saves results to <dir>/range_check_results.csv.
"""
import argparse
import os
import pandas as pd


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

    out_path = os.path.join(data_dir, "range_check_results.csv")
    return {
        "dir": data_dir,
        "is_synthetic": is_synthetic,
        "features": features_path,
        "ranges": ranges_path,
        "out": out_path,
    }


def run_range_check(data_dir="."):
    paths = resolve_paths(data_dir)
    if not os.path.exists(paths["features"]):
        raise FileNotFoundError(f"Features file not found: {paths['features']}")
    if not os.path.exists(paths["ranges"]):
        raise FileNotFoundError(f"Reference ranges file not found: {paths['ranges']}")

    df = pd.read_csv(paths["features"])
    ref = pd.read_csv(paths["ranges"]).set_index("feature")

    if paths["is_synthetic"]:
        print("=" * 70)
        print("SYNTHETIC DATA - NOT EVIDENCE OF REAL-WORLD ACCURACY")
        print("=" * 70)

    print(f"Running range check on: {paths['features']}")
    print(f"Using reference ranges: {paths['ranges']}")

    fault_cols = []
    fault_summaries = []

    # Check each feature in reference ranges
    for feature, ref_row in ref.iterrows():
        if feature not in df.columns:
            continue
        lo = float(ref_row["functional_min"])
        hi = float(ref_row["functional_max"])
        fault_col = f"{feature}_fault"
        fault_cols.append(fault_col)

        vals = pd.to_numeric(df[feature], errors="coerce")
        df[fault_col] = (vals < lo) | (vals > hi)

    # Build per-shot fault summary
    for _, row in df.iterrows():
        shot_faults = []
        for feature, ref_row in ref.iterrows():
            if feature not in df.columns:
                continue
            lo = float(ref_row["functional_min"])
            hi = float(ref_row["functional_max"])
            v = row[feature]
            if pd.isna(v):
                continue
            if v < lo:
                shot_faults.append(f"{feature} low ({v:.1f} < {lo:g})")
            elif v > hi:
                shot_faults.append(f"{feature} high ({v:.1f} > {hi:g})")
        summary_str = "; ".join(shot_faults) if shot_faults else "none"
        fault_summaries.append(summary_str)

    df["fault_summary"] = fault_summaries

    df.to_csv(paths["out"], index=False)
    print(f"Saved range check results to: {paths['out']}")

    # Print summary statistics
    total_shots = len(df)
    shots_with_fault = (df["fault_summary"] != "none").sum()
    print(f"\nTotal shots evaluated: {total_shots}")
    print(f"Shots with at least one fault: {shots_with_fault} ({shots_with_fault / total_shots:.1%})")
    print("\nFault frequency per feature:")
    for fc in fault_cols:
        feat_name = fc.replace("_fault", "")
        count = df[fc].sum()
        pct = count / total_shots * 100
        lo = ref.loc[feat_name, "functional_min"]
        hi = ref.loc[feat_name, "functional_max"]
        print(f"  {feat_name:22s} [{lo:g}-{hi:g}]: {count:3d}/{total_shots} ({pct:5.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Range-based fault detection")
    parser.add_argument("data_dir", nargs="?", default=".", help="Data directory (. or synthetic/)")
    parser.add_argument("--dir", dest="dir_flag", default=None, help="Alternative flag for data directory")
    args = parser.parse_args()

    data_dir = args.dir_flag if args.dir_flag else args.data_dir
    run_range_check(data_dir)


if __name__ == "__main__":
    main()
