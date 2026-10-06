"""Deviation Scoring Module (Secondary View: Consistency / Z-Score).

Computes each shot's deviation (z-score) from the shooter's own average form.
Accepts an optional data directory argument (default '.' = real data).
Outputs self_deviation_report.csv into the data directory.
"""
import argparse
import os
import pandas as pd

FEATURE_COLS = [
    "elbow_angle_load",
    "elbow_angle_release",
    "knee_flex_min",
    "knee_angle_release",
    "followthrough_frames",
]


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

    out_path = os.path.join(data_dir, "self_deviation_report.csv")
    return {
        "dir": data_dir,
        "is_synthetic": is_synthetic,
        "features": features_path,
        "out": out_path,
    }


def compute_deviation_scoring(data_dir="."):
    paths = resolve_paths(data_dir)
    if not os.path.exists(paths["features"]):
        raise FileNotFoundError(f"Features file not found: {paths['features']}")

    if paths["is_synthetic"]:
        print("=" * 70)
        print("SYNTHETIC DATA - NOT EVIDENCE OF REAL-WORLD ACCURACY")
        print("=" * 70)

    print(f"Reading features from: {paths['features']}")
    df = pd.read_csv(paths["features"])
    df["shot_num"] = df["shot_num"].astype(int)

    report_rows = []
    n_shooters = df["shooter_id"].nunique()
    print(f"Analyzing {len(df)} shot(s) across {n_shooters} shooter(s)...")

    for sid, group in df.groupby("shooter_id", sort=False):
        mean = group[FEATURE_COLS].mean()
        std = group[FEATURE_COLS].std().replace(0, 1e-6).fillna(1e-6)

        if n_shooters == 1:
            print(f"\nShooter {sid} average (across {len(group)} shots):")
            print(mean.round(2).to_string())
            print("\nPer-shot deviation (z-score from own average):\n")

        for _, row in group.iterrows():
            z = {col: round(float((row[col] - mean[col]) / std[col]), 2) for col in FEATURE_COLS}
            flagged = [c for c, v in z.items() if abs(v) > 1.0]
            inconsistent_str = f"inconsistent: {', '.join(flagged)}" if flagged else "consistent"
            record = {
                "shooter_id": sid,
                "shot_num": int(row["shot_num"]),
                **z,
                "inconsistent_features": inconsistent_str,
            }
            report_rows.append(record)

            if n_shooters == 1:
                flag_str = f"  <-- {inconsistent_str}" if flagged else ""
                print(f"Shot {int(row['shot_num'])}: {z}{flag_str}")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(paths["out"], index=False)
    print(f"\nSaved self-deviation report to: {paths['out']}")
    if n_shooters > 1:
        print(f"Report includes {len(report_df)} shots. Sample:")
        print(report_df.head(10).to_string(index=False))

    return report_df


def main():
    parser = argparse.ArgumentParser(description="Compute per-shot self-deviation scores")
    parser.add_argument("data_dir", nargs="?", default=".", help="Data directory (. or synthetic/)")
    parser.add_argument("--dir", dest="dir_flag", default=None, help="Alternative flag for data directory")
    args = parser.parse_args()

    data_dir = args.dir_flag if args.dir_flag else args.data_dir
    compute_deviation_scoring(data_dir)


if __name__ == "__main__":
    main()
