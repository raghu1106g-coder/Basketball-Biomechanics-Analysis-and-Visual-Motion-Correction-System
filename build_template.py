"""Personal Best Template Builder.

Builds shooter-specific personal best templates from made and good-quality shots.
Accepts an optional data directory argument (default '.' = real data).
Outputs are written into the same directory they read from.
"""
import argparse
import os
import pandas as pd

FEATURES = [
    "elbow_angle_load",
    "elbow_angle_release",
    "knee_flex_min",
    "knee_angle_release",
    "followthrough_frames",
]
TRAIT_COLS = ["arc_quality", "followthrough_quality", "elbow_quality", "knee_flex_quality"]


def resolve_paths(data_dir="."):
    data_dir = os.path.normpath(data_dir)
    is_synthetic = (
        os.path.exists(os.path.join(data_dir, "synthetic_features.csv"))
        or os.path.basename(data_dir).lower() == "synthetic"
    )
    if is_synthetic:
        features_path = os.path.join(data_dir, "synthetic_features.csv")
        metadata_path = os.path.join(data_dir, "synthetic_metadata.csv")
    else:
        features_path = os.path.join(data_dir, "features.csv")
        metadata_path = os.path.join(data_dir, "metadata.csv")

    out_path = os.path.join(data_dir, "personal_best_templates.csv")
    return {
        "dir": data_dir,
        "is_synthetic": is_synthetic,
        "features": features_path,
        "metadata": metadata_path,
        "out": out_path,
    }


def build_personal_best_templates(data_dir="."):
    paths = resolve_paths(data_dir)
    if not os.path.exists(paths["features"]):
        raise FileNotFoundError(f"Features file not found: {paths['features']}")
    if not os.path.exists(paths["metadata"]):
        raise FileNotFoundError(f"Metadata file not found: {paths['metadata']}")

    if paths["is_synthetic"]:
        print("=" * 70)
        print("SYNTHETIC DATA - NOT EVIDENCE OF REAL-WORLD ACCURACY")
        print("=" * 70)

    print(f"Building templates from: {paths['features']} and {paths['metadata']}")
    feats = pd.read_csv(paths["features"])
    meta = pd.read_csv(paths["metadata"]).dropna(subset=["shooter_id", "shot_num"])

    feats["shot_num"] = feats["shot_num"].astype(int)
    meta["shot_num"] = meta["shot_num"].astype(int)

    available_traits = [c for c in TRAIT_COLS if c in meta.columns]
    merge_cols = ["shooter_id", "shot_num", "made_shot"] + available_traits
    df = feats.merge(meta[merge_cols], on=["shooter_id", "shot_num"])

    if available_traits:
        df["all_good"] = (df[available_traits] == "good").all(axis=1)
    else:
        df["all_good"] = True
    df["made"] = df["made_shot"].astype(str).str.strip().str.lower() == "made"

    template_rows = []
    source = "synthetic" if paths["is_synthetic"] else "real_own"

    for sid, group in df.groupby("shooter_id", sort=False):
        for basis, mask in (
            ("made+all_good", group["made"] & group["all_good"]),
            ("made_only", group["made"]),
            ("all_good_only", group["all_good"]),
            ("all_shots", group["made"] | ~group["made"]),
        ):
            if mask.sum() > 0:
                selected_basis = basis
                selected_mask = mask
                break
        else:
            selected_basis = "all_shots"
            selected_mask = pd.Series(True, index=group.index)

        mean_vals = group.loc[selected_mask, FEATURES].mean().round(2).to_dict()
        row = {
            "shooter_id": sid,
            **mean_vals,
            "n_template_shots": int(selected_mask.sum()),
            "template_basis": selected_basis,
            "source": source,
        }
        template_rows.append(row)

    tmpl_df = pd.DataFrame(template_rows)
    tmpl_df.to_csv(paths["out"], index=False)
    print(f"Saved {len(tmpl_df)} shooter template(s) to: {paths['out']}")
    print("\nTemplate Summary:")
    print(tmpl_df.to_string(index=False))
    return tmpl_df


def main():
    parser = argparse.ArgumentParser(description="Build shooter personal best templates")
    parser.add_argument("data_dir", nargs="?", default=".", help="Data directory (. or synthetic/)")
    parser.add_argument("--dir", dest="dir_flag", default=None, help="Alternative flag for data directory")
    args = parser.parse_args()

    data_dir = args.dir_flag if args.dir_flag else args.data_dir
    build_personal_best_templates(data_dir)


if __name__ == "__main__":
    main()
