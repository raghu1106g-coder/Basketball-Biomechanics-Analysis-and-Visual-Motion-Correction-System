"""One-off script to normalize metadata.csv conventions.

Unifies trait labels to good / moderate / poor, verifies made_shot in {made, miss},
and adds source, angle_deg, fps, and wrist_habit columns.
Prints the mapping and review before overwriting; saves backup to metadata.csv.bak.
"""
import argparse
import os
import shutil
import pandas as pd

ARC_QUALITY_MAP = {
    "low": "poor",
    "bad": "poor",
    "high": "moderate",
    "moderate": "moderate",
    "good": "good",
}

VALID_TRAITS = {"good", "moderate", "poor"}
VALID_MADE = {"made", "miss"}


def extract_wrist_habit(note):
    if not isinstance(note, str) or pd.isna(note):
        return ""
    n = note.lower()
    if "outward" in n:
        return "outward"
    elif "inward" in n:
        return "inward"
    elif "neutral" in n:
        return "neutral"
    return ""


def normalize_metadata(df):
    clean = df.dropna(subset=["shooter_id", "shot_num"]).copy()
    clean["shot_num"] = clean["shot_num"].astype(int)

    # Arc quality mapping
    clean["arc_quality_orig"] = clean["arc_quality"]
    clean["arc_quality"] = clean["arc_quality"].astype(str).str.strip().str.lower().map(
        lambda x: ARC_QUALITY_MAP.get(x, x)
    )

    # Trait validation
    for col in ["arc_quality", "followthrough_quality", "elbow_quality", "knee_flex_quality"]:
        clean[col] = clean[col].astype(str).str.strip().str.lower()
        invalid = set(clean[col]) - VALID_TRAITS
        if invalid:
            raise ValueError(f"Unexpected values in {col}: {invalid}")

    # made_shot validation
    clean["made_shot"] = clean["made_shot"].astype(str).str.strip().str.lower()
    invalid_made = set(clean["made_shot"]) - VALID_MADE
    if invalid_made:
        raise ValueError(f"Unexpected values in made_shot: {invalid_made}")

    # Add columns
    clean["source"] = "real_own"
    clean["angle_deg"] = 90
    clean["fps"] = 60
    clean["wrist_habit"] = clean["notes"].apply(extract_wrist_habit)

    return clean


def main():
    parser = argparse.ArgumentParser(description="Normalize metadata.csv conventions")
    parser.add_argument("--file", default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--apply", action="store_true", help="Apply changes and overwrite file after backup")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found.")
        return

    df = pd.read_csv(args.file)
    norm = normalize_metadata(df)

    print("=" * 70)
    print("METADATA NORMALIZATION REVIEW")
    print("=" * 70)
    print("Label Mapping Rules for arc_quality:")
    for k, v in ARC_QUALITY_MAP.items():
        if k != v:
            print(f"  {k:10s} -> {v}")
    print("\nAdded Columns:")
    print("  source     = 'real_own'")
    print("  angle_deg  = 90")
    print("  fps        = 60")
    print("  wrist_habit = derived from notes (inward / outward / neutral)")

    print("\nRow-by-Row Review:")
    review_cols = [
        "shooter_id", "shot_num", "made_shot", "arc_quality_orig",
        "arc_quality", "wrist_habit", "notes"
    ]
    print(norm[review_cols].rename(columns={"arc_quality_orig": "arc_before", "arc_quality": "arc_after"}).to_string(index=False))

    final_cols = [
        "shooter_id", "view", "shot_num", "filepath", "made_shot",
        "arc_quality", "followthrough_quality", "elbow_quality",
        "knee_flex_quality", "notes", "start_sec", "end_sec",
        "source", "angle_deg", "fps", "wrist_habit"
    ]
    norm_final = norm[final_cols]

    backup_path = args.file + ".bak"
    if args.apply:
        shutil.copyfile(args.file, backup_path)
        print(f"\nCreated backup at: {backup_path}")
        norm_final.to_csv(args.file, index=False)
        print(f"Successfully overwrote {args.file} with normalized conventions.")
    else:
        print(f"\nDRY RUN: File NOT modified. Run with --apply to write backup ({backup_path}) and overwrite {args.file}.")


if __name__ == "__main__":
    main()
