"""Classifier Training and Evaluation Module.

Trains MultiOutputClassifier(RandomForestClassifier) using GroupKFold grouped
by shooter_id. Reports macro F1 / precision / recall per trait first.
Guarded to skip if fewer than 3 shooters.
Includes prominent synthetic warning banner when evaluating synthetic data.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

FEATURES = [
    "elbow_angle_load",
    "elbow_angle_release",
    "knee_flex_min",
    "knee_angle_release",
    "followthrough_frames",
]
TRAITS = ["arc_quality", "followthrough_quality", "elbow_quality", "knee_flex_quality"]


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

    return {
        "dir": data_dir,
        "is_synthetic": is_synthetic,
        "features": features_path,
        "metadata": metadata_path,
    }


def train_and_evaluate(data_dir="."):
    paths = resolve_paths(data_dir)
    if not os.path.exists(paths["features"]):
        raise FileNotFoundError(f"Features file not found: {paths['features']}")
    if not os.path.exists(paths["metadata"]):
        raise FileNotFoundError(f"Metadata file not found: {paths['metadata']}")

    if paths["is_synthetic"]:
        print("=" * 80)
        print("SYNTHETIC DATA - NOT EVIDENCE OF REAL-WORLD ACCURACY")
        print("=" * 80)

    print(f"Loading features from: {paths['features']}")
    print(f"Loading metadata from: {paths['metadata']}")

    feats = pd.read_csv(paths["features"])
    meta = pd.read_csv(paths["metadata"]).dropna(subset=["shooter_id", "shot_num"])

    feats["shot_num"] = feats["shot_num"].astype(int)
    meta["shot_num"] = meta["shot_num"].astype(int)

    # Validate target columns
    available_traits = [t for t in TRAITS if t in meta.columns]
    if not available_traits:
        raise ValueError(f"No trait columns found in metadata: {TRAITS}")

    merged = feats.merge(
        meta[["shooter_id", "shot_num"] + available_traits],
        on=["shooter_id", "shot_num"]
    ).dropna(subset=FEATURES + available_traits)

    shooters = merged["shooter_id"].unique()
    n_shooters = len(shooters)
    print(f"Dataset contains {len(merged)} shots across {n_shooters} shooter(s).")

    # Guard: skip if fewer than 3 shooters
    if n_shooters < 3:
        print("\n" + "#" * 70)
        print(f"[GUARD] SKIPPING CLASSIFIER TRAINING: Found only {n_shooters} shooter(s) ({', '.join(shooters)}).")
        print("GroupKFold cross-validation requires at least 3 shooters.")
        print("#" * 70 + "\n")
        return None

    X = merged[FEATURES].values
    Y = merged[available_traits].values
    groups = merged["shooter_id"].values

    n_splits = min(5, n_shooters)
    gkf = GroupKFold(n_splits=n_splits)

    # MultiOutput Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf = MultiOutputClassifier(rf)

    oof_preds = np.empty_like(Y, dtype=object)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=groups), 1):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val = X[val_idx]

        clf.fit(X_train, Y_train)
        pred_val = clf.predict(X_val)
        oof_preds[val_idx] = pred_val

    # Compute metrics per trait: Macro F1, Precision, Recall first, then Accuracy
    results = []
    for i, trait in enumerate(available_traits):
        y_true = Y[:, i]
        y_pred = oof_preds[:, i]

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        results.append({
            "Trait": trait,
            "Macro F1": round(f1, 4),
            "Macro Precision": round(prec, 4),
            "Macro Recall": round(rec, 4),
            "Accuracy": round(acc, 4),
        })

    res_df = pd.DataFrame(results)

    # Overall macro average
    mean_row = {
        "Trait": "-- MACRO AVERAGE --",
        "Macro F1": round(res_df["Macro F1"].mean(), 4),
        "Macro Precision": round(res_df["Macro Precision"].mean(), 4),
        "Macro Recall": round(res_df["Macro Recall"].mean(), 4),
        "Accuracy": round(res_df["Accuracy"].mean(), 4),
    }
    summary_df = pd.concat([res_df, pd.DataFrame([mean_row])], ignore_index=True)

    print(f"\nGroupKFold Evaluation ({n_splits} folds across {n_shooters} shooters):")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print("-" * 80)
    print("Note: Reported metrics prioritize Macro F1 / Precision / Recall to capture multi-class trait balance.")
    if paths["is_synthetic"]:
        print("REMINDER: Results are on SYNTHETIC data and do not reflect real-world model accuracy.")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Train multi-output trait classifier")
    parser.add_argument("data_dir", nargs="?", default=".", help="Data directory (. or synthetic/)")
    parser.add_argument("--dir", dest="dir_flag", default=None, help="Alternative flag for data directory")
    args = parser.parse_args()

    data_dir = args.dir_flag if args.dir_flag else args.data_dir
    train_and_evaluate(data_dir)


if __name__ == "__main__":
    main()
