"""
Step 4: Batch Feature Extraction (idempotent version)
----------------------------------------------------------
Loops over every processed .npy, computes joint angles, segments phases,
checks tracking reliability, and writes to features.csv. Skips any
(shooter_id, shot_num) pair already present in features.csv, and appends
only new rows -- safe to re-run anytime as new clips are added.
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd

SHOULDER, ELBOW, WRIST = 12, 14, 16
HIP, KNEE, ANKLE = 24, 26, 28
VISIBILITY_THRESHOLD = 0.5
FEATURES_CSV = "features.csv"
FEATURE_COLUMNS = [
    "elbow_angle_load",
    "elbow_angle_release",
    "knee_flex_min",
    "knee_angle_release",
    "followthrough_frames",
    "knee_reliable_pct",
    "elbow_reliable_pct",
    "shooter_id",
    "shot_num",
]


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def compute_angle_series(coords, a_idx, b_idx, c_idx):
    return np.array([
        angle_between(coords[i, a_idx], coords[i, b_idx], coords[i, c_idx])
        for i in range(coords.shape[0])
    ])


def segment_phases(knee_angle, elbow_angle):
    n = len(knee_angle)
    load = int(np.argmin(knee_angle[: int(0.7 * n)]))
    release = load + int(np.argmax(elbow_angle[load:]))

    ft_frames = 0
    erel = elbow_angle[release]
    for idx in range(release + 1, n):
        if elbow_angle[idx] >= erel - 10.0:
            ft_frames += 1
        else:
            break
    return load, release, ft_frames


def extract_features_for_clip(landmarks, width, height):
    coords = landmarks[:, :, :3].copy()
    coords[:, :, 0] *= width
    coords[:, :, 1] *= height
    coords[:, :, 2] *= width

    elbow_series = compute_angle_series(coords, SHOULDER, ELBOW, WRIST)
    knee_series = compute_angle_series(coords, HIP, KNEE, ANKLE)
    load, release, ft_frames = segment_phases(knee_series, elbow_series)

    knee_reliable_pct = (landmarks[:, KNEE, 3] > VISIBILITY_THRESHOLD).mean() * 100
    elbow_reliable_pct = (landmarks[:, ELBOW, 3] > VISIBILITY_THRESHOLD).mean() * 100

    return {
        "elbow_angle_load": round(float(elbow_series[load]), 2),
        "elbow_angle_release": round(float(elbow_series[release]), 2),
        "knee_flex_min": round(float(knee_series[load]), 2),
        "knee_angle_release": round(float(knee_series[release]), 2),
        "followthrough_frames": int(ft_frames),
        "knee_reliable_pct": round(knee_reliable_pct, 1),
        "elbow_reliable_pct": round(elbow_reliable_pct, 1),
    }


def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 3840.0, 2160.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if w <= 0 or h <= 0:
        return 3840.0, 2160.0
    return float(w), float(h)


def main():
    parser = argparse.ArgumentParser(description="Batch Feature Extraction")
    parser.add_argument("--force", action="store_true", help="Reprocess rows already in features.csv")
    args = parser.parse_args()

    metadata = pd.read_csv("metadata.csv")
    metadata = metadata.dropna(subset=["shooter_id", "view", "shot_num", "filepath"])

    existing_df = pd.DataFrame()
    done_keys = set()
    if os.path.exists(FEATURES_CSV) and not args.force:
        existing_df = pd.read_csv(FEATURES_CSV)
        done_keys = set(zip(existing_df["shooter_id"], existing_df["shot_num"].astype(int)))
        print(f"Found existing {FEATURES_CSV} with {len(existing_df)} rows already processed.")
    elif args.force:
        print(f"--force flag set: reprocessing all rows.")
    else:
        print(f"No existing {FEATURES_CSV} found -- starting fresh.")

    new_rows = []
    skipped_no_npy = 0
    skipped_already_done = 0

    for _, row in metadata.iterrows():
        shooter_id = str(row["shooter_id"]).strip()
        shot_num = int(float(row["shot_num"]))
        view = str(row["view"]).strip()

        key = (shooter_id, shot_num)
        if key in done_keys:
            skipped_already_done += 1
            continue

        npy_path = f"processed/{shooter_id}_{view}_{shot_num}.npy"
        if not os.path.exists(npy_path):
            print(f"Skipping {npy_path} -- not yet processed by pose extraction")
            skipped_no_npy += 1
            continue

        filepath = str(row["filepath"]).strip()
        trimmed_path = filepath.replace("raw/", "trimmed/", 1)
        if not os.path.exists(trimmed_path):
            alt_trimmed = f"trimmed/{shooter_id}/{view}/{shooter_id}_{view}_{shot_num:02d}.mp4"
            if os.path.exists(alt_trimmed):
                trimmed_path = alt_trimmed

        width, height = get_video_dimensions(trimmed_path)
        landmarks = np.load(npy_path)
        features = extract_features_for_clip(landmarks, width, height)
        features["shooter_id"] = shooter_id
        features["shot_num"] = shot_num
        new_rows.append(features)

        flag = "  <-- LOW RELIABILITY" if features["knee_reliable_pct"] < 60 or features["elbow_reliable_pct"] < 60 else ""
        print(f"{npy_path}: elbow_release={features['elbow_angle_release']}, "
              f"knee_min={features['knee_flex_min']}, "
              f"ft={features['followthrough_frames']}, "
              f"reliability=knee {features['knee_reliable_pct']}%/elbow {features['elbow_reliable_pct']}%{flag}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)[FEATURE_COLUMNS]
        if args.force or existing_df.empty:
            final_df = new_df
        else:
            final_df = pd.concat([existing_df[FEATURE_COLUMNS], new_df], ignore_index=True)
        final_df = final_df[FEATURE_COLUMNS]
        final_df.to_csv(FEATURES_CSV, index=False)
        print(f"\nWrote {len(final_df)} row(s) to {FEATURES_CSV}.")
    else:
        print("\nNo new rows to add.")

    print(f"Skipped (already in features.csv): {skipped_already_done}")
    print(f"Skipped (no .npy found yet): {skipped_no_npy}")


if __name__ == "__main__":
    main()