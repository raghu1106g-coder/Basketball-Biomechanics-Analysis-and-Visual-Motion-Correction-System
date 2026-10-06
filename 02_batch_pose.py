"""
Batch Pose Extraction + Smoothing
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter

MODEL_PATH = "pose_landmarker.task"


def build_landmarker(model_path):
    print("Building landmarker...", flush=True)
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    lm = vision.PoseLandmarker.create_from_options(options)
    print("Landmarker built OK.", flush=True)
    return lm


def iter_pose_requests_from_dataframe(metadata):
    for _, row in metadata.iterrows():
        filepath = row.get("filepath")
        if pd.isna(filepath):
            continue
        filepath = str(filepath).strip()
        if not filepath:
            continue
        shooter_id = row.get("shooter_id")
        view = row.get("view")
        shot_num = row.get("shot_num")
        if pd.isna(shooter_id) or pd.isna(view) or pd.isna(shot_num):
            print(f"Skipping {filepath}: missing metadata", flush=True)
            continue
        shooter_id = str(shooter_id).strip()
        view = str(view).strip()
        shot_num = str(int(float(shot_num)))
        trimmed_path = filepath.replace("raw/", "trimmed/", 1)
        out_path = os.path.join("processed", f"{shooter_id}_{view}_{shot_num}.npy")
        yield trimmed_path, out_path


def extract_landmarks(video_path, landmarker):
    print(f"  Opening {video_path}", flush=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  fps={fps}", flush=True)
    frame_duration_ms = int(1000 / fps) if fps > 0 else 33
    frames_landmarks = []
    timestamp_ms = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])
        else:
            coords = np.full((33, 4), np.nan)
        frames_landmarks.append(coords)
        timestamp_ms += frame_duration_ms
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"  ...{frame_count} frames processed", flush=True)

    cap.release()
    print(f"  Done reading: {frame_count} total frames", flush=True)
    return np.array(frames_landmarks)


def interpolate_missing(landmarks):
    for j in range(33):
        for k in range(4):
            series = landmarks[:, j, k]
            nans = np.isnan(series)
            if nans.any() and not nans.all():
                series[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), series[~nans])
                landmarks[:, j, k] = series
    return landmarks


def smooth_landmarks(landmarks, window=7, poly=2):
    num_frames = landmarks.shape[0]
    if num_frames <= window:
        return landmarks
    smoothed = np.copy(landmarks)
    for j in range(33):
        for k in range(4):
            smoothed[:, j, k] = savgol_filter(landmarks[:, j, k], window, poly)
    return smoothed


def process_pose_metadata(metadata_path="metadata.csv", model_path=MODEL_PATH):
    print("Reading metadata.csv...", flush=True)
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded {len(metadata)} rows.", flush=True)

    landmarker = build_landmarker(model_path)
    os.makedirs("processed", exist_ok=True)

    requests = list(iter_pose_requests_from_dataframe(metadata))
    print(f"Found {len(requests)} valid rows to process.", flush=True)

    try:
        for trimmed_path, out_path in requests:
            if os.path.exists(out_path):
                print(f"Skipping (already processed): {out_path}", flush=True)
                continue
            if not os.path.exists(trimmed_path):
                print(f"Skipping {trimmed_path}: file not found", flush=True)
                continue

            print(f"Processing {trimmed_path}...", flush=True)
            raw = extract_landmarks(trimmed_path, landmarker)
            missed = np.isnan(raw[:, 0, 0]).sum()
            filled = interpolate_missing(raw)
            smoothed = smooth_landmarks(filled)
            np.save(out_path, smoothed)
            print(f"  {raw.shape[0]} frames, {missed} interpolated -> saved {out_path}", flush=True)
    finally:
        landmarker.close()


def main():
    process_pose_metadata()
    print("Batch pose extraction complete.", flush=True)


if __name__ == "__main__":
    main()