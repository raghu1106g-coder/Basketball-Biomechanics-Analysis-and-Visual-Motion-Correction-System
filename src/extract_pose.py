"""
Module 1 — Pose Extraction
===========================
Extract 33 MediaPipe Pose landmarks per frame from a shooting video.

Outputs a CSV with columns:
    frame_idx, timestamp_ms, lm_0_x … lm_32_vis

Handles:
    • Missing detections (interpolate gaps < config threshold)
    • Mid-clip subject entry
    • Frame cap to prevent hanging on long videos
    • Savitzky-Golay smoothing

CLI usage:
    python -m src.extract_pose --video path/to/video.mp4 --output landmarks.csv
    python -m src.extract_pose --metadata metadata.csv   # batch mode
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter

from src.utils import (
    load_config, NUM_LANDMARKS, LANDMARK_COLS_PER_LM,
    save_landmarks_csv, landmarks_to_dataframe,
)


# ── Core extraction ────────────────────────────────────────────────────────

def build_landmarker(model_path):
    """Create a MediaPipe PoseLandmarker in VIDEO mode."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(video_path, landmarker, max_frames=3000):
    """
    Run pose detection on every frame of *video_path*.

    Returns
    -------
    landmarks : ndarray (T, 33, 4) — x, y, z, visibility
    fps       : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Cannot open video: {}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_duration_ms = int(1000 / fps)

    frames = []
    timestamp_ms = 0
    frame_count = 0

    try:
        while cap.isOpened() and frame_count < max_frames:
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
                coords = np.full((NUM_LANDMARKS, LANDMARK_COLS_PER_LM), np.nan)

            frames.append(coords)
            timestamp_ms += frame_duration_ms
            frame_count += 1

            if frame_count % 50 == 0:
                print("  ... {} frames processed".format(frame_count), flush=True)
    finally:
        cap.release()

    print("  Total frames read: {}".format(frame_count), flush=True)
    return np.array(frames), fps


# ── Post-processing ────────────────────────────────────────────────────────

def interpolate_missing(landmarks, max_gap=5):
    """
    Fill NaN gaps shorter than *max_gap* frames via linear interpolation.
    Longer gaps are left as-is (subject not in frame).
    """
    T, J, C = landmarks.shape
    for j in range(J):
        for c in range(C):
            series = landmarks[:, j, c].copy()
            nans = np.isnan(series)
            if not nans.any() or nans.all():
                continue

            # Identify contiguous NaN runs
            nan_idx = np.where(nans)[0]
            if len(nan_idx) == 0:
                continue

            # Split into contiguous groups
            splits = np.where(np.diff(nan_idx) > 1)[0] + 1
            groups = np.split(nan_idx, splits)

            short_gaps = []
            for g in groups:
                if len(g) <= max_gap:
                    short_gaps.extend(g.tolist())

            if short_gaps:
                valid = np.where(~nans)[0]
                if len(valid) >= 2:
                    series[short_gaps] = np.interp(short_gaps, valid, series[valid])
                    landmarks[:, j, c] = series

    return landmarks


def smooth_landmarks(landmarks, window=7, poly=2):
    """Apply Savitzky-Golay smoothing per joint per coordinate."""
    T = landmarks.shape[0]
    if T <= window:
        return landmarks
    smoothed = landmarks.copy()
    for j in range(NUM_LANDMARKS):
        for c in range(LANDMARK_COLS_PER_LM):
            col = landmarks[:, j, c]
            if np.isnan(col).any():
                continue  # skip columns with remaining NaNs
            smoothed[:, j, c] = savgol_filter(col, window, poly)
    return smoothed


# ── Single-video pipeline ──────────────────────────────────────────────────

def process_video(video_path, output_path, config=None):
    """Full extraction pipeline for one video.  Returns the output path."""
    if config is None:
        config = load_config()

    ext_cfg = config["extraction"]
    model_path = ext_cfg["model_path"]
    max_frames = ext_cfg["max_frames"]
    gap_max = ext_cfg["interpolate_gap_max"]
    win = ext_cfg["smoothing_window"]
    poly = ext_cfg["smoothing_poly"]

    print("Processing: {}".format(video_path), flush=True)

    landmarker = build_landmarker(model_path)
    try:
        raw, fps = extract_landmarks(video_path, landmarker, max_frames)
    finally:
        landmarker.close()

    missed = int(np.isnan(raw[:, 0, 0]).sum())
    filled = interpolate_missing(raw, max_gap=gap_max)
    smoothed = smooth_landmarks(filled, window=win, poly=poly)

    save_landmarks_csv(smoothed, output_path, fps)
    print("  {} frames, {} interpolated -> {}".format(
        raw.shape[0], missed, output_path), flush=True)
    return output_path


# ── Batch mode via metadata ────────────────────────────────────────────────

def process_batch(metadata_path, output_dir="data/processed/landmarks", config=None):
    """Process all rows in metadata CSV."""
    if config is None:
        config = load_config()

    metadata = pd.read_csv(metadata_path)
    os.makedirs(output_dir, exist_ok=True)

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
            continue

        shooter_id = str(shooter_id).strip()
        view = str(view).strip()
        shot_num = str(int(float(shot_num)))

        # Use trimmed video if available
        trimmed_path = filepath.replace("raw/", "trimmed/", 1)
        if not os.path.exists(trimmed_path):
            trimmed_path = filepath

        out_name = "{}_{}_{}.csv".format(shooter_id, view, shot_num)
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            print("Skipping (exists): {}".format(out_path), flush=True)
            continue
        if not os.path.exists(trimmed_path):
            print("Skipping (not found): {}".format(trimmed_path), flush=True)
            continue

        process_video(trimmed_path, out_path, config)

    print("Batch pose extraction complete.", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose landmarks from basketball shooting videos."
    )
    parser.add_argument("--video", type=str, help="Path to a single video file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (single-video mode).")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to metadata CSV for batch processing.")
    parser.add_argument("--output-dir", type=str, default="data/processed/landmarks",
                        help="Output directory for batch mode.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to rules.yaml config.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.video:
        out = args.output or args.video.replace(".mp4", "_landmarks.csv")
        process_video(args.video, out, config)
    elif args.metadata:
        process_batch(args.metadata, args.output_dir, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
