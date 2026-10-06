"""
Landmark-Level Data Augmentation
=================================
Apply augmentations at the LANDMARK level (not pixel level):
    1. Mirror-flip: swap left/right landmarks, negate x
    2. Temporal resample: stretch/compress by 0.8x–1.25x
    3. Gaussian jitter: add N(0, σ) to x, y, z

All augmented rows include an `is_augmented` flag column.

CLI usage:
    python scripts/augment.py --input data/processed/landmarks/ --output data/processed/landmarks_augmented/
    python scripts/augment.py --input single_file.csv --output augmented.csv --methods mirror jitter temporal
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, load_landmarks_csv, save_landmarks_csv,
    dataframe_to_landmarks, landmarks_to_dataframe,
    NUM_LANDMARKS,
)


def mirror_flip(landmarks, swap_pairs):
    """
    Mirror-flip: swap left/right landmark pairs and negate x-coordinate.

    Parameters
    ----------
    landmarks : ndarray (T, 33, 4)
    swap_pairs : list of (int, int)

    Returns
    -------
    ndarray (T, 33, 4) — flipped
    """
    flipped = landmarks.copy()

    # Negate x (flip horizontally) — MediaPipe x is normalised [0, 1]
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]

    # Swap left/right pairs
    for (left, right) in swap_pairs:
        temp = flipped[:, left, :].copy()
        flipped[:, left, :] = flipped[:, right, :]
        flipped[:, right, :] = temp

    return flipped


def temporal_resample(landmarks, factor):
    """
    Stretch/compress time-series by *factor* using linear interpolation.
    factor < 1 → compress (fewer frames), factor > 1 → stretch (more frames).
    """
    T = landmarks.shape[0]
    new_T = max(3, int(T * factor))
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_T)

    resampled = np.zeros((new_T, NUM_LANDMARKS, 4))
    for j in range(NUM_LANDMARKS):
        for c in range(4):
            resampled[:, j, c] = np.interp(x_new, x_old, landmarks[:, j, c])

    return resampled


def gaussian_jitter(landmarks, std=0.005):
    """
    Add Gaussian noise to x, y, z coordinates (not visibility).
    """
    jittered = landmarks.copy()
    noise = np.random.normal(0, std, size=(landmarks.shape[0], NUM_LANDMARKS, 3))
    jittered[:, :, :3] += noise
    return jittered


def augment_landmarks(landmarks, methods, config=None):
    """
    Apply specified augmentation methods.

    Parameters
    ----------
    landmarks : ndarray (T, 33, 4)
    methods : list of str — subset of ["mirror", "temporal", "jitter"]
    config : dict or None

    Returns
    -------
    list of (augmented_landmarks, method_name)
    """
    if config is None:
        config = load_config()

    aug_cfg = config["augmentation"]
    results = []

    if "mirror" in methods:
        swap_pairs = [tuple(p) for p in aug_cfg["mirror_swap_pairs"]]
        flipped = mirror_flip(landmarks, swap_pairs)
        results.append((flipped, "mirror"))

    if "temporal" in methods:
        lo, hi = aug_cfg["temporal_resample_range"]
        # Generate a few factors spread across the range
        for factor in [lo, 1.0, hi]:
            if abs(factor - 1.0) < 0.01:
                continue  # skip identity
            resampled = temporal_resample(landmarks, factor)
            results.append((resampled, "temporal_{:.2f}".format(factor)))

    if "jitter" in methods:
        std = aug_cfg["jitter_std"]
        jittered = gaussian_jitter(landmarks, std)
        results.append((jittered, "jitter"))

    return results


def augment_file(input_path, output_dir, methods, config=None):
    """Augment a single landmark file and save results."""
    landmarks = load_landmarks_csv(input_path) if input_path.endswith(".csv") else np.load(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]

    augmented = augment_landmarks(landmarks, methods, config)
    os.makedirs(output_dir, exist_ok=True)

    for aug_lm, method_name in augmented:
        out_name = "{}_{}.csv".format(base, method_name)
        out_path = os.path.join(output_dir, out_name)

        df = landmarks_to_dataframe(aug_lm)
        df["is_augmented"] = True
        df["augmentation_method"] = method_name
        df.to_csv(out_path, index=False)
        print("  Saved: {} ({} frames)".format(out_path, len(df)))


def main():
    parser = argparse.ArgumentParser(description="Landmark-level data augmentation.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input landmark CSV or directory of CSVs.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for augmented files.")
    parser.add_argument("--methods", nargs="+", default=["mirror", "temporal", "jitter"],
                        choices=["mirror", "temporal", "jitter"],
                        help="Augmentation methods to apply.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.csv"))
        if not files:
            # Try .npy
            files = glob.glob(os.path.join(args.input, "*.npy"))
        print("Found {} files to augment".format(len(files)))
        for f in sorted(files):
            print("Augmenting: {}".format(f))
            augment_file(f, args.output, args.methods, config)
    else:
        augment_file(args.input, args.output, args.methods, config)

    print("\nAugmentation complete.")


if __name__ == "__main__":
    main()
