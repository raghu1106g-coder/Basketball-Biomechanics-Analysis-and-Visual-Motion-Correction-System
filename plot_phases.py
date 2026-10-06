"""Plot elbow and knee angle curves with detected phases for manual spot-checking.

Saves phase diagnostic figures to diagnostics/ for visual verification.
"""
import argparse
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import importlib
batch_features = importlib.import_module("03_batch_features")
SHOULDER = batch_features.SHOULDER
ELBOW = batch_features.ELBOW
WRIST = batch_features.WRIST
HIP = batch_features.HIP
KNEE = batch_features.KNEE
ANKLE = batch_features.ANKLE
compute_angle_series = batch_features.compute_angle_series
segment_phases = batch_features.segment_phases
get_video_dimensions = batch_features.get_video_dimensions


def plot_clip_phases(shooter_id, view, shot_num, trimmed_path, npy_path, out_dir="diagnostics"):
    if not os.path.exists(npy_path):
        print(f"Skipping {npy_path}: file does not exist")
        return None

    landmarks = np.load(npy_path)
    width, height = get_video_dimensions(trimmed_path)

    coords = landmarks[:, :, :3].copy()
    coords[:, :, 0] *= width
    coords[:, :, 1] *= height
    coords[:, :, 2] *= width

    elbow = compute_angle_series(coords, SHOULDER, ELBOW, WRIST)
    knee = compute_angle_series(coords, HIP, KNEE, ANKLE)
    load, release, ft_frames = segment_phases(knee, elbow)

    n_frames = len(elbow)
    frames = np.arange(n_frames)

    fig, (ax_elbow, ax_knee) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Elbow subplot
    ax_elbow.plot(frames, elbow, color="#1f77b4", lw=2, label="Elbow angle")
    ax_elbow.axvline(load, color="red", linestyle="--", alpha=0.7, label=f"Load (f={load}, {elbow[load]:.1f}°)")
    ax_elbow.axvline(release, color="green", linestyle="--", alpha=0.7, label=f"Release (f={release}, {elbow[release]:.1f}°)")
    ax_elbow.scatter([load], [elbow[load]], color="red", s=60, zorder=5)
    ax_elbow.scatter([release], [elbow[release]], color="green", s=60, zorder=5)

    if ft_frames > 0:
        ft_end = min(release + ft_frames, n_frames - 1)
        ax_elbow.axvspan(release, ft_end, color="green", alpha=0.15, label=f"Follow-through ({ft_frames} frames)")
        ax_elbow.axhline(elbow[release] - 10.0, color="gray", linestyle=":", alpha=0.6, label="Release - 10° threshold")

    ax_elbow.set_ylabel("Elbow Angle (deg)")
    ax_elbow.set_title(f"{shooter_id} Shot {shot_num} ({view}) — Detected Load & Release Phases", fontsize=12, fontweight="bold")
    ax_elbow.grid(True, alpha=0.3)
    ax_elbow.legend(loc="upper left", fontsize=9)

    # Knee subplot
    ax_knee.plot(frames, knee, color="#ff7f0e", lw=2, label="Knee angle")
    ax_knee.axvline(load, color="red", linestyle="--", alpha=0.7, label=f"Knee min dip ({knee[load]:.1f}°)")
    ax_knee.axvline(release, color="green", linestyle="--", alpha=0.7, label=f"Knee at release ({knee[release]:.1f}°)")
    ax_knee.scatter([load], [knee[load]], color="red", s=60, zorder=5)
    ax_knee.scatter([release], [knee[release]], color="green", s=60, zorder=5)

    ax_knee.set_xlabel("Frame Number")
    ax_knee.set_ylabel("Knee Angle (deg)")
    ax_knee.grid(True, alpha=0.3)
    ax_knee.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{shooter_id}_{view}_{shot_num}_phases.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved phase plot: {out_path} [load={load}, release={release}, ft={ft_frames}]")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot detected phases for spot check")
    parser.add_argument("--metadata", default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--out", default="diagnostics", help="Output directory")
    parser.add_argument("--shots", nargs="*", type=int, help="Optional specific shot numbers to plot")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata).dropna(subset=["shooter_id", "view", "shot_num"])
    if args.shots:
        meta = meta[meta["shot_num"].astype(int).isin(args.shots)]

    for _, row in meta.iterrows():
        sid = str(row["shooter_id"]).strip()
        view = str(row["view"]).strip()
        shot = int(float(row["shot_num"]))
        filepath = str(row.get("filepath", "")).strip()
        trimmed = filepath.replace("raw/", "trimmed/", 1)
        if not os.path.exists(trimmed):
            alt_trimmed = f"trimmed/{sid}/{view}/{sid}_{view}_{shot:02d}.mp4"
            if os.path.exists(alt_trimmed):
                trimmed = alt_trimmed
        npy = f"processed/{sid}_{view}_{shot}.npy"
        plot_clip_phases(sid, view, shot, trimmed, npy, out_dir=args.out)


if __name__ == "__main__":
    main()
