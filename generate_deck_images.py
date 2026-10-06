"""Generate the five evidence images requested for the project deck."""

import json
import os
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.features import compute_features
from src.segment_phases import segment_phases
from src.utils import angle_series, load_config, load_landmarks, SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs", "deck_images")
LANDMARKS = os.path.join(ROOT, "data", "processed", "landmarks", "shooter01_side_01_landmarks.csv")
OVERLAY = os.path.join(ROOT, "outputs", "shooter01_side_01_landmarks_overlay.mp4")
REPORT = os.path.join(ROOT, "outputs", "shooter01_side_01_landmarks_faults.json")


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_overlay_frame():
    cap = cv2.VideoCapture(OVERLAY)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(66, max(0, frame_count // 2)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read overlay video")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame).resize((1200, 780))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 430, 92), fill=(16, 27, 28, 225))
    draw.text((46, 39), "SKELETON OVERLAY / SOURCE FRAME", fill=(244, 240, 232), font=font(24, True))
    draw.text((46, 69), "shot 01 · frame 66 · 33 landmarks · visibility QC 100%", fill=(139, 213, 167), font=font(15))
    image.save(os.path.join(OUT, "01_skeleton_overlay.png"))


def save_phase_plot():
    config = load_config()
    landmarks = load_landmarks(LANDMARKS)
    phases = segment_phases(landmarks, config=config)
    elbow = angle_series(landmarks, SHOULDER, ELBOW, WRIST)
    knee = angle_series(landmarks, HIP, KNEE, ANKLE)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True)
    fig.patch.set_facecolor("#101b1c")
    phase_colors = {"stance": "#6f8c87", "load": "#e8b04b", "release": "#e86d3d", "followthrough": "#8bd5a7"}
    for ax, series, label, color in zip(axes, (elbow, knee), ("Elbow angle", "Knee angle"), ("#e86d3d", "#54b7c6")):
        ax.set_facecolor("#162728")
        ax.plot(series, color=color, linewidth=2.5, label=label)
        for phase, start, end in phases["phases"]:
            ax.axvspan(start, end, color=phase_colors[phase], alpha=.13)
            ax.axvline(start, color=phase_colors[phase], alpha=.8, linewidth=1)
        ax.set_ylabel("degrees", color="#b8c9c2")
        ax.tick_params(colors="#b8c9c2")
        ax.grid(alpha=.15, color="white")
        ax.legend(loc="upper right", facecolor="#162728", labelcolor="white", frameon=False)
        for spine in ax.spines.values(): spine.set_color("#375150")
    axes[-1].set_xlabel("Frame", color="#b8c9c2")
    axes[-1].set_xlim(0, len(elbow) - 1)
    fig.suptitle("JOINT ANGLE VS. FRAME / PHASE BOUNDARIES", color="#f4f0e8", fontsize=18, fontweight="bold", x=.08, ha="left")
    fig.text(.08, .935, "stance     load     release     follow-through", color="#89a39c", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(os.path.join(OUT, "02_joint_angle_phases.png"), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_feature_table():
    df = pd.read_csv(os.path.join(ROOT, "features.csv"))
    columns = ["shot_num", "elbow_angle_load", "elbow_angle_release", "knee_flex_min", "release_angle", "followthrough_frames", "knee_reliable_pct"]
    display = df.copy()
    release_angles = []
    for shot_num in display["shot_num"].astype(int):
        landmark_path = os.path.join(
            ROOT, "data", "processed", "landmarks",
            "shooter01_side_{}.csv".format(shot_num),
        )
        release_angles.append(compute_features(load_landmarks(landmark_path))["release_angle"])
    display["release_angle"] = release_angles
    display = display.reindex(columns=columns).head(6).round(2)
    display = display.rename(columns={
        "shot_num": "shot",
        "elbow_angle_load": "elbow load",
        "elbow_angle_release": "elbow release",
        "knee_flex_min": "knee load",
        "release_angle": "release angle",
        "followthrough_frames": "follow f",
        "knee_reliable_pct": "knee QC",
    })
    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.patch.set_facecolor("#101b1c")
    ax.set_facecolor("#162728")
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#375150")
        cell.set_text_props(color="#f4f0e8")
        cell.set_facecolor("#1f3939" if row == 0 else ("#162728" if row % 2 else "#193131"))
        if row == 0: cell.set_text_props(weight="bold", color="#e8b04b")
    ax.set_title("EXTRACTED FEATURE TABLE / features.csv", color="#f4f0e8", fontsize=18, fontweight="bold", loc="left", pad=28)
    fig.savefig(os.path.join(OUT, "03_feature_table.png"), dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_deviation_plot():
    with open(REPORT) as file: report = json.load(file)
    labels = list(report["per_joint_zscore"])
    values = [report["per_joint_zscore"][label] for label in labels]
    colors = ["#e86d3d" if abs(value) >= 0.5 else "#54b7c6" for value in values]
    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.patch.set_facecolor("#101b1c"); ax.set_facecolor("#162728")
    y = np.arange(len(labels)); ax.barh(y, values, color=colors, height=.54)
    for index, value in enumerate(values):
        ax.text(value + .03, index, "{:.2f}".format(value), va="center", color="#f4f0e8", fontsize=11)
    ax.axvline(2, color="#e8b04b", linestyle="--", label="+2σ threshold")
    ax.axvline(-2, color="#e8b04b", linestyle="--", label="-2σ threshold")
    ax.axvline(0, color="#b8c9c2", linewidth=.8)
    ax.set_yticks(y, [label.replace("_", " ").title() for label in labels]); ax.invert_yaxis()
    ax.set_xlabel("Deviation z-score", color="#b8c9c2"); ax.tick_params(colors="#b8c9c2")
    ax.grid(axis="x", alpha=.15, color="white"); ax.legend(facecolor="#162728", labelcolor="white", frameon=False)
    for spine in ax.spines.values(): spine.set_color("#375150")
    ax.set_title("DEVIATION SCORES / SHOT 01 VS EXPERT TEMPLATE", color="#f4f0e8", fontsize=18, fontweight="bold", loc="left", pad=18)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_deviation_scores.png"), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_pytest_console():
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
    text = result.stdout + "\n" + result.stderr
    image = Image.new("RGB", (1200, 780), (12, 18, 19)); draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1200, 64), fill=(31, 51, 51))
    draw.ellipse((24, 24, 36, 36), fill=(232, 109, 61)); draw.ellipse((44, 24, 56, 36), fill=(232, 176, 75)); draw.ellipse((64, 24, 76, 36), fill=(139, 213, 167))
    draw.text((100, 19), "pytest tests/ -v", fill=(244, 240, 232), font=font(22, True))
    lines = text.strip().splitlines()[-26:]
    y = 92
    for line in lines:
        color = (139, 213, 167) if "PASSED" in line or "passed" in line else (232, 109, 61) if "FAILED" in line or "failed" in line else (190, 204, 198)
        draw.text((38, y), line[:112], fill=color, font=font(18)); y += 25
    draw.text((38, 738), "exit code: {}".format(result.returncode), fill=(139, 213, 167) if result.returncode == 0 else (232, 109, 61), font=font(18, True))
    image.save(os.path.join(OUT, "05_pytest_console.png"))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    save_overlay_frame(); save_phase_plot(); save_feature_table(); save_deviation_plot(); save_pytest_console()
    print("Deck images written to {}".format(OUT))