"""
Module 6 — Visualization
==========================
(a) Skeleton overlay on original video with faulty joints in red.
(b) Side-by-side "your motion vs corrected motion" animation using
    forward kinematics to interpolate flagged joints toward the template.

CLI usage:
    python -m src.visualize --video vid.mp4 --landmarks shot.csv --faults faults.json --mode both
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import (
    load_config, load_landmarks, angle_series, normalize_timeseries,
    POSE_CONNECTIONS, NUM_LANDMARKS,
    SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE,
    forward_kinematics_chain, average_limb_lengths,
)
from src.build_template import TEMPLATE_JOINTS
from src.deviation import load_template


# ── Drawing helpers ─────────────────────────────────────────────────────────

def _pixel_pos(landmark_xy, frame_w, frame_h):
    """Convert normalised (x, y) to pixel coordinates."""
    return int(landmark_xy[0] * frame_w), int(landmark_xy[1] * frame_h)


def draw_skeleton(frame, landmarks_frame, fault_joints=None, config=None):
    """
    Draw the MediaPipe pose skeleton on *frame* (in-place).

    Parameters
    ----------
    frame : ndarray (H, W, 3) BGR
    landmarks_frame : ndarray (33, 4)  x, y, z, vis
    fault_joints : set of int — landmark indices with faults (drawn red)
    config : dict
    """
    if config is None:
        config = load_config()

    vis_cfg = config["visualization"]
    color_ok    = tuple(vis_cfg["skeleton_color_ok"])
    color_fault = tuple(vis_cfg["skeleton_color_fault"])
    thickness   = vis_cfg["skeleton_thickness"]
    radius      = vis_cfg["joint_radius"]
    h, w = frame.shape[:2]

    if fault_joints is None:
        fault_joints = set()

    # Draw connections
    for (a, b) in POSE_CONNECTIONS:
        if np.isnan(landmarks_frame[a, 0]) or np.isnan(landmarks_frame[b, 0]):
            continue
        pa = _pixel_pos(landmarks_frame[a, :2], w, h)
        pb = _pixel_pos(landmarks_frame[b, :2], w, h)
        conn_color = color_fault if (a in fault_joints or b in fault_joints) else color_ok
        cv2.line(frame, pa, pb, conn_color, thickness)

    # Draw joints
    for j in range(NUM_LANDMARKS):
        if np.isnan(landmarks_frame[j, 0]):
            continue
        pos = _pixel_pos(landmarks_frame[j, :2], w, h)
        jcolor = color_fault if j in fault_joints else color_ok
        cv2.circle(frame, pos, radius, jcolor, -1)

    return frame


def _fault_joint_indices(faults):
    """Map fault joint names to MediaPipe landmark indices."""
    name_to_idx = {
        "elbow": ELBOW,
        "knee": KNEE,
        "wrist": WRIST,
        "shoulder": SHOULDER,
        "forearm": ELBOW,  # forearm faults highlight elbow
        "hip": HIP,
        "ankle": ANKLE,
    }
    indices = set()
    for f in faults:
        joint_name = f.get("joint", "")
        if joint_name in name_to_idx:
            indices.add(name_to_idx[joint_name])
    return indices


# ── (a) Skeleton Overlay Video ─────────────────────────────────────────────

def render_overlay_video(video_path, landmarks, faults, output_path, config=None):
    """
    Overlay skeleton on original video. Faulty joints in red.
    """
    if config is None:
        config = load_config()

    vis_cfg = config["visualization"]
    fault_indices = _fault_joint_indices(faults)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Cannot open video: {}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or vis_cfg["video_fps"]
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = landmarks.shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < T:
            ret, frame = cap.read()
            if not ret:
                break

            # Determine which joints are faulty in THIS frame's phase
            active_faults = set()
            for f in faults:
                fr = f.get("frame_range", [0, T])
                if fr[0] <= frame_idx <= fr[1]:
                    active_faults.update(_fault_joint_indices([f]))
            if not active_faults:
                active_faults = fault_indices  # fallback: always highlight

            draw_skeleton(frame, landmarks[frame_idx], active_faults, config)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print("Overlay video saved: {} ({} frames)".format(output_path, frame_idx))
    return output_path


# ── (b) Corrected Skeleton via Forward Kinematics ──────────────────────────

def _correct_joint_angles(landmarks, template, faults, config=None):
    """
    Generate corrected landmarks by blending faulty joint angles toward
    the template while keeping limb lengths fixed.

    Returns corrected (T, 33, 4) landmarks.
    """
    if config is None:
        config = load_config()

    blend = config["visualization"]["correction_blend"]
    norm_len = config["dtw"]["normalize_length"]
    T = landmarks.shape[0]
    corrected = landmarks.copy()

    fault_joint_names = set(f["joint"] for f in faults)

    for joint_name, (a_idx, b_idx, c_idx) in TEMPLATE_JOINTS.items():
        if joint_name not in fault_joint_names and joint_name != "forearm":
            continue
        if joint_name not in template:
            continue

        # Original angle series
        orig_angles = angle_series(landmarks, a_idx, b_idx, c_idx)

        # Template mean (normalised to T frames)
        tmpl_mean = template[joint_name]["mean"]
        tmpl_resampled = normalize_timeseries(tmpl_mean, T)

        # Blend: corrected_angle = orig + blend * (template - orig)
        corrected_angles = orig_angles + blend * (tmpl_resampled - orig_angles)

        # Apply corrected angles using simple 2-D IK around the parent joint
        for i in range(T):
            if np.isnan(landmarks[i, b_idx, 0]):
                continue

            # Get the parent (a) and child (c) positions
            parent = landmarks[i, a_idx, :2].copy()
            vertex = landmarks[i, b_idx, :2].copy()
            child  = landmarks[i, c_idx, :2].copy()

            # Original limb lengths
            len_ab = np.linalg.norm(vertex - parent)
            len_bc = np.linalg.norm(child - vertex)

            if len_ab < 1e-6 or len_bc < 1e-6:
                continue

            # Direction from parent to vertex (keep this fixed)
            dir_ab = (vertex - parent) / len_ab

            # Compute the new vertex-to-child direction using corrected angle
            orig_angle = orig_angles[i]
            new_angle  = corrected_angles[i]
            delta_rad  = np.radians(new_angle - orig_angle)

            # Rotate the bc vector by delta_rad
            bc_vec = child - vertex
            cos_d, sin_d = np.cos(delta_rad), np.sin(delta_rad)
            new_bc = np.array([
                cos_d * bc_vec[0] - sin_d * bc_vec[1],
                sin_d * bc_vec[0] + cos_d * bc_vec[1],
            ])
            # Normalise and scale to original length
            new_bc = new_bc / (np.linalg.norm(new_bc) + 1e-8) * len_bc

            corrected[i, c_idx, 0] = vertex[0] + new_bc[0]
            corrected[i, c_idx, 1] = vertex[1] + new_bc[1]

    return corrected


def render_correction_video(landmarks, corrected, faults, output_path, config=None):
    """
    Side-by-side skeleton animation: original (left) vs corrected (right).
    """
    if config is None:
        config = load_config()

    vis_cfg = config["visualization"]
    panel_w = vis_cfg["side_by_side_width"]
    panel_h = vis_cfg["side_by_side_height"]
    fps = vis_cfg["video_fps"]
    T = landmarks.shape[0]

    total_w = panel_w * 2
    fault_indices = _fault_joint_indices(faults)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (total_w, panel_h))

    for i in range(T):
        canvas = np.zeros((panel_h, total_w, 3), dtype=np.uint8)

        # Background panels
        canvas[:, :panel_w] = (30, 30, 30)         # dark grey left
        canvas[:, panel_w:] = (20, 20, 35)          # dark blue-grey right

        # Labels
        cv2.putText(canvas, "Your Shot", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, "Corrected", (panel_w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

        # Draw original skeleton (left panel)
        left_lm = landmarks[i].copy()
        left_lm[:, 0] = left_lm[:, 0] * panel_w  # un-normalise for drawing
        left_lm[:, 1] = left_lm[:, 1] * panel_h
        _draw_skeleton_pixel(canvas, left_lm, fault_indices, config,
                             offset_x=0, w=panel_w, h=panel_h)

        # Draw corrected skeleton (right panel)
        right_lm = corrected[i].copy()
        right_lm[:, 0] = right_lm[:, 0] * panel_w
        right_lm[:, 1] = right_lm[:, 1] * panel_h
        _draw_skeleton_pixel(canvas, right_lm, set(), config,
                             offset_x=panel_w, w=panel_w, h=panel_h,
                             override_color=(100, 255, 100))

        writer.write(canvas)

    writer.release()
    print("Correction video saved: {} ({} frames)".format(output_path, T))
    return output_path


def _draw_skeleton_pixel(frame, lm_pixels, fault_joints, config,
                         offset_x=0, w=640, h=480, override_color=None):
    """Draw skeleton using pixel-scale landmarks with an x offset."""
    vis_cfg = config["visualization"]
    color_ok    = override_color or tuple(vis_cfg["skeleton_color_ok"])
    color_fault = tuple(vis_cfg["skeleton_color_fault"])
    thickness   = vis_cfg["skeleton_thickness"]
    radius      = vis_cfg["joint_radius"]

    for (a, b) in POSE_CONNECTIONS:
        if np.isnan(lm_pixels[a, 0]) or np.isnan(lm_pixels[b, 0]):
            continue
        pa = (int(lm_pixels[a, 0]) + offset_x, int(lm_pixels[a, 1]))
        pb = (int(lm_pixels[b, 0]) + offset_x, int(lm_pixels[b, 1]))
        c = color_fault if (a in fault_joints or b in fault_joints) else color_ok
        cv2.line(frame, pa, pb, c, thickness)

    for j in range(NUM_LANDMARKS):
        if np.isnan(lm_pixels[j, 0]):
            continue
        pos = (int(lm_pixels[j, 0]) + offset_x, int(lm_pixels[j, 1]))
        c = color_fault if j in fault_joints else color_ok
        cv2.circle(frame, pos, radius, c, -1)


# ── Angle-vs-Time Plots ───────────────────────────────────────────────────

def render_angle_plots(landmarks, template, phases, output_dir, config=None):
    """
    Generate angle-vs-time plots for each joint, with template band overlay.
    Saves PNGs to output_dir.
    """
    if config is None:
        config = load_config()

    norm_len = config["dtw"]["normalize_length"]
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    for joint_name, (a, b, c) in TEMPLATE_JOINTS.items():
        ts = angle_series(landmarks, a, b, c)
        normed = normalize_timeseries(ts, norm_len)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#16213e")

        x = np.arange(norm_len)

        # Template band
        if joint_name in template:
            mean = template[joint_name]["mean"]
            std  = template[joint_name]["std"]
            ax.fill_between(x, mean - std, mean + std,
                            alpha=0.3, color="#00d2ff", label="Template ±1σ")
            ax.plot(x, mean, color="#00d2ff", linewidth=1.5,
                    linestyle="--", label="Template mean")

        # Test shot
        ax.plot(x, normed, color="#ff6b35", linewidth=2, label="Your shot")

        # Phase markers
        if phases:
            total_frames = landmarks.shape[0]
            for phase_name, start, end in phases.get("phases", []):
                norm_start = int(start / total_frames * norm_len)
                ax.axvline(norm_start, color="#ffffff", alpha=0.3, linestyle=":")
                ax.text(norm_start + 1, ax.get_ylim()[1] * 0.95,
                        phase_name, color="#ffffff", fontsize=8, alpha=0.7)

        ax.set_xlabel("Normalised Frame", color="white", fontsize=11)
        ax.set_ylabel("Angle (degrees)", color="white", fontsize=11)
        ax.set_title("{} Angle".format(joint_name.replace("_", " ").title()),
                     color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")
        ax.grid(alpha=0.15)

        out_path = os.path.join(output_dir, "{}_angle.png".format(joint_name))
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        plot_paths[joint_name] = out_path

    print("Angle plots saved to {}".format(output_dir))
    return plot_paths


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise biomechanics analysis.")
    parser.add_argument("--video", type=str, default=None,
                        help="Original video (needed for overlay mode).")
    parser.add_argument("--landmarks", type=str, required=True,
                        help="Landmark file (CSV or NPY).")
    parser.add_argument("--faults", type=str, default=None,
                        help="Faults JSON file (from deviation module).")
    parser.add_argument("--template", type=str, default="data/expert_template.csv",
                        help="Expert template CSV (for correction + plots).")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["overlay", "correction", "plots", "both"],
                        help="Visualization mode.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    landmarks = load_landmarks(args.landmarks)

    # Load faults
    faults_list = []
    if args.faults and os.path.exists(args.faults):
        with open(args.faults) as f:
            faults_data = json.load(f)
            faults_list = faults_data.get("faults", faults_data if isinstance(faults_data, list) else [])

    # Load template
    template = {}
    if os.path.exists(args.template):
        template = load_template(args.template)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.landmarks))[0]

    if args.mode in ("overlay", "both"):
        if args.video:
            overlay_path = os.path.join(args.output_dir, "{}_overlay.mp4".format(base))
            render_overlay_video(args.video, landmarks, faults_list, overlay_path, config)
        else:
            print("WARNING: --video required for overlay mode")

    if args.mode in ("correction", "both"):
        if template:
            corrected = _correct_joint_angles(landmarks, template, faults_list, config)
            corr_path = os.path.join(args.output_dir, "{}_correction.mp4".format(base))
            render_correction_video(landmarks, corrected, faults_list, corr_path, config)
        else:
            print("WARNING: Template needed for correction mode")

    if args.mode in ("plots", "both"):
        from src.segment_phases import segment_phases as _seg
        phases = _seg(landmarks, config=config)
        plots_dir = os.path.join(args.output_dir, "plots")
        render_angle_plots(landmarks, template, phases, plots_dir, config)


if __name__ == "__main__":
    main()
