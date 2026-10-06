"""
Shared utilities for the Basketball Shooting Biomechanics pipeline.

Provides:
    - YAML config loading
    - Joint-angle computation
    - Landmark I/O (CSV read/write)
    - Limb-length computation
    - Forward-kinematics helper
    - MediaPipe landmark index constants
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# ── MediaPipe Pose landmark indices ────────────────────────────────────────
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
NOSE = 0
LEFT_SHOULDER = 11;  RIGHT_SHOULDER = 12
LEFT_ELBOW    = 13;  RIGHT_ELBOW    = 14
LEFT_WRIST    = 15;  RIGHT_WRIST    = 16
LEFT_HIP      = 23;  RIGHT_HIP      = 24
LEFT_KNEE     = 25;  RIGHT_KNEE     = 26
LEFT_ANKLE    = 27;  RIGHT_ANKLE    = 28

# Dominant-side defaults (right-handed shooter → right-side landmarks)
SHOULDER = RIGHT_SHOULDER
ELBOW    = RIGHT_ELBOW
WRIST    = RIGHT_WRIST
HIP      = RIGHT_HIP
KNEE     = RIGHT_KNEE
ANKLE    = RIGHT_ANKLE

NUM_LANDMARKS = 33
LANDMARK_COLS_PER_LM = 4  # x, y, z, visibility

# MediaPipe Pose skeleton connections for drawing
POSE_CONNECTIONS = [
    (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),   # right arm
    (11, 12),             # shoulders
    (11, 23), (12, 24),   # torso
    (23, 24),             # hips
    (23, 25), (25, 27),   # left leg
    (24, 26), (26, 28),   # right leg
    (15, 17), (15, 19), (15, 21),  # left hand
    (16, 18), (16, 20), (16, 22),  # right hand
    (27, 29), (27, 31),   # left foot
    (28, 30), (28, 32),   # right foot
]

# ── Config ──────────────────────────────────────────────────────────────────

_CONFIG_CACHE = {}

def load_config(config_path=None):
    """Load rules.yaml.  Result is cached after first read."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "rules.yaml"
        )
    config_path = os.path.abspath(config_path)
    if config_path not in _CONFIG_CACHE:
        with open(config_path, "r") as f:
            _CONFIG_CACHE[config_path] = yaml.safe_load(f)
    return _CONFIG_CACHE[config_path]


# ── Geometry helpers ────────────────────────────────────────────────────────

def angle_between(a, b, c):
    """
    Angle at vertex *b* formed by points a-b-c, in degrees.

    Parameters
    ----------
    a, b, c : array-like, shape (2,) or (3,)

    Returns
    -------
    float  – angle in [0, 180]
    """
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def angle_series(landmarks, a_idx, b_idx, c_idx):
    """
    Compute angle at vertex b_idx for every frame.

    Parameters
    ----------
    landmarks : ndarray, shape (T, 33, 4)  — x, y, z, vis
    a_idx, b_idx, c_idx : int — landmark indices

    Returns
    -------
    ndarray, shape (T,)
    """
    T = landmarks.shape[0]
    angles = np.empty(T)
    for i in range(T):
        angles[i] = angle_between(
            landmarks[i, a_idx, :3],
            landmarks[i, b_idx, :3],
            landmarks[i, c_idx, :3],
        )
    return angles


def angular_velocity(angle_ts, fps=30.0):
    """First derivative of angle time-series (deg/s)."""
    dt = 1.0 / fps
    return np.gradient(angle_ts, dt)


def angular_acceleration(angle_ts, fps=30.0):
    """Second derivative of angle time-series (deg/s^2)."""
    dt = 1.0 / fps
    vel = np.gradient(angle_ts, dt)
    return np.gradient(vel, dt)


def limb_length(landmarks_frame, idx_a, idx_b):
    """Euclidean distance between two landmarks in a single frame."""
    a = landmarks_frame[idx_a, :3]
    b = landmarks_frame[idx_b, :3]
    return float(np.linalg.norm(a - b))


def average_limb_lengths(landmarks, pairs):
    """
    Average limb lengths over all frames for given index pairs.

    Parameters
    ----------
    landmarks : ndarray (T, 33, 4)
    pairs : list of (int, int)

    Returns
    -------
    dict  {(a,b): float}
    """
    T = landmarks.shape[0]
    result = {}
    for (a, b) in pairs:
        lengths = [limb_length(landmarks[i], a, b) for i in range(T)]
        result[(a, b)] = float(np.mean(lengths))
    return result


# ── Forward kinematics (2-D, for correction skeleton) ──────────────────────

def forward_kinematics_chain(root_pos, angles_deg, lengths):
    """
    Given a root position, a chain of angles (absolute, measured from
    vertical-down in screen coords where y increases downward), and
    segment lengths, return the position of each joint.

    Parameters
    ----------
    root_pos : (2,) starting position
    angles_deg : list of float — one per segment
    lengths : list of float — one per segment

    Returns
    -------
    list of ndarray (2,) — positions including root
    """
    positions = [np.array(root_pos, dtype=np.float64)]
    for ang, ln in zip(angles_deg, lengths):
        rad = np.radians(ang)
        dx = ln * np.sin(rad)
        dy = ln * np.cos(rad)   # y-down convention
        positions.append(positions[-1] + np.array([dx, dy]))
    return positions


# ── Landmark I/O ────────────────────────────────────────────────────────────

def landmarks_to_dataframe(landmarks, fps=30.0):
    """
    Convert (T, 33, 4) ndarray → DataFrame with columns:
        frame_idx, timestamp_ms, lm_0_x, lm_0_y, lm_0_z, lm_0_vis, …, lm_32_vis
    """
    T = landmarks.shape[0]
    records = []
    for i in range(T):
        row = {"frame_idx": i, "timestamp_ms": round(i * 1000.0 / fps, 1)}
        for j in range(NUM_LANDMARKS):
            row["lm_{}_x".format(j)]   = landmarks[i, j, 0]
            row["lm_{}_y".format(j)]   = landmarks[i, j, 1]
            row["lm_{}_z".format(j)]   = landmarks[i, j, 2]
            row["lm_{}_vis".format(j)] = landmarks[i, j, 3]
        records.append(row)
    return pd.DataFrame(records)


def dataframe_to_landmarks(df):
    """
    Convert landmark CSV DataFrame → (T, 33, 4) ndarray.
    """
    T = len(df)
    landmarks = np.zeros((T, NUM_LANDMARKS, 4), dtype=np.float64)
    for j in range(NUM_LANDMARKS):
        landmarks[:, j, 0] = df["lm_{}_x".format(j)].values
        landmarks[:, j, 1] = df["lm_{}_y".format(j)].values
        landmarks[:, j, 2] = df["lm_{}_z".format(j)].values
        landmarks[:, j, 3] = df["lm_{}_vis".format(j)].values
    return landmarks


def save_landmarks_csv(landmarks, output_path, fps=30.0):
    """Save (T,33,4) landmarks to CSV."""
    df = landmarks_to_dataframe(landmarks, fps)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_landmarks_csv(csv_path):
    """Load CSV → (T, 33, 4) ndarray."""
    df = pd.read_csv(csv_path)
    return dataframe_to_landmarks(df)


def load_landmarks_npy(npy_path):
    """Load legacy .npy → (T, 33, 4) ndarray."""
    return np.load(npy_path)


def load_landmarks(path):
    """Auto-detect CSV or NPY and load landmarks."""
    if path.endswith(".npy"):
        return load_landmarks_npy(path)
    else:
        return load_landmarks_csv(path)


# ── Normalise time-series to fixed length ──────────────────────────────────

def normalize_timeseries(ts, target_length=100):
    """
    Resample a 1-D time-series to *target_length* using linear interpolation.
    """
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    if n == 0:
        return np.zeros(target_length)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, ts)
