import json
import os

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_landmark_csv_exists():
    path = os.path.join(ROOT, "data", "processed", "landmarks", "shooter01_side_01_landmarks.csv")
    assert os.path.exists(path)
    assert len(pd.read_csv(path)) > 0


def test_features_have_numeric_shot_rows():
    features = pd.read_csv(os.path.join(ROOT, "features.csv"))
    assert len(features) >= 6
    assert features["elbow_angle_load"].notna().all()
    assert features["knee_flex_min"].notna().all()


def test_deviation_report_has_joint_scores():
    path = os.path.join(ROOT, "outputs", "shooter01_side_01_landmarks_faults.json")
    with open(path) as file:
        report = json.load(file)
    assert set(report["per_joint_zscore"]) >= {"elbow", "knee"}
    assert isinstance(report["faults"], list)


def test_phase_artifacts_exist():
    plot_dir = os.path.join(ROOT, "outputs", "shooter01_side_01_landmarks_plots")
    assert os.path.exists(os.path.join(plot_dir, "elbow_angle.png"))
    assert os.path.exists(os.path.join(plot_dir, "knee_angle.png"))


def test_api_health_payload():
    from app import app

    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"