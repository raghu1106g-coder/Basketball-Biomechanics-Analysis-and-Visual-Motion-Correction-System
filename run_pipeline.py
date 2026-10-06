"""
Run the end-to-end basketball shooting analysis pipeline for one video.

Usage:
    python run_pipeline.py --video trimmed/shooter01/side/shooter01_side_01.mp4

The runner reuses existing landmark CSVs and templates when available. When
needed, it extracts missing metadata shots so auto template selection can use
made, good-quality reference shots.
"""

import argparse
import json
import os

from src.build_template import auto_select_reference_shots, build_template
from src.deviation import compute_deviation, load_template
from src.extract_pose import process_batch, process_video
from src.feedback import generate_feedback
from src.features import compute_features
from src.segment_phases import segment_phases
from src.utils import load_config, load_landmarks
from src.visualize import (
    _correct_joint_angles,
    render_angle_plots,
    render_correction_video,
    render_overlay_video,
)


def landmark_path_for_video(video_path, landmark_dir):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(landmark_dir, basename + "_landmarks.csv")


def ensure_landmarks(video_path, landmark_path, config):
    if not os.path.exists(landmark_path):
        process_video(video_path, landmark_path, config)
    return landmark_path


def ensure_template(metadata_path, landmark_dir, template_path, config):
    if os.path.exists(template_path):
        return template_path

    reference_paths = auto_select_reference_shots(
        metadata_path, landmark_dir, config
    )
    if not reference_paths:
        raise FileNotFoundError(
            "No reference landmarks found. Run extraction with --metadata first."
        )
    build_template(reference_paths, config, template_path)
    return template_path


def run_pipeline(video_path, output_dir="outputs", metadata_path="metadata.csv",
                 landmark_dir="data/processed/landmarks",
                 template_path="data/expert_template.csv", config_path=None):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(landmark_dir, exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError("Video not found: {}".format(video_path))
    if not os.path.exists(config["extraction"]["model_path"]):
        raise FileNotFoundError(
            "Pose model not found: {}".format(config["extraction"]["model_path"])
        )

    # Auto templates require reference shots. Extract missing metadata shots once.
    if not os.path.exists(template_path) and os.path.exists(metadata_path):
        process_batch(metadata_path, landmark_dir, config=config)

    landmark_path = ensure_landmarks(
        video_path, landmark_path_for_video(video_path, landmark_dir), config
    )
    template_path = ensure_template(
        metadata_path, landmark_dir, template_path, config
    )

    landmarks = load_landmarks(landmark_path)
    phases = segment_phases(landmarks, config=config)
    features = compute_features(landmarks, config=config)
    template = load_template(template_path)
    deviation = compute_deviation(landmarks, template, config=config)
    feedback = generate_feedback(deviation["faults"])

    base = os.path.splitext(os.path.basename(landmark_path))[0]
    faults_path = os.path.join(output_dir, base + "_faults.json")
    with open(faults_path, "w") as file:
        json.dump(deviation, file, indent=2)

    overlay_path = os.path.join(output_dir, base + "_overlay.mp4")
    render_overlay_video(video_path, landmarks, deviation["faults"], overlay_path, config)

    corrected = _correct_joint_angles(landmarks, template, deviation["faults"], config)
    correction_path = os.path.join(output_dir, base + "_correction.mp4")
    render_correction_video(
        landmarks, corrected, deviation["faults"], correction_path, config
    )

    plots_dir = os.path.join(output_dir, base + "_plots")
    plot_paths = render_angle_plots(
        landmarks, template, phases, plots_dir, config
    )

    feedback_path = os.path.join(output_dir, base + "_feedback.txt")
    with open(feedback_path, "w", encoding="utf-8") as file:
        file.write(feedback)

    summary = {
        "video": video_path,
        "landmarks": landmark_path,
        "template": template_path,
        "features": features,
        "phases": {
            key: phases[key]
            for key in ("stance", "load", "release", "followthrough", "end")
        },
        "deviation": deviation,
        "feedback": feedback,
        "artifacts": {
            "faults": faults_path,
            "overlay_video": overlay_path,
            "correction_video": correction_path,
            "plots": plot_paths,
            "feedback": feedback_path,
        },
    }
    summary_path = os.path.join(output_dir, base + "_summary.json")
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    print("Pipeline complete.")
    print("Summary: {}".format(summary_path))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run basketball shot analysis.")
    parser.add_argument("--video", required=True, help="Input MP4 video path.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--landmark-dir", default="data/processed/landmarks")
    parser.add_argument("--template", default="data/expert_template.csv")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    run_pipeline(
        args.video,
        args.output,
        args.metadata,
        args.landmark_dir,
        args.template,
        args.config,
    )


if __name__ == "__main__":
    main()
