"""Flask API for the basketball shooting analysis pipeline."""

import os
import tempfile
import uuid

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from run_pipeline import run_pipeline


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def public_output_url(path):
    if not path:
        return None
    return "/outputs/" + os.path.relpath(path, OUTPUT_DIR).replace(os.sep, "/")


def public_summary(summary):
    artifacts = summary.get("artifacts", {})
    result = dict(summary)
    result["artifacts"] = dict(artifacts)
    for key in ("faults", "overlay_video", "correction_video", "feedback"):
        if key in result["artifacts"]:
            result["artifacts"][key] = public_output_url(result["artifacts"][key])
    result["artifacts"]["plots"] = {
        name: public_output_url(path)
        for name, path in result["artifacts"].get("plots", {}).items()
    }
    return result


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "basketball-biomechanics"})


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.post("/analyze")
def analyze():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "Upload a video using the 'video' field."}), 400
    if not allowed_file(video.filename):
        return jsonify({"error": "Supported formats: mp4, mov, avi, mkv."}), 400

    job_id = uuid.uuid4().hex[:10]
    extension = os.path.splitext(secure_filename(video.filename))[1].lower()
    upload_path = os.path.join(UPLOAD_DIR, job_id + extension)
    job_output = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output, exist_ok=True)
    video.save(upload_path)

    try:
        summary = run_pipeline(
            upload_path,
            output_dir=job_output,
            metadata_path=os.path.join(ROOT, "metadata.csv"),
            landmark_dir=os.path.join(ROOT, "data", "processed", "landmarks"),
            template_path=os.path.join(ROOT, "data", "expert_template.csv"),
        )
        return jsonify({
            "job_id": job_id,
            "features": summary["features"],
            "phases": summary["phases"],
            "deviation_scores": summary["deviation"],
            "faults": summary["deviation"]["faults"],
            "feedback": summary["feedback"],
            "artifacts": {
                key: (
                    {name: "/outputs/" + os.path.relpath(path, OUTPUT_DIR).replace(os.sep, "/")
                     for name, path in value.items()}
                    if key == "plots" else public_output_url(value)
                )
                for key, value in summary["artifacts"].items()
            },
        })
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify({"error": str(exc), "job_id": job_id}), 500
    finally:
        try:
            os.remove(upload_path)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
