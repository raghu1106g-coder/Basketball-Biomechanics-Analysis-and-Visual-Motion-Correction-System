# Basketball Shooting Biomechanics

An end-to-end Python and React system for extracting basketball shooting landmarks, measuring joint mechanics, comparing shots with a reference template, identifying faults, and rendering visual feedback.

## Environment

The project is pinned for Python 3.8 and uses the local virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The MediaPipe model must remain at `pose_landmarker.task`. FFmpeg must be installed and available on PATH for the legacy trimming script.

## Pipeline

Run the complete analysis for a trimmed video:

```powershell
python run_pipeline.py --video trimmed/shooter01/side/shooter01_side_01.mp4 --output outputs
```

The runner extracts missing landmark CSVs, auto-selects made/good reference shots, builds `data/expert_template.csv`, calculates phases and features, detects deviations, generates feedback, and writes overlay/correction videos and angle plots.

Standalone modules are available with `python -m src.<module>`:

```powershell
python -m src.extract_pose --metadata metadata.csv
python -m src.features --batch metadata.csv
python -m src.build_template --auto --metadata metadata.csv
python -m src.deviation --landmarks data/processed/landmarks/shooter01_side_1.csv --template data/expert_template.csv
```

## API

Start the Flask server:

```powershell
python app.py
```

Check it:

```powershell
curl http://127.0.0.1:5000/health
```

Analyze a video:

```powershell
curl -X POST -F "video=@trimmed/shooter01/side/shooter01_side_01.mp4" http://127.0.0.1:5000/analyze
```

The API returns JSON with features, phases, faults, feedback, and URLs for generated artifacts. Output files are served from `/outputs/`.

## Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The Vite development server proxies `/analyze` and `/outputs` to Flask on port 5000.

For a production build:

```powershell
npm run build
```

## Project phases

- Phases 1–9: modular extraction, segmentation, features, templates, deviation, visualization, feedback, augmentation, and configuration.
- Phase 10: `run_pipeline.py` orchestrator.
- Phase 11: `app.py` Flask API.
- Phase 12: `frontend/` React + Vite interface.
- Phase 13: this documentation and dependency updates.
