# Dataset Optimiser

A Flask web app to upload a CSV dataset, profile data quality, generate charts, optionally clean/optimize the dataset, optionally run a simple ML evaluation (classification or regression), and download results.

## Features

- CSV upload + profiling (rows/cols, missing values, duplicates, dtypes, memory)
- Auto-generated charts (missing distribution, histograms, correlation heatmap, etc.)
- Cleaning strategies for missing values (drop / mean+mode / KNN)
- Optimization options (dtype optimization, outliers, one-hot encoding, scaling, correlated-drop, PCA, feature selection, SMOTE/oversampling)
- Model evaluation (auto-detects classification vs regression)
- Download processed CSV + download a PDF report

## Run locally

```bash
# create venv (optional)
python -m venv venv

# activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# run
python app.py
```

Then open:

- http://127.0.0.1:5000

- http://127.0.0.1:5000

## Deploy as a website (recommended)

This app is deploy-ready on common platforms. It binds to `0.0.0.0:$PORT` and includes a production server entrypoint in `serve.py`.

### Render

1. Push this repo to GitHub.
2. Render → **New** → **Web Service** → connect this repo.
3. Set:
	- Build command: `pip install -r requirements.txt`
	- Start command: `python serve.py`
4. Deploy.

### Railway

1. Railway → **New Project** → **Deploy from GitHub repo**.
2. Set the start command to: `python serve.py` (if it doesn’t auto-detect).
3. Deploy.

### Notes about storage

- `uploads/` and generated images under `static/images/` are stored on the server filesystem.
- On many free/cheap hosting plans the filesystem is **ephemeral** (resets on redeploy), so treat uploads as temporary.

## Notes

- Uploaded files are stored under `uploads/` (ignored by git).
- Generated charts are written to `static/images/` (images ignored by git).
