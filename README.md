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

## Notes

- Uploaded files are stored under `uploads/` (ignored by git).
- Generated charts are written to `static/images/` (images ignored by git).
