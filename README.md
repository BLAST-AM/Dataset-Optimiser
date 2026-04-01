# Dataset Optimiser

A Flask web app to upload a CSV, generate a quick quality report (metrics + charts), optionally clean/optimize the dataset, run a simple ML evaluation (classification or regression), and download outputs.

## Features

- CSV upload + profiling (rows/cols, missing values, duplicates, dtypes, memory)
- Auto-generated charts (missing distribution, histograms, correlation heatmap, etc.)
- Cleaning strategies for missing values (drop / mean+mode / KNN)
- Optimization options (dtype optimization, outliers, encoding, scaling, PCA, feature selection, imbalance handling)
- Model evaluation (auto-detects classification vs regression)
- Download processed CSV + PDF report

## Run locally

```bash
# install deps
pip install -r requirements.txt

# run
python app.py
```

Open http://127.0.0.1:5000


## Notes

- Uploaded CSVs are stored under `uploads/` (temporary on most hosts).
- Generated charts are written to `static/images/`.
- On free/cheap hosting plans the filesystem can be ephemeral, so treat uploads/outputs as temporary.
