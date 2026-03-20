"""Dataset Optimiser (Flask)

This file intentionally keeps all features working as before, but is now organized
into clear sections with helper functions to reduce repetition.

High-level flow:
- `/` shows the upload page.
- `/upload` reads a CSV and renders the dashboard report.
- `/clean` runs missing-value cleaning and returns an export page.
- `/optimize` runs (simple) optimizations and returns an export page.
- `/evaluate` optionally computes a confusion matrix (only when viable).

Notes for future changes:
- Keep template variable names stable (they are referenced in Jinja templates).
- Plot generation writes images into `static/images/`.
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, redirect, render_template, request, send_file
from werkzeug.utils import secure_filename

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

# Matplotlib: force a non-GUI backend (required for server environments)
plt.switch_backend('Agg')

app = Flask(__name__)

# ------------------------------
# App configuration
# ------------------------------

UPLOAD_FOLDER = 'uploads'
STATIC_IMAGE_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'csv'}

# Dashboard/model-evaluation constraints (keep these aligned with UI expectations)
DEFAULT_EVAL_TEST_SIZE = 0.2
MAX_CONFUSION_MATRIX_CLASSES = 20

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_IMAGE_FOLDER'] = STATIC_IMAGE_FOLDER

# Ensure directories exist at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGE_FOLDER, exist_ok=True)


# ------------------------------
# Small file/path helpers
# ------------------------------

def _safe_stem(filename: str) -> str:
    """Return a safe filename stem for image outputs."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in stem)
    return stem or 'dataset'

def _save_current_figure(image_filename: str) -> str:
    """Save the current Matplotlib figure into `static/images/` and return the filename."""
    plot_path = os.path.join(app.config['STATIC_IMAGE_FOLDER'], image_filename)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return image_filename


def _read_uploaded_csv(filename: str) -> pd.DataFrame:
    """Load a previously uploaded CSV from disk.

    Centralizing this makes it easier to enforce consistent read options later.
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return _read_csv_safely(filepath)


def _read_csv_safely(filepath: str) -> pd.DataFrame:
    """Best-effort CSV reader.

    Handles common real-world issues:
    - Unknown delimiter (uses pandas sniffing)
    - UTF-8 vs Latin-1 encoding

    This intentionally stays conservative (no silent type coercion beyond pandas defaults).
    """
    read_kwargs: dict[str, Any] = {
        'sep': None,
        'engine': 'python',
        'low_memory': False,
    }

    last_err: Exception | None = None
    for enc in ('utf-8', 'utf-8-sig', 'latin1'):
        try:
            return pd.read_csv(filepath, encoding=enc, **read_kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # If sniffing fails, try a standard comma-separated read for this encoding.
            last_err = e
            try:
                return pd.read_csv(filepath, encoding=enc, sep=',', engine='c', low_memory=False)
            except Exception as e2:
                last_err = e2
                continue

    raise last_err or RuntimeError('Failed to read CSV.')


def _eval_capabilities(
    df: pd.DataFrame,
    default_test_size: float = DEFAULT_EVAL_TEST_SIZE,
    max_classes: int = MAX_CONFUSION_MATRIX_CLASSES,
) -> tuple[bool, list[str], str | None]:
    """Decide which columns are viable evaluation targets (classification or regression).

    Returns a tuple of (possible, targets, reason).
    """
    if df is None or df.empty or df.shape[0] < 6 or df.shape[1] < 2:
        return False, [], 'Dataset is too small for evaluation.'

    candidates: list[str] = []
    n_rows = int(df.shape[0])
    min_test_rows = max(2, int(np.ceil(n_rows * float(default_test_size))))
    if min_test_rows < 2:
        return False, [], 'Not enough rows for a train/test split.'

    for col in df.columns:
        y = df[col].dropna()
        if len(y) < 6:
            continue
        nunique = int(y.nunique(dropna=True))
        if nunique < 2:
            continue

        # Classification-like: small number of classes and enough samples per class.
        if nunique <= max_classes:
            y_enc = y.astype(str)
            counts = y_enc.value_counts()
            if counts.min() >= 2 and int(np.ceil(len(y_enc) * float(default_test_size))) >= nunique:
                candidates.append(str(col))
                continue

        # Regression-like: numeric and reasonably high cardinality.
        if pd.api.types.is_numeric_dtype(y) and nunique > max_classes:
            candidates.append(str(col))

    if candidates:
        return True, sorted(list(dict.fromkeys(candidates))), None
    return False, [], 'No suitable target column found (needs either a classification-like label or a numeric regression target).'


def _infer_task_type(y: pd.Series, max_classes: int = MAX_CONFUSION_MATRIX_CLASSES) -> str:
    """Infer whether a target should be treated as classification or regression."""
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return 'unknown'
    nunique = int(y_nonnull.nunique(dropna=True))
    if nunique < 2:
        return 'unknown'
    if (not pd.api.types.is_numeric_dtype(y_nonnull)) or nunique <= max_classes:
        return 'classification'
    return 'regression'


def _build_recommendations(df: pd.DataFrame) -> list[str]:
    recs: list[str] = []

    if df is None or df.empty:
        return recs

    rows, cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    missing_pct = (missing_total / (rows * cols) * 100) if (rows * cols) else 0.0
    dup_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if missing_total > 0:
        if len(numeric_cols) > 0 and missing_pct <= 15:
            recs.append('Missing values detected: try KNN Imputation for numeric columns (keeps relationships).')
        else:
            recs.append('Missing values detected: try Simple Imputation (Mean/Mode) or Drop Rows for a quick cleanup.')
    else:
        recs.append('No missing values: you can run optimization/feature engineering safely.')

    if dup_rows > 0:
        recs.append(f'Duplicate rows detected ({dup_rows}). Consider removing duplicates before modeling.')

    mem_mb = float(df.memory_usage(deep=True).sum()) / (1024**2)
    if mem_mb >= 5:
        recs.append('Dataset is moderately large: enable Data Type Optimization to reduce memory usage.')

    if len(numeric_cols) >= 2:
        recs.append('Enable “Remove Redundant Features” to drop highly correlated numeric columns.')

    # Skew hint
    skewed = 0
    for c in numeric_cols[:20]:
        s = df[c].dropna()
        if len(s) >= 20:
            sk = s.skew()
            if pd.notna(sk) and abs(float(sk)) >= 1.0:
                skewed += 1
    if skewed > 0:
        recs.append('Skewed numeric columns detected: consider “Log Transform Skewed Numeric Columns”.')

    if len(cat_cols) > 0:
        recs.append('Categorical columns detected: enable one-hot encoding if you plan to train ML models.')

    return recs

def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Heuristically detect a datetime-like column (best-effort)."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.select_dtypes(include=['object']).columns:
        parsed = pd.to_datetime(df[col], errors='coerce')
        non_null_ratio = parsed.notna().mean()
        if non_null_ratio >= 0.6 and parsed.nunique(dropna=True) >= 5:
            return col
    return None

def generate_additional_visualizations(df: pd.DataFrame, filename: str) -> dict:
    """Generate additional charts requested for the analysis page.

    Returns a dict of {key: image_filename}. Keys are stable for templates.
    """
    images: dict[str, str] = {}
    stem = _safe_stem(filename)
    nonce = str(int(time.time() * 1000))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1) Categorical bar chart (top categories)
    if len(categorical_cols) > 0:
        cat_col = None
        for c in categorical_cols:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 30:
                cat_col = c
                break
        if cat_col is None:
            cat_col = categorical_cols[0]

        plt.figure(figsize=(10, 6))
        counts = df[cat_col].astype(str).fillna('Missing').value_counts().head(10)
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='deep', legend=False)
        plt.title(f'Top Categories in {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        images['categorical_bar'] = _save_current_figure(f"cat_{stem}_{nonce}.png")

    # 2) Time-series line graph (if a datetime-like column exists)
    dt_col = _detect_datetime_column(df)
    if dt_col is not None and len(numeric_cols) > 0:
        parsed = pd.to_datetime(df[dt_col], errors='coerce')
        ts = df.copy()
        ts[dt_col] = parsed
        ts = ts.dropna(subset=[dt_col])
        if not ts.empty:
            y_col = numeric_cols[0]
            ts['_date'] = ts[dt_col].dt.date
            series = ts.groupby('_date')[y_col].mean(numeric_only=True)
            if len(series) >= 2:
                plt.figure(figsize=(10, 4))
                plt.plot(series.index, series.values, marker='o', linewidth=2)
                plt.title(f'{y_col} Trend Over Time ({dt_col})')
                plt.xlabel('Date')
                plt.ylabel(f'Average {y_col}')
                plt.xticks(rotation=45, ha='right')
                images['time_series_line'] = _save_current_figure(f"line_{stem}_{nonce}.png")

    # 3) Histograms for numerical distributions (up to 2 columns)
    if len(numeric_cols) > 0:
        cols_to_plot = numeric_cols[:2]
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(10, 4))
        if len(cols_to_plot) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            data = df[col].dropna()
            ax.hist(data, bins=20, color='#4c78a8', alpha=0.85)
            ax.set_title(f'Histogram: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        images['numerical_hist'] = _save_current_figure(f"hist_{stem}_{nonce}.png")

    # 4) Scatter plot to identify correlations (top 2 numeric columns)
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.7)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        images['scatter_corr'] = _save_current_figure(f"scatter_{stem}_{nonce}.png")

    # 5) Box plot for distribution & outliers (numeric columns, up to 10)
    if len(numeric_cols) > 0:
        cols_to_plot = numeric_cols[:10]
        plt.figure(figsize=(10, max(3, 0.5 * len(cols_to_plot) + 1)))
        sns.boxplot(data=df[cols_to_plot], orient='h')
        plt.title('Box Plot (Distribution & Outliers)')
        images['boxplot'] = _save_current_figure(f"box_{stem}_{nonce}.png")

    # 6) Heatmap / correlation matrix for numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap (Correlation Matrix)')
        images['corr_heatmap'] = _save_current_figure(f"heatmap_{stem}_{nonce}.png")

    return images

def compute_dataset_details(df: pd.DataFrame) -> dict:
    """Compute summary stats shown in the dashboard "Dataset Details" panel."""
    rows, cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    missing_pct = float((missing_total / (rows * cols) * 100) if (rows * cols) else 0)
    dup_rows = int(df.duplicated().sum())
    mem_bytes = int(df.memory_usage(deep=True).sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    dt_col = _detect_datetime_column(df)

    top_missing = (
        df.isnull().sum()
        .sort_values(ascending=False)
        .head(8)
    )
    top_missing = [(k, int(v)) for k, v in top_missing.items() if int(v) > 0]

    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    dtype_counts = {str(k): int(v) for k, v in dtype_counts.items()}

    return {
        'rows': int(rows),
        'cols': int(cols),
        'missing_total': missing_total,
        'missing_pct': round(missing_pct, 2),
        'duplicate_rows': dup_rows,
        'memory_mb': round(mem_bytes / 1024 / 1024, 2),
        'numeric_features': int(len(numeric_cols)),
        'categorical_features': int(len(categorical_cols)),
        'datetime_detected': dt_col,
        'dtype_counts': dtype_counts,
        'top_missing': top_missing,
    }

def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, out_filename: str) -> str:
    """Render and save a confusion matrix heatmap."""
    plt.clf()
    plt.figure(figsize=(8, 6))
    annotate = len(labels) <= 10
    sns.heatmap(
        cm,
        annot=annotate,
        fmt='d',
        cmap='mako',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    return _save_current_figure(out_filename)

def _eval_capability_for_confusion_matrix(
    df: pd.DataFrame,
    default_test_size: float = DEFAULT_EVAL_TEST_SIZE,
    max_classes: int = MAX_CONFUSION_MATRIX_CLASSES,
) -> tuple[bool, list[str], str | None]:
    """Decide if any column can reasonably be used for a confusion matrix.

    This mirrors the /evaluate constraints (classification-like targets with a small
    number of classes and enough samples to do a stratified split).
    """
    candidates: list[str] = []

    if df is None or df.empty or df.shape[0] < 4 or df.shape[1] < 2:
        return False, [], 'Dataset is too small for evaluation.'

    # Conservative: choose viability for default test size (UI default is 20%).
    # We require at least 1 sample per class in the test set and at least 2 samples per class overall.
    for col in df.columns:
        y = df[col].dropna()
        if y.empty:
            continue

        # Treat high-cardinality numeric columns as non-classification labels.
        nunique = int(y.nunique(dropna=True))
        if nunique < 2 or nunique > max_classes:
            continue

        y_enc = y.astype(str)
        counts = y_enc.value_counts()
        if counts.min() < 2:
            continue

        n_samples = int(len(y_enc))
        test_n = int(np.ceil(n_samples * float(default_test_size)))
        if test_n < nunique:
            continue
        if (counts * float(default_test_size)).min() < 1:
            continue

        candidates.append(str(col))

    if candidates:
        return True, candidates, None

    # Provide a best-effort reason (used for helpful errors)
    if df.shape[0] < 10:
        return False, [], 'Not enough rows for a reliable train/test split.'
    return False, [], f'No suitable target column found (needs 2–{max_classes} classes with enough samples per class).'


# ------------------------------
# Report rendering helpers (DRY)
# ------------------------------

def _build_report_context(
    df: pd.DataFrame,
    filename: str,
    *,
    eval_result: dict[str, Any] | None,
    cm_image: str | None,
    eval_image: str | None = None,
    eval_image_title: str | None = None,
) -> dict[str, Any]:
    """Build the Jinja context used by `results.html`.

    Keeping this in one place prevents subtle drift when adding features.
    """
    rows, cols = df.shape
    missing_values = int(df.isnull().sum().sum())
    columns = df.columns.tolist()

    dataset_details = compute_dataset_details(df)
    eval_possible, eval_targets, _eval_reason = _eval_capabilities(df)
    recommendations = _build_recommendations(df)

    head_data = df.head(5).to_html(classes='table table-striped table-bordered', index=False)
    plot_image = generate_plot(df, filename)
    extra_images = generate_additional_visualizations(df, filename)

    return {
        'filename': filename,
        'rows': rows,
        'cols': cols,
        'missing': missing_values,
        'dataset_details': dataset_details,
        'columns': columns,
        'table': head_data,
        'plot_image': plot_image,
        'extra_images': extra_images,
        'eval_possible': eval_possible,
        'eval_targets': eval_targets,
        'eval_result': eval_result,
        'cm_image': cm_image,
        'eval_image': eval_image,
        'eval_image_title': eval_image_title,
        'recommendations': recommendations,
    }


def _plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_filename: str,
) -> str:
    """Create a simple regression plot (Actual vs Predicted) and persist it as an image."""
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    # identity line
    try:
        lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        if np.isfinite(lo) and np.isfinite(hi):
            plt.plot([lo, hi], [lo, hi], linestyle='--', color='white', linewidth=1)
    except Exception:
        pass
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    return _save_current_figure(out_filename)

def _memory_optimize(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    before = df.memory_usage(deep=True).sum()
    out = df.copy()

    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        col_data = out[col]
        if pd.api.types.is_float_dtype(col_data):
            out[col] = pd.to_numeric(col_data, downcast='float')
        elif pd.api.types.is_integer_dtype(col_data):
            out[col] = pd.to_numeric(col_data, downcast='integer')

    obj_cols = out.select_dtypes(include=['object']).columns
    for col in obj_cols:
        nunique = out[col].nunique(dropna=True)
        total = len(out[col])
        # Convert to category when it looks categorical (low cardinality)
        if total > 0 and (nunique / total) <= 0.5:
            out[col] = out[col].astype('category')

    after = out.memory_usage(deep=True).sum()
    saved_pct = (before - after) / before * 100 if before else 0
    note = f"Memory optimized: {before/1024**2:.2f}MB -> {after/1024**2:.2f}MB ({saved_pct:.1f}% saved)."
    return out, note

def _winsorize_outliers(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    changed = 0
    for col in num_cols:
        s = out[col]
        if s.dropna().empty:
            continue
        low = s.quantile(lower_q)
        high = s.quantile(upper_q)
        out[col] = s.clip(lower=low, upper=high)
        changed += 1
    return out, f"Outliers capped for {changed} numeric columns (q{int(lower_q*100)}–q{int(upper_q*100)})."

def _log_transform_skewed(df: pd.DataFrame, skew_threshold: float = 1.0) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    transformed = []
    for col in num_cols:
        s = out[col].dropna()
        if s.empty:
            continue
        skew = s.skew()
        if pd.isna(skew) or abs(skew) < skew_threshold:
            continue
        min_val = s.min()
        if min_val <= 0:
            out[col] = np.log1p(out[col] - min_val)
        else:
            out[col] = np.log1p(out[col])
        transformed.append(col)
    return out, f"Log transform applied to {len(transformed)} skewed columns." if transformed else "No columns met skew threshold for log transform."

def _bin_first_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return out, "No numeric columns available for binning."
    col = num_cols[0]
    try:
        out[f"{col}_bin"] = pd.qcut(out[col], q=4, duplicates='drop')
        return out, f"Binning added: {col}_bin (quartiles from {col})."
    except Exception:
        return out, f"Binning skipped for {col} (insufficient unique values)."

def _one_hot_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if not cat_cols:
        return df, "No categorical columns available for one-hot encoding."
    out = pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=True)
    return out, f"One-hot encoded {len(cat_cols)} categorical columns."

def _scale_numeric(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    exclude_cols = exclude_cols or []
    out = df.copy()
    num_cols = [c for c in out.select_dtypes(include=[np.number]).columns.tolist() if c not in exclude_cols]
    if not num_cols:
        return out, "No numeric columns available for scaling."
    scaler = StandardScaler()
    out[num_cols] = scaler.fit_transform(out[num_cols])
    return out, f"Scaled {len(num_cols)} numeric columns (StandardScaler)."

def _drop_correlated(df: pd.DataFrame, threshold: float = 0.9, exclude_cols: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    exclude_cols = set(exclude_cols or [])
    out = df.copy()
    num_cols = [c for c in out.select_dtypes(include=[np.number]).columns.tolist() if c not in exclude_cols]
    if len(num_cols) < 2:
        return out, "Not enough numeric columns for correlation-based reduction."
    corr = out[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    out = out.drop(columns=to_drop, errors='ignore')
    return out, f"Dropped {len(to_drop)} highly correlated columns (>|{threshold}|)." if to_drop else "No highly correlated columns found to drop."

def _apply_pca(df: pd.DataFrame, target_col: str | None = None, variance: float = 0.95) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    exclude = [target_col] if target_col else []
    num_cols = [c for c in out.select_dtypes(include=[np.number]).columns.tolist() if c not in exclude]
    if len(num_cols) < 2:
        return out, "Not enough numeric columns for PCA."
    X = out[num_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=variance, svd_solver='full')
    comps = pca.fit_transform(Xs)
    comp_cols = [f"PCA{i+1}" for i in range(comps.shape[1])]
    out = out.drop(columns=num_cols, errors='ignore')
    out[comp_cols] = comps
    return out, f"PCA applied: reduced {len(num_cols)} numeric cols -> {len(comp_cols)} components (95% variance)."

def _feature_select_rf(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, str]:
    if target not in df.columns:
        return df, "Feature selection skipped (target column not found)."

    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=False, dummy_na=True)
    X = X.fillna(0)

    # Decide classifier vs regressor
    is_classification = (not pd.api.types.is_numeric_dtype(y)) or (y.nunique(dropna=True) <= 20)
    if is_classification:
        y_enc = y.astype(str).fillna('Missing')
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, y_enc)
    else:
        y_num = pd.to_numeric(y, errors='coerce')
        y_num = y_num.fillna(y_num.median())
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, y_num)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    keep_n = min(30, max(5, int(len(importances) * 0.3)))
    keep = importances.head(keep_n).index.tolist()
    selected = X[keep].copy()
    selected[target] = y
    return selected, f"Feature selection kept top {keep_n} features by Random Forest importance."

def _handle_class_imbalance(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, str]:
    if target not in df.columns:
        return df, "Imbalance handling skipped (target column not found)."

    y = df[target]
    # Only attempt for classification-like targets
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return df, "Imbalance handling skipped (target looks continuous)."

    try:
        from imblearn.over_sampling import SMOTE
        X = df.drop(columns=[target])
        X = pd.get_dummies(X, drop_first=False, dummy_na=True)
        X = X.fillna(0)
        y_enc = y.astype(str).fillna('Missing')
        # SMOTE requires at least 2 classes and enough minority samples
        if y_enc.nunique() < 2:
            return df, "Imbalance handling skipped (only one class present)."
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y_enc)
        out = X_res.copy()
        out[target] = y_res
        return out, "Class imbalance handled using SMOTE."
    except Exception:
        # Fallback: simple random oversampling to max class size
        y_enc = y.astype(str).fillna('Missing')
        if y_enc.nunique() < 2:
            return df, "Imbalance handling skipped (only one class present)."
        counts = y_enc.value_counts()
        max_n = counts.max()
        frames = []
        for cls, n in counts.items():
            cls_df = df[y_enc == cls]
            frames.append(cls_df)
            if n < max_n and not cls_df.empty:
                frames.append(cls_df.sample(max_n - n, replace=True, random_state=42))
        out = pd.concat(frames, ignore_index=True)
        return out, "Class imbalance handled using random oversampling (SMOTE unavailable)."

def allowed_file(filename: str) -> bool:
    """Validate upload extension (CSV only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_plot(df: pd.DataFrame, filename: str) -> str:
    """
    Generates a simple bar chart of missing values.
    Saves it to static/images/ and returns the relative path.
    """
    # clear any existing plots to prevent overlap
    plt.clf() 
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot missing values count per column
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0] # Only plot columns with missing info
    
    if not missing_data.empty:
        sns.barplot(x=missing_data.index, y=missing_data.values, hue=missing_data.index, palette='viridis', legend=False)
        plt.title(f'Missing Values in {filename}')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    else:
        # If no missing data, plot a simple row count distribution or empty message
        plt.text(0.5, 0.5, 'No Missing Values Detected!', 
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.title(f'Data Integrity Check: {filename}')

    # Save the plot.
    # NOTE: This intentionally reuses the same filename per dataset stem,
    # so repeated uploads overwrite the previous plot (keeps folder tidy).
    plot_filename = f"plot_{filename.split('.')[0]}.png"
    plot_path = os.path.join(app.config['STATIC_IMAGE_FOLDER'], plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file. Please upload a CSV."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = _read_csv_safely(filepath)
        ctx = _build_report_context(df, filename, eval_result=None, cm_image=None)
        return render_template('results.html', **ctx)
    except Exception as e:
        return f"Error processing file: {e}"

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    filename = request.form.get('filename')
    target = request.form.get('target')
    model_name = request.form.get('model')
    test_size = request.form.get('test_size', '0.2')

    try:
        test_size = float(test_size)
        if not (0.05 <= test_size <= 0.5):
            test_size = 0.2
    except Exception:
        test_size = 0.2

    try:
        df = _read_uploaded_csv(filename)
        eval_possible, eval_targets, eval_reason = _eval_capabilities(df)
        if not eval_possible:
            raise ValueError(eval_reason or 'Model evaluation is not available for this dataset.')

        if not target or target not in df.columns:
            raise ValueError('Please select a valid target column to evaluate.')

        if target not in eval_targets:
            raise ValueError('Selected target is not suitable for evaluation for this dataset.')

        work = df.dropna(subset=[target]).copy()
        if work.shape[0] < 6:
            raise ValueError('Not enough non-missing rows in the target column to evaluate.')

        y_raw = work[target]
        task = _infer_task_type(y_raw)
        if task == 'unknown':
            raise ValueError('Target column is not suitable for evaluation.')

        X = work.drop(columns=[target])

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns.tolist() if c not in numeric_features]

        numeric_pipe = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=False)),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipe, numeric_features),
                ('cat', categorical_pipe, categorical_features),
            ],
            remainder='drop',
            sparse_threshold=0.3,
        )

        stem = _safe_stem(filename)
        nonce = str(int(time.time() * 1000))

        if task == 'classification':
            y = y_raw.astype(str).fillna('Missing')
            class_count = int(y.nunique(dropna=True))
            if class_count < 2:
                raise ValueError('Target must have at least 2 classes for classification.')
            if class_count > MAX_CONFUSION_MATRIX_CLASSES:
                raise ValueError('Too many classes for a readable confusion matrix (max 20).')

            strat = y if y.nunique() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=strat,
            )

            if model_name == 'rf':
                estimator = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
                model_label = 'Random Forest'
            elif model_name == 'svm':
                estimator = SVC(kernel='rbf', gamma='scale')
                model_label = 'SVM (RBF)'
            elif model_name == 'knn':
                estimator = KNeighborsClassifier(n_neighbors=5)
                model_label = 'KNN (k=5)'
            else:
                estimator = LogisticRegression(max_iter=2000)
                model_label = 'Logistic Regression'

            clf = Pipeline(steps=[('prep', preprocessor), ('model', estimator)])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            labels = sorted(list(set(y_test) | set(y_pred)))
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            eval_result = {
                'task': 'classification',
                'target': target,
                'model': model_label,
                'test_size': test_size,
                'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'precision_macro': round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4),
                'recall_macro': round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4),
                'f1_macro': round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
                'classes': labels,
            }

            cm_image = _plot_confusion_matrix(
                cm,
                labels,
                f'Confusion Matrix ({model_label})',
                f'cm_{stem}_{nonce}.png',
            )
            ctx = _build_report_context(df, filename, eval_result=eval_result, cm_image=cm_image)
            return render_template('results.html', **ctx)

        # Regression
        y_num = pd.to_numeric(y_raw, errors='coerce')
        y_num = y_num.dropna()
        if y_num.empty:
            raise ValueError('Target column could not be converted to numeric for regression.')
        work2 = work.loc[y_num.index].copy()
        X2 = work2.drop(columns=[target])
        y2 = y_num

        X_train, X_test, y_train, y_test = train_test_split(
            X2,
            y2,
            test_size=test_size,
            random_state=42,
        )

        if model_name == 'rf':
            estimator = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
            model_label = 'Random Forest Regressor'
        elif model_name == 'svm':
            estimator = SVR(kernel='rbf', gamma='scale')
            model_label = 'SVR (RBF)'
        elif model_name == 'knn':
            estimator = KNeighborsRegressor(n_neighbors=5)
            model_label = 'KNN Regressor (k=5)'
        else:
            estimator = LinearRegression()
            model_label = 'Linear Regression'

        reg = Pipeline(steps=[('prep', preprocessor), ('model', estimator)])
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        eval_result = {
            'task': 'regression',
            'target': target,
            'model': model_label,
            'test_size': test_size,
            'r2': round(float(r2_score(y_test, y_pred)), 4),
            'mae': round(float(mean_absolute_error(y_test, y_pred)), 4),
            'rmse': round(rmse, 4),
        }

        eval_image = _plot_regression_diagnostics(
            np.asarray(y_test),
            np.asarray(y_pred),
            f'Actual vs Predicted ({model_label})',
            f'reg_{stem}_{nonce}.png',
        )
        ctx = _build_report_context(
            df,
            filename,
            eval_result=eval_result,
            cm_image=None,
            eval_image=eval_image,
            eval_image_title='Regression: Actual vs Predicted',
        )
        return render_template('results.html', **ctx)

    except Exception as e:
        df = _read_uploaded_csv(filename)
        ctx = _build_report_context(df, filename, eval_result={'error': str(e)}, cm_image=None)
        return render_template('results.html', **ctx)


@app.route('/report/<filename>')
def download_report(filename: str):
    """Generate a PDF report for an uploaded dataset."""
    try:
        # Lazy import so the server can still run even if reportlab isn't installed yet.
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        df = _read_uploaded_csv(filename)
        ctx = _build_report_context(df, filename, eval_result=None, cm_image=None)

        buf = BytesIO()
        styles = getSampleStyleSheet()
        title = styles['Title']
        h = styles['Heading2']
        body = ParagraphStyle(
            'Body',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=10.5,
            leading=14,
            spaceAfter=8,
        )

        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            topMargin=1.6 * cm,
            bottomMargin=1.6 * cm,
            title=f"Dataset Report — {filename}",
        )

        story = []
        story.append(Paragraph('Dataset Optimiser — Report', title))
        story.append(Paragraph(f"File: {filename}", body))
        story.append(Spacer(1, 10))

        story.append(Paragraph('Summary', h))
        story.append(Paragraph(f"Rows: {ctx['rows']} · Columns: {ctx['cols']} · Missing values: {ctx['missing']}", body))
        dd = ctx.get('dataset_details') or {}
        if dd:
            story.append(Paragraph(
                f"Duplicates: {dd.get('duplicate_rows', '—')} · Memory: {dd.get('memory_mb', '—')} MB · Datetime: {dd.get('datetime_detected') or 'None'}",
                body,
            ))
        story.append(Spacer(1, 6))

        recs = ctx.get('recommendations') or []
        if recs:
            story.append(Paragraph('Suggested next actions', h))
            for r in recs[:10]:
                story.append(Paragraph(f"• {r}", body))
            story.append(Spacer(1, 6))

        def add_image(caption: str, image_filename: str) -> None:
            path = os.path.join(app.config['STATIC_IMAGE_FOLDER'], image_filename)
            if not os.path.exists(path):
                return
            story.append(Paragraph(caption, h))
            img = RLImage(path)
            img.drawWidth = 16.5 * cm
            img.drawHeight = img.drawHeight * (img.drawWidth / img.imageWidth)
            story.append(img)
            story.append(Spacer(1, 12))

        # Main + extra charts
        if ctx.get('plot_image'):
            add_image('Missing value distribution', ctx['plot_image'])

        extra = ctx.get('extra_images') or {}
        captions = {
            'categorical_bar': 'Categorical comparison',
            'time_series_line': 'Time-series trend',
            'numerical_hist': 'Numerical distributions',
            'scatter_corr': 'Scatter plot',
            'boxplot': 'Box plot',
            'corr_heatmap': 'Correlation heatmap',
        }
        for k, cap in captions.items():
            if extra.get(k):
                add_image(cap, extra.get(k))

        pdf_name = f"report_{_safe_stem(filename)}.pdf"
        doc.build(story)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=pdf_name, mimetype='application/pdf')
    except ImportError:
        return 'PDF report requires the reportlab package. Add it to requirements and install it.', 500
    except Exception as e:
        return f"Error generating report: {e}", 500

@app.route('/optimize', methods=['POST'])
def optimize_data():
    filename = request.form.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Options from UI (checkboxes submit value only when checked)
    opt_dtype = request.form.get('opt_dtype') is not None
    opt_onehot = request.form.get('opt_onehot') is not None
    opt_log = request.form.get('opt_log') is not None
    opt_bin = request.form.get('opt_bin') is not None
    opt_outliers = request.form.get('opt_outliers') is not None
    opt_scale = request.form.get('opt_scale') is not None
    opt_corr_drop = request.form.get('opt_corr_drop') is not None
    opt_pca = request.form.get('opt_pca') is not None
    opt_feature_select = request.form.get('opt_feature_select') is not None
    opt_smote = request.form.get('opt_smote') is not None
    target = request.form.get('target') or None

    try:
        df = _read_csv_safely(filepath)
        notes: list[str] = []

        # Feature selection / imbalance handling depend on target.
        # Run those first, because one-hot / PCA etc will change columns.
        if target and opt_smote:
            df, note = _handle_class_imbalance(df, target)
            notes.append(note)

        if target and opt_feature_select:
            df, note = _feature_select_rf(df, target)
            notes.append(note)

        if opt_dtype:
            df, note = _memory_optimize(df)
            notes.append(note)

        if opt_outliers:
            df, note = _winsorize_outliers(df)
            notes.append(note)

        if opt_log:
            df, note = _log_transform_skewed(df)
            notes.append(note)

        if opt_bin:
            df, note = _bin_first_numeric(df)
            notes.append(note)

        if opt_onehot:
            df, note = _one_hot_encode(df)
            notes.append(note)

        # Scaling/correlation/PCA should not include target
        exclude = [target] if target and target in df.columns else []

        if opt_scale:
            df, note = _scale_numeric(df, exclude_cols=exclude)
            notes.append(note)

        if opt_corr_drop:
            df, note = _drop_correlated(df, threshold=0.9, exclude_cols=exclude)
            notes.append(note)

        if opt_pca:
            df, note = _apply_pca(df, target_col=target, variance=0.95)
            notes.append(note)

        opt_filename = "opt_" + filename
        opt_path = os.path.join(app.config['UPLOAD_FOLDER'], opt_filename)
        df.to_csv(opt_path, index=False)

        preview_table = df.head(5).to_html(classes='table table-success table-striped', index=False)
        note = " | ".join(notes) if notes else "No optimization options selected."

        return render_template('export.html',
                               orig_filename=filename,
                               clean_filename=opt_filename,
                               note=note,
                               rows=df.shape[0],
                               table=preview_table)
    except Exception as e:
        return f"Error during optimization: {e}"


@app.route('/clean', methods=['POST'])
def clean_data():
    filename = request.form.get('filename')
    strategy = request.form.get('strategy')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = _read_csv_safely(filepath)
        original_shape = df.shape
        
        # separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        if strategy == 'drop':
            # Strategy 1: Drop Rows
            df_clean = df.dropna()
            note = f"Removed {original_shape[0] - df_clean.shape[0]} rows containing missing values."
            
        elif strategy == 'mean':
            # Strategy 2: Simple Mean Imputation
            # For numeric, use mean. For text, use 'Most Frequent'
            num_imputer = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            if len(numeric_cols) > 0:
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            if len(non_numeric_cols) > 0:
                df[non_numeric_cols] = cat_imputer.fit_transform(df[non_numeric_cols])
            
            df_clean = df
            note = "Filled numeric missing values with Mean, and text with Mode."

        elif strategy == 'knn':
            # Strategy 3: ML - KNN Imputation
            # KNN only works on numbers. We must isolate numeric columns.
            if len(numeric_cols) > 0:
                knn = KNNImputer(n_neighbors=3) # Look at 3 nearest neighbors
                df[numeric_cols] = knn.fit_transform(df[numeric_cols])
                
                # Round values (e.g. Age 24.3 -> 24) for cleanliness
                df[numeric_cols] = df[numeric_cols].round(1)
            
            # For non-numeric, we still fall back to 'most_frequent' as KNN doesn't handle text easily
            if len(non_numeric_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[non_numeric_cols] = cat_imputer.fit_transform(df[non_numeric_cols])
            
            df_clean = df
            note = "Used K-Nearest Neighbors (k=3) algorithm to estimate missing numeric values based on similarity."

        # Save the Cleaned File
        clean_filename = "clean_" + filename
        clean_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
        df_clean.to_csv(clean_path, index=False)

        # Generate "After" stats for the report
        clean_rows = df_clean.shape[0]
        preview_table = df_clean.head(5).to_html(classes='table table-success table-striped', index=False)

        return render_template('export.html', 
                               orig_filename=filename,
                               clean_filename=clean_filename,
                               note=note,
                               rows=clean_rows,
                               table=preview_table)

    except Exception as e:
        return f"Error during cleaning: {e}"

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)