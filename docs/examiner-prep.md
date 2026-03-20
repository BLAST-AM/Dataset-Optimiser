# Dataset Optimiser — Examiner Prep (Tech + Uses + Q&A)

Date: 2026-01-30

## 1) One-minute project explanation (you can speak this)

This project is a Flask web application that helps a user upload a CSV dataset, automatically profile its quality (missing values, duplicates, data types, memory usage), generate visual insights (distributions, correlations, categorical counts, and time-series when possible), and then optionally apply cleaning and optimization steps. The user can also evaluate a simple ML model on a chosen target column (when classification is feasible) and download the cleaned/optimized CSV output. Uploaded files are stored in `uploads/` and generated charts are saved in `static/images/` and displayed in the results dashboard.

---

## 2) Tech stack used in THIS project (and what each was used for)

### Backend / Web
- **Python**: Implements the complete pipeline: read uploaded CSV → compute profiling stats → clean/optimize → optionally evaluate → export/download results.
- **Flask**: Web server + routing. Connects UI actions to Python functions (upload, clean, optimize, evaluate, download).
- **Jinja2 (via Flask templates)**: Renders dynamic pages by injecting dataset stats, previews, chart filenames, and evaluation results into HTML templates.
- **Werkzeug (via Flask)**: Used for secure upload handling (sanitizing uploaded filenames to prevent unsafe paths).

### Frontend / UI
- **HTML**: Defines the app pages (upload form, results dashboard, export/download page).
- **CSS**: Custom dashboard styling (layout, glass panels, responsive visuals).
- **JavaScript**: Small UI effect (mousemove-based background animation) to improve the feel of the upload page.
- **Bootstrap 5 (CDN)**: Provides layout/grid and common UI components (form styling, buttons, responsive structure).
- **Google Fonts (CDN: Inter, Orbitron)**: Used for consistent typography/branding across the UI.

### Data analysis / Data quality
- **Pandas**: Core data tool. Reads CSVs, computes dataset details (shape, missing %, duplicates, dtypes), drives cleaning/feature steps, and writes output CSVs.
- **NumPy**: Supports numerical operations and dtype handling used alongside Pandas and plotting.

### ML / Evaluation
- **scikit-learn**: Used when evaluation is enabled:
  - preprocessing: train/test split, scaling, imputers
  - models: common baseline learners (e.g., logistic regression, SVM, KNN, random forest)
  - metrics: accuracy/precision/recall/F1 and confusion matrix (only when conditions are appropriate)
  - PCA: optional dimensionality reduction step
- **imbalanced-learn**: Used for **SMOTE** oversampling during optimization/evaluation when classes are imbalanced (to avoid a model that just predicts the majority class).

### Visualizations
- **Matplotlib**: Generates and saves plots as image files under `static/images/` for the dashboard.
- **Seaborn**: Produces higher-level statistical charts (especially correlation heatmaps and nicer defaults on plots).

### Export / Documentation tooling
- **openpyxl**: Included in dependencies to support Excel workflows via Pandas (e.g., if exporting/handling `.xlsx` is needed). In the current codebase it’s not directly imported, but it enables Pandas Excel operations.
- **ReportLab** (used in `tools/` scripts): Generates examiner-friendly PDFs (e.g., exporting diagrams/markdown documentation to PDF).

---

## 3) Viva-style questions WITH model answers

### A) Problem + scope
**Q1. What problem does your project solve?**  
A. It reduces the manual effort of dataset preparation by automatically profiling a CSV for common quality issues, generating visual insights, and providing one-click cleaning/optimization options, plus a basic evaluation step to sanity-check a target column.

**Q2. Who is the target user?**  
A. Students/analysts who need quick data quality checks and basic preprocessing without writing code—especially for early-stage exploration.

**Q3. What is the main workflow?**  
A. Upload CSV → server profiles data and generates plots → user optionally cleans/optimizes → user can run evaluation (when feasible) → user downloads output CSV.

**Q4. What outputs does the system generate?**  
A. (1) A results dashboard (HTML) with stats + plots, (2) cleaned/optimized CSV files saved in `uploads/`, and (3) plot images saved in `static/images/`.

### B) Data profiling (dashboard)
**Q5. What “data quality” metrics do you compute and why?**  
A. Rows/columns (size), missing values (amount and percentage), duplicate rows, dtype distribution, and memory usage. These quantify cleanliness, complexity, and cost (e.g., missing values drive cleaning choice; memory size affects performance).

**Q6. Why do you generate histograms, scatter plots, box plots, and correlation heatmaps?**  
A. Histograms show distributions/skew; scatter plots show relationships; box plots show outliers/spread; correlation heatmaps highlight multicollinearity or strongly-related features.

**Q7. How do you handle datasets with a lot of columns?**  
A. The dashboard prioritizes a subset (e.g., top numeric columns) to keep visuals readable and performance reasonable.

**Q8. How do you handle time-series plots?**  
A. The app tries to detect a datetime-like column; if found and numeric columns exist, it aggregates by date and plots a trend (best-effort).

### C) Cleaning
**Q9. What cleaning strategies are implemented?**  
A. The project supports missing-value handling (simple imputation and KNN imputation depending on settings), and typical dataset fixes like removing unusable rows/columns (based on the chosen option).

**Q10. How do you decide between simple imputation vs KNN imputation?**  
A. Simple imputation is fast and stable (good default). KNN imputation can preserve relationships between features but is slower and needs numeric features; it’s best for moderate-sized datasets where that extra quality matters.

**Q11. How do you prevent “data leakage” when evaluating models?**  
A. The correct approach is to split into train/test first and fit transformations (imputer/scaler/SMOTE) only on the training set, then apply to the test set. (If asked, mention this is the best-practice pipeline and where you’d enforce it in a future iteration.)

### D) Optimization / preprocessing
**Q12. What does “optimization” mean in your project?**  
A. It refers to optional preprocessing steps that can improve model-readiness and data usability—like handling imbalance (SMOTE), scaling, feature selection, encoding, dropping correlated features, and PCA.

**Q13. Why do you include scaling?**  
A. Distance-based models (SVM, KNN) are sensitive to feature scale; scaling prevents one large-range feature from dominating.

**Q14. Why do you include one-hot encoding?**  
A. Many ML models require numeric inputs; one-hot encoding turns categorical features into model-compatible numeric columns.

**Q15. Why drop highly correlated features?**  
A. Correlated features can cause redundancy, unstable coefficients in some models, and unnecessary complexity. Dropping them can simplify the model without losing much information.

**Q16. Why include PCA, and what are the tradeoffs?**  
A. PCA can reduce dimensionality and noise, improving speed and sometimes generalization. The tradeoff is interpretability—principal components are less explainable than original features.

### E) Evaluation / ML
**Q17. When can you produce a confusion matrix?**  
A. Only when the target looks like a classification label with a reasonable number of classes and enough samples. The app avoids generating a confusion matrix when classes are too many or the target is unsuitable.

**Q18. Why did you choose these models (Logistic Regression, SVM, KNN, Random Forest)?**  
A. They’re standard baselines that cover linear, margin-based, distance-based, and ensemble approaches. Together they give a broad sanity-check without overcomplicating the UI.

**Q19. Why use multiple metrics (precision/recall/F1) instead of only accuracy?**  
A. Accuracy can be misleading with imbalanced classes. Precision/recall/F1 capture false positives/false negatives more explicitly.

**Q20. Why use SMOTE and what can go wrong?**  
A. SMOTE balances classes by synthesizing minority samples, improving learning when data is skewed. Risks: it can amplify noise/outliers and must be applied only on the training set to avoid leakage.

### F) Engineering / security / reliability
**Q21. How do you handle file uploads safely?**  
A. The app restricts uploads to CSV and uses secure filename handling to avoid path traversal issues, storing files in a controlled `uploads/` directory.

**Q22. Where do plots go and why?**  
A. Plots are saved under `static/images/` so they can be served as static files and embedded into the results page via a simple filename reference.

**Q23. What are the main performance bottlenecks?**  
A. Reading very large CSVs into memory, running more expensive steps like KNN imputation/SMOTE, and generating multiple plots.

**Q24. What happens with bad input (empty CSV, non-numeric target, too many classes)?**  
A. The expected behavior is to fall back gracefully: skip steps that don’t apply, show a clear message, and still allow download of the cleaned/optimized dataset when possible.

### G) UI / usability
**Q25. How does the UI make the project usable for non-coders?**  
A. The user only needs to upload a CSV and select options via checkboxes/buttons; results are shown as readable stats and plots, and outputs are downloadable.

### H) Documentation tooling
**Q26. What is the purpose of the PDF tools in `tools/`?**  
A. They generate examiner-friendly PDF documentation (e.g., turning the diagrams/markdown documentation into a portable PDF).

---

## 4) Quick improvements you can mention if asked “what next?”

- Add a proper scikit-learn `Pipeline` so imputation/scaling/SMOTE happen only on training data (prevents leakage).
- Add background jobs for long operations (large datasets) with progress feedback.
- Add robust CSV parsing options (delimiter detection, encoding handling).
- Add automated tests for the cleaning/optimization functions.
