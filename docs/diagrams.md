# Dataset Optimiser — DFD + UML Diagrams

This document provides Data Flow Diagrams (DFD) and UML diagrams for the **Dataset Optimiser** Flask application (see [app.py](../app.py)).

## Notation

- **External Entity**: outside the system boundary (User/Browser)
- **Process**: transformation of data (Upload/Analyze/Clean/Optimize/Evaluate/Download)
- **Data Store**: persistent storage (CSV files in `uploads/`, images in `static/images/`)

---

## DFD — Level 0 (Context Diagram)

```mermaid
flowchart LR
  U[External Entity: User (Browser)]
  S((Process: Dataset Optimiser Web App))

  DS1[(Data Store: uploads/ (CSV files))]
  DS2[(Data Store: static/images/ (plots))]

  U -- "Upload CSV" --> S
  S -- "Save uploaded CSV" --> DS1

  S -- "Analysis dashboard (HTML)" --> U
  S -- "Generate plots" --> DS2

  U -- "Request clean/optimize/evaluate" --> S
  S -- "Write cleaned/optimized CSV" --> DS1

  U -- "Download request" --> S
  S -- "Send CSV file" --> U
```

---

## DFD — Level 1 (Major Processes)

```mermaid
flowchart TB
  U[External Entity: User (Browser)]

  P1((1. Upload Dataset
POST /upload))
  P2((2. Analyze Dataset
_build_report_context))
  P3((3. Clean Dataset
POST /clean))
  P4((4. Optimize Dataset
POST /optimize))
  P5((5. Evaluate Model
POST /evaluate))
  P6((6. Download Output
GET /download/<filename>))

  DS1[(uploads/: original/clean_/opt_ CSVs)]
  DS2[(static/images/: plots & confusion matrices)]

  U -- CSV file --> P1
  P1 -- saved CSV path --> DS1
  P1 -- filename --> P2

  DS1 -- read CSV --> P2
  P2 -- plots --> DS2
  P2 -- dashboard HTML (results.html) --> U

  U -- strategy + filename --> P3
  DS1 -- read CSV --> P3
  P3 -- cleaned CSV --> DS1
  P3 -- export HTML (export.html) --> U

  U -- options + filename + optional target --> P4
  DS1 -- read CSV --> P4
  P4 -- optimized CSV --> DS1
  P4 -- export HTML (export.html) --> U

  U -- target + model + test_size + filename --> P5
  DS1 -- read CSV --> P5
  P5 -- confusion matrix plot --> DS2
  P5 -- dashboard HTML (results.html with metrics) --> U

  U -- output filename --> P6
  DS1 -- file bytes --> P6
  P6 -- downloaded file --> U
```

---

## DFD — Level 2 (Optimization Pipeline Detail)

```mermaid
flowchart LR
  U[User]
  P4((Optimize Dataset
POST /optimize))
  DS1[(uploads/)]

  subgraph OPT[Optimization Steps (conditional by checkboxes)]
    O1[SMOTE / Oversampling\n_handle_class_imbalance]
    O2[RF Feature Selection\n_feature_select_rf]
    O3[Memory Optimize DTypes\n_memory_optimize]
    O4[Winsorize Outliers\n_winsorize_outliers]
    O5[Log Transform Skewed\n_log_transform_skewed]
    O6[Binning\n_bin_first_numeric]
    O7[One-Hot Encoding\n_one_hot_encode]
    O8[Scaling\n_scale_numeric]
    O9[Drop Correlated\n_drop_correlated]
    O10[PCA\n_apply_pca]
  end

  U -- "options + target (optional)" --> P4
  DS1 -- "read original CSV" --> P4

  P4 --> O1 --> O2 --> O3 --> O4 --> O5 --> O6 --> O7 --> O8 --> O9 --> O10

  O10 -- "write opt_<filename>.csv" --> DS1
  P4 -- "export.html + preview + notes" --> U
```

---

# UML Diagrams (PlantUML)

These are **conceptual UML diagrams** that map directly to what the code in [app.py](../app.py) does.

## UML — Use Case Diagram

```plantuml
@startuml
left to right direction
skinparam packageStyle rectangle

actor User as U

rectangle "Dataset Optimiser (Flask Web App)" {
  usecase "Upload CSV" as UC1
  usecase "View Analysis Dashboard" as UC2
  usecase "Generate Plots" as UC3
  usecase "Clean Dataset" as UC4
  usecase "Optimize Dataset" as UC5
  usecase "Evaluate Model" as UC6
  usecase "Download Output CSV" as UC7

  UC2 ..> UC3 : <<include>>
  UC4 ..> UC7 : <<include>>
  UC5 ..> UC7 : <<include>>
}

U --> UC1
U --> UC2
U --> UC4
U --> UC5
U --> UC6
U --> UC7

note right of UC6
Only enabled when dataset supports
classification-like target with limited
classes and enough samples.
end note
@enduml
```

---

## UML — Class Diagram (Conceptual)

```plantuml
@startuml
skinparam classAttributeIconSize 0

class FlaskApp {
  +home()
  +upload_file()
  +clean_data()
  +optimize_data()
  +evaluate_model()
  +download_file(filename)
}

class FileStorage {
  +save_upload(file): filename
  +read_csv(filename): DataFrame
  +write_csv(df, filename): void
  +send_file(filename): bytes
}

class ReportService {
  +build_report_context(df, filename, eval_result, cm_image): dict
  +compute_dataset_details(df): dict
}

class VisualizationService {
  +generate_missing_plot(df, filename): imageName
  +generate_additional_visualizations(df, filename): dict
  +plot_confusion_matrix(cm, labels, title): imageName
}

class CleaningService {
  +clean_drop_rows(df): df
  +clean_mean_mode(df): df
  +clean_knn(df): df
}

class OptimizationService {
  +memory_optimize(df): (df, note)
  +winsorize_outliers(df): (df, note)
  +log_transform_skewed(df): (df, note)
  +bin_first_numeric(df): (df, note)
  +one_hot_encode(df): (df, note)
  +scale_numeric(df, exclude): (df, note)
  +drop_correlated(df, exclude): (df, note)
  +apply_pca(df, target): (df, note)
  +feature_select_rf(df, target): (df, note)
  +handle_class_imbalance(df, target): (df, note)
}

class EvaluationService {
  +eval_capability_for_confusion_matrix(df): (bool, targets, reason)
  +evaluate_classifier(df, target, model, test_size): (metrics, cm)
}

FlaskApp ..> FileStorage
FlaskApp ..> ReportService
FlaskApp ..> VisualizationService
FlaskApp ..> CleaningService
FlaskApp ..> OptimizationService
FlaskApp ..> EvaluationService

ReportService ..> VisualizationService
EvaluationService ..> VisualizationService
@enduml
```

---

## UML — Sequence Diagram (Upload + Analyze)

```plantuml
@startuml
actor User
participant "Browser" as B
participant "Flask Routes\n(app.py)" as R
participant "File System\nuploads/" as FS
participant "Pandas" as PD
participant "Report/Plots" as RP
participant "File System\nstatic/images/" as IMG

User -> B: Select CSV + submit
B -> R: POST /upload (file)
R -> FS: Save file (uploads/<filename>)
R -> PD: read_csv(path)
PD --> R: DataFrame df
R -> RP: build_report_context(df)
RP -> IMG: write plot_*.png + extra *.png
RP --> R: context (rows/cols/missing/images/...)
R --> B: results.html (dashboard)
B --> User: Render dashboard
@enduml
```

---

## UML — Sequence Diagram (Clean Dataset)

```plantuml
@startuml
actor User
participant "Browser" as B
participant "Flask Routes" as R
participant "File System\nuploads/" as FS
participant "Pandas/Sklearn Imputer" as ML

User -> B: Choose cleaning strategy
B -> R: POST /clean (filename, strategy)
R -> FS: read uploads/<filename>
R -> ML: Apply cleaning strategy
ML --> R: df_clean
R -> FS: write uploads/clean_<filename>
R --> B: export.html + preview + download link
B --> User: Show download
@enduml
```

---

## UML — Sequence Diagram (Optimize Dataset)

```plantuml
@startuml
actor User
participant "Browser" as B
participant "Flask Routes" as R
participant "File System\nuploads/" as FS
participant "OptimizationService" as OPT

User -> B: Select optimization checkboxes
B -> R: POST /optimize (filename, options, target?)
R -> FS: read uploads/<filename>
FS --> R: df

R -> OPT: apply selected steps (conditional)
OPT --> R: df_opt + notes

R -> FS: write uploads/opt_<filename>
R --> B: export.html + preview + download link
B --> User: Show download
@enduml
```

---

## UML — Sequence Diagram (Evaluate Model)

```plantuml
@startuml
actor User
participant "Browser" as B
participant "Flask Routes" as R
participant "File System\nuploads/" as FS
participant "EvaluationService" as EV
participant "Sklearn" as SK
participant "File System\nstatic/images/" as IMG

User -> B: Choose target + model + test size
B -> R: POST /evaluate
R -> FS: read uploads/<filename>
FS --> R: df
R -> EV: check eval capability + validate target
EV --> R: ok/targets
R -> SK: preprocess (impute + onehot)
R -> SK: train_test_split(stratified)
R -> SK: fit model + predict
R -> SK: compute metrics + confusion matrix
R -> IMG: write cm_<stem>_<nonce>.png
R --> B: results.html (metrics + cm image)
B --> User: Show evaluation results
@enduml
```

---

## UML — Activity Diagram (Cleaning)

```plantuml
@startuml
start
:Receive filename + strategy;
:Read CSV from uploads/;
if (strategy == drop?) then (yes)
  :Drop rows with NA;
elseif (strategy == mean?) then (yes)
  :Impute numeric mean;
  :Impute categorical mode;
else (knn)
  :KNN impute numeric;
  :Impute categorical mode;
endif
:Write clean_<filename>.csv to uploads/;
:Render export.html with preview;
stop
@enduml
```

---

## UML — Activity Diagram (Optimization)

```plantuml
@startuml
start
:Receive filename + options + optional target;
:Read CSV from uploads/;

if (SMOTE selected and target set?) then (yes)
  :Handle imbalance (SMOTE or oversampling);
endif
if (Feature select selected and target set?) then (yes)
  :Random Forest feature selection;
endif

if (Optimize dtypes?) then (yes)
  :Downcast numeric / category-ize objects;
endif
if (Cap outliers?) then (yes)
  :Winsorize numeric columns;
endif
if (Log transform?) then (yes)
  :Log1p skewed numeric columns;
endif
if (Binning?) then (yes)
  :Add quartile bin feature;
endif
if (One-hot?) then (yes)
  :Get dummies for categorical;
endif

:Exclude target from scaling/corr/PCA;
if (Scale numeric?) then (yes)
  :StandardScaler numeric;
endif
if (Drop correlated?) then (yes)
  :Remove highly correlated features;
endif
if (PCA?) then (yes)
  :PCA to 95% variance;
endif

:Write opt_<filename>.csv to uploads/;
:Render export.html with preview + notes;
stop
@enduml
```

---

## UML — Component Diagram

```plantuml
@startuml
skinparam componentStyle rectangle

component "Web Browser" as Browser
component "Flask App\n(app.py)" as Flask
component "Data Processing\n(pandas/numpy)" as DP
component "ML / Metrics\n(scikit-learn, imblearn)" as ML
component "Plotting\n(matplotlib/seaborn)" as Plot

database "uploads/\nCSV Store" as Uploads
database "static/images/\nImage Store" as Images

Browser --> Flask : HTTP requests
Flask --> Uploads : read/write CSV
Flask --> DP : DataFrame ops
Flask --> ML : train/evaluate + imputers
Flask --> Plot : generate plots
Plot --> Images : write PNG
Flask --> Browser : HTML responses + downloads
@enduml
```

---

## UML — Deployment Diagram

```plantuml
@startuml
node "Client Machine" {
  artifact "Browser" as Br
}

node "Server (Python runtime)" {
  artifact "Flask process" as F
  folder "uploads/" as U
  folder "static/images/" as I
}

Br --> F : HTTP (GET/POST)
F --> U : file I/O (CSV)
F --> I : file I/O (PNG)
@enduml
```
