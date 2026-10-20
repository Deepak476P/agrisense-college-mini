# AgriSense Mini

A college-level partial implementation of the supplied **AgriSense AI** project.
It covers **all four major features** with a small Python application and five detailed,
executed Jupyter notebooks explaining what each step does, why it is needed and how to read its output.

See [STEP_BY_STEP.md](STEP_BY_STEP.md) for the six development milestones and how to follow the Git history.

## Features

| Module | Included features |
|---|---|
| Crop recommendation | Seven soil/climate inputs; top three ranked crops; model-score chart; crop season, water, duration and soil reference notes; downloadable result |
| Fertilizer advice | Product classification; nutrient/reference comparison; gap chart; quantity and bag calculator using an editable assumed rate; source formulation/application/organic-alternative notes |
| Yield and economics | Yield in tons/ha and quintals/acre; acreage conversion; total production; editable price/cost assumptions; revenue, cost, profit/loss and margin; ±25% rainfall scenarios |
| Leaf image analysis | Five illustrated samples or PNG/JPEG upload; explicit crop hint; leaf and marked-area masks; overlay; marked-area percentage; severity band; connected-region count; illustrative comparison |
| Learning and evaluation | Data inspection, charts, validation, model comparison, measured reports, confusion matrices, regression errors, limitations, viva questions and integration checks |

This is an educational demonstration using **synthetic data** and **illustrated leaves**.
Leaf analysis uses colour thresholds, not a trained disease classifier. Its percentage describes marked
image area, not diagnostic confidence. Nutrient scales are undocumented; reference gaps do not
prescribe application doses. Financial numbers use editable assumptions, not current prices.

## Open in VS Code

1. Open this project folder.
2. Use **Python: Select Interpreter** and select `.venv\Scripts\python.exe`.
3. Open `notebooks/00_project_guide.ipynb` and select the same environment as its kernel.
4. Use **Run All**, then continue through notebooks 01–04.
5. For the app, use the **Run AgriSense app** debug configuration or the terminal command below.

Python and Jupyter extensions are recommended in `.vscode/extensions.json`.

### On this computer

The project environment was prepared using the existing installed Python packages:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. This needs no paid API, account login or GPU.
Saved models are included; ordinary app startup does not retrain them.

### Fresh clone or another computer

The tested runtime is **Python 3.14.6**; exact package versions are in `requirements.txt`.
Run these commands from this folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

On Linux/macOS use `python3.14` and `.venv/bin/python` instead.
Retrain after changing library versions rather than relying on cross-version joblib compatibility.
Load only the model artifacts you trust; joblib is a Python serialization format.

## Detailed learning notebooks

| Notebook | What and why it teaches |
|---|---|
| [00 — Project guide](notebooks/00_project_guide.ipynb) | Scope, all features, architecture, file responsibilities, implementation steps, setup, demonstration and viva |
| [01 — Crop recommendation](notebooks/01_crop_recommendation.ipynb) | Data dictionary, exploration, stratification, Decision Tree/Random Forest comparison, scores and confusion matrix |
| [02 — Fertilizer advice](notebooks/02_fertilizer_advice.ipynb) | Categorical encoding, pipelines, synthetic rule labels, nutrient arithmetic and bag calculation |
| [03 — Yield and economics](notebooks/03_yield_and_economics.ipynb) | Regression, leakage prevention, MAE/RMSE/R², per-crop errors, unit conversion and financial assumptions |
| [04 — Leaf image analysis](notebooks/04_leaf_image_analysis.ipynb) | RGB/HSV, thresholds, masks, morphological operations, overlays, area ratios and failure cases |

The notebooks contain the executed outputs. Each major code block has a **What**, **Why**, and
**Read the output** explanation. Shared implementation remains visible in the small `src/` modules.

## Project layout

```text
app.py                    Streamlit interface
train_models.py           Reproduce the three fitted models and metric reports
src/data.py               Schemas, paths, data quality checks and validation
src/ml.py                 Pipelines, model comparison, training and prediction
src/advisory.py           Nutrient comparisons, quantities, economics and scenarios
src/vision.py             Deterministic colour segmentation
data/                     Three CSVs, source metadata and provenance
samples/                  Five illustrated leaves
models/                   Locally trained model bundles
notebooks/                Five teaching notebooks with outputs
reports/                  Measured metrics and execution/verification evidence
tests/                    Core and Streamlit integration checks
tools/                    Notebook generation/execution and reference preparation
.vscode/                  Interpreter and debug configurations
```

## Training and evaluation

The three datasets contain 2,200 crop, 1,200 fertilizer and 2,500 yield rows.
Each uses an 80/20 train/test split with random seed 42. Classification splits are stratified.
Three-fold validation **inside the training portion** compares a dummy baseline, Decision Tree and Random Forest.
The best tree/forest candidate is selected using macro F1 for classification or MAE for regression.
It is fitted on the training set and then evaluated on the held-out portion.

One-hot encoding is fitted inside the pipeline. Tree models do not need feature standardization.
`Production_tons` is excluded from yield inputs because it is derived from the target.

Read the actual results in `reports/crop_metrics.json`, `reports/fertilizer_metrics.json` and
`reports/yield_metrics.json`. Notebook outputs also show them. High scores here measure recovery of
synthetic relationships; they do not establish field performance.

## Verification and reproduction

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\execute_notebooks.py
.\.venv\Scripts\python.exe tools\verify.py
```

The tests cover dataset schemas, target leakage, predictions, probability totals, invalid inputs,
unknown categories, range warnings, arithmetic, zero-revenue/loss cases, image determinism,
mask containment, unchanged background pixels, blank-image rejection and app submissions.
Notebook execution writes `reports/notebook_execution.json`.

`tools/build_notebooks.py` rebuilds notebook source and clears saved outputs. Run it only when
intentionally regenerating the notebooks, then execute them again. The app and notebooks do not
require access to the original ZIP after this project is prepared.

## What is deliberately simplified

- This project covers all four modules, with a small interface and no database/accounts/deployment.
- The original serialized models and random leaf-confidence values are not reused.
- Crop names carry between forms only when supported; nutrient values do not cross incompatible dataset scales.
- Rainfall sensitivity is a model exercise, not a causal climate forecast.
- Fertilizer quantities use a user-entered assumption; automatic field dosages and chemical treatment prescriptions are not implemented.
- Real leaf disease classification, live weather/prices, field validation and calibrated uncertainty are future extensions.

## Attribution and provenance

Based on the user-supplied **agri-ai-suite.zip / AgriSense AI**.
The three CSVs and five PNGs were copied unchanged. Selected crop/fertilizer descriptions were retained
as labelled reference text; old market-price fields were omitted.
File hashes and data limitations are recorded in `data/provenance.json`.

The source archive contains generation code for the synthetic datasets and sample drawings.
No license file was supplied. Keep attribution and confirm rights before publicly redistributing
the reference data/assets. The repository is intended to remain private unless the owner decides otherwise.
