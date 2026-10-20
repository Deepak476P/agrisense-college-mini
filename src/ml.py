"""Train on synthetic data and report measured, held-out performance."""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.data import ROOT, SEED, SPECS, load_data, validate_input


def make_pipeline(kind, estimator):
    spec = SPECS[kind]
    steps = [("numeric", "passthrough", spec["numeric"])]
    if spec["categorical"]:
        steps.append(("category", OneHotEncoder(handle_unknown="ignore"), spec["categorical"]))
    # Tree models split on thresholds, so feature scaling is unnecessary here.
    return Pipeline([
        ("prepare", ColumnTransformer(steps)),
        ("model", estimator),
    ])


def split_data(kind, frame=None):
    frame = load_data(kind) if frame is None else frame
    spec = SPECS[kind]
    X = frame[spec["numeric"] + spec["categorical"]]
    y = frame[spec["target"]]
    return train_test_split(
        X, y, test_size=0.2, random_state=SEED,
        stratify=y if spec["task"] == "classification" else None,
    )


def train_model(kind, save=True):
    spec = SPECS[kind]
    X_train, X_test, y_train, y_test = split_data(kind)
    classification = spec["task"] == "classification"
    if classification:
        candidates = {
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=1),
        }
        dummy = DummyClassifier(strategy="most_frequent")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scoring = "f1_macro"
    else:
        candidates = {
            "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
            "Random Forest": RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=1),
        }
        dummy = DummyRegressor(strategy="median")
        cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
        scoring = "neg_mean_absolute_error"
    comparison = []
    for name, estimator in {"Dummy baseline": dummy, **candidates}.items():
        scores = cross_val_score(
            make_pipeline(kind, estimator), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        comparison.append({
            "model": name,
            "cv_score": float(scores.mean()) if classification else float(-scores.mean()),
            "cv_std": float(scores.std()),
        })
    eligible = [row for row in comparison if row["model"] != "Dummy baseline"]
    selected = (max if classification else min)(eligible, key=lambda row: row["cv_score"])["model"]
    pipeline = make_pipeline(kind, candidates[selected])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    metrics = {
        "dataset": spec["file"],
        "data_type": "synthetic",
        "seed": SEED,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "selected_model": selected,
        "cv_metric": "macro F1 (higher is better)" if classification else "MAE in tons/ha (lower is better)",
        "comparison": comparison,
    }
    if classification:
        labels = list(pipeline.named_steps["model"].classes_)
        report = classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0)
        metrics.update({
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "classification_report": report,
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        })
    else:
        errors = X_test[["Crop"]].copy()
        errors["absolute_error"] = np.abs(y_test.to_numpy() - predicted)
        metrics.update({
            "r2": float(r2_score(y_test, predicted)),
            "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "mae_by_crop": errors.groupby("Crop")["absolute_error"].mean().to_dict(),
        })
    bundle = {
        "kind": kind,
        "pipeline": pipeline,
        "features": spec["numeric"] + spec["categorical"],
        "ranges": {name: [float(X_train[name].min()), float(X_train[name].max())] for name in spec["numeric"]},
        "categories": {name: sorted(X_train[name].unique().tolist()) for name in spec["categorical"]},
        "metrics": metrics,
    }
    if save:
        (ROOT / "models").mkdir(exist_ok=True)
        (ROOT / "reports").mkdir(exist_ok=True)
        joblib.dump(bundle, ROOT / "models" / f"{kind}_model.joblib")
        (ROOT / "reports" / f"{kind}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return bundle


def load_model(kind):
    path = ROOT / "models" / f"{kind}_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Models are missing. Run: python train_models.py")
    # Load only files trained locally by this project. Pickle/joblib is not an untrusted-file format.
    return joblib.load(path)


def predict(kind, values, bundle=None):
    bundle = load_model(kind) if bundle is None else bundle
    frame, warnings = validate_input(kind, values, bundle)
    pipeline = bundle["pipeline"]
    value = pipeline.predict(frame)[0]
    result = {"prediction": value.item() if isinstance(value, np.generic) else value, "warnings": warnings}
    if SPECS[kind]["task"] == "classification":
        probabilities = pipeline.predict_proba(frame)[0]
        result["ranking"] = pd.DataFrame({
            "Label": pipeline.named_steps["model"].classes_,
            "Model score (%)": probabilities * 100,
        }).sort_values("Model score (%)", ascending=False).reset_index(drop=True)
    return result


def feature_importance(bundle):
    pipeline = bundle["pipeline"]
    return pd.DataFrame({
        "Feature": pipeline.named_steps["prepare"].get_feature_names_out(),
        "Importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
