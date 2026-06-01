"""Model training and evaluation for insider-threat scoring."""

import json
import warnings
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """Train Logistic Regression, Random Forest, and XGBoost with
    stratified 5-fold CV and SMOTE inside each fold.

    Returns
    -------
    dict
        {model_name: {'model': fitted_pipeline, 'cv_scores': {...}}}
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["f1", "precision", "recall", "roc_auc"]

    model_configs = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }

    results = {}

    for name, clf in model_configs.items():
        pipe = Pipeline([("smote", SMOTE(random_state=42)), ("clf", clf)])

        cv_results = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )

        pipe.fit(X_train, y_train)

        results[name] = {
            "model": pipe,
            "cv_scores": {
                "f1_mean": float(cv_results["test_f1"].mean()),
                "f1_std": float(cv_results["test_f1"].std()),
                "precision_mean": float(cv_results["test_precision"].mean()),
                "precision_std": float(cv_results["test_precision"].std()),
                "recall_mean": float(cv_results["test_recall"].mean()),
                "recall_std": float(cv_results["test_recall"].std()),
                "roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
                "roc_auc_std": float(cv_results["test_roc_auc"].std()),
            },
        }

        print(f"  {name:<22s}  CV F1={results[name]['cv_scores']['f1_mean']:.4f} "
              f"(±{results[name]['cv_scores']['f1_std']:.4f})  "
              f"ROC-AUC={results[name]['cv_scores']['roc_auc_mean']:.4f}")

    return results


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str
) -> Dict[str, float]:
    """Evaluate a fitted pipeline on hold-out data.

    Returns dict of F1, precision, recall, ROC-AUC; prints a formatted table.
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0,
    }

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<15} {'Value':>10}")
    print(f"  {'-'*25}")
    for k, v in metrics.items():
        print(f"  {k:<15} {v:>10.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    return metrics


def save_model(model: Pipeline, path: str) -> None:
    """Save a fitted pipeline to disk using joblib."""
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")


def get_roc_data(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) for ROC plotting."""
    from sklearn.metrics import roc_curve

    y_proba = model.predict_proba(X_test)[:, 1]
    return roc_curve(y_test, y_proba)[:2]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.feature_engineering import aggregate_user_features
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/synthetic_cert.csv", parse_dates=["date"])
    feats = aggregate_user_features(df)
    feat_cols = [c for c in feats.columns if c not in ("user_id", "label")]

    X = feats[feat_cols]
    y = feats["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train: {len(X_train)} users, Test: {len(X_test)} users")
    print(f"Train label rates: {y_train.mean():.4f}")
    print(f"Test  label rates: {y_test.mean():.4f}\n")

    print("=== Cross-Validation Results ===")
    trained = train_models(X_train, y_train)

    print("\n=== Hold-Out Evaluation ===")
    all_metrics = {}
    for name, entry in trained.items():
        m = evaluate_model(entry["model"], X_test, y_test, name)
        all_metrics[name] = {
            **m,
            "cv_f1_mean": entry["cv_scores"]["f1_mean"],
            "cv_f1_std": entry["cv_scores"]["f1_std"],
            "cv_roc_auc_mean": entry["cv_scores"]["roc_auc_mean"],
            "cv_roc_auc_std": entry["cv_scores"]["roc_auc_std"],
        }

    best = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
    save_model(trained[best]["model"], "outputs/best_model.pkl")
    with open("outputs/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved to outputs/metrics.json")
