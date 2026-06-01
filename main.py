#!/usr/bin/env python
"""Insider Threat Scoring — CLI entry point.

Usage:
    python main.py --mode train
    python main.py --mode score
    python main.py --mode explain
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import aggregate_user_features
from src.train import train_models, evaluate_model, save_model
from src.explain import (
    generate_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    explain_single_prediction,
    find_high_risk_user,
    generate_shap_report,
)
from src.score import build_risk_scores_csv

warnings.filterwarnings("ignore")


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_and_featurize(data_path: str) -> tuple:
    """Return (X, y, user_ids, feature_names)."""
    df = pd.read_csv(data_path, parse_dates=["date"])
    feats = aggregate_user_features(df)
    feat_cols = [c for c in feats.columns if c not in ("user_id", "label")]
    X = feats[feat_cols]
    y = feats["label"] if "label" in feats.columns else None
    user_ids = feats["user_id"].tolist()
    return X, y, user_ids, feat_cols


def mode_train(data_path: str, output_dir: str) -> None:
    print("=== MODE: train ===\n")
    X, y, _, _ = load_and_featurize(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    trained = train_models(X_train, y_train)

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

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
    best_model = trained[best_name]["model"]

    model_path = os.path.join(output_dir, "best_model.pkl")
    save_model(best_model, model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def mode_score(data_path: str, output_dir: str) -> None:
    print("=== MODE: score ===\n")
    model_path = os.path.join(output_dir, "best_model.pkl")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Run --mode train first.", file=sys.stderr)
        sys.exit(1)

    X, _, user_ids, _ = load_and_featurize(data_path)
    model = joblib.load(model_path)
    shap_vals, _ = generate_shap_values(model, X)

    csv_path = os.path.join(output_dir, "risk_scores.csv")
    result = build_risk_scores_csv(model, X, shap_vals, user_ids, output_path=csv_path)

    print("\nRisk score distribution:")
    print(result["risk_level"].value_counts().to_string())
    print(f"\nTop 5 highest-risk users:")
    print(result.sort_values("risk_score", ascending=False).head(5).to_string())


def mode_explain(data_path: str, output_dir: str) -> None:
    print("=== MODE: explain ===\n")
    model_path = os.path.join(output_dir, "best_model.pkl")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Run --mode train first.", file=sys.stderr)
        sys.exit(1)

    X, _, _, _ = load_and_featurize(data_path)
    model = joblib.load(model_path)

    print("Generating SHAP values (this may take a moment) ...")
    shap_vals, explainer = generate_shap_values(model, X)

    print("Saving plots ...")
    plot_shap_summary(shap_vals, X, os.path.join(output_dir, "shap_beeswarm.png"))
    plot_shap_bar(shap_vals, X, os.path.join(output_dir, "shap_bar.png"))

    idx = find_high_risk_user(model, X)
    print(f"Waterfall for highest-risk user (index {idx}) ...")
    explain_single_prediction(
        model, X, idx, os.path.join(output_dir, "shap_waterfall.png")
    )

    report_path = os.path.join(output_dir, "shap_report.md")
    generate_shap_report(shap_vals, X, top_n=10, output_path=report_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Insider Threat Scoring — end-to-end ML pipeline."
    )
    parser.add_argument(
        "--data",
        default="data/synthetic_cert.csv",
        help="Path to input CSV (default: data/synthetic_cert.csv).",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "score", "explain"],
        required=True,
        help='Pipeline mode: train, score, or explain.',
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory (default: outputs).",
    )
    args = parser.parse_args()

    out = ensure_dir(args.output)

    print(f"Data   : {args.data}")
    print(f"Mode   : {args.mode}")
    print(f"Output : {out}\n")

    if args.mode == "train":
        mode_train(args.data, out)
    elif args.mode == "score":
        mode_score(args.data, out)
    elif args.mode == "explain":
        mode_explain(args.data, out)


if __name__ == "__main__":
    main()
