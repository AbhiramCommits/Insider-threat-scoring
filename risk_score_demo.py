#!/usr/bin/env python
"""Interactive risk-score demo for insider-threat detection.

Usage:
    python risk_score_demo.py --input data/sample_user.json
"""

import argparse
import json
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore")

MODEL_PATH = "outputs/best_model.pkl"
FEATURE_ORDER_PATH = "outputs/feature_order.json"


def load_model() -> Any:
    return joblib.load(MODEL_PATH)


def score_user(model: Any, user_features: dict) -> tuple:
    """Score a single user dict and return (risk_score, risk_level, top_factors)."""
    feature_cols = list(model.feature_names_in_)

    row = {}
    for col in feature_cols:
        row[col] = user_features.get(col, 0.0)

    X = pd.DataFrame([row])[feature_cols]

    proba = model.predict_proba(X)[0, 1]
    risk_score = round(proba * 100, 1)

    if risk_score < 25:
        risk_level = "Low"
    elif risk_score < 50:
        risk_level = "Medium"
    elif risk_score < 75:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # SHAP for this single row
    clf = model.named_steps["clf"]
    try:
        if hasattr(clf, "get_booster") or hasattr(clf, "estimators_"):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.LinearExplainer(clf, X)
            shap_vals = explainer.shap_values(X)

        row_shap = shap_vals[0]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:3]

        top_factors = []
        for i in top_idx:
            direction = "increases risk" if row_shap[i] > 0 else "decreases risk"
            top_factors.append(
                {
                    "feature": feature_cols[i],
                    "shap_value": round(float(row_shap[i]), 4),
                    "feature_value": round(float(X.iloc[0, i]), 4),
                    "direction": direction,
                }
            )
    except Exception as e:
        top_factors = [{"feature": "N/A", "shap_value": 0, "feature_value": 0, "direction": str(e)}]

    return risk_score, risk_level, top_factors


def print_report(user_id: str, risk_score: float, risk_level: str, top_factors: list) -> None:
    """Print a formatted risk report to the console."""
    sep = "=" * 56

    print(f"\n{sep}")
    print(f"  INSIDER THREAT — RISK SCORING REPORT")
    print(f"{sep}")
    print(f"  User ID        : {user_id}")
    print(f"  Risk Score     : {risk_score:.1f} / 100")
    print(f"  Risk Level     : {risk_level}")
    print(f"{sep}")
    print(f"  TOP RISK FACTORS")
    print(f"  {'Feature':<35s} {'SHAP':>8s}  {'Direction'}")
    print(f"  {'-'*35} {'-'*8}  {'-'*15}")

    for factor in top_factors:
        arrow = "▲" if "increases" in factor["direction"] else "▼"
        print(
            f"  {factor['feature']:<35s} "
            f"{factor['shap_value']:>+8.4f}  "
            f"{arrow} {factor['direction']}"
        )

    print(f"{sep}")
    print(f"  INTERPRETATION")
    if risk_level == "Low":
        print(f"  This user exhibits normal behavioural patterns with no")
        print(f"  significant risk indicators. Routine monitoring recommended.")
    elif risk_level == "Medium":
        print(f"  This user shows mild behavioural anomalies. Consider")
        print(f"  reviewing activity logs and increasing monitoring frequency.")
    elif risk_level == "High":
        print(f"  This user displays multiple elevated risk indicators.")
        print(f"  Immediate review of recent activity is warranted.")
    else:
        print(f"  CRITICAL: This user exhibits strong indicators of insider")
        print(f"  threat activity. Escalate to security team immediately.")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Score a user for insider-threat risk using aggregated telemetry features."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file containing aggregated user features.",
    )
    args = parser.parse_args()

    # Load user features
    with open(args.input, "r") as f:
        user_data = json.load(f)

    user_id = user_data.pop("user_id", "UNKNOWN")

    # Load model and score
    model = load_model()
    risk_score, risk_level, top_factors = score_user(model, user_data)

    print_report(user_id, risk_score, risk_level, top_factors)


if __name__ == "__main__":
    main()
