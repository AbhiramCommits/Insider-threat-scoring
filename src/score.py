"""Risk scoring for insider-threat detection.

Converts model probabilities to 0-100 risk scores and extracts
top contributing factors using SHAP.
"""

import json
import warnings
from typing import Any, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_risk_score(
    model: Any, X: pd.DataFrame, user_ids: pd.Series | list | None = None
) -> pd.DataFrame:
    """Convert model-predicted threat probability to a 0–100 risk score.

    Parameters
    ----------
    model : fitted sklearn/imblearn Pipeline
    X : pd.DataFrame
        Feature matrix (one row per user).
    user_ids : sequence, optional
        User identifiers; defaults to X.index if omitted.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, risk_score, risk_level.
    """
    proba = model.predict_proba(X)[:, 1]

    if user_ids is None:
        user_ids = X.index.tolist()

    scores = pd.DataFrame(
        {
            "user_id": user_ids,
            "risk_score": (proba * 100).round(1),
        }
    )

    def _classify(score: float) -> str:
        if score < 25:
            return "Low"
        elif score < 50:
            return "Medium"
        elif score < 75:
            return "High"
        else:
            return "Critical"

    scores["risk_level"] = scores["risk_score"].apply(_classify)
    return scores


def get_top_risk_factors(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    user_idx: int,
    n: int = 3,
) -> List[dict]:
    """Return the top *n* features driving the threat score for a given user.

    Parameters
    ----------
    shap_values : np.ndarray
        2-D SHAP array, shape (n_samples, n_features).
    X : pd.DataFrame
        Feature matrix used for SHAP computation.
    user_idx : int
        Row index of the target user within *X*.
    n : int
        Number of top factors to return.

    Returns
    -------
    list[dict]
        Each dict has keys: feature, shap_value, feature_value, direction.
    """
    row = shap_values[user_idx]
    top_indices = np.argsort(np.abs(row))[::-1][:n]

    factors = []
    for i in top_indices:
        direction = "increases risk" if row[i] > 0 else "decreases risk"
        factors.append(
            {
                "feature": str(X.columns[i]),
                "shap_value": round(float(row[i]), 6),
                "feature_value": round(float(X.iloc[user_idx, i]), 4),
                "direction": direction,
            }
        )
    return factors


def build_risk_scores_csv(
    model: Any,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    user_ids: pd.Series | list,
    output_path: str = "outputs/risk_scores.csv",
) -> pd.DataFrame:
    """Compute risk scores and top factors for all users; save as CSV.

    Returns the full DataFrame.
    """
    scores = compute_risk_score(model, X, user_ids)

    top_factors = []
    for i in range(len(X)):
        factors = get_top_risk_factors(shap_values, X, i, n=3)
        top_factors.append(
            {
                "top_factor_1": factors[0]["feature"] if len(factors) > 0 else "",
                "top_factor_2": factors[1]["feature"] if len(factors) > 1 else "",
                "top_factor_3": factors[2]["feature"] if len(factors) > 2 else "",
            }
        )

    factors_df = pd.DataFrame(top_factors)
    result = pd.concat([scores, factors_df], axis=1)
    result.to_csv(output_path, index=False)
    print(f"Risk scores saved to {output_path}")
    return result


# ---------------------------------------------------------------------------
# __main__ — generate risk_scores.csv for the full dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import joblib
    from src.feature_engineering import aggregate_user_features
    from src.explain import generate_shap_values

    df = pd.read_csv("data/synthetic_cert.csv", parse_dates=["date"])
    feats = aggregate_user_features(df)
    feat_cols = [c for c in feats.columns if c not in ("user_id", "label")]

    X = feats[feat_cols]
    user_ids = feats["user_id"].tolist()

    model = joblib.load("outputs/best_model.pkl")
    shap_vals, _ = generate_shap_values(model, X)

    result = build_risk_scores_csv(model, X, shap_vals, user_ids)

    print(f"\nRisk score distribution:")
    print(result["risk_level"].value_counts().to_string())
    print(f"\nTop 5 highest-risk users:")
    print(result.sort_values("risk_score", ascending=False).head(5).to_string())
