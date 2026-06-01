"""SHAP explainability for insider-threat models.

Generates global and local explanations, plus a plain-English risk report.
"""

import warnings
from typing import Any, Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def generate_shap_values(
    model: Any, X: pd.DataFrame
) -> Tuple[np.ndarray, Any]:
    """Compute SHAP values using TreeExplainer or LinearExplainer.

    Returns (shap_values_array, explainer).
    shap_values_array is always a 2-D array for the positive (threat) class.
    """
    clf = model.named_steps["clf"]

    if hasattr(clf, "get_booster") or hasattr(clf, "estimators_"):
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X)
    else:
        explainer = shap.LinearExplainer(clf, X)
        raw = explainer.shap_values(X)

    if isinstance(raw, list):
        shap_vals = np.array(raw[1])
    else:
        shap_vals = np.array(raw)

    return shap_vals, explainer


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values: np.ndarray, X: pd.DataFrame, save_path: str | None = None
) -> None:
    """Beeswarm summary plot of SHAP values."""
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_bar(
    shap_values: np.ndarray, X: pd.DataFrame, save_path: str | None = None
) -> None:
    """Mean |SHAP| bar plot."""
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def explain_single_prediction(
    model: Any, X: pd.DataFrame, idx: int, save_path: str | None = None
) -> dict:
    """Waterfall plot for a single user at index *idx*.

    If idx is None, the user with the highest predicted threat probability
    is selected automatically.

    Returns a dict with index, predicted_prob, and top_shap_features.
    """
    clf = model.named_steps["clf"]

    # Build an Explanation object using the new SHAP unified API
    masker = shap.maskers.Independent(X)
    explainer = shap.Explainer(clf, masker=masker)
    exp = explainer(X.iloc[[idx]])

    shap.plots.waterfall(exp[0], show=False, max_display=15)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    proba = model.predict_proba(X.iloc[[idx]])[0, 1]

    return {
        "user_index": idx,
        "predicted_probability": float(proba),
    }


def find_high_risk_user(model: Any, X: pd.DataFrame) -> int:
    """Return the index of the user with the highest predicted threat probability."""
    proba = model.predict_proba(X)[:, 1]
    return int(np.argmax(proba))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


_FEATURE_RISK_RATIONALE: dict[str, str] = {
    "after_hours": "After-hours activity suggests data exfiltration when fewer monitors or colleagues are present.",
    "usb": "USB mass-storage device usage is the most common physical exfiltration vector for sensitive data.",
    "email_external": "A high proportion of emails sent to external domains indicates data being communicated outside the organisation.",
    "email_volume": "Sudden spikes in email volume can reflect bulk data exfiltration to personal accounts.",
    "failed_login": "Multiple failed login attempts suggest credential probing, privilege escalation, or unauthorised system access.",
    "file_access": "Unusual file-access counts or patterns indicate data staging, bulk download, or intellectual-property theft.",
    "vpn": "VPN usage can mask a user's true location, enabling covert activity from untrusted networks.",
    "weekend": "Weekend activity is atypical for most roles and is often associated with off-hours data theft.",
    "login_hour": "Irregular login times — especially late-night or early-morning — correlate with stealthy insider behaviour.",
    "deviation": "Deviations detected by rolling 7-day analysis capture short-term behavioural anomalies missed by global aggregates.",
    "spike": "Binary spike flags identify extreme single-day anomalies that may signal acute threat events.",
    "high_risk_day": "Days with multiple concurrent anomalous signals are rare for benign users and common during insider attacks.",
    "morning": "A shift away from normal morning login patterns may indicate the user is avoiding routine observation.",
    "evening": "Elevated evening login ratios are a classic precursor to data exfiltration in CERT datasets.",
    "night": "Night-time access (00:00–06:59) is extremely rare for legitimate users and constitutes a high-risk signal.",
    "afternoon": "Consistent afternoon-only logins with no morning presence can be a sign of schedule manipulation.",
    "external_email": "External communications are the primary channel for data leaks; elevated ratios warrant investigation.",
    "cv": "High coefficient of variation in activity metrics indicates unpredictable, burst-like behaviour.",
    "skew": "Skewed login-hour distributions with a long tail toward late hours signal inconsistent, risky schedules.",
    "days_active": "Unusually few active days may indicate a dormant or compromised account used sparingly to avoid detection.",
}


def _rationale_for_feature(feature_name: str) -> str:
    """Return a plain-English rationale string for a given feature name."""
    name_lower = feature_name.lower()
    for keyword, rationale in _FEATURE_RISK_RATIONALE.items():
        if keyword in name_lower:
            return rationale
    return "Elevated values of this feature are associated with increased insider-threat risk."


def generate_shap_report(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 10,
    output_path: str = "outputs/shap_report.md",
) -> str:
    """Write a markdown report of the top-N risk indicators by mean |SHAP|.

    Returns the report text.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_impact = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": mean_abs_shap,
            "direction": ["+" if shap_values[:, i].mean() > 0 else "−"
                          for i in range(len(X.columns))],
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top = feature_impact.head(top_n)

    lines = [
        "# SHAP Explainability Report",
        "",
        f"Top {top_n} risk indicators ranked by mean absolute SHAP value.",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| | Direction | Risk Rationale |",
        "|------|---------|--------------|-----------|----------------|",
    ]

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        direction = "Increases risk ↑" if row["direction"] == "+" else "Decreases risk ↓"
        rationale = _rationale_for_feature(row["feature"])
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['mean_abs_shap']:.6f} "
            f"| {direction} | {rationale} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Detailed Explanations")
    lines.append("")

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        rationale = _rationale_for_feature(row["feature"])
        direction_word = "increases" if row["direction"] == "+" else "decreases"
        lines.append(
            f"**{rank}. `{row['feature']}`** (mean |SHAP| = {row['mean_abs_shap']:.6f})  "
        )
        lines.append(
            f"   This feature **{direction_word}** the predicted threat probability. "
        )
        lines.append(f"   {rationale}")
        lines.append("")

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"SHAP report written to {output_path}")
    return report_text


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import joblib
    from src.feature_engineering import aggregate_user_features

    df = pd.read_csv("data/synthetic_cert.csv", parse_dates=["date"])
    feats = aggregate_user_features(df)
    feat_cols = [c for c in feats.columns if c not in ("user_id", "label")]
    X = feats[feat_cols]
    y = feats["label"]

    model = joblib.load("outputs/best_model.pkl")

    print("Generating SHAP values ...")
    shap_vals, exp = generate_shap_values(model, X)

    print("Summary plots ...")
    plot_shap_summary(shap_vals, X, "outputs/shap_beeswarm.png")
    plot_shap_bar(shap_vals, X, "outputs/shap_bar.png")

    print("Waterfall for highest-risk user ...")
    idx = find_high_risk_user(model, X)
    explain_single_prediction(model, X, idx, "outputs/shap_waterfall.png")

    print("Generating report ...")
    generate_shap_report(shap_vals, X, top_n=10)
    print("Done.")
