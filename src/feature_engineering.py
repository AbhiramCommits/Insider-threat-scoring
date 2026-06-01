"""Feature engineering pipeline for insider threat scoring.

Produces 40+ behavioural features aggregated per user from daily telemetry.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per user with 40+ features.

    Merges output of temporal, activity, deviation, and bonus features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    temporal = build_temporal_features(df)
    activity = build_activity_features(df)
    deviation = build_deviation_features(df)
    bonus = _build_bonus_features(df)

    features = (
        temporal
        .merge(activity, on="user_id", how="outer")
        .merge(deviation, on="user_id", how="outer")
        .merge(bonus, on="user_id", how="outer")
    )

    # Bring label along if present
    if "label" in df.columns:
        label = df.groupby("user_id")["label"].max().reset_index()
        features = features.merge(label, on="user_id", how="left")

    return features


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce after_hours_ratio, weekend_ratio, login_hour_variance,
    login_hour_mean, days_active."""
    temporal = df.groupby("user_id").agg(
        days_active=("date", "nunique"),
        after_hours_ratio=("after_hours_login", "mean"),
        weekend_ratio=("is_weekend", "mean"),
        login_hour_mean=("login_hour", "mean"),
        login_hour_variance=("login_hour", "var"),
    ).reset_index()
    temporal["login_hour_variance"] = temporal["login_hour_variance"].fillna(0)
    return temporal


# ---------------------------------------------------------------------------
# Activity features
# ---------------------------------------------------------------------------

def build_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce usb_event_total, usb_event_spike, email_volume_spike,
    external_email_ratio_mean, failed_login_total, file_access_anomaly_score."""
    activity = df.groupby("user_id").agg(
        usb_event_total=("usb_events", "sum"),
        external_email_ratio_mean=("email_external_ratio", "mean"),
        failed_login_total=("failed_logins", "sum"),
    ).reset_index()

    # Per-user z-score spike detection
    spike_data = []
    for uid, grp in df.groupby("user_id"):
        row = {"user_id": uid}

        for col, out_col in [
            ("usb_events", "usb_event_spike"),
            ("email_sent_count", "email_volume_spike"),
        ]:
            vals = grp[col]
            mu, sigma = vals.mean(), vals.std(ddof=0)
            if sigma == 0:
                row[out_col] = 0
            else:
                row[out_col] = int(((vals - mu).abs() / sigma).max() > 2)

        # file_access_anomaly_score = max z-score across days
        fa = grp["file_access_count"]
        fa_mu, fa_sigma = fa.mean(), fa.std(ddof=0)
        if fa_sigma == 0:
            row["file_access_anomaly_score"] = 0.0
        else:
            row["file_access_anomaly_score"] = ((fa - fa_mu).abs() / fa_sigma).max()

        spike_data.append(row)

    spikes = pd.DataFrame(spike_data)
    return activity.merge(spikes, on="user_id", how="outer")


# ---------------------------------------------------------------------------
# Deviation features (7-day rolling)
# ---------------------------------------------------------------------------

def build_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user rolling 7-day mean; flag deviations >2 std for login_hour,
    email_sent_count, file_access_count."""
    df = df.sort_values(["user_id", "date"])
    deviation_features = ["login_hour", "email_sent_count", "file_access_count"]

    results = []
    for uid, grp in df.groupby("user_id"):
        row = {"user_id": uid}
        for feat in deviation_features:
            rmean = grp[feat].rolling(7, min_periods=3).mean()
            rstd = grp[feat].rolling(7, min_periods=3).std().replace(0, np.nan)
            dev = (grp[feat] - rmean).abs() / rstd
            over_2 = dev > 2
            row[f"{feat}_deviation_count"] = int(over_2.sum())
            row[f"{feat}_max_deviation"] = (
                0.0 if dev.isna().all() else float(dev.max())
            )
            row[f"{feat}_deviation_ratio"] = float(over_2.mean())
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Bonus / derived features (pushing total above 40)
# ---------------------------------------------------------------------------

def _build_bonus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra features: time-of-day buckets, ratios, interaction terms."""
    # Time buckets from login_hour
    df = df.copy()
    df["login_morning"] = ((df["login_hour"] >= 7) & (df["login_hour"] < 12)).astype(int)
    df["login_afternoon"] = ((df["login_hour"] >= 12) & (df["login_hour"] < 19)).astype(int)
    df["login_evening"] = (df["login_hour"] >= 19).astype(int)
    df["login_night"] = (df["login_hour"] < 7).astype(int)

    bonus = df.groupby("user_id").agg(
        # login_hour distribution
        login_hour_std=("login_hour", "std"),
        login_hour_min=("login_hour", "min"),
        login_hour_max=("login_hour", "max"),
        login_hour_median=("login_hour", "median"),
        login_hour_skew=("login_hour", lambda x: x.skew() if x.std() > 0 else 0),
        morning_ratio=("login_morning", "mean"),
        afternoon_ratio=("login_afternoon", "mean"),
        evening_ratio=("login_evening", "mean"),
        night_ratio=("login_night", "mean"),
        # email
        email_volume_total=("email_sent_count", "sum"),
        email_volume_mean=("email_sent_count", "mean"),
        email_volume_max=("email_sent_count", "max"),
        email_volume_std=("email_sent_count", "std"),
        email_external_ratio_max=("email_external_ratio", "max"),
        email_external_ratio_std=("email_external_ratio", "std"),
        # USB
        usb_event_mean=("usb_events", "mean"),
        usb_event_max=("usb_events", "max"),
        usb_days_ratio=("usb_events", lambda x: (x > 0).mean()),
        # file
        file_access_total=("file_access_count", "sum"),
        file_access_mean=("file_access_count", "mean"),
        file_access_max=("file_access_count", "max"),
        file_access_std=("file_access_count", "std"),
        file_access_cv=("file_access_count", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        # failed logins
        failed_login_mean=("failed_logins", "mean"),
        failed_login_max=("failed_logins", "max"),
        failed_login_ratio=("failed_logins", lambda x: (x > 0).mean()),
        # VPN
        vpn_usage_ratio=("vpn_usage", "mean"),
        vpn_usage_total=("vpn_usage", "sum"),
        # interaction terms
        usb_after_hours_total=(
            "usb_events",
            lambda x: (x * df.loc[x.index, "after_hours_login"]).sum(),
        ),
        email_after_hours_total=(
            "email_sent_count",
            lambda x: (x * df.loc[x.index, "after_hours_login"]).sum(),
        ),
    ).reset_index()

    # high_risk_day_count: days with >= 3 anomalous indicators
    indicators = [
        "after_hours_login",
        "vpn_usage",
    ]
    # binary flags for USB / failed logins / email spike
    _fu = df.groupby("user_id")
    for col in ["usb_events", "failed_logins"]:
        df[f"{col}_high"] = _fu[col].transform(
            lambda x: (x > x.mean() + 2 * x.std(ddof=0)).astype(int)
            if x.std(ddof=0) > 0
            else 0
        )
        indicators.append(f"{col}_high")
    df["email_high"] = _fu["email_sent_count"].transform(
        lambda x: (x > x.mean() + 2 * x.std(ddof=0)).astype(int)
        if x.std(ddof=0) > 0
        else 0
    )
    indicators.append("email_high")
    df["file_high"] = _fu["file_access_count"].transform(
        lambda x: (x > x.mean() + 2 * x.std(ddof=0)).astype(int)
        if x.std(ddof=0) > 0
        else 0
    )
    indicators.append("file_high")

    df["risk_flags"] = df[indicators].sum(axis=1)
    high_risk = df.groupby("user_id")["risk_flags"].agg(
        high_risk_day_count=lambda x: (x >= 3).sum(),
        avg_risk_flags="mean",
    ).reset_index()

    bonus = bonus.merge(high_risk, on="user_id", how="outer")

    return bonus


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline and print summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_cert.csv", parse_dates=["date"])
    feats = aggregate_user_features(df)
    print(f"Feature matrix shape: {feats.shape}")
    print(f"Columns ({len(feats.columns)}):")
    for i, c in enumerate(feats.columns, 1):
        print(f"  {i:2d}. {c}")
