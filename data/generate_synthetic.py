"""Generate a synthetic CERT-like insider threat dataset.

2000 users × 90 days of behavioral telemetry with ~5 % threat-positive rows.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_USERS = 2000
N_DAYS = 90
POS_RATE = 0.05

start_date = datetime(2024, 1, 1)

user_ids = [f"USER_{i:04d}" for i in range(N_USERS)]
date_range = [start_date + timedelta(days=i) for i in range(N_DAYS)]

total = N_USERS * N_DAYS
n_pos = int(total * POS_RATE)

df = pd.DataFrame(
    {
        "user_id": np.repeat(user_ids, N_DAYS),
        "date": np.tile(date_range, N_USERS),
    }
)
df["date"] = pd.to_datetime(df["date"])

labels = np.zeros(total, dtype=int)
labels[np.random.choice(total, size=n_pos, replace=False)] = 1
df["label"] = labels

# Per-user baseline parameters
uid_idx = np.repeat(np.arange(N_USERS), N_DAYS)

login_hour_base = np.random.normal(9, 2, N_USERS).clip(6, 18)
email_rate = np.random.gamma(3, 6, N_USERS)
file_rate = np.random.gamma(3, 15, N_USERS)
vpn_prob = np.random.beta(1, 5, N_USERS)
usb_prob = np.random.beta(0.5, 15, N_USERS)
failed_login_base = np.random.exponential(0.3, N_USERS)

# -- Feature generation --

df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

login_hour = np.random.normal(login_hour_base[uid_idx], 2).clip(0, 23)
df["login_hour"] = login_hour.astype(int)

df["after_hours_login"] = (
    (df["login_hour"] < 7) | (df["login_hour"] > 19)
).astype(int)

# usb_events – zero-inflated Poisson
usb_lambda = usb_prob[uid_idx]
usb_mask = np.random.random(total) < usb_lambda
usb_events = np.zeros(total, dtype=int)
usb_events[usb_mask] = np.random.poisson(
    np.random.gamma(2, 2, usb_mask.sum())
)
df["usb_events"] = usb_events

df["email_sent_count"] = np.random.poisson(email_rate[uid_idx])
df["email_external_ratio"] = np.random.beta(1.5, 8, total).round(3)

df["file_access_count"] = np.random.poisson(file_rate[uid_idx])

# failed_logins – zero-inflated Poisson
fl_mask = np.random.random(total) < 0.25
failed = np.zeros(total, dtype=int)
failed[fl_mask] = np.random.poisson(failed_login_base[uid_idx][fl_mask])
df["failed_logins"] = failed

df["vpn_usage"] = (np.random.random(total) < vpn_prob[uid_idx]).astype(int)

# -- Inject threat signal --
t = df["label"] == 1
n = t.sum()

df.loc[t, "after_hours_login"] = np.random.choice(
    [0, 1], size=n, p=[0.25, 0.75]
)
df.loc[t, "usb_events"] = np.random.poisson(np.random.gamma(3, 2, n))
df.loc[t, "email_external_ratio"] = np.random.beta(3, 1.5, n).round(3)
df.loc[t, "file_access_count"] = np.random.poisson(
    np.random.gamma(5, 20, n)
)
df.loc[t, "failed_logins"] = np.random.poisson(np.random.gamma(2, 4, n))
df.loc[t, "vpn_usage"] = np.random.choice(
    [0, 1], size=n, p=[0.35, 0.65]
)
df.loc[t, "login_hour"] = (
    np.random.normal(20, 3, n).clip(0, 23).astype(int)
)

# Column order & save
cols = [
    "user_id",
    "date",
    "login_hour",
    "is_weekend",
    "after_hours_login",
    "usb_events",
    "email_sent_count",
    "email_external_ratio",
    "file_access_count",
    "failed_logins",
    "vpn_usage",
    "label",
]
df = df[cols]

out = "data/synthetic_cert.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(f"Positive rate: {df['label'].mean():.3%}")
print(df.groupby("label").size().to_string())
print("\nHead:")
print(df.head().to_string())
print("\nDescribe:")
print(df.describe().to_string())
