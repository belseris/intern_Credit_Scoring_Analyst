"""Telecom features for the airtime micro-credit data (T05).

Input: the raw Kaggle file `data/sample_data_intw.csv` (one row per customer
per snapshot date), or any DataFrame with the columns in REQUIRED_COLUMNS.
Output: one row of features per input row, same index.

Usage:
    import pandas as pd
    from features.telecom import build_features

    raw = pd.read_csv("data/sample_data_intw.csv")
    feats = build_features(raw)

Rules every feature follows:
- Missing input gives a missing feature. Nothing is filled with 0.
- A customer with no top-ups in a window gets count 0 and amount 0, but the
  balance-before-top-up features are missing, because there was no top-up to
  measure.

Data quirks handled (see docs/data_dictionary_telecom.md):
- Q3 corrupted values: decimals in columns that must be whole numbers, and
  `medianmarechprebal30` >= 100,000. Set to missing.
- A whole-number day count of 10,000 or more is also corrupted (one row has
  `last_rech_date_ma` = 835,720). Set to missing.
- Q4 impossible negatives in tenure, daily spend and days since top-up. Set to
  missing.
- `last_rech_date_ma` is 0 for 98.9% of customers with no top-up in 90 days.
  There 0 means "no top-up", not "topped up today", so it is set to missing.
"""

import pandas as pd

# Columns that count things or days, so must be whole numbers (T03 nb 2.6).
INT_LIKE_COLUMNS = [
    "aon", "last_rech_date_ma", "cnt_ma_rech30", "cnt_ma_rech90",
    "cnt_loans30", "cnt_loans90",
]
# Columns where a negative value is impossible (T03 nb 2.5, Q4).
NON_NEGATIVE_COLUMNS = ["aon", "last_rech_date_ma"]
# A day count this large (about 27 years) is corrupted. Catches the rare
# corrupted value that happens to be a whole number, e.g. 835,720.
MAX_DAYS = 10_000
DAY_COLUMNS = ["aon", "last_rech_date_ma"]
# Corrupted balance values sit near 100,000 or 1,000,000; the real 90-day
# maximum is 41,457 (T03 nb 2.6).
PREBAL_MAX = 100_000

REQUIRED_COLUMNS = [
    "aon",
    "cnt_ma_rech30", "cnt_ma_rech90",
    "sumamnt_ma_rech30", "sumamnt_ma_rech90",
    "last_rech_date_ma",
    "medianmarechprebal30", "medianmarechprebal90",
    "cnt_loans30", "cnt_loans90",
]

FEATURE_COLUMNS = [
    # recharge frequency
    "recharge_count_30", "recharge_count_90",
    # recharge amount
    "recharge_amount_30", "recharge_amount_90",
    # recency
    "days_since_last_recharge",
    # balance dips
    "prebal_median_30", "prebal_median_90",
    # loan-taking frequency
    "loan_count_30", "loan_count_90",
    # tenure
    "tenure_days",
]


def clean_telecom(raw):
    """Return a numeric copy of the required columns with bad values set to missing.

    Text that is not a number becomes missing. The input is not changed.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    df = raw[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)

    for c in INT_LIKE_COLUMNS:
        df[c] = df[c].where(df[c] % 1 == 0)
    for c in NON_NEGATIVE_COLUMNS:
        df[c] = df[c].where(df[c] >= 0)
    for c in DAY_COLUMNS:
        df[c] = df[c].where(df[c] < MAX_DAYS)
    df["medianmarechprebal30"] = df["medianmarechprebal30"].where(
        df["medianmarechprebal30"] < PREBAL_MAX)
    return df


def build_features(raw):
    """Build telecom features from the raw data. Returns columns FEATURE_COLUMNS."""
    df = clean_telecom(raw)
    out = pd.DataFrame(index=raw.index)

    cnt30, cnt90 = df["cnt_ma_rech30"], df["cnt_ma_rech90"]
    amt30, amt90 = df["sumamnt_ma_rech30"], df["sumamnt_ma_rech90"]

    # Recharge frequency: number of main-account top-ups.
    out["recharge_count_30"] = cnt30
    out["recharge_count_90"] = cnt90

    # Recharge amount: total top-ups.
    out["recharge_amount_30"] = amt30
    out["recharge_amount_90"] = amt90

    # Days since last top-up. 0 with no top-up in 90 days means "none".
    out["days_since_last_recharge"] = df["last_rech_date_ma"].where(cnt90 != 0)

    # Balance dips. The data has no daily balance, so the proxy is the median
    # balance just before a top-up: near 0 means the customer usually runs
    # out before topping up. Without a top-up in the window it is undefined.
    out["prebal_median_30"] = df["medianmarechprebal30"].where(cnt30 != 0)
    out["prebal_median_90"] = df["medianmarechprebal90"].where(cnt90 != 0)

    # Loan-taking frequency.
    loans30, loans90 = df["cnt_loans30"], df["cnt_loans90"]
    out["loan_count_30"] = loans30
    out["loan_count_90"] = loans90

    # Tenure.
    out["tenure_days"] = df["aon"]

    return out[FEATURE_COLUMNS]
