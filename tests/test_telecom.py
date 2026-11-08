"""Unit tests for features/telecom.py. Run: python -m pytest"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from features.telecom import FEATURE_COLUMNS, REQUIRED_COLUMNS, build_features

TYPICAL = {
    "aon": 500, "cnt_ma_rech30": 3, "cnt_ma_rech90": 6,
    "sumamnt_ma_rech30": 3000, "sumamnt_ma_rech90": 9000,
    "last_rech_date_ma": 2, "medianmarechprebal30": 40,
    "medianmarechprebal90": 5, "cnt_loans30": 2, "cnt_loans90": 4,
}
# A customer who never topped up and never borrowed. Zeros are how the raw
# file stores this case (last_rech_date_ma and prebal are 0 too).
NEVER = {c: 0 for c in REQUIRED_COLUMNS}


def frame(*rows):
    return pd.DataFrame(list(rows))


def row(**changes):
    return {**TYPICAL, **changes}


def one(feats, col):
    return feats[col].iloc[0]


# ---------- typical customer ----------

def test_typical_customer_values():
    f = build_features(frame(TYPICAL))
    assert list(f.columns) == FEATURE_COLUMNS
    assert one(f, "recharge_count_30") == 3
    assert one(f, "recharge_count_90") == 6
    assert one(f, "recharge_amount_30") == 3000
    assert one(f, "recharge_amount_90") == 9000
    assert one(f, "days_since_last_recharge") == 2
    assert one(f, "prebal_median_30") == 40
    assert one(f, "prebal_median_90") == 5
    assert one(f, "loan_count_30") == 2
    assert one(f, "loan_count_90") == 4
    assert one(f, "tenure_days") == 500


# ---------- zero cases ----------

def test_never_recharged_does_not_crash_or_divide_by_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any divide warning fails the test
        f = build_features(frame(NEVER))
    values = f.iloc[0]
    assert not np.isinf(values.to_numpy(dtype=float)).any()
    assert values["recharge_count_30"] == 0
    assert values["recharge_count_90"] == 0
    assert values["recharge_amount_30"] == 0
    assert values["recharge_amount_90"] == 0
    assert values["loan_count_30"] == 0
    assert values["loan_count_90"] == 0


def test_never_recharged_zero_days_is_not_today():
    f = build_features(frame(NEVER))
    assert math.isnan(one(f, "days_since_last_recharge"))


def test_never_recharged_has_no_balance_dip_value():
    # prebal is stored as 0 when there was no top-up; that is not "ran out".
    f = build_features(frame(NEVER))
    for col in ["prebal_median_30", "prebal_median_90"]:
        assert math.isnan(one(f, col)), col


def test_topped_up_today_keeps_zero_days():
    f = build_features(frame(row(last_rech_date_ma=0)))
    assert one(f, "days_since_last_recharge") == 0


def test_no_top_up_in_30_days_but_some_in_90():
    f = build_features(frame(row(cnt_ma_rech30=0, sumamnt_ma_rech30=0,
                                 medianmarechprebal30=0, last_rech_date_ma=45)))
    assert one(f, "recharge_count_30") == 0
    assert math.isnan(one(f, "prebal_median_30"))
    assert one(f, "prebal_median_90") == 5
    assert one(f, "days_since_last_recharge") == 45


def test_loans_without_any_top_up():
    f = build_features(frame(row(cnt_ma_rech30=0, cnt_ma_rech90=0,
                                 sumamnt_ma_rech30=0, sumamnt_ma_rech90=0)))
    assert one(f, "recharge_count_90") == 0
    assert one(f, "loan_count_90") == 4


def test_zero_balance_with_top_ups_is_a_dip():
    f = build_features(frame(row(medianmarechprebal30=0)))
    assert one(f, "prebal_median_30") == 0


# ---------- missing cases ----------

def test_all_missing_row_gives_all_missing_features():
    f = build_features(frame({c: np.nan for c in REQUIRED_COLUMNS}))
    assert f.iloc[0].isna().all()


@pytest.mark.parametrize("col", REQUIRED_COLUMNS)
def test_one_missing_input_does_not_crash(col):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = build_features(frame(row(**{col: np.nan})))
    assert not np.isinf(f.to_numpy(dtype=float)).any()


def test_missing_count_gives_missing_count():
    f = build_features(frame(row(cnt_ma_rech90=np.nan)))
    assert math.isnan(one(f, "recharge_count_90"))
    # Unknown count: keep the reported values rather than guess "no top-up".
    assert one(f, "days_since_last_recharge") == 2
    assert one(f, "prebal_median_90") == 5
