# Thai alternative-data spec v0

Status: draft v0 for mentor review
Author: Ratthapak Sinsiri (Bell)
Date: 2026-11-04

Data spec for scoring Thai thin-file applicants with prepaid top-up data from a telco partner.

## 1. Thin-file definition

An applicant is **thin-file** if **either** condition is true at the observation point:

- fewer than **3 tradelines** in the credit bureau (NCB), or
- credit history shorter than **36 months** (time since the oldest tradeline was opened).

A tradeline is one credit account in the bureau report, e.g. one credit card or one car loan.
Applicants with no bureau record (0 tradelines) are also thin-file.

## 2. Bad definition

| Item | Definition |
|---|---|
| Bad | 90+ days past due (DPD) at least once within 12 months after the loan is opened |
| Good | Never 30+ DPD within 12 months |
| Indeterminate | Worst status 30–89 DPD. Excluded from model training |
| Insufficient performance | Loans open less than 12 months at data extraction. Excluded |

## 3. Observation and performance windows

```
   observation window (180 days)        performance window (12 months)
|-------------------------------------|----------------------------------------|
                                 application date
                              (observation point)
```

| Window | Length | Used for |
|---|---|---|
| Observation point | Application date | The date the score is calculated |
| Observation window | 180 days before the application date | Building telco features (30 / 90 / 180-day summaries) |
| Performance window | 12 months after the loan is opened | Measuring good / bad |

Rule: every telco feature uses **only data before the application date** (no leakage).

## 4. Prepaid top-up fields from the telco partner

The telco computes these summaries on its side and sends one row per applicant.

| Field | Meaning | Window | Type | Unit |
|---|---|---|---|---|
| `customer_key` | Hashed key to match bank and telco records (no raw phone number or ID) | – | string | – |
| `as_of_date` | Date the features are computed (= application date) | – | date | – |
| `consent_flag` | Customer gave consent to share telco data for credit scoring | – | int (0/1) | – |
| `tenure_days` | Days since the number was activated | at as_of_date | int | days |
| `recharge_count_30` / `_90` / `_180` | Number of top-ups | 30 / 90 / 180 days | int | count |
| `recharge_amount_30` / `_90` / `_180` | Total top-up amount | 30 / 90 / 180 days | float | THB |
| `days_since_last_recharge` | Days since the last top-up | at as_of_date | int | days |
| `recharge_interval_std_90` | Standard deviation of days between top-ups (regularity) | 90 days | float | days |
| `low_balance_days_90` | Days with balance below 10 THB | 90 days | int | days |
| `airtime_loan_count_90` | Number of airtime advances taken from the telco | 90 days | int | count |
| `airtime_loan_ontime_ratio_90` | Share of airtime advances repaid on time | 90 days | float | 0–1 |
| `sim_change_count_180` | Number of SIM changes on the same number | 180 days | int | count |
