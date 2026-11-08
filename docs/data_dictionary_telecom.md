# Data dictionary: telecom micro-credit data (Indonesia)

Author: Ratthapak Sinsiri (Bell)
Date: 2026-11-04 (updated 2026-11-05 with T05 findings: Q3 and Q10 recount, new Q12)
Status: draft for mentor review

Every number in this document comes from [`notebooks/t03_profile_telecom.ipynb`](../notebooks/t03_profile_telecom.ipynb), except those marked (T05), which come from [`features/telecom.py`](../features/telecom.py). Section numbers in brackets, e.g. (nb 2.3), point to the notebook section.

## 1. Source and file check

| Item | Value |
|---|---|
| Source | Kaggle, `sivakrishna3311/delinquency-telecom-dataset` |
| Downloaded | 2026-11-04 |
| File | `data/sample_data_intw.csv` (30,095,468 bytes) |
| Product | Prepaid airtime loan. Borrow 5 and pay back 6, or borrow 10 and pay back 12, within 5 days |
| Rows | **209,593** |
| Columns | **36**, not 37 as stated in the task (see note) |
| Customers (`msisdn`) | 186,243 |
| Period (`pdate`) | 2016-06-01 to 2016-08-21, 82 days, no missing days |
| Missing values | 0 |
| Fully duplicated rows | 1 |
| Licence | To confirm from the Kaggle page |

**Note on 36 vs 37 columns.** The file has 36 columns. The owner's variable description table also lists 36. The 37th column in the task is most likely a row-index column that some copies of the file include. It carries no information.

Amounts are in Indonesian Rupiah (IDR).

## 2. Target and bad rate

### 2.1 Target definition

| Item | Value |
|---|---|
| Target column | `label` |
| Meaning | Loan repaid within 5 days of issue |
| `label` = 1 | Repaid: **good** |
| `label` = 0 | Not repaid: **bad** |
| Values found | Only 0 and 1 (nb 3.1) |

⚠️ The coding is reversed from the usual credit-scoring convention, where 1 = bad. The notebook creates `bad = 1 - label` before computing bad rates. Do the same before modelling.

### 2.2 Direction check

The data supports 0 = bad (nb 3.2, 3.3). Label-0 customers are weaker on every activity measure:

| Median | `label` 0 | `label` 1 |
|---|---|---|
| Daily spend, 30 days (`daily_decr30`) | 33 | 2,128 |
| Top-ups, 30 days (`cnt_ma_rech30`) | 1 | 3 |
| Top-up total, 30 days (`sumamnt_ma_rech30`) | 773 | 5,498 |
| Balance before top-up (`medianmarechprebal30`) | 3.6 | 39 |
| Days on network (`aon`) | 399 | 545 |

The share of label 0 falls steadily as top-ups rise:

| Top-ups in 30 days | Rows | % `label` 0 |
|---|---|---|
| 0 | 27,979 | 39.8% |
| 1 | 37,238 | 20.6% |
| 2–4 | 76,993 | 7.5% |
| 5+ | 67,383 | 2.3% |

### 2.3 Bad rate

| Population | Rows | Bads | Bad rate |
|---|---|---|---|
| Whole file (nb 4.1) | 209,593 | 26,162 | **12.48%** |
| June 2016 | 83,154 | 13,261 | 15.95% |
| July 2016 | 85,765 | 12,901 | 15.04% |
| **August 2016** | **40,674** | **0** | **0.00%** |
| Recommended development sample (nb 4.4) | 168,289 | 26,162 | **15.55%** |

The whole-file rate of 12.48% matches the "about 12.5%" quoted for this dataset.

**⚠️ August has no bads.** 40,674 rows and not one default is not credible. The most likely cause is that August loans had not reached their due date, or outcomes were not updated, when the data was extracted. These rows are unobserved, not good. Keeping them understates risk by about 3 percentage points.

**No-loan rows.** 3,259 rows have no loan in the last 30 days (`cnt_loans30` = 0). All of them have `label` = 1. A customer without a loan cannot default, so these are not real outcomes.

**Recommended development sample:** June–July 2016, `cnt_loans30` ≥ 1, duplicate row removed. Bad rate 15.55%.

## 3. Data quality issues

| # | Issue | Columns | Size | Action |
|---|---|---|---|---|
| Q1 | **Immature target**: August has no bads | `pdate`, `label` | 40,674 rows | Exclude from development. Can be used as an unlabelled sample for stability checks |
| Q2 | **No-loan rows** labelled good | `cnt_loans30`, `label` | 3,259 rows | Exclude from development |
| Q3 | **Corrupted values**: random decimals far outside the normal range, e.g. `aon` = 999,860 days (nb 2.6) | 9 columns, see section 4 | About 1,047 per column; 10,232 rows have at least one. One of them is a whole number, `last_rech_date_ma` = 835,720 days, found in T05 | Set to missing, then impute or flag |
| Q4 | **Negative values** where negatives are impossible (nb 2.5) | `aon`, `daily_decr30/90`, `last_rech_date_ma/da` | 14 to 1,839 rows per column | Set to missing |
| Q5 | **Negative values** that may be real (balance in debit) | `rental30/90`, `medianmarechprebal30/90` | 1,646 to 5,929 rows per column | Keep. Confirm meaning with the data owner |
| Q6 | **Repeat customers** (nb 2.5) | `msisdn` | 19,756 customers have more than one row (23,350 extra rows); 591 have both labels | Split train/test **by `msisdn`** so one customer is never in both |
| Q7 | **Values do not match the description** | `medianamnt_loans30/90` | Values are only 0, 0.5, 1, 1.5, 2, 3. A median loan should be 6 or 12 | Do not use until the meaning is known |
| Q8 | **Loan amounts stored as repayment** | `maxamnt_loans30/90` | Values are 0/6/12, not 5/10 as described | None. Note only |
| Q9 | **Constant column** | `pcircle` | One value (`UPW`) | Drop |
| Q10 | **Last top-up over 90 days ago** in 90-day data (nb 2.7) | `last_rech_date_ma/da` | 47 and 345 rows (91 to 115 days). A 48th `last_rech_date_ma` value is 835,720 days and is counted in Q3 (T05) | Keep. May be real inactive customers |
| Q11 | Fully duplicated row | all | 1 row | Drop |
| Q12 | **Zero means "no top-up"**, not a real zero (T05) | `last_rech_date_ma`, `medianmarechprebal30/90` | No top-up in 90 days: 20,950 rows, 20,729 (98.9%) have `last_rech_date_ma` = 0 and all have `medianmarechprebal90` = 0. No top-up in 30 days: 27,979 rows, 27,821 (99.4%) have `medianmarechprebal30` = 0 | Set to missing when there was no top-up in the window. Otherwise 0 days reads as "topped up today" and 0 balance as "ran out" |

Consistency checks that passed (nb 2.7): 30-day counts and amounts never exceed 90-day ones, `amnt_loans30/90` are always multiples of 6, `maxamnt_loans90` is always 0/6/12.

## 4. Column dictionary

Use: **Target**, **ID** (identifier, not a feature), **Use**, **Clean** (use after fixing issues), **Review** (meaning unclear), **Drop**.

Median is over the whole file and is shown to give a sense of typical values.

| # | Column | Meaning | Type | Unit | Median | Issues | Use |
|---|---|---|---|---|---|---|---|
| 1 | `label` | Loan repaid within 5 days (1 = good, 0 = bad) | int | flag | 1 | Reversed coding | Target |
| 2 | `msisdn` | Mobile number (masked) | text | – | – | Q6 | ID |
| 3 | `aon` | Age on network | float | days | 527 | Q3 (2,089), Q4 (1,539) | Clean |
| 4 | `daily_decr30` | Average daily spend from main account, last 30 days | float | IDR | 1,469 | Q4 (1,839) | Clean |
| 5 | `daily_decr90` | Average daily spend from main account, last 90 days | float | IDR | 1,500 | Q4 (1,839) | Clean |
| 6 | `rental30` | Average main account balance, last 30 days | float | IDR | 1,084 | Q5 (5,628); owner unsure of definition | Review |
| 7 | `rental90` | Average main account balance, last 90 days | float | IDR | 1,334 | Q5 (5,929); owner unsure of definition | Review |
| 8 | `last_rech_date_ma` | Days since last main account top-up | float | days | 3 | Q3 (1,047), Q4 (1,315), Q10 (47), Q12 | Clean |
| 9 | `last_rech_date_da` | Days since last data account top-up | float | days | 0 | Q3 (1,047), Q4 (14), Q10 (345) | Clean |
| 10 | `last_rech_amt_ma` | Amount of last main account top-up | int | IDR | 1,539 | – | Use |
| 11 | `cnt_ma_rech30` | Number of main account top-ups, last 30 days | int | count | 3 | – | Use |
| 12 | `fr_ma_rech30` | Frequency of main account top-ups, last 30 days | float | – | 2 | Q3 (1,047); owner unsure of definition | Review |
| 13 | `sumamnt_ma_rech30` | Total main account top-ups, last 30 days | float | IDR | 4,628 | – | Use |
| 14 | `medianamnt_ma_rech30` | Median top-up amount, main account, last 30 days | float | IDR | 1,539 | – | Use |
| 15 | `medianmarechprebal30` | Median main account balance just before top-up, last 30 days | float | IDR | 34 | Q3 (1,047), Q5 (1,646), Q12 | Clean |
| 16 | `cnt_ma_rech90` | Number of main account top-ups, last 90 days | int | count | 4 | – | Use |
| 17 | `fr_ma_rech90` | Frequency of main account top-ups, last 90 days | int | – | 2 | Owner unsure of definition | Review |
| 18 | `sumamnt_ma_rech90` | Total main account top-ups, last 90 days | int | IDR | 7,226 | – | Use |
| 19 | `medianamnt_ma_rech90` | Median top-up amount, main account, last 90 days | float | IDR | 1,539 | – | Use |
| 20 | `medianmarechprebal90` | Median main account balance just before top-up, last 90 days | float | IDR | 36 | Q5 (1,730), Q12 | Clean |
| 21 | `cnt_da_rech30` | Number of data account top-ups, last 30 days | float | count | 0 | Q3 (1,047); 98% zero | Clean |
| 22 | `fr_da_rech30` | Frequency of data account top-ups, last 30 days | float | – | 0 | Q3 (1,047); 99% zero | Clean |
| 23 | `cnt_da_rech90` | Number of data account top-ups, last 90 days | int | count | 0 | 97% zero | Use |
| 24 | `fr_da_rech90` | Frequency of data account top-ups, last 90 days | int | – | 0 | 99.6% zero | Use |
| 25 | `cnt_loans30` | Number of loans, last 30 days | int | count | 2 | 0 = no loan (Q2) | Use |
| 26 | `amnt_loans30` | Total repayment due on loans, last 30 days | int | IDR | 12 | – | Use |
| 27 | `maxamnt_loans30` | Largest loan, last 30 days (stored as repayment: 6 or 12) | float | IDR | 6 | Q3 (1,047), Q8 | Clean |
| 28 | `medianamnt_loans30` | Described as median loan amount, last 30 days | float | ? | 0 | Q7 | Review |
| 29 | `cnt_loans90` | Number of loans, last 90 days | float | count | 2 | Q3 (1,047) | Clean |
| 30 | `amnt_loans90` | Total repayment due on loans, last 90 days | int | IDR | 12 | – | Use |
| 31 | `maxamnt_loans90` | Largest loan, last 90 days (stored as repayment: 6 or 12) | int | IDR | 6 | Q8 | Use |
| 32 | `medianamnt_loans90` | Described as median loan amount, last 90 days | float | ? | 0 | Q7 | Review |
| 33 | `payback30` | Average days to repay, last 30 days | float | days | 0 | – | Use |
| 34 | `payback90` | Average days to repay, last 90 days | float | days | 1.7 | – | Use |
| 35 | `pcircle` | Telecom circle (service area) | text | – | `UPW` | Q9 | Drop |
| 36 | `pdate` | Snapshot date (format `dd-mm-yyyy`) | text | date | – | Q1 | ID (for time split) |

Abbreviations: `ma` = main account (airtime balance), `da` = data account, `rech` = recharge (top-up), `cnt` = count, `fr` = frequency, `amnt` = amount, `30`/`90` = look-back window in days.

## 5. Open questions for the data owner

1. What is the 37th column in the task description? Is it a row index?
2. Were August 2016 outcomes observed when the data was extracted?
3. What do `rental30/90` measure, and can they be negative?
4. What do `fr_ma_rech30/90` and `fr_da_rech30/90` measure: days between top-ups, or something else?
5. What do `medianamnt_loans30/90` contain? The values (0 to 3) do not match "median loan amount".
6. Where do the corrupted values (Q3) come from? Were they added on purpose, e.g. as noise for a test?
7. What licence covers this data?
