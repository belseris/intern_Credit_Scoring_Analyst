# Credit Scoring Analyst internship (Bell)

Credit scorecard practice on public and synthetic data, building towards a Thai thin-file scorecard that uses prepaid top-up data from a telco partner.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data (not in git)

The `data/` folder is ignored by git. Put the files here before running:

| File | Source | Used by |
|---|---|---|
| `data/credit.csv` | Taiwan credit card default data, 24,000-row copy from the team | `reference_solution.py` |
| `data/synthetic_th_altdata.csv` | Created automatically by `synthetic_th_altdata.py` (seed 42) | `th_altdata_scorecard.py` |

## Run

```bash
python reference_solution.py        # Taiwan scorecard, steps 1-6
python th_altdata_scorecard.py      # Thai thin-file scorecard on synthetic top-up data
```

On Windows, if Thai text in the output fails to print, run with `python -X utf8 ...`.

## Files

| File | What it is |
|---|---|
| `reference_solution.py` | Taiwan credit card scorecard: cleaning, features, WoE/IV, selection, logistic regression, points, validation |
| `synthetic_th_altdata.py` | Generator for synthetic Thai applicants with prepaid top-up fields (field list in `docs/data_spec.md`, added with T04) |
| `th_altdata_scorecard.py` | Scorecard on the synthetic data: population filter, missing bins, out-of-time test, positive-only variant, fairness check |
| `docs/model_5questions.md` | T01: 5 questions about the two models |
