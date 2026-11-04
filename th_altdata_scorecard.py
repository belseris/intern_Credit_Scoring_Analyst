# %% ขั้นที่ 1: โหลดข้อมูล synthetic + กำหนดกลุ่มลูกค้าตาม data_spec.md
# Scorecard สำหรับผู้สมัคร thin-file ไทย ด้วยข้อมูลเติมเงินจากค่ายมือถือ (docs/data_spec.md)
# ขั้นตอนเหมือน reference_solution.py แต่ต่างกัน 4 จุด:
#   1) ต้องคัดกลุ่มลูกค้า (thin-file, ยินยอม, ผลการผ่อนครบ 12 เดือน) ก่อนสร้างโมเดล
#   2) มีค่าว่างที่มีความหมาย (เช่น ไม่เคยยืมค่าโทร) -> ให้เป็นกลุ่ม "missing" ของตัวเอง
#   3) ทดสอบข้ามเวลา (out-of-time) ไม่ใช่แค่สุ่มแบ่ง
#   4) ลองแบบ "ใช้ข้อมูลมือถือในทางบวกเท่านั้น" ตามที่ research notes (R1) แนะนำ
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_th_altdata import EXTRACTION_DATE, OUT as DATA_FILE, generate

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def section(title):
    """พิมพ์หัวข้อคั่น ให้อ่านผลใน Terminal ง่าย"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check(label, value, ok=None):
    """พิมพ์ผลตรวจ 1 บรรทัด: [ OK ] = ผ่าน, [WARN] = ต้องจัดการ, ว่าง = ข้อมูลเฉยๆ"""
    status = {True: "[ OK ]", False: "[WARN]", None: "      "}[ok]
    print(f"  {status}  {label:<38} {value}")


# ---------------------------------------------------------------------------
section("1.1 LOAD: synthetic Thai applicants (one row per applicant)")
if not Path(DATA_FILE).exists():                      # ยังไม่มีไฟล์ -> สร้างจาก generator (seed เดิม ได้ข้อมูลเดิมทุกครั้ง)
    Path(DATA_FILE).parent.mkdir(exist_ok=True)
    generate().to_csv(DATA_FILE, index=False)
    print(f"  generated {DATA_FILE}")
raw = pd.read_csv(DATA_FILE, parse_dates=["as_of_date"])

check("applicants (rows)", f"{len(raw):,}")
check("columns", raw.shape[1])
check("application dates", f"{raw['as_of_date'].min().date()} to {raw['as_of_date'].max().date()}")
check("outcome extracted on", EXTRACTION_DATE.date())
print("\n  column groups: ID/date | consent + exclusion flags | bureau (thin-file test) | "
      "telco top-up fields (data_spec 5) | outcome")
print(raw.head(3).T.to_string())

# ---------------------------------------------------------------------------
section("1.2 POPULATION: who is in the development sample (data_spec 2-3)")
# คัดออกทีละเงื่อนไขตามลำดับ แล้วนับว่าแต่ละเงื่อนไขตัดไปกี่คน
thin = (raw["tradeline_count"] < 3) | (raw["credit_history_months"] < 36)
steps = [
    ("no telco consent (consent_flag = 0)", raw["consent_flag"] == 0),
    ("under 20 years old", raw["age"] < 20),
    ("fraud", raw["fraud_flag"] == 1),
    ("deceased", raw["deceased_flag"] == 1),
    ("staff", raw["staff_flag"] == 1),
    ("not thin-file (3+ tradelines and 36+ months)", ~thin),
    ("insufficient performance (< 12 months)", raw["months_on_book"] < 12),
    ("indeterminate (worst 30-89 DPD)", raw["worst_dpd_12m"] == "30-89"),
]
keep = pd.Series(True, index=raw.index)
check("start", f"{len(raw):,}")
for label, rule in steps:
    removed = int((keep & rule).sum())
    keep &= ~rule
    check(f"- {label}", f"{removed:>6,}  -> {int(keep.sum()):,} left")
df = raw[keep].copy()
df["bad"] = (df["worst_dpd_12m"] == "90+").astype(int)   # bad = 90+ DPD ใน 12 เดือน, good = ไม่เคยถึง 30 DPD
check("development population", f"{len(df):,} applicants, bad rate {df['bad'].mean():.2%}")

# ทำไมต้องตัดคนที่ผ่อนไม่ถึง 12 เดือน: ดูอัตรา bad ของ thin-file ที่ยินยอม แยกตามเดือนที่สมัคร
eligible = raw["consent_flag"].eq(1) & thin & raw["worst_dpd_12m"].ne("30-89")
by_month = raw[eligible].assign(bad=raw["worst_dpd_12m"].eq("90+"), month=raw["as_of_date"].dt.to_period("M"))
by_month = by_month.groupby("month").agg(applicants=("bad", "size"), bad_rate=("bad", "mean"),
                                         months_on_book=("months_on_book", "min"))
by_month["status"] = np.where(by_month["months_on_book"] >= 12, "used", "EXCLUDED: < 12 months")
by_month["bad_rate"] = by_month["bad_rate"].map("{:.1%}".format)
print("\n  bad rate by application month (thin-file, consented)")
print(by_month.to_string())
print("  -> 2026 applications have had less time to reach 90 DPD, so their bad rate looks too low")

# ---------------------------------------------------------------------------
section("1.3 HEALTH CHECK: missing values, duplicates, impossible values")
telco = ["tenure_days", "recharge_count_30", "recharge_count_90", "recharge_count_180",
         "recharge_amount_30", "recharge_amount_90", "recharge_amount_180", "days_since_last_recharge",
         "recharge_interval_std_90", "low_balance_days_90", "airtime_loan_count_90",
         "airtime_loan_ontime_ratio_90", "sim_change_count_180"]
check("duplicate customer_key", int(df["customer_key"].duplicated().sum()), ok=not df["customer_key"].duplicated().any())
check("negative values in telco fields", int((df[telco] < 0).sum().sum()), ok=not (df[telco] < 0).any().any())
order_ok = (df["recharge_count_30"] <= df["recharge_count_90"]) & (df["recharge_count_90"] <= df["recharge_count_180"])
check("count_30 <= count_90 <= count_180", f"{order_ok.mean():.1%} of rows", ok=bool(order_ok.all()))
check("ontime ratio within 0-1", bool(df["airtime_loan_ontime_ratio_90"].dropna().between(0, 1).all()), ok=True)

missing = df[telco].isna().sum()
for col, n_missing in missing[missing > 0].items():
    check(f"missing: {col}", f"{n_missing:,} ({n_missing / len(df):.1%})")
both = df["recharge_interval_std_90"].isna() == (df["recharge_count_90"] < 2)
none = df["airtime_loan_ontime_ratio_90"].isna() == (df["airtime_loan_count_90"] == 0)
check("interval_std missing = fewer than 2 top-ups", bool(both.all()), ok=bool(both.all()))
check("ontime_ratio missing = no airtime loan", bool(none.all()), ok=bool(none.all()))
print("  -> missing values here MEAN something (no airtime loan, too few top-ups): keep them as their own bin")


# %% ขั้นที่ 2: สร้างตัวแปร
section("2.1 BUILD FEATURES from the top-up fields")
feat = pd.DataFrame(index=df.index)
# ใช้ตรงๆ
for col in ["tenure_days", "recharge_count_90", "recharge_count_180", "recharge_amount_90",
            "days_since_last_recharge", "recharge_interval_std_90", "low_balance_days_90",
            "airtime_loan_count_90", "airtime_loan_ontime_ratio_90", "sim_change_count_180"]:
    feat[col] = df[col]
feat["sim_change_count_180"] = feat["sim_change_count_180"].clip(upper=2)   # 2 = "2 ครั้งขึ้นไป" เพราะ 3 ครั้งมีคนน้อยมาก
# สร้างใหม่ 2 ตัว
cnt90 = df["recharge_count_90"].where(df["recharge_count_90"] > 0)          # 0 ครั้ง -> ว่าง (หารไม่ได้)
feat["avg_topup_90"] = df["recharge_amount_90"] / cnt90                      # เติมครั้งละกี่บาท
feat["recharge_trend"] = (df["recharge_count_30"] / (cnt90 / 3)).clip(0, 3)  # เดือนล่าสุดเติมบ่อยกว่าค่าเฉลี่ย 3 เดือนไหม (1 = เท่าเดิม)

FEATURE_MEANING = {
    "tenure_days": "days since the number was activated",
    "recharge_count_90": "top-ups in 90 days",
    "recharge_count_180": "top-ups in 180 days",
    "recharge_amount_90": "THB topped up in 90 days",
    "days_since_last_recharge": "days since the last top-up",
    "recharge_interval_std_90": "irregularity of top-up gaps (days)",
    "low_balance_days_90": "days with balance < 10 THB, of 90",
    "airtime_loan_count_90": "airtime advances taken in 90 days",
    "airtime_loan_ontime_ratio_90": "share of airtime advances repaid on time",
    "sim_change_count_180": "SIM changes in 180 days (2 = 2 or more)",
    "avg_topup_90": "THB per top-up (90 days)",
    "recharge_trend": "last 30 days vs 90-day pace (1 = same)",
}
candidates = list(FEATURE_MEANING)
reference_only = ["age", "sex"]              # เก็บไว้ตรวจความเป็นธรรม ไม่ใส่ในโมเดล
feat = feat[candidates].join(df[reference_only + ["bad", "as_of_date"]])

check("candidate features", len(candidates), ok=True)
summary = feat[candidates].describe().T[["mean", "min", "50%", "max"]]
summary.insert(0, "missing", feat[candidates].isna().mean().map("{:.1%}".format))
summary.insert(0, "measures", [FEATURE_MEANING[c] for c in candidates])
print(summary.round(2).to_string())


# %% ขั้นที่ 3: แบ่งข้อมูลตามเวลา + WoE และ IV
from sklearn.model_selection import train_test_split

section("3.1 SPLIT: development (Jan-Sep 2025) vs out-of-time (Oct-Dec 2025)")
OOT_FROM = pd.Timestamp("2025-10-01")
dev = feat[feat["as_of_date"] < OOT_FROM]
oot = feat[feat["as_of_date"] >= OOT_FROM]
# ช่วงพัฒนาแบ่ง 70/30 แบบสุ่ม (stratify) ส่วน OOT คือ "ข้อสอบจากอนาคต" ไม่ได้ใช้ตอนสร้างโมเดล
train, test = train_test_split(dev, test_size=0.3, stratify=dev["bad"], random_state=42)
for name, part in [("train (Jan-Sep 2025, 70%)", train), ("test (Jan-Sep 2025, 30%)", test),
                   ("out-of-time (Oct-Dec 2025)", oot)]:
    check(name, f"{len(part):,} applicants, bad rate {part['bad'].mean():.2%}")
print("  -> bins, WoE and IV come from TRAIN only; test and out-of-time stay unseen until step 6")


def make_bins(series, n_bins=5):
    """จุดตัด quintile ของตัวแปรต่อเนื่อง (ไม่นับค่าว่าง) หรือ None ถ้ามีค่าไม่กี่แบบ"""
    s = series.dropna()
    if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 7:
        return None
    edges = s.quantile(np.linspace(0, 1, n_bins + 1)).unique()
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def label_value(v):
    """ชื่อกลุ่มของตัวแปรที่ใช้ค่าตัวเอง: 3.0 -> '3', ค่าว่าง -> 'missing'"""
    if pd.isna(v):
        return "missing"
    if isinstance(v, (int, float, np.number)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def assign_bins(series, edges):
    """ชื่อกลุ่มของแต่ละคน (จุดตัดจาก train); ค่าว่างได้กลุ่ม 'missing' ของตัวเอง"""
    if edges is None:
        return series.map(label_value)
    return pd.cut(series, bins=edges).astype(str).where(series.notna(), "missing")


def bin_order(series, edges):
    """ลำดับกลุ่มสำหรับแสดงผล: น้อยไปมาก แล้วตามด้วย missing"""
    if edges is None:
        values = series.dropna().unique()
        values = sorted(values) if pd.api.types.is_numeric_dtype(series) else sorted(map(str, values))
        order = [label_value(v) for v in values]
    else:   # ใช้ชื่อช่วงจาก pd.cut เอง เพราะ pd.cut ปัดตัวเลขในชื่อ (เช่น 164.29 -> 164.0)
        order = list(pd.cut(pd.Series([], dtype=float), bins=edges).cat.categories.astype(str))
    return order + (["missing"] if series.isna().any() else [])


def woe_table(bins, target, order):
    """จำนวนคน, อัตรา bad, WoE และ IV ของแต่ละกลุ่ม"""
    t = pd.DataFrame({"bin": bins, "bad": target}).groupby("bin")["bad"].agg(customers="count", bads="sum")
    t = t.reindex([o for o in order if o in t.index])
    t["goods"] = t["customers"] - t["bads"]
    t["bad_rate"] = t["bads"] / t["customers"]
    pct_good = (t["goods"] + 0.5) / (t["goods"].sum() + 0.5)
    pct_bad = (t["bads"] + 0.5) / (t["bads"].sum() + 0.5)
    t["woe"] = np.log(pct_good / pct_bad)          # บวก = ปลอดภัยกว่าค่าเฉลี่ย, ลบ = เสี่ยงกว่า
    t["iv"] = (pct_good - pct_bad) * t["woe"]
    return t.round(3)


def iv_strength(v):
    if v < 0.02:
        return "useless"
    if v < 0.1:
        return "weak"
    if v < 0.3:
        return "medium"
    if v < 0.5:
        return "strong"
    return "very strong - check"


def show_woe(col):
    """ตาราง WoE ที่อ่านง่าย"""
    t = woe_tables[col]
    out = pd.DataFrame({
        "customers": t["customers"].map("{:,}".format),
        "%goods": (t["goods"] / t["goods"].sum()).map("{:.1%}".format),
        "%bads": (t["bads"] / t["bads"].sum()).map("{:.1%}".format),
        "bad_rate": t["bad_rate"].map("{:.1%}".format),
        "woe": t["woe"].map("{:+.3f}".format),
        "iv": t["iv"].map("{:.3f}".format),
        "risk": [("safer " + "+" * min(int(abs(w) * 4) + 1, 8)) if w > 0 else ("riskier " + "-" * min(int(abs(w) * 4) + 1, 8))
                 for w in t["woe"]],
    })
    out.index = t.index
    out.index.name = col
    print(out.to_string())
    print(f"  {'':>12} total IV = {t['iv'].sum():.3f} ({iv_strength(t['iv'].sum())})")


def woe_direction(col):
    """นับว่า WoE เปลี่ยนทิศกี่ครั้ง (ไม่นับกลุ่ม missing): 0 = เรียงทางเดียว อธิบายง่าย"""
    w = woe_tables[col]["woe"].drop("missing", errors="ignore").to_numpy()
    signs = np.sign(np.diff(w))
    signs = signs[signs != 0]
    return int((np.diff(signs) != 0).sum()) if len(signs) > 1 else 0


bin_edges, bin_orders, woe_tables = {}, {}, {}
for col in candidates + reference_only:
    bin_edges[col] = make_bins(train[col])
    bin_orders[col] = bin_order(train[col], bin_edges[col])
    woe_tables[col] = woe_table(assign_bins(train[col], bin_edges[col]), train["bad"], bin_orders[col])

# ---------------------------------------------------------------------------
section("3.2 BINNING: how each variable was cut into groups")
for col in candidates + reference_only:
    edges = bin_edges[col]
    method = "own values" if edges is None else f"quintiles {np.round(edges[1:-1], 2).tolist()}"
    extra = " + missing bin" if "missing" in woe_tables[col].index else ""
    check(col, f"{len(woe_tables[col])} bins | {method}{extra}")

# ---------------------------------------------------------------------------
section("3.3 WoE TABLES: which groups are safer / riskier (train)")
print("  WoE = ln(%goods / %bads)   + = safer than average   - = riskier   0 = average")
for col in ["airtime_loan_ontime_ratio_90", "tenure_days", "recharge_interval_std_90"]:
    print()
    show_woe(col)

# ---------------------------------------------------------------------------
section("3.4 IV RANKING")
iv = pd.Series({col: woe_tables[col]["iv"].sum() for col in woe_tables}).sort_values(ascending=False)
print("  scale: <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 very strong (check)\n")
for col, value in iv.items():
    role = "reference only" if col in reference_only else "candidate"
    print(f"  {col:<30} {value:6.3f}  {iv_strength(value):<20} {role:<15} {'#' * max(1, int(value * 40))}")


# %% ขั้นที่ 4: คัดตัวแปร
def to_woe(data, col):
    """แทนค่าจริงด้วย WoE ของกลุ่ม (จุดตัดและ WoE จาก train); กลุ่มที่ไม่เคยเห็น -> 0"""
    mapping = woe_tables[col]["woe"]
    return assign_bins(data[col], bin_edges[col]).map(mapping).astype(float).fillna(0)


section("4.1 GATE 1: drop useless variables (IV < 0.02)")
strong = [c for c in candidates if iv[c] >= 0.02]
for col in candidates:
    check(col, f"IV {iv[col]:.3f}", ok=iv[col] >= 0.02)
print(f"  -> dropped: {[c for c in candidates if iv[c] < 0.02] or 'none'} | {len(strong)} variables pass")

# ---------------------------------------------------------------------------
section("4.2 GATE 2: drop duplicates (|correlation of WoE| > 0.7)")
train_woe = pd.DataFrame({c: to_woe(train, c) for c in strong})
corr = train_woe.corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(strong) for b in strong[i + 1:] if abs(corr.loc[a, b]) > 0.7]
to_drop = set()
for a, b, r in sorted(pairs, key=lambda p: -abs(p[2])):
    if a in to_drop or b in to_drop:
        continue
    loser, winner = (a, b) if iv[a] < iv[b] else (b, a)
    to_drop.add(loser)
    check(f"{a} vs {b}", f"corr {r:.2f} -> keep {winner} (IV {iv[winner]:.3f}), drop {loser}", ok=False)
if not pairs:
    print("  no pair above 0.7")
after_rules = [c for c in strong if c not in to_drop]
print(f"  -> {len(after_rules)} variables left: {after_rules}")

# ---------------------------------------------------------------------------
section("4.3 GATE 3: judgement - does the WoE pattern make business sense?")
print("  direction changes = how often WoE switches between going up and going down (0 = one direction)\n")
for col in after_rules:
    turns = woe_direction(col)
    check(col, f"{turns} direction change(s)", ok=turns <= 1)
JUDGEMENT_REASON = {
    "avg_topup_90": "WoE goes up, down, up (no clear story); its 'missing' bin (no top-up) repeats recharge_count_180",
}
for col, reason in JUDGEMENT_REASON.items():
    print()
    show_woe(col)
    print(f"  -> DROP: {reason}")
selected = [c for c in after_rules if c not in JUDGEMENT_REASON]

# ---------------------------------------------------------------------------
section("4.4 FUNNEL: candidates -> final features")
check("start (candidates)", len(candidates))
check("after gate 1 (IV >= 0.02)", len(strong))
check("after gate 2 (no duplicates)", f"{len(after_rules)}  dropped {sorted(to_drop)}")
check("after gate 3 (judgement)", f"{len(selected)}  dropped {list(JUDGEMENT_REASON) or 'none'}")
print()
for col in selected:
    check(col, f"IV {iv[col]:.3f}  {FEATURE_MEANING[col]}", ok=True)

X_train = pd.DataFrame({c: to_woe(train, c) for c in selected})
X_test = pd.DataFrame({c: to_woe(test, c) for c in selected})
X_oot = pd.DataFrame({c: to_woe(oot, c) for c in selected})
y_train, y_test, y_oot = train["bad"], test["bad"], oot["bad"]


# %% ขั้นที่ 5: Logistic Regression + แปลงเป็นคะแนน
from sklearn.linear_model import LogisticRegression


def show_coefs(model, columns):
    for col, c in zip(columns, model.coef_[0]):
        check(col, f"{c:+.3f}  {'correct sign' if c < 0 else 'WRONG SIGN'}", ok=c < 0)
    check("intercept", f"{model.intercept_[0]:+.3f}")


section("5.1 FIT + SIGN FIX: every coefficient must be negative")
logit = LogisticRegression(C=1e6, max_iter=1000)
logit.fit(X_train, y_train)
coefs = pd.Series(logit.coef_[0], index=X_train.columns)
show_coefs(logit, X_train.columns)
while (coefs > 0).any():
    worst = min(coefs[coefs > 0].index, key=lambda c: iv[c])
    print(f"  -> dropping {worst}: coefficient {coefs[worst]:+.3f} has the wrong sign (IV {iv[worst]:.3f})")
    selected = [c for c in selected if c != worst]
    X_train, X_test, X_oot = X_train[selected], X_test[selected], X_oot[selected]
    logit.fit(X_train, y_train)
    coefs = pd.Series(logit.coef_[0], index=X_train.columns)

section("5.2 FINAL MODEL")
show_coefs(logit, X_train.columns)
check("all coefficients negative", bool((coefs < 0).all()), ok=bool((coefs < 0).all()))
b0 = logit.intercept_[0]
print(f"\n  average applicant (all WoE = 0): PD = {1 / (1 + np.exp(-b0)):.1%}  (actual train bad rate {y_train.mean():.1%})")

section("5.3 SCORECARD: points for each group (600 = odds 50:1, PDO 20)")
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)
n = len(selected)
base_share = (offset - factor * b0) / n
base_points = round(base_share)
print(f"  factor {factor:.2f} | offset {offset:.2f} | points = {base_share:.1f} - {factor:.2f} x coefficient x WoE\n")
rows = []
for col in selected:
    for bin_label, woe in woe_tables[col]["woe"].items():
        rows.append({"feature": col, "bin": bin_label, "woe": woe,
                     "points": round(base_share - factor * coefs[col] * woe)})
scorecard = pd.DataFrame(rows)
for col in selected:
    part = scorecard[scorecard["feature"] == col]
    print(f"  {col}  (coefficient {coefs[col]:+.3f}, spread {part['points'].max() - part['points'].min()} points)")
    for _, r in part.iterrows():
        print(f"      {r['bin']:<22} WoE {r['woe']:+.3f}  -> {int(r['points']):>4} pts  {'#' * max(1, (int(r['points']) - 40) // 2)}")
    print()


def points_by_feature(data, positive_only=False):
    """คะแนนแยกตามตัวแปร; positive_only = ห้ามต่ำกว่าคะแนนพื้นฐาน (ข้อมูลมือถือเพิ่มคะแนนได้ หักไม่ได้)"""
    pts = {}
    for col in selected:
        lookup = scorecard[scorecard["feature"] == col].set_index("bin")["points"]
        p = assign_bins(data[col], bin_edges[col]).map(lookup).fillna(base_points).astype(float)
        pts[col] = p.clip(lower=base_points) if positive_only else p
    return pd.DataFrame(pts, index=data.index)


def score(data, positive_only=False):
    return points_by_feature(data, positive_only).sum(axis=1)


def pd_from_score(s):
    return 1 / (1 + np.exp((s - offset) / factor))


train_score = score(train)
gap = np.abs(pd_from_score(train_score) - logit.predict_proba(X_train)[:, 1]).max()
check("score range (train)", f"{int(train_score.min())} to {int(train_score.max())}")
check("max difference in PD vs model", f"{gap:.4f} (from rounding points)", ok=gap < 0.02)


# %% ขั้นที่ 6: วัดผล (test + out-of-time), PSI, Red/Amber/Green, ทางบวกเท่านั้น, ความเป็นธรรม
from sklearn.metrics import roc_auc_score, roc_curve


def metrics(y, s):
    """AUC/Gini/KS จากคะแนน (คะแนนต่ำ = เสี่ยง จึงใช้ -score)"""
    fpr, tpr, _ = roc_curve(y, -s)
    auc = roc_auc_score(y, -s)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": (tpr - fpr).max()}


def psi(expected, actual, n_bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True, sort=False) + 1e-6
    a = pd.cut(actual, edges).value_counts(normalize=True, sort=False) + 1e-6
    return float(((a - e) * np.log(a / e)).sum())


section("6.1 DISCRIMINATION: train vs test vs out-of-time")
test_score, oot_score = score(test), score(oot)
results = pd.DataFrame({"train": metrics(y_train, train_score), "test": metrics(y_test, test_score),
                        "out-of-time": metrics(y_oot, oot_score)}).T
print(results.round(3).to_string())
gap_test = results.loc["train", "Gini"] - results.loc["test", "Gini"]
gap_oot = results.loc["train", "Gini"] - results.loc["out-of-time", "Gini"]
check("Gini gap train - test", f"{gap_test:.3f}", ok=abs(gap_test) < 0.05)
check("Gini gap train - out-of-time", f"{gap_oot:.3f}", ok=abs(gap_oot) < 0.05)

section("6.2 DECILES (out-of-time): 1 = lowest score = riskiest")
decile = pd.qcut(oot_score.rank(method="first"), 10, labels=range(1, 11))
tab = pd.DataFrame({"decile": decile, "score": oot_score, "pred": pd_from_score(oot_score), "bad": y_oot})
tab = tab.groupby("decile", observed=True).agg(lo=("score", "min"), hi=("score", "max"), applicants=("bad", "size"),
                                               pred_pd=("pred", "mean"), actual=("bad", "mean"), bads=("bad", "sum"))
tab["cum%bads"] = (tab["bads"].cumsum() / tab["bads"].sum())
print(pd.DataFrame({"score": tab["lo"].astype(int).astype(str) + "-" + tab["hi"].astype(int).astype(str),
                    "applicants": tab["applicants"], "pred PD": tab["pred_pd"].map("{:.1%}".format),
                    "actual": tab["actual"].map("{:.1%}".format), "cum%bads": tab["cum%bads"].map("{:.1%}".format),
                    "": ["#" * int(a * 150) for a in tab["actual"]]}).to_string())

section("6.3 STABILITY: PSI of score")
psi_test, psi_oot = psi(train_score, test_score), psi(train_score, oot_score)
check("PSI train vs test (same months)", f"{psi_test:.4f}", ok=psi_test < 0.1)
check("PSI train vs out-of-time", f"{psi_oot:.4f}  (< 0.1 stable | 0.1-0.25 watch | > 0.25 shifted)", ok=psi_oot < 0.1)
for col in selected:                                  # ตัวแปรไหนขยับมากสุด (ต้นเหตุถ้า PSI สูง)
    v = psi(to_woe(train, col), to_woe(oot, col)) if train[col].nunique() > 1 else 0.0
    check(f"  CSI {col}", f"{v:.4f}", ok=v < 0.1)

section("6.4 RED / AMBER / GREEN on out-of-time (cut-offs from cost, not from data)")
cut_red, cut_amber = offset + factor * np.log(3), offset + factor * np.log(5)
check("Red   (score <= cut)", f"{cut_red:.1f} = PD 25.0%")
check("Amber (score <= cut)", f"{cut_amber:.1f} = PD 16.7%")
band = pd.cut(oot_score, [-np.inf, cut_red, cut_amber, np.inf], labels=["Red", "Amber", "Green"])
bands = pd.DataFrame({"band": band, "bad": y_oot}).groupby("band", observed=False)["bad"].agg(applicants="count", bads="sum")
bands["% applicants"] = (bands["applicants"] / bands["applicants"].sum()).map("{:.1%}".format)
bands["bad rate"] = (bands["bads"] / bands["applicants"].where(bands["applicants"] > 0)).map("{:.1%}".format)
bands["% of all bads"] = (bands["bads"] / bands["bads"].sum()).map("{:.1%}".format)
print(bands.to_string())

section("6.5 POSITIVE-ONLY VARIANT: phone data can add points but never remove them")
pos_score = score(oot, positive_only=True)
m_pos = metrics(y_oot, pos_score)
check("AUC out-of-time, standard", f"{results.loc['out-of-time', 'AUC']:.3f}")
check("AUC out-of-time, positive-only", f"{m_pos['AUC']:.3f}")
check("applicants whose score goes up", f"{(pos_score > oot_score).mean():.1%}")
red_pos = (pos_score <= cut_red).mean()
check("Red share: standard -> positive-only", f"{(oot_score <= cut_red).mean():.1%} -> {red_pos:.1%}")
print("  -> here phone data is the ONLY source, so positive-only puts everyone at or above average: nobody can be Red")
print("     positive-only fits when phone data is added ON TOP of another score (e.g. a thin bureau file), not alone")

section("6.6 FAIRNESS CHECK (age and sex are NOT in the model)")
fair = pd.DataFrame({"score": oot_score, "bad": y_oot, "red": oot_score <= cut_red,
                     "sex": oot["sex"], "age_group": pd.cut(oot["age"], [0, 25, 35, 50, 120],
                                                            labels=["20-25", "26-35", "36-50", "51+"])})
for group in ["sex", "age_group"]:
    g = fair.groupby(group, observed=True).agg(applicants=("bad", "size"), bad_rate=("bad", "mean"),
                                               red_share=("red", "mean"), avg_score=("score", "mean"))
    print(f"\n  by {group}")
    print(pd.DataFrame({"applicants": g["applicants"].map("{:,}".format), "bad rate": g["bad_rate"].map("{:.1%}".format),
                        "% in Red": g["red_share"].map("{:.1%}".format), "avg score": g["avg_score"].round(0)}).to_string())
print("\n  -> compare '% in Red' with 'bad rate': a group sent to Red much more often than its bad rate suggests needs review")

section("FINAL SUMMARY")
print(f"  population: {len(df):,} thin-file applicants with consent, bad rate {df['bad'].mean():.2%}")
print(f"  features ({len(selected)}): {', '.join(selected)}")
print(f"  AUC  train {results.loc['train', 'AUC']:.3f} | test {results.loc['test', 'AUC']:.3f} | "
      f"out-of-time {results.loc['out-of-time', 'AUC']:.3f}")
print(f"  PSI  train vs out-of-time {psi_oot:.4f}")
print("  NOTE: synthetic data; the numbers test the pipeline, not real Thai applicants")
