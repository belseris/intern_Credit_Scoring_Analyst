# %% ขั้นที่ 1: โหลดข้อมูล + ทำความสะอาด
import numpy as np
import pandas as pd

pd.set_option("display.width", 120)        # ให้ตารางกว้างพอ ไม่ตัดขึ้นบรรทัดใหม่
pd.set_option("display.max_columns", 30)   # แสดงครบทุกคอลัมน์ ไม่ซ่อนเป็น "..."


def section(title):
    """พิมพ์หัวข้อคั่น ให้อ่านผลใน Terminal ง่าย"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check(label, value, ok=None):
    """พิมพ์ผลตรวจ 1 บรรทัด: [ OK ] = ผ่าน, [WARN] = ต้องจัดการ, ว่าง = ข้อมูลเฉยๆ"""
    status = {True: "[ OK ]", False: "[WARN]", None: "      "}[ok]
    print(f"  {status}  {label:<34} {value}")


def code_table(data, col, meanings):
    """ตารางรหัส: ความหมาย, จำนวนคน, สัดส่วน, อัตราผิดนัด (รหัสที่ไม่มีใน dictionary ขึ้น ??)"""
    t = data.groupby(col)["default"].agg(customers="count", default_rate="mean")
    out = pd.DataFrame({
        "meaning": [meanings.get(c, "?? not in dictionary") for c in t.index],
        "customers": t["customers"].map("{:,}".format),
        "share": (t["customers"] / len(data)).map("{:.1%}".format),
        "default_rate": t["default_rate"].map("{:.1%}".format),
    }, index=t.index)
    print(out.to_string())


pay_cols = [f"PAY_{i}" for i in range(1, 7)]        # สถานะการจ่าย ก.ย. (1) -> เม.ย. (6)
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]   # ยอดบิล ก.ย. -> เม.ย.
pamt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]    # ยอดจ่ายจริง ก.ย. -> เม.ย.

# ความหมายของรหัสตาม data dictionary
EDUCATION_MEANING = {1: "graduate school", 2: "university", 3: "high school", 4: "others"}
MARRIAGE_MEANING = {1: "married", 2: "single", 3: "others"}
PAY_MEANING = {-2: "no card use", -1: "paid in full", 0: "paid minimum (revolving)",
               **{k: f"late {k} month" + ("s" if k > 1 else "") for k in range(1, 10)}}

# ---------------------------------------------------------------------------
section("1.1 LOAD + LOOK: what is in the file")
# ไฟล์ 24k มีชื่อคอลัมน์ PAY_1 และ default อยู่แล้ว ไม่ต้อง rename และไม่มีคอลัมน์ ID
df_raw = pd.read_csv("data/credit.csv")

check("customers (rows)", f"{df_raw.shape[0]:,}")
check("columns", df_raw.shape[1])
check("all columns numeric", bool(df_raw.dtypes.apply(pd.api.types.is_numeric_dtype).all()), ok=True)
print("\n  first 5 customers (one column per customer)")
print(df_raw.head().T)

# ---------------------------------------------------------------------------
section("1.2 HEALTH CHECK: missing values + duplicate rows")
dups = df_raw[df_raw.duplicated(keep=False)]          # keep=False = เอาทุกแถวในกลุ่มที่ซ้ำ
n_dup = int(df_raw.duplicated().sum())

check("missing values", int(df_raw.isna().sum().sum()), ok=df_raw.isna().sum().sum() == 0)
check("duplicate rows", f"{n_dup} ({n_dup / len(df_raw):.1%} of data)", ok=False)
check("  rows involved (pairs x 2)", len(dups))
check("  of which all 6 bills = 0", int((dups[bill_cols] == 0).all(axis=1).sum()))
check("  default rate of duplicates", f"{dups['default'].mean():.1%} (all data {df_raw['default'].mean():.1%})")
print("  -> decision: KEEP (no customer ID; mostly non-users who match by chance)")

# ---------------------------------------------------------------------------
section("1.3 RANGE CHECK: min / median / max of every column")
print(df_raw.describe().T[["mean", "min", "50%", "max"]].round(1))

# ---------------------------------------------------------------------------
section("1.4 CODE CHECK: codes not in the data dictionary")
print("\n  EDUCATION (dictionary: 1-4)")
code_table(df_raw, "EDUCATION", EDUCATION_MEANING)

print("\n  MARRIAGE (dictionary: 1-3)")
code_table(df_raw, "MARRIAGE", MARRIAGE_MEANING)

print("\n  PAY_1 = September payment status (dictionary: -2 to 9)")
code_table(df_raw, "PAY_1", PAY_MEANING)

print()
pay_values = df_raw[pay_cols].stack()                 # รวมรหัส PAY ทั้ง 6 เดือนเป็นคอลัมน์เดียว
check("SEX codes", sorted(df_raw["SEX"].unique().tolist()), ok=set(df_raw["SEX"]) <= {1, 2})
check("PAY_1..6 code range", f"{int(pay_values.min())} to {int(pay_values.max())}",
      ok=pay_values.between(-2, 9).all())

# ---------------------------------------------------------------------------
section("1.5 MONEY CHECK: values that matter for step 2")
neg_bill = (df_raw[bill_cols] < 0).any(axis=1)
over_limit = df_raw["BILL_AMT1"] > df_raw["LIMIT_BAL"]
check("LIMIT_BAL <= 0 (divide by zero)", int((df_raw["LIMIT_BAL"] <= 0).sum()), ok=(df_raw["LIMIT_BAL"] > 0).all())
check("any negative payment", int((df_raw[pamt_cols] < 0).any(axis=1).sum()), ok=(df_raw[pamt_cols] >= 0).all().all())
check("AGE range", f"{df_raw['AGE'].min()} to {df_raw['AGE'].max()}", ok=df_raw["AGE"].between(18, 100).all())
check("any negative bill (overpaid)", f"{int(neg_bill.sum()):,} customers ({neg_bill.mean():.1%})", ok=False)
check("latest bill > limit", f"{int(over_limit.sum()):,} customers ({over_limit.mean():.1%})", ok=False)
print("  -> negative bills and over-limit bills are handled with clip() in step 2")

# ---------------------------------------------------------------------------
section("1.6 CLEAN: merge undocumented codes")
df = df_raw.copy()   # แก้ที่สำเนา เก็บข้อมูลดิบไว้ตรวจย้อนได้

# EDUCATION: 0 = ไม่มีคำอธิบาย, 5/6 = ไม่ทราบ -> รวมเป็น 4 = อื่นๆ/ไม่ทราบ
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
# MARRIAGE: 0 = ไม่มีคำอธิบาย -> รวมเป็น 3 = อื่นๆ
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("  EDUCATION 0, 5, 6 -> 4 (others) | MARRIAGE 0 -> 3 (others)")
print("\n  EDUCATION after")
code_table(df, "EDUCATION", EDUCATION_MEANING)
print("\n  MARRIAGE after")
code_table(df, "MARRIAGE", MARRIAGE_MEANING)

# ---------------------------------------------------------------------------
section("1.7 VERIFY: nothing lost, raw data untouched")
check("shape after cleaning", df.shape, ok=df.shape == df_raw.shape)
check("rows lost", len(df_raw) - len(df), ok=len(df) == len(df_raw))
check("undocumented codes left", int((~df["EDUCATION"].isin(EDUCATION_MEANING)).sum()
                                     + (~df["MARRIAGE"].isin(MARRIAGE_MEANING)).sum()), ok=True)
check("raw data still has EDUCATION 5", bool((df_raw["EDUCATION"] == 5).any()),
      ok=(df_raw["EDUCATION"] == 5).any())                                      # True = copy() ทำงาน
check("default rate", f"{df['default'].mean():.2%} ({int(df['default'].sum()):,} of {len(df):,})")

# ---------------------------------------------------------------------------
section("STEP 1 SUMMARY")
print(f"  clean data: {len(df):,} customers x {df.shape[1]} columns, no missing values")
print(f"  merged undocumented codes: EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3")
print(f"  kept {n_dup} duplicate rows (no ID to prove they are the same person)")
print(f"  default rate {df['default'].mean():.2%} -> about 1 in 5 customers defaults")


# %% ขั้นที่ 2: สร้าง feature 11 ตัว
feat = pd.DataFrame(index=df.index)   # ตารางใหม่ แถวเดียวกับ df เก็บเฉพาะตัวแปรที่จะคัดเลือก


def pay_status(code):
    """แปลงรหัส PAY เป็นชื่อกลุ่ม เพราะ -2/-1/0 ไม่ได้เรียงตามความเสี่ยง"""
    if code == -2:
        return "no_use"
    if code == -1:
        return "paid_full"
    if code == 0:
        return "revolving"
    if code == 1:
        return "late_1m"
    if 2 <= code <= 9:
        return "late_2m_plus"   # รวมรหัส 2–9 เพราะแต่ละรหัสมีคนน้อย
    return "unknown"            # รหัสที่ไม่มีในคำอธิบาย จะได้ WoE = 0 ไม่หลุดเข้ากลุ่มผิด


# ---------------------------------------------------------------------------
section("2.1 EVIDENCE: this month's payment pays LAST month's bill")
same_month = (df["PAY_AMT1"] == df["BILL_AMT1"]).sum()   # จ่าย ก.ย. = บิล ก.ย.?
prev_month = (df["PAY_AMT1"] == df["BILL_AMT2"]).sum()   # จ่าย ก.ย. = บิล ส.ค.?
check("PAY_AMT1 == BILL_AMT1 (same month)", f"{same_month:,} customers")
check("PAY_AMT1 == BILL_AMT2 (previous month)", f"{prev_month:,} customers", ok=prev_month > same_month)
print(f"  -> {prev_month / same_month:.1f}x more matches with last month's bill: pair PAY_AMT(i) with BILL_AMT(i+1)")

# ---------------------------------------------------------------------------
section("2.2 BUILD 11 FEATURES")
# 1) สถานะการจ่ายเดือนล่าสุด (ก.ย.)
feat["pay_status_1"] = df["PAY_1"].apply(pay_status)
# 2) ใน 6 เดือน จ่ายช้ากี่เดือน (รหัส >= 1)
feat["months_late"] = (df[pay_cols] >= 1).sum(axis=1)
# 3) ใน 6 เดือน ไม่ได้ใช้บัตรกี่เดือน (รหัส -2)
feat["months_no_use"] = (df[pay_cols] == -2).sum(axis=1)

# 4–5) อัตราการใช้วงเงิน = บิล / วงเงิน จำกัดที่ 0–1.5 (ติดลบ = จ่ายเกิน = ไม่มีหนี้)
feat["util_1"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(0, 1.5)                # เดือนล่าสุด
feat["util_avg"] = (df[bill_cols].mean(axis=1) / df["LIMIT_BAL"]).clip(0, 1.5)   # เฉลี่ย 6 เดือน นิ่งกว่า

# 6–7) สัดส่วนการจ่ายคืน = ยอดจ่ายเดือนนี้ / บิลเดือนก่อน
ratios = []
for i in range(1, 6):                                   # จับคู่ได้ 5 คู่: (PAY_AMT1, BILL_AMT2) ... (PAY_AMT5, BILL_AMT6)
    bill = df[f"BILL_AMT{i + 1}"]
    paid = df[f"PAY_AMT{i}"]
    ratio = (paid / bill.where(bill > 0)).clip(0, 1)    # บิล <= 0 เป็น NaN กันหารด้วยศูนย์, จ่ายเกินนับเป็น 1
    ratios.append(ratio.fillna(1.0))                    # ไม่มีหนี้ = ไม่มีอะไรค้าง นับเป็นจ่ายครบ
feat["pay_ratio_1"] = ratios[0]                         # ล่าสุด: จ่าย ก.ย. ต่อบิล ส.ค.
feat["pay_ratio_avg"] = pd.concat(ratios, axis=1).mean(axis=1)

# 8) หนี้เพิ่มหรือลด: (บิล ก.ย. - บิล เม.ย.) / วงเงิน จำกัด -1 ถึง 1
feat["bill_trend"] = ((df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"]).clip(-1, 1)
# 9) กำลังจ่าย: ยอดจ่ายเฉลี่ยต่อเดือน / วงเงิน จำกัด 0–0.5
feat["pay_to_limit"] = (df[pamt_cols].mean(axis=1) / df["LIMIT_BAL"]).clip(0, 0.5)

# 10–11) ตัวแปรเดิมที่ใช้ตรงๆ
feat["LIMIT_BAL"] = df["LIMIT_BAL"]
feat["EDUCATION"] = df["EDUCATION"]
feat["default"] = df["default"]   # target เก็บไว้คำนวณ WoE ในขั้นที่ 3 (ไม่ใช่ตัวทำนาย)

FEATURE_MEANING = {
    "pay_status_1": "Sep payment status (5 groups)",
    "months_late": "months paid late, of 6",
    "months_no_use": "months card not used, of 6",
    "util_1": "Sep bill / limit",
    "util_avg": "avg 6-month bill / limit",
    "pay_ratio_1": "Sep payment / Aug bill",
    "pay_ratio_avg": "avg payment / prev bill (5 pairs)",
    "bill_trend": "(Sep bill - Apr bill) / limit",
    "pay_to_limit": "avg monthly payment / limit",
    "LIMIT_BAL": "credit limit (NT$)",
    "EDUCATION": "education code 1-4 (cleaned)",
}
check("candidate features", feat.shape[1] - 1, ok=feat.shape[1] - 1 == 11)
check("missing values", int(feat.isna().sum().sum()), ok=feat.isna().sum().sum() == 0)

# ---------------------------------------------------------------------------
section("2.3 CLIP IMPACT: how many customers were capped")
# เช็กว่าการ clip กระทบคนกี่ % (ถ้ามากเกินไปแปลว่าเลือกขอบผิด)
raw = {
    "util_1 (0..1.5)": df["BILL_AMT1"] / df["LIMIT_BAL"],
    "util_avg (0..1.5)": df[bill_cols].mean(axis=1) / df["LIMIT_BAL"],
    "bill_trend (-1..1)": (df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"],
    "pay_to_limit (0..0.5)": df[pamt_cols].mean(axis=1) / df["LIMIT_BAL"],
}
bounds = {"util_1 (0..1.5)": (0, 1.5), "util_avg (0..1.5)": (0, 1.5),
          "bill_trend (-1..1)": (-1, 1), "pay_to_limit (0..0.5)": (0, 0.5)}
for name, values in raw.items():
    lo, hi = bounds[name]
    below, above = (values < lo).mean(), (values > hi).mean()
    check(name, f"raised to min {below:5.1%} | capped at max {above:5.1%}")
print("  -> 'raised to min' for util = negative bills set to 0 (no debt); caps hit only a few customers")

# ---------------------------------------------------------------------------
section("2.4 FEATURE SUMMARY")
summary = feat.drop(columns=["default", "pay_status_1"]).describe().T[["mean", "min", "50%", "max"]]
summary.insert(0, "measures", [FEATURE_MEANING[c] for c in summary.index])
print(summary.round(3).to_string())

# ---------------------------------------------------------------------------
section("2.5 PAY_STATUS_1 GROUPS: codes -> named groups")
STATUS_MEANING = {"no_use": "code -2", "paid_full": "code -1", "revolving": "code 0",
                  "late_1m": "code 1", "late_2m_plus": "codes 2-9", "unknown": "other"}
code_table(feat, "pay_status_1", STATUS_MEANING)

# ---------------------------------------------------------------------------
section("2.6 EXAMPLE: customer 0, raw data -> features")
print("  PAY_1..6 (Sep -> Apr):", df.loc[0, pay_cols].tolist())
print("  BILL_AMT1..6         :", df.loc[0, bill_cols].astype(int).tolist())
print("  PAY_AMT1..6          :", df.loc[0, pamt_cols].astype(int).tolist())
print("  LIMIT_BAL            :", f"{df.loc[0, 'LIMIT_BAL']:,}")
print()
for col in FEATURE_MEANING:
    value = feat.loc[0, col]
    check(col, round(value, 3) if isinstance(value, float) else value)
check("default", int(feat.loc[0, "default"]))


# %% ขั้นที่ 3: แบ่ง train/test + WoE และ IV
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
section("3.1 SPLIT: train 70% (learn) / test 30% (exam)")
# แบ่ง 70/30; stratify ให้สัดส่วนผิดนัดเท่ากันทั้งสองชุด; random_state ให้ผลเหมือนเดิมทุกครั้ง
train, test = train_test_split(feat, test_size=0.3, stratify=feat["default"], random_state=42)

# ข้อมูลส่วนตัว: ต่อเข้ามาเพื่อวัด IV เปรียบเทียบเท่านั้น ไม่ใส่ในโมเดล
demo = df[["SEX", "AGE", "MARRIAGE"]]
train = train.join(demo)
test = test.join(demo)

check("train", f"{len(train):,} customers, default rate {train['default'].mean():.2%}")
check("test", f"{len(test):,} customers, default rate {test['default'].mean():.2%}",
      ok=abs(train["default"].mean() - test["default"].mean()) < 0.001)
check("columns per set", f"{train.shape[1]} = 11 features + default + SEX/AGE/MARRIAGE")
print("  -> everything below (bins, WoE, IV) is learned from TRAIN only; test stays unseen until step 6")


def make_bins(series, n_bins=5):
    """คืนจุดตัดของตัวแปรต่อเนื่อง หรือ None ถ้ามีค่าไม่กี่แบบ (ใช้ค่าเดิมเป็นช่วง)"""
    if series.dtype == object or series.nunique() <= 7:
        return None
    edges = series.quantile(np.linspace(0, 1, n_bins + 1)).unique()   # จุดตัดตาม quintile, unique() ตัดจุดที่ซ้ำ
    edges[0], edges[-1] = -np.inf, np.inf                              # ขอบนอกสุดเป็น ±∞ รองรับค่าในอนาคต
    return edges


def assign_bins(series, edges):
    """จัดแต่ละค่าเข้าช่วงด้วยจุดตัดจาก train (pd.cut ใช้จุดตัดที่ให้ ไม่คำนวณใหม่)"""
    if edges is None:
        return series
    return pd.cut(series, bins=edges)


def woe_table(bins, target):
    """คืนจำนวนคน, อัตราผิดนัด, WoE และ IV ของแต่ละช่วง"""
    t = pd.DataFrame({"bin": bins, "default": target}).groupby("bin", observed=True)["default"].agg(customers="count", bads="sum")
    t["goods"] = t["customers"] - t["bads"]
    t["default_rate"] = t["bads"] / t["customers"]
    pct_good = (t["goods"] + 0.5) / (t["goods"].sum() + 0.5)   # สัดส่วนคนดีที่อยู่ช่วงนี้ (+0.5 กัน log(0))
    pct_bad = (t["bads"] + 0.5) / (t["bads"].sum() + 0.5)      # สัดส่วนคนผิดนัดที่อยู่ช่วงนี้
    t["woe"] = np.log(pct_good / pct_bad)                      # บวก = ปลอดภัยกว่าค่าเฉลี่ย, ลบ = เสี่ยงกว่า
    t["iv"] = (pct_good - pct_bad) * t["woe"]                  # ส่วนของ IV ที่ช่วงนี้สร้าง
    return t.round(3)


def show_woe(col):
    """พิมพ์ตาราง WoE ให้อ่านง่าย: % ของคนดี/คนผิดนัด, อัตราผิดนัด, WoE, IV และแถบแสดงทิศทาง"""
    t = woe_tables[col]
    if bin_edges[col] is None and not pd.api.types.is_numeric_dtype(t.index):   # กลุ่มชื่อ: เรียงจากเสี่ยงสุดไปปลอดภัยสุด
        t = t.sort_values("woe")
    out = pd.DataFrame({
        "customers": t["customers"].map("{:,}".format),
        "goods": t["goods"].map("{:,}".format),
        "bads": t["bads"].map("{:,}".format),
        "%goods": (t["goods"] / t["goods"].sum()).map("{:.1%}".format),
        "%bads": (t["bads"] / t["bads"].sum()).map("{:.1%}".format),
        "default_rate": t["default_rate"].map("{:.1%}".format),
        "woe": t["woe"].map("{:+.3f}".format),
        "iv": t["iv"].map("{:.3f}".format),
        "risk": [("safer " + "+" * min(int(abs(w) * 4) + 1, 8)) if w > 0 else ("riskier " + "-" * min(int(abs(w) * 4) + 1, 8))
                 for w in t["woe"]],
    })
    out.index = t.index.astype(str)   # ตั้ง index หลังสร้าง ไม่งั้น pandas จับคู่ช่วงกับข้อความไม่ได้ -> NaN
    out.index.name = col
    print(out.to_string())
    print(f"  {'':>12} total IV = {t['iv'].sum():.3f} ({iv_strength(t['iv'].sum())})")


def iv_strength(v):
    """แปลค่า IV เป็นระดับความแรง"""
    if v < 0.02:
        return "useless"
    if v < 0.1:
        return "weak"
    if v < 0.3:
        return "medium"
    if v < 0.5:
        return "strong"
    return "very strong - check"


candidates = ["pay_status_1", "months_late", "months_no_use", "util_1", "util_avg",
              "pay_ratio_1", "pay_ratio_avg", "bill_trend", "pay_to_limit", "LIMIT_BAL", "EDUCATION"]
reference_only = ["SEX", "AGE", "MARRIAGE"]

# หาจุดตัดและตาราง WoE ของทุกตัวแปร จาก train อย่างเดียว
bin_edges, woe_tables = {}, {}
for col in candidates + reference_only:
    bin_edges[col] = make_bins(train[col])
    woe_tables[col] = woe_table(assign_bins(train[col], bin_edges[col]), train["default"])

# ---------------------------------------------------------------------------
section("3.2 BINNING: how each variable was cut into groups")
for col in candidates + reference_only:
    edges = bin_edges[col]
    method = "own values (few distinct)" if edges is None else f"quintiles -> cut points {np.round(edges[1:-1], 3).tolist()}"
    check(col, f"{len(woe_tables[col])} bins | {method}")

# ---------------------------------------------------------------------------
section("3.3 WoE TABLES: which groups are safer / riskier (train)")
print("  WoE = ln(%goods / %bads)   + = safer than average   - = riskier   0 = average")
for col in ["pay_status_1", "util_avg", "LIMIT_BAL"]:
    print()
    show_woe(col)

# ---------------------------------------------------------------------------
section("3.4 IV RANKING: how strongly each variable separates good from bad")
# IV ของตัวแปร = ผลรวม IV ทุกช่วง แล้วเรียงจากมากไปน้อย
iv = pd.Series({col: woe_tables[col]["iv"].sum() for col in woe_tables}).sort_values(ascending=False)
print("  scale: <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 very strong (check)\n")
for col, value in iv.items():
    role = "reference only" if col in reference_only else "candidate"
    bar = "#" * max(1, int(value * 40))
    print(f"  {col:<14} {value:6.3f}  {iv_strength(value):<20} {role:<15} {bar}")


# %% ขั้นที่ 4: คัดตัวแปร (IV < 0.02, correlation > 0.7, ตัดตามวิจารณญาณ)
def to_woe(data, col):
    """แทนค่าจริงด้วย WoE ของช่วง (จุดตัดและ WoE จาก train)"""
    mapping = woe_tables[col]["woe"]
    woe = assign_bins(data[col], bin_edges[col]).map(mapping)
    return pd.Series(woe, index=data.index).astype(float).fillna(0)   # ค่าที่ไม่เคยเห็น -> 0 = เสี่ยงเท่าค่าเฉลี่ย


# ---------------------------------------------------------------------------
section("4.1 GATE 1: drop useless variables (IV < 0.02)")
# ด่าน 1: ตัดตัวที่ IV < 0.02 (useless)
strong = [c for c in candidates if iv[c] >= 0.02]
for col in candidates:
    check(col, f"IV {iv[col]:.3f}", ok=iv[col] >= 0.02)
print(f"  -> dropped: {[c for c in candidates if iv[c] < 0.02] or 'none'} | {len(strong)} variables pass")

# ---------------------------------------------------------------------------
section("4.2 GATE 2: drop duplicates (|correlation of WoE| > 0.7)")
# ด่าน 2: หาคู่ที่ correlation ของ WoE เกิน 0.7 (สองตัววัดเรื่องเดียวกัน)
train_woe = pd.DataFrame({c: to_woe(train, c) for c in strong})
corr = train_woe.corr().round(2)
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(strong) for b in strong[i + 1:] if abs(corr.loc[a, b]) > 0.7]

short = {c: c.replace("pay_status_1", "status").replace("months_", "m_").replace("pay_ratio", "ratio")
            .replace("pay_to_limit", "pay_lim").replace("LIMIT_BAL", "limit").replace("EDUCATION", "edu")
         for c in strong}                                          # ชื่อย่อให้ตารางพอดีจอ
print(corr.rename(index=short, columns=short).to_string())

# ในแต่ละคู่ ตัดตัวที่ IV ต่ำกว่า
to_drop = set()
print()
for a, b, r in pairs:
    loser, winner = (a, b) if iv[a] < iv[b] else (b, a)
    to_drop.add(loser)
    check(f"{a} vs {b}", f"corr {r:.2f} -> keep {winner} (IV {iv[winner]:.3f}), drop {loser} (IV {iv[loser]:.3f})", ok=False)
after_rules = [c for c in strong if c not in to_drop]
print(f"  -> {len(after_rules)} variables left: {after_rules}")

# ---------------------------------------------------------------------------
section("4.3 GATE 3: judgement - does the WoE pattern make business sense?")
# ด่าน 3: ดูตาราง WoE ของตัวที่จะตัดตามวิจารณญาณ ว่ารูปแบบมีเหตุผลทางธุรกิจไหม
judgement_drop = ["pay_to_limit", "months_no_use", "EDUCATION"]
JUDGEMENT_REASON = {
    "pay_to_limit": "WoE zig-zags (- + - - +): paying more is not steadily safer -> no business story",
    "months_no_use": "flat for 0-4, then 5 risky / 6 safe: the useful part repeats pay_status_1 'no_use'",
    "EDUCATION": "IV mostly from group 4 (merged unknown codes, small); personal data -> fairness risk",
}
for col in judgement_drop:
    print()
    show_woe(col)
    print(f"  -> DROP: {JUDGEMENT_REASON[col]}")

selected = [c for c in after_rules if c not in judgement_drop]

# ---------------------------------------------------------------------------
section("4.4 FUNNEL: 11 candidates -> final features")
check("start (candidates)", len(candidates))
check("after gate 1 (IV >= 0.02)", len(strong))
check("after gate 2 (no duplicates)", f"{len(after_rules)}  dropped {sorted(to_drop)}")
check("after gate 3 (judgement)", f"{len(selected)}  dropped {judgement_drop}")
print()
for col in selected:
    check(col, f"IV {iv[col]:.3f} ({iv_strength(iv[col])})  {FEATURE_MEANING[col]}", ok=True)

# ---------------------------------------------------------------------------
section("4.5 WoE TRANSFORM: replace raw values with WoE (bins + WoE from train)")
# แปลง train และ test เป็น WoE (test ใช้จุดตัดและ WoE จาก train เท่านั้น)
X_train = pd.DataFrame({c: to_woe(train, c) for c in selected})
X_test = pd.DataFrame({c: to_woe(test, c) for c in selected})
y_train, y_test = train["default"], test["default"]
check("X_train", f"{X_train.shape[0]:,} customers x {X_train.shape[1]} features")
check("X_test", f"{X_test.shape[0]:,} customers x {X_test.shape[1]} features")
check("missing after transform", int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
      ok=X_train.notna().all().all() and X_test.notna().all().all())

example = train.index[0]                                          # ลูกค้าคนแรกใน train
print(f"\n  example: customer {example} (default = {int(train.loc[example, 'default'])})")
print(pd.DataFrame({"raw value": train.loc[example, selected],
                    "WoE": X_train.loc[example].map("{:+.3f}".format)}).to_string())


# %% ขั้นที่ 5: Logistic Regression + เช็กเครื่องหมาย + แปลงเป็นคะแนน
from sklearn.linear_model import LogisticRegression


def show_coefs(model, columns):
    """ตารางน้ำหนักของโมเดล + เช็กว่าติดลบไหม (WoE บวก = ปลอดภัย จึงต้องลด PD = น้ำหนักติดลบ)"""
    for col, c in zip(columns, model.coef_[0]):
        check(col, f"{c:+.3f}  {'correct sign' if c < 0 else 'WRONG SIGN: higher WoE (safer) -> higher PD?'}", ok=c < 0)
    check("intercept", f"{model.intercept_[0]:+.3f}")


# ---------------------------------------------------------------------------
section("5.1 FIT: logistic regression on WoE (train)")
# C=1e6 = แทบไม่หดน้ำหนัก (regularization) ให้คะแนนตรงกับข้อมูลจริง
# ไม่ใช้ class_weight เพื่อให้ PD ตรงกับอัตราผิดนัดจริง
logit = LogisticRegression(C=1e6, max_iter=1000)
logit.fit(X_train, y_train)
coefs = pd.Series(logit.coef_[0], index=X_train.columns).round(3)
show_coefs(logit, X_train.columns)
# เช็กเครื่องหมาย: WoE บวก = ปลอดภัย และโมเดลทำนายการผิดนัด น้ำหนักจึงต้องติดลบทุกตัว
check("all coefficients negative", bool((coefs < 0).all()), ok=bool((coefs < 0).all()))

# ---------------------------------------------------------------------------
section("5.2 SIGN FIX: drop wrong-sign variables one by one (lowest IV first)")
check("WoE corr pay_ratio_1 vs pay_ratio_avg", round(X_train["pay_ratio_1"].corr(X_train["pay_ratio_avg"]), 2))
# ตัดตัวที่เครื่องหมายผิดทีละตัว (IV ต่ำสุดก่อน) แล้วฝึกใหม่ จนติดลบครบทุกตัว
while (coefs > 0).any():
    wrong = coefs[coefs > 0].index
    worst = min(wrong, key=lambda c: iv[c])
    print(f"  -> dropping {worst}: coefficient {coefs[worst]:+.3f} has the wrong sign (IV {iv[worst]:.3f})")
    selected = [c for c in selected if c != worst]
    X_train, X_test = X_train[selected], X_test[selected]
    logit.fit(X_train, y_train)
    coefs = pd.Series(logit.coef_[0], index=X_train.columns).round(3)

# ---------------------------------------------------------------------------
section("5.3 FINAL MODEL")
show_coefs(logit, X_train.columns)
check("all coefficients negative", bool((coefs < 0).all()), ok=bool((coefs < 0).all()))
b0 = logit.intercept_[0]
print(f"\n  average customer (all WoE = 0): PD = 1 / (1 + e^-({b0:.3f})) = {1 / (1 + np.exp(-b0)):.1%}"
      f"  (actual train default rate {y_train.mean():.1%})")
print("  model: log-odds of default = intercept + sum(coefficient x WoE)")

# ---------------------------------------------------------------------------
section("5.4 SCALE: turn log-odds into points")
# สเกลคะแนน: 600 คะแนน = odds ดี 50 : เสีย 1, ทุก 20 คะแนน odds ดีขึ้น 2 เท่า
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
factor = PDO / np.log(2)                          # คะแนนต่อ log-odds 1 หน่วย
offset = BASE_SCORE - factor * np.log(BASE_ODDS)  # คะแนนที่ log-odds = 0 (odds 1:1)
n = len(selected)
check("BASE_SCORE / BASE_ODDS", f"{BASE_SCORE} points = odds good:bad {BASE_ODDS}:1 (PD {1 / (BASE_ODDS + 1):.1%})")
check("PDO (points to double odds)", f"+{PDO} points = odds x2")
check("factor = PDO / ln(2)", f"{factor:.2f} points per 1 unit of log-odds")
check("offset = 600 - factor x ln(50)", f"{offset:.2f} points at odds 1:1 (PD 50%)")
for s in [460, 500, 540, 580, 600]:
    odds = np.exp((s - offset) / factor)
    check(f"  score {s}", f"odds {odds:6.1f}:1  -> PD {1 / (1 + odds):5.1%}")

# ---------------------------------------------------------------------------
section("5.5 SCORECARD: points for each group")
# ตารางแต้ม: คะแนนพื้นฐานแบ่งเท่ากันทุกตัวแปร + คะแนนตามความเสี่ยงของช่วง
rows = []
for col in selected:
    for bin_label, woe in woe_tables[col]["woe"].items():
        points = (offset - factor * b0) / n - factor * coefs[col] * woe
        rows.append({"feature": col, "bin": str(bin_label), "woe": woe, "points": round(points)})
scorecard = pd.DataFrame(rows)

base_points = round((offset - factor * b0) / n)   # คะแนนของช่วงที่เสี่ยงเท่าค่าเฉลี่ย (WoE = 0)
print(f"  points = {(offset - factor * b0) / n:.1f} (base share) - {factor:.2f} x coefficient x WoE\n")
for col in selected:
    part = scorecard[scorecard["feature"] == col]
    spread = part["points"].max() - part["points"].min()
    print(f"  {col}  (coefficient {coefs[col]:+.3f}, spread {spread} points)")
    for _, r in part.iterrows():
        bar = "#" * max(1, (int(r["points"]) - 45) // 2)   # แถบยาวตามคะแนน (ย่อครึ่งให้พอดีจอ)
        print(f"      {r['bin']:<22} WoE {r['woe']:+.3f}  -> {int(r['points']):>4} pts  {bar}")
    print()


def points_by_feature(data):
    """คะแนนของลูกค้าแยกตามตัวแปร (ช่วงที่ไม่เคยเห็นได้คะแนนพื้นฐาน)"""
    pts = {}
    for col in selected:
        lookup = scorecard[scorecard["feature"] == col].set_index("bin")["points"]
        pts[col] = assign_bins(data[col], bin_edges[col]).astype(str).map(lookup).fillna(base_points).astype(float)
    return pd.DataFrame(pts, index=data.index)


def score(data):
    """คะแนนรวมของลูกค้า = ผลรวมคะแนนทุกตัวแปร"""
    return points_by_feature(data).sum(axis=1)


# ---------------------------------------------------------------------------
section("5.6 VERIFY: scorecard gives the same answer as the model")
train_score = score(train)
# เช็กว่าคะแนนให้ผลเหมือนโมเดล (ต่างกันได้นิดหน่อยจากการปัดคะแนน)
p_from_score = 1 / (1 + np.exp((train_score - offset) / factor))
p_from_model = logit.predict_proba(X_train)[:, 1]
check("base points (WoE = 0)", base_points)
check("score range (train)", f"{int(train_score.min())} to {int(train_score.max())}")
check("max difference in PD", f"{np.abs(p_from_score - p_from_model).max():.4f} (from rounding points)",
      ok=np.abs(p_from_score - p_from_model).max() < 0.02)

# ---------------------------------------------------------------------------
section("5.7 EXAMPLE: score two customers")
riskiest, safest = train_score.idxmin(), train_score.idxmax()
for who, cid in [("lowest score", riskiest), ("highest score", safest)]:
    pts = points_by_feature(train.loc[[cid]]).iloc[0]
    total = pts.sum()
    print(f"\n  {who}: customer {cid} (actual default = {int(train.loc[cid, 'default'])})")
    for col in selected:
        raw_value = train.loc[cid, col]
        shown = raw_value if isinstance(raw_value, str) else f"{raw_value:,.3f}".rstrip("0").rstrip(".")
        print(f"      {col:<14} {shown:>12}  -> {int(pts[col]):>4} pts")
    print(f"      {'TOTAL':<14} {'':>12}     {int(total):>4} pts  -> PD {1 / (1 + np.exp((total - offset) / factor)):.1%}")


# %% ขั้นที่ 6: วัดผล AUC/Gini/KS, ตาราง decile, PSI และกลุ่ม Red/Amber/Green
from sklearn.metrics import roc_auc_score, roc_curve


def metrics(y, p):
    """AUC: โอกาสที่คนผิดนัดถูกจัดว่าเสี่ยงกว่าคนดี; Gini = 2*AUC - 1; KS = ระยะห่างมากที่สุดของสองกลุ่ม"""
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": (tpr - fpr).max()}


def psi(expected, actual, n_bins=10):
    """PSI: การกระจายของคะแนนเปลี่ยนไปไหม (< 0.1 stable, 0.1–0.25 watch, > 0.25 shifted)"""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))   # แบ่ง 10 ช่วงตามคะแนนของ train
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True, sort=False) + 1e-6   # สัดส่วนคนแต่ละช่วงใน train
    a = pd.cut(actual, edges).value_counts(normalize=True, sort=False) + 1e-6     # สัดส่วนคนแต่ละช่วงใน test
    return float(((a - e) * np.log(a / e)).sum())


# ---------------------------------------------------------------------------
section("6.1 DISCRIMINATION: does the model rank bad customers below good ones?")
p_train_logit = logit.predict_proba(X_train)[:, 1]
p_test_logit = logit.predict_proba(X_test)[:, 1]   # ใช้ test เป็นครั้งแรก
results = pd.DataFrame({"train": metrics(y_train, p_train_logit), "test": metrics(y_test, p_test_logit)}).T
gini_gap = results.loc["train", "Gini"] - results.loc["test", "Gini"]

print(f"  {'':<8}{'AUC':>8}{'Gini':>8}{'KS':>8}")
for name, r in results.iterrows():
    print(f"  {name:<8}{r['AUC']:>8.3f}{r['Gini']:>8.3f}{r['KS']:>8.3f}")
print("  guide: AUC 0.5 = coin flip, 0.7-0.8 = acceptable/good for credit | Gini 0.4-0.6 typical | KS > 0.3 good\n")
check("test AUC", f"{results.loc['test', 'AUC']:.3f}", ok=results.loc["test", "AUC"] >= 0.7)
check("Gini gap (train - test)", f"{gini_gap:.3f} (> 0.05 would suggest overfitting)", ok=abs(gini_gap) < 0.05)

# ---------------------------------------------------------------------------
section("6.2 DECILES (test): 10 equal groups, 1 = lowest score = riskiest")
# ตาราง decile: แบ่ง test เป็น 10 กลุ่มเท่ากันตามคะแนน (1 = คะแนนต่ำสุด = เสี่ยงสุด)
test_score = score(test)
decile = pd.qcut(test_score.rank(method="first"), 10, labels=range(1, 11))   # rank แก้คะแนนซ้ำ ให้ทุกกลุ่มได้ 720 คน
bands = pd.DataFrame({"decile": decile, "score": test_score, "predicted": p_test_logit, "actual": y_test})
deciles = bands.groupby("decile", observed=True).agg(
    min_score=("score", "min"), max_score=("score", "max"), customers=("actual", "count"),
    predicted_rate=("predicted", "mean"), actual_rate=("actual", "mean"), defaulters=("actual", "sum"))
deciles["share_of_all_defaulters"] = deciles["defaulters"] / deciles["defaulters"].sum()
deciles["cumulative_share"] = deciles["share_of_all_defaulters"].cumsum()   # สะสม: คะแนนต่ำสุด X กลุ่มจับคนผิดนัดได้กี่ %

print(f"  {'dec':>3} {'score':>9} {'cust':>5} {'pred PD':>8} {'actual':>7} {'bads':>5} {'%bads':>6} {'cum%':>6}  actual default rate")
for d, r in deciles.iterrows():
    bar = "#" * int(r["actual_rate"] * 50)
    print(f"  {d:>3} {int(r['min_score']):>4}-{int(r['max_score']):<4} {int(r['customers']):>5} {r['predicted_rate']:>8.1%} "
          f"{r['actual_rate']:>7.1%} {int(r['defaulters']):>5} {r['share_of_all_defaulters']:>6.1%} {r['cumulative_share']:>6.1%}  {bar}")
print()
check("lowest 10% of scores catch", f"{deciles.loc[1, 'cumulative_share']:.1%} of all defaulters (random = 10%)")
check("lowest 30% of scores catch", f"{deciles.loc[3, 'cumulative_share']:.1%} of all defaulters (random = 30%)")
check("predicted vs actual (max gap)", f"{(deciles['predicted_rate'] - deciles['actual_rate']).abs().max():.1%} per decile",
      ok=(deciles["predicted_rate"] - deciles["actual_rate"]).abs().max() < 0.05)

# ---------------------------------------------------------------------------
section("6.3 STABILITY: PSI of score, train vs test")
psi_value = psi(train_score, test_score)
check("PSI", f"{psi_value:.4f}  (< 0.1 stable | 0.1-0.25 watch | > 0.25 shifted)", ok=psi_value < 0.1)

# ---------------------------------------------------------------------------
section("6.4 RED / AMBER / GREEN: cut-offs from cost, not from test")
# จุดตัดกลุ่มสีจากสูตรต้นทุน แปลงเป็นคะแนนด้วยสเกลของ scorecard (ไม่ได้ใช้ test เลือก)
cut_red = offset + factor * np.log(3)     # ต้นทุน 3:1 -> PD > 1/4 -> odds ดี:เสีย < 3
cut_amber = offset + factor * np.log(5)   # ต้นทุน 5:1 -> PD > 1/6 -> odds ดี:เสีย < 5
check("Red   (score <= cut)", f"{cut_red:.1f}  = odds 3:1 = PD 25.0%  (a default costs 3x a good customer's profit)")
check("Amber (score <= cut)", f"{cut_amber:.1f}  = odds 5:1 = PD 16.7%  (stricter view: 5x)")
check("Green", f"> {cut_amber:.1f}")

band = pd.cut(test_score, bins=[-np.inf, cut_red, cut_amber, np.inf], labels=["Red", "Amber", "Green"])
band_check = pd.DataFrame({"band": band, "default": y_test}).groupby("band", observed=True)["default"].agg(
    accounts="count", default_rate="mean", defaulters="sum")
band_check["share_of_accounts"] = band_check["accounts"] / band_check["accounts"].sum()
band_check["share_of_defaulters"] = band_check["defaulters"] / band_check["defaulters"].sum()

print(f"\n  {'band':<6} {'accounts':>9} {'% accts':>8} {'default rate':>13} {'defaulters':>11} {'% of all bads':>14}")
for b, r in band_check.iterrows():
    print(f"  {b:<6} {int(r['accounts']):>9,} {r['share_of_accounts']:>8.1%} {r['default_rate']:>13.1%} "
          f"{int(r['defaulters']):>11,} {r['share_of_defaulters']:>14.1%}")

# ---------------------------------------------------------------------------
section("FINAL SUMMARY")
print(f"  features ({len(selected)}): {', '.join(selected)}")
print(f"  test AUC {results.loc['test', 'AUC']:.3f} | Gini {results.loc['test', 'Gini']:.3f} | KS {results.loc['test', 'KS']:.3f}"
      f" | Gini gap {gini_gap:.3f} | PSI {psi_value:.4f}")
print(f"  score range (test): {int(test_score.min())} to {int(test_score.max())}")
print(f"  Red band: {band_check.loc['Red', 'share_of_accounts']:.1%} of accounts, "
      f"{band_check.loc['Red', 'share_of_defaulters']:.1%} of defaulters "
      f"(default rate {band_check.loc['Red', 'default_rate']:.1%} vs {y_test.mean():.1%} overall)")
