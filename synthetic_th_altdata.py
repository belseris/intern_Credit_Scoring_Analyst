"""สร้างข้อมูล synthetic ของผู้สมัครสินเชื่อไทย ตาม docs/data_spec.md (ข้อมูลเติมเงินจากค่ายมือถือ)

ข้อมูลนี้สมมุติขึ้นทั้งหมด ใช้ฝึกและทดสอบ pipeline เท่านั้น ไม่ใช่ข้อมูลลูกค้าจริง
การผิดนัดถูกกำหนดจาก "ตัวแปรแฝง" 3 ตัวที่ไม่มีในไฟล์ (ความมั่นคง, วินัยการจ่าย, รายได้)
ตัวแปรในไฟล์เป็นแค่ร่องรอยที่มีสัญญาณรบกวน เหมือนข้อมูลจริงที่วัดนิสัยคนได้ไม่ตรง 100%

Run: python synthetic_th_altdata.py   -> data/synthetic_th_altdata.csv
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/synthetic_th_altdata.csv")
EXTRACTION_DATE = pd.Timestamp("2026-12-31")   # วันที่ดึงผลการผ่อน: สมัครหลัง ธ.ค. 2025 จะยังไม่ครบ 12 เดือน
PRICE_UP_FROM = pd.Timestamp("2025-10-01")     # สมมุติว่าแพ็กเกจขึ้นราคา 5% ตั้งแต่ ต.ค. 2025 (ให้ PSI มีอะไรให้วัด)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=40_000, seed=42):
    """คืน DataFrame ผู้สมัคร n คน หนึ่งแถวต่อหนึ่งคน คอลัมน์ตาม data_spec.md ส่วนที่ 5 + ข้อมูลบูโรและผลการผ่อน"""
    rng = np.random.default_rng(seed)

    # ตัวแปรแฝง (ไม่อยู่ในไฟล์)
    stability = rng.normal(size=n)                                        # ใช้เบอร์เดิมนาน เติมเงินสม่ำเสมอ
    discipline = 0.5 * stability + np.sqrt(0.75) * rng.normal(size=n)    # วินัยการจ่าย (สัมพันธ์กับความมั่นคง)
    income = rng.normal(size=n)                                           # ระดับรายได้

    # วันสมัคร ม.ค. 2025 ถึง มี.ค. 2026
    month_start = pd.date_range("2025-01-01", "2026-03-01", freq="MS")
    as_of = month_start[rng.integers(0, len(month_start), n)] + pd.to_timedelta(rng.integers(0, 28, n), unit="D")
    price = np.where(as_of >= PRICE_UP_FROM, 1.05, 1.0)

    # ข้อมูลส่วนตัวและข้อมูลบูโร (ใช้กำหนดกลุ่มลูกค้า ไม่ใช่ตัวทำนาย)
    age = np.clip(np.round(17 + rng.gamma(3.2, 5, n)), 18, 70).astype(int)
    sex = rng.choice(["F", "M"], n)
    no_file = rng.random(n) < 0.35
    tradelines = np.where(no_file, 0, 1 + rng.poisson(1.8, n))
    history = np.where(no_file, 0, rng.integers(1, 121, n))

    # ข้อมูลเติมเงินจากค่ายมือถือ
    tenure = np.clip(np.exp(6.4 + 0.6 * stability + 0.5 * rng.normal(size=n)), 15, 6000).round()
    lam90 = np.exp(1.7 + 0.35 * stability + 0.25 * income)
    cnt90 = rng.poisson(lam90)
    cnt30 = rng.binomial(cnt90, sigmoid(-0.7 + 0.25 * discipline + 0.3 * rng.normal(size=n)))
    cnt180 = cnt90 + rng.poisson(lam90 * np.exp(0.15 * rng.normal(size=n)))
    unit = np.exp(4.3 + 0.45 * income + 0.3 * rng.normal(size=n)) * price   # บาทต่อการเติมหนึ่งครั้ง

    def amount(cnt):
        return np.round(cnt * unit * np.exp(0.1 * rng.normal(size=n)), 2)

    days_since = np.select(
        [cnt30 > 0, cnt90 > 0, cnt180 > 0],
        [np.minimum(rng.geometric(np.clip((cnt30 + 0.5) / 30, 0.02, 1)) - 1, 29),
         30 + rng.integers(0, 60, n), 90 + rng.integers(0, 90, n)],
        180 + rng.integers(0, 200, n))
    interval_std = np.where(cnt90 >= 2, np.round(np.exp(1.6 - 0.45 * discipline + 0.35 * rng.normal(size=n)), 1), np.nan)
    low_balance = rng.binomial(90, sigmoid(-1.6 - 0.5 * stability - 0.4 * income))
    airtime_cnt = np.minimum(rng.poisson(np.exp(-0.3 - 0.5 * income - 0.2 * stability)), 20)
    ontime = np.where(airtime_cnt > 0,
                      np.round(rng.binomial(airtime_cnt, sigmoid(1.6 + 1.0 * discipline)) / np.maximum(airtime_cnt, 1), 3),
                      np.nan)
    sim_change = rng.poisson(0.06 + 0.12 * (stability < -1))

    # ผลการผ่อน: แย่สุดใน 12 เดือน (bad = 90+ DPD) จากตัวแปรแฝง + โชค
    risk = -0.75 * discipline - 0.45 * stability - 0.25 * income + 0.35 * sim_change
    latent = -2.7 + risk + rng.logistic(size=n)
    worst = np.where(latent > 0, "90+", np.where(latent > -0.8, "30-89", "0-29")).astype(object)

    # สินเชื่อที่เปิดไม่ถึง 12 เดือน: ยังไม่ทันได้เห็นการผิดนัดทั้งหมด (เหมือนเดือน ส.ค. ในข้อมูล T03)
    mob = ((EXTRACTION_DATE - as_of).days // 30).to_numpy()
    unseen = (mob < 12) & (worst == "90+") & (rng.random(n) > mob / 12)
    worst[unseen] = "0-29"

    df = pd.DataFrame({
        "customer_key": [hashlib.sha256(f"cust{i}".encode()).hexdigest()[:16] for i in range(n)],
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "consent_flag": (rng.random(n) < 0.93).astype(int),
        "age": age,
        "sex": sex,
        "tradeline_count": tradelines,
        "credit_history_months": history,
        "fraud_flag": (rng.random(n) < 0.003).astype(int),
        "deceased_flag": (rng.random(n) < 0.001).astype(int),
        "staff_flag": (rng.random(n) < 0.005).astype(int),
        "tenure_days": tenure,
        "recharge_count_30": cnt30,
        "recharge_count_90": cnt90,
        "recharge_count_180": cnt180,
        "recharge_amount_30": amount(cnt30),
        "recharge_amount_90": amount(cnt90),
        "recharge_amount_180": amount(cnt180),
        "days_since_last_recharge": days_since,
        "recharge_interval_std_90": interval_std,
        "low_balance_days_90": low_balance,
        "airtime_loan_count_90": airtime_cnt,
        "airtime_loan_ontime_ratio_90": ontime,
        "sim_change_count_180": sim_change,
        "months_on_book": mob,
        "worst_dpd_12m": worst,
    })
    # ไม่ยินยอม = ค่ายมือถือไม่ส่งข้อมูลมา (data_spec ส่วนที่ 6)
    telco_cols = df.columns[df.columns.get_loc("tenure_days"):df.columns.get_loc("sim_change_count_180") + 1]
    df.loc[df["consent_flag"] == 0, telco_cols] = np.nan
    return df


if __name__ == "__main__":
    data = generate()
    OUT.parent.mkdir(exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(data):,} rows x {data.shape[1]} columns")
