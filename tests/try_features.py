import sys
from pathlib import Path

# โฟลเดอร์หลักของโปรเจกต์ = โฟลเดอร์แม่ของ tests/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # ให้ Python หาโฟลเดอร์ features เจอ

import pandas as pd
from features.telecom import build_features

raw = pd.read_csv(ROOT / "data" / "sample_data_intw.csv")
feats = build_features(raw)

print(feats.shape)        # (209593, 18) คือ 209,593 แถว 18 feature
print(feats.head())
