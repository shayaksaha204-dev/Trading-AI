"""Quick test: count all features from all engines."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
n = 300
df = pd.DataFrame({
    "Open": np.cumsum(np.random.randn(n) * 0.5) + 100,
    "High": 0, "Low": 0, "Close": 0,
    "Volume": np.abs(np.random.randn(n)) * 1e6,
}, index=pd.date_range("2024-01-01", periods=n, freq="D"))
df["High"] = df["Open"] + np.abs(np.random.randn(n)) * 1.5
df["Low"] = df["Open"] - np.abs(np.random.randn(n)) * 1.5
df["Close"] = df["Open"] + np.random.randn(n) * 0.8

from features.technical import TechnicalFeatures
from features.market_structure import MarketStructure
from features.price_action import PriceAction
from features.smc import SmartMoneyConcepts
from features.ict import ICTConcepts
from features.wyckoff import WyckoffAnalysis
from features.elliott_wave import ElliottWave

base_cols = set(df.columns)
print(f"Base columns: {len(base_cols)}")

df = TechnicalFeatures().compute_all(df)
tech_new = set(df.columns) - base_cols
print(f"+ Technical:      {len(tech_new):>3} features")

prev = set(df.columns)
df = MarketStructure().compute_all(df)
ms_new = set(df.columns) - prev
print(f"+ Market Structure: {len(ms_new):>3} features")

prev = set(df.columns)
df = PriceAction().compute_all(df)
pa_new = set(df.columns) - prev
print(f"+ Price Action:   {len(pa_new):>3} features")

prev = set(df.columns)
df = SmartMoneyConcepts().compute_all(df)
smc_new = set(df.columns) - prev
print(f"+ SMC:            {len(smc_new):>3} features")

prev = set(df.columns)
df = ICTConcepts().compute_all(df)
ict_new = set(df.columns) - prev
print(f"+ ICT:            {len(ict_new):>3} features")

prev = set(df.columns)
df = WyckoffAnalysis().compute_all(df)
wy_new = set(df.columns) - prev
print(f"+ Wyckoff:        {len(wy_new):>3} features")

prev = set(df.columns)
df = ElliottWave().compute_all(df)
ew_new = set(df.columns) - prev
print(f"+ Elliott Wave:   {len(ew_new):>3} features")

# Count numeric features (what the model actually uses)
exclude = {"Ticker", "Category", "Date", "Datetime"}
numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
print(f"\n{'='*40}")
print(f"TOTAL NUMERIC FEATURES: {len(numeric)}")
print(f"{'='*40}")
print(f"\nDataFrame shape: {df.shape}")
print(f"NaN rows before drop: {df.isna().any(axis=1).sum()}")
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Clean rows: {len(df_clean)} / {len(df)}")
