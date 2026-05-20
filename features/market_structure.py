"""
Market Structure — Support/Resistance, Regimes, Cross-Asset Analysis
=====================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class MarketStructure:
    """Analyzes market microstructure: support/resistance, regimes, correlations."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._support_resistance(df)
        df = self._regime_features(df)
        df = self._price_structure(df)
        return df

    def compute_cross_asset(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add cross-asset correlation features to each asset's data."""
        if len(all_data) < 2:
            return all_data

        # Build returns matrix
        returns = {}
        for ticker, df in all_data.items():
            if "Close" in df.columns:
                returns[ticker] = df["Close"].pct_change()
        if not returns:
            return all_data
        ret_df = pd.DataFrame(returns)

        # Rolling correlations
        for ticker, df in all_data.items():
            if ticker not in returns:
                continue
            corr_cols = [t for t in returns if t != ticker][:5]  # Top 5 correlations
            for other in corr_cols:
                safe = other.replace("=","_").replace("^","_").replace("-","_")
                col_name = f"Corr_{safe}_20"
                aligned = pd.concat([returns[ticker], returns[other]], axis=1).dropna()
                if len(aligned) > 20:
                    corr = aligned.iloc[:,0].rolling(20).corr(aligned.iloc[:,1])
                    df[col_name] = corr.reindex(df.index)
            all_data[ticker] = df
        return all_data

    def _support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]; h = df["High"]; l = df["Low"]
        for w in [20, 50]:
            df[f"Resistance_{w}"] = h.rolling(w).max()
            df[f"Support_{w}"] = l.rolling(w).min()
            df[f"Dist_Resistance_{w}"] = (df[f"Resistance_{w}"] - c) / (c + 1e-10) * 100
            df[f"Dist_Support_{w}"] = (c - df[f"Support_{w}"]) / (c + 1e-10) * 100
            rng = df[f"Resistance_{w}"] - df[f"Support_{w}"]
            df[f"Range_Position_{w}"] = (c - df[f"Support_{w}"]) / (rng + 1e-10)
        return df

    def _regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]; ret = c.pct_change()
        # Volatility regime
        vol_20 = ret.rolling(20).std()
        vol_60 = ret.rolling(60).std()
        df["Vol_Regime"] = vol_20 / (vol_60 + 1e-10)
        # Trend regime
        sma_20 = c.rolling(20).mean()
        sma_50 = c.rolling(50).mean()
        df["Trend_Regime"] = np.where(sma_20 > sma_50, 1, -1)
        # Mean reversion regime
        df["MR_Regime"] = (c - sma_20) / (vol_20 * c + 1e-10)
        # Momentum regime
        mom_5 = ret.rolling(5).sum()
        mom_20 = ret.rolling(20).sum()
        df["Mom_Regime"] = np.sign(mom_5) + np.sign(mom_20)
        # Market breadth proxy (using rolling positive returns ratio)
        df["Pos_Return_Ratio_20"] = ret.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)
        return df

    def _price_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]; h = df["High"]; l = df["Low"]
        # Pivot points
        pp = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
        df["Pivot"] = pp
        df["Pivot_R1"] = 2*pp - l.shift(1)
        df["Pivot_S1"] = 2*pp - h.shift(1)
        df["Pivot_R2"] = pp + (h.shift(1) - l.shift(1))
        df["Pivot_S2"] = pp - (h.shift(1) - l.shift(1))
        # Price channels
        df["Price_Channel_Upper_10"] = h.rolling(10).max()
        df["Price_Channel_Lower_10"] = l.rolling(10).min()
        # Fibonacci retracement levels from recent swing
        swing_high = h.rolling(20).max()
        swing_low = l.rolling(20).min()
        rng = swing_high - swing_low
        df["Fib_236"] = swing_high - 0.236 * rng
        df["Fib_382"] = swing_high - 0.382 * rng
        df["Fib_500"] = swing_high - 0.500 * rng
        df["Fib_618"] = swing_high - 0.618 * rng
        return df


if __name__ == "__main__":
    ms = MarketStructure()
    dummy = pd.DataFrame({
        "Open": np.random.randn(200)+100, "High": np.random.randn(200)+101,
        "Low": np.random.randn(200)+99, "Close": np.random.randn(200)+100,
        "Volume": np.abs(np.random.randn(200))*1e6,
    })
    result = ms.compute_all(dummy)
    new_cols = [c for c in result.columns if c not in dummy.columns]
    print(f"Market structure features: {len(new_cols)}")
    for c in new_cols: print(f"  {c}")
