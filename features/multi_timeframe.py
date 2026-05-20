"""
Multi-Timeframe Features — Higher-TF Context for Intraday Models
==================================================================
Computes trend/momentum/volatility on 15m and 1h data,
then resamples them to the primary (5m) timeframe.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class MultiTimeframeFeatures:
    """
    Adds higher-timeframe context to the primary DataFrame.
    
    For each higher TF, computes:
    - Trend direction (SMA cross)
    - RSI
    - ATR (volatility)
    - MACD histogram sign
    Then forward-fills onto the primary timeframe (no future leak).
    """

    def compute_mtf_features(
        self,
        primary_df: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Enrich primary_df with features from higher timeframes.
        
        Args:
            primary_df: Primary (e.g., 5m) DataFrame with DatetimeIndex
            htf_data: {"15m": df_15m, "1h": df_1h, ...}
        """
        df = primary_df.copy()

        for tf_name, htf_df in htf_data.items():
            if htf_df is None or len(htf_df) < 30:
                logger.debug(f"  MTF: Skipping {tf_name} (insufficient data)")
                continue

            prefix = f"MTF_{tf_name}"
            htf = htf_df.copy()

            # Compute features on the higher TF
            c = htf["Close"]

            # Trend: SMA cross
            sma_fast = c.rolling(10).mean()
            sma_slow = c.rolling(30).mean()
            htf[f"{prefix}_Trend"] = np.where(sma_fast > sma_slow, 1.0, -1.0)
            htf[f"{prefix}_Trend_Strength"] = ((sma_fast / sma_slow) - 1.0) * 100

            # RSI
            delta = c.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            ag = gain.rolling(14).mean()
            al = loss.rolling(14).mean()
            htf[f"{prefix}_RSI"] = 100 - 100 / (1 + ag / (al + 1e-10))

            # ATR (volatility)
            tr = pd.concat([
                htf["High"] - htf["Low"],
                (htf["High"] - c.shift(1)).abs(),
                (htf["Low"] - c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            htf[f"{prefix}_ATR_Pct"] = (tr.rolling(14).mean() / (c + 1e-10)) * 100

            # MACD histogram sign
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            htf[f"{prefix}_MACD_Sign"] = np.where(macd > signal, 1.0, -1.0)

            # Bollinger Band position
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            htf[f"{prefix}_BB_Pos"] = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)

            # Select only the MTF feature columns
            mtf_cols = [col for col in htf.columns if col.startswith(prefix)]
            htf_features = htf[mtf_cols].copy()

            # Resample to primary timeframe via forward-fill (no future leak)
            # Reindex to primary df's index, forward-filling from the HTF
            htf_features = htf_features.reindex(df.index, method='ffill')

            # Merge into primary df
            for col in mtf_cols:
                df[col] = htf_features[col]

        # Cross-timeframe alignment signal
        trend_cols = [c for c in df.columns if c.endswith("_Trend") and c.startswith("MTF_")]
        if trend_cols:
            # All HTFs agree on direction
            trends = df[trend_cols].fillna(0)
            df["MTF_Alignment"] = trends.mean(axis=1)  # -1 to 1
            df["MTF_All_Bullish"] = (trends > 0).all(axis=1).astype(float)
            df["MTF_All_Bearish"] = (trends < 0).all(axis=1).astype(float)

        n_added = len([c for c in df.columns if c.startswith("MTF_")])
        logger.debug(f"  MTF: Added {n_added} multi-timeframe features")

        return df

    def get_feature_names(self, timeframes: List[str] = None) -> List[str]:
        """Return expected feature names."""
        tfs = timeframes or ["15m", "1h"]
        names = []
        for tf in tfs:
            p = f"MTF_{tf}"
            names.extend([
                f"{p}_Trend", f"{p}_Trend_Strength", f"{p}_RSI",
                f"{p}_ATR_Pct", f"{p}_MACD_Sign", f"{p}_BB_Pos",
            ])
        names.extend(["MTF_Alignment", "MTF_All_Bullish", "MTF_All_Bearish"])
        return names
