"""
Price Action Engine — Structural Analysis (BOS, CHoCH, Swing Points)
======================================================================
Pure price action: swing highs/lows, structure breaks, trend structure.
Optimized for intraday/scalping timeframes.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from loguru import logger


class PriceAction:
    """
    Pure price action analysis engine.
    
    Detects:
        - Swing Highs / Swing Lows (with configurable lookback)
        - Higher Highs, Higher Lows, Lower Highs, Lower Lows
        - Break of Structure (BOS) — trend continuation
        - Change of Character (CHoCH) — trend reversal signal
        - Market structure (bullish/bearish/ranging)
        - Impulse vs corrective moves
        - Equal highs/lows (liquidity targets)
    """

    def compute_all(self, df: pd.DataFrame, swing_lookback: int = 5) -> pd.DataFrame:
        df = df.copy()
        df = self._detect_swings(df, swing_lookback)
        df = self._swing_structure(df)
        df = self._bos_choch(df)
        df = self._impulse_correction(df)
        df = self._equal_levels(df)
        df = self._displacement(df)
        df = self._market_structure_score(df)
        return df

    def _detect_swings(self, df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """Identify swing highs and swing lows."""
        h = df["High"].values
        l = df["Low"].values
        n = len(df)
        
        swing_high = np.zeros(n, dtype=np.float64)
        swing_low = np.zeros(n, dtype=np.float64)
        is_swing_high = np.zeros(n, dtype=np.int32)
        is_swing_low = np.zeros(n, dtype=np.int32)

        for i in range(lookback, n - lookback):
            # Swing High: higher than all neighbors in lookback window
            if h[i] == max(h[i - lookback:i + lookback + 1]):
                is_swing_high[i] = 1
                swing_high[i] = h[i]
            # Swing Low: lower than all neighbors in lookback window
            if l[i] == min(l[i - lookback:i + lookback + 1]):
                is_swing_low[i] = 1
                swing_low[i] = l[i]

        # Forward-fill last known swing levels
        last_sh = 0.0
        last_sl = 0.0
        current_sh = np.zeros(n)
        current_sl = np.zeros(n)
        for i in range(n):
            if swing_high[i] > 0:
                last_sh = swing_high[i]
            if swing_low[i] > 0:
                last_sl = swing_low[i]
            current_sh[i] = last_sh
            current_sl[i] = last_sl

        df["PA_Swing_High"] = current_sh
        df["PA_Swing_Low"] = current_sl
        df["PA_Is_Swing_High"] = is_swing_high
        df["PA_Is_Swing_Low"] = is_swing_low
        
        # Distance from current swing levels
        c = df["Close"]
        df["PA_Dist_Swing_High"] = (current_sh - c) / (c + 1e-10) * 100
        df["PA_Dist_Swing_Low"] = (c - current_sl) / (c + 1e-10) * 100
        df["PA_Range_Position"] = (c - current_sl) / (current_sh - current_sl + 1e-10)
        
        return df

    def _swing_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify HH, HL, LH, LL pattern."""
        sh = df["PA_Swing_High"].values
        sl = df["PA_Swing_Low"].values
        n = len(df)
        
        # Track swing sequences
        hh = np.zeros(n, dtype=np.int32)  # Higher High
        hl = np.zeros(n, dtype=np.int32)  # Higher Low
        lh = np.zeros(n, dtype=np.int32)  # Lower High
        ll = np.zeros(n, dtype=np.int32)  # Lower Low
        
        prev_sh = 0.0
        prev_sl = 0.0
        
        for i in range(1, n):
            if sh[i] != sh[i-1] and sh[i] > 0:  # New swing high
                if prev_sh > 0:
                    if sh[i] > prev_sh:
                        hh[i] = 1
                    elif sh[i] < prev_sh:
                        lh[i] = 1
                prev_sh = sh[i]
            if sl[i] != sl[i-1] and sl[i] > 0:  # New swing low
                if prev_sl > 0:
                    if sl[i] > prev_sl:
                        hl[i] = 1
                    elif sl[i] < prev_sl:
                        ll[i] = 1
                prev_sl = sl[i]

        df["PA_HH"] = hh
        df["PA_HL"] = hl
        df["PA_LH"] = lh
        df["PA_LL"] = ll
        
        # Cumulative structure score: +1 for bullish structure, -1 for bearish
        bullish = (hh + hl)
        bearish = (lh + ll)
        df["PA_Structure_Score"] = pd.Series(bullish - bearish, index=df.index).rolling(20, min_periods=1).sum().fillna(0)
        
        return df

    def _bos_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Break of Structure (BOS) and Change of Character (CHoCH).
        BOS = price breaks a swing level in the direction of the trend (continuation)
        CHoCH = price breaks a swing level against the trend (reversal)
        """
        c = df["Close"].values
        sh = df["PA_Swing_High"].values
        sl = df["PA_Swing_Low"].values
        structure = df["PA_Structure_Score"].values
        n = len(c)
        
        bos_bull = np.zeros(n, dtype=np.int32)
        bos_bear = np.zeros(n, dtype=np.int32)
        choch_bull = np.zeros(n, dtype=np.int32)
        choch_bear = np.zeros(n, dtype=np.int32)

        for i in range(1, n):
            # Bullish break above swing high
            if c[i] > sh[i-1] and sh[i-1] > 0:
                if structure[i] >= 0:
                    bos_bull[i] = 1   # Continuation (BOS)
                else:
                    choch_bull[i] = 1  # Reversal (CHoCH)
            # Bearish break below swing low
            if c[i] < sl[i-1] and sl[i-1] > 0:
                if structure[i] <= 0:
                    bos_bear[i] = 1
                else:
                    choch_bear[i] = 1

        df["PA_BOS_Bull"] = bos_bull
        df["PA_BOS_Bear"] = bos_bear
        df["PA_CHoCH_Bull"] = choch_bull
        df["PA_CHoCH_Bear"] = choch_bear
        
        # Rolling BOS/CHoCH signals
        df["PA_BOS_Signal"] = pd.Series(bos_bull - bos_bear, index=df.index).rolling(10, min_periods=1).sum().fillna(0)
        df["PA_CHoCH_Signal"] = pd.Series(choch_bull - choch_bear, index=df.index).rolling(10, min_periods=1).sum().fillna(0)
        
        return df

    def _impulse_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect impulse vs corrective moves based on candle body/range ratio."""
        body = (df["Close"] - df["Open"]).abs()
        total_range = df["High"] - df["Low"] + 1e-10
        
        # Impulse: large body relative to range (>70%)
        df["PA_Body_Ratio"] = body / total_range
        df["PA_Is_Impulse"] = (df["PA_Body_Ratio"] > 0.7).astype(int)
        df["PA_Is_Corrective"] = (df["PA_Body_Ratio"] < 0.3).astype(int)
        
        # Impulse strength (rolling)
        df["PA_Impulse_Strength"] = df["PA_Body_Ratio"].rolling(5).mean()
        
        # Consecutive impulse candles
        direction = np.sign(df["Close"] - df["Open"])
        df["PA_Impulse_Direction"] = direction * df["PA_Is_Impulse"]
        
        return df

    def _equal_levels(self, df: pd.DataFrame, tolerance: float = 0.001) -> pd.DataFrame:
        """Detect equal highs and equal lows (liquidity pools)."""
        h = df["High"].values
        l = df["Low"].values
        n = len(h)
        
        eq_high = np.zeros(n, dtype=np.int32)
        eq_low = np.zeros(n, dtype=np.int32)
        
        lookback = 20
        for i in range(lookback, n):
            for j in range(i - lookback, i):
                if abs(h[i] - h[j]) / (h[j] + 1e-10) < tolerance:
                    eq_high[i] = 1
                    break
            for j in range(i - lookback, i):
                if abs(l[i] - l[j]) / (l[j] + 1e-10) < tolerance:
                    eq_low[i] = 1
                    break

        df["PA_Equal_High"] = eq_high
        df["PA_Equal_Low"] = eq_low
        
        return df

    def _displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect displacement candles (large, aggressive moves indicating institutional activity)."""
        body = (df["Close"] - df["Open"]).abs()
        avg_body = body.rolling(20).mean()
        
        # Displacement = body > 2x average
        df["PA_Displacement"] = (body > 2.0 * avg_body).astype(int)
        df["PA_Displacement_Bull"] = ((df["Close"] > df["Open"]) & (body > 2.0 * avg_body)).astype(int)
        df["PA_Displacement_Bear"] = ((df["Close"] < df["Open"]) & (body > 2.0 * avg_body)).astype(int)
        
        return df

    def _market_structure_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite market structure assessment."""
        score = (
            df.get("PA_Structure_Score", 0) * 0.3 +
            df.get("PA_BOS_Signal", 0) * 0.3 +
            df.get("PA_CHoCH_Signal", 0) * 0.2 +
            df.get("PA_Impulse_Direction", 0).rolling(10, min_periods=1).sum().fillna(0) * 0.2
        )
        df["PA_Market_Structure"] = score
        # Classify: >1 bullish, <-1 bearish, else ranging
        df["PA_Structure_Class"] = np.where(score > 1, 1, np.where(score < -1, -1, 0))
        return df
