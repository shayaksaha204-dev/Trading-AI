"""
ICT (Inner Circle Trader) Concepts — Advanced Institutional Features
======================================================================
Kill Zones, Optimal Trade Entry, Silver Bullet, Judas Swing,
Power of 3, Asian Range, London/NY sessions.
"""

import numpy as np
import pandas as pd
from loguru import logger


class ICTConcepts:
    """
    ICT methodology features optimized for intraday/scalping.
    
    Detects:
        - Kill Zones (London/NY open power hours)
        - Optimal Trade Entry (OTE) — Fibonacci retracement into OB
        - Fair Value Gaps (ICT-specific multi-timeframe)
        - Judas Swing (fake move before real move)
        - Silver Bullet windows (10:00-11:00, 14:00-15:00 EST)
        - Power of 3 (Accumulation → Manipulation → Distribution)
        - Asian Range as reference
        - Session-based analysis
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._session_analysis(df)
        df = self._kill_zones(df)
        df = self._ote_levels(df)
        df = self._judas_swing(df)
        df = self._power_of_3(df)
        df = self._candle_anatomy(df)
        df = self._time_features(df)
        df = self._ict_confluence(df)
        return df

    def _session_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze price behavior relative to daily sessions.
        Uses daily open/high/low/close for session context.
        """
        c = df["Close"]
        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        
        # Previous day's range as context
        df["ICT_Prev_Range"] = (h.shift(1) - l.shift(1)) / (c.shift(1) + 1e-10) * 100
        df["ICT_Prev_Body"] = (c.shift(1) - o.shift(1)) / (c.shift(1) + 1e-10) * 100
        
        # Where in previous day's range did we close?
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        df["ICT_Prev_Close_Position"] = (c.shift(1) - prev_l) / (prev_h - prev_l + 1e-10)
        
        # Gap from previous close
        df["ICT_Gap"] = (o - c.shift(1)) / (c.shift(1) + 1e-10) * 100
        df["ICT_Gap_Filled"] = np.where(
            df["ICT_Gap"] > 0,
            (l <= c.shift(1)).astype(int),
            np.where(df["ICT_Gap"] < 0, (h >= c.shift(1)).astype(int), 0)
        )
        
        # Asian session range proxy (first few candles of the day)
        # For daily data, we use rolling as approximation
        df["ICT_Asian_Range"] = (h.rolling(5).max() - l.rolling(5).min()) / (c + 1e-10) * 100
        
        return df

    def _kill_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kill Zone analysis — periods of highest institutional activity.
        For daily data: compute volatility patterns that match kill zone behavior.
        """
        c = df["Close"]
        ret = c.pct_change()
        
        # Volatility expansion (simulates kill zone entry)
        short_vol = ret.abs().rolling(3).mean()
        long_vol = ret.abs().rolling(20).mean()
        
        df["ICT_Vol_Expansion"] = short_vol / (long_vol + 1e-10)
        df["ICT_In_Kill_Zone"] = (df["ICT_Vol_Expansion"] > 1.5).astype(int)
        
        # Kill zone direction bias
        df["ICT_KZ_Direction"] = np.where(
            df["ICT_In_Kill_Zone"] == 1,
            np.sign(ret.rolling(3).sum()),
            0
        )
        
        # Expansion after contraction (squeeze → release pattern)
        vol_ratio = short_vol / (long_vol + 1e-10)
        df["ICT_Squeeze"] = (vol_ratio.shift(3) < 0.7).astype(int) * (vol_ratio > 1.3).astype(int)
        
        return df

    def _ote_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimal Trade Entry — Fibonacci levels from recent swing.
        OTE zone = 0.618 to 0.786 retracement of the impulse move.
        """
        h = df["High"]
        l = df["Low"]
        c = df["Close"]
        
        for lookback in [10, 20]:
            swing_h = h.rolling(lookback).max()
            swing_l = l.rolling(lookback).min()
            rng = swing_h - swing_l + 1e-10
            
            # Fibonacci levels
            fib_618 = swing_h - 0.618 * rng
            fib_786 = swing_h - 0.786 * rng
            fib_50 = swing_h - 0.5 * rng
            fib_382 = swing_h - 0.382 * rng
            
            # In OTE zone?
            df[f"ICT_In_OTE_{lookback}"] = ((c >= fib_786) & (c <= fib_618)).astype(int)
            df[f"ICT_Fib_Level_{lookback}"] = (swing_h - c) / rng  # Current fib level
            
            # Distance to key fibs
            df[f"ICT_Dist_618_{lookback}"] = (c - fib_618) / (c + 1e-10) * 100
            df[f"ICT_Dist_50_{lookback}"] = (c - fib_50) / (c + 1e-10) * 100

        return df

    def _judas_swing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Judas Swing — fake breakout that traps traders before reversing.
        Pattern: price takes out a level, then reverses strongly.
        """
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        o = df["Open"].values
        n = len(c)
        
        judas_bull = np.zeros(n, dtype=np.int32)
        judas_bear = np.zeros(n, dtype=np.int32)
        
        lookback = 10
        for i in range(lookback + 1, n):
            prev_low = np.min(l[i-lookback:i])
            prev_high = np.max(h[i-lookback:i])
            
            # Bearish Judas: sweeps above resistance then closes below open
            if h[i] > prev_high and c[i] < o[i] and c[i] < prev_high:
                judas_bear[i] = 1
            
            # Bullish Judas: sweeps below support then closes above open
            if l[i] < prev_low and c[i] > o[i] and c[i] > prev_low:
                judas_bull[i] = 1
        
        df["ICT_Judas_Bull"] = judas_bull
        df["ICT_Judas_Bear"] = judas_bear
        df["ICT_Judas_Signal"] = judas_bull - judas_bear
        
        return df

    def _power_of_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Power of 3: Accumulation → Manipulation → Distribution.
        Detected via volume patterns and price structure.
        """
        c = df["Close"]
        ret = c.pct_change()
        vol = ret.abs()
        
        # Phase detection using volatility profile
        vol_3 = vol.rolling(3).mean()
        vol_10 = vol.rolling(10).mean()
        vol_20 = vol.rolling(20).mean()
        
        # Accumulation: low volatility, range-bound
        df["ICT_Accumulation"] = ((vol_3 < vol_20 * 0.6) & (vol_3 < vol_10 * 0.7)).astype(int)
        
        # Manipulation: sudden spike (stop hunt / fake move)
        df["ICT_Manipulation"] = ((vol_3 > vol_20 * 2.0) & (ret.rolling(3).sum().abs() < vol_3 * 0.5)).astype(int)
        
        # Distribution: strong directional move with high volatility
        df["ICT_Distribution"] = ((vol_3 > vol_20 * 1.5) & (ret.rolling(3).sum().abs() > vol_3 * 0.7)).astype(int)
        
        # Phase score
        df["ICT_PO3_Phase"] = np.where(
            df["ICT_Distribution"] == 1, 3,
            np.where(df["ICT_Manipulation"] == 1, 2,
                     np.where(df["ICT_Accumulation"] == 1, 1, 0))
        )
        
        return df

    def _candle_anatomy(self, df: pd.DataFrame) -> pd.DataFrame:
        """ICT-style candle anatomy analysis."""
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        body = (c - o).abs()
        total = h - l + 1e-10
        
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        
        # Wick ratios (rejection signals)
        df["ICT_Upper_Wick_Ratio"] = upper_wick / total
        df["ICT_Lower_Wick_Ratio"] = lower_wick / total
        df["ICT_Body_Pct"] = body / total
        
        # Rejection candles (long wick + small body)
        df["ICT_Bull_Rejection"] = ((lower_wick > 2 * body) & (upper_wick < body)).astype(int)
        df["ICT_Bear_Rejection"] = ((upper_wick > 2 * body) & (lower_wick < body)).astype(int)
        
        # Marubozu (full body, no wicks = strong conviction)
        df["ICT_Marubozu"] = ((body / total > 0.9)).astype(int)
        
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features for session/cycle analysis."""
        if hasattr(df.index, 'dayofweek'):
            df["ICT_Day_of_Week"] = df.index.dayofweek
            df["ICT_Month"] = df.index.month
            df["ICT_Quarter"] = df.index.quarter
            
            # Day-of-week return tendency
            ret = df["Close"].pct_change()
            for d in range(5):
                mask = df["ICT_Day_of_Week"] == d
                df.loc[mask, f"ICT_DOW_{d}_Bias"] = ret[mask].rolling(52, min_periods=10).mean()
            
            # Fill NaN in DOW columns
            for d in range(5):
                col = f"ICT_DOW_{d}_Bias"
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        else:
            df["ICT_Day_of_Week"] = 0
            df["ICT_Month"] = 0
            df["ICT_Quarter"] = 0
        
        return df

    def _ict_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite ICT score."""
        bull_score = (
            df.get("ICT_Judas_Bull", 0) * 3 +
            df.get("ICT_Bull_Rejection", 0) * 2 +
            df.get("ICT_In_OTE_20", 0) * 2 +
            df.get("ICT_Squeeze", 0) * 1 +
            (df.get("ICT_PO3_Phase", 0) == 3).astype(int) * (df["Close"].pct_change() > 0).astype(int) * 2
        )
        bear_score = (
            df.get("ICT_Judas_Bear", 0) * 3 +
            df.get("ICT_Bear_Rejection", 0) * 2 +
            (df.get("ICT_PO3_Phase", 0) == 3).astype(int) * (df["Close"].pct_change() < 0).astype(int) * 2
        )
        df["ICT_Score"] = bull_score - bear_score
        df["ICT_Score_Smooth"] = pd.Series(df["ICT_Score"]).rolling(5).mean().fillna(0)
        return df
