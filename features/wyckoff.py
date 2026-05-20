"""
Wyckoff Method — Phase Detection & Volume Analysis
=====================================================
Accumulation, Distribution, Markup, Markdown phases.
Springs, Upthrusts, Composite Man theory.
"""

import numpy as np
import pandas as pd
from loguru import logger


class WyckoffAnalysis:
    """
    Wyckoff method feature engine.
    
    Detects:
        - Market Phases (Accumulation → Markup → Distribution → Markdown)
        - Spring / Upthrust (false breakout traps)
        - Effort vs Result (volume-price divergence)
        - Selling/Buying Climax
        - Tests of supply/demand
        - Composite Man activity
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        has_volume = "Volume" in df.columns and (df["Volume"] > 0).any()
        df = self._phases(df, has_volume)
        df = self._spring_upthrust(df)
        if has_volume:
            df = self._effort_result(df)
            df = self._climax(df)
            df = self._volume_spread(df)
        df = self._tests(df)
        df = self._wyckoff_score(df)
        return df

    def _phases(self, df: pd.DataFrame, has_volume: bool) -> pd.DataFrame:
        """
        Detect Wyckoff market phases based on price range and trend.
        Phase 1: Accumulation (range after downtrend)
        Phase 2: Markup (uptrend breakout)
        Phase 3: Distribution (range after uptrend)
        Phase 4: Markdown (downtrend breakout)
        """
        c = df["Close"]
        ret = c.pct_change()
        
        # Trend detection
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        trend = np.sign(sma20 - sma50)
        
        # Range detection (low volatility = ranging)
        vol_10 = ret.abs().rolling(10).mean()
        vol_50 = ret.abs().rolling(50).mean()
        ranging = (vol_10 < vol_50 * 0.7).astype(int)
        
        # Phase classification
        phase = np.zeros(len(df))
        for i in range(50, len(df)):
            # Recent trend direction
            recent_trend = trend.iloc[i] if not np.isnan(trend.iloc[i]) else 0
            prev_trend = trend.iloc[max(i-20, 0)] if not np.isnan(trend.iloc[max(i-20, 0)]) else 0
            is_ranging = ranging.iloc[i]
            
            if is_ranging and prev_trend < 0:
                phase[i] = 1  # Accumulation
            elif not is_ranging and recent_trend > 0:
                phase[i] = 2  # Markup
            elif is_ranging and prev_trend > 0:
                phase[i] = 3  # Distribution
            elif not is_ranging and recent_trend < 0:
                phase[i] = 4  # Markdown

        df["WY_Phase"] = phase
        # One-hot encode phases
        for p in range(1, 5):
            df[f"WY_Phase_{p}"] = (phase == p).astype(int)
        
        df["WY_Ranging"] = ranging
        df["WY_Trend"] = trend
        
        return df

    def _spring_upthrust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Spring: price dips below support briefly then reverses up (bullish trap of shorts).
        Upthrust: price pokes above resistance briefly then reverses down (bearish trap of longs).
        """
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        o = df["Open"].values
        n = len(c)
        
        spring = np.zeros(n, dtype=np.int32)
        upthrust = np.zeros(n, dtype=np.int32)
        
        lookback = 20
        for i in range(lookback + 1, n):
            support = np.min(l[i-lookback:i])
            resistance = np.max(h[i-lookback:i])
            
            # Spring: low breaks support but close is above it + bullish close
            if l[i] < support and c[i] > support and c[i] > o[i]:
                spring[i] = 1
            
            # Upthrust: high breaks resistance but close is below it + bearish close
            if h[i] > resistance and c[i] < resistance and c[i] < o[i]:
                upthrust[i] = 1

        df["WY_Spring"] = spring
        df["WY_Upthrust"] = upthrust
        df["WY_Trap_Signal"] = spring - upthrust
        
        return df

    def _effort_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Effort vs Result: compare volume (effort) with price movement (result).
        High effort + low result = absorption (smart money absorbing supply/demand).
        Low effort + high result = lack of opposition (easy trend move).
        """
        v = df["Volume"]
        ret = df["Close"].pct_change().abs()
        
        # Normalize
        v_norm = v / (v.rolling(20).mean() + 1e-10)
        r_norm = ret / (ret.rolling(20).mean() + 1e-10)
        
        # Effort-Result ratio
        df["WY_Effort_Result"] = v_norm / (r_norm + 1e-10)
        
        # Divergences
        df["WY_Absorption"] = ((v_norm > 1.5) & (r_norm < 0.7)).astype(int)  # High effort, low result
        df["WY_Easy_Move"] = ((v_norm < 0.7) & (r_norm > 1.5)).astype(int)   # Low effort, high result
        
        # Volume increasing with price movement = healthy trend
        df["WY_Volume_Trend_Agree"] = np.sign(df["Close"].pct_change()) * np.sign(v.diff())
        
        return df

    def _climax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selling Climax (SC): extreme volume + wide range + closes off lows = bottom near.
        Buying Climax (BC): extreme volume + wide range + closes off highs = top near.
        """
        v = df["Volume"]
        rng = df["High"] - df["Low"]
        body = (df["Close"] - df["Open"])
        
        v_z = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-10)
        r_z = (rng - rng.rolling(50).mean()) / (rng.rolling(50).std() + 1e-10)
        
        # Selling Climax: huge volume + wide range + bullish close (reversal off lows)
        df["WY_Selling_Climax"] = ((v_z > 2) & (r_z > 1.5) & (body > 0) & 
                                    ((df["Close"] - df["Low"]) / (rng + 1e-10) > 0.6)).astype(int)
        
        # Buying Climax: huge volume + wide range + bearish close (reversal off highs)
        df["WY_Buying_Climax"] = ((v_z > 2) & (r_z > 1.5) & (body < 0) &
                                   ((df["High"] - df["Close"]) / (rng + 1e-10) > 0.6)).astype(int)
        
        df["WY_Climax_Signal"] = df["WY_Selling_Climax"] - df["WY_Buying_Climax"]
        
        return df

    def _volume_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Spread Analysis (VSA) — Wyckoff-inspired."""
        v = df["Volume"]
        c = df["Close"]
        rng = df["High"] - df["Low"]
        body = c - df["Open"]
        
        v_avg = v.rolling(20).mean()
        r_avg = rng.rolling(20).mean()
        
        # No Supply bar: narrow range, low volume, bullish close
        df["WY_No_Supply"] = ((rng < r_avg * 0.5) & (v < v_avg * 0.5) & (body > 0)).astype(int)
        
        # No Demand bar: narrow range, low volume, bearish close
        df["WY_No_Demand"] = ((rng < r_avg * 0.5) & (v < v_avg * 0.5) & (body < 0)).astype(int)
        
        # Stopping Volume: high volume but price stops going down
        ret = c.pct_change()
        df["WY_Stopping_Vol"] = ((v > v_avg * 2) & (ret.abs() < ret.abs().rolling(20).mean() * 0.5)).astype(int)
        
        return df

    def _tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Test of supply/demand — price revisits a level on lower volume."""
        c = df["Close"]
        l = df["Low"]
        h = df["High"]
        
        # Distance from recent lows/highs (testing levels)
        for w in [10, 20]:
            recent_low = l.rolling(w).min()
            recent_high = h.rolling(w).max()
            
            df[f"WY_Test_Low_{w}"] = ((l - recent_low).abs() / (c + 1e-10) < 0.005).astype(int)
            df[f"WY_Test_High_{w}"] = ((h - recent_high).abs() / (c + 1e-10) < 0.005).astype(int)
        
        return df

    def _wyckoff_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite Wyckoff bullish/bearish score."""
        bull = (
            df.get("WY_Spring", 0) * 3 +
            df.get("WY_Phase_1", 0) * 1 +  # Accumulation
            df.get("WY_Phase_2", 0) * 2 +  # Markup
            df.get("WY_Selling_Climax", 0) * 2 +
            df.get("WY_No_Supply", 0) * 1
        )
        bear = (
            df.get("WY_Upthrust", 0) * 3 +
            df.get("WY_Phase_3", 0) * 1 +  # Distribution
            df.get("WY_Phase_4", 0) * 2 +  # Markdown
            df.get("WY_Buying_Climax", 0) * 2 +
            df.get("WY_No_Demand", 0) * 1
        )
        df["WY_Score"] = bull - bear
        df["WY_Score_Smooth"] = pd.Series(df["WY_Score"]).rolling(5).mean().fillna(0)
        return df
