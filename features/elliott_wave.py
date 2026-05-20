"""
Elliott Wave — Automated Wave Structure Detection
====================================================
Impulse/corrective wave identification, wave counting,
Fibonacci wave extensions.
"""

import numpy as np
import pandas as pd
from loguru import logger


class ElliottWave:
    """
    Elliott Wave feature engine.
    
    Detects:
        - Impulse waves (5-wave trending moves)
        - Corrective waves (3-wave counter-trend moves)
        - Wave position estimation (which wave are we in?)
        - Fibonacci extensions for wave targets
        - Wave momentum and exhaustion signals
    """

    def compute_all(self, df: pd.DataFrame, swing_lookback: int = 5) -> pd.DataFrame:
        df = df.copy()
        df = self._wave_swings(df, swing_lookback)
        df = self._wave_counting(df)
        df = self._wave_fibonacci(df)
        df = self._wave_momentum(df)
        df = self._wave_score(df)
        return df

    def _wave_swings(self, df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """Identify alternating swing points for wave structure."""
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        n = len(df)
        
        # Detect swings
        pivots = np.zeros(n)  # +1 = swing high, -1 = swing low
        pivot_prices = np.zeros(n)
        
        for i in range(lookback, n - lookback):
            if h[i] == max(h[i-lookback:i+lookback+1]):
                pivots[i] = 1
                pivot_prices[i] = h[i]
            if l[i] == min(l[i-lookback:i+lookback+1]):
                if pivots[i] == 0:  # Don't overwrite
                    pivots[i] = -1
                    pivot_prices[i] = l[i]
        
        # Build alternating pivot sequence
        last_pivot_type = 0
        last_pivot_price = c[0]
        wave_amplitude = np.zeros(n)
        wave_length = np.zeros(n)
        pivot_count = 0
        last_pivot_idx = 0
        
        for i in range(n):
            if pivots[i] != 0 and pivots[i] != last_pivot_type:
                amplitude = abs(pivot_prices[i] - last_pivot_price)
                wave_amplitude[i] = amplitude / (c[i] + 1e-10) * 100
                wave_length[i] = i - last_pivot_idx
                last_pivot_type = pivots[i]
                last_pivot_price = pivot_prices[i]
                last_pivot_idx = i
                pivot_count += 1
        
        df["EW_Pivot"] = pivots
        df["EW_Wave_Amplitude"] = pd.Series(wave_amplitude, index=df.index).replace(0, np.nan).ffill().fillna(0)
        df["EW_Wave_Length"] = pd.Series(wave_length, index=df.index).replace(0, np.nan).ffill().fillna(0)
        
        return df

    def _wave_counting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate wave count position.
        Counts alternating swings and maps to 5-wave impulse or 3-wave correction.
        """
        pivots = df["EW_Pivot"].values
        c = df["Close"].values
        n = len(c)
        
        wave_count = np.zeros(n)
        wave_type = np.zeros(n)  # 1 = impulse, -1 = corrective
        
        # Track last 7 swing points (enough for 5+2 waves)
        swing_buffer = []
        
        for i in range(n):
            if pivots[i] != 0:
                swing_buffer.append((i, pivots[i], c[i]))
                if len(swing_buffer) > 8:
                    swing_buffer = swing_buffer[-8:]
            
            if len(swing_buffer) >= 5:
                # Check if last 5 swings form impulse pattern
                prices = [s[2] for s in swing_buffer[-5:]]
                types = [s[1] for s in swing_buffer[-5:]]
                
                # Bullish impulse: wave 3 > wave 1, wave 5 > wave 3
                if types[0] == -1:  # Starts from low
                    w1 = prices[1] - prices[0]
                    w2 = prices[1] - prices[2]
                    w3 = prices[3] - prices[2]
                    w4 = prices[3] - prices[4] if len(prices) > 4 else 0
                    
                    if w3 > w1 and w3 > 0 and w2 > 0 and w2 < w1:
                        wave_type[i] = 1
                        # Estimate position in wave
                        wave_count[i] = len(swing_buffer) % 5 + 1
                
                # Bearish impulse
                elif types[0] == 1:  # Starts from high
                    w1 = prices[0] - prices[1]
                    w2 = prices[2] - prices[1]
                    w3 = prices[2] - prices[3]
                    
                    if w3 > w1 and w3 > 0 and w2 > 0 and w2 < w1:
                        wave_type[i] = -1
                        wave_count[i] = len(swing_buffer) % 5 + 1

        df["EW_Wave_Count"] = pd.Series(wave_count, index=df.index).replace(0, np.nan).ffill().bfill().fillna(0)
        df["EW_Wave_Type"] = pd.Series(wave_type, index=df.index).replace(0, np.nan).ffill().bfill().fillna(0)
        
        # Is likely wave 3? (strongest wave — best for trading)
        df["EW_In_Wave3"] = (df["EW_Wave_Count"] == 3).astype(int)
        # Is likely wave 5? (exhaustion wave — prepare for reversal)
        df["EW_In_Wave5"] = (df["EW_Wave_Count"] == 5).astype(int)
        
        return df

    def _wave_fibonacci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fibonacci extensions for wave targets."""
        h = df["High"]
        l = df["Low"]
        c = df["Close"]
        
        for w in [20, 50]:
            swing_h = h.rolling(w).max()
            swing_l = l.rolling(w).min()
            rng = swing_h - swing_l + 1e-10
            
            # Common Elliott Wave Fibonacci targets
            df[f"EW_Ext_1618_{w}"] = swing_l + rng * 1.618  # Wave 3 target
            df[f"EW_Ext_2618_{w}"] = swing_l + rng * 2.618  # Extended wave
            df[f"EW_Ret_382_{w}"] = swing_h - rng * 0.382   # Wave 4 retrace
            df[f"EW_Ret_500_{w}"] = swing_h - rng * 0.500   # Deep retrace
            
            # Distance to extension levels
            df[f"EW_Dist_Ext_{w}"] = (df[f"EW_Ext_1618_{w}"] - c) / (c + 1e-10) * 100
        
        return df

    def _wave_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wave momentum — detect wave exhaustion and divergence."""
        c = df["Close"]
        ret = c.pct_change()
        
        # Momentum per wave (declining momentum = exhaustion)
        mom_5 = ret.rolling(5).sum()
        mom_10 = ret.rolling(10).sum()
        mom_20 = ret.rolling(20).sum()
        
        # Momentum divergence (price making new high but momentum declining)
        price_hh = (c > c.rolling(20).max().shift(1)).astype(int)
        mom_lh = (mom_10 < mom_10.rolling(20).max().shift(1)).astype(int)
        
        df["EW_Bearish_Divergence"] = (price_hh & mom_lh).astype(int)
        
        price_ll = (c < c.rolling(20).min().shift(1)).astype(int)
        mom_hl = (mom_10 > mom_10.rolling(20).min().shift(1)).astype(int)
        
        df["EW_Bullish_Divergence"] = (price_ll & mom_hl).astype(int)
        
        # Wave exhaustion signal
        df["EW_Exhaustion"] = df["EW_Bearish_Divergence"] + df["EW_Bullish_Divergence"]
        df["EW_Momentum_Ratio"] = mom_5 / (mom_20 + 1e-10)
        
        return df

    def _wave_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite Elliott Wave score."""
        bull = (
            df.get("EW_In_Wave3", 0) * (df.get("EW_Wave_Type", 0) > 0).astype(int) * 3 +
            df.get("EW_Bullish_Divergence", 0) * 2 +
            (df.get("EW_Momentum_Ratio", 0) > 1).astype(int)
        )
        bear = (
            df.get("EW_In_Wave3", 0) * (df.get("EW_Wave_Type", 0) < 0).astype(int) * 3 +
            df.get("EW_Bearish_Divergence", 0) * 2 +
            df.get("EW_In_Wave5", 0) * 2 +
            (df.get("EW_Momentum_Ratio", 0) < -1).astype(int)
        )
        df["EW_Score"] = bull - bear
        df["EW_Score_Smooth"] = pd.Series(df["EW_Score"]).rolling(5).mean().fillna(0)
        return df
