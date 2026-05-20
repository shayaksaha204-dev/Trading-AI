"""
Smart Money Concepts (SMC) — Institutional Trading Features
==============================================================
Order Blocks, Fair Value Gaps, Breaker Blocks, Liquidity Zones.
"""

import numpy as np
import pandas as pd
from loguru import logger


class SmartMoneyConcepts:
    """
    SMC feature engine detecting institutional footprints:
    
    - Order Blocks (OB): last opposing candle before an impulse
    - Fair Value Gaps (FVG): imbalance zones (3-candle gaps)
    - Breaker Blocks: failed order blocks turned into S/R
    - Mitigation Blocks: partially-filled order blocks
    - Liquidity Zones: clusters of equal highs/lows
    - Premium/Discount zones
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._order_blocks(df)
        df = self._fair_value_gaps(df)
        df = self._breaker_blocks(df)
        df = self._liquidity_zones(df)
        df = self._premium_discount(df)
        df = self._institutional_candles(df)
        df = self._smc_confluence(df)
        return df

    def _order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks — the last opposing candle before a strong impulse move.
        Bullish OB: last bearish candle before a bullish impulse
        Bearish OB: last bullish candle before a bearish impulse
        """
        o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
        n = len(df)
        
        bull_ob = np.zeros(n)       # Bullish order block level
        bear_ob = np.zeros(n)       # Bearish order block level
        in_bull_ob = np.zeros(n, dtype=np.int32)
        in_bear_ob = np.zeros(n, dtype=np.int32)
        ob_strength = np.zeros(n)
        
        body = np.abs(c - o)
        avg_body = pd.Series(body).rolling(20).mean().values
        
        for i in range(3, n):
            # Check for strong bullish impulse (body > 2x average)
            if c[i] > o[i] and body[i] > 2.0 * (avg_body[i] if not np.isnan(avg_body[i]) else body[i]):
                # Find last bearish candle before this impulse
                for j in range(i - 1, max(i - 10, 0), -1):
                    if c[j] < o[j]:  # Bearish candle = bullish order block
                        bull_ob[i] = l[j]  # OB zone = low of the bearish candle
                        ob_strength[i] = body[i] / (avg_body[i] + 1e-10)
                        break
            
            # Check for strong bearish impulse
            if c[i] < o[i] and body[i] > 2.0 * (avg_body[i] if not np.isnan(avg_body[i]) else body[i]):
                for j in range(i - 1, max(i - 10, 0), -1):
                    if c[j] > o[j]:  # Bullish candle = bearish order block
                        bear_ob[i] = h[j]
                        ob_strength[i] = body[i] / (avg_body[i] + 1e-10)
                        break

        # Forward-fill active OB zones
        last_bull_ob = 0.0
        last_bear_ob = 0.0
        for i in range(n):
            if bull_ob[i] > 0:
                last_bull_ob = bull_ob[i]
            if bear_ob[i] > 0:
                last_bear_ob = bear_ob[i]
            
            # Check if price is in OB zone
            if last_bull_ob > 0 and l[i] <= last_bull_ob * 1.002:
                in_bull_ob[i] = 1
            if last_bear_ob > 0 and h[i] >= last_bear_ob * 0.998:
                in_bear_ob[i] = 1
            
            # Invalidate OB if price breaks through
            if last_bull_ob > 0 and c[i] < last_bull_ob * 0.99:
                last_bull_ob = 0
            if last_bear_ob > 0 and c[i] > last_bear_ob * 1.01:
                last_bear_ob = 0

        df["SMC_Bull_OB"] = in_bull_ob
        df["SMC_Bear_OB"] = in_bear_ob
        df["SMC_OB_Strength"] = pd.Series(ob_strength, index=df.index).rolling(10, min_periods=1).mean().fillna(0)
        df["SMC_Dist_Bull_OB"] = np.where(bull_ob > 0, (c - bull_ob) / (c + 1e-10) * 100, 0)
        df["SMC_Dist_Bear_OB"] = np.where(bear_ob > 0, (bear_ob - c) / (c + 1e-10) * 100, 0)
        
        return df

    def _fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fair Value Gaps (FVG) — imbalance zones where price moved too fast.
        Bullish FVG: candle[i-2].high < candle[i].low (gap up)
        Bearish FVG: candle[i-2].low > candle[i].high (gap down)
        """
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        n = len(df)
        
        bull_fvg = np.zeros(n, dtype=np.int32)
        bear_fvg = np.zeros(n, dtype=np.int32)
        fvg_size = np.zeros(n)
        in_fvg = np.zeros(n, dtype=np.int32)
        
        # Track active FVG zones
        active_bull_fvgs = []  # (top, bottom) of gap
        active_bear_fvgs = []

        for i in range(2, n):
            # Bullish FVG: gap between candle[i-2] high and candle[i] low
            if l[i] > h[i - 2]:
                bull_fvg[i] = 1
                gap = l[i] - h[i - 2]
                fvg_size[i] = gap / (c[i] + 1e-10) * 100
                active_bull_fvgs.append((l[i], h[i - 2]))
            
            # Bearish FVG: gap between candle[i-2] low and candle[i] high
            if h[i] < l[i - 2]:
                bear_fvg[i] = 1
                gap = l[i - 2] - h[i]
                fvg_size[i] = gap / (c[i] + 1e-10) * 100
                active_bear_fvgs.append((l[i - 2], h[i]))

            # Check if price is inside any active FVG
            new_bull = []
            for top, bottom in active_bull_fvgs:
                if bottom <= c[i] <= top:
                    in_fvg[i] = 1
                if c[i] > bottom * 0.995:  # FVG not yet fully filled
                    new_bull.append((top, bottom))
            active_bull_fvgs = new_bull[-20:]  # Keep last 20

            new_bear = []
            for top, bottom in active_bear_fvgs:
                if bottom <= c[i] <= top:
                    in_fvg[i] = -1
                if c[i] < top * 1.005:
                    new_bear.append((top, bottom))
            active_bear_fvgs = new_bear[-20:]

        df["SMC_Bull_FVG"] = bull_fvg
        df["SMC_Bear_FVG"] = bear_fvg
        df["SMC_FVG_Size"] = fvg_size
        df["SMC_In_FVG"] = in_fvg
        df["SMC_FVG_Count_Bull"] = pd.Series(bull_fvg, index=df.index).rolling(20, min_periods=1).sum().fillna(0)
        df["SMC_FVG_Count_Bear"] = pd.Series(bear_fvg, index=df.index).rolling(20, min_periods=1).sum().fillna(0)
        
        return df

    def _breaker_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Breaker Block — a failed order block that now acts as opposition.
        When an OB fails (price breaks through), the zone flips to become a breaker.
        """
        c = df["Close"].values
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        n = len(c)
        
        body = np.abs(c - o)
        avg_body = pd.Series(body).rolling(20).mean().values
        
        breaker_bull = np.zeros(n, dtype=np.int32)
        breaker_bear = np.zeros(n, dtype=np.int32)
        
        # Simplified: detect when price breaks a recent structure level
        # then returns to test it from the other side
        recent_highs = pd.Series(h).rolling(10).max().values
        recent_lows = pd.Series(l).rolling(10).min().values
        
        for i in range(20, n):
            # Price broke below support, then came back up to test = bearish breaker
            if c[i-1] < recent_lows[i-10] and c[i] > c[i-1]:
                breaker_bear[i] = 1
            # Price broke above resistance, then pulled back = bullish breaker
            if c[i-1] > recent_highs[i-10] and c[i] < c[i-1]:
                breaker_bull[i] = 1

        df["SMC_Breaker_Bull"] = breaker_bull
        df["SMC_Breaker_Bear"] = breaker_bear
        
        return df

    def _liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect buy-side and sell-side liquidity.
        - Buy-side liquidity: clusters of swing highs (stop losses of shorts)
        - Sell-side liquidity: clusters of swing lows (stop losses of longs)
        """
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        n = len(c)
        
        bsl = np.zeros(n)  # Buy-side liquidity proximity
        ssl = np.zeros(n)  # Sell-side liquidity proximity
        liq_sweep = np.zeros(n, dtype=np.int32)
        
        lookback = 50
        tolerance = 0.003  # 0.3% cluster threshold
        
        for i in range(lookback, n):
            window_h = h[i-lookback:i]
            window_l = l[i-lookback:i]
            
            # Count how many highs are clustered near the max
            max_h = np.max(window_h)
            cluster_h = np.sum(np.abs(window_h - max_h) / (max_h + 1e-10) < tolerance)
            bsl[i] = cluster_h / lookback
            
            # Count how many lows are clustered near the min
            min_l = np.min(window_l)
            cluster_l = np.sum(np.abs(window_l - min_l) / (min_l + 1e-10) < tolerance)
            ssl[i] = cluster_l / lookback
            
            # Liquidity sweep: price takes out a cluster then reverses
            if h[i] > max_h and c[i] < h[i] - (h[i] - l[i]) * 0.5:
                liq_sweep[i] = -1  # Swept buy-side, bearish
            if l[i] < min_l and c[i] > l[i] + (h[i] - l[i]) * 0.5:
                liq_sweep[i] = 1   # Swept sell-side, bullish

        df["SMC_BSL"] = bsl
        df["SMC_SSL"] = ssl
        df["SMC_Liq_Sweep"] = liq_sweep
        df["SMC_Liq_Sweep_Sum"] = pd.Series(liq_sweep, index=df.index).rolling(10, min_periods=1).sum().fillna(0)
        
        return df

    def _premium_discount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Premium/Discount zones based on recent range.
        Premium = above 50% of range (look to sell)
        Discount = below 50% of range (look to buy)
        """
        for w in [20, 50]:
            rh = df["High"].rolling(w).max()
            rl = df["Low"].rolling(w).min()
            equilibrium = (rh + rl) / 2
            rng = rh - rl + 1e-10
            
            df[f"SMC_Premium_Discount_{w}"] = (df["Close"] - equilibrium) / rng * 2
            df[f"SMC_In_Premium_{w}"] = (df["Close"] > equilibrium + rng * 0.25).astype(int)
            df[f"SMC_In_Discount_{w}"] = (df["Close"] < equilibrium - rng * 0.25).astype(int)
        
        return df

    def _institutional_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect institutional candle patterns (large body, directional conviction)."""
        body = (df["Close"] - df["Open"]).abs()
        total_range = df["High"] - df["Low"] + 1e-10
        avg_range = total_range.rolling(20).mean()
        
        # Institutional candle: range > 1.5x average, body > 60% of range
        inst = ((total_range > 1.5 * avg_range) & (body / total_range > 0.6)).astype(int)
        direction = np.sign(df["Close"] - df["Open"])
        
        df["SMC_Inst_Candle"] = inst
        df["SMC_Inst_Direction"] = inst * direction
        df["SMC_Inst_Count"] = df["SMC_Inst_Candle"].rolling(10).sum().fillna(0)
        
        return df

    def _smc_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite SMC score combining all signals."""
        bull = (
            df.get("SMC_Bull_OB", 0) * 2 +
            df.get("SMC_Bull_FVG", 0) * 1.5 +
            df.get("SMC_Breaker_Bull", 0) * 1 +
            (df.get("SMC_Liq_Sweep", 0) == 1).astype(int) * 2 +
            df.get("SMC_In_Discount_20", 0) * 1
        )
        bear = (
            df.get("SMC_Bear_OB", 0) * 2 +
            df.get("SMC_Bear_FVG", 0) * 1.5 +
            df.get("SMC_Breaker_Bear", 0) * 1 +
            (df.get("SMC_Liq_Sweep", 0) == -1).astype(int) * 2 +
            df.get("SMC_In_Premium_20", 0) * 1
        )
        df["SMC_Score"] = bull - bear
        df["SMC_Score_Smooth"] = pd.Series(df["SMC_Score"]).rolling(5).mean().fillna(0)
        return df
