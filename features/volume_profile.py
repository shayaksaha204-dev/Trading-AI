"""
Volume Profile Features — POC, VWAP, Value Area, Delta
========================================================
Computes volume-at-price analysis critical for intraday SMC/ICT.
"""

import numpy as np
import pandas as pd
from loguru import logger


class VolumeProfileFeatures:
    """
    Computes volume profile features:
    - Session VWAP (resets daily)
    - Point of Control (POC)
    - Value Area High/Low
    - Volume Delta (buy/sell pressure estimate)
    - Cumulative Delta
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        has_volume = "Volume" in df.columns and (df["Volume"] > 0).any()

        if not has_volume:
            logger.debug("  VolumeProfile: No volume data, skipping")
            return df

        c = df["Close"]; h = df["High"]; l = df["Low"]; o = df["Open"]; v = df["Volume"]
        tp = (h + l + c) / 3.0

        # ── Session VWAP (resets each day) ──
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
            vwap = pd.Series(np.nan, index=df.index, dtype=float)
            vwap_upper = pd.Series(np.nan, index=df.index, dtype=float)
            vwap_lower = pd.Series(np.nan, index=df.index, dtype=float)

            for date in pd.unique(dates):
                mask = dates == date
                day_tp = tp[mask]
                day_v = v[mask]
                cum_tpv = (day_tp * day_v).cumsum()
                cum_v = day_v.cumsum()
                day_vwap = cum_tpv / (cum_v + 1e-10)
                vwap[mask] = day_vwap

                # VWAP bands (±1 std)
                cum_sq = ((day_tp - day_vwap) ** 2 * day_v).cumsum()
                vwap_std = np.sqrt(cum_sq / (cum_v + 1e-10))
                vwap_upper[mask] = day_vwap + vwap_std
                vwap_lower[mask] = day_vwap - vwap_std

            df["VP_VWAP"] = vwap
            df["VP_VWAP_Upper"] = vwap_upper
            df["VP_VWAP_Lower"] = vwap_lower
            df["VP_VWAP_Dist"] = ((c - vwap) / (c + 1e-10) * 100).fillna(0)
        else:
            # Non-datetime index: use running VWAP
            cum_tpv = (tp * v).cumsum()
            cum_v = v.cumsum()
            vwap = cum_tpv / (cum_v + 1e-10)
            df["VP_VWAP"] = vwap
            df["VP_VWAP_Dist"] = ((c - vwap) / (c + 1e-10) * 100).fillna(0)

        # ── Point of Control (price with highest volume in rolling window) ──
        poc_window = 78  # ~1 trading day for 5m data
        poc = pd.Series(np.nan, index=df.index, dtype=float)
        va_high = pd.Series(np.nan, index=df.index, dtype=float)
        va_low = pd.Series(np.nan, index=df.index, dtype=float)

        for i in range(poc_window, len(df)):
            window_slice = slice(i - poc_window, i)
            w_close = c.iloc[window_slice].values
            w_vol = v.iloc[window_slice].values

            if w_vol.sum() == 0:
                continue

            # Simple histogram: bin prices and sum volume per bin
            n_bins = min(20, poc_window // 4)
            price_min, price_max = w_close.min(), w_close.max()
            if price_max - price_min < 1e-10:
                continue

            bins = np.linspace(price_min, price_max, n_bins + 1)
            bin_vol = np.zeros(n_bins)
            for j in range(len(w_close)):
                bin_idx = min(int((w_close[j] - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
                bin_vol[bin_idx] += w_vol[j]

            # POC: bin with highest volume
            poc_bin = np.argmax(bin_vol)
            poc.iloc[i] = (bins[poc_bin] + bins[poc_bin + 1]) / 2

            # Value Area: 70% of total volume around POC
            total_vol = bin_vol.sum()
            target_vol = total_vol * 0.7
            sorted_bins = np.argsort(bin_vol)[::-1]
            cum_vol = 0
            va_bins = []
            for b in sorted_bins:
                cum_vol += bin_vol[b]
                va_bins.append(b)
                if cum_vol >= target_vol:
                    break
            va_bins_sorted = sorted(va_bins)
            va_high.iloc[i] = bins[max(va_bins_sorted) + 1] if va_bins_sorted else price_max
            va_low.iloc[i] = bins[min(va_bins_sorted)] if va_bins_sorted else price_min

        df["VP_POC"] = poc
        df["VP_POC_Dist"] = ((c - poc) / (c + 1e-10) * 100).fillna(0)
        df["VP_VA_High"] = va_high
        df["VP_VA_Low"] = va_low
        df["VP_In_Value_Area"] = ((c >= va_low) & (c <= va_high)).astype(float)

        # ── Volume Delta (buy/sell pressure estimate) ──
        # Approximation: if close > open, volume is buying pressure, else selling
        body_ratio = (c - o) / (h - l + 1e-10)
        buy_vol = v * np.clip((body_ratio + 1) / 2, 0, 1)
        sell_vol = v - buy_vol
        df["VP_Delta"] = buy_vol - sell_vol
        df["VP_Delta_Pct"] = df["VP_Delta"] / (v + 1e-10)
        df["VP_Cum_Delta"] = df["VP_Delta"].cumsum()

        # Cumulative delta trend
        cd = df["VP_Cum_Delta"]
        df["VP_Cum_Delta_SMA"] = cd.rolling(20).mean()
        df["VP_Cum_Delta_Trend"] = np.where(cd > cd.rolling(20).mean(), 1.0, -1.0)

        logger.debug(f"  VolumeProfile: Added volume profile features")
        return df

    def get_feature_names(self) -> list:
        return [
            "VP_VWAP", "VP_VWAP_Upper", "VP_VWAP_Lower", "VP_VWAP_Dist",
            "VP_POC", "VP_POC_Dist", "VP_VA_High", "VP_VA_Low", "VP_In_Value_Area",
            "VP_Delta", "VP_Delta_Pct", "VP_Cum_Delta",
            "VP_Cum_Delta_SMA", "VP_Cum_Delta_Trend",
        ]
