"""
Options Flow Features — Put/Call Ratio from yfinance
======================================================
Basic options analysis for US stocks using free yfinance data.
Returns neutral values for non-stock tickers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class OptionsFlowFeatures:
    """
    Computes basic options flow features from yfinance:
    - Put/Call ratio (open interest)
    - Put/Call volume ratio
    - Max pain estimate
    - Unusual volume flag
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 3600  # 1 hour

    def compute_all(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """Add options flow features. Only works for US stocks."""
        df = df.copy()

        category = cfg.assets.get_category(ticker)
        if category != "stocks":
            # Non-stock: add neutral values
            df["Opt_PC_Ratio"] = 1.0
            df["Opt_PC_Vol_Ratio"] = 1.0
            df["Opt_Unusual_Vol"] = 0.0
            return df

        # Fetch options data (cached)
        opts = self._get_options_data(ticker)
        if opts is None:
            df["Opt_PC_Ratio"] = 1.0
            df["Opt_PC_Vol_Ratio"] = 1.0
            df["Opt_Unusual_Vol"] = 0.0
            return df

        # Apply as constant features (options data is daily-level)
        df["Opt_PC_Ratio"] = opts.get("pc_ratio", 1.0)
        df["Opt_PC_Vol_Ratio"] = opts.get("pc_vol_ratio", 1.0)
        df["Opt_Unusual_Vol"] = opts.get("unusual_vol", 0.0)

        return df

    def _get_options_data(self, ticker: str) -> Optional[Dict]:
        """Fetch and cache options data."""
        now = time.time()
        if ticker in self._cache and (now - self._cache_time.get(ticker, 0)) < self._cache_ttl:
            return self._cache[ticker]

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            dates = stock.options

            if not dates:
                return None

            # Use nearest expiry
            chain = stock.option_chain(dates[0])
            calls = chain.calls
            puts = chain.puts

            total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
            total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0

            pc_ratio = total_put_oi / max(total_call_oi, 1)
            pc_vol_ratio = total_put_vol / max(total_call_vol, 1)

            # Unusual volume: total options volume vs typical
            total_vol = total_call_vol + total_put_vol
            unusual = 1.0 if total_vol > 10000 else 0.0  # Simple threshold

            result = {
                "pc_ratio": float(np.clip(pc_ratio, 0, 5)),
                "pc_vol_ratio": float(np.clip(pc_vol_ratio, 0, 5)),
                "unusual_vol": unusual,
            }

            self._cache[ticker] = result
            self._cache_time[ticker] = now
            return result

        except Exception as e:
            logger.debug(f"  Options: Failed for {ticker}: {e}")
            return None

    def get_feature_names(self) -> list:
        return ["Opt_PC_Ratio", "Opt_PC_Vol_Ratio", "Opt_Unusual_Vol"]
