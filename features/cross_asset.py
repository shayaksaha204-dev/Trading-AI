"""
Cross-Asset Correlation Features — Intermarket Analysis
==========================================================
Markets don't move in isolation. This module computes:
  - Rolling correlation with reference assets (SPY, BTC, DXY, VIX)
  - Intermarket divergence flags
  - VIX regime as context feature
  - Relative strength vs benchmark
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class CrossAssetFeatures:
    """
    Computes cross-asset correlation and intermarket features.
    
    Reference assets:
    - SPY: US equity market proxy
    - BTC-USD: Crypto proxy
    - DX-Y.NYB: US Dollar index
    - ^VIX: Volatility/Fear index
    """

    def __init__(self):
        self.reference_data: Dict[str, pd.DataFrame] = {}
        self._loaded = False
        self.reference_tickers = getattr(
            cfg.data, 'cross_asset_references',
            ["SPY", "BTC-USD", "DX-Y.NYB", "^VIX"]
        )

    def load_references(self, fetcher=None, timeframe: str = None):
        """Load reference asset data. Call once before computing features."""
        if self._loaded:
            return

        timeframe = timeframe or cfg.data.primary_timeframe

        if fetcher is None:
            try:
                from data.fetcher import DataFetcher
                fetcher = DataFetcher()
            except Exception as e:
                logger.warning(f"CrossAsset: Could not create fetcher: {e}")
                return

        for ticker in self.reference_tickers:
            try:
                df = fetcher.fetch_ticker(ticker, timeframe=timeframe)
                if df is not None and len(df) > 20:
                    self.reference_data[ticker] = df
                    logger.debug(f"  CrossAsset: Loaded {ticker} ({len(df)} bars)")
            except Exception as e:
                logger.debug(f"  CrossAsset: Failed to load {ticker}: {e}")

        self._loaded = True
        logger.info(f"  CrossAsset: {len(self.reference_data)}/{len(self.reference_tickers)} references loaded")

    def compute_all(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """Add cross-asset features to a ticker's DataFrame."""
        df = df.copy()

        if not self.reference_data:
            logger.debug("  CrossAsset: No reference data loaded, skipping")
            return df

        close = df["Close"]

        # Skip self-correlation
        for ref_ticker, ref_df in self.reference_data.items():
            if ref_ticker == ticker:
                continue

            safe_name = ref_ticker.replace("-", "").replace("=", "").replace("^", "").replace(".", "")
            ref_close = ref_df["Close"]

            # Align reference data to primary index
            ref_aligned = ref_close.reindex(df.index, method='ffill')

            if ref_aligned.isna().sum() > len(ref_aligned) * 0.5:
                continue

            # Returns for correlation
            primary_ret = close.pct_change()
            ref_ret = ref_aligned.pct_change()

            # Rolling correlations
            for window in [20, 60]:
                corr = primary_ret.rolling(window).corr(ref_ret)
                df[f"Corr_{safe_name}_{window}"] = corr.fillna(0)

            # Relative strength
            df[f"RelStrength_{safe_name}"] = (
                close.pct_change(20) - ref_aligned.pct_change(20)
            ).fillna(0)

            # Intermarket divergence flag
            # (e.g., gold up while USD up = unusual)
            primary_direction = np.sign(close.pct_change(5))
            ref_direction = np.sign(ref_aligned.pct_change(5))

            if ref_ticker in ["DX-Y.NYB"] and ticker in ["GC=F", "SI=F"]:
                # Gold/Silver and USD normally inversely correlated
                df[f"Divergence_{safe_name}"] = (
                    (primary_direction == ref_direction).astype(float) - 0.5
                ) * 2  # -1 = expected, +1 = divergence
            elif ref_ticker == "^VIX":
                # High VIX = risk-off
                pass
            else:
                df[f"Divergence_{safe_name}"] = (
                    (primary_direction != ref_direction).astype(float) - 0.5
                ) * 2

        # ── VIX-specific features ──
        vix_ticker = next((t for t in self.reference_data if "VIX" in t), None)
        if vix_ticker and vix_ticker != ticker:
            vix = self.reference_data[vix_ticker]["Close"].reindex(df.index, method='ffill')
            if vix.notna().sum() > 20:
                # VIX regime
                vix_sma = vix.rolling(20).mean()
                df["VIX_Level"] = vix.fillna(20)
                df["VIX_Regime"] = np.where(
                    vix > 30, 3.0,  # High fear
                    np.where(vix > 20, 2.0,  # Elevated
                    np.where(vix > 12, 1.0, 0.0))  # Normal / Low
                )
                df["VIX_Trend"] = np.where(vix > vix_sma, 1.0, -1.0)

        return df

    def get_feature_names(self) -> List[str]:
        """Return expected feature names (approximate — depends on loaded refs)."""
        names = []
        for ref in self.reference_tickers:
            safe = ref.replace("-", "").replace("=", "").replace("^", "").replace(".", "")
            names.extend([
                f"Corr_{safe}_20", f"Corr_{safe}_60",
                f"RelStrength_{safe}", f"Divergence_{safe}",
            ])
        names.extend(["VIX_Level", "VIX_Regime", "VIX_Trend"])
        return names
