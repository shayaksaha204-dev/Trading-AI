"""
Data Fetcher — Multi-Source Market Data Acquisition
=====================================================
Fetches OHLCV data from Yahoo Finance for stocks, forex, crypto, commodities.
Supports caching, rate limiting, and multiple timeframes.
"""

import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, DATA_DIR


class DataFetcher:
    """
    Multi-source market data fetcher with caching and rate limiting.
    
    Supports:
        - US Stocks & ETFs (Yahoo Finance)
        - Forex pairs (Yahoo Finance)
        - Crypto (Yahoo Finance / CoinGecko)
        - Commodities futures (Yahoo Finance)
        - Market indices (Yahoo Finance)
    """

    def __init__(self):
        self.cache_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = 0.25  # seconds between requests
        self._last_request_time = 0.0
        logger.info(f"DataFetcher initialized — cache: {self.cache_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all(
        self,
        timeframe: str = None,
        years: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all configured assets.
        
        Args:
            timeframe: OHLCV interval ('1h', '1d', '1wk'). Default from config.
            years: Years of history. Default from config.
            
        Returns:
            Dict mapping ticker -> DataFrame with OHLCV data.
        """
        timeframe = timeframe or cfg.data.primary_timeframe
        years = years or cfg.data.history_years
        tickers = cfg.assets.all_tickers
        
        logger.info(f"Fetching {len(tickers)} assets | {timeframe} | {years}y history")
        
        results = {}
        failed = []
        
        for i, ticker in enumerate(tickers):
            try:
                df = self.fetch_ticker(ticker, timeframe=timeframe, years=years)
                if df is not None and len(df) > 0:
                    results[ticker] = df
                    logger.success(
                        f"  [{i+1}/{len(tickers)}] {ticker}: "
                        f"{len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})"
                    )
                else:
                    failed.append(ticker)
                    logger.warning(f"  [{i+1}/{len(tickers)}] {ticker}: No data returned")
            except Exception as e:
                failed.append(ticker)
                logger.error(f"  [{i+1}/{len(tickers)}] {ticker}: {e}")
        
        logger.info(
            f"Fetch complete: {len(results)} succeeded, {len(failed)} failed"
        )
        if failed:
            logger.warning(f"Failed tickers: {failed}")
        
        return results

    def fetch_ticker(
        self,
        ticker: str,
        timeframe: str = None,
        years: int = None,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker with caching.
        
        Args:
            ticker: Yahoo Finance ticker symbol.
            timeframe: Data interval (1m, 3m, 5m, 15m, 1h, 1d).
            years: Years of history.
            force_refresh: Bypass cache.
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Returns, LogReturns
        """
        timeframe = timeframe or cfg.data.primary_timeframe
        years = years or cfg.data.history_years
        
        # Check cache first
        if not force_refresh:
            cached = self._load_cache(ticker, timeframe)
            if cached is not None:
                return cached
        
        # Rate limiting
        self._rate_limit()
        
        # Determine date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Handle yfinance period limits for intraday
        # yfinance limits: 1m=7d, 3m/5m/15m/30m/1h=60d, >=1d=730d
        if timeframe == "1m":
            start_date = end_date - timedelta(days=6)  # Max 7 days for 1m
        elif timeframe in ["3m", "5m", "15m", "30m", "1h"]:
            start_date = max(start_date, end_date - timedelta(days=59))  # Max 60 days
        elif timeframe in ["90m", "1d"]:
            start_date = max(start_date, end_date - timedelta(days=729))  # Max 730 days
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=timeframe,
                auto_adjust=True,
            )
            
            if df is None or df.empty:
                return None
            
            # Clean and standardize
            df = self._clean_data(df, ticker)
            
            # Cache result
            self._save_cache(df, ticker, timeframe)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return None

    def fetch_category(
        self,
        category: str,
        timeframe: str = None,
        years: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all assets in a category (stocks, forex, crypto, commodities, indices).
        """
        category_map = {
            "stocks": cfg.assets.stocks,
            "indian_stocks": cfg.assets.indian_stocks,
            "forex": cfg.assets.forex,
            "crypto": cfg.assets.crypto,
            "commodities": cfg.assets.commodities,
            "indices": cfg.assets.indices,
        }
        
        tickers = category_map.get(category, [])
        if not tickers:
            logger.warning(f"Unknown category: {category}")
            return {}
        
        results = {}
        for ticker in tickers:
            df = self.fetch_ticker(ticker, timeframe=timeframe, years=years)
            if df is not None:
                results[ticker] = df
        
        return results

    def fetch_multi_timeframe(
        self,
        ticker: str,
        primary_tf: str = None,
        higher_tfs: list = None,
        years: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes for a single ticker.
        
        Args:
            ticker: Asset symbol
            primary_tf: Primary timeframe (default from config)
            higher_tfs: List of higher timeframes (default from config)
            years: Years of history
            
        Returns:
            Dict of {timeframe: DataFrame} e.g. {"5m": df_5m, "15m": df_15m, "1h": df_1h}
        """
        primary_tf = primary_tf or cfg.data.primary_timeframe
        higher_tfs = higher_tfs or getattr(cfg.data, 'higher_timeframes', ["15m", "1h"])
        years = years or cfg.data.history_years
        
        result = {}
        
        # Primary timeframe
        df = self.fetch_ticker(ticker, timeframe=primary_tf, years=years)
        if df is not None:
            result[primary_tf] = df
        
        # Higher timeframes
        for tf in higher_tfs:
            if tf != primary_tf:
                htf_df = self.fetch_ticker(ticker, timeframe=tf, years=years)
                if htf_df is not None:
                    result[tf] = htf_df
        
        return result

    def get_latest_prices(self, tickers: List[str] = None) -> pd.DataFrame:
        """Get the latest price for given tickers."""
        tickers = tickers or cfg.assets.all_tickers
        records = []
        
        for ticker in tickers:
            try:
                self._rate_limit()
                stock = yf.Ticker(ticker)
                info = stock.fast_info
                records.append({
                    "ticker": ticker,
                    "price": getattr(info, "last_price", None),
                    "prev_close": getattr(info, "previous_close", None),
                    "market_cap": getattr(info, "market_cap", None),
                    "category": cfg.assets.get_category(ticker),
                })
            except Exception as e:
                logger.warning(f"Could not get latest price for {ticker}: {e}")
        
        return pd.DataFrame(records)

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize and enrich raw OHLCV data."""
        # Keep core columns
        core_cols = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in core_cols if c in df.columns]
        df = df[available].copy()
        
        # Ensure numeric
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Drop rows where Close is NaN
        df = df.dropna(subset=["Close"])
        
        # Sort by date
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        
        # Add derived columns
        df["Returns"] = df["Close"].pct_change()
        df["LogReturns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Ticker"] = ticker
        df["Category"] = cfg.assets.get_category(ticker)
        
        # Forward-fill volume if missing
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0).astype(float)
        
        return df

    def _cache_path(self, ticker: str, timeframe: str) -> Path:
        """Generate cache file path."""
        safe_ticker = ticker.replace("=", "_").replace("^", "_").replace("-", "_")
        return self.cache_dir / f"{safe_ticker}_{timeframe}.parquet"

    def _load_cache(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from cache if fresh enough."""
        path = self._cache_path(ticker, timeframe)
        if not path.exists():
            return None
        
        # Check age
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > cfg.data.cache_expiry_hours:
            return None
        
        try:
            df = pd.read_parquet(path)
            logger.debug(f"Cache hit: {ticker} ({len(df)} rows)")
            return df
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """Save data to cache."""
        path = self._cache_path(ticker, timeframe)
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {e}")

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def clear_cache(self):
        """Remove all cached data files."""
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached files")


# ── Standalone Usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Test single fetch
    print("\n--- Testing single ticker fetch ---")
    df = fetcher.fetch_ticker("AAPL", timeframe="1d", years=2)
    if df is not None:
        print(f"AAPL: {len(df)} rows")
        print(df.tail(3))
    
    # Test category fetch
    print("\n--- Testing crypto category ---")
    crypto = fetcher.fetch_category("crypto", years=1)
    for ticker, data in crypto.items():
        print(f"  {ticker}: {len(data)} rows")
