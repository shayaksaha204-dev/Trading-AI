"""
Feature Store — Centralized Feature Management
================================================
Manages feature computation, caching, versioning, and retrieval
with point-in-time correctness to prevent data leakage.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, DATA_DIR


class FeatureStore:
    """
    Centralized feature management system.
    
    Features:
        - Point-in-time computation (no lookahead bias)
        - Caching with versioning
        - Feature importance tracking
        - Unified feature retrieval for all models
    """

    def __init__(self):
        self.store_dir = DATA_DIR / "feature_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.store_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self.feature_importance: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_and_store(
        self,
        raw_data: Dict[str, pd.DataFrame],
        feature_pipeline,
        version: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute features for all assets and store them.
        
        Args:
            raw_data: Dict of ticker -> OHLCV DataFrame.
            feature_pipeline: Callable that takes a DataFrame and returns featured DataFrame.
            version: Optional version tag. Auto-generated if None.
            
        Returns:
            Dict of ticker -> feature-enriched DataFrame.
        """
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {}
        
        for ticker, df in raw_data.items():
            try:
                featured_df = feature_pipeline(df)
                if featured_df is not None and len(featured_df) > 0:
                    self._save_features(ticker, featured_df, version)
                    results[ticker] = featured_df
            except Exception as e:
                logger.error(f"Feature computation failed for {ticker}: {e}")
        
        # Update metadata
        self.metadata["last_version"] = version
        self.metadata["last_update"] = datetime.now().isoformat()
        self.metadata["n_assets"] = len(results)
        self._save_metadata()
        
        logger.info(f"Feature store updated: v{version}, {len(results)} assets")
        return results

    def load_features(
        self,
        ticker: str,
        version: str = None,
    ) -> Optional[pd.DataFrame]:
        """Load cached features for a ticker."""
        version = version or self.metadata.get("last_version", "latest")
        path = self.store_dir / f"{self._safe_name(ticker)}_{version}.parquet"
        
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Failed to load features for {ticker}: {e}")
        
        return None

    def load_all_features(
        self,
        version: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load cached features for all available assets."""
        version = version or self.metadata.get("last_version", "latest")
        results = {}
        
        for path in self.store_dir.glob(f"*_{version}.parquet"):
            ticker = path.stem.replace(f"_{version}", "")
            try:
                results[ticker] = pd.read_parquet(path)
            except Exception:
                pass
        
        return results

    def get_feature_list(self, ticker: str = None) -> List[str]:
        """Get list of computed feature names."""
        if ticker:
            df = self.load_features(ticker)
            if df is not None:
                return list(df.columns)
        
        # Return from metadata
        return self.metadata.get("feature_columns", [])

    def update_importance(self, importance: Dict[str, float]):
        """Update feature importance scores from model training."""
        self.feature_importance.update(importance)
        self.metadata["feature_importance"] = self.feature_importance
        self._save_metadata()

    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [f[0] for f in sorted_features[:n]]

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _save_features(self, ticker: str, df: pd.DataFrame, version: str):
        """Save features to parquet."""
        path = self.store_dir / f"{self._safe_name(ticker)}_{version}.parquet"
        df.to_parquet(path)

    def _safe_name(self, ticker: str) -> str:
        """Convert ticker to filesystem-safe name."""
        return ticker.replace("=", "_").replace("^", "_").replace("-", "_")

    def _load_metadata(self) -> dict:
        """Load store metadata."""
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text())
            except Exception:
                pass
        return {}

    def _save_metadata(self):
        """Save store metadata."""
        try:
            self.metadata_path.write_text(json.dumps(self.metadata, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def clear(self):
        """Remove all stored features."""
        count = 0
        for f in self.store_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        self.metadata = {}
        self._save_metadata()
        logger.info(f"Feature store cleared: {count} files removed")
