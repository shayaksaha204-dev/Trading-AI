"""
Data Preprocessor — Cleaning, Normalization, and Dataset Creation
==================================================================
Transforms raw OHLCV data into ML-ready datasets with proper
train/val/test splits, no lookahead bias, and rolling windows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class DataPreprocessor:
    """
    Preprocesses raw market data for model consumption.
    
    Pipeline:
        1. Handle missing values (forward-fill, interpolation)
        2. Detect and clip outliers (winsorization)
        3. Normalize / scale features
        4. Create rolling window sequences
        5. Split into train / validation / test (time-aware)
    """

    def __init__(self):
        self.scalers: Dict[str, RobustScaler] = {}
        self.feature_columns: List[str] = []
        self.target_column: str = "Returns"

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare_dataset(
        self,
        data: Dict[str, pd.DataFrame],
        feature_df: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, dict]:
        """
        Full preprocessing pipeline for all assets.
        
        Args:
            data: Dict of ticker -> raw OHLCV DataFrame.
            feature_df: Optional dict of ticker -> feature-enriched DataFrame.
            
        Returns:
            Dict of ticker -> {
                'train': (X, y),
                'val': (X, y),
                'test': (X, y),
                'scaler': fitted scaler,
                'feature_columns': list of feature names,
                'dates': {'train': dates, 'val': dates, 'test': dates}
            }
        """
        datasets = {}
        
        for ticker, df in data.items():
            try:
                # Use feature-enriched data if available
                if feature_df and ticker in feature_df:
                    enriched = feature_df[ticker]
                else:
                    enriched = df
                
                result = self.prepare_single(enriched, ticker)
                if result is not None:
                    datasets[ticker] = result
                    
            except Exception as e:
                logger.error(f"Failed to preprocess {ticker}: {e}")
        
        logger.info(f"Preprocessed {len(datasets)} / {len(data)} assets")
        return datasets

    def prepare_single(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
    ) -> Optional[dict]:
        """
        Preprocess a single asset's data.
        
        Returns dict with train/val/test splits as numpy arrays.
        """
        if df is None or len(df) < cfg.data.sequence_length + cfg.data.prediction_horizon + 50:
            logger.warning(
                f"{ticker}: Not enough data ({len(df) if df is not None else 0} rows, "
                f"need {cfg.data.sequence_length + cfg.data.prediction_horizon + 50})"
            )
            return None
        
        # Defragment the DataFrame to avoid PerformanceWarnings
        df = df.copy()
        
        # Step 1: Clean
        df = self._handle_missing(df)
        df = self._clip_outliers(df)
        
        # Step 2: Identify feature columns (numeric only, exclude metadata)
        exclude_cols = {"Ticker", "Category", "Date", "Datetime"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.feature_columns = feature_cols
        
        # Step 2.5: Replace any remaining inf values with NaN, then drop
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Step 3: Drop rows with NaN in features
        df = df.dropna(subset=feature_cols)
        
        if len(df) < cfg.data.sequence_length + cfg.data.prediction_horizon + 50:
            logger.warning(f"{ticker}: Not enough clean data after NaN removal")
            return None
        
        # Defragment again before adding new columns
        df = df.copy()
        
        # Step 4: Create target variable
        label_method = getattr(cfg.data, 'label_method', 'simple')
        
        if label_method == "triple_barrier":
            # Triple Barrier Method — labels based on which barrier price hits first
            from data.labeling import TripleBarrierLabeler
            labeler = TripleBarrierLabeler(
                tp_atr_mult=cfg.data.tp_atr_multiplier,
                sl_atr_mult=cfg.data.sl_atr_multiplier,
                max_holding_period=cfg.data.prediction_horizon,
            )
            target, direction = labeler.label(df)
            df["Target"] = target
            df["TargetDirection"] = direction
            logger.debug(f"{ticker}: Using triple barrier labeling")
        else:
            # Simple method — future returns over prediction horizon
            target = df["Close"].shift(-cfg.data.prediction_horizon) / df["Close"] - 1
            threshold = 0.005  # 0.5% threshold for "no change"
            direction = np.where(target > threshold, 1, np.where(target < -threshold, -1, 0))
            df["Target"] = target
            df["TargetDirection"] = direction
        
        # Drop rows where target is NaN (last N rows)
        df = df.dropna(subset=["Target"])
        
        # Step 5: Scale features
        scaler = RobustScaler()
        feature_matrix = df[feature_cols].values
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Clip extreme values and clean any remaining NaN/inf post-scaling
        scaled_features = np.clip(scaled_features, -10, 10)
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=10.0, neginf=-10.0)
        self.scalers[ticker] = scaler
        
        # Step 6: Create rolling window sequences
        X, y_reg, y_cls, dates = self._create_sequences(
            scaled_features,
            df["Target"].values,
            df["TargetDirection"].values,
            df.index,
        )
        
        if X is None:
            return None
        
        # Step 7: Time-aware split
        n = len(X)
        train_end = int(n * cfg.data.train_ratio)
        val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))
        
        result = {
            "train": {
                "X": X[:train_end],
                "y_regression": y_reg[:train_end],
                "y_classification": y_cls[:train_end],
            },
            "val": {
                "X": X[train_end:val_end],
                "y_regression": y_reg[train_end:val_end],
                "y_classification": y_cls[train_end:val_end],
            },
            "test": {
                "X": X[val_end:],
                "y_regression": y_reg[val_end:],
                "y_classification": y_cls[val_end:],
            },
            "scaler": scaler,
            "feature_columns": feature_cols,
            "dates": {
                "train": dates[:train_end],
                "val": dates[train_end:val_end],
                "test": dates[val_end:],
            },
            "ticker": ticker,
            "n_features": len(feature_cols),
        }
        
        logger.info(
            f"{ticker}: train={train_end}, val={val_end - train_end}, "
            f"test={n - val_end}, features={len(feature_cols)}"
        )
        
        return result

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward-fill then interpolation."""
        # Forward-fill first (most appropriate for time series)
        df = df.ffill()
        
        # Then interpolate remaining gaps
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5)
        
        # Fill any remaining NaNs with column median
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        n_std: float = 5.0,
    ) -> pd.DataFrame:
        """Winsorize outliers beyond n standard deviations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude price columns from clipping (they're naturally variable)
        clip_cols = [c for c in numeric_cols if c not in ["Open", "High", "Low", "Close", "Volume"]]
        
        for col in clip_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower = mean - n_std * std
                upper = mean + n_std * std
                df[col] = df[col].clip(lower, upper)
        
        return df

    def _create_sequences(
        self,
        features: np.ndarray,
        targets_reg: np.ndarray,
        targets_cls: np.ndarray,
        index: pd.DatetimeIndex,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create rolling window sequences for time series models.
        
        Args:
            features: Scaled feature matrix [N, F]
            targets_reg: Regression targets [N]
            targets_cls: Classification targets [N]
            index: DatetimeIndex for tracking dates
            
        Returns:
            X: [num_sequences, sequence_length, n_features]
            y_reg: [num_sequences]
            y_cls: [num_sequences]
            dates: [num_sequences]
        """
        seq_len = cfg.data.sequence_length
        n = len(features)
        
        if n < seq_len + 1:
            return None, None, None, None
        
        num_sequences = n - seq_len
        n_features = features.shape[1]
        
        X = np.zeros((num_sequences, seq_len, n_features), dtype=np.float32)
        y_reg = np.zeros(num_sequences, dtype=np.float32)
        y_cls = np.zeros(num_sequences, dtype=np.int64)
        dates = np.empty(num_sequences, dtype=object)
        
        for i in range(num_sequences):
            X[i] = features[i : i + seq_len]
            y_reg[i] = targets_reg[i + seq_len - 1]
            y_cls[i] = targets_cls[i + seq_len - 1]
            dates[i] = index[i + seq_len - 1]
        
        return X, y_reg, y_cls, dates

    def inverse_transform(
        self,
        ticker: str,
        data: np.ndarray,
    ) -> np.ndarray:
        """Inverse-transform scaled data back to original scale."""
        if ticker in self.scalers:
            return self.scalers[ticker].inverse_transform(data)
        return data

    def get_scaler(self, ticker: str) -> Optional[RobustScaler]:
        """Get the fitted scaler for a ticker."""
        return self.scalers.get(ticker)


# ── Standalone Usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from fetcher import DataFetcher
    
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    
    # Fetch test data
    df = fetcher.fetch_ticker("AAPL", years=3)
    if df is not None:
        result = preprocessor.prepare_single(df, "AAPL")
        if result:
            print(f"\nAAPL Preprocessing Results:")
            print(f"  Features: {result['n_features']}")
            print(f"  Train X shape: {result['train']['X'].shape}")
            print(f"  Val X shape:   {result['val']['X'].shape}")
            print(f"  Test X shape:  {result['test']['X'].shape}")
            print(f"  Feature columns: {result['feature_columns'][:10]}...")
