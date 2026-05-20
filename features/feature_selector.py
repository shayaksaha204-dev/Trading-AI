"""
Feature Selection — Remove Noise, Keep Signal
================================================
Removes redundant/correlated features and selects
the most predictive ones for the model.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from loguru import logger


class FeatureSelector:
    """
    Automatic feature selection pipeline:
    1. Remove zero-variance features
    2. Remove highly correlated features (>0.95)
    3. Keep features with meaningful target correlation
    """

    def __init__(self, correlation_threshold: float = 0.95, min_target_corr: float = 0.01):
        self.correlation_threshold = correlation_threshold
        self.min_target_corr = min_target_corr
        self.selected_features: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "Target") -> "FeatureSelector":
        """Fit the selector on training data."""
        exclude = {"Ticker", "Category", "Date", "Datetime", "Target", "TargetDirection"}
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        if not feature_cols:
            self.selected_features = []
            self._fitted = True
            return self

        features = df[feature_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Step 1: Remove zero/near-zero variance
        variances = features.var()
        nonzero = variances[variances > 1e-8].index.tolist()
        removed_zv = len(feature_cols) - len(nonzero)
        features = features[nonzero]

        # Step 2: Remove highly correlated features (keep one from each cluster)
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > self.correlation_threshold].tolist()
            if correlated and col not in to_drop:
                # Among correlated features, keep the one with highest target correlation
                if target_col in df.columns:
                    target = df[target_col].fillna(0)
                    correlations = {c: abs(features[c].corr(target)) for c in correlated}
                    col_corr = abs(features[col].corr(target))
                    correlations[col] = col_corr
                    best = max(correlations, key=correlations.get)
                    to_drop.update(c for c in correlations if c != best)
                else:
                    to_drop.update(correlated)

        remaining = [c for c in features.columns if c not in to_drop]
        removed_corr = len(features.columns) - len(remaining)
        features = features[remaining]

        # Step 3: Remove features with near-zero target correlation
        if target_col in df.columns:
            target = df[target_col].fillna(0)
            target_corrs = features.corrwith(target).abs()
            useful = target_corrs[target_corrs >= self.min_target_corr].index.tolist()
            removed_useless = len(remaining) - len(useful)
        else:
            useful = remaining
            removed_useless = 0

        self.selected_features = useful
        self._fitted = True

        logger.info(f"  Feature Selection: {len(feature_cols)} → {len(useful)} features")
        logger.info(f"    Removed {removed_zv} zero-variance, {removed_corr} correlated, {removed_useless} low-signal")

        return self

    def transform(self, df: pd.DataFrame) -> List[str]:
        """Return the selected feature columns."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        # Only return features that exist in this DataFrame
        return [c for c in self.selected_features if c in df.columns]

    def fit_transform(self, df: pd.DataFrame, target_col: str = "Target") -> List[str]:
        """Fit and return selected features."""
        self.fit(df, target_col)
        return self.transform(df)
