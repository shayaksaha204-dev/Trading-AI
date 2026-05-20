"""
Walk-Forward Validation — Time-Series Cross-Validation
========================================================
Expanding/sliding window validation for unbiased model evaluation.
Properly integrated into the training pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class WalkForwardValidator:
    """
    Walk-forward validation for time series.
    Splits data into expanding or sliding training windows
    with out-of-sample validation at each fold.
    """

    def __init__(self, n_folds: int = None, expanding: bool = None):
        self.n_folds = n_folds or cfg.training.walk_forward_folds
        self.expanding = expanding if expanding is not None else cfg.training.expanding_window

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation index splits.
        
        Returns list of (train_indices, val_indices) tuples.
        """
        min_train = n_samples // (self.n_folds + 1)
        fold_size = (n_samples - min_train) // self.n_folds
        splits = []

        for i in range(self.n_folds):
            if self.expanding:
                train_end = min_train + i * fold_size
                train_idx = np.arange(0, train_end)
            else:
                train_start = i * fold_size
                train_end = min_train + i * fold_size
                train_idx = np.arange(train_start, train_end)

            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)
            val_idx = np.arange(val_start, val_end)

            if len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        logger.info(f"Walk-forward: {len(splits)} folds, {'expanding' if self.expanding else 'sliding'} window")
        for i, (tr, va) in enumerate(splits):
            logger.debug(f"  Fold {i+1}: train={len(tr)}, val={len(va)}")
        return splits

    def run_pipeline(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        trainer,
        model_type: str,
        n_features: int,
        feature_columns: List[str] = None,
    ) -> Tuple:
        """
        Run walk-forward validation integrated with the training pipeline.
        
        Trains the model on each fold and returns the best model
        (from the final fold, which has the most training data).
        
        Args:
            X: Feature array [N, seq_len, n_features]
            y_reg: Regression targets [N]
            y_cls: Classification targets [N]
            trainer: ModelTrainer instance
            model_type: "tft", "cnn", or "lstm"
            n_features: Number of features
            feature_columns: Feature column names
            
        Returns:
            (best_model, fold_results_dict)
        """
        splits = self.split(len(X))
        fold_results = []
        best_model = None
        best_val_loss = float("inf")

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n--- Walk-Forward Fold {fold_idx+1}/{len(splits)} ---")
            logger.info(f"  Train: {len(train_idx)} samples (idx {train_idx[0]}-{train_idx[-1]})")
            logger.info(f"  Val:   {len(val_idx)} samples (idx {val_idx[0]}-{val_idx[-1]})")

            train_data = {
                "X": X[train_idx],
                "y_regression": y_reg[train_idx],
                "y_classification": y_cls[train_idx],
            }
            val_data = {
                "X": X[val_idx],
                "y_regression": y_reg[val_idx],
                "y_classification": y_cls[val_idx],
            }

            # Train model on this fold
            model = self._train_fold(
                trainer, model_type, train_data, val_data,
                n_features, feature_columns
            )

            val_loss = trainer.best_val_loss
            fold_results.append({
                "fold": fold_idx + 1,
                "best_val_loss": val_loss,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            })

            # Keep the best model (typically the last fold has most data)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        avg_val_loss = np.mean([r["best_val_loss"] for r in fold_results])
        logger.info(f"\nWalk-forward complete. Avg val loss: {avg_val_loss:.6f}")

        result = {
            "folds": fold_results,
            "avg_val_loss": avg_val_loss,
            "best_fold": min(fold_results, key=lambda r: r["best_val_loss"])["fold"],
        }

        return best_model, result

    def _train_fold(self, trainer, model_type, train_data, val_data, n_features, feature_columns):
        """Train a single fold."""
        if model_type == "tft":
            return trainer.train_tft(train_data, val_data, n_features, feature_columns)
        elif model_type == "cnn":
            return trainer.train_cnn(train_data, val_data, n_features, feature_columns)
        elif model_type == "lstm":
            return trainer.train_lstm(train_data, val_data, n_features, feature_columns)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_final_split(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get just the final fold's split (for use without full walk-forward).
        This gives the most training data with a proper temporal split.
        """
        splits = self.split(n_samples)
        if splits:
            return splits[-1]
        # Fallback: simple split
        train_end = int(n_samples * 0.85)
        return np.arange(0, train_end), np.arange(train_end, n_samples)
