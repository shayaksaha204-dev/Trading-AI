"""
Hyperparameter Tuner — Optuna-based Bayesian Optimization
===========================================================
"""

import optuna
import numpy as np
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class HyperparameterTuner:
    """Automated hyperparameter search using Optuna."""

    def __init__(self, n_trials=None, timeout_hours=None):
        self.n_trials = n_trials or cfg.training.hpo_n_trials
        self.timeout = (timeout_hours or cfg.training.hpo_timeout_hours) * 3600

    def tune_tft(self, train_data, val_data, n_features):
        """Tune TFT hyperparameters."""
        def objective(trial):
            from config import TFTConfig
            from training.trainer import ModelTrainer

            config = TFTConfig(
                hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
                num_attention_heads=trial.suggest_categorical("num_heads", [2, 4]),
                lstm_layers=trial.suggest_int("lstm_layers", 1, 3),
                dropout=trial.suggest_float("dropout", 0.05, 0.4),
                learning_rate=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            )
            # Override global config temporarily
            original = cfg.tft
            cfg.tft = config
            cfg.training.max_epochs = 30  # Shorter for HPO

            try:
                trainer = ModelTrainer()
                trainer.train_tft(train_data, val_data, n_features)
                return trainer.best_val_loss
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("inf")
            finally:
                cfg.tft = original

        study = optuna.create_study(direction="minimize", study_name="tft_hpo")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        logger.info(f"Best TFT params: {study.best_params}")
        logger.info(f"Best val loss: {study.best_value:.6f}")
        return study.best_params, study.best_value
