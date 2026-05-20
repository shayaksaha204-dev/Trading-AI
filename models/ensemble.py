"""
Ensemble Model — Meta-Learner combining TFT, CNN, and LSTM predictions
========================================================================
Aggregates signals with regime-adaptive weighting.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from loguru import logger
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, CHECKPOINT_DIR
from training.adaptive_strategies import get_strategy_tracker


class EnsembleModel:
    """
    Meta-learner that combines predictions from TFT, PatternCNN, and RegimeLSTM.
    
    Methods:
        - Simple weighted average
        - XGBoost/LightGBM meta-learner
        - Regime-adaptive weighting
    """

    def __init__(self, config=None):
        self.cfg = config or cfg.ensemble
        self.meta_model = None
        self.weights = {"tft": 0.5, "cnn": 0.25, "lstm": 0.25}
        self.regime_weights = {
            "bull": {"tft": 0.5, "cnn": 0.2, "lstm": 0.3},
            "bear": {"tft": 0.5, "cnn": 0.3, "lstm": 0.2},
            "sideways": {"tft": 0.3, "cnn": 0.4, "lstm": 0.3},
            "volatile": {"tft": 0.4, "cnn": 0.2, "lstm": 0.4},
        }
        self._fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Train the meta-learner on stacked model predictions."""
        if self.cfg.method == "xgboost":
            try:
                from xgboost import XGBClassifier
                self.meta_model = XGBClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    learning_rate=self.cfg.learning_rate,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                )
                self.meta_model.fit(features, targets)
                self._fitted = True
                logger.info("XGBoost meta-learner trained")
            except ImportError:
                logger.warning("XGBoost not available, falling back to weighted average")
        elif self.cfg.method == "lightgbm":
            try:
                import lightgbm as lgb
                self.meta_model = lgb.LGBMClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    learning_rate=self.cfg.learning_rate,
                    verbose=-1,
                )
                self.meta_model.fit(features, targets)
                self._fitted = True
                logger.info("LightGBM meta-learner trained")
            except ImportError:
                logger.warning("LightGBM not available, falling back to weighted average")

    def predict(
        self,
        tft_pred: Dict[str, np.ndarray],
        cnn_pred: Dict[str, np.ndarray],
        lstm_pred: Dict[str, np.ndarray],
        regime: str = None,
        strategy_result: Dict = None,
    ) -> Dict[str, np.ndarray]:
        """
        Combine predictions from all models + strategy confluence.
        Uses ADAPTIVE strategy weights that improve over time.
        
        Returns:
            Dict with: signal (-1/0/1), confidence (0-1), direction_probs,
            regime, strategy breakdown, reasoning
        """
        # Get direction probabilities from each model
        tft_dir = tft_pred.get("direction_probs", np.array([[0.33, 0.34, 0.33]]))
        cnn_dir = cnn_pred.get("direction_probs", np.array([[0.33, 0.34, 0.33]]))
        lstm_regime = lstm_pred.get("regime_probs", np.array([[0.25, 0.25, 0.25, 0.25]]))

        # Determine regime
        if regime is None:
            regime_names = ["bull", "bear", "sideways", "volatile"]
            regime = regime_names[lstm_regime[0].argmax()]

        # Get weights based on regime
        if self.cfg.regime_adaptive:
            w = self.regime_weights.get(regime, self.weights)
        else:
            w = self.weights

        # If meta-learner is trained, use it
        if self._fitted and self.meta_model is not None:
            meta_features = np.concatenate([
                tft_dir, cnn_dir, lstm_regime,
                tft_pred.get("confidence", np.array([0.5])).reshape(1, -1),
                cnn_pred.get("pattern_strength", np.array([0.5])).reshape(1, -1),
            ], axis=-1)
            try:
                meta_probs = self.meta_model.predict_proba(meta_features)
                direction_probs = meta_probs[0] if len(meta_probs.shape) > 1 else meta_probs
            except Exception:
                direction_probs = self._weighted_average(tft_dir, cnn_dir, w)
        else:
            direction_probs = self._weighted_average(tft_dir, cnn_dir, w)

        # Neural network signal
        nn_signal = int(np.argmax(direction_probs) - 1)
        nn_confidence = float(np.max(direction_probs))

        # Regime confidence
        regime_conf = float(lstm_pred.get("confidence", np.array([0.5]))[0])

        # ── Merge with Strategy Confluence (ADAPTIVE WEIGHTS) ──
        if strategy_result and strategy_result.get("signal", 0) != 0:
            strat_signal = strategy_result["signal"]
            strat_conf = strategy_result["confidence"]
            agreement = strategy_result.get("bull_count", 0) if strat_signal > 0 else strategy_result.get("bear_count", 0)
            
            # Get ADAPTIVE strategy weights based on performance
            try:
                strategy_tracker = get_strategy_tracker()
                adaptive_weights = strategy_tracker.get_weights(regime)
                
                # Weight strategy confidence by adaptive weights
                breakdown = strategy_result.get("breakdown", {})
                weighted_conf = 0.0
                total_weight = 0.0
                for strat_name, data in breakdown.items():
                    strat_key = strat_name.replace(" ", "_")
                    weight = adaptive_weights.get(strat_key, 1.0/8)
                    weighted_conf += data.get("confidence", 0.5) * weight
                    total_weight += weight
                
                if total_weight > 0:
                    strat_conf = weighted_conf / total_weight
            except Exception:
                pass  # Use default strat_conf if adaptive weighting fails

            if nn_signal == strat_signal:
                # Models and strategies AGREE → boost confidence
                signal = nn_signal
                confidence = min(nn_confidence * 0.5 + strat_conf * 0.3 + regime_conf * 0.2 + 0.1, 0.99)
            elif nn_signal == 0:
                # Model says HOLD but strategies have a signal → follow strategies with lower confidence
                signal = strat_signal
                confidence = strat_conf * 0.6 + regime_conf * 0.2
            elif strat_signal == 0:
                # Strategies say HOLD, model has signal → follow model with lower confidence
                signal = nn_signal
                confidence = nn_confidence * 0.5 + regime_conf * 0.2
            else:
                # Models and strategies DISAGREE → HOLD (conflicting signals are dangerous)
                signal = 0
                confidence = 0.3
        else:
            # No strategy data → use model output directly
            signal = nn_signal
            confidence = nn_confidence * 0.8 + regime_conf * 0.2

        result = {
            "signal": np.array([signal]),
            "signal_label": ["SELL", "HOLD", "BUY"][signal + 1],
            "confidence": np.array([confidence]),
            "direction_probs": direction_probs.reshape(1, -1) if direction_probs.ndim == 1 else direction_probs,
            "regime": regime,
            "regime_confidence": np.array([regime_conf]),
        }

        # Add strategy breakdown
        if strategy_result:
            result["strategy_agreement"] = strategy_result.get("agreement", "0/5")
            result["strategy_breakdown"] = strategy_result.get("breakdown", {})
            result["reasoning"] = strategy_result.get("reasoning", [])

        # Add forecast from TFT if available
        if "median_forecast" in tft_pred:
            result["median_forecast"] = tft_pred["median_forecast"]
            result["lower_bound"] = tft_pred.get("lower_bound", tft_pred["median_forecast"])
            result["upper_bound"] = tft_pred.get("upper_bound", tft_pred["median_forecast"])

        # Add pattern info from CNN
        if "predicted_pattern" in cnn_pred:
            result["pattern"] = cnn_pred["predicted_pattern"][0] if isinstance(cnn_pred["predicted_pattern"], list) else cnn_pred["predicted_pattern"]
            result["pattern_strength"] = cnn_pred.get("pattern_strength", np.array([0.5]))

        return result

    def _weighted_average(self, tft_dir, cnn_dir, w):
        """Weighted average of direction probabilities."""
        combined = w["tft"] * tft_dir + w["cnn"] * cnn_dir
        # Normalize
        combined = combined / (combined.sum(axis=-1, keepdims=True) + 1e-10)
        return combined[0] if combined.ndim > 1 else combined

    def save(self, path: str = None):
        """Save meta-learner to disk."""
        path = path or str(CHECKPOINT_DIR / "ensemble_meta.joblib")
        data = {"weights": self.weights, "regime_weights": self.regime_weights, "meta_model": self.meta_model, "fitted": self._fitted}
        joblib.dump(data, path)
        logger.info(f"Ensemble saved to {path}")

    def load(self, path: str = None):
        """Load meta-learner from disk."""
        path = path or str(CHECKPOINT_DIR / "ensemble_meta.joblib")
        if Path(path).exists():
            data = joblib.load(path)
            self.weights = data["weights"]
            self.regime_weights = data["regime_weights"]
            self.meta_model = data["meta_model"]
            self._fitted = data["fitted"]
            logger.info("Ensemble loaded")


if __name__ == "__main__":
    ensemble = EnsembleModel()
    # Simulate predictions
    tft_pred = {"direction_probs": np.array([[0.2, 0.3, 0.5]]), "confidence": np.array([0.75]), "median_forecast": np.array([[0.01, 0.02, 0.015, 0.01, 0.005]])}
    cnn_pred = {"direction_probs": np.array([[0.15, 0.25, 0.6]]), "pattern_strength": np.array([0.8]), "predicted_pattern": ["breakout"]}
    lstm_pred = {"regime_probs": np.array([[0.6, 0.1, 0.2, 0.1]]), "predicted_regime": ["bull"], "confidence": np.array([0.7])}
    result = ensemble.predict(tft_pred, cnn_pred, lstm_pred)
    print(f"Signal: {result['signal_label']}")
    print(f"Confidence: {result['confidence'][0]:.2%}")
    print(f"Regime: {result['regime']}")
    print(f"Pattern: {result.get('pattern', 'N/A')}")
