"""
Trading Predictor — Real-time Prediction Engine
==================================================
Loads trained models and generates predictions for given assets.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, CHECKPOINT_DIR
from models.tft_predictor import TFTPredictor
from models.pattern_cnn import PatternCNN
from models.regime_lstm import RegimeLSTM
from models.ensemble import EnsembleModel
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from features.technical import TechnicalFeatures
from features.market_structure import MarketStructure
from features.price_action import PriceAction
from features.smc import SmartMoneyConcepts
from features.ict import ICTConcepts
from features.wyckoff import WyckoffAnalysis
from features.elliott_wave import ElliottWave
from features.strategy_confluence import StrategyConfluence
from training.continuous_learner import get_continuous_learner
from training.adaptive_strategies import AdaptiveConfluenceScorer, get_strategy_tracker


class TradingPredictor:
    """
    Inference engine that loads trained models and generates predictions.
    """

    def __init__(self, n_features: int = None):
        self.device = cfg.gpu.device
        self.n_features = n_features
        self.tft = None
        self.cnn = None
        self.lstm = None
        self.ensemble = EnsembleModel()
        self.fetcher = DataFetcher()
        self.preprocessor = DataPreprocessor()
        self.tech = TechnicalFeatures()
        self.structure = MarketStructure()
        self.price_action = PriceAction()
        self.smc = SmartMoneyConcepts()
        self.ict = ICTConcepts()
        self.wyckoff = WyckoffAnalysis()
        self.elliott = ElliottWave()
        self.confluence = StrategyConfluence(min_agreement=3, min_confidence=0.55)
        
        # NEW: Continuous learning components
        self.tft_learner = get_continuous_learner("tft")
        self.cnn_learner = get_continuous_learner("cnn")
        self.lstm_learner = get_continuous_learner("lstm")
        self.adaptive_scorer = AdaptiveConfluenceScorer()
        
        # Track pending outcomes for learning
        self._pending_outcomes: List[Dict] = []
        
        self._loaded = False

    def _detect_n_features(self, checkpoint_path: Path, key: str = "input_proj.weight") -> int:
        """Detect n_features from a saved checkpoint by inspecting weight shapes."""
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if key in state_dict:
                # input_proj.weight shape is [hidden_size, n_features]
                return state_dict[key].shape[1]
        except Exception as e:
            logger.warning(f"Could not detect n_features from {checkpoint_path}: {e}")
        return None

    def load_models(self, n_features: int = None):
        """Load all trained model checkpoints.
        
        Auto-detects n_features from the checkpoint if not provided,
        so the model architecture matches the saved weights.
        """
        # Auto-detect from TFT checkpoint first (most reliable)
        tft_path = CHECKPOINT_DIR / "tft_best.pt"
        cnn_path = CHECKPOINT_DIR / "cnn_best.pt"
        lstm_path = CHECKPOINT_DIR / "lstm_best.pt"

        if n_features is None and self.n_features is None:
            for path in [tft_path, cnn_path, lstm_path]:
                if path.exists():
                    detected = self._detect_n_features(path)
                    if detected:
                        n_features = detected
                        logger.info(f"Auto-detected n_features={n_features} from {path.name}")
                        break

        n_features = n_features or self.n_features or 64

        # TFT
        if tft_path.exists():
            self.tft = TFTPredictor(n_features=n_features).to(self.device)
            self.tft.load_state_dict(torch.load(tft_path, map_location=self.device, weights_only=True))
            self.tft.eval()
            logger.info(f"TFT model loaded (n_features={n_features})")

        # CNN
        if cnn_path.exists():
            self.cnn = PatternCNN(n_features=n_features).to(self.device)
            self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
            self.cnn.eval()
            logger.info(f"CNN model loaded (n_features={n_features})")

        # LSTM
        if lstm_path.exists():
            self.lstm = RegimeLSTM(n_features=n_features).to(self.device)
            self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
            self.lstm.eval()
            logger.info(f"LSTM model loaded (n_features={n_features})")

        # Store detected value for predict_ticker to use
        self.n_features = n_features

        # Ensemble
        self.ensemble.load()
        self._loaded = True

    def predict_ticker(self, ticker: str) -> Optional[Dict]:
        """Generate prediction for a single ticker."""
        if not self._loaded:
            logger.error("Models not loaded. Call load_models() first.")
            return None

        # Fetch latest data
        df = self.fetcher.fetch_ticker(ticker, force_refresh=True)
        if df is None or len(df) < cfg.data.sequence_length + 10:
            logger.error(f"Insufficient data for {ticker}")
            return None

        # Compute features (all engines)
        df = self.tech.compute_all(df)
        df = self.structure.compute_all(df)
        df = self.price_action.compute_all(df)
        df = self.smc.compute_all(df)
        df = self.ict.compute_all(df)
        df = self.wyckoff.compute_all(df)
        df = self.elliott.compute_all(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Prepare input
        exclude = {"Ticker", "Category", "Date", "Datetime"}
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        # Get last sequence
        seq = df[feature_cols].values[-cfg.data.sequence_length:]
        if len(seq) < cfg.data.sequence_length:
            logger.error(f"Not enough data points for {ticker}")
            return None

        # Normalize
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        seq_scaled = scaler.fit_transform(seq)

        # Clip and sanitize scaled data
        seq_scaled = np.clip(seq_scaled, -10, 10)
        seq_scaled = np.nan_to_num(seq_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        # Match feature dimension to model's expected n_features
        actual_features = seq_scaled.shape[1]
        expected_features = self.n_features
        if actual_features != expected_features:
            if not getattr(self, '_feature_mismatch_warned', False):
                logger.warning(
                    f"Feature count mismatch: got {actual_features}, model expects {expected_features}. "
                    f"Selecting core features to match. (This warning will not repeat)"
                )
                self._feature_mismatch_warned = True

            if actual_features > expected_features:
                # Select the CORE features the model was trained on (OHLCV + derived)
                # Priority: Open, High, Low, Close, Volume, then first computed features
                core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                # Add derived features in order until we reach expected count
                derived_priority = ['Returns', 'Log_Returns', 'Volatility', 'ATR_14',
                                    'RSI_14', 'MACD_12_26_9', 'BB_Upper', 'SMA_20']
                selected_indices = []
                for col in core_cols:
                    if col in feature_cols:
                        selected_indices.append(feature_cols.index(col))
                for col in derived_priority:
                    if len(selected_indices) >= expected_features:
                        break
                    if col in feature_cols and feature_cols.index(col) not in selected_indices:
                        selected_indices.append(feature_cols.index(col))
                # Fill remaining with sequential columns if needed
                for i in range(len(feature_cols)):
                    if len(selected_indices) >= expected_features:
                        break
                    if i not in selected_indices:
                        selected_indices.append(i)
                seq_scaled = seq_scaled[:, selected_indices[:expected_features]]
            else:
                # Pad with zeros
                pad = np.zeros((seq_scaled.shape[0], expected_features - actual_features))
                seq_scaled = np.hstack([seq_scaled, pad])

        # To tensor
        x = torch.FloatTensor(seq_scaled).unsqueeze(0).to(self.device)

        # Run Strategy Confluence on latest bar's features
        strategy_result = self.confluence.analyze_dataframe(df)

        # Predict with each model
        with torch.no_grad():
            tft_pred = self.tft.predict(x) if self.tft else {"direction_probs": np.array([[0.33, 0.34, 0.33]]), "confidence": np.array([0.5])}
            cnn_pred = self.cnn.predict(x) if self.cnn else {"direction_probs": np.array([[0.33, 0.34, 0.33]]), "pattern_strength": np.array([0.5]), "predicted_pattern": ["unknown"]}
            lstm_pred = self.lstm.predict(x) if self.lstm else {"regime_probs": np.array([[0.25, 0.25, 0.25, 0.25]]), "predicted_regime": ["unknown"], "confidence": np.array([0.5])}

        # Ensemble (with strategy confluence)
        result = self.ensemble.predict(tft_pred, cnn_pred, lstm_pred, strategy_result=strategy_result)

        # Add context
        result["ticker"] = ticker
        result["current_price"] = float(df["Close"].iloc[-1])
        result["category"] = cfg.assets.get_category(ticker)
        result["timestamp"] = str(df.index[-1])
        result["n_features_used"] = len(feature_cols)
        
        # Store for outcome tracking
        result["features"] = seq_scaled
        result["regime"] = result.get("regime", "unknown")

        return result

    def predict_all(self) -> List[Dict]:
        """Generate predictions for all configured assets."""
        all_tickers = cfg.assets.stocks + cfg.assets.forex + cfg.assets.crypto + cfg.assets.commodities
        results = []
        for ticker in all_tickers:
            try:
                pred = self.predict_ticker(ticker)
                if pred:
                    results.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")
        return results

    def predict_category(self, category: str) -> List[Dict]:
        """Generate predictions for a specific category."""
        category_map = {
            "stocks": cfg.assets.stocks,
            "forex": cfg.assets.forex,
            "crypto": cfg.assets.crypto,
            "commodities": cfg.assets.commodities,
        }
        tickers = category_map.get(category, [])
        results = []
        for ticker in tickers:
            pred = self.predict_ticker(ticker)
            if pred:
                results.append(pred)
        return results
    
    def record_outcome(
        self,
        ticker: str,
        prediction: Dict,
        actual_return: float,
        profit: float,
        hours_elapsed: int = 24,
    ) -> Dict:
        """
        Record the outcome of a prediction and learn from it.
        
        This is the KEY method for continuous improvement.
        Call this after the prediction horizon to learn from outcomes.
        
        Args:
            ticker: Asset symbol
            prediction: The original prediction dict from predict_ticker()
            actual_return: Actual return after prediction horizon
            profit: P&L of the trade (signal * actual_return * position_size)
            hours_elapsed: Hours since prediction was made
            
        Returns:
            Learning update summary
        """
        signal = prediction.get("signal", [0])[0] if isinstance(prediction.get("signal"), np.ndarray) else prediction.get("signal", 0)
        confidence = prediction.get("confidence", [0.5])[0] if isinstance(prediction.get("confidence"), np.ndarray) else prediction.get("confidence", 0.5)
        features = prediction.get("features")
        regime = prediction.get("regime", "unknown")
        strategy_breakdown = prediction.get("strategy_breakdown", {})
        
        if features is None:
            return {"learned": False, "reason": "no_features_stored"}
        
        # Learn from all 3 models
        results = {}
        
        for model_type, learner in [
            ("tft", self.tft_learner),
            ("cnn", self.cnn_learner),
            ("lstm", self.lstm_learner),
        ]:
            try:
                result = learner.learn_from_trade(
                    features=features,
                    signal=signal,
                    confidence=confidence,
                    strategy_breakdown=strategy_breakdown,
                    outcome=actual_return,
                    profit=profit,
                    ticker=ticker,
                    regime=regime,
                )
                results[model_type] = result
            except Exception as e:
                logger.warning(f"Learning failed for {model_type}: {e}")
                results[model_type] = {"learned": False, "error": str(e)}
        
        # Get updated strategy weights
        strategy_tracker = get_strategy_tracker()
        updated_weights = strategy_tracker.get_weights(regime)
        
        return {
            "ticker": ticker,
            "prediction_signal": signal,
            "prediction_confidence": confidence,
            "actual_return": actual_return,
            "profit": profit,
            "was_correct": profit > 0,
            "learning_results": results,
            "updated_strategy_weights": updated_weights,
        }
    
    def get_learning_status(self) -> Dict:
        """Get current learning status and improvements."""
        return {
            "tft": self.tft_learner.get_learning_summary_for_ollama(),
            "cnn": self.cnn_learner.get_learning_summary_for_ollama(),
            "lstm": self.lstm_learner.get_learning_summary_for_ollama(),
            "strategy_tracker": get_strategy_tracker().get_summary(),
        }
