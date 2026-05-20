"""
Pipeline — End-to-End Orchestration
=====================================
Data fetch → Features → Train → Evaluate → Predict
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import cfg, CHECKPOINT_DIR, RESULTS_DIR
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from data.feature_store import FeatureStore
from features.technical import TechnicalFeatures
from features.market_structure import MarketStructure
from features.price_action import PriceAction
from features.smc import SmartMoneyConcepts
from features.ict import ICTConcepts
from features.wyckoff import WyckoffAnalysis
from features.elliott_wave import ElliottWave
from features.sentiment import SentimentAnalyzer
from features.feature_selector import FeatureSelector
from features.temporal import TemporalFeatures
from features.multi_timeframe import MultiTimeframeFeatures
from features.cross_asset import CrossAssetFeatures
from features.volume_profile import VolumeProfileFeatures
from features.options_flow import OptionsFlowFeatures
from training.trainer import ModelTrainer
from training.continuous_learner import get_continuous_learner
from training.experience import get_experience_buffer
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics


class TradingPipeline:
    """End-to-end pipeline for training and evaluating the trading AI."""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.preprocessor = DataPreprocessor()
        self.feature_store = FeatureStore()
        # Original 8 strategies
        self.tech = TechnicalFeatures()
        self.structure = MarketStructure()
        self.price_action = PriceAction()
        self.smc = SmartMoneyConcepts()
        self.ict = ICTConcepts()
        self.wyckoff = WyckoffAnalysis()
        self.elliott = ElliottWave()
        self.sentiment = SentimentAnalyzer()
        # New feature modules
        self.temporal = TemporalFeatures()
        self.multi_tf = MultiTimeframeFeatures()
        self.cross_asset = CrossAssetFeatures()
        self.volume_profile = VolumeProfileFeatures()
        self.options_flow = OptionsFlowFeatures()
        # Utilities
        self.feature_selector = FeatureSelector()
        self.trainer = ModelTrainer()
        self.backtester = BacktestEngine()
        # Continuous learning components
        self.tft_learner = get_continuous_learner("tft")
        self.cnn_learner = get_continuous_learner("cnn")
        self.lstm_learner = get_continuous_learner("lstm")
        self.experience_buffer = get_experience_buffer()

    def run_full_pipeline(
        self,
        categories: List[str] = None,
        tickers: List[str] = None,
        skip_fetch: bool = False,
        timeframe: str = None,
    ):
        """
        Run the complete pipeline: fetch → features → train → backtest.
        
        Args:
            categories: List of asset categories to train on.
            tickers: Specific tickers (overrides categories).
            skip_fetch: Skip data fetching (use cached data).
            timeframe: Intraday timeframe (1m, 3m, 5m, 15m, 1h, 1d).
        """
        start = time.time()
        timeframe = timeframe or cfg.data.primary_timeframe
        
        logger.info("=" * 60)
        logger.info("  TRADING AI — Full Pipeline")
        logger.info("=" * 60)
        logger.info(f"  Timeframe: {timeframe} (Intraday Trading)")
        logger.info(cfg.summary())

        # Step 1: Fetch Data
        logger.info("\n📥 Step 1: Fetching market data...")
        if tickers:
            raw_data = {}
            for t in tickers:
                df = self.fetcher.fetch_ticker(t, timeframe=timeframe, force_refresh=not skip_fetch)
                if df is not None:
                    raw_data[t] = df
        elif categories:
            raw_data = {}
            for cat in categories:
                raw_data.update(self.fetcher.fetch_category(cat, timeframe=timeframe))
        else:
            raw_data = self.fetcher.fetch_all(timeframe=timeframe)

        if not raw_data:
            logger.error("No data fetched. Aborting.")
            return
        logger.success(f"  Fetched {len(raw_data)} assets")

        # Step 1.5: Load cross-asset reference data
        logger.info("\n📊 Step 1.5: Loading cross-asset references...")
        self.cross_asset.load_references(self.fetcher, timeframe)

        # Step 1.6: Fetch multi-timeframe data for each ticker
        logger.info("\n📊 Step 1.6: Fetching multi-timeframe data...")
        htf_data = {}  # {ticker: {"15m": df, "1h": df}}
        higher_tfs = getattr(cfg.data, 'higher_timeframes', ["15m", "1h"])
        for ticker in raw_data:
            try:
                mtf = {}
                for htf in higher_tfs:
                    if htf != timeframe:
                        htf_df = self.fetcher.fetch_ticker(ticker, timeframe=htf)
                        if htf_df is not None:
                            mtf[htf] = htf_df
                if mtf:
                    htf_data[ticker] = mtf
            except Exception as e:
                logger.debug(f"  MTF fetch failed for {ticker}: {e}")

        # Step 2: Feature Engineering
        logger.info("\n🔧 Step 2: Computing features (8 strategies + new modules)...")
        featured_data = {}
        for ticker, df in raw_data.items():
            try:
                # All 8 original strategy features
                featured = self.tech.compute_all(df)
                featured = self.structure.compute_all(featured)
                featured = self.price_action.compute_all(featured)
                featured = self.smc.compute_all(featured)
                featured = self.ict.compute_all(featured)
                featured = self.wyckoff.compute_all(featured)
                featured = self.elliott.compute_all(featured)
                # NEW: Real sentiment from news headlines (Issue #1)
                featured = self.sentiment.create_sentiment_features_from_news(featured, ticker)
                # NEW: Time-of-day / seasonality features (Issue #6)
                featured = self.temporal.compute_all(featured)
                # NEW: Volume profile features (Issue #7)
                featured = self.volume_profile.compute_all(featured)
                # NEW: Cross-asset correlation features (Issue #5)
                featured = self.cross_asset.compute_all(featured, ticker)
                # NEW: Multi-timeframe features (Issue #4)
                if ticker in htf_data:
                    featured = self.multi_tf.compute_mtf_features(featured, htf_data[ticker])
                # NEW: Options flow (Issue #10, US stocks only)
                featured = self.options_flow.compute_all(featured, ticker)
                featured_data[ticker] = featured
            except Exception as e:
                logger.warning(f"  Feature computation failed for {ticker}: {e}")

        logger.success(f"  Features computed for {len(featured_data)} assets")

        # Store features
        def compute_all_features(df):
            df = self.tech.compute_all(df)
            df = self.structure.compute_all(df)
            df = self.price_action.compute_all(df)
            df = self.smc.compute_all(df)
            df = self.ict.compute_all(df)
            df = self.wyckoff.compute_all(df)
            df = self.elliott.compute_all(df)
            df = self.sentiment.create_sentiment_features(df)
            df = self.temporal.compute_all(df)
            df = self.volume_profile.compute_all(df)
            return df
        self.feature_store.compute_and_store(raw_data, compute_all_features)

        # Step 2.5: Feature Selection (remove noisy/redundant features)
        logger.info("\n🎯 Step 2.5: Feature selection...")
        # Use the first well-populated asset for fitting the selector
        first_key = next(iter(featured_data))
        sample_df = featured_data[first_key].copy()
        sample_df = sample_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample_df) > 100:
            # Compute target for selection
            sample_df["Target"] = sample_df["Close"].pct_change(cfg.data.prediction_horizon).shift(-cfg.data.prediction_horizon)
            sample_df = sample_df.dropna(subset=["Target"])
            selected = self.feature_selector.fit_transform(sample_df)
            # Apply selection to all featured data
            for ticker in featured_data:
                available = [c for c in selected if c in featured_data[ticker].columns]
                keep_cols = list(set(available + ["Open", "High", "Low", "Close", "Volume"]))
                keep_cols = [c for c in keep_cols if c in featured_data[ticker].columns]
                featured_data[ticker] = featured_data[ticker][keep_cols]

        # Step 3: Preprocess
        logger.info("\n📊 Step 3: Preprocessing datasets...")
        datasets = self.preprocessor.prepare_dataset(raw_data, featured_data)
        if not datasets:
            logger.error("No datasets prepared. Aborting.")
            return
        logger.success(f"  Prepared {len(datasets)} datasets")

        # Step 4: Train models (on combined data)
        logger.info("\n🧠 Step 4: Training models...")
        combined = self._combine_datasets(datasets)
        if combined is None:
            logger.error("Could not combine datasets")
            return

        n_features = combined["n_features"]
        
        # Sanitize combined data — ensure no NaN/inf reaches training
        # Use memory-efficient in-place sanitization to avoid OOM on large arrays
        def sanitize_array_inplace(arr, nan_val=0.0, posinf_val=10.0, neginf_val=-10.0, chunk_size=10000):
            """Sanitize array in-place without allocating large boolean masks."""
            # Handle read-only arrays by working on a writeable view
            if not arr.flags.writeable:
                arr = arr.copy()
            flat = arr.ravel()
            for i in range(0, len(flat), chunk_size):
                chunk = flat[i:i+chunk_size]
                nan_mask = np.isnan(chunk)
                chunk[nan_mask] = nan_val
                posinf_mask = np.isposinf(chunk)
                chunk[posinf_mask] = posinf_val
                neginf_mask = np.isneginf(chunk)
                chunk[neginf_mask] = neginf_val
            return arr
        
        for split in ["train", "val", "test"]:
            combined[split]["X"] = sanitize_array_inplace(combined[split]["X"], nan_val=0.0, posinf_val=10.0, neginf_val=-10.0)
            combined[split]["y_regression"] = sanitize_array_inplace(combined[split]["y_regression"], nan_val=0.0, posinf_val=1.0, neginf_val=-1.0)
            combined[split]["y_classification"] = sanitize_array_inplace(combined[split]["y_classification"], nan_val=0.0, posinf_val=0.0, neginf_val=0.0)
        
        logger.info(f"  Combined dataset: {combined['train']['X'].shape[0]} train, "
                     f"{combined['val']['X'].shape[0]} val, "
                     f"{combined['test']['X'].shape[0]} test samples, "
                     f"{n_features} features")

        # Train TFT with continuous learning
        logger.info("\n  --- Training TFT (Temporal Fusion Transformer) ---")
        from models.tft_predictor import TFTPredictor
        tft_model = TFTPredictor(n_features=n_features)
        tft_model, tft_session = self.tft_learner.train_with_experience(
            tft_model, combined["train"], combined["val"], 
            combined.get("feature_columns", []), self.trainer
        )

        # Train CNN with continuous learning
        logger.info("\n  --- Training Pattern CNN ---")
        from models.pattern_cnn import PatternCNN
        cnn_model = PatternCNN(n_features=n_features)
        cnn_model, cnn_session = self.cnn_learner.train_with_experience(
            cnn_model, combined["train"], combined["val"],
            combined.get("feature_columns", []), self.trainer
        )

        # Train LSTM with continuous learning
        logger.info("\n  --- Training Regime LSTM ---")
        from models.regime_lstm import RegimeLSTM
        lstm_model = RegimeLSTM(n_features=n_features)
        lstm_model, lstm_session = self.lstm_learner.train_with_experience(
            lstm_model, combined["train"], combined["val"],
            combined.get("feature_columns", []), self.trainer
        )

        # Step 5: Backtest on test set
        logger.info("\n📈 Step 5: Backtesting...")
        backtest_result = self._run_backtest(tft_model, cnn_model, lstm_model, combined, n_features)

        # Step 5.5: Monte Carlo validation (Issue #12)
        if getattr(cfg, 'backtest', None) and getattr(cfg.backtest, 'run_monte_carlo', False):
            logger.info("\n🎲 Step 5.5: Monte Carlo Validation...")
            try:
                from backtest.monte_carlo import MonteCarloValidator
                mc = MonteCarloValidator(n_simulations=cfg.backtest.monte_carlo_simulations)
                if backtest_result:
                    mc.run(
                        backtest_result.get('_signals', np.array([])),
                        backtest_result.get('_prices', np.array([])),
                        backtest_result.get('_confidences', np.array([])),
                    )
            except Exception as e:
                logger.warning(f"  Monte Carlo failed: {e}")

        # Step 6: Generate AI Training Summary with Ollama
        logger.info("\n🤖 Step 6: Generating AI Training Summary...")
        training_summary = self._generate_training_summary(tft_session, cnn_session, lstm_session)
        
        elapsed = time.time() - start
        logger.success(f"\n✅ Pipeline complete in {elapsed/60:.1f} minutes")
        logger.info(f"  Models saved to: {CHECKPOINT_DIR}")
        logger.info(f"  Results saved to: {RESULTS_DIR}")
        
        return training_summary

    def _combine_datasets(self, datasets: Dict) -> Optional[dict]:
        """Combine multiple ticker datasets into one training set."""
        all_train_X, all_train_yr, all_train_yc = [], [], []
        all_val_X, all_val_yr, all_val_yc = [], [], []
        all_test_X, all_test_yr, all_test_yc = [], [], []
        n_features = None
        feature_columns = None

        for ticker, ds in datasets.items():
            if n_features is None:
                n_features = ds["n_features"]
                feature_columns = ds.get("feature_columns", [])
            elif ds["n_features"] != n_features:
                continue

            all_train_X.append(ds["train"]["X"])
            all_train_yr.append(ds["train"]["y_regression"])
            all_train_yc.append(ds["train"]["y_classification"])
            all_val_X.append(ds["val"]["X"])
            all_val_yr.append(ds["val"]["y_regression"])
            all_val_yc.append(ds["val"]["y_classification"])
            all_test_X.append(ds["test"]["X"])
            all_test_yr.append(ds["test"]["y_regression"])
            all_test_yc.append(ds["test"]["y_classification"])

        if not all_train_X:
            return None

        return {
            "train": {
                "X": np.concatenate(all_train_X, axis=0),
                "y_regression": np.concatenate(all_train_yr, axis=0),
                "y_classification": np.concatenate(all_train_yc, axis=0),
            },
            "val": {
                "X": np.concatenate(all_val_X, axis=0),
                "y_regression": np.concatenate(all_val_yr, axis=0),
                "y_classification": np.concatenate(all_val_yc, axis=0),
            },
            "test": {
                "X": np.concatenate(all_test_X, axis=0),
                "y_regression": np.concatenate(all_test_yr, axis=0),
                "y_classification": np.concatenate(all_test_yc, axis=0),
            },
            "n_features": n_features,
            "feature_columns": feature_columns or [],
        }

    def _run_backtest(self, tft_model, cnn_model, lstm_model, combined, n_features):
        """Run backtest with full strategy confluence on every trade."""
        import torch
        from models.ensemble import EnsembleModel
        from features.strategy_confluence import StrategyConfluence

        test_X = combined["test"]["X"]
        test_y = combined["test"]["y_regression"]
        feature_columns = combined.get("feature_columns", [])

        if len(test_X) == 0:
            logger.warning("No test data for backtesting")
            return

        # Initialize
        device = cfg.gpu.device
        ensemble = EnsembleModel()
        confluence = StrategyConfluence(min_agreement=3, min_confidence=0.50)
        signals = []
        confidences = []

        logger.info(f"  Running backtest with strategy confluence on {len(test_X)} samples...")

        batch_size = 32
        for i in range(0, len(test_X), batch_size):
            batch = torch.FloatTensor(test_X[i:i+batch_size]).to(device)
            with torch.no_grad():
                tft_pred = tft_model.predict(batch) if tft_model else {}
                cnn_pred = cnn_model.predict(batch) if cnn_model else {}
                lstm_pred = lstm_model.predict(batch) if lstm_model else {}

            for j in range(len(batch)):
                tp = {k: v[j:j+1] if isinstance(v, np.ndarray) else v for k, v in tft_pred.items()}
                cp = {k: (v[j:j+1] if isinstance(v, np.ndarray) else [v[j]] if isinstance(v, list) else v) for k, v in cnn_pred.items()}
                lp = {k: (v[j:j+1] if isinstance(v, np.ndarray) else [v[j]] if isinstance(v, list) else v) for k, v in lstm_pred.items()}

                # Run Strategy Confluence on this sample's features
                # Use the LAST timestep of the sequence (most recent bar)
                strategy_result = None
                if feature_columns and (i + j) < len(test_X):
                    last_bar = test_X[i + j][-1]  # Last timestep in sequence
                    if len(last_bar) == len(feature_columns):
                        feature_dict = dict(zip(feature_columns, last_bar))
                        strategy_result = confluence.analyze(feature_dict)

                result = ensemble.predict(tp, cp, lp, strategy_result=strategy_result)
                signals.append(result["signal"][0])
                confidences.append(result["confidence"][0])

        signals = np.array(signals)
        confidences = np.array(confidences)

        # Log signal distribution
        n_buy = (signals == 1).sum()
        n_sell = (signals == -1).sum()
        n_hold = (signals == 0).sum()
        logger.info(f"  Signals: {n_buy} BUY, {n_sell} SELL, {n_hold} HOLD")
        logger.info(f"  Avg confidence: {confidences.mean():.1%} (min: {confidences.min():.1%}, max: {confidences.max():.1%})")

        # Use actual returns as proxy for price movement
        prices = np.cumprod(1 + test_y) * 100
        bt_result = self.backtester.run(signals, prices, confidences)
        logger.info(f"\n  Backtest: {bt_result['n_trades']} trades, "
                     f"Final equity: ${bt_result['final_equity']:,.2f}")

        # Store for Monte Carlo access
        bt_result["_signals"] = signals
        bt_result["_prices"] = prices
        bt_result["_confidences"] = confidences
        return bt_result

    def _generate_training_summary(self, tft_session: Dict, cnn_session: Dict, lstm_session: Dict) -> Dict:
        """
        Generate AI-framed summary of what the models learned.
        Uses Ollama to convert training metrics into natural language.
        """
        from inference.ollama_chat import get_chat_service
        
        chat_service = get_chat_service()
        
        # Aggregate learning summaries from all models
        tft_learning = self.tft_learner.get_learning_summary_for_ollama()
        cnn_learning = self.cnn_learner.get_learning_summary_for_ollama()
        lstm_learning = self.lstm_learner.get_learning_summary_for_ollama()
        
        # Combined summary
        combined_summary = {
            "total_training_sessions": tft_learning.get("total_training_sessions", 0),
            "current_best_accuracy": max(
                tft_learning.get("current_best_accuracy", 0),
                cnn_learning.get("current_best_accuracy", 0),
                lstm_learning.get("current_best_accuracy", 0)
            ),
            "improvement_trend": tft_learning.get("improvement_trend", "unknown"),
            "total_trades_learned": self.experience_buffer.get_learning_summary().get("total_experiences", 0),
            "overall_win_rate": self.experience_buffer.get_learning_summary().get("win_rate", 0),
            "best_strategy_combinations": tft_learning.get("best_strategy_combinations", []),
            "top_strategies": tft_learning.get("top_strategies", []),
            "patterns_recognized": list(set(
                tft_learning.get("patterns_recognized", []) +
                cnn_learning.get("patterns_recognized", []) +
                lstm_learning.get("patterns_recognized", [])
            ))[:10],
            "strategy_agreement_rate": np.mean([
                tft_learning.get("strategy_agreement_rate", 0),
                cnn_learning.get("strategy_agreement_rate", 0),
                lstm_learning.get("strategy_agreement_rate", 0)
            ]),
            "model_performance": {
                "TFT": {
                    "accuracy": tft_session.get("val_accuracy", 0),
                    "improvement": tft_session.get("improvement", 0),
                },
                "CNN": {
                    "accuracy": cnn_session.get("val_accuracy", 0),
                    "improvement": cnn_session.get("improvement", 0),
                },
                "LSTM": {
                    "accuracy": lstm_session.get("val_accuracy", 0),
                    "improvement": lstm_session.get("improvement", 0),
                }
            }
        }
        
        # Generate AI commentary
        ai_summary = chat_service.frame_training_summary(combined_summary)
        
        logger.info(f"\n  📝 AI Training Summary (using 8 strategies: Technical, Market Structure, Price Action, SMC, ICT, Wyckoff, Elliott Wave, Sentiment):")
        logger.info(f"  {ai_summary}")
        
        # Print model-specific insights
        for model_name, perf in combined_summary["model_performance"].items():
            if perf["accuracy"] > 0:
                logger.info(f"  {model_name}: {perf['accuracy']:.1%} accuracy, {perf['improvement']:+.1%} improvement")
        
        # Print top strategies
        if combined_summary["top_strategies"]:
            logger.info(f"\n  🎯 Top Performing Strategies:")
            for strat in combined_summary["top_strategies"][:3]:
                logger.info(f"    • {strat['name']}: {strat['win_rate']} win rate, {strat['profit']} profit")
        
        # Print patterns learned
        if combined_summary["patterns_recognized"]:
            logger.info(f"\n  🔍 Patterns Learned: {', '.join(combined_summary['patterns_recognized'][:5])}")
        
        return {
            **combined_summary,
            "ai_commentary": ai_summary,
        }
