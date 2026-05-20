"""
Continuous Learner — Incremental Learning from Strategies
=========================================================
Enables the model to improve with each training session.
Uses strategy confluence as additional supervision signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, CHECKPOINT_DIR
from training.experience import get_experience_buffer, TradeExperience
from training.online_learner import get_online_learner
from training.curriculum import get_curriculum_manager
from training.adaptive_strategies import get_strategy_tracker
from features.strategy_confluence import StrategyConfluence


class StrategyLoss(nn.Module):
    """
    Loss function that incorporates strategy signals.
    Penalizes predictions that disagree with confluence.
    """
    
    def __init__(self, strategy_weight: float = 0.3):
        super().__init__()
        self.strategy_weight = strategy_weight
        self.base_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def forward(
        self,
        logits: torch.Tensor,           # [B, 3] direction predictions
        targets: torch.Tensor,           # [B] actual direction labels
        strategy_signals: torch.Tensor,  # [B] confluence signals (-1, 0, 1)
        strategy_confidence: torch.Tensor,  # [B] confluence confidence
    ) -> torch.Tensor:
        # Base classification loss
        base = self.base_loss(logits, targets)
        
        # Strategy alignment loss
        # Convert strategy signal to class: -1 -> 0, 0 -> 1, 1 -> 2
        strategy_target = (strategy_signals + 1).long()
        
        # Only apply strategy loss where confidence is high
        mask = strategy_confidence > 0.5
        
        if mask.sum() > 0:
            strat_loss = F.cross_entropy(
                logits[mask], 
                strategy_target[mask],
                reduction='mean'
            )
            return base + self.strategy_weight * strat_loss
        
        return base


class ContinuousLearner:
    """
    Manages continuous learning across training sessions.
    
    Features:
    - Loads previous best model and continues training
    - Uses experience buffer for replay learning
    - Incorporates strategy confluence as supervision
    - Tracks improvement over time
    - Generates learning summaries for Ollama
    """
    
    def __init__(self, model_type: str = "tft"):
        self.model_type = model_type
        self.experience_buffer = get_experience_buffer()
        self.confluence = StrategyConfluence()
        self.device = cfg.gpu.device
        
        # NEW: Online learning components
        self.online_learner = get_online_learner(model_type)
        self.curriculum = get_curriculum_manager()
        self.strategy_tracker = get_strategy_tracker()
        
        # Training history
        self.session_history: List[Dict] = []
        self.best_accuracy = 0.0
        self.improvement_rate = 0.0
        
        # Track if model is initialized for online learning
        self._online_initialized = False
    
    def load_previous_model(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """
        Load previous best model if exists.
        Returns (model, loaded) tuple.

        Uses strict=False so that a feature-count change (e.g. a new indicator
        was added to the pipeline) does NOT discard all previously learned
        weights.  Layers whose shapes still match are warm-started; only the
        input-projection rows / VSN heads that touch the new feature are
        reinitialised from scratch.
        """
        checkpoint_path = CHECKPOINT_DIR / f"{self.model_type}_best.pt"

        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                if missing or unexpected:
                    logger.info(
                        f"  Loaded previous {self.model_type} model (partial — feature count changed). "
                        f"Re-initialised {len(missing)} keys, ignored {len(unexpected)} stale keys."
                    )
                    logger.debug(f"  Missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
                    logger.debug(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
                else:
                    logger.info(f"  Loaded previous {self.model_type} model for incremental learning")

                return model, True
            except Exception as e:
                logger.warning(f"  Could not load previous model: {e}")

        return model, False
    
    def compute_strategy_targets(
        self,
        features: np.ndarray,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute strategy confluence signals for training data.
        
        Returns:
            signals: [N] confluence signals (-1, 0, 1)
            confidences: [N] confluence confidence (0-1)
        """
        signals = []
        confidences = []
        
        for i in range(len(features)):
            # Get last bar features
            last_bar = features[i][-1]  # Last timestep
            
            if len(last_bar) == len(feature_columns):
                feature_dict = dict(zip(feature_columns, last_bar))
                result = self.confluence.analyze(feature_dict)
                signals.append(result["signal"])
                confidences.append(result["confidence"])
            else:
                signals.append(0)
                confidences.append(0.0)
        
        return np.array(signals), np.array(confidences)
    
    def train_with_experience(
        self,
        model: nn.Module,
        train_data: Dict,
        val_data: Dict,
        feature_columns: List[str],
        base_trainer,  # ModelTrainer instance
    ) -> Tuple[nn.Module, Dict]:
        """
        Train model with experience replay and strategy supervision.
        
        Args:
            model: The model to train
            train_data: Training data dict
            val_data: Validation data dict
            feature_columns: List of feature names
            base_trainer: Base trainer for core training loop
            
        Returns:
            (trained_model, session_summary)
        """
        session_start = datetime.now()
        session_id = len(self.session_history) + 1
        
        logger.info(f"\n  📚 Continuous Learning Session #{session_id}")
        logger.info(f"  Previous experiences: {len(self.experience_buffer.experiences)}")
        
        # Load previous model
        model, was_loaded = self.load_previous_model(model)
        
        # Get strategy targets for training data
        strategy_signals, strategy_conf = self.compute_strategy_targets(
            train_data["X"], feature_columns
        )
        
        # Experience replay — always runs, but caps total memory usage.
        #
        # Strategy:
        #   1. If the main dataset exceeds MAX_SAMPLES, subsample it first so the
        #      combined (main + replay) footprint stays bounded.
        #   2. Add up to 20% replay samples on top of the (possibly subsampled) set.
        #
        # This replaces the old "skip replay for large datasets" approach, which
        # meant the model never benefited from past experiences after the dataset
        # grew beyond 100K rows.
        MAX_SAMPLES = 100_000          # Hard cap on main-dataset rows fed to training
        REPLAY_RATIO = 0.20            # Replay = 20 % of (capped) main dataset size

        n_main = len(train_data["X"])

        # ── Step 1: subsample main dataset if too large ──────────────────────
        if n_main > MAX_SAMPLES:
            keep_idx = np.random.choice(n_main, MAX_SAMPLES, replace=False)
            train_data = {
                "X":                train_data["X"][keep_idx],
                "y_regression":     train_data["y_regression"][keep_idx],
                "y_classification": train_data["y_classification"][keep_idx],
            }
            logger.info(
                f"  Large dataset ({n_main:,} rows): subsampled to {MAX_SAMPLES:,} "
                f"for memory-safe training."
            )

        # ── Step 2: mix in experience replay ─────────────────────────────────
        exp_features, exp_signals, exp_labels = self.experience_buffer.get_features_and_labels(500)

        if exp_features is not None and len(exp_features) > 10:
            # Filter to experiences whose feature dimension matches current data
            expected_n_features = train_data["X"].shape[-1]
            valid_mask = np.array([f.shape[-1] == expected_n_features for f in exp_features])
            if valid_mask.sum() > 10:
                exp_features = exp_features[valid_mask]
                exp_signals = exp_signals[valid_mask]
            else:
                exp_features = None  # Skip replay if too few valid experiences

        if exp_features is not None and len(exp_features) > 10:
            n_replay = min(len(exp_features), int(len(train_data["X"]) * REPLAY_RATIO))
            if n_replay > 0:
                indices = np.random.choice(len(exp_features), n_replay, replace=False)
                train_data["X"] = np.concatenate(
                    [train_data["X"], exp_features[indices]], axis=0
                )
                # Keep signals in raw form [-1, 0, 1] — the trainer shifts by +1
                exp_cls = exp_signals[indices].astype(np.int64)
                train_data["y_classification"] = np.concatenate(
                    [train_data["y_classification"], exp_cls], axis=0
                )
                train_data["y_regression"] = np.concatenate(
                    [train_data["y_regression"], np.zeros(n_replay)], axis=0
                )
                logger.info(
                    f"  Experience replay: +{n_replay} samples "
                    f"({n_replay / len(train_data['X']):.0%} of training set)"
                )
        
        # Run training — optionally with WALK-FORWARD VALIDATION
        n_features = train_data["X"].shape[-1]
        
        if cfg.training.use_walk_forward:
            # Walk-forward: combine train+val, then split via expanding window
            from training.walk_forward import WalkForwardValidator
            
            combined_X = np.concatenate([train_data["X"], val_data["X"]], axis=0)
            combined_y_reg = np.concatenate([train_data["y_regression"], val_data["y_regression"]], axis=0)
            combined_y_cls = np.concatenate([train_data["y_classification"], val_data["y_classification"]], axis=0)
            
            wfv = WalkForwardValidator()
            model, wf_result = wfv.run_pipeline(
                combined_X, combined_y_reg, combined_y_cls,
                base_trainer, self.model_type, n_features, feature_columns
            )
            logger.info(
                f"  Walk-forward: {len(wf_result['folds'])} folds, "
                f"avg val loss: {wf_result['avg_val_loss']:.6f}"
            )
        else:
            # Simple training (no walk-forward)
            if self.model_type == "tft":
                model = base_trainer.train_tft(train_data, val_data, n_features, feature_columns)
            elif self.model_type == "cnn":
                model = base_trainer.train_cnn(train_data, val_data, n_features, feature_columns)
            elif self.model_type == "lstm":
                model = base_trainer.train_lstm(train_data, val_data, n_features, feature_columns)
        
        # Calculate session results
        session_result = self._evaluate_session(model, val_data, feature_columns)
        
        # Update experience buffer with new trades from validation
        self._add_val_experiences(val_data, feature_columns, session_result)
        
        # Track improvement
        prev_best = self.best_accuracy
        self.best_accuracy = max(self.best_accuracy, session_result["val_accuracy"])
        self.improvement_rate = (self.best_accuracy - prev_best) / max(prev_best, 0.01)
        
        # Build session summary
        session_summary = {
            "session_id": session_id,
            "timestamp": session_start.isoformat(),
            "model_type": self.model_type,
            "was_incremental": was_loaded,
            "previous_experiences": len(self.experience_buffer.experiences),
            "val_accuracy": session_result["val_accuracy"],
            "val_loss": session_result["val_loss"],
            "improvement": self.improvement_rate,
            "best_accuracy_so_far": self.best_accuracy,
            "strategy_agreement": session_result.get("strategy_agreement", 0),
            "win_rate": session_result.get("win_rate", 0),
            "patterns_learned": session_result.get("patterns_learned", []),
        }
        
        self.session_history.append(session_summary)
        
        # Update experience buffer stats
        self.experience_buffer.update_session_stats({
            "win_rate": session_result.get("win_rate", 0),
            "strategy_stats": session_result.get("strategy_stats", {}),
        })
        
        # NEW: Initialize online learner with trained model
        self.online_learner.initialize(model)
        self._online_initialized = True
        
        logger.info(f"  Session complete: Val Acc = {session_result['val_accuracy']:.1%}")
        if self.improvement_rate > 0:
            logger.success(f"  Model improved by {self.improvement_rate:.1%}")
        
        return model, session_summary
    
    def _evaluate_session(
        self,
        model: nn.Module,
        val_data: Dict,
        feature_columns: List[str]
    ) -> Dict:
        """Evaluate model performance on validation data."""
        model.eval()
        
        X = torch.FloatTensor(val_data["X"]).to(self.device)
        y_cls = torch.LongTensor(val_data["y_classification"] + 1).clamp(0, 2).to(self.device)
        
        correct = 0
        total = 0
        strategy_agreements = 0
        strategy_total = 0
        wins = 0
        losses = 0
        
        patterns_seen = set()
        
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y_cls[i:i+batch_size]
                
                out = model(batch_X)
                # Handle different model output keys:
                # TFT/CNN use "direction", RegimeLSTM uses "regime_logits"
                if "direction" in out:
                    preds = out["direction"].argmax(dim=-1)
                elif "regime_logits" in out:
                    preds = out["regime_logits"].argmax(dim=-1)
                elif "pattern_class" in out:
                    preds = out["pattern_class"].argmax(dim=-1)
                else:
                    # Fallback: use the first tensor that looks like class logits
                    for key, val in out.items():
                        if val.dim() >= 1 and val.shape[-1] >= 3:
                            preds = val.argmax(dim=-1)
                            break
                    else:
                        logger.warning(f"  Model output keys {list(out.keys())} — no classification head found, skipping batch")
                        continue
                
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
                
                # Check strategy agreement
                for j in range(len(batch_X)):
                    idx = i + j
                    if idx < len(val_data["X"]):
                        last_bar = val_data["X"][idx][-1]
                        if len(last_bar) == len(feature_columns):
                            feature_dict = dict(zip(feature_columns, last_bar))
                            strat_result = self.confluence.analyze(feature_dict)
                            
                            pred_signal = preds[j].item() - 1  # Convert back to -1, 0, 1
                            if strat_result["signal"] == pred_signal and strat_result["confidence"] > 0.5:
                                strategy_agreements += 1
                            
                            # Track patterns
                            for strat, data in strat_result.get("breakdown", {}).items():
                                if data["confidence"] > 0.6:
                                    patterns_seen.add(f"{strat}_{data['signal']}")
                            
                            strategy_total += 1
                            
                            # Simulate trade outcome
                            actual_return = val_data["y_regression"][idx]
                            if pred_signal != 0:
                                pnl = pred_signal * actual_return
                                if pnl > 0:
                                    wins += 1
                                else:
                                    losses += 1
        
        return {
            "val_accuracy": correct / max(total, 1),
            "val_loss": 0,  # Computed in base trainer
            "strategy_agreement": strategy_agreements / max(strategy_total, 1),
            "win_rate": wins / max(wins + losses, 1),
            "patterns_learned": list(patterns_seen)[:10],
            "strategy_stats": {},
        }
    
    def _add_val_experiences(
        self,
        val_data: Dict,
        feature_columns: List[str],
        session_result: Dict
    ):
        """Add validation trades to experience buffer."""
        for i in range(len(val_data["X"])):
            # Sample 20% of validation data
            if np.random.random() > 0.2:
                continue
            
            last_bar = val_data["X"][i][-1]
            if len(last_bar) != len(feature_columns):
                continue
            
            feature_dict = dict(zip(feature_columns, last_bar))
            strat_result = self.confluence.analyze(feature_dict)
            
            actual_return = val_data["y_regression"][i]
            pred_signal = val_data["y_classification"][i]  # -1, 0, 1
            
            # Simulate trade
            pnl = pred_signal * actual_return
            
            exp = TradeExperience(
                ticker="VAL",
                features=val_data["X"][i],
                signal=pred_signal,
                confidence=0.5,  # Placeholder
                strategy_breakdown=strat_result.get("breakdown", {}),
                outcome=actual_return,
                profit=pnl,
            )
            
            self.experience_buffer.add(exp)
    
    def learn_from_trade(
        self,
        features: np.ndarray,
        signal: int,
        confidence: float,
        strategy_breakdown: Dict,
        outcome: float,
        profit: float,
        ticker: str = "UNKNOWN",
        regime: str = "unknown",
    ) -> Dict:
        """
        Learn from a single trade outcome in real-time.
        This is the key method for continuous improvement.
        
        Args:
            features: Input features [seq_len, n_features]
            signal: Predicted signal (-1, 0, 1)
            confidence: Prediction confidence
            strategy_breakdown: Strategy analysis for this trade
            outcome: Actual return after prediction
            profit: P&L of the trade
            ticker: Asset symbol
            regime: Market regime at time of trade
            
        Returns:
            Learning update summary
        """
        if not self._online_initialized:
            return {"learned": False, "reason": "online_learner_not_initialized"}
        
        # 1. Learn from outcome (online gradient update)
        online_result = self.online_learner.learn_from_outcome(
            features=features,
            signal=signal,
            confidence=confidence,
            strategy_breakdown=strategy_breakdown,
            outcome=outcome,
            profit=profit,
            ticker=ticker,
        )
        
        # 2. Update adaptive strategy weights
        self.strategy_tracker.record_outcome(
            strategy_breakdown=strategy_breakdown,
            actual_outcome=outcome,
            profit=profit,
            regime=regime,
        )
        
        # 3. Track if this was a hard example
        if profit < 0 and confidence > 0.6:
            self.curriculum.add_hard_example(
                features=features,
                predicted_signal=signal,
                actual_outcome=outcome,
                profit=profit,
                confidence=confidence,
                strategy_breakdown=strategy_breakdown,
            )
        
        return {
            "learned": online_result.get("learned", False),
            "online_update": online_result,
            "strategy_weights_updated": True,
            "hard_example_added": profit < 0 and confidence > 0.6,
        }
    
    def get_learning_summary_for_ollama(self) -> Dict:
        """
        Generate a summary of what the model learned for Ollama to frame.
        """
        buffer_summary = self.experience_buffer.get_learning_summary()
        
        # Recent session summaries
        recent_sessions = self.session_history[-5:] if self.session_history else []
        
        # Calculate trends
        if len(self.session_history) >= 2:
            recent_acc = np.mean([s["val_accuracy"] for s in self.session_history[-3:]])
            early_acc = np.mean([s["val_accuracy"] for s in self.session_history[:3]])
            trend = "improving" if recent_acc > early_acc + 0.02 else "stable" if abs(recent_acc - early_acc) < 0.02 else "needs_attention"
        else:
            trend = "starting"
        
        # Best performing strategies
        strategy_patterns = buffer_summary.get("best_patterns", [])
        strategy_perf = buffer_summary.get("best_strategies", [])
        
        # Get online learning stats
        online_stats = self.online_learner.get_learning_stats()
        
        # Get curriculum stats
        curriculum_summary = self.curriculum.get_curriculum_summary()
        
        # Get adaptive strategy weights
        strategy_summary = self.strategy_tracker.get_summary()
        
        return {
            "total_training_sessions": len(self.session_history),
            "current_best_accuracy": self.best_accuracy,
            "improvement_trend": trend,
            "recent_accuracy": recent_sessions[-1]["val_accuracy"] if recent_sessions else 0,
            "total_trades_learned": buffer_summary["total_experiences"],
            "overall_win_rate": buffer_summary["win_rate"],
            "best_strategy_combinations": [
                {"pattern": p, "avg_profit": f"{ap:.2%}", "count": c}
                for p, ap, c in strategy_patterns[:3]
            ],
            "top_strategies": [
                {"name": s, "profit": f"${p:.2f}", "win_rate": f"{wr:.1%}"}
                for s, p, wr in strategy_perf[:3]
            ],
            "patterns_recognized": list(set(
                p for s in self.session_history 
                for p in s.get("patterns_learned", [])
            ))[:10],
            "strategy_agreement_rate": np.mean([s.get("strategy_agreement", 0) for s in recent_sessions]) if recent_sessions else 0,
            "last_session": recent_sessions[-1] if recent_sessions else None,
            # NEW: Online learning stats
            "online_learning": {
                "total_updates": online_stats["total_online_updates"],
                "loss_trend": online_stats["loss_trend"],
                "hard_examples": online_stats["hard_examples_tracked"],
            },
            # NEW: Curriculum stats
            "curriculum": {
                "hard_examples": curriculum_summary["total_hard_examples"],
                "weak_patterns": curriculum_summary["top_weak_patterns"][:3],
            },
            # NEW: Adaptive strategy weights
            "adaptive_weights": strategy_summary["current_weights"],
        }


# Module-level learners for each model type
_tft_learner = None
_cnn_learner = None
_lstm_learner = None


def get_continuous_learner(model_type: str) -> ContinuousLearner:
    """Get or create a continuous learner for the specified model type."""
    global _tft_learner, _cnn_learner, _lstm_learner
    
    if model_type == "tft":
        if _tft_learner is None:
            _tft_learner = ContinuousLearner("tft")
        return _tft_learner
    elif model_type == "cnn":
        if _cnn_learner is None:
            _cnn_learner = ContinuousLearner("cnn")
        return _cnn_learner
    elif model_type == "lstm":
        if _lstm_learner is None:
            _lstm_learner = ContinuousLearner("lstm")
        return _lstm_learner
    else:
        return ContinuousLearner(model_type)
