"""
Experience Replay Buffer — Learning from Past Trades
======================================================
Stores successful/failed trades with strategy context.
Enables the model to learn from its own trading history.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHECKPOINT_DIR


class TradeExperience:
    """Single trade experience with strategy context."""
    
    def __init__(
        self,
        ticker: str,
        features: np.ndarray,
        signal: int,
        confidence: float,
        strategy_breakdown: Dict,
        outcome: float,  # Actual return
        profit: float,    # P&L
        timestamp: str = None,
    ):
        self.ticker = ticker
        self.features = features
        self.signal = signal
        self.confidence = confidence
        self.strategy_breakdown = strategy_breakdown
        self.outcome = outcome
        self.profit = profit
        self.timestamp = timestamp or datetime.now().isoformat()
        
        # Label: 1 = good trade, 0 = bad trade
        self.was_profitable = profit > 0
        
    @staticmethod
    def _to_python(val):
        """Convert numpy scalars to native Python types for JSON serialization."""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "features": self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            "signal": self._to_python(self.signal),
            "confidence": self._to_python(self.confidence),
            "strategy_breakdown": self.strategy_breakdown,
            "outcome": self._to_python(self.outcome),
            "profit": self._to_python(self.profit),
            "timestamp": self.timestamp,
            "was_profitable": self._to_python(self.was_profitable),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TradeExperience":
        return cls(
            ticker=data["ticker"],
            features=np.array(data["features"]),
            signal=data["signal"],
            confidence=data["confidence"],
            strategy_breakdown=data["strategy_breakdown"],
            outcome=data["outcome"],
            profit=data["profit"],
            timestamp=data.get("timestamp"),
        )


class ExperienceBuffer:
    """
    Replay buffer for storing and sampling trade experiences.
    
    The model learns from:
    - Profitable trades (reinforce correct predictions)
    - Failed trades (learn what NOT to do)
    - Strategy confluence patterns
    """
    
    def __init__(self, max_size: int = 10000, save_path: Path = None):
        self.max_size = max_size
        self.save_path = save_path or CHECKPOINT_DIR / "experience_buffer.json"
        self.experiences: List[TradeExperience] = []
        self.training_stats: Dict = {
            "total_sessions": 0,
            "total_trades": 0,
            "win_rate_history": [],
            "strategy_performance": {},
            "last_session": None,
        }
        
        self._load()
    
    def add(self, experience: TradeExperience):
        """Add a new experience to the buffer."""
        self.experiences.append(experience)
        
        # Trim to max size (keep most recent)
        if len(self.experiences) > self.max_size:
            self.experiences = self.experiences[-self.max_size:]
        
        # Update stats
        self.training_stats["total_trades"] += 1
    
    def add_batch(self, experiences: List[TradeExperience]):
        """Add multiple experiences."""
        for exp in experiences:
            self.add(exp)
    
    def sample(self, n: int, profitable_only: bool = False) -> List[TradeExperience]:
        """Sample experiences for training."""
        if profitable_only:
            pool = [e for e in self.experiences if e.was_profitable]
        else:
            pool = self.experiences
        
        if len(pool) <= n:
            return pool
        
        indices = np.random.choice(len(pool), n, replace=False)
        return [pool[i] for i in indices]
    
    def get_features_and_labels(self, n: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get features, signals, and success labels for training.
        
        Returns:
            features: [N, seq_len, n_features]
            signals: [N] - predicted signals
            labels: [N] - 1 if trade was profitable, 0 otherwise
        """
        experiences = self.experiences[:n] if n else self.experiences
        
        if not experiences:
            return None, None, None
        
        features = np.array([e.features for e in experiences])
        signals = np.array([e.signal for e in experiences])
        labels = np.array([1.0 if e.was_profitable else 0.0 for e in experiences])
        
        return features, signals, labels
    
    def get_strategy_patterns(self) -> Dict:
        """
        Analyze which strategy combinations led to profitable trades.
        Returns patterns the model should learn.
        """
        if not self.experiences:
            return {}
        
        patterns = {
            "profitable": {},
            "unprofitable": {},
        }
        
        for exp in self.experiences:
            key = "profitable" if exp.was_profitable else "unprofitable"
            
            # Get agreeing strategies
            agreeing = []
            for strat, data in exp.strategy_breakdown.items():
                if data["signal"] == exp.signal and data["confidence"] > 0.5:
                    agreeing.append(strat)
            
            pattern_key = ",".join(sorted(agreeing)) if agreeing else "none"
            
            if pattern_key not in patterns[key]:
                patterns[key][pattern_key] = {"count": 0, "avg_profit": 0, "total_profit": 0}
            
            patterns[key][pattern_key]["count"] += 1
            patterns[key][pattern_key]["total_profit"] += exp.profit
        
        # Calculate averages
        for key in patterns:
            for p in patterns[key]:
                if patterns[key][p]["count"] > 0:
                    patterns[key][p]["avg_profit"] = patterns[key][p]["total_profit"] / patterns[key][p]["count"]
        
        return patterns
    
    def update_session_stats(self, session_result: Dict):
        """Update stats after a training session."""
        self.training_stats["total_sessions"] += 1
        self.training_stats["last_session"] = datetime.now().isoformat()
        
        if "win_rate" in session_result:
            self.training_stats["win_rate_history"].append(session_result["win_rate"])
        
        # Update strategy performance
        if "strategy_stats" in session_result:
            for strat, stats in session_result["strategy_stats"].items():
                if strat not in self.training_stats["strategy_performance"]:
                    self.training_stats["strategy_performance"][strat] = {
                        "total_trades": 0,
                        "wins": 0,
                        "total_profit": 0,
                    }
                self.training_stats["strategy_performance"][strat]["total_trades"] += stats.get("trades", 0)
                self.training_stats["strategy_performance"][strat]["wins"] += stats.get("wins", 0)
                self.training_stats["strategy_performance"][strat]["total_profit"] += stats.get("profit", 0)
        
        self._save()
    
    def get_learning_summary(self) -> Dict:
        """Get summary of what the model has learned for Ollama."""
        patterns = self.get_strategy_patterns()
        
        # Find best performing patterns
        best_patterns = sorted(
            patterns.get("profitable", {}).items(),
            key=lambda x: x[1]["avg_profit"],
            reverse=True
        )[:5]
        
        # Calculate overall stats
        total = len(self.experiences)
        wins = sum(1 for e in self.experiences if e.was_profitable)
        win_rate = wins / total if total > 0 else 0
        
        # Best strategies
        strat_perf = self.training_stats.get("strategy_performance", {})
        best_strategies = sorted(
            strat_perf.items(),
            key=lambda x: x[1].get("total_profit", 0),
            reverse=True
        )[:3]
        
        return {
            "total_experiences": total,
            "win_rate": win_rate,
            "total_sessions": self.training_stats["total_sessions"],
            "best_patterns": [(p, d["avg_profit"], d["count"]) for p, d in best_patterns],
            "best_strategies": [(s, d["total_profit"], d["wins"]/max(d["total_trades"],1)) for s, d in best_strategies],
            "improvement_trend": self._calculate_improvement(),
        }
    
    def _calculate_improvement(self) -> str:
        """Calculate if model is improving over time."""
        history = self.training_stats.get("win_rate_history", [])
        if len(history) < 2:
            return "insufficient_data"
        
        recent = np.mean(history[-3:]) if len(history) >= 3 else history[-1]
        earlier = np.mean(history[:3]) if len(history) >= 3 else history[0]
        
        if recent > earlier + 0.05:
            return "improving"
        elif recent < earlier - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _sanitize_stats(self, stats: Dict) -> Dict:
        """Recursively convert numpy types in training_stats to native Python."""
        sanitized = {}
        for k, v in stats.items():
            if isinstance(v, dict):
                sanitized[k] = self._sanitize_stats(v)
            elif isinstance(v, list):
                sanitized[k] = [self._sanitize_stats(i) if isinstance(i, dict)
                                else int(i) if isinstance(i, np.integer)
                                else float(i) if isinstance(i, np.floating)
                                else i for i in v]
            elif isinstance(v, (np.integer,)):
                sanitized[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sanitized[k] = float(v)
            elif isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                sanitized[k] = v
        return sanitized

    def _save(self):
        """Save buffer to disk."""
        try:
            data = {
                "experiences": [e.to_dict() for e in self.experiences[-1000:]],  # Keep last 1000
                "training_stats": self._sanitize_stats(self.training_stats),
            }
            # Write to temp file first, then rename for atomicity
            tmp_path = self.save_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self.save_path)
        except Exception as e:
            logger.warning(f"Could not save experience buffer: {e}")
    
    def _load(self):
        """Load buffer from disk."""
        if self.save_path.exists():
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                
                self.experiences = [TradeExperience.from_dict(e) for e in data.get("experiences", [])]
                self.training_stats = data.get("training_stats", self.training_stats)
                logger.info(f"Loaded {len(self.experiences)} experiences from buffer")
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                logger.warning(f"Experience buffer is corrupted, backing up and starting fresh: {e}")
                # Back up the corrupted file for debugging
                backup = self.save_path.with_suffix(".json.bak")
                try:
                    import shutil
                    shutil.copy2(self.save_path, backup)
                    logger.info(f"Corrupted buffer backed up to {backup}")
                    # Delete the original so we don't keep failing
                    self.save_path.unlink()
                    logger.info("Deleted corrupted buffer file")
                except Exception:
                    pass
                # Reset to empty state
                self.experiences = []
            except Exception as e:
                logger.warning(f"Could not load experience buffer: {e}")
    
    def clear(self):
        """Clear the buffer."""
        self.experiences = []
        self.training_stats = {
            "total_sessions": 0,
            "total_trades": 0,
            "win_rate_history": [],
            "strategy_performance": {},
            "last_session": None,
        }
        self._save()


# Singleton
_experience_buffer = None

def get_experience_buffer() -> ExperienceBuffer:
    """Get or create the experience buffer."""
    global _experience_buffer
    if _experience_buffer is None:
        _experience_buffer = ExperienceBuffer()
    return _experience_buffer
