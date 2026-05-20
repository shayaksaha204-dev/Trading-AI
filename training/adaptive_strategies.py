"""
Adaptive Strategy Weights — Strategies That Improve Over Time
==============================================================
Like human traders, learns which strategies work best in different conditions.
Weights adapt based on actual performance, not static rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from loguru import logger
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHECKPOINT_DIR


class StrategyPerformanceTracker:
    """
    Tracks performance of each strategy over time.
    Enables adaptive weighting based on what's actually working.
    """
    
    def __init__(self, save_path: Path = None):
        self.save_path = save_path or CHECKPOINT_DIR / "strategy_performance.json"
        
        # Strategy names
        self.strategies = [
            "Technical",
            "Market_Structure",
            "Price_Action",
            "SMC",
            "ICT",
            "Wyckoff",
            "Elliott_Wave",
            "Sentiment",
        ]
        
        # Performance tracking
        self.strategy_stats = {
            strat: {
                "total_signals": 0,
                "correct_signals": 0,
                "total_profit": 0.0,
                "wins": 0,
                "losses": 0,
                "avg_confidence": 0.0,
                "recent_performance": [],  # Last 20 trades
            }
            for strat in self.strategies
        }
        
        # Condition-specific performance (regime, volatility, etc.)
        self.condition_performance = {
            "bull": {s: {"wins": 0, "total": 0} for s in self.strategies},
            "bear": {s: {"wins": 0, "total": 0} for s in self.strategies},
            "sideways": {s: {"wins": 0, "total": 0} for s in self.strategies},
            "volatile": {s: {"wins": 0, "total": 0} for s in self.strategies},
        }
        
        # Adaptive weights (start equal)
        self.weights = {s: 1.0 / len(self.strategies) for s in self.strategies}
        
        # Learning rate for weight updates
        self.weight_learning_rate = 0.05
        
        self._load()
    
    def record_outcome(
        self,
        strategy_breakdown: Dict,
        actual_outcome: float,
        profit: float,
        regime: str = "unknown",
    ):
        """
        Record the outcome of a trade with strategy breakdown.
        
        Args:
            strategy_breakdown: Dict of strategy_name -> {signal, confidence, reason}
            actual_outcome: Actual return after prediction
            profit: P&L of the trade
            regime: Market regime at time of trade
        """
        for strat, data in strategy_breakdown.items():
            strat_name = strat.replace(" ", "_")
            if strat_name not in self.strategy_stats:
                continue
            
            stats = self.strategy_stats[strat_name]
            
            # Update basic stats
            stats["total_signals"] += 1
            stats["total_profit"] += profit
            stats["avg_confidence"] = stats["avg_confidence"] * 0.9 + data.get("confidence", 0.5) * 0.1
            
            # Check if strategy was correct
            strat_signal = data.get("signal", 0)
            actual_direction = 1 if actual_outcome > 0 else -1 if actual_outcome < 0 else 0
            
            if strat_signal == actual_direction and strat_signal != 0:
                stats["correct_signals"] += 1
                stats["wins"] += 1
            elif strat_signal != 0:
                stats["losses"] += 1
            
            # Track recent performance
            stats["recent_performance"].append(profit if strat_signal == actual_direction else -abs(profit))
            if len(stats["recent_performance"]) > 20:
                stats["recent_performance"] = stats["recent_performance"][-20:]
            
            # Track condition-specific performance
            if regime in self.condition_performance:
                self.condition_performance[regime][strat_name]["total"] += 1
                if strat_signal == actual_direction and strat_signal != 0:
                    self.condition_performance[regime][strat_name]["wins"] += 1
        
        # Update adaptive weights
        self._update_weights()
        self._save()
    
    def _update_weights(self):
        """
        Update strategy weights based on recent performance.
        Uses exponential moving average of win rates.
        """
        new_weights = {}
        
        for strat in self.strategies:
            stats = self.strategy_stats[strat]
            
            # Calculate recent win rate
            recent = stats["recent_performance"]
            if len(recent) >= 5:
                recent_wr = sum(1 for r in recent if r > 0) / len(recent)
            else:
                recent_wr = 0.5  # Default
            
            # Calculate overall win rate
            total = stats["wins"] + stats["losses"]
            if total > 0:
                overall_wr = stats["wins"] / total
            else:
                overall_wr = 0.5
            
            # Combine recent and overall (favor recent)
            combined_wr = recent_wr * 0.7 + overall_wr * 0.3
            
            # Weight is proportional to win rate
            new_weights[strat] = combined_wr
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        else:
            new_weights = self.weights  # Keep old weights if no data
        
        # Smooth update (don't change too fast)
        for strat in self.strategies:
            self.weights[strat] = (
                self.weights[strat] * (1 - self.weight_learning_rate) +
                new_weights[strat] * self.weight_learning_rate
            )
        
        # Re-normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def get_weights(self, regime: str = None) -> Dict[str, float]:
        """
        Get strategy weights, optionally regime-adjusted.
        
        Args:
            regime: Current market regime for condition-specific weights
            
        Returns:
            Dict of strategy_name -> weight
        """
        if regime and regime in self.condition_performance:
            # Adjust weights based on regime-specific performance
            regime_weights = {}
            for strat in self.strategies:
                data = self.condition_performance[regime][strat]
                if data["total"] > 5:
                    regime_wr = data["wins"] / data["total"]
                    regime_weights[strat] = self.weights[strat] * (0.5 + regime_wr)
                else:
                    regime_weights[strat] = self.weights[strat]
            
            # Normalize
            total = sum(regime_weights.values())
            if total > 0:
                regime_weights = {k: v / total for k, v in regime_weights.items()}
            
            return regime_weights
        
        return self.weights.copy()
    
    def get_best_strategies(self, top_n: int = 3) -> List[Tuple]:
        """
        Get top performing strategies.
        
        Returns:
            List of (strategy_name, win_rate, total_profit)
        """
        performance = []
        for strat in self.strategies:
            stats = self.strategy_stats[strat]
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total if total > 0 else 0
            performance.append((strat, wr, stats["total_profit"]))
        
        performance.sort(key=lambda x: x[1], reverse=True)
        return performance[:top_n]
    
    def get_worst_strategies(self, top_n: int = 3) -> List[Tuple]:
        """
        Get worst performing strategies (need improvement).
        
        Returns:
            List of (strategy_name, win_rate, total_profit)
        """
        performance = []
        for strat in self.strategies:
            stats = self.strategy_stats[strat]
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total if total > 0 else 0
            performance.append((strat, wr, stats["total_profit"]))
        
        performance.sort(key=lambda x: x[1])
        return performance[:top_n]
    
    def get_regime_recommendations(self, regime: str) -> Dict:
        """
        Get strategy recommendations for a specific regime.
        
        Returns:
            Dict with best/worst strategies for this regime
        """
        if regime not in self.condition_performance:
            return {"regime": regime, "recommendations": "insufficient_data"}
        
        regime_data = self.condition_performance[regime]
        
        # Sort by win rate
        sorted_strats = sorted(
            [(s, d["wins"] / max(d["total"], 1), d["total"]) for s, d in regime_data.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "regime": regime,
            "best_strategies": [(s, wr, n) for s, wr, n in sorted_strats[:3] if n > 5],
            "avoid_strategies": [(s, wr, n) for s, wr, n in sorted_strats[-3:] if n > 5],
            "total_trades_in_regime": sum(d["total"] for d in regime_data.values()),
        }
    
    def get_summary(self) -> Dict:
        """Get overall summary of strategy performance."""
        return {
            "current_weights": self.weights,
            "best_strategies": self.get_best_strategies(3),
            "worst_strategies": self.get_worst_strategies(3),
            "total_trades_tracked": sum(s["total_signals"] for s in self.strategy_stats.values()),
            "regime_breakdown": {
                regime: sum(d["total"] for d in data.values())
                for regime, data in self.condition_performance.items()
            },
        }
    
    def _save(self):
        """Save performance data to disk."""
        try:
            data = {
                "strategy_stats": self.strategy_stats,
                "condition_performance": self.condition_performance,
                "weights": self.weights,
                "timestamp": datetime.now().isoformat(),
            }
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save strategy performance: {e}")
    
    def _load(self):
        """Load performance data from disk."""
        if self.save_path.exists():
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                
                self.strategy_stats = data.get("strategy_stats", self.strategy_stats)
                self.condition_performance = data.get("condition_performance", self.condition_performance)
                self.weights = data.get("weights", self.weights)
                
                logger.info(f"Loaded strategy performance: {sum(s['total_signals'] for s in self.strategy_stats.values())} trades tracked")
            except Exception as e:
                logger.warning(f"Could not load strategy performance: {e}")


class AdaptiveConfluenceScorer:
    """
    Uses adaptive strategy weights to compute confluence scores.
    Improves over time as it learns which strategies work best.
    """
    
    def __init__(self, tracker: StrategyPerformanceTracker = None):
        self.tracker = tracker or StrategyPerformanceTracker()
    
    def compute_weighted_confluence(
        self,
        strategy_breakdown: Dict,
        regime: str = None,
    ) -> Tuple[int, float, Dict]:
        """
        Compute confluence signal using adaptive weights.
        
        Args:
            strategy_breakdown: Dict of strategy_name -> {signal, confidence}
            regime: Current market regime
            
        Returns:
            (signal, confidence, weighted_breakdown)
        """
        weights = self.tracker.get_weights(regime)
        
        # Weighted vote
        bull_score = 0.0
        bear_score = 0.0
        total_weight = 0.0
        
        for strat, data in strategy_breakdown.items():
            strat_name = strat.replace(" ", "_")
            weight = weights.get(strat_name, 1.0 / 8)
            signal = data.get("signal", 0)
            confidence = data.get("confidence", 0.5)
            
            weighted_conf = weight * confidence
            
            if signal == 1:
                bull_score += weighted_conf
            elif signal == -1:
                bear_score += weighted_conf
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            bull_score /= total_weight
            bear_score /= total_weight
        
        # Determine signal
        threshold = 0.3  # Minimum weighted confidence for a signal
        
        if bull_score > bear_score + threshold:
            signal = 1
            confidence = bull_score
        elif bear_score > bull_score + threshold:
            signal = -1
            confidence = bear_score
        else:
            signal = 0
            confidence = 0.3
        
        return signal, confidence, {
            "bull_score": bull_score,
            "bear_score": bear_score,
            "weights_used": weights,
        }
    
    def record_outcome(self, strategy_breakdown: Dict, outcome: float, profit: float, regime: str = None):
        """Record outcome to improve future weights."""
        self.tracker.record_outcome(strategy_breakdown, outcome, profit, regime)


# Singleton
_strategy_tracker = None

def get_strategy_tracker() -> StrategyPerformanceTracker:
    """Get or create the strategy performance tracker."""
    global _strategy_tracker
    if _strategy_tracker is None:
        _strategy_tracker = StrategyPerformanceTracker()
    return _strategy_tracker
