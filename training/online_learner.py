"""
Online Learner — Real-time Learning from Trade Outcomes
=========================================================
Like modern AI models, learns from each prediction's outcome.
Implements continual learning with catastrophic forgetting prevention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, CHECKPOINT_DIR
from training.experience import get_experience_buffer, TradeExperience
from training.curriculum import get_curriculum_manager


class ElasticWeightConsolidation:
    """
    Prevents catastrophic forgetting in continual learning.
    Stores Fisher information matrix for important weights.
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.1):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optimal_params = {}
        self._initialized = False
    
    def compute_fisher(self, dataloader, device: str):
        """Compute Fisher information matrix for current task."""
        self.model.eval()
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            self.model.zero_grad()
            out = self.model(X)
            if isinstance(out, dict):
                logits = out.get("direction", out.get("logits", list(out.values())[0]))
            else:
                logits = out
            
            # Sample from predictive distribution
            probs = F.softmax(logits, dim=-1)
            labels = probs.multinomial(num_samples=1).squeeze()
            
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher_info[n] += p.grad.data ** 2
        
        # Normalize
        for n in self.fisher_info:
            self.fisher_info[n] /= len(dataloader)
        
        # Store optimal params
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self._initialized = True
        logger.info("  EWC: Fisher information computed")
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self._initialized:
            return torch.tensor(0.0)
        
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_info and n in self.optimal_params:
                loss += (self.fisher_info[n] * (p - self.optimal_params[n]) ** 2).sum()
        
        return self.lambda_ewc * loss


class OnlineLearner:
    """
    Real-time learning from trade outcomes.
    
    Features:
    - Learns from each prediction outcome immediately
    - Uses Elastic Weight Consolidation to prevent forgetting
    - Implements replay learning with prioritized sampling
    - Tracks performance improvement over time
    """
    
    def __init__(self, model_type: str = "tft"):
        self.model_type = model_type
        self.device = cfg.gpu.device
        self.experience_buffer = get_experience_buffer()
        self.curriculum = get_curriculum_manager()
        
        # Learning state
        self.total_online_updates = 0
        self.recent_losses: List[float] = []
        self.performance_trend: List[Dict] = []
        
        # EWC for continual learning
        self.ewc = None
        
        # Optimizer (will be set when model is loaded)
        self.optimizer = None
        self.model = None
        
        # Learning hyperparameters
        self.learning_rate = 1e-5  # Small LR for online updates
        self.mini_batch_size = 8
        self.update_frequency = 5  # Update every N trades
    
    def initialize(self, model: nn.Module):
        """Initialize online learner with a trained model."""
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.ewc = ElasticWeightConsolidation(model)
        logger.info(f"  Online learner initialized for {self.model_type}")
    
    def learn_from_outcome(
        self,
        features: np.ndarray,
        signal: int,
        confidence: float,
        strategy_breakdown: Dict,
        outcome: float,
        profit: float,
        ticker: str = "UNKNOWN",
    ) -> Dict:
        """
        Learn from a single trade outcome.
        
        Args:
            features: Input features [seq_len, n_features]
            signal: Predicted signal (-1, 0, 1)
            confidence: Prediction confidence
            strategy_breakdown: Strategy analysis for this trade
            outcome: Actual return after prediction
            profit: P&L of the trade
            ticker: Asset symbol
            
        Returns:
            Learning update summary
        """
        if self.model is None:
            return {"learned": False, "reason": "model_not_initialized"}
        
        # Store experience
        exp = TradeExperience(
            ticker=ticker,
            features=features,
            signal=signal,
            confidence=confidence,
            strategy_breakdown=strategy_breakdown,
            outcome=outcome,
            profit=profit,
        )
        self.experience_buffer.add(exp)
        
        # Track if this was a hard example (high confidence but wrong)
        is_hard = self._is_hard_example(signal, confidence, outcome, profit)
        if is_hard:
            self.curriculum.add_hard_example(features, signal, outcome, profit)
        
        # Incremental update counter
        self.total_online_updates += 1
        
        # Only perform gradient update every N trades (efficiency)
        if self.total_online_updates % self.update_frequency != 0:
            return {"learned": False, "reason": "waiting_for_batch"}
        
        # Perform online learning step
        result = self._online_update_step()
        
        return result
    
    def _is_hard_example(self, signal: int, confidence: float, outcome: float, profit: float) -> bool:
        """Determine if this was a hard example (model was confident but wrong)."""
        if signal == 0:  # HOLD predictions aren't "hard"
            return False
        
        # High confidence but wrong direction
        was_wrong = profit < 0
        was_confident = confidence > 0.6
        
        return was_wrong and was_confident
    
    def _online_update_step(self) -> Dict:
        """Perform one online learning update with replay."""
        self.model.train()
        
        # Sample from experience buffer (prioritize hard/recent examples)
        batch = self._sample_training_batch()
        
        if batch is None:
            return {"learned": False, "reason": "insufficient_experiences"}
        
        X, y_signals, y_outcomes, weights = batch
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y_signals + 1).to(self.device)  # Map -1,0,1 to 0,1,2
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        out = self.model(X)
        
        if isinstance(out, dict):
            logits = out.get("direction", out.get("logits", list(out.values())[0]))
        else:
            logits = out
        
        # Classification loss (weighted by importance)
        cls_loss = F.cross_entropy(logits, y, reduction='none')
        cls_loss = (cls_loss * weights).mean()
        
        # EWC regularization (prevent forgetting)
        ewc_loss = self.ewc.ewc_loss()
        
        # Total loss
        total_loss = cls_loss + ewc_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track loss
        loss_val = total_loss.item()
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses = self.recent_losses[-100:]
        
        # Track performance
        self._update_performance_trend()
        
        return {
            "learned": True,
            "loss": loss_val,
            "cls_loss": cls_loss.item(),
            "ewc_loss": ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0,
            "batch_size": len(X),
            "total_updates": self.total_online_updates,
            "avg_recent_loss": np.mean(self.recent_losses[-10:]),
        }
    
    def _sample_training_batch(self) -> Optional[Tuple]:
        """
        Sample a batch with curriculum learning.
        Prioritizes hard examples and recent experiences.
        """
        # Get hard examples from curriculum
        hard_batch = self.curriculum.sample_hard_examples(self.mini_batch_size // 2)
        
        # Get random experiences
        exp_batch = self.experience_buffer.sample(self.mini_batch_size // 2)
        
        if not hard_batch and not exp_batch:
            return None
        
        # Combine
        all_features = []
        all_signals = []
        all_outcomes = []
        all_weights = []
        
        # Hard examples get higher weight
        for features, signal, outcome in hard_batch:
            all_features.append(features)
            all_signals.append(signal)
            all_outcomes.append(outcome)
            all_weights.append(2.0)  # Higher weight for hard examples
        
        # Regular experiences
        for exp in exp_batch:
            all_features.append(exp.features)
            all_signals.append(exp.signal)
            all_outcomes.append(exp.outcome)
            all_weights.append(1.0)
        
        if not all_features:
            return None
        
        return (
            np.array(all_features),
            np.array(all_signals),
            np.array(all_outcomes),
            np.array(all_weights),
        )
    
    def _update_performance_trend(self):
        """Track performance improvement over time."""
        if self.total_online_updates % 50 == 0:
            recent_wr = self._calculate_recent_win_rate()
            self.performance_trend.append({
                "update": self.total_online_updates,
                "win_rate": recent_wr,
                "avg_loss": np.mean(self.recent_losses[-20:]) if self.recent_losses else 0,
                "timestamp": datetime.now().isoformat(),
            })
    
    def _calculate_recent_win_rate(self) -> float:
        """Calculate win rate from recent experiences."""
        recent = self.experience_buffer.experiences[-100:]
        if not recent:
            return 0.0
        wins = sum(1 for e in recent if e.was_profitable)
        return wins / len(recent)
    
    def consolidate_knowledge(self, dataloader):
        """
        Consolidate learned knowledge using EWC.
        Call this periodically to prevent catastrophic forgetting.
        """
        if self.ewc is not None:
            self.ewc.compute_fisher(dataloader, self.device)
            logger.info(f"  Knowledge consolidated for {self.model_type}")
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about online learning progress."""
        return {
            "model_type": self.model_type,
            "total_online_updates": self.total_online_updates,
            "recent_avg_loss": np.mean(self.recent_losses[-20:]) if self.recent_losses else 0,
            "loss_trend": self._calculate_loss_trend(),
            "win_rate_trend": [p["win_rate"] for p in self.performance_trend[-10:]],
            "total_experiences": len(self.experience_buffer.experiences),
            "hard_examples_tracked": len(self.curriculum.hard_examples),
        }
    
    def _calculate_loss_trend(self) -> str:
        """Determine if loss is decreasing (improving)."""
        if len(self.recent_losses) < 20:
            return "insufficient_data"
        
        recent = np.mean(self.recent_losses[-10:])
        earlier = np.mean(self.recent_losses[-20:-10])
        
        if recent < earlier * 0.95:
            return "improving"
        elif recent > earlier * 1.05:
            return "worsening"
        else:
            return "stable"


# Module-level online learners
_online_learners = {}

def get_online_learner(model_type: str) -> OnlineLearner:
    """Get or create an online learner for the specified model type."""
    if model_type not in _online_learners:
        _online_learners[model_type] = OnlineLearner(model_type)
    return _online_learners[model_type]
