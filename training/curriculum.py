"""
Curriculum Learning — Focus on Hard Examples
=============================================
Like human learning, focuses on challenging patterns.
Prioritizes examples where the model was confident but wrong.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from loguru import logger
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHECKPOINT_DIR


class HardExample:
    """A challenging example where the model failed."""
    
    def __init__(
        self,
        features: np.ndarray,
        predicted_signal: int,
        actual_outcome: float,
        profit: float,
        confidence: float,
        strategy_breakdown: Dict = None,
        timestamp: str = None,
    ):
        self.features = features
        self.predicted_signal = predicted_signal
        self.actual_outcome = actual_outcome
        self.profit = profit
        self.confidence = confidence
        self.strategy_breakdown = strategy_breakdown or {}
        self.timestamp = timestamp or str(np.datetime64('now'))
        
        # How "hard" this example is (higher = harder)
        self.difficulty = self._compute_difficulty()
    
    def _compute_difficulty(self) -> float:
        """
        Compute difficulty score.
        Higher when model was confident but very wrong.
        """
        # Confidence penalty (high confidence + wrong = very hard)
        confidence_penalty = self.confidence if self.profit < 0 else 0
        
        # Loss magnitude
        loss_magnitude = abs(self.profit)
        
        # Direction error
        direction_error = 1.0 if (self.predicted_signal * self.actual_outcome < 0) else 0.5
        
        return confidence_penalty * 2 + loss_magnitude * 10 + direction_error
    
    def to_dict(self) -> Dict:
        return {
            "features": self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            "predicted_signal": self.predicted_signal,
            "actual_outcome": self.actual_outcome,
            "profit": self.profit,
            "confidence": self.confidence,
            "strategy_breakdown": self.strategy_breakdown,
            "timestamp": self.timestamp,
            "difficulty": self.difficulty,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HardExample":
        return cls(
            features=np.array(data["features"]),
            predicted_signal=data["predicted_signal"],
            actual_outcome=data["actual_outcome"],
            profit=data["profit"],
            confidence=data["confidence"],
            strategy_breakdown=data.get("strategy_breakdown", {}),
            timestamp=data.get("timestamp"),
        )


class CurriculumManager:
    """
    Manages curriculum learning for the trading model.
    
    Features:
    - Tracks hard examples (confident but wrong predictions)
    - Prioritizes hard examples during training
    - Graduates examples that are no longer hard
    - Tracks pattern-specific weaknesses
    """
    
    def __init__(self, max_hard_examples: int = 500, save_path: Path = None):
        self.max_hard_examples = max_hard_examples
        self.save_path = save_path or CHECKPOINT_DIR / "curriculum.json"
        self.hard_examples: List[HardExample] = []
        
        # Pattern weakness tracking
        self.pattern_weaknesses: Dict[str, Dict] = {}
        
        # Training history
        self.graduation_history: List[Dict] = []
        
        self._load()
    
    def add_hard_example(
        self,
        features: np.ndarray,
        predicted_signal: int,
        actual_outcome: float,
        profit: float,
        confidence: float = 0.5,
        strategy_breakdown: Dict = None,
    ):
        """Add a hard example to the curriculum."""
        example = HardExample(
            features=features,
            predicted_signal=predicted_signal,
            actual_outcome=actual_outcome,
            profit=profit,
            confidence=confidence,
            strategy_breakdown=strategy_breakdown,
        )
        
        self.hard_examples.append(example)
        
        # Track pattern weaknesses
        if strategy_breakdown:
            for strat, data in strategy_breakdown.items():
                if strat not in self.pattern_weaknesses:
                    self.pattern_weaknesses[strat] = {
                        "total": 0,
                        "wrong": 0,
                        "avg_confidence_when_wrong": 0,
                    }
                self.pattern_weaknesses[strat]["total"] += 1
                if profit < 0:
                    self.pattern_weaknesses[strat]["wrong"] += 1
                    self.pattern_weaknesses[strat]["avg_confidence_when_wrong"] = (
                        self.pattern_weaknesses[strat]["avg_confidence_when_wrong"] * 0.9 +
                        data.get("confidence", 0.5) * 0.1
                    )
        
        # Trim to max size (keep hardest)
        if len(self.hard_examples) > self.max_hard_examples:
            self.hard_examples.sort(key=lambda x: x.difficulty, reverse=True)
            self.hard_examples = self.hard_examples[:self.max_hard_examples]
        
        self._save()
    
    def sample_hard_examples(self, n: int) -> List[Tuple]:
        """
        Sample hard examples for training.
        Prioritizes by difficulty with some randomness.
        
        Returns:
            List of (features, predicted_signal, actual_outcome) tuples
        """
        if not self.hard_examples:
            return []
        
        # Sort by difficulty
        sorted_examples = sorted(self.hard_examples, key=lambda x: x.difficulty, reverse=True)
        
        # Sample from top 50% hardest (with randomness)
        top_half = sorted_examples[:max(1, len(sorted_examples) // 2)]
        
        n = min(n, len(top_half))
        if n == 0:
            return []
        
        # Weighted sampling by difficulty
        difficulties = np.array([e.difficulty for e in top_half])
        probs = difficulties / (difficulties.sum() + 1e-10)
        
        indices = np.random.choice(len(top_half), size=n, replace=False, p=probs)
        
        return [
            (top_half[i].features, top_half[i].predicted_signal, top_half[i].actual_outcome)
            for i in indices
        ]
    
    def graduate_examples(self, model, device: str, threshold: float = 0.7) -> int:
        """
        Graduate examples that the model now handles correctly.
        
        Args:
            model: The trained model
            device: Device to run on
            threshold: Confidence threshold for graduation
            
        Returns:
            Number of graduated examples
        """
        import torch
        import torch.nn.functional as F
        
        if not self.hard_examples:
            return 0
        
        model.eval()
        graduated = []
        
        with torch.no_grad():
            for i, example in enumerate(self.hard_examples):
                X = torch.FloatTensor(example.features).unsqueeze(0).to(device)
                
                try:
                    out = model(X)
                    if isinstance(out, dict):
                        logits = out.get("direction", out.get("logits", list(out.values())[0]))
                    else:
                        logits = out
                    
                    probs = F.softmax(logits, dim=-1)
                    pred_class = probs.argmax(dim=-1).item()
                    pred_signal = pred_class - 1  # Map back to -1, 0, 1
                    confidence = probs.max().item()
                    
                    # Check if model now predicts correctly with high confidence
                    actual_direction = 1 if example.actual_outcome > 0 else -1 if example.actual_outcome < 0 else 0
                    
                    if pred_signal == actual_direction and confidence > threshold:
                        graduated.append(i)
                        
                except Exception as e:
                    logger.warning(f"Error checking example: {e}")
        
        # Remove graduated examples
        if graduated:
            self.hard_examples = [e for i, e in enumerate(self.hard_examples) if i not in graduated]
            
            self.graduation_history.append({
                "timestamp": str(np.datetime64('now')),
                "n_graduated": len(graduated),
                "remaining": len(self.hard_examples),
            })
            
            logger.info(f"  Curriculum: Graduated {len(graduated)} examples, {len(self.hard_examples)} remaining")
            self._save()
        
        return len(graduated)
    
    def get_weak_patterns(self, top_n: int = 5) -> List[Tuple]:
        """
        Get patterns where the model struggles most.
        
        Returns:
            List of (pattern_name, error_rate, avg_confidence_when_wrong)
        """
        weaknesses = []
        for strat, data in self.pattern_weaknesses.items():
            if data["total"] > 5:  # Need enough samples
                error_rate = data["wrong"] / data["total"]
                weaknesses.append((strat, error_rate, data["avg_confidence_when_wrong"]))
        
        # Sort by error rate
        weaknesses.sort(key=lambda x: x[1], reverse=True)
        return weaknesses[:top_n]
    
    def get_curriculum_summary(self) -> Dict:
        """Get summary of curriculum learning status."""
        return {
            "total_hard_examples": len(self.hard_examples),
            "avg_difficulty": np.mean([e.difficulty for e in self.hard_examples]) if self.hard_examples else 0,
            "top_weak_patterns": self.get_weak_patterns(3),
            "total_graduated": sum(h["n_graduated"] for h in self.graduation_history),
            "recent_graduations": self.graduation_history[-3:] if self.graduation_history else [],
        }
    
    def _save(self):
        """Save curriculum to disk."""
        try:
            data = {
                "hard_examples": [e.to_dict() for e in self.hard_examples[-200:]],  # Keep last 200
                "pattern_weaknesses": self.pattern_weaknesses,
                "graduation_history": self.graduation_history[-20:],
            }
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save curriculum: {e}")
    
    def _load(self):
        """Load curriculum from disk."""
        if self.save_path.exists():
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                
                self.hard_examples = [HardExample.from_dict(e) for e in data.get("hard_examples", [])]
                self.pattern_weaknesses = data.get("pattern_weaknesses", {})
                self.graduation_history = data.get("graduation_history", [])
                
                logger.info(f"Loaded curriculum: {len(self.hard_examples)} hard examples")
            except Exception as e:
                logger.warning(f"Could not load curriculum: {e}")
    
    def clear(self):
        """Clear the curriculum."""
        self.hard_examples = []
        self.pattern_weaknesses = {}
        self.graduation_history = []
        self._save()


# Singleton
_curriculum_manager = None

def get_curriculum_manager() -> CurriculumManager:
    """Get or create the curriculum manager."""
    global _curriculum_manager
    if _curriculum_manager is None:
        _curriculum_manager = CurriculumManager()
    return _curriculum_manager
