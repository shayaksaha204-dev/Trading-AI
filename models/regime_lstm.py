"""
Regime LSTM — Bidirectional LSTM for Market Regime Detection
==============================================================
Classifies market into: bull, bear, sideways, volatile regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class TemporalAttention(nn.Module):
    """Attention mechanism over LSTM hidden states."""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h):
        # h: [batch, seq, hidden]
        scores = self.v(torch.tanh(self.W(h)))  # [batch, seq, 1]
        weights = F.softmax(scores, dim=1)
        context = (h * weights).sum(dim=1)  # [batch, hidden]
        return context, weights.squeeze(-1)


class RegimeLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for market regime classification.
    
    Regimes:
        0 = Bull (strong uptrend)
        1 = Bear (strong downtrend)
        2 = Sideways (range-bound)
        3 = Volatile (high volatility, no clear direction)
    """

    def __init__(self, n_features: int = None, config=None):
        super().__init__()
        c = config or cfg.lstm
        n_features = n_features or c.input_size
        self.hidden_size = c.hidden_size
        direction_mult = 2 if c.bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, c.hidden_size),
            nn.LayerNorm(c.hidden_size),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=c.hidden_size,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            batch_first=True,
            bidirectional=c.bidirectional,
            dropout=c.dropout if c.num_layers > 1 else 0,
        )

        # Attention
        self.attention = TemporalAttention(c.hidden_size * direction_mult) if c.attention else None

        effective_hidden = c.hidden_size * direction_mult

        # Regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(effective_hidden, c.hidden_size),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_size, c.num_regimes),
        )

        # Regime strength/confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(effective_hidden, c.hidden_size // 2),
            nn.GELU(),
            nn.Linear(c.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Transition probability head (predicts next regime)
        self.transition_head = nn.Sequential(
            nn.Linear(effective_hidden, c.hidden_size),
            nn.GELU(),
            nn.Linear(c.hidden_size, c.num_regimes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            Dict with regime_probs, confidence, transition_probs, attention_weights
        """
        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)

        if self.attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
            attn_weights = None

        return {
            "regime_logits": self.regime_head(context),
            "confidence": self.confidence_head(context).squeeze(-1),
            "transition_logits": self.transition_head(context),
            "attention_weights": attn_weights,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        self.eval()
        regime_names = ["bull", "bear", "sideways", "volatile"]
        with torch.no_grad():
            out = self.forward(x)
        probs = F.softmax(out["regime_logits"], dim=-1).cpu().numpy()
        return {
            "regime_probs": probs,
            "predicted_regime": [regime_names[i] for i in probs.argmax(axis=-1)],
            "confidence": out["confidence"].cpu().numpy(),
            "transition_probs": F.softmax(out["transition_logits"], dim=-1).cpu().numpy(),
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = RegimeLSTM(n_features=64)
    print(f"RegimeLSTM Parameters: {model.get_param_count():,}")
    x = torch.randn(4, 60, 64)
    out = model(x)
    print(f"Regime logits: {out['regime_logits'].shape}")
    print(f"Confidence: {out['confidence'].shape}")
    pred = model.predict(x)
    print(f"Predicted regimes: {pred['predicted_regime']}")
