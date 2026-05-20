"""
Pattern CNN — 1D Convolutional Network for Chart Pattern Recognition
=====================================================================
Multi-scale CNN that detects breakout, reversal, continuation, and consolidation patterns.
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


class MultiScaleConvBlock(nn.Module):
    """Parallel convolutions with different kernel sizes to capture multi-scale patterns."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ) for k in kernel_sizes
        ])
        self.merge = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        merged = torch.cat(outs, dim=1)
        return self.norm(F.gelu(self.merge(merged)))


class PatternCNN(nn.Module):
    """
    1D CNN for chart pattern detection.
    
    Architecture:
        - Multi-scale convolution blocks (3, 5, 7 kernel sizes)
        - Residual connections
        - Global average pooling
        - Classification head for pattern type
        - Regression head for pattern strength/confidence
    """

    def __init__(self, n_features: int = None, config=None):
        super().__init__()
        c = config or cfg.cnn
        n_features = n_features or 5  # OHLCV default

        # Multi-scale conv blocks
        self.block1 = MultiScaleConvBlock(n_features, c.num_filters[0], c.kernel_sizes)
        self.block2 = MultiScaleConvBlock(c.num_filters[0], c.num_filters[1], c.kernel_sizes)
        self.block3 = MultiScaleConvBlock(c.num_filters[1], c.num_filters[2], c.kernel_sizes)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(c.dropout)

        # Pattern classification head
        self.pattern_head = nn.Sequential(
            nn.Linear(c.num_filters[2], c.fc_size),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fc_size, c.num_classes),
        )

        # Pattern strength head (0-1)
        self.strength_head = nn.Sequential(
            nn.Linear(c.num_filters[2], c.fc_size // 2),
            nn.GELU(),
            nn.Linear(c.fc_size // 2, 1),
            nn.Sigmoid(),
        )

        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(c.num_filters[2], c.fc_size // 2),
            nn.GELU(),
            nn.Linear(c.fc_size // 2, 3),  # down, neutral, up
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            Dict with pattern_class, pattern_strength, direction
        """
        # Conv1d expects [batch, channels, length]
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)  # [batch, filters]
        x = self.dropout(x)

        return {
            "pattern_class": self.pattern_head(x),
            "pattern_strength": self.strength_head(x).squeeze(-1),
            "direction": self.direction_head(x),
        }

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        pattern_names = ["breakout", "reversal_up", "reversal_down", "continuation", "consolidation"]
        probs = F.softmax(out["pattern_class"], dim=-1).cpu().numpy()
        return {
            "pattern_probs": probs,
            "predicted_pattern": [pattern_names[i] for i in probs.argmax(axis=-1)],
            "pattern_strength": out["pattern_strength"].cpu().numpy(),
            "direction_probs": F.softmax(out["direction"], dim=-1).cpu().numpy(),
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = PatternCNN(n_features=64)
    print(f"PatternCNN Parameters: {model.get_param_count():,}")
    x = torch.randn(4, 60, 64)
    out = model(x)
    print(f"Pattern class: {out['pattern_class'].shape}")
    print(f"Strength: {out['pattern_strength'].shape}")
    print(f"Direction: {out['direction'].shape}")
