"""
TFT Predictor — Temporal Fusion Transformer for Multi-Horizon Forecasting
===========================================================================
Primary prediction model using attention-based architecture.
Built on PyTorch with custom implementation for flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.sigmoid(self.fc1(x)) * self.dropout(self.fc2(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1, context_size=None):
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(output_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        h = self.elu(self.fc1(x))
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = self.fc2(h)
        h = self.glu(h)
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_vars, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vars = num_vars
        self.flattened_grn = GatedResidualNetwork(
            num_vars * input_size, hidden_size, num_vars, dropout, context_size
        )
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout) for _ in range(num_vars)
        ])

    def forward(self, x, context=None):
        # x: [batch, time, num_vars * input_per_var] or [batch, num_vars * input_per_var]
        has_time = x.dim() == 3
        if has_time:
            batch, time, _ = x.shape
            flat = x.reshape(batch * time, -1)
        else:
            flat = x
            batch = x.shape[0]
            time = 1

        weights = F.softmax(self.flattened_grn(flat, context.reshape(-1, context.shape[-1]) if context is not None and has_time else context), dim=-1)
        var_size = flat.shape[-1] // self.num_vars
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_input = flat[:, i*var_size:(i+1)*var_size]
            var_outputs.append(grn(var_input))
        var_outputs = torch.stack(var_outputs, dim=-1)  # [B*T, hidden, num_vars]
        weights = weights.unsqueeze(1)  # [B*T, 1, num_vars]
        output = (var_outputs * weights).sum(dim=-1)  # [B*T, hidden]

        if has_time:
            output = output.reshape(batch, time, -1)
            weights = weights.reshape(batch, time, -1)
        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, self.d_k)
        self.W_o = nn.Linear(self.d_k, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch = q.shape[0]
        Q = self.W_q(q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.mean(dim=1)  # Average across heads for interpretability
        output = self.W_o(context)
        return output, attn.mean(dim=1)  # Return averaged attention weights


class TFTPredictor(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon time series prediction.
    
    Features:
        - Variable Selection Networks for feature importance
        - Gated Residual Networks for flexible nonlinearity
        - LSTM encoder-decoder for temporal patterns
        - Interpretable multi-head attention for long-range dependencies
        - Quantile outputs for uncertainty estimation
    """

    def __init__(self, n_features: int, config=None):
        super().__init__()
        c = config or cfg.tft
        self.hidden_size = c.hidden_size
        self.n_features = n_features
        self.pred_len = c.max_prediction_length
        self.num_quantiles = c.output_size

        # Input projection
        self.input_proj = nn.Linear(n_features, c.hidden_size)

        # Variable selection
        self.vsn = VariableSelectionNetwork(
            input_size=1, num_vars=n_features,
            hidden_size=c.hidden_size, dropout=c.dropout
        )

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=c.hidden_size, hidden_size=c.hidden_size,
            num_layers=c.lstm_layers, batch_first=True, dropout=c.dropout
        )

        # Gated skip connection
        self.gate1 = GatedLinearUnit(c.hidden_size, c.hidden_size, c.dropout)
        self.norm1 = nn.LayerNorm(c.hidden_size)

        # Self-attention
        self.attention = InterpretableMultiHeadAttention(
            c.hidden_size, c.num_attention_heads, c.dropout
        )

        # Post-attention processing
        self.gate2 = GatedLinearUnit(c.hidden_size, c.hidden_size, c.dropout)
        self.norm2 = nn.LayerNorm(c.hidden_size)

        # Position-wise feedforward
        self.grn_final = GatedResidualNetwork(c.hidden_size, c.hidden_size, dropout=c.dropout)

        # Output heads
        # Regression: predict future returns (quantiles)
        quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        self.quantile_heads = nn.ModuleList([
            nn.Linear(c.hidden_size, self.pred_len) for _ in quantiles
        ])
        self.quantiles = quantiles

        # Classification: predict direction (down, neutral, up)
        self.direction_head = nn.Linear(c.hidden_size, 3)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(c.hidden_size, c.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(c.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, n_features]
            
        Returns:
            Dict with keys: quantiles, direction, confidence, attention_weights
        """
        batch, seq_len, _ = x.shape

        # Input projection
        h = self.input_proj(x)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(h)

        # Gated skip connection
        gated = self.gate1(lstm_out)
        h = self.norm1(gated + h)

        # Self-attention
        attn_out, attn_weights = self.attention(h, h, h)

        # Post-attention gate
        gated2 = self.gate2(attn_out)
        h = self.norm2(gated2 + h)

        # Final GRN
        h = self.grn_final(h)

        # Use last timestep for prediction
        last = h[:, -1, :]  # [batch, hidden]

        # Quantile regression outputs
        q_outputs = torch.stack([head(last) for head in self.quantile_heads], dim=-1)
        # [batch, pred_len, num_quantiles]

        # Direction classification
        direction = self.direction_head(last)  # [batch, 3]

        # Confidence
        confidence = self.confidence_head(last)  # [batch, 1]

        return {
            "quantiles": q_outputs,
            "direction": direction,
            "confidence": confidence.squeeze(-1),
            "attention_weights": attn_weights,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Inference-mode prediction returning numpy arrays."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        return {
            "median_forecast": out["quantiles"][:, :, 3].cpu().numpy(),  # 0.5 quantile
            "lower_bound": out["quantiles"][:, :, 0].cpu().numpy(),      # 0.02 quantile
            "upper_bound": out["quantiles"][:, :, -1].cpu().numpy(),     # 0.98 quantile
            "direction_probs": F.softmax(out["direction"], dim=-1).cpu().numpy(),
            "confidence": out["confidence"].cpu().numpy(),
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class QuantileLoss(nn.Module):
    """Quantile regression loss function."""
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions, targets):
        # predictions: [batch, pred_len, num_quantiles]
        # targets: [batch] — expand to match
        targets = targets.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
        errors = targets - predictions
        losses = []
        for i, q in enumerate(self.quantiles):
            e = errors[:, :, i]
            loss = torch.max(q * e, (q - 1) * e)
            losses.append(loss.mean())
        return sum(losses) / len(losses)


if __name__ == "__main__":
    model = TFTPredictor(n_features=64)
    print(f"TFT Parameters: {model.get_param_count():,}")
    x = torch.randn(4, 60, 64)
    out = model(x)
    print(f"Quantiles shape: {out['quantiles'].shape}")
    print(f"Direction shape: {out['direction'].shape}")
    print(f"Confidence shape: {out['confidence'].shape}")
