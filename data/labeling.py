"""
Triple Barrier Labeling — Realistic Trade Outcome Labels
==========================================================
Labels each bar based on which barrier the price hits first:
  - Take Profit barrier → label = +1
  - Stop Loss barrier → label = -1
  - Time Expiry → label based on P&L at expiry

This aligns training labels with actual trading outcomes,
unlike simple pct_change which ignores the path.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class TripleBarrierLabeler:
    """
    Triple Barrier Method for labeling trading data.
    
    For each bar, simulates forward through future bars and checks
    which barrier (take-profit, stop-loss, or time expiry) is hit first.
    
    Barriers can be fixed or ATR-scaled (dynamic).
    """

    def __init__(
        self,
        tp_atr_mult: float = None,
        sl_atr_mult: float = None,
        max_holding_period: int = None,
        atr_window: int = 14,
        fixed_tp_pct: float = None,
        fixed_sl_pct: float = None,
    ):
        """
        Args:
            tp_atr_mult: Take-profit barrier as multiple of ATR (default from config)
            sl_atr_mult: Stop-loss barrier as multiple of ATR (default from config)
            max_holding_period: Max bars to hold (default: prediction_horizon)
            atr_window: ATR calculation window
            fixed_tp_pct: Fixed take-profit percentage (overrides ATR-based if set)
            fixed_sl_pct: Fixed stop-loss percentage (overrides ATR-based if set)
        """
        self.tp_atr_mult = tp_atr_mult or getattr(cfg.data, 'tp_atr_multiplier', 2.0)
        self.sl_atr_mult = sl_atr_mult or getattr(cfg.data, 'sl_atr_multiplier', 1.0)
        self.max_holding_period = max_holding_period or cfg.data.prediction_horizon
        self.atr_window = atr_window
        self.fixed_tp_pct = fixed_tp_pct
        self.fixed_sl_pct = fixed_sl_pct

    def label(
        self,
        df: pd.DataFrame,
        threshold: float = 0.002,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply triple barrier labeling to a DataFrame.
        
        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns
            threshold: Minimum P&L at expiry to classify as +1/-1 (otherwise 0)
            
        Returns:
            y_regression: float array of actual P&L at barrier touch [N]
            y_classification: int array of direction labels (-1, 0, +1) [N]
        """
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        n = len(close)

        # Compute ATR for dynamic barriers
        atr = self._compute_atr(df)

        y_reg = np.zeros(n, dtype=np.float32)
        y_cls = np.zeros(n, dtype=np.int64)

        for i in range(n):
            entry_price = close[i]
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            # Determine barrier levels
            if self.fixed_tp_pct is not None:
                tp_level = entry_price * (1 + self.fixed_tp_pct)
                sl_level = entry_price * (1 - (self.fixed_sl_pct or self.fixed_tp_pct))
            else:
                atr_val = atr[i] if not np.isnan(atr[i]) else entry_price * 0.01
                tp_level = entry_price + self.tp_atr_mult * atr_val
                sl_level = entry_price - self.sl_atr_mult * atr_val

            # Walk forward through future bars
            barrier_hit = 0  # 0 = no barrier, 1 = TP, -1 = SL
            barrier_pnl = 0.0
            max_bars = min(self.max_holding_period, n - i - 1)

            for j in range(1, max_bars + 1):
                future_idx = i + j
                if future_idx >= n:
                    break

                future_high = high[future_idx]
                future_low = low[future_idx]
                future_close = close[future_idx]

                # Check stop-loss first (conservative: assume worst case hit first)
                if future_low <= sl_level:
                    barrier_hit = -1
                    barrier_pnl = (sl_level - entry_price) / entry_price
                    break

                # Check take-profit
                if future_high >= tp_level:
                    barrier_hit = 1
                    barrier_pnl = (tp_level - entry_price) / entry_price
                    break

            # If no barrier was hit → time expiry
            if barrier_hit == 0 and max_bars > 0:
                expiry_idx = min(i + max_bars, n - 1)
                expiry_price = close[expiry_idx]
                barrier_pnl = (expiry_price - entry_price) / entry_price

                # Classify based on P&L at expiry
                if barrier_pnl > threshold:
                    barrier_hit = 1
                elif barrier_pnl < -threshold:
                    barrier_hit = -1
                else:
                    barrier_hit = 0

            y_reg[i] = barrier_pnl
            y_cls[i] = barrier_hit

        # Stats
        n_tp = (y_cls == 1).sum()
        n_sl = (y_cls == -1).sum()
        n_exp = (y_cls == 0).sum()
        logger.debug(
            f"  Triple Barrier: {n_tp} TP ({n_tp/max(n,1):.1%}), "
            f"{n_sl} SL ({n_sl/max(n,1):.1%}), "
            f"{n_exp} Expiry ({n_exp/max(n,1):.1%})"
        )

        return y_reg, y_cls

    def _compute_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Compute Average True Range."""
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        n = len(close)
        tr = np.zeros(n)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Simple moving average of TR
        atr = np.zeros(n)
        for i in range(self.atr_window, n):
            atr[i] = np.mean(tr[i - self.atr_window + 1 : i + 1])

        # Fill initial values with the first valid ATR
        first_valid = atr[self.atr_window] if self.atr_window < n else 0
        atr[:self.atr_window] = first_valid

        return atr


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open": prices + np.random.randn(n) * 0.1,
        "High": prices + abs(np.random.randn(n) * 0.3),
        "Low": prices - abs(np.random.randn(n) * 0.3),
        "Close": prices,
    })

    labeler = TripleBarrierLabeler(tp_atr_mult=2.0, sl_atr_mult=1.0, max_holding_period=12)
    y_reg, y_cls = labeler.label(df)

    print(f"\nTriple Barrier Labels:")
    print(f"  Take Profit: {(y_cls == 1).sum()}")
    print(f"  Stop Loss:   {(y_cls == -1).sum()}")
    print(f"  Time Expiry: {(y_cls == 0).sum()}")
    print(f"  Avg P&L at TP: {y_reg[y_cls == 1].mean():.4f}")
    print(f"  Avg P&L at SL: {y_reg[y_cls == -1].mean():.4f}")
