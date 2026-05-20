"""
Model Trainer — V2 High-Performance Training Pipeline
========================================================
Focal loss, cosine annealing warmup, class balancing,
label smoothing, gradient clipping, NaN safety.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, CHECKPOINT_DIR, LOG_DIR
from models.tft_predictor import TFTPredictor, QuantileLoss
from models.pattern_cnn import PatternCNN
from models.regime_lstm import RegimeLSTM
from features.strategy_confluence import StrategyConfluence


def _sanitize_array_inplace(arr, nan_val=0.0, posinf_val=10.0, neginf_val=-10.0, chunk_size=10000):
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


# ── Custom Loss Functions ────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, focuses on hard ones."""
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        n_classes = logits.size(-1)
        smooth_targets = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / n_classes

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        loss = -(focal_weight * smooth_targets * log_probs).sum(dim=-1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


class HuberQuantileLoss(nn.Module):
    """Huber quantile loss — robust to outliers."""
    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), delta=0.5):
        super().__init__()
        self.quantiles = quantiles
        self.delta = delta

    def forward(self, preds, targets):
        # preds: [batch, horizon, n_quantiles] or [batch, n_quantiles]
        # targets: [batch] or [batch, horizon]
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)

        if targets.dim() == 1:
            if preds.dim() == 3:
                # Expand [batch] → [batch, horizon, 1] for 3D preds
                targets = targets.view(-1, 1, 1).expand(-1, preds.size(1), preds.size(2))
            else:
                targets = targets.unsqueeze(-1).expand_as(preds)
        elif targets.dim() == 2 and preds.dim() == 3:
            targets = targets.unsqueeze(-1).expand_as(preds)

        losses = []
        for i, q in enumerate(self.quantiles):
            if i < preds.size(-1):
                error = targets[..., min(i, targets.size(-1)-1)] - preds[..., i]
                abs_error = torch.abs(error)
                # Huber: quadratic for small errors, linear for large
                huber = torch.where(abs_error <= self.delta,
                                     0.5 * error ** 2,
                                     self.delta * (abs_error - 0.5 * self.delta))
                # Asymmetric weighting for quantiles
                weight = torch.where(error >= 0, q, 1 - q)
                losses.append((weight * huber).mean())

        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=preds.device)


# ── Learning Rate Scheduler with Warmup ──────────────────────────────────────

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


# ── Trainer ──────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    V2 GPU-optimized training with:
    - Focal loss (handles class imbalance)
    - Cosine annealing with warmup
    - Class-balanced sampling
    - Label smoothing
    - Gradient clipping
    - NaN-safe training loop
    - STRATEGY-GUIDED TRAINING (uses 8 strategies as supervision)
    """

    def __init__(self, device=None):
        self.device = device or cfg.gpu.device
        self.scaler = GradScaler("cuda", enabled=(self.device == "cuda"))
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.strategy_confluence = StrategyConfluence()  # For strategy-guided training

    def _compute_class_weights(self, labels):
        """Compute inverse-frequency weights for class balancing."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = {int(u): total / (len(unique) * c) for u, c in zip(unique, counts)}
        # Normalize so max weight = 3.0 (prevent extreme weighting)
        max_w = max(weights.values())
        weights = {k: min(v / max_w * 2.0, 3.0) for k, v in weights.items()}
        return weights

    def train_tft(self, train_data: dict, val_data: dict, n_features: int, feature_columns: List[str] = None) -> TFTPredictor:
        """Train the TFT model with STRATEGY-GUIDED training."""
        model = TFTPredictor(n_features=n_features).to(self.device)
        logger.info(f"TFT model: {model.get_param_count():,} parameters")

        # Compute class weights for direction labels
        cls_labels = train_data["y_classification"] + 1  # Shift to [0,1,2]
        class_weights = self._compute_class_weights(cls_labels)
        weight_tensor = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(3)]).to(self.device)
        
        # Compute STRATEGY TARGETS - model learns to predict what strategies say
        strategy_targets, strategy_confidence = self._compute_strategy_targets(
            train_data["X"], feature_columns
        )
        logger.info(f"  Strategy-guided training: {strategy_targets.shape[0]} samples")

        # Optimizer with lower learning rate for stability
        lr = cfg.tft.learning_rate * 0.5  # Reduce LR for more features
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay * 2)

        # Scheduler: warmup + cosine decay
        steps_per_epoch = max(1, len(train_data["X"]) // cfg.training.batch_size)
        total_steps = steps_per_epoch * cfg.training.max_epochs
        warmup_steps = steps_per_epoch * 3  # 3 epochs warmup
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

        # Loss functions
        q_loss = HuberQuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])
        focal = FocalLoss(gamma=2.0, alpha=weight_tensor, label_smoothing=0.1)

        train_loader = self._make_loader(train_data["X"], train_data["y_regression"], train_data["y_classification"])
        val_loader = self._make_loader(val_data["X"], val_data["y_regression"], val_data["y_classification"], shuffle=False)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        best_val_acc = 0

        # Create strategy-weighted sample weights
        strategy_weights = torch.FloatTensor(strategy_confidence).to(self.device)
        
        for epoch in range(cfg.training.max_epochs):
            model.train()
            train_loss = 0; n_batches = 0; correct = 0; total = 0; strat_correct = 0
            optimizer.zero_grad()

            for step, (X, y_reg, y_cls) in enumerate(train_loader):
                X = X.to(self.device)
                y_reg = y_reg.to(self.device)
                y_cls = (y_cls + 1).long().clamp(0, 2).to(self.device)
                
                # Get strategy targets for this batch
                batch_idx = step * cfg.training.batch_size
                batch_strat_tgt = torch.LongTensor(
                    strategy_targets[batch_idx:batch_idx + cfg.training.batch_size]
                ).to(self.device) if len(strategy_targets) > batch_idx else None
                batch_strat_conf = torch.FloatTensor(
                    strategy_confidence[batch_idx:batch_idx + cfg.training.batch_size]
                ).to(self.device) if len(strategy_confidence) > batch_idx else None

                if torch.isnan(X).any():
                    continue

                with autocast("cuda", enabled=(self.device == "cuda")):
                    out = model(X)
                    loss_q = q_loss(out["quantiles"], y_reg)
                    loss_d = focal(out["direction"], y_cls)
                    
                    # STRATEGY-GUIDED LOSS: Learn to predict strategy confluence
                    loss_strategy = 0
                    if batch_strat_tgt is not None and batch_strat_conf is not None:
                        # Only use high-confidence strategy signals as supervision
                        high_conf_mask = batch_strat_conf > 0.5
                        if high_conf_mask.sum() > 0:
                            loss_strategy = F.cross_entropy(
                                out["direction"][high_conf_mask],
                                batch_strat_tgt[high_conf_mask],
                                reduction='mean'
                            )
                    
                    # Combined loss: price prediction + strategy alignment
                    loss = loss_q + 0.5 * loss_d + 0.3 * loss_strategy
                    loss = loss / cfg.gpu.gradient_accumulation_steps

                if torch.isnan(loss):
                    optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()
                if (step + 1) % cfg.gpu.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter clipping
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                train_loss += loss.item() * cfg.gpu.gradient_accumulation_steps
                n_batches += 1

                # Track accuracy
                pred = out["direction"].argmax(dim=-1)
                correct += (pred == y_cls).sum().item()
                total += len(y_cls)
                
                # Track strategy agreement accuracy
                if batch_strat_tgt is not None:
                    strat_correct += (pred == batch_strat_tgt).sum().item()

            train_loss /= max(n_batches, 1)
            train_acc = correct / max(total, 1)
            strat_acc = strat_correct / max(total, 1)

            # Validate
            val_loss, val_acc = self._validate_tft_v2(model, val_loader, q_loss, focal)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  TFT Epoch {epoch+1:>3}/{cfg.training.max_epochs} | "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc: {train_acc:.1%}/{val_acc:.1%} | "
                    f"Strategy Acc: {strat_acc:.1%} | "
                    f"LR: {lr_now:.2e}"
                )

            # Early stopping on BOTH loss and accuracy
            score = val_loss - 0.1 * val_acc  # Reward lower loss AND higher accuracy
            if score < self.best_val_loss - cfg.training.min_delta:
                self.best_val_loss = score
                best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_DIR / "tft_best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.training.early_stopping_patience:
                    logger.info(f"  TFT early stopping at epoch {epoch+1}")
                    break

        best_path = CHECKPOINT_DIR / "tft_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, weights_only=True))
        logger.success(f"TFT complete. Best val acc: {best_val_acc:.1%}")
        return model

    def train_cnn(self, train_data: dict, val_data: dict, n_features: int, feature_columns: List[str] = None) -> PatternCNN:
        """Train CNN with STRATEGY-GUIDED training for pattern recognition.
        
        Issue #9 (Ensemble Diversity): CNN only receives OHLCV + pattern features
        to specialize in pattern recognition rather than duplicating TFT.
        """
        # Feature subsetting for ensemble diversity (Issue #9)
        feature_mask = self._get_feature_mask("cnn", feature_columns)
        if feature_mask is not None:
            train_X = train_data["X"][:, :, feature_mask]
            val_X = val_data["X"][:, :, feature_mask]
            actual_n_features = feature_mask.sum()
            masked_columns = [feature_columns[i] for i in range(len(feature_columns)) if feature_mask[i]] if feature_columns else None
            logger.info(f"  CNN Ensemble Diversity: Using {actual_n_features}/{n_features} features (pattern-focused)")
        else:
            train_X = train_data["X"]
            val_X = val_data["X"]
            actual_n_features = n_features
            masked_columns = feature_columns
        
        model = PatternCNN(n_features=actual_n_features).to(self.device)
        logger.info(f"CNN model: {model.get_param_count():,} parameters")

        cls_labels = train_data["y_classification"] + 1
        class_weights = self._compute_class_weights(cls_labels)
        weight_tensor = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(3)]).to(self.device)
        
        # Compute STRATEGY TARGETS
        strategy_targets, strategy_confidence = self._compute_strategy_targets(
            train_data["X"], feature_columns  # Use full features for strategy analysis
        )
        logger.info(f"  CNN Strategy-guided training: {strategy_targets.shape[0]} samples")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.cnn.learning_rate * 0.5, weight_decay=0.01)
        steps_per_epoch = max(1, len(train_X) // cfg.training.batch_size)
        scheduler = CosineWarmupScheduler(optimizer, steps_per_epoch * 2, steps_per_epoch * cfg.training.max_epochs)
        focal = FocalLoss(gamma=2.0, alpha=weight_tensor, label_smoothing=0.1)

        # Use masked features for data loaders
        train_masked = {"X": train_X, "y_regression": train_data["y_regression"], "y_classification": train_data["y_classification"]}
        val_masked = {"X": val_X, "y_regression": val_data["y_regression"], "y_classification": val_data["y_classification"]}
        train_loader = self._make_loader(train_masked["X"], train_masked["y_regression"], train_masked["y_classification"])
        val_loader = self._make_loader(val_masked["X"], val_masked["y_regression"], val_masked["y_classification"], shuffle=False)


        self.best_val_loss = float("inf"); self.patience_counter = 0
        best_acc = 0

        for epoch in range(cfg.training.max_epochs):
            model.train(); train_loss = 0; n = 0; correct = 0; total = 0; strat_correct = 0
            for step, (X, _, y_cls) in enumerate(train_loader):
                X, y_cls = X.to(self.device), (y_cls + 1).long().clamp(0, 2).to(self.device)
                
                # Get strategy targets for this batch
                batch_idx = step * cfg.training.batch_size
                batch_strat_tgt = torch.LongTensor(
                    strategy_targets[batch_idx:batch_idx + cfg.training.batch_size]
                ).to(self.device) if len(strategy_targets) > batch_idx else None
                batch_strat_conf = torch.FloatTensor(
                    strategy_confidence[batch_idx:batch_idx + cfg.training.batch_size]
                ).to(self.device) if len(strategy_confidence) > batch_idx else None
                
                with autocast("cuda", enabled=(self.device == "cuda")):
                    out = model(X)
                    loss_d = focal(out["direction"], y_cls)
                    
                    # STRATEGY-GUIDED LOSS
                    loss_strategy = 0
                    if batch_strat_tgt is not None and batch_strat_conf is not None:
                        high_conf_mask = batch_strat_conf > 0.5
                        if high_conf_mask.sum() > 0:
                            loss_strategy = F.cross_entropy(
                                out["direction"][high_conf_mask],
                                batch_strat_tgt[high_conf_mask],
                                reduction='mean'
                            )
                    
                    loss = loss_d + 0.3 * loss_strategy
                    
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                train_loss += loss.item(); n += 1
                correct += (out["direction"].argmax(-1) == y_cls).sum().item()
                total += len(y_cls)
                if batch_strat_tgt is not None:
                    strat_correct += (out["direction"].argmax(-1) == batch_strat_tgt).sum().item()

            model.eval(); val_loss = 0; vn = 0; vc = 0; vt = 0
            with torch.no_grad():
                for X, _, y_cls in val_loader:
                    X, y_cls = X.to(self.device), (y_cls + 1).long().clamp(0, 2).to(self.device)
                    out = model(X)
                    val_loss += focal(out["direction"], y_cls).item(); vn += 1
                    vc += (out["direction"].argmax(-1) == y_cls).sum().item()
                    vt += len(y_cls)
            val_loss /= max(vn, 1)
            val_acc = vc / max(vt, 1)
            strat_acc = strat_correct / max(total, 1)

            if (epoch+1) % 10 == 0:
                logger.info(f"  CNN Epoch {epoch+1} | Loss: {train_loss/max(n,1):.4f}/{val_loss:.4f} | Acc: {correct/max(total,1):.1%}/{val_acc:.1%} | Strategy Acc: {strat_acc:.1%}")

            score = val_loss - 0.1 * val_acc
            if score < self.best_val_loss - cfg.training.min_delta:
                self.best_val_loss = score; self.patience_counter = 0; best_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_DIR / "cnn_best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.training.early_stopping_patience:
                    break

        best_path = CHECKPOINT_DIR / "cnn_best.pt"
        if best_path.exists(): model.load_state_dict(torch.load(best_path, weights_only=True))
        logger.success(f"CNN complete. Best val acc: {best_acc:.1%}")
        return model

    def train_lstm(self, train_data: dict, val_data: dict, n_features: int, feature_columns: List[str] = None) -> RegimeLSTM:
        """Train LSTM with STRATEGY-GUIDED regime detection.
        
        Issue #9 (Ensemble Diversity): LSTM only receives momentum/volatility/structure
        features to specialize in regime detection.
        """
        # Feature subsetting for ensemble diversity (Issue #9)
        feature_mask = self._get_feature_mask("lstm", feature_columns)
        if feature_mask is not None:
            train_X = train_data["X"][:, :, feature_mask]
            val_X = val_data["X"][:, :, feature_mask]
            actual_n_features = feature_mask.sum()
            masked_columns = [feature_columns[i] for i in range(len(feature_columns)) if feature_mask[i]] if feature_columns else None
            logger.info(f"  LSTM Ensemble Diversity: Using {actual_n_features}/{n_features} features (regime-focused)")
        else:
            train_X = train_data["X"]
            val_X = val_data["X"]
            actual_n_features = n_features
            masked_columns = feature_columns
        
        model = RegimeLSTM(n_features=actual_n_features).to(self.device)
        logger.info(f"LSTM model: {model.get_param_count():,} parameters")

        train_labels = self._create_regime_labels(train_data["y_regression"])
        val_labels = self._create_regime_labels(val_data["y_regression"])

        class_weights = self._compute_class_weights(train_labels)
        weight_tensor = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(4)]).to(self.device)
        
        # Compute STRATEGY-BASED REGIME TARGETS
        strategy_regime_targets, strategy_confidence = self._compute_strategy_regime_targets(
            train_data["X"], feature_columns  # Use full features for strategy analysis
        )
        logger.info(f"  LSTM Strategy-guided training: {strategy_regime_targets.shape[0]} samples")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lstm.learning_rate * 0.5, weight_decay=0.01)
        steps_per_epoch = max(1, len(train_X) // cfg.training.batch_size)
        scheduler = CosineWarmupScheduler(optimizer, steps_per_epoch * 2, steps_per_epoch * cfg.training.max_epochs)
        focal = FocalLoss(gamma=2.0, alpha=weight_tensor, label_smoothing=0.05)

        # Use masked features for data loaders
        train_masked_loader = self._make_loader(train_X, train_data["y_regression"], train_labels)
        val_masked_loader = self._make_loader(val_X, val_data["y_regression"], val_labels, shuffle=False)

        train_loader = train_masked_loader
        val_loader = val_masked_loader

        self.best_val_loss = float("inf"); self.patience_counter = 0; best_acc = 0

        for epoch in range(cfg.training.max_epochs):
            model.train(); train_loss = 0; n = 0; correct = 0; total = 0; strat_correct = 0
            for step, (X, _, y_regime) in enumerate(train_loader):
                X, y_regime = X.to(self.device), y_regime.long().to(self.device)
                
                # Get strategy regime targets for this batch
                batch_idx = step * cfg.training.batch_size
                batch_strat_regime = torch.LongTensor(
                    strategy_regime_targets[batch_idx:batch_idx + cfg.training.batch_size]
                ).to(self.device) if len(strategy_regime_targets) > batch_idx else None
                batch_strat_conf = torch.FloatTensor(
                    strategy_confidence[batch_idx:batch_idx + cfg.training.batch_size]
                ).to(self.device) if len(strategy_confidence) > batch_idx else None
                
                with autocast("cuda", enabled=(self.device == "cuda")):
                    out = model(X)
                    loss_r = focal(out["regime_logits"], y_regime)
                    
                    # STRATEGY-GUIDED LOSS for regime detection
                    loss_strategy = 0
                    if batch_strat_regime is not None and batch_strat_conf is not None:
                        high_conf_mask = batch_strat_conf > 0.5
                        if high_conf_mask.sum() > 0:
                            loss_strategy = F.cross_entropy(
                                out["regime_logits"][high_conf_mask],
                                batch_strat_regime[high_conf_mask],
                                reduction='mean'
                            )
                    
                    loss = loss_r + 0.3 * loss_strategy
                    
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                train_loss += loss.item(); n += 1
                correct += (out["regime_logits"].argmax(-1) == y_regime).sum().item()
                total += len(y_regime)
                if batch_strat_regime is not None:
                    strat_correct += (out["regime_logits"].argmax(-1) == batch_strat_regime).sum().item()

            model.eval(); val_loss = 0; vn = 0; vc = 0; vt = 0
            with torch.no_grad():
                for X, _, y_regime in val_loader:
                    X, y_regime = X.to(self.device), y_regime.long().to(self.device)
                    out = model(X)
                    val_loss += focal(out["regime_logits"], y_regime).item(); vn += 1
                    vc += (out["regime_logits"].argmax(-1) == y_regime).sum().item()
                    vt += len(y_regime)
            val_loss /= max(vn, 1)
            val_acc = vc / max(vt, 1)
            strat_acc = strat_correct / max(total, 1)

            if (epoch+1) % 10 == 0:
                logger.info(f"  LSTM Epoch {epoch+1} | Loss: {train_loss/max(n,1):.4f}/{val_loss:.4f} | Acc: {correct/max(total,1):.1%}/{val_acc:.1%} | Strategy Acc: {strat_acc:.1%}")

            score = val_loss - 0.1 * val_acc
            if score < self.best_val_loss - cfg.training.min_delta:
                self.best_val_loss = score; self.patience_counter = 0; best_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_DIR / "lstm_best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.training.early_stopping_patience:
                    break

        best_path = CHECKPOINT_DIR / "lstm_best.pt"
        if best_path.exists(): model.load_state_dict(torch.load(best_path, weights_only=True))
        logger.success(f"LSTM complete. Best val acc: {best_acc:.1%}")
        return model

    def _validate_tft_v2(self, model, val_loader, q_loss, focal):
        """Validate TFT and return both loss and accuracy."""
        model.eval()
        total_loss = 0; n = 0; correct = 0; total_samples = 0
        with torch.no_grad():
            for X, y_reg, y_cls in val_loader:
                X = X.to(self.device)
                y_reg = y_reg.to(self.device)
                y_cls = (y_cls + 1).long().clamp(0, 2).to(self.device)
                out = model(X)
                loss = q_loss(out["quantiles"], y_reg) + 0.5 * focal(out["direction"], y_cls)
                total_loss += loss.item(); n += 1
                correct += (out["direction"].argmax(-1) == y_cls).sum().item()
                total_samples += len(y_cls)
        return total_loss / max(n, 1), correct / max(total_samples, 1)

    def _make_loader(self, X, y_reg, y_cls, shuffle=True):
        X = _sanitize_array_inplace(X, nan_val=0.0, posinf_val=10.0, neginf_val=-10.0)
        y_reg = _sanitize_array_inplace(y_reg, nan_val=0.0, posinf_val=1.0, neginf_val=-1.0)
        y_cls = _sanitize_array_inplace(y_cls, nan_val=0.0, posinf_val=0.0, neginf_val=0.0)
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y_reg), torch.LongTensor(y_cls))
        return DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=shuffle, num_workers=0, pin_memory=cfg.gpu.pin_memory)

    def _get_feature_mask(self, model_type: str, feature_columns: List[str]) -> Optional[np.ndarray]:
        """
        Get a boolean feature mask for ensemble diversity (Issue #9).
        
        - TFT: ALL features (returns None → no masking)
        - CNN: OHLCV + pattern/candlestick features (pattern recognition specialist)
        - LSTM: Momentum + volatility + market structure (regime detection specialist)
        
        Returns None if feature_columns is empty or model_type is TFT.
        """
        if not feature_columns or model_type == "tft":
            return None
        
        n = len(feature_columns)
        mask = np.zeros(n, dtype=bool)
        
        # Always include base OHLCV for all models
        base_cols = {"Open", "High", "Low", "Close", "Volume", "Returns", "Log_Returns"}
        
        if model_type == "cnn":
            # CNN specializes in pattern recognition
            # Include: OHLCV, candlestick patterns, price action, SMC, ICT
            pattern_keywords = [
                "open", "high", "low", "close", "volume", "returns", "log_returns",
                "candle", "pattern", "doji", "hammer", "engulf", "star", "marubozu",
                "body", "shadow", "wick", "gap",
                "smc", "ict", "ob_", "fvg_", "bos_", "choch",
                "support", "resistance", "swing", "fractal",
                "price_action", "pin_bar", "inside_bar",
                "bb_", "bollinger", "keltner",
                "atr", "vp_",  # Volume profile
            ]
            for i, col in enumerate(feature_columns):
                col_lower = col.lower()
                if col in base_cols or any(kw in col_lower for kw in pattern_keywords):
                    mask[i] = True
                    
        elif model_type == "lstm":
            # LSTM specializes in regime/trend detection
            # Include: momentum, volatility, structure, temporal, multi-TF
            regime_keywords = [
                "open", "high", "low", "close", "volume", "returns", "log_returns",
                "rsi", "macd", "stoch", "cci", "williams", "mfi", "roc", "momentum",
                "sma", "ema", "trend", "adx", "di_plus", "di_minus",
                "atr", "volatility", "std", "variance",
                "regime", "wyckoff", "elliott", "wave", "phase",
                "mtf_", "corr_", "relstrength", "vix",
                "hour_", "dayof", "session", "minute",
                "sentiment",
            ]
            for i, col in enumerate(feature_columns):
                col_lower = col.lower()
                if col in base_cols or any(kw in col_lower for kw in regime_keywords):
                    mask[i] = True
        
        # Ensure at least 30% of features are selected (fallback)
        if mask.sum() < n * 0.3:
            return None
        
        return mask

    def _create_regime_labels(self, returns: np.ndarray) -> np.ndarray:
        """Create regime labels: 0=bull, 1=bear, 2=sideways, 3=volatile."""
        labels = np.zeros(len(returns), dtype=np.int64)
        abs_ret = np.abs(returns)
        median_ret = np.median(abs_ret[abs_ret > 0]) if (abs_ret > 0).any() else 0.01
        for i in range(len(returns)):
            r = returns[i]; ar = abs_ret[i]
            if ar > median_ret * 2:
                labels[i] = 3
            elif r > median_ret * 0.5:
                labels[i] = 0
            elif r < -median_ret * 0.5:
                labels[i] = 1
            else:
                labels[i] = 2
        return labels
    
    def _compute_strategy_targets(
        self, 
        features: np.ndarray, 
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute strategy confluence targets for training.
        
        Issue #8 (Circularity Fix): Instead of computing strategy signals on the
        exact same features the model receives, we use a TIME-OFFSET view.
        We look at an earlier timestep in the sequence (not the last) to compute
        the strategy signal. This means the model must learn to predict what the
        strategies WOULD say from a different temporal perspective, breaking the
        circularity.
        
        Returns:
            targets: [N] class indices (0=SELL, 1=HOLD, 2=BUY) from strategy confluence
            confidence: [N] strategy confidence (0-1)
        """
        if feature_columns is None or len(feature_columns) == 0:
            return np.zeros(len(features), dtype=np.int64), np.zeros(len(features), dtype=np.float32)
        
        targets = []
        confidences = []
        
        # Use a time-offset: compute strategy on a DIFFERENT timestep than the
        # last bar to break circularity. The model sees the full sequence but
        # the strategy target comes from a mid-sequence view.
        seq_len = features.shape[1] if features.ndim >= 2 else 1
        # Use bar at ~75% of sequence (earlier temporal context)
        offset_idx = max(0, int(seq_len * 0.5) - 1)
        
        for i in range(len(features)):
            # Use mid-sequence bar instead of last bar (circularity fix)
            bar = features[i][offset_idx] if features.ndim >= 2 else features[i]
            
            if len(bar) == len(feature_columns):
                feature_dict = dict(zip(feature_columns, bar))
                result = self.strategy_confluence.analyze(feature_dict)
                signal = result["signal"]
                target = signal + 1
                confidence = result["confidence"]
            else:
                target = 1
                confidence = 0.0
            
            targets.append(target)
            confidences.append(confidence)
        
        return np.array(targets, dtype=np.int64), np.array(confidences, dtype=np.float32)
    
    def _compute_strategy_regime_targets(
        self,
        features: np.ndarray,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute strategy-based regime targets for LSTM training.
        
        Uses strategy confluence to determine market regime:
        - 0 (bull): Multiple bullish strategies with high confidence
        - 1 (bear): Multiple bearish strategies with high confidence  
        - 2 (sideways): Low agreement or mostly HOLD signals
        - 3 (volatile): High strategy disagreement
        
        Returns:
            targets: [N] regime indices (0=bull, 1=bear, 2=sideways, 3=volatile)
            confidence: [N] strategy confidence (0-1)
        """
        if feature_columns is None or len(feature_columns) == 0:
            return np.zeros(len(features), dtype=np.int64), np.zeros(len(features), dtype=np.float32)
        
        targets = []
        confidences = []
        
        for i in range(len(features)):
            last_bar = features[i][-1]
            
            if len(last_bar) == len(feature_columns):
                feature_dict = dict(zip(feature_columns, last_bar))
                result = self.strategy_confluence.analyze(feature_dict)
                
                bull_count = result.get("bull_count", 0)
                bear_count = result.get("bear_count", 0)
                confidence = result["confidence"]
                
                # Determine regime from strategy agreement
                if bull_count >= 5:
                    regime = 0  # bull
                elif bear_count >= 5:
                    regime = 1  # bear
                elif bull_count >= 3 and bear_count >= 3:
                    regime = 3  # volatile (disagreement)
                elif confidence < 0.3:
                    regime = 2  # sideways
                elif bull_count > bear_count:
                    regime = 0  # bull
                elif bear_count > bull_count:
                    regime = 1  # bear
                else:
                    regime = 2  # sideways
            else:
                regime = 2  # sideways
                confidence = 0.0
            
            targets.append(regime)
            confidences.append(confidence)
        
        return np.array(targets, dtype=np.int64), np.array(confidences, dtype=np.float32)
