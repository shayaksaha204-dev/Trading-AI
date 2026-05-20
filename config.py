"""
Trading AI — Central Configuration
===================================
All system-wide settings in one place.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data" / "cache"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"

# Create directories
for d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── GPU Configuration ────────────────────────────────────────────────────────
@dataclass
class GPUConfig:
    """RTX 5070 optimized settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True           # FP16/BF16 training
    bf16: bool = True                      # Use BF16 if available (Blackwell)
    gradient_checkpointing: bool = True    # Trade compute for memory
    gradient_accumulation_steps: int = 8   # Effective batch = actual * 8
    pin_memory: bool = True
    num_workers: int = 4                   # DataLoader workers
    max_memory_gb: float = 11.0            # Leave 1GB headroom from 12GB

    @property
    def precision(self) -> str:
        if self.bf16 and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return "bf16-mixed"
            except Exception:
                pass
        return "16-mixed" if self.mixed_precision else "32-true"


# ── Asset Universe ────────────────────────────────────────────────────────────
@dataclass
class AssetConfig:
    """Defines the tradeable universe."""
    # Stocks (US Large Cap)
    stocks: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "WMT", "JNJ", "PG", "XOM", "BAC", "DIS",
    ])

    # Indian Stocks (NSE Large Cap)
    indian_stocks: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS",   # Reliance Industries
        "TCS.NS",        # Tata Consultancy Services
        "HDFCBANK.NS",   # HDFC Bank
        "INFY.NS",       # Infosys
        "ICICIBANK.NS",  # ICICI Bank
        "HINDUNILVR.NS", # Hindustan Unilever
        "ITC.NS",        # ITC Limited
        "SBIN.NS",       # State Bank of India
        "BHARTIARTL.NS", # Bharti Airtel
        "KOTAKBANK.NS",  # Kotak Mahindra Bank
        "LT.NS",         # Larsen & Toubro
        "AXISBANK.NS",   # Axis Bank
        "WIPRO.NS",      # Wipro
        "HCLTECH.NS",    # HCL Technologies
        "ADANIENT.NS",   # Adani Enterprises
    ])

    # Forex pairs
    forex: List[str] = field(default_factory=lambda: [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
        "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDINR=X",
    ])

    # Crypto
    crypto: List[str] = field(default_factory=lambda: [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    ])

    # Commodities
    commodities: List[str] = field(default_factory=lambda: [
        "GC=F",   # Gold
        "SI=F",   # Silver
        "CL=F",   # Crude Oil
        "NG=F",   # Natural Gas
        "HG=F",   # Copper
    ])

    # Market indices (for context, not trading)
    indices: List[str] = field(default_factory=lambda: [
        "^GSPC",  # S&P 500
        "^IXIC",  # NASDAQ
        "^DJI",   # Dow Jones
        "^VIX",   # Volatility Index
        "^TNX",   # 10Y Treasury Yield
        "^NSEI",  # NIFTY 50
        "^BSESN", # BSE SENSEX
        "^NSEBANK",  # NIFTY Bank
    ])

    @property
    def all_tickers(self) -> List[str]:
        return (self.stocks + self.indian_stocks + self.forex + self.crypto
                + self.commodities + self.indices)

    def get_category(self, ticker: str) -> str:
        """Return which category a ticker belongs to."""
        if ticker in self.stocks: return "stocks"
        if ticker in self.indian_stocks: return "indian_stocks"
        if ticker in self.forex: return "forex"
        if ticker in self.crypto: return "crypto"
        if ticker in self.commodities: return "commodities"
        if ticker in self.indices: return "indices"
        return "unknown"


@dataclass
class DataConfig:
    """Data fetching and preprocessing settings."""
    # Time periods
    history_years: int = 2              # Years of historical data (max 2 for 1h)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Timeframes - INTRADAY FOCUSED
    primary_timeframe: str = "5m"       # Primary intraday timeframe
    supported_timeframes: List[str] = field(default_factory=lambda: [
        "1m", "3m", "5m", "15m"  # Intraday scalping/day trading
    ])
    
    # Higher timeframes for multi-TF analysis
    higher_timeframes: List[str] = field(default_factory=lambda: ["15m", "1h"])
    
    # Cross-asset reference tickers for correlation features
    cross_asset_references: List[str] = field(default_factory=lambda: [
        "SPY", "BTC-USD", "DX-Y.NYB", "^VIX", "^NSEI", "USDINR=X"
    ])
    
    # Sequence lengths per timeframe
    sequence_lengths: Dict[str, int] = field(default_factory=lambda: {
        "1m": 390,   # 390 minutes = 6.5 hours (1 trading day)
        "3m": 130,   # 130 x 3m = 6.5 hours
        "5m": 78,    # 78 x 5m = 6.5 hours
        "15m": 26,   # 26 x 15m = 6.5 hours
    })
    
    # Prediction horizons per timeframe
    prediction_horizons: Dict[str, int] = field(default_factory=lambda: {
        "1m": 30,    # Predict 30 minutes ahead
        "3m": 20,    # Predict 1 hour ahead
        "5m": 12,    # Predict 1 hour ahead
        "15m": 8,    # Predict 2 hours ahead
    })

    # Preprocessing (defaults for primary timeframe)
    sequence_length: int = 78           # 78 x 5m = 6.5 hours (1 trading day)
    prediction_horizon: int = 12        # Forecast 1 hour ahead (12 x 5m)
    max_prediction_horizon: int = 24    # Max forecast horizon

    # Target labeling
    label_method: str = "triple_barrier"  # "triple_barrier" or "simple"
    tp_atr_multiplier: float = 2.0       # Take-profit barrier (ATR multiples)
    sl_atr_multiplier: float = 1.0       # Stop-loss barrier (ATR multiples)

    # Cache
    cache_expiry_hours: int = 1         # Re-fetch after 1 hour (intraday needs fresh data)
    data_format: str = "parquet"        # Storage format


# ── Model Configurations ─────────────────────────────────────────────────────
@dataclass
class TFTConfig:
    """Temporal Fusion Transformer hyperparameters."""
    hidden_size: int = 128
    attention_head_size: int = 4
    num_attention_heads: int = 4
    lstm_layers: int = 2
    dropout: float = 0.1
    hidden_continuous_size: int = 64
    output_size: int = 7               # Quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    learning_rate: float = 1e-3
    reduce_on_plateau_patience: int = 5
    max_encoder_length: int = 120      # Match sequence_length (120h)
    max_prediction_length: int = 8     # Match prediction_horizon (8h)


@dataclass
class CNNConfig:
    """Pattern Recognition CNN hyperparameters."""
    in_channels: int = 1               # Price channel
    num_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    fc_size: int = 256
    num_classes: int = 5               # breakout, reversal_up, reversal_down, continuation, consolidation
    dropout: float = 0.3
    learning_rate: float = 5e-4


@dataclass
class LSTMConfig:
    """Regime Detection LSTM hyperparameters."""
    input_size: int = 64               # Feature dimension (set dynamically)
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    num_regimes: int = 4               # bull, bear, sideways, volatile
    dropout: float = 0.2
    learning_rate: float = 1e-3
    attention: bool = True


@dataclass
class EnsembleConfig:
    """Meta-learner ensemble settings."""
    method: str = "xgboost"            # xgboost, lightgbm, or neural
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    regime_adaptive: bool = True       # Adjust weights by regime


# ── Training Configuration ───────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Training pipeline settings."""
    # General
    max_epochs: int = 100
    batch_size: int = 8                # Actual batch size (GPU limited)
    effective_batch_size: int = 64     # After gradient accumulation
    early_stopping_patience: int = 15
    min_delta: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    max_lr: float = 1e-3
    min_lr: float = 1e-6

    # Scheduler
    scheduler: str = "one_cycle"       # one_cycle, cosine, plateau
    warmup_epochs: int = 5

    # Walk-forward
    use_walk_forward: bool = True      # Enable walk-forward validation
    walk_forward_folds: int = 5
    expanding_window: bool = True      # True = expanding, False = sliding

    # Hyperparameter tuning
    hpo_n_trials: int = 50
    hpo_timeout_hours: float = 4.0


# ── Feature Configuration ────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    # Technical indicator windows
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_windows: List[int] = field(default_factory=lambda: [5, 12, 26, 50])
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    adx_window: int = 14
    stoch_window: int = 14
    cci_window: int = 20

    # Sentiment
    sentiment_model: str = "ProsusAI/finbert"
    sentiment_max_length: int = 512
    sentiment_batch_size: int = 32

    # News data for real sentiment
    newsapi_key: str = ""               # Leave empty for RSS-only mode
    news_lookback_hours: int = 48       # How far back to search for news
    news_max_headlines: int = 50        # Max headlines per ticker

    # Rolling windows for features
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])


# ── Dashboard Configuration ──────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    """Backtesting settings."""
    use_kelly: bool = True              # Kelly criterion position sizing
    run_monte_carlo: bool = False       # Monte Carlo robustness testing
    monte_carlo_simulations: int = 1000 # Number of MC simulations


@dataclass
class DashboardConfig:
    """Web dashboard settings."""
    host: str = "0.0.0.0"
    port: int = 5555
    debug: bool = False
    refresh_interval_seconds: int = 5   # Live updates every 5 seconds
    price_update_interval_ms: int = 1000  # Price updates every 1 second
    chart_timeframes: List[str] = field(default_factory=lambda: [
        "1m", "3m", "5m", "15m", "1h", "1d"
    ])


# ── Automation Configuration ──────────────────────────────────────────────────
@dataclass
class AutomationConfig:
    """Automated trading settings."""
    enabled: bool = False
    trading_style: str = "intraday"      # intraday, daytrading, swing, longterm
    broker: str = ""                      # dhan, angelone, zerodha, groww, fxpro
    paper_mode: bool = True               # Start in paper mode (simulated)
    max_positions: int = 20               # Max concurrent trades
    max_daily_loss_pct: float = 5.0       # Kill-switch: stop if daily loss > X%
    per_trade_risk_pct: float = 2.0       # Max capital risk per trade
    min_confidence: float = 0.60          # Min prediction confidence to trade
    cooldown_after_losses: int = 3        # Pause after N consecutive losses
    auto_square_off_minutes: int = 15     # Minutes before close for intraday
    scan_interval_seconds: int = 60       # Opportunity scan frequency
    default_quantity: int = 1             # Default trade quantity


# ── Master Configuration ─────────────────────────────────────────────────────
@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    assets: AssetConfig = field(default_factory=AssetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)

    def summary(self) -> str:
        """Print configuration summary."""
        lines = [
            "=" * 60,
            "  TRADING AI — Configuration Summary",
            "=" * 60,
            f"  Device:          {self.gpu.device}",
            f"  Precision:       {self.gpu.precision}",
            f"  Grad Checkpoint: {self.gpu.gradient_checkpointing}",
            f"  Batch Size:      {self.training.batch_size} (effective: {self.training.effective_batch_size})",
            f"  Max Epochs:      {self.training.max_epochs}",
            f"  Sequence Length:  {self.data.sequence_length}",
            f"  Prediction:      {self.data.prediction_horizon} steps ahead",
            f"  Assets:          {len(self.assets.all_tickers)} total",
            f"    US Stocks:     {len(self.assets.stocks)}",
            f"    Indian Stocks: {len(self.assets.indian_stocks)}",
            f"    Forex:         {len(self.assets.forex)}",
            f"    Crypto:        {len(self.assets.crypto)}",
            f"    Commodities:   {len(self.assets.commodities)}",
            f"    Indices:       {len(self.assets.indices)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ── Global Instance ───────────────────────────────────────────────────────────
cfg = Config()
