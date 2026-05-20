"""
Trading AI — Main Entry Point
================================
CLI interface for training, predicting, backtesting, and dashboard.

Usage:
    python main.py train                        # Train on all assets
    python main.py train --category stocks      # Train on stocks only
    python main.py train --tickers AAPL MSFT    # Train on specific tickers
    python main.py predict AAPL                 # Predict for AAPL
    python main.py predict --all                # Predict for all assets
    python main.py backtest                     # Run backtest
    python main.py dashboard                    # Launch web dashboard
    python main.py status                       # Check system status
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
import numpy as np

# Suppress noisy pandas warnings (harmless during feature computation)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

# Ensure project root is in path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
logger.add(str(ROOT / "logs" / "trading_ai.log"), rotation="10 MB", level="DEBUG")


def cmd_status():
    """Check system status and GPU availability."""
    from config import cfg
    print(cfg.summary())
    
    try:
        import torch
        print(f"\n  PyTorch:         {torch.__version__}")
        print(f"  CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version:    {torch.version.cuda}")
            print(f"  GPU:             {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory
            print(f"  VRAM:            {mem / 1024**3:.1f} GB")
        else:
            print("\n  ⚠️  CUDA not available. Install PyTorch with CUDA:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132")
    except ImportError:
        print("\n  ❌ PyTorch not installed!")


def cmd_learning(args):
    """Check continuous learning progress."""
    from training.continuous_learner import get_continuous_learner
    from training.experience import get_experience_buffer
    from training.curriculum import get_curriculum_manager
    from training.adaptive_strategies import get_strategy_tracker
    from inference.ollama_chat import get_chat_service
    
    print("\n" + "=" * 60)
    print("  CONTINUOUS LEARNING STATUS")
    print("=" * 60)
    
    # Get learners
    tft_learner = get_continuous_learner("tft")
    cnn_learner = get_continuous_learner("cnn")
    lstm_learner = get_continuous_learner("lstm")
    experience_buffer = get_experience_buffer()
    curriculum = get_curriculum_manager()
    strategy_tracker = get_strategy_tracker()
    chat_service = get_chat_service()
    
    # Get summaries
    tft_summary = tft_learner.get_learning_summary_for_ollama()
    cnn_summary = cnn_learner.get_learning_summary_for_ollama()
    lstm_summary = lstm_learner.get_learning_summary_for_ollama()
    buffer_summary = experience_buffer.get_learning_summary()
    curriculum_summary = curriculum.get_curriculum_summary()
    strategy_summary = strategy_tracker.get_summary()
    
    # Print stats
    print(f"\n  📊 Training Sessions: {tft_summary.get('total_training_sessions', 0)}")
    print(f"  📈 Total Trades Analyzed: {buffer_summary.get('total_experiences', 0)}")
    print(f"  🎯 Overall Win Rate: {buffer_summary.get('win_rate', 0):.1%}")
    print(f"  📉 Improvement Trend: {tft_summary.get('improvement_trend', 'unknown')}")
    
    # Online learning stats
    online_stats = tft_summary.get("online_learning", {})
    print(f"\n  ⚡ Online Learning:")
    print(f"     Total Updates: {online_stats.get('total_updates', 0)}")
    print(f"     Loss Trend: {online_stats.get('loss_trend', 'unknown')}")
    print(f"     Hard Examples Tracked: {online_stats.get('hard_examples', 0)}")
    
    # Curriculum stats
    curriculum_stats = tft_summary.get("curriculum", {})
    print(f"\n  📚 Curriculum Learning:")
    print(f"     Hard Examples: {curriculum_stats.get('hard_examples', 0)}")
    weak_patterns = curriculum_stats.get('weak_patterns', [])
    if weak_patterns:
        print(f"     Weak Patterns:")
        for pattern, error_rate, _ in weak_patterns[:3]:
            print(f"       - {pattern}: {error_rate:.1%} error rate")
    
    print(f"\n  🧠 Model Performance:")
    print(f"     TFT:  {tft_summary.get('current_best_accuracy', 0):.1%} accuracy")
    print(f"     CNN:  {cnn_summary.get('current_best_accuracy', 0):.1%} accuracy")
    print(f"     LSTM: {lstm_summary.get('current_best_accuracy', 0):.1%} accuracy")
    
    # Adaptive strategy weights
    print(f"\n  ⚖️  Adaptive Strategy Weights:")
    weights = tft_summary.get('adaptive_weights', {})
    for strat, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {strat}: {weight:.1%}")
    
    # Best patterns
    patterns = buffer_summary.get("best_patterns", [])
    if patterns:
        print(f"\n  🔍 Best Strategy Patterns:")
        for pattern, avg_profit, count in patterns[:3]:
            print(f"     • {pattern}: {avg_profit:.2%} avg profit ({count} trades)")
    
    # Best strategies
    strategies = buffer_summary.get("best_strategies", [])
    if strategies:
        print(f"\n  🎯 Top Strategies:")
        for name, profit, win_rate in strategies[:3]:
            print(f"     • {name}: ${profit:.2f} profit, {win_rate:.1%} win rate")
    
    # Generate AI summary
    if chat_service.is_available():
        print(f"\n  🤖 AI Summary:")
        combined = {
            **tft_summary,
            "model_breakdown": {
                "TFT": tft_summary.get("current_best_accuracy", 0),
                "CNN": cnn_summary.get("current_best_accuracy", 0),
                "LSTM": lstm_summary.get("current_best_accuracy", 0),
            }
        }
        ai_summary = chat_service.frame_training_summary(combined)
        print(f"  {ai_summary}")
    else:
        print(f"\n  ⚠️  Ollama not available. Run 'ollama run llama3' to enable AI summaries.")
    
    print("\n" + "=" * 60)


def cmd_train(args):
    """Train models with intraday timeframes."""
    from pipeline import TradingPipeline
    from config import cfg
    
    # Get timeframe from args or config
    timeframe = getattr(args, "timeframe", None) or cfg.data.primary_timeframe
    
    print(f"\n  📊 Training with timeframe: {timeframe}")
    print(f"  🎯 Intraday trading focused (scalping/day trading)\n")
    
    pipeline = TradingPipeline()
    
    tickers = args.tickers if hasattr(args, "tickers") and args.tickers else None
    categories = [args.category] if hasattr(args, "category") and args.category else None
    
    pipeline.run_full_pipeline(
        categories=categories,
        tickers=tickers,
        skip_fetch=getattr(args, "skip_fetch", False),
        timeframe=timeframe,
    )


def cmd_predict(args):
    """Generate predictions."""
    from inference.predictor import TradingPredictor
    
    predictor = TradingPredictor()
    predictor.load_models()
    
    if hasattr(args, "all") and args.all:
        results = predictor.predict_all()
    elif hasattr(args, "category") and args.category:
        results = predictor.predict_category(args.category)
    elif hasattr(args, "tickers") and args.tickers:
        results = []
        for t in args.tickers:
            pred = predictor.predict_ticker(t)
            if pred:
                results.append(pred)
    else:
        print("Specify --all, --category, or ticker symbols")
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("  TRADING AI — Predictions")
    print("=" * 70)
    for r in results:
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r["signal_label"], "⚪")
        print(f"\n  {signal_emoji} {r['ticker']:>10} | {r['signal_label']:>4} | "
              f"Confidence: {r['confidence'][0]:.1%} | "
              f"Regime: {r['regime']} | "
              f"Price: ${r['current_price']:.2f}")
        if "pattern" in r:
            print(f"{'':>16} Pattern: {r['pattern']} (strength: {r.get('pattern_strength', [0])[0]:.1%})")
    print("\n" + "=" * 70)


def cmd_learn(args):
    """Run continuous learning loop - learn from recent trade outcomes."""
    from inference.predictor import TradingPredictor
    from data.fetcher import DataFetcher
    import time
    
    print("\n" + "=" * 60)
    print("  CONTINUOUS LEARNING LOOP")
    print("=" * 60)
    print("\n  This will:")
    print("  1. Load trained models")
    print("  2. Fetch recent data for tracked tickers")
    print("  3. Simulate trades and learn from outcomes")
    print("  4. Update strategy weights based on performance")
    print("\n  Press Ctrl+C to stop.\n")
    
    predictor = TradingPredictor()
    predictor.load_models()
    fetcher = DataFetcher()
    
    tickers = args.tickers if hasattr(args, "tickers") and args.tickers else ["AAPL", "BTC-USD", "EURUSD=X"]
    interval_minutes = getattr(args, "interval", 60) or 60
    
    print(f"  Tracking: {', '.join(tickers)}")
    print(f"  Update interval: {interval_minutes} minutes\n")
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n  --- Iteration {iteration} ---")
        
        for ticker in tickers:
            try:
                # Get prediction
                pred = predictor.predict_ticker(ticker)
                if not pred:
                    continue
                
                signal = pred.get("signal", [0])[0] if isinstance(pred.get("signal"), np.ndarray) else pred.get("signal", 0)
                confidence = pred.get("confidence", [0.5])[0] if isinstance(pred.get("confidence"), np.ndarray) else pred.get("confidence", 0.5)
                
                signal_label = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(signal, "HOLD")
                print(f"  {ticker}: {signal_label} (conf: {confidence:.1%})")
                
                # Simulate outcome (in real use, you'd wait for actual outcome)
                # Here we simulate by using next bar's return
                df = fetcher.fetch_ticker(ticker, force_refresh=True)
                if df is not None and len(df) > 1:
                    actual_return = df["Close"].pct_change().iloc[-1]
                    profit = signal * actual_return
                    
                    # Learn from outcome
                    result = predictor.record_outcome(
                        ticker=ticker,
                        prediction=pred,
                        actual_return=actual_return,
                        profit=profit,
                    )
                    
                    if result.get("learned"):
                        print(f"    ✓ Learned from outcome: {'profit' if profit > 0 else 'loss'} of {abs(profit):.2%}")
                    
            except Exception as e:
                print(f"  {ticker}: Error - {e}")
        
        
        # Show learning status every 10 iterations
        if iteration % 10 == 0:
            status = predictor.get_learning_status()
            tft_status = status.get("tft", {})
            online = tft_status.get("online_learning", {})
            print(f"\n  📊 Learning Progress:")
            print(f"     Online updates: {online.get('total_updates', 0)}")
            print(f"     Loss trend: {online.get('loss_trend', 'unknown')}")
        
        
        # Wait for next iteration
        print(f"\n  Waiting {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def cmd_dashboard(args):
    """Launch web dashboard with real-time charts and AI chat."""
    from dashboard.app import create_app, run_server
    app = create_app()
    port = getattr(args, "port", 5555) or 5555
    print(f"\n  🌐 Dashboard: http://localhost:{port}")
    print(f"  📊 Features: Live charts, AI chat (Ollama), Real-time price updates")
    print(f"  💡 Make sure Ollama is running with llama3: ollama run llama3")
    run_server(app, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Trading AI — Multi-Model Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status
    subparsers.add_parser("status", help="Check system status")

    # Learning progress
    subparsers.add_parser("learning", help="Check continuous learning progress")

    # Continuous learning loop
    learn_parser = subparsers.add_parser("learn", help="Run continuous learning loop")
    learn_parser.add_argument("--tickers", nargs="+", type=str, default=["AAPL", "BTC-USD"], help="Tickers to track")
    learn_parser.add_argument("--interval", type=int, default=60, help="Update interval in minutes")

    # Train
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--category", type=str, choices=["stocks", "indian_stocks", "forex", "crypto", "commodities"])
    train_parser.add_argument("--tickers", nargs="+", type=str, help="Specific tickers")
    train_parser.add_argument("--timeframe", type=str, choices=["1m", "3m", "5m", "15m", "1h", "1d"], 
                              default="5m", help="Intraday timeframe (default: 5m)")
    train_parser.add_argument("--skip-fetch", action="store_true", help="Use cached data")

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Generate predictions")
    pred_parser.add_argument("tickers", nargs="*", help="Ticker symbols")
    pred_parser.add_argument("--all", action="store_true", help="Predict all assets")
    pred_parser.add_argument("--category", type=str)

    # Dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dash_parser.add_argument("--port", type=int, default=5555)

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "learning":
        cmd_learning(args)
    elif args.command == "learn":
        cmd_learn(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()
        print("\n  Quick start:")
        print("    python main.py status              # Check GPU & system")
        print("    python main.py learning            # Check learning progress")
        print("    python main.py learn               # Run continuous learning loop")
        print("    python main.py train               # Train on all assets (5m intraday)")
        print("    python main.py train --timeframe 1m  # Train for 1m scalping")
        print("    python main.py train --timeframe 15m # Train for 15m day trading")
        print("    python main.py train --category crypto")
        print("    python main.py predict AAPL BTC-USD")
        print("    python main.py dashboard           # Live charts with intraday timeframes")


if __name__ == "__main__":
    main()
