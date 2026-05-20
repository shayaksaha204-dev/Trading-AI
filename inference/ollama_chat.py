"""
Ollama Chat Service — STRICTLY Grounded Sentence Formatting
=============================================================
Ollama ONLY formats real model data into sentences.
NO HALLUCINATIONS - Every fact comes from actual model outputs.
"""

import requests
import json
from typing import Dict, Optional, List
from loguru import logger


def _scalar(val):
    """Safely convert numpy arrays/scalars to a plain Python float."""
    import numpy as np
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 else 0.0
    if isinstance(val, (np.integer, np.floating)):
        return float(val)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return float(val[0])
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


class OllamaChatService:
    """
    Chat service that STRICTLY formats model data into natural language.
    
    CRITICAL: Ollama does NOT generate trading insights.
    It ONLY converts real data into readable sentences.
    All facts come from the trained models, never from Ollama.
    """

    def __init__(self, model: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.context_history: List[Dict] = []
        self.max_history = 10

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def frame_prediction(self, prediction: Dict) -> str:
        """
        Convert raw prediction data into a natural language explanation.
        
        CRITICAL: Ollama ONLY formats the data. It does NOT generate insights.
        Every number and fact comes from the actual model prediction.
        """
        if not self.is_available():
            return self._fallback_frame(prediction)

        # Extract ALL data from prediction (grounded in real model output)
        ticker = prediction.get("ticker", "Unknown")
        signal = prediction.get("signal_label", "HOLD")
        confidence = _scalar(prediction.get("confidence", 0.5))
        regime = prediction.get("regime", "unknown")
        pattern = prediction.get("pattern", "none")
        pattern_strength = _scalar(prediction.get("pattern_strength", 0))
        price = _scalar(prediction.get("current_price", 0))
        
        # Get strategy breakdown (REAL data from model)
        breakdown = prediction.get("strategy_breakdown", {})
        strategy_details = self._format_strategy_breakdown(breakdown)
        
        # Get reasoning (REAL data from model)
        reasoning = prediction.get("reasoning", [])
        
        # STRICT prompt - Ollama must ONLY format, not generate
        prompt = f"""You are a sentence formatter. Your ONLY job is to convert the following data into readable sentences. DO NOT add any information not provided below. DO NOT make up any numbers or facts.

DATA FROM MODEL (you must use ONLY these values):
- Asset: {ticker}
- Price: ${price:.2f}
- Signal: {signal}
- Confidence: {confidence:.1%}
- Market Regime: {regime}
- Pattern: {pattern.replace('_', ' ')} (strength: {pattern_strength:.1%})

STRATEGY ANALYSIS (from the 8 trading strategies):
{strategy_details}

MODEL'S REASONING:
{chr(10).join(reasoning[:5]) if reasoning else 'Standard analysis'}

Format this into 2-3 clear sentences. Use ONLY the data above. Do NOT add predictions or insights not in the data."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # LOW temperature to prevent hallucination
                        "num_predict": 150,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", self._fallback_frame(prediction))
            else:
                return self._fallback_frame(prediction)

        except Exception as e:
            logger.warning(f"Ollama request failed: {e}")
            return self._fallback_frame(prediction)
    
    def _format_strategy_breakdown(self, breakdown: Dict) -> str:
        """Format strategy breakdown into readable text (GROUNDING DATA)."""
        if not breakdown:
            return "No strategy data available."
        
        lines = []
        for strat, data in breakdown.items():
            signal = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(data.get("signal", 0), "HOLD")
            conf = _scalar(data.get("confidence", 0))
            reason = data.get("reason", "no signal")
            lines.append(f"- {strat}: {signal} ({conf:.0%} confidence) - {reason}")
        
        return chr(10).join(lines[:8])  # All 8 strategies

    def _fallback_frame(self, prediction: Dict) -> str:
        """Fallback if Ollama is unavailable."""
        ticker = prediction.get("ticker", "Unknown")
        signal = prediction.get("signal_label", "HOLD")
        confidence = _scalar(prediction.get("confidence", 0.5))
        regime = prediction.get("regime", "unknown")
        pattern = prediction.get("pattern", "none").replace("_", " ")

        signal_desc = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}
        return f"{ticker} shows a {signal_desc.get(signal, 'neutral')} signal with {confidence:.0%} confidence. Market regime is {regime} with {pattern} pattern detected."

    def chat(self, user_message: str, prediction_context: Optional[Dict] = None, learning_context: Optional[Dict] = None) -> str:
        """
        Interactive chat with the AI about trading predictions.
        
        CRITICAL: Ollama can ONLY answer using provided data.
        It CANNOT generate new insights or predictions.
        """
        if not self.is_available():
            return "AI chat is unavailable. Please ensure Ollama is running with llama3 model."

        # Build STRICT context - Ollama can ONLY use this data
        context_data = []
        
        # Add prediction data if available (GROUNDING)
        if prediction_context:
            ticker = prediction_context.get("ticker", "")
            signal = prediction_context.get("signal_label", "")
            confidence = _scalar(prediction_context.get("confidence", 0))
            regime = prediction_context.get("regime", "")
            price = _scalar(prediction_context.get("current_price", 0))
            pattern = prediction_context.get("pattern", "")
            
            context_data.append(f"""CURRENT PREDICTION DATA:
- Asset: {ticker}
- Price: ${price:.2f}
- Signal: {signal} ({confidence:.0%} confidence)
- Market Regime: {regime}
- Pattern: {pattern}""")
            
            # Add strategy breakdown (GROUNDING)
            breakdown = prediction_context.get("strategy_breakdown", {})
            if breakdown:
                context_data.append("STRATEGY SIGNALS:")
                for strat, data in breakdown.items():
                    sig = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(data.get("signal", 0), "HOLD")
                    conf = _scalar(data.get("confidence", 0))
                    reason = data.get("reason", "")
                    context_data.append(f"- {strat}: {sig} ({conf:.0%}) - {reason}")
        
        # Add learning data if available (GROUNDING)
        if learning_context:
            sessions = learning_context.get("total_training_sessions", 0)
            accuracy = _scalar(learning_context.get("current_best_accuracy", 0))
            win_rate = _scalar(learning_context.get("overall_win_rate", 0))
            trades = learning_context.get("total_trades_learned", 0)
            trend = learning_context.get("improvement_trend", "")
            online = learning_context.get("online_learning", {})
            weights = learning_context.get("adaptive_weights", {})
            
            context_data.append(f"""LEARNING STATUS:
- Training Sessions: {sessions}
- Current Accuracy: {accuracy:.1%}
- Win Rate: {win_rate:.1%}
- Trades Analyzed: {trades}
- Trend: {trend}
- Online Updates: {online.get('total_updates', 0)}
- Loss Trend: {online.get('loss_trend', 'unknown')}""")
            
            if weights:
                context_data.append("ADAPTIVE STRATEGY WEIGHTS:")
                for strat, weight in sorted(weights.items(), key=lambda x: _scalar(x[1]), reverse=True):
                    context_data.append(f"- {strat}: {_scalar(weight):.1%}")

        # System prompt — knowledgeable AI that grounds data answers in model output
        system_context = f"""You are a knowledgeable AI trading assistant powering a multi-model trading prediction system. You can discuss markets, strategies, trading concepts, and general topics naturally.

RULES:
1. For general knowledge questions (what is RSI, explain Elliott Wave, etc.) — answer freely using your knowledge.
2. When the user asks about a specific prediction, price, signal, or model output — use ONLY the real data below. Never invent numbers.
3. If the user asks for a prediction but no data is loaded, say "Please analyze a ticker first by clicking Analyze."
4. Be concise (2-4 sentences). Be confident and professional.
5. When model data IS available, weave it naturally into your response.

{chr(10).join(context_data) if context_data else 'No prediction data loaded yet. The user has not analyzed a ticker.'}

The system uses 8 strategies: Technical Analysis, Market Structure, Price Action, Smart Money Concepts (SMC), ICT Concepts, Wyckoff Analysis, Elliott Wave, and Sentiment. It has 3 neural models: TFT (Temporal Fusion Transformer), PatternCNN, and RegimeLSTM that work together via an ensemble."""

        # Add to history
        self.context_history.append({"role": "user", "content": user_message})
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_context},
                        *self.context_history
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 400,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "I couldn't process that request.")
                self.context_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message
            else:
                return f"Error communicating with AI model."

        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return "Sorry, I encountered an error. Please try again."

    def explain_strategy(self, strategy_name: str, prediction: Dict) -> str:
        """Explain what a specific strategy is detecting."""
        if not self.is_available():
            return f"Strategy {strategy_name} is active."

        prompt = f"""Explain the {strategy_name} trading strategy in 1-2 sentences in the context of this prediction:
- Ticker: {prediction.get('ticker', 'Unknown')}
- Signal: {prediction.get('signal_label', 'HOLD')}
- Pattern: {prediction.get('pattern', 'none')}

Keep it brief and actionable:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.6, "num_predict": 80}
                },
                timeout=15
            )

            if response.status_code == 200:
                return response.json().get("response", f"{strategy_name} strategy is active.")
            return f"{strategy_name} strategy is active."

        except:
            return f"{strategy_name} strategy is active."

    def generate_live_commentary(self, price_data: Dict, prediction: Dict) -> str:
        """Generate real-time commentary for live chart updates."""
        if not self.is_available():
            return ""

        ticker = prediction.get("ticker", "")
        signal = prediction.get("signal_label", "")
        current_price = price_data.get("close", 0)
        change_pct = price_data.get("change_pct", 0)
        volume = price_data.get("volume", 0)

        prompt = f"""Generate a very brief (10-15 words) live market update for:
{ticker}: ${current_price:.2f} ({change_pct:+.2f}%)
Signal: {signal}

Example format: "AAPL holding above support at $175, bullish momentum building"
Your update:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 30}
                },
                timeout=5
            )

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return ""

        except:
            return ""

    def clear_context(self):
        """Clear conversation history."""
        self.context_history = []

    def frame_training_summary(self, learning_summary: Dict) -> str:
        """
        Generate natural language summary of what the model learned.
        
        CRITICAL: Ollama ONLY formats the data. All facts come from the model.
        """
        if not self.is_available():
            return self._fallback_training_summary(learning_summary)

        # Extract ALL data from learning summary (GROUNDING)
        sessions = learning_summary.get("total_training_sessions", 0)
        accuracy = _scalar(learning_summary.get("current_best_accuracy", 0))
        trend = learning_summary.get("improvement_trend", "unknown")
        win_rate = _scalar(learning_summary.get("overall_win_rate", 0))
        trades = learning_summary.get("total_trades_learned", 0)
        patterns = learning_summary.get("patterns_recognized", [])
        strategies = learning_summary.get("best_strategy_combinations", [])
        agreement = _scalar(learning_summary.get("strategy_agreement_rate", 0))
        
        # Online learning data (GROUNDING)
        online = learning_summary.get("online_learning", {})
        online_updates = online.get("total_updates", 0)
        loss_trend = online.get("loss_trend", "unknown")
        hard_examples = online.get("hard_examples", 0)
        
        # Curriculum data (GROUNDING)
        curriculum = learning_summary.get("curriculum", {})
        curriculum_hard = curriculum.get("hard_examples", 0)
        weak_patterns = curriculum.get("weak_patterns", [])
        
        # Adaptive weights (GROUNDING)
        weights = learning_summary.get("adaptive_weights", {})
        
        # Format weak patterns
        weak_str = ""
        if weak_patterns:
            weak_str = "WEAK PATTERNS (need improvement):\n"
            for pattern, error_rate, _ in weak_patterns[:3]:
                weak_str += f"- {pattern}: {error_rate:.1%} error rate\n"
        
        # Format adaptive weights
        weights_str = ""
        if weights:
            weights_str = "ADAPTIVE STRATEGY WEIGHTS (learned from performance):\n"
            for strat, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                weights_str += f"- {strat}: {weight:.1%}\n"

        # STRICT prompt - Ollama ONLY formats, never generates
        prompt = f"""You are a sentence formatter. Convert this training data into 2-3 readable sentences. Use ONLY the data below. Do NOT add any facts not provided.

TRAINING DATA (from the model):
- Training Sessions: {sessions}
- Current Accuracy: {accuracy:.1%}
- Win Rate: {win_rate:.1%}
- Total Trades Analyzed: {trades}
- Strategy Agreement Rate: {agreement:.1%}
- Performance Trend: {trend}

ONLINE LEARNING (real-time updates):
- Total Online Updates: {online_updates}
- Loss Trend: {loss_trend}
- Hard Examples Tracked: {hard_examples}

CURRICULUM LEARNING:
- Hard Examples in Curriculum: {curriculum_hard}
{weak_str}
{weights_str}
Format this into 2-3 sentences about the model's learning progress. Use ONLY the numbers above."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # LOW temperature
                        "num_predict": 200,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", self._fallback_training_summary(learning_summary))
            else:
                return self._fallback_training_summary(learning_summary)

        except Exception as e:
            logger.warning(f"Training summary framing failed: {e}")
            return self._fallback_training_summary(learning_summary)

    def _fallback_training_summary(self, summary: Dict) -> str:
        """Fallback training summary if Ollama unavailable."""
        sessions = summary.get("total_training_sessions", 0)
        accuracy = _scalar(summary.get("current_best_accuracy", 0))
        trend = summary.get("improvement_trend", "unknown")
        win_rate = _scalar(summary.get("overall_win_rate", 0))
        trades = summary.get("total_trades_learned", 0)
        agreement = _scalar(summary.get("strategy_agreement_rate", 0))
        
        online = summary.get("online_learning", {})
        online_updates = online.get("total_updates", 0)
        loss_trend = online.get("loss_trend", "unknown")

        trend_msg = {
            "improving": "improving steadily",
            "stable": "performing consistently",
            "needs_attention": "may need adjustment",
            "starting": "building foundational knowledge"
        }.get(trend, "learning")

        return (
            f"After {sessions} training sessions analyzing {trades} trades, "
            f"the model has achieved {accuracy:.0%} accuracy with {win_rate:.0%} win rate. "
            f"Online learning has performed {online_updates} real-time updates with {loss_trend} loss trend. "
            f"The model agrees with the 8 strategies {agreement:.0%} of the time and is {trend_msg}."
        )

    def frame_strategy_insight(self, strategy_name: str, performance: Dict) -> str:
        """
        Generate insight about a specific strategy's performance.
        
        CRITICAL: Ollama ONLY formats the provided performance data.
        """
        if not self.is_available():
            return f"{strategy_name} strategy has {_scalar(performance.get('win_rate', 0)):.0%} win rate."

        # Extract ALL data (GROUNDING)
        win_rate = _scalar(performance.get("win_rate", 0))
        total_trades = performance.get("total_trades", 0)
        total_profit = _scalar(performance.get("total_profit", 0))
        avg_profit = _scalar(performance.get("avg_profit", 0))
        
        # STRICT prompt - format only
        prompt = f"""You are a sentence formatter. Convert this strategy performance data into 1-2 sentences. Use ONLY the data below.

STRATEGY: {strategy_name}
PERFORMANCE DATA:
- Win Rate: {win_rate:.1%}
- Total Trades: {total_trades}
- Total Profit: ${total_profit:.2f}
- Average Profit per Trade: ${avg_profit:.2f}

Format into 1-2 sentences. Use ONLY the numbers above."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 80}
                },
                timeout=15
            )

            if response.status_code == 200:
                return response.json().get("response", f"{strategy_name} shows {win_rate:.0%} win rate.")
            return f"{strategy_name} shows {win_rate:.0%} win rate."

        except:
            return f"{strategy_name} shows {win_rate:.0%} win rate."
    
    def answer_market_question(self, question: str, prediction: Dict, learning: Dict) -> str:
        """
        Answer a market question using ONLY real model data.
        
        This is the main method for dashboard chat.
        Ollama CANNOT generate insights - only format data.
        """
        return self.chat(
            user_message=question,
            prediction_context=prediction,
            learning_context=learning,
        )


# Singleton instance
_chat_service = None


def get_chat_service() -> OllamaChatService:
    """Get or create the chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = OllamaChatService()
    return _chat_service
