"""
Strategy Confluence Engine — Multi-Strategy Signal Aggregation
================================================================
Each trading methodology generates its own signal. 
Trades only fire when strategies AGREE for maximum profit.

Includes ALL 8 strategies:
1. Technical Indicators (RSI, MACD, BB, etc.)
2. Market Structure (Support/Resistance, Regimes)
3. Price Action (BOS/CHoCH/structure)
4. Smart Money Concepts (OB/FVG/liquidity)
5. ICT (kill zones/OTE/Judas/PO3)
6. Wyckoff (phases/spring/upthrust)
7. Elliott Wave (wave position/momentum)
8. Sentiment (FinBERT sentiment analysis)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger


class StrategySignal:
    """Individual strategy signal with reasoning."""
    def __init__(self, name: str, signal: int, confidence: float, reason: str, details: Dict = None):
        self.name = name
        self.signal = signal  # -1, 0, 1
        self.confidence = confidence  # 0-1
        self.reason = reason
        self.details = details or {}

    def __repr__(self):
        label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[self.signal]
        return f"{self.name}: {label} ({self.confidence:.0%}) — {self.reason}"


class StrategyConfluence:
    """
    Runs ALL trading strategies on feature data and combines signals.
    
    Only generates a trade when:
    - At least 4/8 strategies agree on direction (50%+)
    - Average confidence across agreeing strategies > threshold
    - No strategy has a strong OPPOSING signal
    
    Strategies included:
    1. Technical Indicators (RSI, MACD, BB, ADX, etc.)
    2. Market Structure (Support/Resistance, Regimes, Pivot Points)
    3. Price Action (BOS/CHoCH/HH/HL/Impulse)
    4. Smart Money Concepts (OB/FVG/Breaker/Liquidity)
    5. ICT (Kill Zones/OTE/Judas Swing/Power of 3)
    6. Wyckoff (Phases/Spring/Upthrust/Climax)
    7. Elliott Wave (Wave 3/5/Divergence/Extensions)
    8. Sentiment (FinBERT positive/negative/neutral)
    """

    def __init__(self, min_agreement: int = 4, min_confidence: float = 0.50):
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence

    def analyze(self, features: Dict[str, float]) -> Dict:
        """
        Run all 8 strategies and return confluence signal.
        
        Args:
            features: Dict of feature_name -> value (latest bar's features)
            
        Returns:
            Dict with signal, confidence, strategy breakdown, reasoning
        """
        strategies = [
            self._technical_signal(features),
            self._market_structure_signal(features),
            self._price_action_signal(features),
            self._smc_signal(features),
            self._ict_signal(features),
            self._wyckoff_signal(features),
            self._elliott_wave_signal(features),
            self._sentiment_signal(features),
        ]

        # Count votes
        bull_votes = [s for s in strategies if s.signal == 1]
        bear_votes = [s for s in strategies if s.signal == -1]
        hold_votes = [s for s in strategies if s.signal == 0]

        bull_count = len(bull_votes)
        bear_count = len(bear_votes)
        total = len(strategies)

        # Determine signal
        if bull_count >= self.min_agreement:
            signal = 1
            agreeing = bull_votes
            opposing = bear_votes
        elif bear_count >= self.min_agreement:
            signal = -1
            agreeing = bear_votes
            opposing = bull_votes
        else:
            signal = 0
            agreeing = hold_votes
            opposing = []

        # Compute confluence confidence
        if agreeing:
            avg_conf = np.mean([s.confidence for s in agreeing])
            # Penalize if there are strong opposing signals
            if opposing:
                opposition_strength = max(s.confidence for s in opposing)
                avg_conf *= (1 - opposition_strength * 0.3)
        else:
            avg_conf = 0.0

        # Check minimum confidence
        if avg_conf < self.min_confidence:
            signal = 0
            avg_conf = avg_conf * 0.5  # Reduce confidence further

        # Build reasoning
        reasons = []
        for s in strategies:
            emoji = "🟢" if s.signal == 1 else "🔴" if s.signal == -1 else "⚪"
            reasons.append(f"{emoji} {s}")

        signal_label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[signal]

        return {
            "signal": signal,
            "signal_label": signal_label,
            "confidence": float(avg_conf),
            "bull_count": bull_count,
            "bear_count": bear_count,
            "hold_count": len(hold_votes),
            "agreement": f"{max(bull_count, bear_count)}/{total}",
            "strategies": strategies,
            "reasoning": reasons,
            "breakdown": {s.name: {"signal": s.signal, "confidence": s.confidence, "reason": s.reason, "details": s.details} for s in strategies},
        }

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """Analyze the last row of a featured DataFrame."""
        if len(df) == 0:
            return {"signal": 0, "signal_label": "HOLD", "confidence": 0.0}
        last = df.iloc[-1].to_dict()
        return self.analyze(last)

    # ── Individual Strategy Signals ──────────────────────────────────────────

    def _price_action_signal(self, f: Dict) -> StrategySignal:
        """Price Action: BOS/CHoCH + structure + impulse."""
        score = 0
        reasons = []

        # Market structure
        struct = f.get("PA_Structure_Score", 0)
        if struct > 2:
            score += 2; reasons.append("bullish structure")
        elif struct < -2:
            score -= 2; reasons.append("bearish structure")

        # BOS signals
        if f.get("PA_BOS_Bull", 0) > 0:
            score += 2; reasons.append("bullish BOS")
        if f.get("PA_BOS_Bear", 0) > 0:
            score -= 2; reasons.append("bearish BOS")

        # CHoCH (reversal — strong signal)
        if f.get("PA_CHoCH_Bull", 0) > 0:
            score += 3; reasons.append("bullish CHoCH")
        if f.get("PA_CHoCH_Bear", 0) > 0:
            score -= 3; reasons.append("bearish CHoCH")

        # Displacement
        if f.get("PA_Displacement_Bull", 0) > 0:
            score += 1; reasons.append("bullish displacement")
        if f.get("PA_Displacement_Bear", 0) > 0:
            score -= 1; reasons.append("bearish displacement")

        # Impulse
        impulse = f.get("PA_Impulse_Direction", 0)
        if impulse > 0:
            score += 1
        elif impulse < 0:
            score -= 1

        # HH/HL vs LH/LL
        if f.get("PA_HH", 0) or f.get("PA_HL", 0):
            score += 1; reasons.append("HH/HL forming")
        if f.get("PA_LH", 0) or f.get("PA_LL", 0):
            score -= 1; reasons.append("LH/LL forming")

        signal = 1 if score >= 3 else (-1 if score <= -3 else 0)
        confidence = min(abs(score) / 8.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "no clear structure"

        return StrategySignal("Price Action", signal, confidence, reason)

    def _smc_signal(self, f: Dict) -> StrategySignal:
        """SMC: Order blocks, FVG, liquidity, premium/discount."""
        score = 0
        reasons = []

        # Order blocks
        if f.get("SMC_Bull_OB", 0) > 0:
            score += 3; reasons.append("in bullish OB")
        if f.get("SMC_Bear_OB", 0) > 0:
            score -= 3; reasons.append("in bearish OB")

        # Fair value gaps
        in_fvg = f.get("SMC_In_FVG", 0)
        if in_fvg > 0:
            score += 2; reasons.append("filling bullish FVG")
        elif in_fvg < 0:
            score -= 2; reasons.append("filling bearish FVG")

        # Recent FVG activity
        bull_fvg = f.get("SMC_FVG_Count_Bull", 0)
        bear_fvg = f.get("SMC_FVG_Count_Bear", 0)
        if bull_fvg > bear_fvg + 2:
            score += 1; reasons.append(f"{bull_fvg:.0f} bullish FVGs")
        elif bear_fvg > bull_fvg + 2:
            score -= 1; reasons.append(f"{bear_fvg:.0f} bearish FVGs")

        # Liquidity sweep
        sweep = f.get("SMC_Liq_Sweep", 0)
        if sweep > 0:
            score += 2; reasons.append("sell-side liquidity swept")
        elif sweep < 0:
            score -= 2; reasons.append("buy-side liquidity swept")

        # Premium / Discount
        if f.get("SMC_In_Discount_20", 0) > 0:
            score += 1; reasons.append("in discount zone")
        if f.get("SMC_In_Premium_20", 0) > 0:
            score -= 1; reasons.append("in premium zone")

        # Breaker blocks
        if f.get("SMC_Breaker_Bull", 0) > 0:
            score += 1; reasons.append("bullish breaker")
        if f.get("SMC_Breaker_Bear", 0) > 0:
            score -= 1; reasons.append("bearish breaker")

        # Institutional candles
        inst_dir = f.get("SMC_Inst_Direction", 0)
        if inst_dir > 0:
            score += 1; reasons.append("institutional buying")
        elif inst_dir < 0:
            score -= 1; reasons.append("institutional selling")

        signal = 1 if score >= 3 else (-1 if score <= -3 else 0)
        confidence = min(abs(score) / 10.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "no SMC setup"

        return StrategySignal("SMC", signal, confidence, reason)

    def _ict_signal(self, f: Dict) -> StrategySignal:
        """ICT: Kill zones, OTE, Judas, Power of 3."""
        score = 0
        reasons = []

        # Judas swing (strong reversal signal)
        if f.get("ICT_Judas_Bull", 0) > 0:
            score += 3; reasons.append("bullish Judas swing")
        if f.get("ICT_Judas_Bear", 0) > 0:
            score -= 3; reasons.append("bearish Judas swing")

        # Optimal Trade Entry zone
        in_ote_10 = f.get("ICT_In_OTE_10", 0)
        in_ote_20 = f.get("ICT_In_OTE_20", 0)
        if in_ote_10 or in_ote_20:
            score += 2; reasons.append("in OTE zone")

        # Kill zone
        if f.get("ICT_In_Kill_Zone", 0) > 0:
            kz_dir = f.get("ICT_KZ_Direction", 0)
            if kz_dir > 0:
                score += 1; reasons.append("bullish kill zone")
            elif kz_dir < 0:
                score -= 1; reasons.append("bearish kill zone")

        # Squeeze → release
        if f.get("ICT_Squeeze", 0) > 0:
            score += 1; reasons.append("squeeze breakout")

        # Power of 3 phase
        po3 = f.get("ICT_PO3_Phase", 0)
        if po3 == 3:  # Distribution phase
            reasons.append("PO3 distribution phase")
            # Direction from recent returns
            score += 1

        # Rejection candles
        if f.get("ICT_Bull_Rejection", 0) > 0:
            score += 2; reasons.append("bullish rejection")
        if f.get("ICT_Bear_Rejection", 0) > 0:
            score -= 2; reasons.append("bearish rejection")

        # Fibonacci level
        fib = f.get("ICT_Fib_Level_20", 0)
        if 0.5 <= fib <= 0.786:
            score += 1; reasons.append(f"at {fib:.1%} fib retracement")

        signal = 1 if score >= 3 else (-1 if score <= -3 else 0)
        confidence = min(abs(score) / 8.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "no ICT setup"

        return StrategySignal("ICT", signal, confidence, reason)

    def _wyckoff_signal(self, f: Dict) -> StrategySignal:
        """Wyckoff: Phases, spring/upthrust, effort-result, climax."""
        score = 0
        reasons = []

        # Phases
        phase = f.get("WY_Phase", 0)
        if phase == 1:
            score += 1; reasons.append("accumulation phase")
        elif phase == 2:
            score += 2; reasons.append("markup phase")
        elif phase == 3:
            score -= 1; reasons.append("distribution phase")
        elif phase == 4:
            score -= 2; reasons.append("markdown phase")

        # Spring / Upthrust (strong reversal)
        if f.get("WY_Spring", 0) > 0:
            score += 3; reasons.append("spring detected!")
        if f.get("WY_Upthrust", 0) > 0:
            score -= 3; reasons.append("upthrust detected!")

        # Effort vs Result
        if f.get("WY_Absorption", 0) > 0:
            reasons.append("absorption (smart money)")
            # Absorption in downtrend = bullish, in uptrend = bearish
            trend = f.get("WY_Trend", 0)
            if trend < 0:
                score += 2
            else:
                score -= 2

        # Climax
        if f.get("WY_Selling_Climax", 0) > 0:
            score += 2; reasons.append("selling climax (bottom)")
        if f.get("WY_Buying_Climax", 0) > 0:
            score -= 2; reasons.append("buying climax (top)")

        # VSA signals
        if f.get("WY_No_Supply", 0) > 0:
            score += 1; reasons.append("no supply bar")
        if f.get("WY_No_Demand", 0) > 0:
            score -= 1; reasons.append("no demand bar")

        # Stopping volume
        if f.get("WY_Stopping_Vol", 0) > 0:
            reasons.append("stopping volume")

        signal = 1 if score >= 3 else (-1 if score <= -3 else 0)
        confidence = min(abs(score) / 8.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "no Wyckoff setup"

        return StrategySignal("Wyckoff", signal, confidence, reason)

    def _elliott_wave_signal(self, f: Dict) -> StrategySignal:
        """Elliott Wave: Wave position, extensions, divergence."""
        score = 0
        reasons = []

        wave_type = f.get("EW_Wave_Type", 0)  # 1=bullish impulse, -1=bearish

        # Wave 3 = strongest (trade with the trend)
        if f.get("EW_In_Wave3", 0) > 0:
            if wave_type > 0:
                score += 3; reasons.append("in bullish Wave 3 (strongest)")
            elif wave_type < 0:
                score -= 3; reasons.append("in bearish Wave 3")

        # Wave 5 = exhaustion (prepare for reversal)
        if f.get("EW_In_Wave5", 0) > 0:
            if wave_type > 0:
                score -= 1; reasons.append("Wave 5 exhaustion (bearish reversal near)")
            elif wave_type < 0:
                score += 1; reasons.append("Wave 5 exhaustion (bullish reversal near)")

        # Divergence
        if f.get("EW_Bullish_Divergence", 0) > 0:
            score += 2; reasons.append("bullish divergence")
        if f.get("EW_Bearish_Divergence", 0) > 0:
            score -= 2; reasons.append("bearish divergence")

        # Momentum ratio
        mom = f.get("EW_Momentum_Ratio", 0)
        if mom > 1.5:
            score += 1; reasons.append("strong upward momentum")
        elif mom < -1.5:
            score -= 1; reasons.append("strong downward momentum")

        # Exhaustion
        if f.get("EW_Exhaustion", 0) > 0:
            reasons.append("wave exhaustion signal")

        signal = 1 if score >= 2 else (-1 if score <= -2 else 0)
        confidence = min(abs(score) / 6.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "no clear wave pattern"

        return StrategySignal("Elliott Wave", signal, confidence, reason)

    def _technical_signal(self, f: Dict) -> StrategySignal:
        """Technical Indicators: RSI, MACD, BB, ADX, Stochastic, etc."""
        score = 0
        reasons = []
        details = {}

        # RSI
        rsi = f.get("RSI_14", 50)
        details["rsi"] = rsi
        if rsi < 30:
            score += 2; reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 2; reasons.append("RSI overbought")
        elif rsi < 40:
            score += 1
        elif rsi > 60:
            score -= 1

        # MACD
        macd = f.get("MACD", 0)
        macd_signal = f.get("MACD_Signal", 0)
        macd_hist = f.get("MACD_Hist", 0)
        details["macd_hist"] = macd_hist
        if macd_hist > 0:
            score += 1; reasons.append("MACD bullish")
        elif macd_hist < 0:
            score -= 1; reasons.append("MACD bearish")

        # Bollinger Bands
        bb_percent = f.get("BB_Percent", 0.5)
        details["bb_percent"] = bb_percent
        if bb_percent < 0.1:
            score += 2; reasons.append("BB lower band bounce")
        elif bb_percent > 0.9:
            score -= 2; reasons.append("BB upper band rejection")

        # ADX (trend strength)
        adx = f.get("ADX", 0)
        plus_di = f.get("Plus_DI", 0)
        minus_di = f.get("Minus_DI", 0)
        details["adx"] = adx
        if adx > 25:  # Strong trend
            if plus_di > minus_di:
                score += 1; reasons.append("strong uptrend (ADX)")
            elif minus_di > plus_di:
                score -= 1; reasons.append("strong downtrend (ADX)")

        # Stochastic
        stoch_k = f.get("Stoch_K", 50)
        details["stoch_k"] = stoch_k
        if stoch_k < 20:
            score += 1; reasons.append("Stoch oversold")
        elif stoch_k > 80:
            score -= 1; reasons.append("Stoch overbought")

        # CCI
        cci = f.get("CCI", 0)
        if cci < -100:
            score += 1
        elif cci > 100:
            score -= 1

        # Aroon
        aroon_osc = f.get("Aroon_Osc", 0)
        if aroon_osc > 50:
            score += 1
        elif aroon_osc < -50:
            score -= 1

        signal = 1 if score >= 3 else (-1 if score <= -3 else 0)
        confidence = min(abs(score) / 8.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "neutral indicators"

        return StrategySignal("Technical", signal, confidence, reason, details)

    def _market_structure_signal(self, f: Dict) -> StrategySignal:
        """Market Structure: Support/Resistance, Regimes, Pivot Points."""
        score = 0
        reasons = []
        details = {}

        # Support/Resistance distance
        dist_resist = f.get("Dist_Resistance_20", 0)
        dist_support = f.get("Dist_Support_20", 0)
        range_pos = f.get("Range_Position_20", 0.5)
        details["range_position"] = range_pos

        # Near support = bullish
        if dist_support < 2:
            score += 2; reasons.append("near support")
        elif dist_resist < 2:
            score -= 2; reasons.append("near resistance")

        # Range position
        if range_pos < 0.2:
            score += 1; reasons.append("lower range (buy zone)")
        elif range_pos > 0.8:
            score -= 1; reasons.append("upper range (sell zone)")

        # Trend Regime
        trend_regime = f.get("Trend_Regime", 0)
        details["trend_regime"] = trend_regime
        if trend_regime > 0:
            score += 1; reasons.append("uptrend regime")
        else:
            score -= 1; reasons.append("downtrend regime")

        # Volatility Regime
        vol_regime = f.get("Vol_Regime", 1)
        details["vol_regime"] = vol_regime
        if vol_regime > 1.5:
            reasons.append("high volatility regime")

        # Pivot Points
        pivot = f.get("Pivot", 0)
        close = f.get("Close", 0)
        if pivot > 0 and close > pivot:
            score += 1; reasons.append("above pivot")
        elif pivot > 0 and close < pivot:
            score -= 1; reasons.append("below pivot")

        # Fibonacci levels
        fib_618 = f.get("Fib_618", 0)
        if fib_618 > 0 and abs(close - fib_618) / close < 0.01:
            reasons.append("at 61.8% fib level")

        signal = 1 if score >= 2 else (-1 if score <= -2 else 0)
        confidence = min(abs(score) / 6.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "neutral structure"

        return StrategySignal("Market Structure", signal, confidence, reason, details)

    def _sentiment_signal(self, f: Dict) -> StrategySignal:
        """Sentiment Analysis: FinBERT sentiment scores."""
        score = 0
        reasons = []
        details = {}

        # Sentiment score (-1 to 1)
        sentiment = f.get("Sentiment_Score", 0)
        sentiment_sma = f.get("Sentiment_SMA_10", 0)
        sentiment_mom = f.get("Sentiment_Momentum", 0)

        details["sentiment"] = sentiment
        details["sentiment_momentum"] = sentiment_mom

        # Strong sentiment
        if sentiment > 0.5:
            score += 2; reasons.append("strong positive sentiment")
        elif sentiment < -0.5:
            score -= 2; reasons.append("strong negative sentiment")
        elif sentiment > 0.2:
            score += 1; reasons.append("positive sentiment")
        elif sentiment < -0.2:
            score -= 1; reasons.append("negative sentiment")

        # Sentiment momentum
        if sentiment_mom > 0.2:
            score += 1; reasons.append("sentiment improving")
        elif sentiment_mom < -0.2:
            score -= 1; reasons.append("sentiment worsening")

        # Sentiment SMA crossover
        if sentiment > sentiment_sma and sentiment_sma < 0:
            score += 1; reasons.append("sentiment turning positive")
        elif sentiment < sentiment_sma and sentiment_sma > 0:
            score -= 1; reasons.append("sentiment turning negative")

        signal = 1 if score >= 2 else (-1 if score <= -2 else 0)
        confidence = min(abs(score) / 5.0, 1.0)
        reason = ", ".join(reasons[:3]) if reasons else "neutral sentiment"

        return StrategySignal("Sentiment", signal, confidence, reason, details)
