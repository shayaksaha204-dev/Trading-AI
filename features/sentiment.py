"""
Sentiment Analyzer — FinBERT NLP for Financial Text
=====================================================
Uses FinBERT to extract sentiment from financial news/text.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.
    Loads the model lazily on first use to save memory.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = cfg.gpu.device
        self._loaded = False

    def _load_model(self):
        """Lazy-load FinBERT model."""
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = cfg.features.sentiment_model
            logger.info(f"Loading sentiment model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.success("Sentiment model loaded")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self._loaded = False

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a list of texts.
        Returns list of dicts with keys: positive, negative, neutral, score.
        """
        self._load_model()
        if not self._loaded:
            return [{"positive": 0.33, "negative": 0.33, "neutral": 0.34, "score": 0.0}] * len(texts)

        import torch
        results = []
        batch_size = cfg.features.sentiment_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=cfg.features.sentiment_max_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                for p in probs:
                    p_np = p.cpu().numpy()
                    # FinBERT: [positive, negative, neutral]
                    results.append({
                        "positive": float(p_np[0]),
                        "negative": float(p_np[1]),
                        "neutral": float(p_np[2]),
                        "score": float(p_np[0] - p_np[1]),  # -1 to 1
                    })
            except Exception as e:
                logger.warning(f"Sentiment batch failed: {e}")
                results.extend([{"positive": 0.33, "negative": 0.33, "neutral": 0.34, "score": 0.0}] * len(batch))

        return results

    def analyze_single(self, text: str) -> Dict[str, float]:
        """Analyze sentiment for a single text."""
        return self.analyze_texts([text])[0]

    def create_sentiment_features(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
    ) -> pd.DataFrame:
        """
        Add sentiment features to a DataFrame that has a text column.
        If no text column exists, creates placeholder neutral sentiment.
        """
        df = df.copy()
        if text_column in df.columns:
            texts = df[text_column].fillna("").tolist()
            sentiments = self.analyze_texts(texts)
            df["Sentiment_Score"] = [s["score"] for s in sentiments]
            df["Sentiment_Positive"] = [s["positive"] for s in sentiments]
            df["Sentiment_Negative"] = [s["negative"] for s in sentiments]
            df["Sentiment_Neutral"] = [s["neutral"] for s in sentiments]
        else:
            # Placeholder — no text data available
            df["Sentiment_Score"] = 0.0
            df["Sentiment_Positive"] = 0.33
            df["Sentiment_Negative"] = 0.33
            df["Sentiment_Neutral"] = 0.34

        # Rolling sentiment windows
        if "Sentiment_Score" in df.columns:
            for w in [5, 10, 20]:
                df[f"Sentiment_SMA_{w}"] = df["Sentiment_Score"].rolling(w).mean()
            df["Sentiment_Momentum"] = df["Sentiment_Score"] - df["Sentiment_Score"].shift(5)

        return df

    def create_sentiment_features_from_news(
        self,
        df: pd.DataFrame,
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Add sentiment features by fetching REAL news headlines for the ticker
        and running FinBERT on them.
        
        Falls back to placeholder if no headlines are found.
        
        Args:
            df: DataFrame with DatetimeIndex
            ticker: Asset symbol (e.g., "AAPL", "BTC-USD")
        """
        df = df.copy()
        
        # Try to fetch real headlines
        try:
            from data.news_fetcher import get_news_fetcher
            fetcher = get_news_fetcher()
            headlines = fetcher.fetch_headlines(ticker)
        except Exception as e:
            logger.debug(f"  Sentiment: Could not fetch news for {ticker}: {e}")
            headlines = []
        
        if headlines and len(headlines) >= 3:
            # Run FinBERT on real headlines
            texts = [h["headline"] for h in headlines]
            sentiments = self.analyze_texts(texts)
            
            # Compute aggregate sentiment
            scores = [s["score"] for s in sentiments]
            positives = [s["positive"] for s in sentiments]
            negatives = [s["negative"] for s in sentiments]
            neutrals = [s["neutral"] for s in sentiments]
            
            # Aggregate: weighted by recency (newer headlines matter more)
            n = len(scores)
            weights = np.exp(-np.arange(n) * 0.1)  # Exponential decay
            weights /= weights.sum()
            
            avg_score = float(np.average(scores, weights=weights))
            avg_positive = float(np.average(positives, weights=weights))
            avg_negative = float(np.average(negatives, weights=weights))
            avg_neutral = float(np.average(neutrals, weights=weights))
            
            # Also compute score distribution features
            score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
            score_range = float(max(scores) - min(scores)) if len(scores) > 1 else 0.0
            
            # Map headline sentiment to DataFrame (time-aligned)
            if isinstance(df.index, pd.DatetimeIndex) and headlines[0].get("published"):
                # Create time-mapped sentiment series
                sentiment_series = self._map_headlines_to_bars(df.index, headlines, sentiments)
                df["Sentiment_Score"] = sentiment_series
            else:
                # Use aggregate for all bars
                df["Sentiment_Score"] = avg_score
            
            df["Sentiment_Positive"] = avg_positive
            df["Sentiment_Negative"] = avg_negative
            df["Sentiment_Neutral"] = avg_neutral
            df["Sentiment_StdDev"] = score_std
            df["Sentiment_Range"] = score_range
            df["Sentiment_Headlines_Count"] = float(len(headlines))
            
            logger.debug(
                f"  Sentiment [{ticker}]: {len(headlines)} headlines, "
                f"score={avg_score:+.3f} (std={score_std:.3f})"
            )
        else:
            # Fallback: placeholder values
            df["Sentiment_Score"] = 0.0
            df["Sentiment_Positive"] = 0.33
            df["Sentiment_Negative"] = 0.33
            df["Sentiment_Neutral"] = 0.34
            df["Sentiment_StdDev"] = 0.0
            df["Sentiment_Range"] = 0.0
            df["Sentiment_Headlines_Count"] = 0.0
        
        # Rolling sentiment windows
        for w in [5, 10, 20]:
            df[f"Sentiment_SMA_{w}"] = df["Sentiment_Score"].rolling(w).mean()
        df["Sentiment_Momentum"] = df["Sentiment_Score"] - df["Sentiment_Score"].shift(5)
        
        return df
    
    def _map_headlines_to_bars(self, index, headlines, sentiments):
        """Map headline sentiments to the nearest bars in the DataFrame."""
        # Create a sentiment value for each bar based on most recent headline
        scores = pd.Series(0.0, index=index)
        
        # Sort headlines by published date
        headline_data = []
        for h, s in zip(headlines, sentiments):
            pub = h.get("published")
            if pub is not None:
                headline_data.append((pub, s["score"]))
        
        if not headline_data:
            return scores
        
        headline_data.sort(key=lambda x: x[0])
        
        # For each bar, find the most recent headline before it
        for pub_time, score in headline_data:
            # Forward-fill: this headline's sentiment applies to all bars after it
            mask = index >= pub_time
            if mask.any():
                scores[mask] = score
        
        # Apply exponential smoothing to avoid sudden jumps
        scores = scores.ewm(span=10, adjust=False).mean()
        
        return scores

    def unload(self):
        """Unload model to free GPU memory."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Sentiment model unloaded")


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_texts = [
        "Apple stock surges to all-time high on strong earnings",
        "Market crashes as recession fears grow",
        "Bitcoin trades sideways in a narrow range",
    ]
    results = analyzer.analyze_texts(test_texts)
    for text, r in zip(test_texts, results):
        print(f"  {r['score']:+.3f} | {text[:60]}")
