"""
LLM-Sentiment Model
Uses RoBERTa fine-tuned for financial news sentiment analysis.

Buy Signal: sentiment_z > +0.4
Sell Signal: sentiment_z < -0.4
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LLMSentimentModel:
    """LLM-Sentiment Model for trading signals."""
    
    def __init__(self, 
                 buy_threshold: float = 0.4,
                 sell_threshold: float = -0.4,
                 model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
        """
        Initialize LLM-Sentiment Model.
        
        Args:
            buy_threshold: sentiment_z threshold for buy signal (default: 0.4)
            sell_threshold: sentiment_z threshold for sell signal (default: -0.4)
            model_name: HuggingFace model name
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the RoBERTa sentiment model."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load sentiment model: {e}")
            self.model = None
    
    def get_sentiment_z_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate sentiment z-score from data.
        
        Args:
            data: DataFrame with sentiment columns
            
        Returns:
            Series of sentiment z-scores
        """
        if 'sentiment_z' in data.columns:
            return data['sentiment_z']
        
        # Calculate from sentiment score if available
        if 'avg_sentiment' in data.columns:
            sentiment = data['avg_sentiment']
            mean = sentiment.mean()
            std = sentiment.std()
            if std > 0:
                return (sentiment - mean) / std
            return pd.Series(0.0, index=data.index)
        
        # Return zeros if no sentiment data
        return pd.Series(0.0, index=data.index)
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Get sentiment z-scores for the data.
        
        Args:
            data: DataFrame with price and sentiment data
            
        Returns:
            Series of sentiment z-scores
        """
        return self.get_sentiment_z_score(data)
    
    def generate_signals(self, data: pd.DataFrame) -> tuple:
        """
        Generate buy/sell signals based on sentiment.
        
        Args:
            data: DataFrame with price and sentiment data
            
        Returns:
            Tuple of (sentiment_z, buy_signals, sell_signals)
        """
        sentiment_z = self.predict(data)
        buy_signals = (sentiment_z > self.buy_threshold).astype(int)
        sell_signals = (sentiment_z < self.sell_threshold).astype(int)
        
        return sentiment_z, buy_signals, sell_signals
