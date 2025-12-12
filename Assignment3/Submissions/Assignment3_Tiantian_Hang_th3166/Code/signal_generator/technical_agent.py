"""
Technical Agent - Analyzes technical indicators to generate trading signals
Based on FinRL technical analysis logic
"""

import pandas as pd
import numpy as np


class TechnicalAgent:
    """
    Technical analysis agent based on FinRL indicators
    """
    
    def __init__(self, indicators_config=None):
        """
        Initialize technical agent
        
        Args:
            indicators_config: dict with thresholds for each indicator
        """
        self.config = indicators_config or self._default_config()
    
    def _default_config(self):
        """Default configuration for technical indicators"""
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'adx_trend_threshold': 25,
            'macd_bullish_threshold': 0
        }
    
    def analyze(self, tech_data):
        """
        Analyze technical indicators and generate signal
        
        Args:
            tech_data: Series or dict with technical indicators
                Required: rsi, macd, cci, adx, etc.
        
        Returns:
            dict: {
                'signal': 1 (buy), -1 (sell), 0 (hold),
                'confidence': 0.0-1.0,
                'indicators': {
                    'rsi': 'oversold'/'overbought'/'neutral',
                    'macd': 'bullish'/'bearish'/'neutral',
                    ...
                },
                'raw_score': float
            }
        """
        scores = []
        indicators_status = {}
        
        # RSI analysis
        rsi = tech_data.get('rsi', 50) if isinstance(tech_data, dict) else tech_data.get('rsi', 50)
        if pd.isna(rsi):
            rsi = 50
        
        if rsi < self.config['rsi_oversold']:
            scores.append(1)
            indicators_status['rsi'] = 'oversold'
        elif rsi > self.config['rsi_overbought']:
            scores.append(-1)
            indicators_status['rsi'] = 'overbought'
        else:
            scores.append(0)
            indicators_status['rsi'] = 'neutral'
        
        # MACD analysis
        macd = tech_data.get('macd', 0) if isinstance(tech_data, dict) else tech_data.get('macd', 0)
        macd_signal = tech_data.get('macd_signal', 0) if isinstance(tech_data, dict) else tech_data.get('macd_signal', 0)
        
        if pd.isna(macd):
            macd = 0
        if pd.isna(macd_signal):
            macd_signal = 0
        
        if macd > macd_signal:
            scores.append(0.5)
            indicators_status['macd'] = 'bullish'
        else:
            scores.append(-0.5)
            indicators_status['macd'] = 'bearish'
        
        # CCI analysis
        cci = tech_data.get('cci', 0) if isinstance(tech_data, dict) else tech_data.get('cci', 0)
        if pd.isna(cci):
            cci = 0
        
        if cci < self.config['cci_oversold']:
            scores.append(1)
            indicators_status['cci'] = 'oversold'
        elif cci > self.config['cci_overbought']:
            scores.append(-1)
            indicators_status['cci'] = 'overbought'
        else:
            scores.append(0)
            indicators_status['cci'] = 'neutral'
        
        # ADX analysis (trend strength)
        adx = tech_data.get('adx', 0) if isinstance(tech_data, dict) else tech_data.get('adx', 0)
        if pd.isna(adx):
            adx = 0
        
        if adx > self.config['adx_trend_threshold']:
            # Strong trend - increase confidence
            indicators_status['adx'] = 'strong_trend'
        else:
            indicators_status['adx'] = 'weak_trend'
        
        # Calculate final signal
        total_score = sum(scores)
        signal = 1 if total_score > 0.5 else (-1 if total_score < -0.5 else 0)
        
        # Confidence based on score magnitude and ADX
        base_confidence = min(abs(total_score) / len(scores), 1.0) if len(scores) > 0 else 0.5
        if indicators_status.get('adx') == 'strong_trend':
            confidence = min(base_confidence * 1.2, 1.0)
        else:
            confidence = base_confidence * 0.8
        
        return {
            'signal': signal,
            'confidence': confidence,
            'indicators': indicators_status,
            'raw_score': total_score
        }


