"""
Portfolio Manager - Uses Mean-Variance Optimization with Sentiment Filtering
New Logic:
1. Filter stocks: Exclude stocks with sentiment predicted_return < -0.025
2. Use Technical weights: Apply mean-variance optimized weights from Technical Agent
3. No weight fusion: Sentiment is used as filter, not for weight calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class PortfolioManager:
    """
    Portfolio manager using mean-variance optimization with sentiment filtering
    
    New Strategy:
    - Sentiment acts as a filter: exclude stocks with predicted_return < -0.025
    - Technical provides optimized weights using mean-variance optimization
    - No weight fusion between sentiment and technical
    """
    
    def __init__(self, sentiment_threshold=-0.025, use_optimization=True):
        """
        Initialize portfolio manager
        
        Args:
            sentiment_threshold: Exclude stocks with sentiment predicted_return below this threshold
            use_optimization: Whether to use mean-variance optimized weights from technical
        """
        self.sentiment_threshold = sentiment_threshold
        self.use_optimization = use_optimization
    
    def make_decision(self, sentiment_result, technical_result, market_state=None):
        """
        Generate portfolio weights using new logic:
        1. Filter stocks based on sentiment predicted_return
        2. Apply technical optimized weights to remaining stocks
        
        Args:
            sentiment_result: dict or DataFrame with sentiment predictions
                - If dict: {'gvkey': {'predicted_return': float, ...}, ...}
                - If DataFrame: columns=['gvkey', 'predicted_return', ...]
            technical_result: dict from TechnicalAgentOptimized
                - Must contain 'weights': {gvkey: weight}
                - May contain 'expected_returns', 'sharpe_ratio', 'confidence'
            market_state: optional market condition dict (not used in new logic)
            
        Returns:
            dict: {
                'weights': {gvkey: weight},  # Final portfolio weights
                'filtered_stocks': [gvkey],   # Stocks excluded by sentiment filter
                'confidence': float,
                'sharpe_ratio': float,        # If available from technical
                'reasoning': str,
                'method': str
            }
        """
        # Convert sentiment_result to dict format if needed
        sentiment_dict = self._normalize_sentiment_result(sentiment_result)
        
        # Get technical weights
        if not isinstance(technical_result, dict) or 'weights' not in technical_result:
            raise ValueError(
                "technical_result must be a dict with 'weights' key. "
                "Use TechnicalAgentOptimized.analyze_portfolio() to generate weights."
            )
        
        technical_weights = technical_result['weights']
        
        # Step 1: Filter stocks based on sentiment predicted_return
        filtered_stocks, valid_stocks = self._filter_by_sentiment(
            sentiment_dict, 
            technical_weights
        )
        
        if len(valid_stocks) == 0:
            # No valid stocks after filtering
            return {
                'weights': {},
                'filtered_stocks': list(filtered_stocks),
                'confidence': 0.0,
                'sharpe_ratio': None,
                'reasoning': f"All stocks filtered out (sentiment predicted_return < {self.sentiment_threshold})",
                'method': 'sentiment_filtered'
            }
        
        # Step 2: Apply technical weights to valid stocks
        final_weights = self._apply_technical_weights(technical_weights, valid_stocks)
        
        # Step 3: Calculate confidence and metrics
        confidence = technical_result.get('confidence', 0.5)
        sharpe_ratio = technical_result.get('sharpe_ratio', None)
        
        reasoning = (
            f"Filtered {len(filtered_stocks)} stocks with sentiment predicted_return < {self.sentiment_threshold}. "
            f"Applied mean-variance optimized weights to {len(valid_stocks)} stocks. "
        )
        if sharpe_ratio is not None:
            reasoning += f"Portfolio Sharpe Ratio: {sharpe_ratio:.3f}"
        
        return {
            'weights': final_weights,
            'filtered_stocks': list(filtered_stocks),
            'confidence': confidence,
            'sharpe_ratio': sharpe_ratio,
            'expected_return': technical_result.get('expected_return', None),
            'volatility': technical_result.get('volatility', None),
            'reasoning': reasoning,
            'method': 'mean_variance_optimized_with_sentiment_filter'
        }
    
    def _normalize_sentiment_result(self, sentiment_result):
        """
        Normalize sentiment result to dict format: {gvkey: {'predicted_return': float, ...}}
        
        Args:
            sentiment_result: Can be:
                - dict: {gvkey: {'predicted_return': float, ...}}
                - DataFrame: with columns ['gvkey', 'predicted_return', ...]
                - dict: {'stocks': [{gvkey: ..., predicted_return: ...}, ...]}
        
        Returns:
            dict: {gvkey: {'predicted_return': float, ...}}
        """
        if isinstance(sentiment_result, pd.DataFrame):
            # DataFrame format
            if 'gvkey' not in sentiment_result.columns:
                raise ValueError("sentiment_result DataFrame must have 'gvkey' column")
            if 'predicted_return' not in sentiment_result.columns:
                raise ValueError("sentiment_result DataFrame must have 'predicted_return' column")
            
            result_dict = {}
            for _, row in sentiment_result.iterrows():
                gvkey = row['gvkey']
                result_dict[gvkey] = {
                    'predicted_return': float(row['predicted_return']),
                    **{k: v for k, v in row.items() if k not in ['gvkey', 'predicted_return']}
                }
            return result_dict
        
        elif isinstance(sentiment_result, dict):
            # Check if it's already in the right format
            if 'stocks' in sentiment_result:
                # Format: {'stocks': [{gvkey: ..., predicted_return: ...}, ...]}
                result_dict = {}
                for stock_data in sentiment_result['stocks']:
                    if 'gvkey' in stock_data:
                        gvkey = stock_data['gvkey']
                        result_dict[gvkey] = {
                            'predicted_return': float(stock_data.get('predicted_return', 0.0)),
                            **{k: v for k, v in stock_data.items() if k not in ['gvkey', 'predicted_return']}
                        }
                return result_dict
            else:
                # Format: {gvkey: {'predicted_return': float, ...}}
                # Verify structure
                for gvkey, data in sentiment_result.items():
                    if not isinstance(data, dict):
                        raise ValueError(
                            f"sentiment_result[{gvkey}] must be a dict with 'predicted_return' key"
                        )
                    if 'predicted_return' not in data:
                        raise ValueError(
                            f"sentiment_result[{gvkey}] must have 'predicted_return' key"
                        )
                return sentiment_result
        else:
            raise TypeError(
                f"sentiment_result must be dict or DataFrame, got {type(sentiment_result)}"
            )
    
    def _filter_by_sentiment(self, sentiment_dict: Dict, technical_weights: Dict) -> tuple:
        """
        Filter stocks based on sentiment predicted_return threshold
        
        Args:
            sentiment_dict: {gvkey: {'predicted_return': float, ...}}
            technical_weights: {gvkey: weight}
        
        Returns:
            tuple: (filtered_stocks_set, valid_stocks_set)
        """
        filtered_stocks = set()
        valid_stocks = set()
        
        # Get all stocks that have both sentiment and technical data
        all_stocks = set(sentiment_dict.keys()) & set(technical_weights.keys())
        
        for gvkey in all_stocks:
            predicted_return = sentiment_dict[gvkey].get('predicted_return', 0.0)
            
            if predicted_return < self.sentiment_threshold:
                # Filter out: predicted_return < threshold
                filtered_stocks.add(gvkey)
            else:
                # Keep: predicted_return >= threshold
                valid_stocks.add(gvkey)
        
        return filtered_stocks, valid_stocks
    
    def _apply_technical_weights(self, technical_weights: Dict, valid_stocks: set) -> Dict:
        """
        Apply technical weights to valid stocks and renormalize
        
        Args:
            technical_weights: {gvkey: weight} from TechnicalAgentOptimized
            valid_stocks: set of gvkeys that passed sentiment filter
        
        Returns:
            dict: {gvkey: normalized_weight} for valid stocks only
        """
        # Extract weights for valid stocks only
        filtered_weights = {
            gvkey: technical_weights.get(gvkey, 0.0)
            for gvkey in valid_stocks
        }
        
        # Remove zero weights
        filtered_weights = {k: v for k, v in filtered_weights.items() if v > 0}
        
        if len(filtered_weights) == 0:
            return {}
        
        # Renormalize to sum to 1
        total_weight = sum(filtered_weights.values())
        if total_weight > 0:
            normalized_weights = {
                gvkey: weight / total_weight
                for gvkey, weight in filtered_weights.items()
            }
        else:
            # If all weights are zero, use equal weights
            equal_weight = 1.0 / len(valid_stocks)
            normalized_weights = {
                gvkey: equal_weight
                for gvkey in valid_stocks
            }
        
        return normalized_weights
    
    def get_portfolio_weights(self, sentiment_result, technical_result, market_state=None):
        """
        Alias for make_decision for clarity
        
        Returns portfolio weights dictionary: {gvkey: weight}
        """
        result = self.make_decision(sentiment_result, technical_result, market_state)
        return result['weights']
    
    def get_filtered_stocks(self, sentiment_result, technical_result, market_state=None):
        """
        Get list of stocks filtered out by sentiment threshold
        
        Returns list of gvkeys that were excluded
        """
        result = self.make_decision(sentiment_result, technical_result, market_state)
        return result['filtered_stocks']

