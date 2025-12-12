"""
Technical Agent Optimized - Uses Mean-Variance Optimization
Based on FinRL portfolio optimization approach
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class TechnicalAgentOptimized:
    """
    Improved Technical Agent that uses mean-variance optimization
    to calculate portfolio weights based on technical indicators
    """
    
    def __init__(self, indicators_config=None, use_optimization=True):
        """
        Initialize optimized technical agent
        
        Args:
            indicators_config: dict with thresholds for each indicator
            use_optimization: whether to use mean-variance optimization
        """
        self.config = indicators_config or self._default_config()
        self.use_optimization = use_optimization
        
        # Try to import PyPortfolioOpt
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt import risk_models, objective_functions
            self.optimization_available = True
        except ImportError:
            print("Warning: PyPortfolioOpt not available. Falling back to simple method.")
            self.optimization_available = False
            self.use_optimization = False
    
    def _default_config(self):
        """Default configuration for technical indicators"""
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'adx_trend_threshold': 25,
            'macd_bullish_threshold': 0,
            'weight_bounds': (0, 0.05),  # Max 5% per stock
            'min_history_days': 252  # 1 year of data
        }
    
    def analyze_portfolio(self, tech_data_dict: Dict, historical_returns: pd.DataFrame) -> Dict:
        """
        Analyze multiple stocks using technical indicators and optimize portfolio weights
        
        Args:
            tech_data_dict: dict of {gvkey: {rsi, macd, cci, adx, ...}}
            historical_returns: DataFrame with historical returns
                - columns: stock identifiers (gvkey or tic)
                - index: dates
                - values: daily returns
        
        Returns:
            dict: {
                'weights': {gvkey: weight},
                'expected_returns': {gvkey: expected_return},
                'confidence': float,
                'sharpe_ratio': float,
                'method': str,
                'signal': float  # Overall portfolio signal
            }
        """
        if not self.use_optimization or not self.optimization_available:
            # Fall back to simple method
            return self._analyze_simple(tech_data_dict)
        
        # Check if we have enough data
        if len(historical_returns) < self.config['min_history_days']:
            print(f"Warning: Insufficient historical data ({len(historical_returns)} days). "
                  f"Need at least {self.config['min_history_days']} days.")
            return self._analyze_simple(tech_data_dict)
        
        # 1. Predict expected returns from technical indicators
        predicted_returns = {}
        for gvkey, tech_data in tech_data_dict.items():
            predicted_returns[gvkey] = self._predict_return_from_indicators(tech_data)
        
        # 2. Filter stocks that exist in both predicted_returns and historical_returns
        common_stocks = set(predicted_returns.keys()) & set(historical_returns.columns)
        if len(common_stocks) == 0:
            print("Warning: No common stocks between predictions and historical data.")
            return self._analyze_simple(tech_data_dict)
        
        # 3. Prepare data for optimization
        mu = pd.Series({s: predicted_returns[s] for s in common_stocks})
        S = historical_returns[list(common_stocks)].cov()
        
        # 4. Mean-variance optimization
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt import risk_models, objective_functions
            
            ef = EfficientFrontier(mu, S, weight_bounds=self.config['weight_bounds'])
            raw_weights = ef.nonconvex_objective(
                objective_functions.sharpe_ratio,
                objective_args=(ef.expected_returns, ef.cov_matrix),
                weights_sum_to_one=True
            )
            cleaned_weights = ef.clean_weights()
            
            # Re-normalize weights after cleaning (clean_weights may remove small weights)
            total_weight = sum(cleaned_weights.values())
            if total_weight > 0:
                cleaned_weights = {k: v / total_weight for k, v in cleaned_weights.items()}
            
            # 5. Calculate portfolio performance metrics
            performance = ef.portfolio_performance(verbose=False)
            expected_return = performance[0]
            volatility = performance[1]
            sharpe_ratio = performance[2]
            
            # 6. Calculate overall signal (weighted average of predicted returns)
            overall_signal = sum(cleaned_weights.get(s, 0) * predicted_returns.get(s, 0) 
                                for s in common_stocks)
            
            # 7. Confidence based on Sharpe ratio and signal strength
            confidence = min(abs(sharpe_ratio) / 2.0, 1.0) if sharpe_ratio is not None else 0.5
            if abs(overall_signal) > 0.1:
                confidence = min(confidence * 1.2, 1.0)
            
            return {
                'weights': cleaned_weights,
                'expected_returns': predicted_returns,
                'confidence': confidence,
                'sharpe_ratio': sharpe_ratio,
                'expected_return': expected_return,
                'volatility': volatility,
                'method': 'mean_variance_optimization',
                'signal': np.tanh(overall_signal * 10)  # Normalize to [-1, 1]
            }
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            return self._analyze_simple(tech_data_dict)
    
    def _predict_return_from_indicators(self, tech_data: Dict) -> float:
        """
        Predict expected return from technical indicators
        
        This is a simple model. Can be replaced with ML models.
        
        Args:
            tech_data: dict with technical indicators
        
        Returns:
            float: predicted return
        """
        rsi = tech_data.get('rsi', 50)
        macd = tech_data.get('macd', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        cci = tech_data.get('cci', 0)
        adx = tech_data.get('adx', 0)
        
        # Handle NaN values
        rsi = 50 if pd.isna(rsi) else rsi
        macd = 0 if pd.isna(macd) else macd
        macd_signal = 0 if pd.isna(macd_signal) else macd_signal
        cci = 0 if pd.isna(cci) else cci
        adx = 0 if pd.isna(adx) else adx
        
        # RSI: Low RSI (< 30) -> oversold -> positive expected return
        #      High RSI (> 70) -> overbought -> negative expected return
        rsi_score = (50 - rsi) / 50  # Normalize to [-1, 1]
        
        # MACD: MACD > Signal -> bullish -> positive expected return
        macd_diff = macd - macd_signal
        macd_score = np.tanh(macd_diff * 10)  # Normalize
        
        # CCI: Low CCI (< -100) -> oversold -> positive expected return
        #      High CCI (> 100) -> overbought -> negative expected return
        cci_score = -cci / 200  # Normalize to approximately [-1, 1]
        
        # ADX: High ADX (> 25) -> strong trend -> increase confidence
        #      But doesn't directly predict direction, so weight it less
        adx_factor = 1.0 + (adx / 50) if adx > self.config['adx_trend_threshold'] else 1.0
        
        # Combine indicators (weighted average)
        predicted_return = (
            rsi_score * 0.4 +
            macd_score * 0.3 +
            cci_score * 0.3
        ) * 0.1 * adx_factor  # Scale to reasonable return range
        
        return predicted_return
    
    def _analyze_simple(self, tech_data_dict: Dict) -> Dict:
        """
        Fallback method: simple analysis without optimization
        
        Args:
            tech_data_dict: dict of {gvkey: {rsi, macd, ...}}
        
        Returns:
            dict: Simple signal-based result
        """
        # Calculate equal weights
        n_stocks = len(tech_data_dict)
        if n_stocks == 0:
            return {
                'weights': {},
                'confidence': 0.0,
                'signal': 0.0,
                'method': 'simple_equal_weight'
            }
        
        equal_weight = 1.0 / n_stocks
        weights = {gvkey: equal_weight for gvkey in tech_data_dict.keys()}
        
        # Calculate average signal
        signals = []
        for gvkey, tech_data in tech_data_dict.items():
            signal = self._calculate_simple_signal(tech_data)
            signals.append(signal)
        
        avg_signal = np.mean(signals) if signals else 0.0
        confidence = min(abs(avg_signal), 1.0)
        
        return {
            'weights': weights,
            'confidence': confidence,
            'signal': np.tanh(avg_signal * 2),  # Normalize to [-1, 1]
            'method': 'simple_equal_weight'
        }
    
    def _calculate_simple_signal(self, tech_data: Dict) -> float:
        """
        Calculate simple signal from technical indicators
        
        Args:
            tech_data: dict with technical indicators
        
        Returns:
            float: signal value
        """
        rsi = tech_data.get('rsi', 50)
        macd = tech_data.get('macd', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        cci = tech_data.get('cci', 0)
        
        # Handle NaN
        rsi = 50 if pd.isna(rsi) else rsi
        macd = 0 if pd.isna(macd) else macd
        macd_signal = 0 if pd.isna(macd_signal) else macd_signal
        cci = 0 if pd.isna(cci) else cci
        
        scores = []
        
        # RSI
        if rsi < self.config['rsi_oversold']:
            scores.append(1.0)
        elif rsi > self.config['rsi_overbought']:
            scores.append(-1.0)
        else:
            scores.append(0.0)
        
        # MACD
        if macd > macd_signal:
            scores.append(0.5)
        else:
            scores.append(-0.5)
        
        # CCI
        if cci < self.config['cci_oversold']:
            scores.append(1.0)
        elif cci > self.config['cci_overbought']:
            scores.append(-1.0)
        else:
            scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def analyze(self, tech_data: Dict) -> Dict:
        """
        Analyze single stock (backward compatibility with original interface)
        
        Args:
            tech_data: dict or Series with technical indicators
        
        Returns:
            dict: {
                'signal': 1 (buy), -1 (sell), 0 (hold),
                'confidence': 0.0-1.0,
                'indicators': {...},
                'raw_score': float
            }
        """
        signal = self._calculate_simple_signal(tech_data)
        
        # Determine action
        if signal > 0.3:
            action_signal = 1
        elif signal < -0.3:
            action_signal = -1
        else:
            action_signal = 0
        
        confidence = min(abs(signal), 1.0)
        
        return {
            'signal': action_signal,
            'confidence': confidence,
            'indicators': {
                'rsi': 'oversold' if tech_data.get('rsi', 50) < 30 
                       else 'overbought' if tech_data.get('rsi', 50) > 70 
                       else 'neutral',
                'macd': 'bullish' if tech_data.get('macd', 0) > tech_data.get('macd_signal', 0)
                       else 'bearish',
            },
            'raw_score': signal
        }


# Example usage
if __name__ == "__main__":
    # Example: Analyze portfolio with optimization
    agent = TechnicalAgentOptimized(use_optimization=True)
    
    # Mock technical data
    tech_data_dict = {
        'AAPL': {'rsi': 35, 'macd': 0.5, 'macd_signal': 0.3, 'cci': -50, 'adx': 30},
        'MSFT': {'rsi': 65, 'macd': -0.2, 'macd_signal': 0.1, 'cci': 80, 'adx': 20},
        'GOOGL': {'rsi': 45, 'macd': 0.1, 'macd_signal': 0.05, 'cci': 10, 'adx': 25}
    }
    
    # Mock historical returns (would be loaded from actual data)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    historical_returns = pd.DataFrame(
        np.random.randn(252, 3) * 0.02,  # Random returns
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL']
    )
    
    # Analyze portfolio
    result = agent.analyze_portfolio(tech_data_dict, historical_returns)
    
    print("Portfolio Analysis Result:")
    print(f"  Method: {result['method']}")
    print(f"  Weights: {result['weights']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    if 'sharpe_ratio' in result:
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"  Signal: {result['signal']:.3f}")
