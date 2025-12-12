"""
Backtester module - Integrates all components for backtesting
Based on FRAMEWORK_DESIGN.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from tqdm import tqdm
import time


class DualInsightBacktester:
    """
    Integrated backtester for Dual-Insight Trader
    """
    
    def __init__(self, 
                 price_data,
                 news_fetcher,
                 sentiment_agent,
                 technical_agent,
                 portfolio_manager):
        """
        Initialize backtester
        
        Args:
            price_data: DataFrame with price data
            news_fetcher: NewsDataFetcher instance
            sentiment_agent: SentimentAgent instance
            technical_agent: TechnicalAgent instance
            portfolio_manager: PortfolioManager instance
        """
        self.price_data = price_data
        self.news_fetcher = news_fetcher
        self.sentiment_agent = sentiment_agent
        self.technical_agent = technical_agent
        self.portfolio_manager = portfolio_manager
        
        self.portfolio_value = []
        self.positions = []
        self.trades = []
    
    def _get_trading_dates(self, start_date, end_date):
        """Get list of trading dates"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start, end)
        return [d.strftime('%Y-%m-%d') for d in dates]
    
    def _get_previous_trading_day(self, date_str):
        """Get previous trading day"""
        date = pd.to_datetime(date_str)
        prev_date = date - BDay(1)
        return prev_date.strftime('%Y-%m-%d')
    
    def _get_news_for_date(self, symbol, date):
        """
        Get news data for trading decision
        Uses previous trading day's news (avoid future information leak)
        """
        prev_date = self._get_previous_trading_day(date)
        n_weeks = 3
        
        try:
            news_data = self.news_fetcher.fetch_all_data(symbol, prev_date, n_weeks)
            return news_data
        except Exception as e:
            print(f"Warning: Failed to fetch news for {symbol} on {date}: {e}")
            return None
    
    def _calculate_technical_indicators(self, price_series):
        """
        Calculate technical indicators from price series
        This is a simplified version - in practice, you'd use a library like ta-lib
        """
        indicators = {}
        
        # Simple RSI calculation (14-period)
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1] if len(rs) > 0 else 50
        
        # Simple MACD
        ema12 = price_series.ewm(span=12).mean()
        ema26 = price_series.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = macd.iloc[-1] if len(macd) > 0 else 0
        indicators['macd_signal'] = signal.iloc[-1] if len(signal) > 0 else 0
        
        # Simple CCI (20-period)
        typical_price = price_series
        sma = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        indicators['cci'] = ((typical_price.iloc[-1] - sma.iloc[-1]) / (0.015 * mad.iloc[-1])) if len(mad) > 0 and mad.iloc[-1] != 0 else 0
        
        # ADX (simplified - would need high/low/close in practice)
        indicators['adx'] = 25  # Placeholder
        
        return indicators
    
    def run_backtest(self, start_date, end_date, initial_capital=1000000, symbols=None, 
                     test_mode=False, max_stocks=None, show_progress=True):
        """
        Run backtest
        
        Args:
            start_date: Start date string
            end_date: End date string
            initial_capital: Initial capital
            symbols: List of symbols to trade (if None, uses all in price_data)
            test_mode: If True, enables test mode with limited processing
            max_stocks: Maximum number of stocks to process (for testing)
            show_progress: If True, shows progress bar
            
        Returns:
            dict with backtest results
        """
        current_capital = initial_capital
        current_positions = {}  # {symbol: shares}
        
        # Get trading dates
        trading_dates = self._get_trading_dates(start_date, end_date)
        
        # Get symbols
        if symbols is None:
            symbols = self.price_data['tic'].unique() if 'tic' in self.price_data.columns else []
        
        # Limit symbols for test mode
        if test_mode and max_stocks and len(symbols) > max_stocks:
            symbols = symbols[:max_stocks]
            print(f"⚠️  TEST MODE: Limiting to {max_stocks} stocks")
        
        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Trading {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        print(f"Total trading days: {len(trading_dates)}")
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(total=len(trading_dates) * len(symbols), 
                      desc="Processing", 
                      unit="stock-day",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        total_operations = 0
        for i, date in enumerate(trading_dates):
            # Process each symbol
            for symbol in symbols:
                if show_progress:
                    pbar.set_description(f"Date {i+1}/{len(trading_dates)}: {date[:10]} | {symbol}")
                
                total_operations += 1
                try:
                    # Get price data for this date
                    date_data = self.price_data[
                        (self.price_data['datadate'] == date) & 
                        (self.price_data['tic'] == symbol)
                    ]
                    
                    if len(date_data) == 0:
                        if show_progress:
                            pbar.update(1)
                        continue
                    
                    current_price = date_data['prccd'].iloc[0]
                    
                    # Get news data
                    news_data = self._get_news_for_date(symbol, date)
                    
                    # Get technical indicators
                    price_history = self.price_data[
                        (self.price_data['tic'] == symbol) &
                        (self.price_data['datadate'] <= date)
                    ]['prccd'].tail(50)
                    
                    if len(price_history) < 20:
                        if show_progress:
                            pbar.update(1)
                        continue
                    
                    tech_indicators = self._calculate_technical_indicators(price_history)
                    
                    # Agent analysis
                    sentiment_result = None
                    if news_data is not None:
                        try:
                            sentiment_result = self.sentiment_agent.analyze(
                                news_data, symbol, date
                            )
                        except Exception as e:
                            print(f"Warning: Sentiment analysis failed for {symbol} on {date}: {e}")
                            sentiment_result = {'signal': 0, 'confidence': 0.3}
                    else:
                        sentiment_result = {'signal': 0, 'confidence': 0.3}
                    
                    technical_result = self.technical_agent.analyze(tech_indicators)
                    
                    # Portfolio manager decision
                    decision = self.portfolio_manager.make_decision(
                        sentiment_result,
                        technical_result
                    )
                    
                    # Execute trades (simplified - single stock at a time)
                    current_shares = current_positions.get(symbol, 0)
                    
                    if decision['action'] == 'BUY' and current_shares == 0:
                        # Buy
                        shares = int(current_capital * 0.1 / current_price)  # Use 10% of capital
                        if shares > 0:
                            cost = shares * current_price
                            current_capital -= cost
                            current_positions[symbol] = shares
                            
                            self.trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'price': current_price,
                                'shares': shares,
                                'reasoning': decision['reasoning']
                            })
                    
                    elif decision['action'] == 'SELL' and current_shares > 0:
                        # Sell
                        revenue = current_shares * current_price
                        current_capital += revenue
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': current_shares,
                            'reasoning': decision['reasoning']
                        })
                        
                        current_positions[symbol] = 0
                
                except Exception as e:
                    if not test_mode:  # Only print errors in non-test mode
                        print(f"Error processing {symbol} on {date}: {e}")
                    if show_progress:
                        pbar.update(1)
                    continue
                
                if show_progress:
                    pbar.update(1)
            
            # Calculate portfolio value
            portfolio_value = current_capital
            for symbol, shares in current_positions.items():
                date_data = self.price_data[
                    (self.price_data['datadate'] == date) & 
                    (self.price_data['tic'] == symbol)
                ]
                if len(date_data) > 0:
                    price = date_data['prccd'].iloc[0]
                    portfolio_value += shares * price
            
            self.portfolio_value.append({
                'date': date,
                'value': portfolio_value,
                'cash': current_capital,
                'positions_value': portfolio_value - current_capital
            })
        
        if show_progress:
            pbar.close()
        
        print(f"\n✓ Completed {total_operations} operations")
        return self._calculate_metrics(initial_capital)
    
    def _calculate_metrics(self, initial_capital):
        """Calculate performance metrics"""
        if len(self.portfolio_value) == 0:
            return {}
        
        df_values = pd.DataFrame(self.portfolio_value)
        df_values['date'] = pd.to_datetime(df_values['date'])
        df_values = df_values.set_index('date')
        
        returns = df_values['value'].pct_change().dropna()
        
        total_return = (df_values['value'].iloc[-1] / initial_capital - 1) * 100
        
        if len(returns) > 0:
            annualized_return = (1 + total_return/100) ** (252 / len(returns)) - 1
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': df_values['value'].iloc[-1],
            'total_trades': len(self.trades),
            'portfolio_values': df_values
        }

