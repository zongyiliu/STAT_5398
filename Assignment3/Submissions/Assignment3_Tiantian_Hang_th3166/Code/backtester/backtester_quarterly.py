"""
Quarterly Rebalancing Backtester
Based on FinRL-Trading-master/fundamental_back_testing.py
Only trades on trade_date (quarterly rebalancing dates), holds positions between trades
No short selling allowed
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Transaction cost constant (0.1% per turnover)
TRANSACTION_COST = 0.001


class QuarterlyBacktester:
    """
    Backtester that only trades on quarterly rebalancing dates
    """
    
    def __init__(self, 
                 price_data,
                 selected_stocks,
                 news_fetcher=None,
                 sentiment_agent=None,
                 technical_agent=None,
                 portfolio_manager=None):
        """
        Initialize backtester
        
        Args:
            price_data: DataFrame with price data (columns: gvkey, tic, datadate, prccd, ajexdi)
            selected_stocks: DataFrame with selected stocks (columns: gvkey, trade_date, predicted_return, etc.)
            news_fetcher: Optional NewsDataFetcher instance
            sentiment_agent: Optional SentimentAgent instance
            technical_agent: Optional TechnicalAgent instance
            portfolio_manager: Optional PortfolioManager instance
        """
        self.price_data = price_data.copy()
        self.selected_stocks = selected_stocks.copy()
        self.news_fetcher = news_fetcher
        self.sentiment_agent = sentiment_agent
        self.technical_agent = technical_agent
        self.portfolio_manager = portfolio_manager
        
        # Prepare price data
        self._prepare_price_data()
        
        # Get trade dates
        self.trade_dates = self._get_trade_dates()
        
        # Results storage
        self.portfolio_values = []
        self.positions_history = []
        self.trades = []
    
    def _prepare_price_data(self):
        """Prepare price data with adjusted close prices and create wide format"""
        # Convert datadate to datetime
        if 'datadate' in self.price_data.columns:
            if self.price_data['datadate'].dtype == 'int64':
                self.price_data['date'] = pd.to_datetime(self.price_data['datadate'], format='%Y%m%d')
            else:
                self.price_data['date'] = pd.to_datetime(self.price_data['datadate'])
        else:
            raise ValueError("price_data must have 'datadate' column")
        
        # Calculate adjusted close price
        if 'ajexdi' in self.price_data.columns and 'prccd' in self.price_data.columns:
            self.price_data['adj_close_q'] = self.price_data['prccd'] / self.price_data['ajexdi']
        elif 'prccd' in self.price_data.columns:
            self.price_data['adj_close_q'] = self.price_data['prccd']
        else:
            raise ValueError("price_data must have 'prccd' column")
        
        # Sort by date
        self.price_data = self.price_data.sort_values(['date', 'gvkey'])
        
        # Create wide format: [date × gvkey] with adjusted close prices
        # This matches the format expected by build_portfolio_daily_returns
        self.price_wide = self.price_data.pivot_table(
            index='date', 
            columns='gvkey', 
            values='adj_close_q', 
            aggfunc='last'
        )
        self.price_wide = self.price_wide.sort_index()
        self.price_wide = self.price_wide.replace([np.inf, -np.inf], np.nan)
    
    def _get_trade_dates(self):
        """Extract unique trade dates from selected stocks"""
        if 'trade_date' in self.selected_stocks.columns:
            # Convert to datetime if needed
            if self.selected_stocks['trade_date'].dtype == 'object':
                self.selected_stocks['trade_date'] = pd.to_datetime(self.selected_stocks['trade_date'])
            elif self.selected_stocks['trade_date'].dtype == 'int64':
                self.selected_stocks['trade_date'] = pd.to_datetime(self.selected_stocks['trade_date'], format='%Y%m%d')
            
            trade_dates = sorted(self.selected_stocks['trade_date'].unique())
            return trade_dates
        else:
            raise ValueError("selected_stocks must have 'trade_date' column")
    
    def _prepare_weights(self):
        """Prepare weights DataFrame in the format expected by build_portfolio_daily_returns
        
        Weight calculation method:
        - If 'weights' column exists, use it directly
        - Otherwise, use predicted_return-based weights (not equal weights)
        - Weights are proportional to predicted_return, normalized to sum to 1
        """
        # Ensure we have the required columns
        required_cols = ['gvkey', 'trade_date']
        if not all(col in self.selected_stocks.columns for col in required_cols):
            raise ValueError(f"selected_stocks must have columns: {required_cols}")
        
        # If no weights column, create weights based on predicted_return
        if 'weights' not in self.selected_stocks.columns:
            # Check if predicted_return column exists
            if 'predicted_return' not in self.selected_stocks.columns:
                # Fallback to equal weights if no predicted_return
                print("Warning: No 'predicted_return' column found, using equal weights as fallback")
                def assign_equal_weights(group):
                    n = len(group)
                    group['weights'] = 1.0 / n if n > 0 else 0.0
                    return group
                self.selected_stocks = self.selected_stocks.groupby('trade_date', group_keys=False).apply(assign_equal_weights)
            else:
                # Use predicted_return-based weights
                def assign_predicted_return_weights(group):
                    """
                    Assign weights based on predicted_return
                    Weight = predicted_return / sum(predicted_return) for positive returns
                    Stocks with negative predicted_return get weight 0
                    """
                    n = len(group)
                    if n == 0:
                        group['weights'] = 0.0
                        return group
                    
                    predicted_returns = group['predicted_return'].values
                    
                    # Filter out negative predicted returns (set weight to 0)
                    positive_mask = predicted_returns > 0
                    
                    if positive_mask.sum() == 0:
                        # All predicted returns are non-positive, use equal weights
                        group['weights'] = 1.0 / n
                        return group
                    
                    # Calculate weights based on predicted_return
                    weights = np.zeros(n)
                    positive_returns = predicted_returns[positive_mask]
                    total_positive = positive_returns.sum()
                    
                    if total_positive > 0:
                        # Normalize by sum of positive returns
                        weights[positive_mask] = positive_returns / total_positive
                    else:
                        # Fallback to equal weights for positive stocks
                        weights[positive_mask] = 1.0 / positive_mask.sum()
                    
                    group['weights'] = weights
                    return group
                
                self.selected_stocks = self.selected_stocks.groupby('trade_date', group_keys=False).apply(assign_predicted_return_weights)
        
        # Normalize weights per trade_date to sum to 1
        def normalize_weights(group):
            s = group['weights'].sum()
            if not np.isclose(s, 1.0, atol=1e-6):
                if s > 0:
                    group['weights'] = group['weights'] / s
                else:
                    # If all weights are zero, use equal weights
                    n = len(group)
                    group['weights'] = 1.0 / n if n > 0 else 0.0
            return group
        
        weights_df = self.selected_stocks.groupby('trade_date', group_keys=False).apply(normalize_weights)
        
        # Ensure no negative weights (no short selling)
        if (weights_df['weights'] < 0).any():
            raise ValueError("Negative weights encountered (short-selling not allowed)")
        
        return weights_df[['gvkey', 'trade_date', 'weights']].copy()
    
    def _get_stocks_for_date(self, trade_date):
        """Get selected stocks for a given trade date"""
        mask = self.selected_stocks['trade_date'] == trade_date
        return self.selected_stocks[mask].copy()
    
    def _get_price_for_date(self, gvkey, date):
        """Get adjusted close price for a stock on a given date"""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        mask = (self.price_data['gvkey'] == gvkey) & (self.price_data['date'] == date)
        
        if mask.sum() > 0:
            return self.price_data[mask]['adj_close_q'].iloc[0]
        else:
            # Try to find nearest date
            stock_data = self.price_data[self.price_data['gvkey'] == gvkey].copy()
            if len(stock_data) > 0:
                stock_data = stock_data.sort_values('date')
                # Forward fill
                stock_data = stock_data[stock_data['date'] <= date]
                if len(stock_data) > 0:
                    return stock_data['adj_close_q'].iloc[-1]
            return None
    
    def _calculate_portfolio_value(self, positions, cash, date):
        """Calculate total portfolio value"""
        portfolio_value = cash
        
        for gvkey, shares in positions.items():
            price = self._get_price_for_date(gvkey, date)
            if price is not None:
                portfolio_value += shares * price
        
        return portfolio_value
    
    def build_portfolio_daily_returns(self, weights_df, cost=TRANSACTION_COST, price_wide=None):
        """
        Build daily returns of a rebalanced long-only portfolio using adjusted prices.
        Based on FinRL-Trading-master/fundamental_back_testing.py
        
        Args:
            weights_df: DataFrame with columns [gvkey, trade_date, weights]
            cost: Transaction cost as fraction (default 0.1%)
            price_wide: Optional filtered price data. If None, uses self.price_wide
            
        Returns:
            pd.Series: daily return series (DatetimeIndex)
        """
        # Use provided price_wide if available, otherwise use full price_wide
        price_data = price_wide if price_wide is not None else self.price_wide
        
        # Daily simple returns per asset
        asset_ret = price_data.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Sort trade dates
        tdates = sorted(weights_df['trade_date'].unique())
        if len(tdates) == 0:
            raise ValueError("No trade_date in weights_df")
        
        # Prepare series
        port_ret = pd.Series(index=asset_ret.index, dtype=float)
        
        prev_w = None
        for i, t in enumerate(tdates):
            # Current target weights at date t
            w_slice = weights_df[weights_df['trade_date'] == t][['gvkey', 'weights']].set_index('gvkey')['weights']
            # Reindex to all assets present
            w = w_slice.reindex(asset_ret.columns).fillna(0.0).astype(float)
            
            # Determine the span: from t to next_t (exclusive), or to end
            # Use 'pad' (forward fill) to find the first date >= trade_date, or 'nearest' as fallback
            try:
                # Try to find exact match or first date >= trade_date
                start_idx = asset_ret.index.get_indexer([t], method='pad')[0]
                if start_idx < 0:
                    # Fallback to nearest if pad fails
                    start_idx = asset_ret.index.get_indexer([t], method='nearest')[0]
            except:
                # Fallback to nearest if any error
                start_idx = asset_ret.index.get_indexer([t], method='nearest')[0]
            
            start_date = asset_ret.index[start_idx]
            if i < len(tdates) - 1:
                end_date = tdates[i+1]
            else:
                end_date = asset_ret.index[-1]
            # Slice dates in (start_date, end_date]
            mask = (asset_ret.index > start_date) & (asset_ret.index <= end_date)
            sub_ret = asset_ret.loc[mask]
            
            # Turnover and cost applied on START (rebalance) date's return
            if prev_w is None:
                turnover = w.abs().sum()
            else:
                turnover = (w - prev_w).abs().sum()
            day_cost = cost * turnover  # proportional
            
            # For the first day after rebalance, deduct cost from portfolio return
            first = True
            for dt, row in sub_ret.iterrows():
                r = float(np.nansum(w.values * row.values))
                if first:
                    r = r - day_cost
                    first = False
                port_ret.loc[dt] = r
            
            prev_w = w
        
        port_ret = port_ret.dropna()
        return port_ret
    
    def run_backtest(self, 
                     start_date='2024-12-01',
                     end_date='2025-11-30',
                     initial_capital=1000000,
                     transaction_cost=TRANSACTION_COST,
                     sp500_data=None,
                     qqq_data=None):
        """
        Run backtest with quarterly rebalancing
        Based on FinRL-Trading-master/fundamental_back_testing.py
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital
            transaction_cost: Transaction cost as fraction (default 0.1%)
            sp500_data: Optional DataFrame with SP500 daily price data (columns: date, close)
            qqq_data: Optional DataFrame with QQQ daily price data (columns: date, close)
            
        Returns:
            dict with backtest results including benchmark metrics
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter price data to backtest period
        price_wide_filtered = self.price_wide.loc[
            (self.price_wide.index >= start_date) & 
            (self.price_wide.index <= end_date)
        ].copy()
        
        if len(price_wide_filtered) == 0:
            raise ValueError(f"No price data found in period {start_date} to {end_date}")
        
        # Filter weights to trade dates in backtest period
        trade_dates_in_period = [td for td in self.trade_dates 
                                if start_date <= pd.to_datetime(td) <= end_date]
        
        if len(trade_dates_in_period) == 0:
            raise ValueError(f"No trade dates found in period {start_date} to {end_date}")
        
        print(f"Running backtest from {start_date.date()} to {end_date.date()}")
        print(f"Found {len(trade_dates_in_period)} rebalancing dates")
        
        # Prepare weights DataFrame
        weights_df = self._prepare_weights()
        weights_df = weights_df[weights_df['trade_date'].isin(trade_dates_in_period)].copy()
        
        if len(weights_df) == 0:
            raise ValueError("No weights found for trade dates in backtest period")
        
        # Build portfolio daily returns using filtered price data
        print("Building portfolio daily returns...")
        daily_returns = self.build_portfolio_daily_returns(
            weights_df, 
            cost=transaction_cost,
            price_wide=price_wide_filtered  # Use filtered price data for consistency
        )
        
        # Filter to backtest period
        daily_returns = daily_returns.loc[
            (daily_returns.index >= start_date) & 
            (daily_returns.index <= end_date)
        ]
        
        # Calculate portfolio equity (cumulative value)
        equity = (1.0 + daily_returns).cumprod() * initial_capital
        
        # Store portfolio values
        self.portfolio_values = equity.to_frame('value')
        self.portfolio_values['returns'] = daily_returns
        self.portfolio_values['cumulative_returns'] = equity / initial_capital - 1.0
        
        # Prepare benchmark data
        sp500_returns = None
        qqq_returns = None
        sp500_equity = None
        qqq_equity = None
        
        if sp500_data is not None:
            sp500_returns, sp500_equity = self._prepare_benchmark_data(
                sp500_data, start_date, end_date, initial_capital
            )
            if sp500_returns is not None:
                print(f"  ✓ SP500: {len(sp500_returns)} daily records")
        
        if qqq_data is not None:
            qqq_returns, qqq_equity = self._prepare_benchmark_data(
                qqq_data, start_date, end_date, initial_capital
            )
            if qqq_returns is not None:
                print(f"  ✓ QQQ: {len(qqq_returns)} daily records")
        
        # Calculate metrics
        results = self._calculate_metrics(
            initial_capital, 
            daily_returns, 
            equity,
            sp500_returns=sp500_returns,
            sp500_equity=sp500_equity,
            qqq_returns=qqq_returns,
            qqq_equity=qqq_equity
        )
        
        return results
    
    def _prepare_benchmark_data(self, benchmark_data, start_date, end_date, initial_capital):
        """
        Prepare benchmark data (SP500 or QQQ) for metrics calculation
        
        Args:
            benchmark_data: DataFrame with benchmark data (columns: date, close)
            start_date: Start date for filtering
            end_date: End date for filtering
            initial_capital: Initial capital for normalization
            
        Returns:
            Tuple of (daily_returns, equity) Series or (None, None) if data unavailable
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(benchmark_data, pd.Series):
                benchmark_df = benchmark_data.to_frame('close')
                benchmark_df.index.name = 'date'
                benchmark_df = benchmark_df.reset_index()
            else:
                benchmark_df = benchmark_data.copy()
            
            # Ensure date column exists and is datetime
            if 'date' in benchmark_df.columns:
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df = benchmark_df.set_index('date')
            elif benchmark_df.index.name == 'date' or isinstance(benchmark_df.index, pd.DatetimeIndex):
                benchmark_df.index = pd.to_datetime(benchmark_df.index)
            
            # Get close price column
            if 'close' in benchmark_df.columns:
                close_prices = benchmark_df['close']
            elif 'value' in benchmark_df.columns:
                close_prices = benchmark_df['value']
            elif len(benchmark_df.columns) == 1:
                close_prices = benchmark_df.iloc[:, 0]
            else:
                return None, None
            
            # Filter to backtest period
            # Use actual data end date if it's earlier than requested end_date
            # Don't artificially limit to 2025-09-30 - use actual data range
            effective_end_date = end_date
            
            # Check actual available data range and use the minimum
            if len(close_prices) > 0:
                actual_max_date = close_prices.index.max()
                if actual_max_date < effective_end_date:
                    effective_end_date = actual_max_date
            
            close_prices = close_prices.loc[
                (close_prices.index >= start_date) & 
                (close_prices.index <= effective_end_date)
            ]
            
            if len(close_prices) == 0:
                return None, None
            
            # Sort by date
            close_prices = close_prices.sort_index()
            
            # Calculate daily returns
            daily_returns = close_prices.pct_change().dropna()
            
            # Calculate equity (normalized to initial capital)
            # Start from first available date's price
            start_price = close_prices.iloc[0]
            equity = (close_prices / start_price) * initial_capital
            
            # Align returns with equity index (returns start from second day)
            equity = equity.loc[daily_returns.index]
            
            return daily_returns, equity
            
        except Exception as e:
            print(f"  Warning: Failed to prepare benchmark data: {e}")
            return None, None
    
    def _calculate_benchmark_metrics(self, initial_capital, daily_returns, equity):
        """
        Calculate performance metrics for a benchmark
        
        Args:
            initial_capital: Initial capital
            daily_returns: Daily returns Series
            equity: Equity Series
            
        Returns:
            dict with benchmark metrics or None if data unavailable
        """
        if daily_returns is None or equity is None or len(daily_returns) == 0:
            return None
        
        returns = daily_returns.dropna()
        if len(returns) == 0:
            return None
        
        # Total return
        total_return = (equity.iloc[-1] / initial_capital - 1) * 100
        
        # Annualized return
        days = (returns.index[-1] - returns.index[0]).days
        years = days / 252.0
        if years > 0:
            annualized_return = returns.mean() * 252 * 100
        else:
            annualized_return = 0
        
        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate = 0.02)
        rf_annual = 0.02
        sharpe_ratio = (annualized_return / 100 - rf_annual) / (annualized_vol / 100) if annualized_vol > 0 else 0
        
        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio
        calmar_ratio = (annualized_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'final_value': equity.iloc[-1],
            'daily_returns': returns,
            'equity': equity,
            'drawdown': drawdown
        }
    
    def _calculate_metrics(self, initial_capital, daily_returns, equity,
                          sp500_returns=None, sp500_equity=None,
                          qqq_returns=None, qqq_equity=None):
        """Calculate performance metrics based on FinRL-Trading approach"""
        if len(daily_returns) == 0:
            return {}
        
        returns = daily_returns.dropna()
        if len(returns) == 0:
            return {}
        
        # Portfolio metrics
        # Total return
        total_return = (equity.iloc[-1] / initial_capital - 1) * 100
        
        # Annualized return
        days = (returns.index[-1] - returns.index[0]).days
        years = days / 252.0
        if years > 0:
            annualized_return = returns.mean() * 252 * 100
        else:
            annualized_return = 0
        
        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate = 0.02)
        rf_annual = 0.02
        sharpe_ratio = (annualized_return / 100 - rf_annual) / (annualized_vol / 100) if annualized_vol > 0 else 0
        
        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio
        calmar_ratio = (annualized_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # Calculate benchmark metrics
        sp500_metrics = None
        qqq_metrics = None
        
        if sp500_returns is not None and sp500_equity is not None:
            sp500_metrics = self._calculate_benchmark_metrics(
                initial_capital, sp500_returns, sp500_equity
            )
        
        if qqq_returns is not None and qqq_equity is not None:
            qqq_metrics = self._calculate_benchmark_metrics(
                initial_capital, qqq_returns, qqq_equity
            )
        
        result = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'final_value': equity.iloc[-1],
            'initial_capital': initial_capital,
            'portfolio_values': self.portfolio_values,
            'daily_returns': returns,
            'drawdown': drawdown
        }
        
        # Add benchmark metrics
        if sp500_metrics is not None:
            result['sp500_metrics'] = sp500_metrics
            # Add SP500 equity to portfolio values for plotting
            if 'sp500_equity' not in self.portfolio_values.columns:
                # Align SP500 equity with portfolio dates
                sp500_aligned = sp500_equity.reindex(
                    self.portfolio_values.index, 
                    method='ffill'
                )
                self.portfolio_values['sp500_value'] = sp500_aligned
        
        if qqq_metrics is not None:
            result['qqq_metrics'] = qqq_metrics
            # Add QQQ equity to portfolio values for plotting
            if 'qqq_equity' not in self.portfolio_values.columns:
                # Align QQQ equity with portfolio dates
                qqq_aligned = qqq_equity.reindex(
                    self.portfolio_values.index, 
                    method='ffill'
                )
                self.portfolio_values['qqq_value'] = qqq_aligned
        
        return result
    
    def plot_portfolio_value(self, output_path='portfolio_value_chart.png', 
                             sp500_data=None, qqq_data=None):
        """
        Plot portfolio value over time with drawdown subplot
        Based on FinRL-Trading visualization style
        Shows daily portfolio value with SP500 and QQQ benchmarks
        
        Args:
            output_path: Path to save the chart
            sp500_data: Optional DataFrame with SP500 data (columns: date, close or value)
            qqq_data: Optional DataFrame with QQQ data (columns: date, close or value)
        """
        if len(self.portfolio_values) == 0:
            print("No portfolio values to plot")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
        
        # Get portfolio start value for normalization
        portfolio_start = self.portfolio_values['value'].iloc[0]
        portfolio_dates = pd.to_datetime(self.portfolio_values.index)
        
        # Plot 1: Portfolio value (daily)
        ax1.plot(portfolio_dates, 
                self.portfolio_values['value'], 
                label='Portfolio (Dual-Insight Trader)', 
                linewidth=2.5,
                color='#1f77b4')
        
        # Plot SP500 benchmark if provided
        # First check if already in portfolio_values (from backtest)
        if 'sp500_value' in self.portfolio_values.columns:
            sp500_values = self.portfolio_values['sp500_value'].dropna()
            if len(sp500_values) > 0:
                # Use actual data range (no artificial limit)
                ax1.plot(sp500_values.index,
                        sp500_values.values,
                        label='SP500',
                        linewidth=2,
                        color='#2ca02c',
                        linestyle='--',
                        alpha=0.8)
        elif sp500_data is not None:
            sp500_df = sp500_data.copy()
            # Handle both DataFrame with date column and Series/DataFrame with datetime index
            if 'date' in sp500_df.columns:
                sp500_df['date'] = pd.to_datetime(sp500_df['date'])
                sp500_df = sp500_df.set_index('date')
            elif isinstance(sp500_df.index, pd.DatetimeIndex):
                pass  # Already has datetime index
            elif sp500_df.index.name == 'date':
                sp500_df.index = pd.to_datetime(sp500_df.index)
            
            # Get value column
            if 'close' in sp500_df.columns:
                sp500_values = sp500_df['close']
            elif 'value' in sp500_df.columns:
                sp500_values = sp500_df['value']
            else:
                sp500_values = None
            
            if sp500_values is not None:
                # Filter to backtest period and actual available data
                end_date = portfolio_dates.max()
                
                sp500_filtered = sp500_values.loc[
                    (sp500_values.index >= portfolio_dates.min()) & 
                    (sp500_values.index <= end_date)
                ]
                
                # Use actual data range (don't artificially limit)
                if len(sp500_filtered) > 0:
                    actual_max_date = sp500_filtered.index.max()
                    # Only filter if actual data ends before portfolio end date
                    if actual_max_date < end_date:
                        sp500_filtered = sp500_values.loc[
                            (sp500_values.index >= portfolio_dates.min()) & 
                            (sp500_values.index <= actual_max_date)
                        ]
                    
                    if len(sp500_filtered) > 0:
                        # Normalize SP500 to start at same value as portfolio
                        sp500_start = sp500_filtered.iloc[0]
                        if sp500_start > 0:
                            sp500_normalized = sp500_filtered * (portfolio_start / sp500_start)
                            
                            ax1.plot(sp500_filtered.index,
                                    sp500_normalized,
                                    label='SP500',
                                    linewidth=2,
                                    color='#2ca02c',
                                    linestyle='--',
                                    alpha=0.8)
        
        # Plot QQQ benchmark if provided
        # First check if already in portfolio_values (from backtest)
        if 'qqq_value' in self.portfolio_values.columns:
            qqq_values = self.portfolio_values['qqq_value'].dropna()
            if len(qqq_values) > 0:
                # Use actual data range (no artificial limit)
                ax1.plot(qqq_values.index,
                        qqq_values.values,
                        label='QQQ',
                        linewidth=2,
                        color='#ff7f0e',
                        linestyle='--',
                        alpha=0.8)
        elif qqq_data is not None:
            qqq_df = qqq_data.copy()
            # Handle both DataFrame with date column and Series/DataFrame with datetime index
            if 'date' in qqq_df.columns:
                qqq_df['date'] = pd.to_datetime(qqq_df['date'])
                qqq_df = qqq_df.set_index('date')
            elif isinstance(qqq_df.index, pd.DatetimeIndex):
                pass  # Already has datetime index
            elif qqq_df.index.name == 'date':
                qqq_df.index = pd.to_datetime(qqq_df.index)
            
            # Get value column
            if 'close' in qqq_df.columns:
                qqq_values = qqq_df['close']
            elif 'value' in qqq_df.columns:
                qqq_values = qqq_df['value']
            else:
                qqq_values = None
            
            if qqq_values is not None:
                # Filter to backtest period and actual available data
                end_date = portfolio_dates.max()
                
                qqq_filtered = qqq_values.loc[
                    (qqq_values.index >= portfolio_dates.min()) & 
                    (qqq_values.index <= end_date)
                ]
                
                # Use actual data range (don't artificially limit)
                if len(qqq_filtered) > 0:
                    actual_max_date = qqq_filtered.index.max()
                    # Only filter if actual data ends before portfolio end date
                    if actual_max_date < end_date:
                        qqq_filtered = qqq_values.loc[
                            (qqq_values.index >= portfolio_dates.min()) & 
                            (qqq_values.index <= actual_max_date)
                        ]
                    
                    if len(qqq_filtered) > 0:
                        # Normalize QQQ to start at same value as portfolio
                        qqq_start = qqq_filtered.iloc[0]
                        if qqq_start > 0:
                            qqq_normalized = qqq_filtered * (portfolio_start / qqq_start)
                            
                            ax1.plot(qqq_filtered.index,
                                    qqq_normalized,
                                    label='QQQ',
                                    linewidth=2,
                                    color='#ff7f0e',
                                    linestyle='--',
                                    alpha=0.8)
        
        # Mark rebalancing dates (quarterly)
        for td in self.trade_dates:
            td_dt = pd.to_datetime(td)
            if td_dt >= portfolio_dates.min() and td_dt <= portfolio_dates.max():
                ax1.axvline(x=td_dt, color='red', linestyle=':', alpha=0.4, linewidth=1, zorder=0)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Portfolio Value Comparison (Daily)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Drawdown (daily)
        if 'drawdown' in self.portfolio_values.columns:
            drawdown_series = self.portfolio_values['drawdown'] * 100
        else:
            # Calculate drawdown if not already in portfolio_values
            equity = self.portfolio_values['value']
            rolling_max = equity.expanding().max()
            drawdown_series = (equity - rolling_max) / rolling_max * 100
        
        ax2.fill_between(portfolio_dates, drawdown_series, 0, 
                        color='red', alpha=0.3)
        ax2.plot(portfolio_dates, drawdown_series, 
                color='darkred', linewidth=1.5)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_title('Portfolio Drawdown (Daily)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio value chart saved to: {output_path}")
        plt.close()

