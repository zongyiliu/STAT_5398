"""
Data fetcher module - adapted from FinGPT_Forecaster
Fetches news data and stock data for signal generation
"""

import os
import finnhub
import yfinance as yf
import pandas as pd
from datetime import date, datetime, timedelta
from collections import defaultdict
import json
import time
import numpy as np

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


class NewsDataFetcher:
    """
    Fetches news and stock data for FinGPT signal generation
    Adapted from FinGPT_Forecaster/data_infererence_fetch.py
    """
    
    def __init__(self, finnhub_key=None, price_data_path=None):
        """
        Initialize the news data fetcher
        
        Args:
            finnhub_key: Finnhub API key (if None, uses FINNHUB_KEY or FINNHUB_API_KEY env var)
            price_data_path: Optional path to local price data CSV file (sp500_tickers_daily_price.csv)
                           If provided, will use local data instead of yfinance API
        """
        # Support both FINNHUB_KEY and FINNHUB_API_KEY environment variables
        api_key = finnhub_key or os.environ.get("FINNHUB_KEY") or os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError(
                "Finnhub API key is required. Set FINNHUB_KEY or FINNHUB_API_KEY environment variable."
            )
        
        self.finnhub_client = finnhub.Client(api_key=api_key)
        
        # Pre-load price data if path is provided
        self.price_data = None
        self.gvkey_to_ticker_map = None
        if price_data_path:
            print(f"Loading price data from {price_data_path} for local price lookup...")
            try:
                # Load price data with necessary columns
                self.price_data = pd.read_csv(
                    price_data_path,
                    usecols=['gvkey', 'tic', 'datadate', 'prccd'],
                    dtype={'gvkey': 'Int64', 'tic': 'str', 'prccd': 'float64'},
                    parse_dates=['datadate']
                )
                # Build gvkey to ticker mapping
                self.gvkey_to_ticker_map = self.price_data.groupby('gvkey')['tic'].last().to_dict()
                print(f"  Loaded {len(self.price_data)} price records for {len(self.gvkey_to_ticker_map)} stocks")
            except Exception as e:
                print(f"  Warning: Failed to load price data from {price_data_path}: {e}")
                print(f"  Will fall back to yfinance API")
                self.price_data = None
                self.gvkey_to_ticker_map = None
    
    def get_curday(self):
        """Get current date as string"""
        return date.today().strftime("%Y-%m-%d")
    
    def n_weeks_before(self, date_string, n):
        """Get date n weeks before given date"""
        date_obj = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
        return date_obj.strftime("%Y-%m-%d")
    
    def get_stock_data(self, stock_symbol, steps):
        """
        Get stock price data for given dates
        
        Args:
            stock_symbol: Stock ticker symbol
            steps: List of date strings
            
        Returns:
            DataFrame with Start Date, End Date, Start Price, End Price
        """
        # Try to use local price data first if available
        if self.price_data is not None:
            return self._get_stock_data_from_local(stock_symbol, steps)
        else:
            return self._get_stock_data_from_api(stock_symbol, steps)
    
    def _get_stock_data_from_local(self, stock_symbol, steps):
        """
        Get stock price data from local CSV file
        
        Args:
            stock_symbol: Stock ticker symbol
            steps: List of date strings
            
        Returns:
            DataFrame with Start Date, End Date, Start Price, End Price
        """
        # Find gvkey for this ticker
        gvkey = None
        for gv, ticker in self.gvkey_to_ticker_map.items():
            if ticker == stock_symbol:
                gvkey = gv
                break
        
        if gvkey is None:
            # Ticker not found in local data, fall back to API
            print(f"  Warning: Ticker {stock_symbol} not found in local price data, using yfinance API")
            return self._get_stock_data_from_api(stock_symbol, steps)
        
        # Filter price data for this gvkey
        stock_price_data = self.price_data[self.price_data['gvkey'] == gvkey].copy()
        
        if len(stock_price_data) == 0:
            # No data for this gvkey, fall back to API
            print(f"  Warning: No price data found for {stock_symbol} (gvkey: {gvkey}), using yfinance API")
            return self._get_stock_data_from_api(stock_symbol, steps)
        
        # Sort by date
        stock_price_data = stock_price_data.sort_values('datadate')
        
        # Convert steps to datetime for comparison
        step_dates = [pd.to_datetime(s) for s in steps]
        start_date = step_dates[0]
        end_date = step_dates[-1]
        
        # Filter data within date range
        date_mask = (stock_price_data['datadate'] >= start_date) & (stock_price_data['datadate'] <= end_date)
        filtered_data = stock_price_data[date_mask].copy()
        
        if len(filtered_data) == 0:
            # No data in date range, fall back to API
            print(f"  Warning: No price data in date range for {stock_symbol}, using yfinance API")
            return self._get_stock_data_from_api(stock_symbol, steps)
        
        # Find closest dates for each step
        dates, prices = [], []
        available_dates = filtered_data['datadate'].tolist()
        available_prices = filtered_data['prccd'].tolist()
        
        for step_date in step_dates[:-1]:
            # Find the closest date >= step_date
            found = False
            for i, avail_date in enumerate(available_dates):
                if avail_date >= step_date:
                    dates.append(avail_date)
                    prices.append(float(available_prices[i]))
                    found = True
                    break
            
            if not found:
                # Use the last available date
                if len(available_dates) > 0:
                    dates.append(available_dates[-1])
                    prices.append(float(available_prices[-1]))
        
        # Add the last date and price
        if len(available_dates) > 0:
            dates.append(available_dates[-1])
            prices.append(float(available_prices[-1]))
        
        if len(dates) < 2 or len(prices) < 2:
            # Insufficient data, fall back to API
            print(f"  Warning: Insufficient local data for {stock_symbol}, using yfinance API")
            return self._get_stock_data_from_api(stock_symbol, steps)
        
        return pd.DataFrame({
            "Start Date": dates[:-1], 
            "End Date": dates[1:],
            "Start Price": prices[:-1], 
            "End Price": prices[1:]
        })
    
    def _get_stock_data_from_api(self, stock_symbol, steps):
        """
        Get stock price data from yfinance API (original method)
        
        Args:
            stock_symbol: Stock ticker symbol
            steps: List of date strings
            
        Returns:
            DataFrame with Start Date, End Date, Start Price, End Price
        """
        # Use yf.Ticker().history() - more reliable for single stocks
        # This method returns a DataFrame with standard columns: Open, High, Low, Close, Volume
        try:
            ticker = yf.Ticker(stock_symbol)
            # Convert date strings to datetime if needed
            start_date = pd.to_datetime(steps[0])
            end_date = pd.to_datetime(steps[-1])
            # Add one day to end_date to include the last day
            end_date = end_date + pd.Timedelta(days=1)
            
            stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            # If history() returns empty, try download() as fallback
            if len(stock_data) == 0:
                stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
                # Handle MultiIndex if download() returns it
                if isinstance(stock_data.columns, pd.MultiIndex):
                    first_symbol = stock_data.columns.levels[0][0]
                    stock_data = stock_data.xs(first_symbol, axis=1, level=0)
        except Exception as e:
            # Fallback to download() if Ticker fails
            try:
                start_date = pd.to_datetime(steps[0])
                end_date = pd.to_datetime(steps[-1]) + pd.Timedelta(days=1)
                stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
                # Handle MultiIndex
                if isinstance(stock_data.columns, pd.MultiIndex):
                    first_symbol = stock_data.columns.levels[0][0]
                    stock_data = stock_data.xs(first_symbol, axis=1, level=0)
            except Exception as e2:
                # Last resort: try old format (positional arguments)
                try:
                    stock_data = yf.download(stock_symbol, steps[0], steps[-1], progress=False)
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        first_symbol = stock_data.columns.levels[0][0]
                        stock_data = stock_data.xs(first_symbol, axis=1, level=0)
                except Exception as e3:
                    raise ValueError(
                        f"Failed to download stock data for {stock_symbol} using all methods. "
                        f"Errors: Ticker.history()={e}, download()={e2}, download(old)={e3}"
                    )
        
        if len(stock_data) == 0:
            raise ValueError(f"Failed to download stock price data for symbol {stock_symbol}")
        
        # Debug: print column structure
        # print(f"DEBUG: Columns for {stock_symbol}: {stock_data.columns}")
        # print(f"DEBUG: Column type: {type(stock_data.columns)}")
        # print(f"DEBUG: DataFrame shape: {stock_data.shape}")
        
        # Handle case where yfinance returns Series instead of DataFrame
        if isinstance(stock_data, pd.Series):
            # Convert Series to DataFrame
            stock_data = stock_data.to_frame(name='Close')
        
        # Handle MultiIndex columns (when downloading multiple symbols)
        # If columns are MultiIndex, take the first level
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Get the first symbol from MultiIndex
            first_symbol = stock_data.columns.levels[0][0]
            stock_data = stock_data.xs(first_symbol, axis=1, level=0)
        
        # Check if columns are just the stock symbol (yfinance bug/feature)
        # In this case, the data might be transposed or in wrong format
        if len(stock_data.columns) == 1 and stock_symbol in str(stock_data.columns[0]):
            # Try to get data using Ticker.history() which is more reliable
            try:
                ticker = yf.Ticker(stock_symbol)
                stock_data = ticker.history(start=steps[0], end=steps[-1], auto_adjust=False)
                if len(stock_data) == 0:
                    raise ValueError(f"No data returned for {stock_symbol}")
            except Exception as e:
                raise ValueError(
                    f"yfinance returned unexpected format for {stock_symbol}. "
                    f"Columns: {list(stock_data.columns)}. Error: {e}"
                )
        
        # Normalize column names (handle case variations)
        # yfinance may return 'Close', 'close', or other variations
        column_mapping = {}
        for col in stock_data.columns:
            col_lower = str(col).lower()
            if col_lower == 'close':
                column_mapping[col] = 'Close'
            elif col_lower == 'adj close' or col_lower == 'adjclose':
                column_mapping[col] = 'Close'  # Use Adj Close as Close if available
            elif col_lower in ['open', 'high', 'low', 'volume']:
                column_mapping[col] = col  # Keep other columns as-is
        
        # Rename columns if needed
        if column_mapping:
            stock_data = stock_data.rename(columns=column_mapping)
        
        # Try to find Close column (case-insensitive)
        close_col = None
        for col in stock_data.columns:
            if str(col).lower() == 'close':
                close_col = col
                break
        
        # If no Close found, try Adj Close
        if close_col is None:
            for col in stock_data.columns:
                if 'adj' in str(col).lower() and 'close' in str(col).lower():
                    close_col = col
                    break
        
        if close_col is None:
            # Print available columns for debugging
            available_cols = list(stock_data.columns)
            raise ValueError(
                f"No 'Close' column found in stock data for {stock_symbol}. "
                f"Available columns: {available_cols}. "
                f"DataFrame shape: {stock_data.shape}. "
                f"Please check yfinance version and stock symbol validity."
            )
        
        dates, prices = [], []
        # Convert index to string format for comparison
        available_dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        
        # Get Close prices as a Series (not DataFrame)
        close_prices = stock_data[close_col]
        if isinstance(close_prices, pd.DataFrame):
            # If still a DataFrame, take first column
            close_prices = close_prices.iloc[:, 0]
        
        for date_str in steps[:-1]:
            found = False
            for i in range(len(stock_data)):
                date_val = available_dates[i]
                # Ensure we're comparing strings
                if isinstance(date_val, pd.Series):
                    date_val = date_val.iloc[0] if len(date_val) > 0 else str(date_val)
                if str(date_val) >= str(date_str):
                    # Use iloc for integer-based indexing
                    price_val = close_prices.iloc[i]
                    # Ensure price is a scalar value
                    if isinstance(price_val, pd.Series):
                        price_val = price_val.iloc[0] if len(price_val) > 0 else float(price_val)
                    prices.append(float(price_val))
                    dates.append(pd.to_datetime(date_val))
                    found = True
                    break
            
            if not found:
                # If no date found, use the closest available date
                if len(available_dates) > 0:
                    dates.append(pd.to_datetime(available_dates[-1]))
                    price_val = close_prices.iloc[-1]
                    if isinstance(price_val, pd.Series):
                        price_val = price_val.iloc[0] if len(price_val) > 0 else float(price_val)
                    prices.append(float(price_val))
        
        # Add the last date and price
        if len(available_dates) > 0:
            dates.append(pd.to_datetime(available_dates[-1]))
            price_val = close_prices.iloc[-1]
            if isinstance(price_val, pd.Series):
                price_val = price_val.iloc[0] if len(price_val) > 0 else float(price_val)
            prices.append(float(price_val))
        
        if len(dates) < 2 or len(prices) < 2:
            raise ValueError(f"Insufficient data for {stock_symbol}: got {len(dates)} dates, {len(prices)} prices")
        
        return pd.DataFrame({
            "Start Date": dates[:-1], 
            "End Date": dates[1:],
            "Start Price": prices[:-1], 
            "End Price": prices[1:]
        })
    
    def get_news(self, symbol, data):
        """
        Get news data for given symbol and date ranges
        
        Args:
            symbol: Stock ticker symbol
            data: DataFrame with Start Date and End Date columns
            
        Returns:
            DataFrame with News column added
        """
        news_list = []
        
        for idx, row in data.iterrows():
            start_date = row['Start Date'].strftime('%Y-%m-%d') if hasattr(row['Start Date'], 'strftime') else row['Start Date']
            end_date = row['End Date'].strftime('%Y-%m-%d') if hasattr(row['End Date'], 'strftime') else row['End Date']
            
            time.sleep(1)  # Rate limiting
            try:
                # Call Finnhub API
                weekly_news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
                
                # Handle empty news data (API returns empty list)
                if not weekly_news or len(weekly_news) == 0:
                    # No news available for this period - this is normal for some stocks/periods
                    weekly_news = []
                else:
                    # Process and format news data
                    weekly_news = [
                        {
                            "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                            "headline": n.get('headline', ''),
                            "summary": n.get('summary', ''),
                        } for n in weekly_news if 'datetime' in n
                    ]
                    weekly_news.sort(key=lambda x: x['date'])
                    
            except KeyError as e:
                # Missing expected fields in API response
                print(f"Warning: Missing fields in news data for {symbol} from {start_date} to {end_date}: {e}")
                weekly_news = []
            except Exception as e:
                # Other API errors (network, rate limit, invalid symbol, etc.)
                print(f"Warning: Failed to fetch news for {symbol} from {start_date} to {end_date}: {e}")
                weekly_news = []
            
            # Always append, even if empty (allows downstream processing to handle empty news)
            news_list.append(json.dumps(weekly_news))
        
        data['News'] = news_list
        return data
    
    def get_current_basics(self, symbol, curday):
        """
        Get current basic financials for a symbol
        
        Args:
            symbol: Stock ticker symbol
            curday: Current date string
            
        Returns:
            Dictionary with basic financials
        """
        try:
            basic_financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            
            basic_dict = defaultdict(dict)
            for metric, value_list in basic_financials['series']['quarterly'].items():
                for value in value_list:
                    basic_dict[value['period']].update({metric: value['v']})
            
            basic_list = []
            for k, v in basic_dict.items():
                v.update({'period': k})
                basic_list.append(v)
            
            basic_list.sort(key=lambda x: x['period'])
            
            for basic in basic_list[::-1]:
                if basic['period'] <= curday:
                    return basic
            
            return {}
        except Exception as e:
            print(f"Warning: Failed to fetch basics for {symbol}: {e}")
            return {}
    
    def fetch_all_data(self, symbol, curday, n_weeks=3):
        """
        Fetch all data (stock prices and news) for a symbol
        
        Args:
            symbol: Stock ticker symbol
            curday: Current date string
            n_weeks: Number of weeks to look back
            
        Returns:
            DataFrame with stock data and news
        """
        steps = [self.n_weeks_before(curday, i) for i in range(n_weeks+1)][::-1]
        
        data = self.get_stock_data(symbol, steps)
        data = self.get_news(symbol, data)
        
        return data
    
    def build_gvkey_to_ticker_map(self, price_data_path=None, price_data=None):
        """
        Build mapping from gvkey to ticker symbol
        
        Args:
            price_data_path: Path to price data CSV file
            price_data: DataFrame with price data (alternative to path)
            
        Returns:
            Dictionary mapping gvkey to ticker symbol
        """
        if price_data is None:
            if price_data_path is None:
                raise ValueError("Either price_data_path or price_data must be provided")
            
            # Read only necessary columns for efficiency
            print(f"Loading price data from {price_data_path}...")
            price_data = pd.read_csv(price_data_path, usecols=['gvkey', 'tic'], dtype={'gvkey': 'Int64', 'tic': 'str'})
        
        # Create mapping: use most recent ticker for each gvkey
        # Group by gvkey and take the last tic (assuming data is sorted by date)
        mapping = price_data.groupby('gvkey')['tic'].last().to_dict()
        
        print(f"Built gvkey to ticker mapping: {len(mapping)} stocks")
        return mapping
    
    def get_news_for_selected_stocks(self, selected_stocks_df, price_data_path=None, 
                                     price_data=None, trade_date=None, n_weeks=3):
        """
        Get news data for selected stocks from stock_selected.csv
        
        Args:
            selected_stocks_df: DataFrame with selected stocks (gvkey, trade_date, etc.)
            price_data_path: Path to price data CSV file
            price_data: DataFrame with price data (alternative to path)
            trade_date: Specific trade date to filter (if None, uses all dates)
            n_weeks: Number of weeks to look back for news
            
        Returns:
            Dictionary mapping (gvkey, trade_date) to news data DataFrame
        """
        # Build gvkey to ticker mapping
        gvkey_to_ticker = self.build_gvkey_to_ticker_map(price_data_path, price_data)
        
        # Filter by trade_date if specified
        if trade_date is not None:
            if isinstance(trade_date, str):
                trade_date = pd.to_datetime(trade_date)
            selected_stocks_df = selected_stocks_df[
                pd.to_datetime(selected_stocks_df['trade_date']) == trade_date
            ]
        
        # Convert trade_date to datetime if needed
        if 'trade_date' in selected_stocks_df.columns:
            selected_stocks_df['trade_date'] = pd.to_datetime(selected_stocks_df['trade_date'])
        
        # Group by trade_date and gvkey
        results = {}
        
        for (td, gvkey), group in selected_stocks_df.groupby(['trade_date', 'gvkey']):
            ticker = gvkey_to_ticker.get(gvkey)
            
            if ticker is None:
                print(f"Warning: No ticker found for gvkey {gvkey}, skipping...")
                continue
            
            # Convert trade_date to string for news fetching
            curday = td.strftime('%Y-%m-%d')
            
            try:
                print(f"Fetching news for {ticker} (gvkey: {gvkey}) on {curday}...")
                news_data = self.fetch_all_data(ticker, curday, n_weeks=n_weeks)
                
                # Check if we got any news data (even if some periods have no news)
                # This is still valid - we'll include it with empty news for some periods
                results[(gvkey, td)] = {
                    'ticker': ticker,
                    'gvkey': gvkey,
                    'trade_date': td,
                    'news_data': news_data
                }
            except ValueError as e:
                # Stock data download failed (invalid symbol, no data available)
                print(f"Warning: Stock data unavailable for {ticker} (gvkey: {gvkey}) on {curday}: {e}")
                # Optionally, you could still create an entry with empty data
                # For now, we skip it
                continue
            except Exception as e:
                # Other errors (API failures, network issues, etc.)
                print(f"Error fetching news for {ticker} (gvkey: {gvkey}) on {curday}: {e}")
                # Skip this stock and continue with others
                continue
        
        return results

