#!/usr/bin/env python3
"""
Dual-Insight Trader - Main Program
Integrates FinRL (stock selection) and FinGPT (signal generation) for trading strategy
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from news_fetcher.data_fetcher import NewsDataFetcher
from news_fetcher.prompt_builder import PromptBuilder
from signal_generator.sentiment_agent import SentimentAgent
from signal_generator.technical_agent import TechnicalAgent
from portfolio_manager.portfolio_manager import PortfolioManager
from backtester.backtester import DualInsightBacktester


def load_price_data(data_dir, price_file_path=None):
    """
    Load price data from processed data or raw CSV
    
    Args:
        data_dir: Directory with processed data
        price_file_path: Path to raw price CSV (if data_dir doesn't have processed data)
        
    Returns:
        DataFrame with price data
    """
    # Try to load from processed data first
    processed_file = os.path.join(data_dir, "final_ratios.csv")
    
    if os.path.exists(processed_file):
        print(f"Loading processed data from: {processed_file}")
        df = pd.read_csv(processed_file)
        # Convert date column
        df['datadate'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        return df
    
    # Otherwise load from raw price file
    if price_file_path and os.path.exists(price_file_path):
        print(f"Loading raw price data from: {price_file_path}")
        df = pd.read_csv(price_file_path, usecols=['gvkey', 'tic', 'datadate', 'prccd', 'ajexdi'])
        return df
    
    raise FileNotFoundError(f"Could not find price data in {data_dir} or {price_file_path}")


def load_selected_stocks(data_dir):
    """
    Load selected stocks from stock selection results
    
    Args:
        data_dir: Directory with stock selection results
        
    Returns:
        DataFrame with selected stocks
    """
    stock_selected_file = os.path.join(data_dir, "stock_selected.csv")
    
    if os.path.exists(stock_selected_file):
        print(f"Loading selected stocks from: {stock_selected_file}")
        return pd.read_csv(stock_selected_file)
    else:
        print(f"Warning: stock_selected.csv not found in {data_dir}")
        print("Will use all stocks from price data")
        return None


def get_symbols_from_gvkeys(price_data, selected_stocks, trade_date):
    """
    Convert gvkeys to ticker symbols for a given trade date
    
    Args:
        price_data: DataFrame with price data
        selected_stocks: DataFrame with selected stocks (gvkey, trade_date)
        trade_date: Trade date string
        
    Returns:
        List of ticker symbols
    """
    if selected_stocks is None:
        # Use all symbols
        return price_data['tic'].unique().tolist()
    
    # Filter selected stocks for this date
    date_selected = selected_stocks[selected_stocks['trade_date'] == trade_date]
    
    if len(date_selected) == 0:
        return []
    
    # Map gvkey to tic
    gvkeys = date_selected['gvkey'].unique()
    symbols = price_data[price_data['gvkey'].isin(gvkeys)]['tic'].unique()
    
    return symbols.tolist()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description='Dual-Insight Trader - Integrated FinRL and FinGPT Trading Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with processed data
  python main.py --start_date "2024-12-01" --end_date "2025-11-30" --data_dir "data_processor/outputs"
  
  # Run with raw price data
  python main.py --start_date "2024-12-01" --end_date "2025-11-30" \\
                 --data_dir "data_processor/outputs" \\
                 --price_file "../dec data/sp500_tickers_daily_price.csv"
        """
    )
    
    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory with processed data (final_ratios.csv) or stock selection results'
    )
    
    parser.add_argument(
        '--price_file',
        type=str,
        default=None,
        help='Path to raw price CSV file (if data_dir doesn\'t have processed data)'
    )
    
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=1000000,
        help='Initial capital for backtesting (default: 1000000)'
    )
    
    parser.add_argument(
        '--fusion_strategy',
        type=str,
        default='weighted',
        choices=['weighted', 'consensus', 'majority', 'adaptive'],
        help='Signal fusion strategy (default: weighted)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--test_mode',
        action='store_true',
        help='Enable test mode (limited stocks and dates for quick testing)'
    )
    
    parser.add_argument(
        '--max_stocks',
        type=int,
        default=None,
        help='Maximum number of stocks to process (for testing, default: None = all)'
    )
    
    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Dual-Insight Trader")
    print("=" * 80)
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Fusion Strategy: {args.fusion_strategy}")
    print("-" * 80)
    
    try:
        # Load data
        print("\n[1/5] Loading data...")
        price_data = load_price_data(args.data_dir, args.price_file)
        selected_stocks = load_selected_stocks(args.data_dir)
        print(f"  Loaded {len(price_data)} price records")
        print(f"  Unique symbols: {len(price_data['tic'].unique())}")
        
        # Test mode: Limit date range and stocks
        if args.test_mode:
            print("\n⚠️  TEST MODE ENABLED")
            # Limit to 1 month of data
            start_dt = pd.to_datetime(args.start_date)
            end_dt = start_dt + pd.DateOffset(months=1)
            args.end_date = end_dt.strftime('%Y-%m-%d')
            print(f"  Limited date range: {args.start_date} to {args.end_date}")
            
            # Limit stocks if not specified
            if args.max_stocks is None:
                args.max_stocks = 5
                print(f"  Limited to {args.max_stocks} stocks (default for test mode)")
        
        # Initialize components
        print("\n[2/5] Initializing components...")
        
        # News fetcher
        news_fetcher = NewsDataFetcher()
        print("  ✓ News fetcher initialized")
        
        # Sentiment agent (FinGPT)
        print("  Loading FinGPT model (this may take a while)...")
        sentiment_agent = SentimentAgent()
        print("  ✓ Sentiment agent initialized")
        
        # Technical agent
        technical_agent = TechnicalAgent()
        print("  ✓ Technical agent initialized")
        
        # Portfolio manager
        portfolio_manager = PortfolioManager(fusion_strategy=args.fusion_strategy)
        print("  ✓ Portfolio manager initialized")
        
        # Backtester
        backtester = DualInsightBacktester(
            price_data=price_data,
            news_fetcher=news_fetcher,
            sentiment_agent=sentiment_agent,
            technical_agent=technical_agent,
            portfolio_manager=portfolio_manager
        )
        print("  ✓ Backtester initialized")
        
        # Run backtest
        print("\n[3/5] Running backtest...")
        
        # Get symbols for test mode
        symbols = None
        if args.test_mode and args.max_stocks:
            all_symbols = price_data['tic'].unique().tolist()
            symbols = all_symbols[:args.max_stocks]
            print(f"  Test mode: Processing {len(symbols)} stocks")
        
        results = backtester.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            symbols=symbols,
            test_mode=args.test_mode,
            max_stocks=args.max_stocks,
            show_progress=not args.no_progress
        )
        
        # Save results
        print("\n[4/5] Saving results...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save portfolio values
        if 'portfolio_values' in results:
            portfolio_file = os.path.join(args.output_dir, 'portfolio_values.csv')
            results['portfolio_values'].to_csv(portfolio_file)
            print(f"  ✓ Portfolio values saved to: {portfolio_file}")
        
        # Save trades
        if len(backtester.trades) > 0:
            trades_df = pd.DataFrame(backtester.trades)
            trades_file = os.path.join(args.output_dir, 'trades.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"  ✓ Trades saved to: {trades_file}")
        
        # Save metrics
        metrics_file = os.path.join(args.output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("Dual-Insight Trader - Backtest Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Period: {args.start_date} to {args.end_date}\n")
            f.write(f"Initial Capital: ${args.initial_capital:,.2f}\n")
            f.write(f"Fusion Strategy: {args.fusion_strategy}\n\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Return: {results.get('total_return', 0):.2f}%\n")
            f.write(f"Annualized Return: {results.get('annualized_return', 0):.2f}%\n")
            f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}\n")
            f.write(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%\n")
            f.write(f"Final Value: ${results.get('final_value', 0):,.2f}\n")
            f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
        print(f"  ✓ Metrics saved to: {metrics_file}")
        
        # Display results
        print("\n[5/5] Backtest Results:")
        print("=" * 80)
        print(f"Total Return: {results.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {results.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

