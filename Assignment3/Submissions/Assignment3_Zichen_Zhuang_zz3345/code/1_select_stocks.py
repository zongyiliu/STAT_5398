import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
import config

def load_assignment1_stocks():
    print("Stock Selection from Assignment 1")
    
    # Load stock selection results
    stock_file = config.STOCK_SELECTION_FILE
    print(f"Loading stock selection from: {stock_file}")
    
    if not stock_file.exists():
        print(f"ERROR: Stock selection file not found: {stock_file}")
        return None
    
    df = pd.read_csv(stock_file)
    print(f"Loaded {len(df)} stock selections from {df['trade_date'].min()} to {df['trade_date'].max()}")
    
    # Convert trade_date to datetime
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Filter for backtest period (or get latest available period)
    backtest_start = pd.to_datetime(config.BACKTEST_START)
    backtest_end = pd.to_datetime(config.BACKTEST_END)
    
    print(f"\nTarget backtest period: {backtest_start.date()} to {backtest_end.date()}")
    
    # Since Assignment 1 data may not extend to 2024-2025, 
    # we'll use the most recent period available as a proxy
    latest_date = df['trade_date'].max()
    print(f"Latest available trade date in Assignment 1 data: {latest_date.date()}")
    
    # Get stocks from the last 12 months of available data
    one_year_ago = latest_date - pd.DateOffset(months=12)
    recent_df = df[df['trade_date'] >= one_year_ago].copy()
    
    print(f"\nUsing last 12 months of data: {recent_df['trade_date'].min().date()} to {recent_df['trade_date'].max().date()}")
    print(f"Total selections in this period: {len(recent_df)}")
    
    # Get unique stocks (gvkey)
    unique_stocks = recent_df['gvkey'].unique()
    print(f"Unique stocks in selection: {len(unique_stocks)}")
    
    # For each rebalance period, select top N stocks
    stock_universe = []
    for trade_date in sorted(recent_df['trade_date'].unique()):
        period_stocks = recent_df[recent_df['trade_date'] == trade_date]
        # Sort by predicted return and select top N
        top_stocks = period_stocks.nlargest(config.TOP_N_STOCKS, 'predicted_return')
        stock_universe.append(top_stocks)
    
    universe_df = pd.concat(stock_universe, ignore_index=True)
    print(f"\nSelected {len(universe_df)} stock-period combinations (top {config.TOP_N_STOCKS} per period)")
    
    # Save universe
    output_file = config.DATA_DIR / "stock_universe.csv"
    universe_df.to_csv(output_file, index=False)
    print(f"Saved stock universe to: {output_file}")
    
    print(f"Total periods: {len(universe_df['trade_date'].unique())}")
    print(f"Stocks per period: {config.TOP_N_STOCKS}")
    print(f"Unique stocks: {len(universe_df['gvkey'].unique())}")
    print("\nTop 10 most selected stocks (gvkey):")
    print(universe_df['gvkey'].value_counts().head(10))
    
    # Map gvkey to ticker symbols (we need to add ticker mapping)
    print("Note: gvkey to ticker symbol mapping needed for news collection")
    return universe_df

def main():
    universe_df = load_assignment1_stocks()
    
    if universe_df is not None:
        print(f"Stock universe saved to: {config.DATA_DIR / 'stock_universe.csv'}")
        return True
    else:
        print("\nStep 1 failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
