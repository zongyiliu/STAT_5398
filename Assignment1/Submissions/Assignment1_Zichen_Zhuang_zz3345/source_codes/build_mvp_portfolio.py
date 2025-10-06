import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import os
import sys


def load_stock_selection(selection_file):
    """
    Load selected stocks from ML model output
    
    Args:
        selection_file (str): Path to stock_selected.csv
        
    Returns:
        pandas.DataFrame: Selected stocks with trade dates
    """
    print("Loading selected stocks...")
    stock_selected = pd.read_csv(selection_file)
    stock_selected["trade_date"] = pd.to_datetime(stock_selected["trade_date"])
    
    print(f"  Total records: {len(stock_selected)}")
    print(f"  Date range: {stock_selected['trade_date'].min()} to {stock_selected['trade_date'].max()}")
    print(f"  Unique stocks: {stock_selected['gvkey'].nunique()}")
    
    return stock_selected


def load_price_data(price_file, start_date='2018-01-01'):
    """
    Load daily price data from WRDS
    
    Args:
        price_file (str): Path to WRDS-Security Daily.csv
        start_date (str): Start date for filtering data
        
    Returns:
        pandas.DataFrame: Daily price data
    """
    print("Loading daily price data...")
    
    # Load only necessary columns
    usecols = ["gvkey", "tic", "datadate", "prccd", "ajexdi"]
    daily_price = pd.read_csv(price_file, usecols=usecols)
    
    daily_price["datadate"] = pd.to_datetime(daily_price["datadate"])
    daily_price = daily_price[daily_price["datadate"] >= start_date]
    
    # Calculate adjusted close price
    daily_price["ajexdi"] = daily_price["ajexdi"].replace(np.nan, 1)
    daily_price["adj_close"] = (daily_price["prccd"] / daily_price["ajexdi"]).astype(float)
    
    # Remove missing values
    daily_price.dropna(subset=["adj_close"], inplace=True)
    
    print(f"  Price data shape: {daily_price.shape}")
    print(f"  Date range: {daily_price['datadate'].min()} to {daily_price['datadate'].max()}")
    
    return daily_price


def calculate_returns(daily_price, stock_list, lookback_days=252):
    """
    Calculate historical returns for selected stocks
    
    Args:
        daily_price (pandas.DataFrame): Daily price data
        stock_list (list): List of stock identifiers (gvkey)
        lookback_days (int): Number of days to look back for return calculation
        
    Returns:
        pandas.DataFrame: Daily returns for selected stocks
    """
    # Filter price data for selected stocks
    price_data = daily_price[daily_price["gvkey"].isin(stock_list)].copy()
    
    # Pivot to have stocks as columns
    price_pivot = price_data.pivot_table(
        index="datadate",
        columns="gvkey",
        values="adj_close",
        aggfunc="first"
    )
    
    # Calculate daily returns
    returns = price_pivot.pct_change().dropna()
    
    # Use most recent lookback_days
    if len(returns) > lookback_days:
        returns = returns.iloc[-lookback_days:]
    
    return returns


def optimize_portfolio(returns, risk_free_rate=0.0):
    """
    Optimize portfolio to maximize Sharpe ratio
    No short sales allowed (weights >= 0, sum = 1)
    
    Args:
        returns (pandas.DataFrame): Historical returns for stocks
        risk_free_rate (float): Risk-free rate (annualized)
        
    Returns:
        dict: Optimal weights for each stock
    """
    n_assets = len(returns.columns)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Objective function: negative Sharpe ratio (minimize)
    def neg_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds: no short sales (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Optimize
    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        optimal_weights = result.x
        # Create weight dictionary
        weight_dict = {stock: weight for stock, weight in zip(returns.columns, optimal_weights)}
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        
        print(f"  Optimization successful!")
        print(f"  Expected Annual Return: {portfolio_return:.2%}")
        print(f"  Annual Volatility: {portfolio_std:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Number of stocks with weight > 0.01: {sum(optimal_weights > 0.01)}")
        
        return weight_dict
    else:
        print(f"  Optimization failed: {result.message}")
        # Return equal weights as fallback
        return {stock: 1.0 / n_assets for stock in returns.columns}


def build_portfolio_weights(stock_selected, daily_price, output_file, start_date='2018-01-01'):
    """
    Build portfolio weights for each trading day
    
    Args:
        stock_selected (pandas.DataFrame): Selected stocks
        daily_price (pandas.DataFrame): Daily price data
        output_file (str): Output file path
        start_date (str): Start date for backtest
    """
    print("\nBuilding Mean-Variance Portfolio...")
    
    # Filter data from start_date
    stock_selected = stock_selected[stock_selected["trade_date"] >= start_date].copy()
    daily_price = daily_price[daily_price["datadate"] >= start_date].copy()
    
    # Get unique trade dates (quarterly)
    trade_dates = sorted(stock_selected["trade_date"].unique())
    
    print(f"  Number of rebalancing periods: {len(trade_dates)}")
    
    # Store all weights
    all_weights = []
    
    for i, trade_date in enumerate(trade_dates):
        print(f"\nProcessing {i+1}/{len(trade_dates)}: {trade_date.date()}")
        
        # Get stocks selected for this quarter
        quarter_stocks = stock_selected[stock_selected["trade_date"] == trade_date]
        stock_list = quarter_stocks["gvkey"].unique().tolist()
        
        print(f"  Selected stocks: {len(stock_list)}")
        
        if len(stock_list) == 0:
            print(f"  Warning: No stocks selected for {trade_date}")
            continue
        
        # Get historical returns up to trade date
        historical_price = daily_price[daily_price["datadate"] < trade_date].copy()
        
        if len(historical_price) < 60:  # Need at least 60 days of history
            print(f"  Warning: Insufficient historical data ({len(historical_price)} days)")
            # Use equal weights
            weights_dict = {stock: 1.0 / len(stock_list) for stock in stock_list}
        else:
            try:
                # Calculate returns (using 252 days lookback)
                returns = calculate_returns(historical_price, stock_list, lookback_days=252)
                
                # Remove stocks with insufficient data
                valid_stocks = returns.columns[returns.notna().sum() > 30].tolist()
                
                if len(valid_stocks) < 2:
                    print(f"  Warning: Too few valid stocks ({len(valid_stocks)}), using equal weights")
                    weights_dict = {stock: 1.0 / len(stock_list) for stock in stock_list}
                else:
                    returns = returns[valid_stocks].dropna()
                    
                    # Optimize portfolio
                    weights_dict = optimize_portfolio(returns)
                    
                    # For stocks not in optimization, assign 0 weight
                    for stock in stock_list:
                        if stock not in weights_dict:
                            weights_dict[stock] = 0.0
                    
            except Exception as e:
                print(f"  Error in optimization: {e}")
                print(f"  Using equal weights as fallback")
                weights_dict = {stock: 1.0 / len(stock_list) for stock in stock_list}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v / total_weight for k, v in weights_dict.items()}
        
        # Determine holding period (until next trade date)
        if i < len(trade_dates) - 1:
            next_trade_date = trade_dates[i + 1]
        else:
            next_trade_date = daily_price["datadate"].max()
        
        # Get all dates in this holding period
        holding_dates = daily_price[
            (daily_price["datadate"] >= trade_date) & 
            (daily_price["datadate"] < next_trade_date)
        ]["datadate"].unique()
        
        # Create weight records for each day
        for date in holding_dates:
            for gvkey, weight in weights_dict.items():
                # Get ticker for this gvkey
                ticker_info = daily_price[(daily_price["gvkey"] == gvkey) & 
                                         (daily_price["datadate"] == date)]
                
                if not ticker_info.empty:
                    tic = ticker_info["tic"].iloc[0]
                    all_weights.append({
                        "trade_date": date,
                        "gvkey": gvkey,
                        "tic": tic,
                        "weights": weight
                    })
    
    # Create DataFrame and save
    weights_df = pd.DataFrame(all_weights)
    
    if len(weights_df) > 0:
        weights_df.to_csv(output_file, index=False)
        print(f"\n✓ Portfolio weights saved to: {output_file}")
        print(f"  Total records: {len(weights_df)}")
        print(f"  Date range: {weights_df['trade_date'].min()} to {weights_df['trade_date'].max()}")
        print(f"  Unique stocks: {weights_df['gvkey'].nunique()}")
    else:
        print("\n✗ Warning: No weights generated!")
    
    return weights_df


def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Mean-Variance Portfolio with Maximum Sharpe Ratio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python build_mvp_portfolio.py --stock_selected "./result/stock_selected.csv" --price_data "../../assets/WRDS-Security Daily.csv" --output "./result/portfolio_weights.csv"
  python build_mvp_portfolio.py --stock_selected "./result/stock_selected.csv" --price_data "../../assets/WRDS-Security Daily.csv" --start_date "2018-01-01"
        """
    )
    
    parser.add_argument(
        '--stock_selected',
        type=str,
        required=True,
        help='Path to stock_selected.csv file'
    )
    
    parser.add_argument(
        '--price_data',
        type=str,
        required=True,
        help='Path to WRDS-Security Daily.csv file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./result/portfolio_weights.csv',
        help='Output file path for portfolio weights (default: ./result/portfolio_weights.csv)'
    )
    
    parser.add_argument(
        '--start_date',
        type=str,
        default='2018-01-01',
        help='Start date for backtest (default: 2018-01-01)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Mean-Variance Portfolio Construction")
    print("=" * 80)
    print(f"Stock selection file: {args.stock_selected}")
    print(f"Price data file: {args.price_data}")
    print(f"Output file: {args.output}")
    print(f"Start date: {args.start_date}")
    print("-" * 80)
    
    # Check if input files exist
    if not os.path.exists(args.stock_selected):
        print(f"Error: Stock selection file not found: {args.stock_selected}")
        sys.exit(1)
    
    if not os.path.exists(args.price_data):
        print(f"Error: Price data file not found: {args.price_data}")
        sys.exit(1)
    
    # Load data
    stock_selected = load_stock_selection(args.stock_selected)
    daily_price = load_price_data(args.price_data, args.start_date)
    
    # Build portfolio
    weights_df = build_portfolio_weights(
        stock_selected,
        daily_price,
        args.output,
        args.start_date
    )
    
    print("\n" + "=" * 80)
    print("Portfolio Construction Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
