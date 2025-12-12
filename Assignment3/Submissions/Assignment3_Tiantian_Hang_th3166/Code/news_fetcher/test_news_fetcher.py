#!/usr/bin/env python3
"""
Test script for NewsDataFetcher
Tests news fetching functionality with selected stocks
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_fetcher.data_fetcher import NewsDataFetcher


def test_basic_news_fetching():
    """Test basic news fetching for a single stock"""
    print("=" * 80)
    print("Test 1: Basic News Fetching")
    print("=" * 80)
    
    # Check for API key
    if not os.environ.get("FINNHUB_KEY") and not os.environ.get("FINNHUB_API_KEY"):
        print("ERROR: FINNHUB_KEY or FINNHUB_API_KEY environment variable not set")
        print("Please set your Finnhub API key:")
        print("  export FINNHUB_KEY='your_api_key_here'")
        return False
    
    try:
        # Initialize fetcher
        api_key = os.environ.get("FINNHUB_KEY") or os.environ.get("FINNHUB_API_KEY")
        fetcher = NewsDataFetcher(finnhub_key=api_key)
        
        # Test with a well-known stock (AAPL)
        symbol = "AAPL"
        curday = "2024-12-01"
        n_weeks = 2
        
        print(f"\nFetching news for {symbol} from {n_weeks} weeks before {curday}...")
        data = fetcher.fetch_all_data(symbol, curday, n_weeks=n_weeks)
        
        print(f"\n✓ Successfully fetched data:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {data.columns.tolist()}")
        
        if 'News' in data.columns:
            # Parse first news entry
            if len(data) > 0 and len(data.iloc[0]['News']) > 0:
                import json
                news = json.loads(data.iloc[0]['News'])
                if len(news) > 0:
                    print(f"\n  Sample news (first entry):")
                    print(f"    Date: {news[0].get('date', 'N/A')}")
                    print(f"    Headline: {news[0].get('headline', 'N/A')[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gvkey_mapping():
    """Test gvkey to ticker mapping"""
    print("\n" + "=" * 80)
    print("Test 2: GVKEY to Ticker Mapping")
    print("=" * 80)
    
    price_data_path = "../../dec data/sp500_tickers_daily_price.csv"
    
    if not os.path.exists(price_data_path):
        print(f"ERROR: Price data file not found: {price_data_path}")
        print("Skipping this test...")
        return False
    
    try:
        api_key = os.environ.get("FINNHUB_KEY") or os.environ.get("FINNHUB_API_KEY")
        fetcher = NewsDataFetcher(finnhub_key=api_key)
        
        print(f"\nBuilding gvkey to ticker mapping from {price_data_path}...")
        mapping = fetcher.build_gvkey_to_ticker_map(price_data_path=price_data_path)
        
        print(f"\n✓ Successfully built mapping:")
        print(f"  Total stocks: {len(mapping)}")
        
        # Show some examples
        sample_gvkeys = list(mapping.keys())[:5]
        print(f"\n  Sample mappings:")
        for gvkey in sample_gvkeys:
            print(f"    gvkey {gvkey} -> {mapping[gvkey]}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selected_stocks_news():
    """Test fetching news for selected stocks"""
    print("\n" + "=" * 80)
    print("Test 3: Fetching News for Selected Stocks")
    print("=" * 80)
    
    selected_stocks_path = "../data_processor/outputs/stock_selected.csv"
    price_data_path = "../../dec data/sp500_tickers_daily_price.csv"
    
    if not os.path.exists(selected_stocks_path):
        print(f"ERROR: Selected stocks file not found: {selected_stocks_path}")
        print("Skipping this test...")
        return False
    
    if not os.path.exists(price_data_path):
        print(f"ERROR: Price data file not found: {price_data_path}")
        print("Skipping this test...")
        return False
    
    try:
        # Load selected stocks
        print(f"\nLoading selected stocks from {selected_stocks_path}...")
        selected_stocks = pd.read_csv(selected_stocks_path)
        selected_stocks['trade_date'] = pd.to_datetime(selected_stocks['trade_date'])
        
        print(f"  Loaded {len(selected_stocks)} records")
        print(f"  Date range: {selected_stocks['trade_date'].min()} to {selected_stocks['trade_date'].max()}")
        print(f"  Unique stocks: {selected_stocks['gvkey'].nunique()}")
        
        # Get a sample trade date (most recent)
        sample_trade_date = selected_stocks['trade_date'].max()
        print(f"\n  Testing with trade_date: {sample_trade_date}")
        
        # Filter to a small sample for testing (first 3 stocks)
        sample_stocks = selected_stocks[
            selected_stocks['trade_date'] == sample_trade_date
        ].head(3)
        
        print(f"  Testing with {len(sample_stocks)} stocks...")
        
        # Initialize fetcher
        api_key = os.environ.get("FINNHUB_KEY") or os.environ.get("FINNHUB_API_KEY")
        fetcher = NewsDataFetcher(finnhub_key=api_key)
        
        # Fetch news
        print(f"\nFetching news (this may take a while due to API rate limits)...")
        results = fetcher.get_news_for_selected_stocks(
            sample_stocks,
            price_data_path=price_data_path,
            trade_date=sample_trade_date,
            n_weeks=2
        )
        
        print(f"\n✓ Successfully fetched news for {len(results)} stocks:")
        for (gvkey, td), result in results.items():
            ticker = result['ticker']
            news_data = result['news_data']
            print(f"  {ticker} (gvkey: {gvkey}): {len(news_data)} time periods")
            if len(news_data) > 0 and 'News' in news_data.columns:
                import json
                news_str = news_data.iloc[0]['News']
                if news_str:
                    news = json.loads(news_str)
                    print(f"    First period: {len(news)} news articles")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("News Fetcher Test Suite")
    print("=" * 80)
    
    results = []
    
    # Test 1: Basic news fetching
    results.append(("Basic News Fetching", test_basic_news_fetching()))
    
    # Test 2: GVKEY mapping
    results.append(("GVKEY to Ticker Mapping", test_gvkey_mapping()))
    
    # Test 3: Selected stocks news
    results.append(("Selected Stocks News", test_selected_stocks_news()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


