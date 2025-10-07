#!/usr/bin/env python3
"""
Manual benchmark data downloader
Use this script to download SP500 and QQQ data when the main backtest script fails
"""

import yfinance as yf
import pandas as pd
import time
import os

def download_with_retry(symbol, start_date, end_date, max_retries=5):
    """Download data with extended retry logic"""
    print(f"Attempting to download {symbol}...")
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            data = yf.download(symbol, start=start_date, end=end_date, interval="1d")
            
            if not data.empty:
                print(f"  ✓ Successfully downloaded {symbol} ({len(data)} records)")
                return data
            else:
                print(f"  ⚠ Empty data returned for {symbol}")
                
        except Exception as e:
            print(f"  ✗ Error downloading {symbol}: {e}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 60  # Increasing wait time: 1min, 2min, 3min...
            print(f"  Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print(f"Failed to download {symbol} after {max_retries} attempts")
    return None

def main():
    """Main function to download benchmark data"""
    print("=" * 60)
    print("Manual Benchmark Data Downloader")
    print("=" * 60)
    
    # Create cache directory
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download SP500
    print("\n1. Downloading S&P 500 data...")
    sp500_data = download_with_retry("^GSPC", "1990-01-01", "2025-09-09")
    if sp500_data is not None:
        cache_file = os.path.join(cache_dir, "GSPC_1990-01-01_2025-09-09.csv")
        sp500_data.to_csv(cache_file)
        print(f"✓ SP500 data saved to: {cache_file}")
    
    # Wait between downloads
    print("\nWaiting 5 minutes before downloading QQQ...")
    time.sleep(300)  # 5 minutes
    
    # Download QQQ
    print("\n2. Downloading QQQ data...")
    qqq_data = download_with_retry("QQQ", "1990-01-01", "2025-09-09")
    if qqq_data is not None:
        cache_file = os.path.join(cache_dir, "QQQ_1990-01-01_2025-09-09.csv")
        qqq_data.to_csv(cache_file)
        print(f"✓ QQQ data saved to: {cache_file}")
    
    print("\n" + "=" * 60)
    print("Download completed!")
    print("You can now run backtest.py and it will use the cached data.")
    print("=" * 60)

if __name__ == "__main__":
    main()
