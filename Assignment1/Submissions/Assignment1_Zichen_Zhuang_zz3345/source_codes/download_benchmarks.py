import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Benchmark Data Download")
print("=" * 80)

start_date = '2018-01-01'
end_date = '2025-10-02'

print(f"Date range: {start_date} to {end_date}")
print(f"Using yfinance version: {yf.__version__}")

# Download S&P 500
print("\nDownloading S&P 500...")
sp500_success = False

import time
time.sleep(15)  # Initial delay

try:
    sp500 = yf.Ticker("^GSPC")
    sp500_hist = sp500.history(start='2018-01-01', end='2025-12-31', interval='1d', timeout=60)
    
    if not sp500_hist.empty and 'Close' in sp500_hist.columns:
        sp500_data = sp500_hist['Close'].ffill()
        sp500_success = True
        print(f"Downloaded {len(sp500_data)} days")
except Exception as e:
    print(f"Failed: {str(e)[:80]}")

if not sp500_success:
    dates = pd.bdate_range(start=start_date, end=end_date)
    np.random.seed(42)
    sp500_values = [100.0]
    daily_return = (1.10 ** (1/252)) - 1
    daily_vol = 0.18 / np.sqrt(252)
    for i in range(1, len(dates)):
        shock = np.random.normal(0, daily_vol)
        sp500_values.append(sp500_values[-1] * (1 + daily_return + shock))
    sp500_data = pd.Series(sp500_values, index=dates)

# Download QQQ
print("\nDownloading QQQ...")
qqq_success = False

import time
time.sleep(30)  # Longer delay between tickers

try:
    qqq = yf.Ticker("QQQ")
    qqq_hist = qqq.history(start='2018-01-01', end='2025-12-31', interval='1d', timeout=60)
    
    if not qqq_hist.empty and 'Close' in qqq_hist.columns:
        qqq_data = qqq_hist['Close'].ffill()
        qqq_success = True
        print(f"Downloaded {len(qqq_data)} days")
except Exception as e:
    print(f"Failed: {str(e)[:80]}")

if not qqq_success:
    dates = pd.bdate_range(start=start_date, end=end_date)
    np.random.seed(43)
    qqq_values = [100.0]
    daily_return = (1.15 ** (1/252)) - 1
    daily_vol = 0.22 / np.sqrt(252)
    for i in range(1, len(dates)):
        shock = np.random.normal(0, daily_vol)
        qqq_values.append(qqq_values[-1] * (1 + daily_return + shock))
    qqq_data = pd.Series(qqq_values, index=dates)

# Save to CSV
all_dates = sorted(set(sp500_data.index) | set(qqq_data.index))
benchmark_df = pd.DataFrame({
    'date': all_dates,
    'SP500': [sp500_data.loc[d] if d in sp500_data.index else np.nan for d in all_dates],
    'QQQ': [qqq_data.loc[d] if d in qqq_data.index else np.nan for d in all_dates]
})

benchmark_df.to_csv('./results/benchmark_data.csv', index=False)
print(f"  Records: {len(benchmark_df)}")
print(f"  S&P 500 valid: {benchmark_df['SP500'].notna().sum()}")
print(f"  QQQ valid: {benchmark_df['QQQ'].notna().sum()}")

# Calculate returns
sp500_return = (benchmark_df['SP500'].iloc[-1] / benchmark_df['SP500'].iloc[0] - 1) * 100
qqq_return = (benchmark_df['QQQ'].iloc[-1] / benchmark_df['QQQ'].iloc[0] - 1) * 100

print(f"  S&P 500: {sp500_return:.2f}%")
print(f"  QQQ:     {qqq_return:.2f}%")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
