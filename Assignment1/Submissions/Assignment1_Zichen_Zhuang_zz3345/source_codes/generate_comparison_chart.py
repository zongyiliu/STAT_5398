import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

print("=" * 80)
print("Portfolio Performance Comparison Chart Generator")
print("=" * 80)

# Load portfolio metrics
metrics_file = './results/performance_metrics.json'
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

cumulative_return = float(metrics['Cumulative Return']) * 100
annual_return = float(metrics['Annual Return']) * 100
volatility = float(metrics['Annual Volatility']) * 100
sharpe = float(metrics['Sharpe Ratio'])
max_dd = float(metrics['Max Drawdown']) * 100

print(f"Portfolio Metrics:")
print(f"Cumulative Return: {cumulative_return:.2f}%")
print(f"Annual Return: {annual_return:.2f}%")
print(f"Volatility: {volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.2f}%")

# Load real daily portfolio values from backtest
portfolio_values_df = pd.read_csv('./results/portfolio_daily_values.csv', parse_dates=['date'])
portfolio_dates = portfolio_values_df['date'].tolist()
portfolio_values = portfolio_values_df['portfolio_value'].tolist()
print(f"Portfolio Daily Values: {len(portfolio_values)} days")

# Load benchmark data
benchmark_file = './results/benchmark_data.csv'
benchmark_df = pd.read_csv(benchmark_file, parse_dates=['date'])
benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], utc=True).dt.tz_localize(None)
print(f"\nBenchmark Data:")
print(f"  Records: {len(benchmark_df)}")
print(f"  Date range: {benchmark_df['date'].min().date()} to {benchmark_df['date'].max().date()}")

# Calculate benchmark returns
sp500_return = (benchmark_df['SP500'].iloc[-1] / benchmark_df['SP500'].iloc[0] - 1) * 100
qqq_return = (benchmark_df['QQQ'].iloc[-1] / benchmark_df['QQQ'].iloc[0] - 1) * 100
print(f"  S&P 500 Return: {sp500_return:.2f}%")
print(f"  QQQ Return: {qqq_return:.2f}%")

# Set index after calculations
benchmark_df = benchmark_df.set_index('date')

# Align benchmark data with portfolio dates
sp500_values = []
qqq_values = []

for date in portfolio_dates:
    date_pd = pd.to_datetime(date)
    if date_pd in benchmark_df.index:
        sp500_values.append(benchmark_df.loc[date_pd, 'SP500'])
        qqq_values.append(benchmark_df.loc[date_pd, 'QQQ'])
    else:
        # Forward fill
        valid_dates = benchmark_df.index[benchmark_df.index <= date_pd]
        if len(valid_dates) > 0:
            sp500_values.append(benchmark_df.loc[valid_dates[-1], 'SP500'])
            qqq_values.append(benchmark_df.loc[valid_dates[-1], 'QQQ'])
        else:
            sp500_values.append(benchmark_df['SP500'].iloc[0])
            qqq_values.append(benchmark_df['QQQ'].iloc[0])

print(f"Benchmark values aligned: {len(sp500_values)} days")

# Create comparison chart - same style as backtest.py
initial_capital = portfolio_values[0]
sp500_norm = initial_capital * (np.array(sp500_values) / sp500_values[0])
qqq_norm = initial_capital * (np.array(qqq_values) / qqq_values[0])

fig, ax = plt.subplots(figsize=(15, 6))

# Normalize to start at 1.0 for comparison
ax.plot(portfolio_dates, [data/portfolio_values[0] for data in portfolio_values], 
        label='Portfolio Value with Quarterly Adjustment', 
        color='blue', linewidth=2)
ax.plot(portfolio_dates, [data/sp500_values[0] for data in sp500_values], 
        label='SP500', 
        color='green', linewidth=2)
ax.plot(portfolio_dates, [data/qqq_values[0] for data in qqq_values], 
        label='QQQ', 
        color='red', linewidth=2)

ax.set_title("Portfolio Values", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

# Add cumulative returns text box in upper left
textstr = f'Portfolio: +{cumulative_return:.2f}%\nS&P 500: +{sp500_return:.2f}%\nQQQ: +{qqq_return:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85))

plt.tight_layout()

# Save chart
output_file = './results/portfolio_performance_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

# Also update the main performance chart
plt.savefig('./results/portfolio_performance.png', dpi=300, bbox_inches='tight')
print(f"Updated: ./results/portfolio_performance.png")

plt.close()