# News Fetcher Module

## Overview

The `news_fetcher` module is responsible for fetching financial news data for selected stocks. It integrates with the Finnhub API to retrieve company news and prepares the data for sentiment analysis by the FinGPT model.

## Components

### 1. `data_fetcher.py`

Main module containing the `NewsDataFetcher` class with the following key methods:

- **`get_stock_data(symbol, steps)`**: Downloads stock price data from Yahoo Finance
- **`get_news(symbol, data)`**: Fetches company news from Finnhub API for given date ranges
- **`get_current_basics(symbol, curday)`**: Retrieves basic financial metrics
- **`fetch_all_data(symbol, curday, n_weeks)`**: Fetches all data (prices + news) for a symbol
- **`build_gvkey_to_ticker_map(price_data_path)`**: Maps gvkey identifiers to ticker symbols
- **`get_news_for_selected_stocks(selected_stocks_df, ...)`**: Fetches news for stocks from `stock_selected.csv`

### 2. `prompt_builder.py`

Contains functions for building prompts for the FinGPT model based on:
- Company profiles
- Historical price movements
- News articles
- Basic financials

## Setup

### 1. Install Dependencies

```bash
pip install finnhub-python yfinance pandas numpy
```

### 2. Set Up API Key

You need a Finnhub API key to fetch news data. Get one from: https://finnhub.io/

Set the environment variable:

```bash
export FINNHUB_KEY='your_api_key_here'
# or
export FINNHUB_API_KEY='your_api_key_here'
```

### 3. Rate Limits

Finnhub has rate limits:
- Free tier: 60 calls/minute
- The code includes `time.sleep(1)` between API calls to respect rate limits

## Usage

### Basic Usage

```python
from news_fetcher.data_fetcher import NewsDataFetcher

# Initialize fetcher
fetcher = NewsDataFetcher()

# Fetch news for a single stock
symbol = "AAPL"
curday = "2024-12-01"
data = fetcher.fetch_all_data(symbol, curday, n_weeks=3)
```

### Fetching News for Selected Stocks

```python
import pandas as pd
from news_fetcher.data_fetcher import NewsDataFetcher

# Load selected stocks
selected_stocks = pd.read_csv("data_processor/outputs/stock_selected.csv")
selected_stocks['trade_date'] = pd.to_datetime(selected_stocks['trade_date'])

# Initialize fetcher
fetcher = NewsDataFetcher()

# Fetch news for a specific trade date
trade_date = pd.to_datetime("2024-12-01")
results = fetcher.get_news_for_selected_stocks(
    selected_stocks,
    price_data_path="../dec data/sp500_tickers_daily_price.csv",
    trade_date=trade_date,
    n_weeks=3
)

# Results is a dictionary: {(gvkey, trade_date): {'ticker': ..., 'news_data': ...}}
```

## Testing

Run the test script to verify the module works:

```bash
cd news_fetcher
python test_news_fetcher.py
```

The test script will:
1. Test basic news fetching for a single stock
2. Test gvkey to ticker mapping
3. Test fetching news for selected stocks

## Data Format

### News Data Structure

News data is stored as JSON strings in the `News` column:

```json
[
  {
    "date": "20241125120000",
    "headline": "Company announces new product",
    "summary": "Full article summary..."
  },
  ...
]
```

### Selected Stocks Format

The `stock_selected.csv` file should have:
- `gvkey`: Company identifier
- `trade_date`: Trading date (YYYY-MM-DD)
- `predicted_return`: Predicted return value

## Integration with Sentiment Agent

The news data fetched by this module is used by the `sentiment_agent` to:
1. Build prompts for the FinGPT model
2. Generate sentiment signals
3. Make predictions about stock price movements

## Notes

- **API Rate Limits**: Be mindful of Finnhub API rate limits. The code includes delays between calls.
- **Error Handling**: The module includes error handling for missing data or API failures.
- **Date Alignment**: News data is aligned with trading dates to avoid look-ahead bias.
- **Ticker Mapping**: The module handles mapping from gvkey (internal identifier) to ticker symbols (used by APIs).

## Next Steps

After fetching news data, the next step is to:
1. Use `prompt_builder.py` to construct prompts
2. Pass prompts to the `sentiment_agent` for analysis
3. Generate sentiment signals for portfolio management


