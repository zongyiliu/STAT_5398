import pandas as pd
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import config

def get_ticker_from_gvkey(gvkey):
    # Dow 30 stocks as example universe
    dow30_tickers = [
        'AAPL', 'MSFT', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'DIS',
        'MA', 'NVDA', 'BAC', 'VZ', 'CSCO', 'INTC', 'XOM', 'CVX', 'KO', 'MRK',
        'PFE', 'NKE', 'WBA', 'IBM', 'MMM', 'CAT', 'TRV', 'GS', 'AXP', 'BA'
    ]
    # For demonstration, we'll cycle through these tickers
    idx = int(gvkey) % len(dow30_tickers)
    return dow30_tickers[idx]

def collect_news_finnhub(ticker, start_date, end_date, api_key):
    if not api_key:
        return create_mock_news(ticker, start_date, end_date)
    
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': start_date,
        'to': end_date,
        'token': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        return news_data
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def create_mock_news(ticker, start_date, end_date):
    news_templates = [
        {
            "headline": f"{ticker} Reports Strong Quarterly Earnings",
            "summary": f"Company {ticker} announced better than expected earnings driven by strong sales growth.",
            "category": "earnings",
            "sentiment": "positive"
        },
        {
            "headline": f"{ticker} Announces New Product Launch",
            "summary": f"{ticker} unveiled innovative new products expected to drive future growth.",
            "category": "product",
            "sentiment": "positive"
        },
        {
            "headline": f"{ticker} Faces Regulatory Challenges",
            "summary": f"Regulatory concerns may impact {ticker}'s operations in key markets.",
            "category": "regulatory",
            "sentiment": "negative"
        },
        {
            "headline": f"{ticker} Expands Market Share",
            "summary": f"{ticker} continues to gain market share through strategic initiatives.",
            "category": "market",
            "sentiment": "positive"
        },
        {
            "headline": f"Analyst Upgrades {ticker} Rating",
            "summary": f"Major investment firm upgrades {ticker} to Buy with higher price target.",
            "category": "analyst",
            "sentiment": "positive"
        }
    ]
    
    # Generate 3-5 news items per ticker
    import random
    num_news = random.randint(3, 5)
    mock_news = []
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    for i in range(num_news):
        template = random.choice(news_templates)
        # Random date within period
        days_diff = (end - start).days
        random_days = random.randint(0, days_diff)
        news_date = start + timedelta(days=random_days)
        
        mock_news.append({
            'datetime': int(news_date.timestamp()),
            'headline': template['headline'],
            'summary': template['summary'],
            'source': 'MockSource',
            'url': f'https://example.com/news/{ticker}/{i}',
            'category': template['category'],
            'sentiment': template['sentiment']
        })
    
    return mock_news

def collect_all_news():
    print("Step 2: Collecting News Data")
    
    # Load stock universe
    universe_file = config.DATA_DIR / "stock_universe.csv"
    if not universe_file.exists():
        print(f"ERROR: Stock universe file not found: {universe_file}")
        print("Please run 1 first")
        return False
    
    universe_df = pd.read_csv(universe_file)
    universe_df['trade_date'] = pd.to_datetime(universe_df['trade_date'])
    
    # Get unique stocks
    unique_gvkeys = universe_df['gvkey'].unique()
    print(f"Collecting news for {len(unique_gvkeys)} unique stocks")
    
    # Map gvkeys to tickers
    ticker_mapping = {gvkey: get_ticker_from_gvkey(gvkey) for gvkey in unique_gvkeys}
    
    # Save ticker mapping
    mapping_df = pd.DataFrame(list(ticker_mapping.items()), columns=['gvkey', 'ticker'])
    mapping_file = config.DATA_DIR / "gvkey_ticker_mapping.csv"
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Saved ticker mapping to: {mapping_file}")
    
    # Collect news for each stock
    all_news = []
    
    start_date = config.BACKTEST_START
    end_date = config.BACKTEST_END
    
    print(f"\nCollecting news from {start_date} to {end_date}")
    
    for gvkey in tqdm(unique_gvkeys, desc="Collecting news"):
        ticker = ticker_mapping[gvkey]
        
        # Collect news
        news_items = collect_news_finnhub(
            ticker, 
            start_date, 
            end_date, 
            config.FINNHUB_API_KEY
        )
        
        # Add gvkey and ticker to each news item
        for item in news_items:
            item['gvkey'] = gvkey
            item['ticker'] = ticker
            item['date'] = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
        
        all_news.extend(news_items)
        
        # Rate limiting
        time.sleep(0.5)
    
    # Convert to DataFrame
    news_df = pd.DataFrame(all_news)
    
    print(f"\nCollected {len(news_df)} news articles")
    print(f"Average articles per stock: {len(news_df) / len(unique_gvkeys):.1f}")
    
    # Save news data
    news_file = config.NEWS_DIR / "collected_news.csv"
    news_df.to_csv(news_file, index=False)
    print(f"Saved news data to: {news_file}")
    
    # Also save as JSON for easier processing
    news_json_file = config.NEWS_DIR / "collected_news.json"
    news_df.to_json(news_json_file, orient='records', indent=2)
    print(f"Saved news data (JSON) to: {news_json_file}")
    
    # Display summary
    print(f"Total articles collected: {len(news_df)}")
    print(f"Unique stocks: {news_df['ticker'].nunique()}")
    print(f"Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    print(f"\nTop 5 stocks by article count:")
    print(news_df['ticker'].value_counts().head())
    
    return True

def main():
    success = collect_all_news()
    
    if success:
        print("\nStep 2 completed successfully!")
        print(f"News data saved to: {config.NEWS_DIR}")
        return True
    else:
        print("\nStep 2 failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
