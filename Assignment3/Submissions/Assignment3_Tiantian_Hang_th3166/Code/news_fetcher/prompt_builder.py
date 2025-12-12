"""
Prompt builder module - adapted from FinGPT_Forecaster
Builds prompts for FinGPT model inference
"""

import os
import json
import random
import finnhub
import pandas as pd
from collections import defaultdict

# Import will be handled at runtime
try:
    from news_fetcher.data_fetcher import NewsDataFetcher
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from news_fetcher.data_fetcher import NewsDataFetcher


class PromptBuilder:
    """
    Builds prompts for FinGPT model
    Adapted from FinGPT_Forecaster/prompt.py
    """
    
    def __init__(self, finnhub_key=None):
        """
        Initialize prompt builder
        
        Args:
            finnhub_key: Finnhub API key (if None, uses FINNHUB_KEY env var)
        """
        self.fetcher = NewsDataFetcher(finnhub_key)
    
    def get_company_prompt(self, symbol):
        """
        Get company introduction prompt
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Formatted company introduction string
        """
        try:
            profile = self.fetcher.finnhub_client.company_profile2(symbol=symbol)
            
            company_template = (
                "[Company Introduction]:\n\n"
                "{name} is a leading entity in the {finnhubIndustry} sector. "
                "Incorporated and publicly traded since {ipo}, the company has established "
                "its reputation as one of the key players in the market. As of today, "
                "{name} has a market capitalization of {marketCapitalization:.2f} in {currency}, "
                "with {shareOutstanding:.2f} shares outstanding.\n\n"
                "{name} operates primarily in the {country}, trading under the ticker {ticker} "
                "on the {exchange}. As a dominant force in the {finnhubIndustry} space, "
                "the company continues to innovate and drive progress within the industry."
            )
            
            formatted_str = company_template.format(**profile)
            return formatted_str
        except Exception as e:
            print(f"Warning: Failed to get company profile for {symbol}: {e}")
            return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."
    
    def get_prompt_by_row(self, symbol, row):
        """
        Get prompt components for a specific row
        
        Args:
            symbol: Stock ticker symbol
            row: DataFrame row with Start Date, End Date, Start Price, End Price, News
            
        Returns:
            Tuple of (head, news_list, basics)
        """
        start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
        term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
        
        head = (
            f"From {start_date} to {end_date}, {symbol}'s stock price {term} "
            f"from {row['Start Price']:.2f} to {row['End Price']:.2f}. "
            f"Company news during this period are listed below:\n\n"
        )
        
        try:
            news = json.loads(row["News"])
            news = [
                f"[Headline]: {n['headline']}\n[Summary]: {n['summary']}\n"
                for n in news 
                if n['date'][:8] <= end_date.replace('-', '') and
                not n['summary'].startswith("Looking for stock market analysis and research with proves results?")
            ]
        except:
            news = []
        
        # Basics would be added separately if needed
        basics = None
        
        return head, news, basics
    
    def sample_news(self, news, k=5):
        """
        Sample k news items from news list
        
        Args:
            news: List of news items
            k: Number of items to sample
            
        Returns:
            List of sampled news items
        """
        if len(news) == 0:
            return []
        k = min(k, len(news))
        return [news[i] for i in sorted(random.sample(range(len(news)), k))]
    
    def get_all_prompts_online(self, symbol, data, curday, with_basics=True):
        """
        Build complete prompt for online inference
        
        Args:
            symbol: Stock ticker symbol
            data: DataFrame with stock data and news
            curday: Current date string
            with_basics: Whether to include basic financials
            
        Returns:
            Tuple of (info, prompt)
        """
        company_prompt = self.get_company_prompt(symbol)
        
        prev_rows = []
        for row_idx, row in data.iterrows():
            head, news, _ = self.get_prompt_by_row(symbol, row)
            prev_rows.append((head, news, None))
        
        prompt = ""
        for i in range(-len(prev_rows), 0):
            prompt += "\n" + prev_rows[i][0]
            sampled_news = self.sample_news(
                prev_rows[i][1],
                min(5, len(prev_rows[i][1]))
            )
            if sampled_news:
                prompt += "\n".join(sampled_news)
            else:
                prompt += "No relative news reported."
        
        period = f"{curday} to {self.fetcher.n_weeks_before(curday, -1)}"
        
        if with_basics:
            basics = self.fetcher.get_current_basics(symbol, curday)
            if basics:
                basics_str = (
                    f"Some recent basic financials of {symbol}, reported at {basics.get('period', 'N/A')}, "
                    f"are presented below:\n\n[Basic Financials]:\n\n" +
                    "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
                )
            else:
                basics_str = "[Basic Financials]:\n\nNo basic financial reported."
        else:
            basics_str = "[Basic Financials]:\n\nNo basic financial reported."
        
        info = company_prompt + '\n' + prompt + '\n' + basics_str
        prompt = (
            info + 
            f"\n\nBased on all the information before {curday}, let's first analyze the positive "
            f"developments and potential concerns for {symbol}. Come up with 2-4 most important "
            f"factors respectively and keep them concise. Most factors should be inferred from "
            f"company related news. Then make your prediction of the {symbol} stock price movement "
            f"for next week ({period}). Provide a summary analysis to support your prediction."
        )
        
        return info, prompt

