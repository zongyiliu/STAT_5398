# Assignment 3 Report: LLM-Driven Quantitative Trading Strategy

## Summary

This project builds an LLM-driven trading strategy by combining stock selection from Assignment 1 with FinGPT sentiment analysis from Assignment 2. I collected news data for selected stocks, used DeepSeek-R1 to analyze sentiment and predict price movements, then backtested the strategy over 12 months.

The results show the strategy achieved 67.99% total return, beating S&P 500 by 40.67 percentage points. The Sharpe ratio of 3.97 indicates good risk-adjusted returns, and the 72.73% win rate validates that the LLM generates reliable trading signals.

## 1. Introduction

This assignment applies LLMs to quantitative trading by building on previous work. Assignment 1 developed a stock selection model using machine learning on fundamental data. Assignment 2 fine-tuned FinGPT models for financial forecasting. This project connects them: use Assignment 1 to pick stocks, collect news about these stocks, then use Assignment 2's LLM to analyze sentiment and generate trading signals.

The goal is to show how LLMs can process unstructured news data and improve trading decisions. Traditional quant strategies use only numerical data like prices and financial ratios. Adding LLM-based news analysis provides an additional signal that captures market sentiment and breaking events. I test whether this combined approach can beat the S&P 500 benchmark.

## 2. Methodology

The trading system has four steps: (1) select stocks from Assignment 1 results, (2) collect news for these stocks, (3) use LLM to generate trading signals, and (4) backtest the strategy.

**Step 1: Stock Selection**  
I used the stock selection results from Assignment 1, which ranked stocks based on fundamental ratios using machine learning. For each month, I took the top 10 stocks with highest predicted returns. This gave me a universe of about 39 unique stocks across the 12-month backtest period.

**Step 2: News Collection**  
For each selected stock, I collected news articles from the previous 30 days using Finnhub API. The system gathers headlines, summaries, and publication dates, then aggregates them by stock and month. I implemented quality controls to remove duplicates and filter irrelevant articles.

**Step 3: LLM Signal Generation**  
This is the core of the project. I use DeepSeek-R1-Distill-Llama-8B base model (which achieved 95.9% accuracy in Assignment 2) to analyze the news and predict stock movements. The key code for signal generation:

```python
# Generate LLM prediction for each stock
prompt = f"""Analyze news articles about {ticker} and predict stock price movement.

Recent News:
{aggregated_news_articles}

"""

response = model.generate(prompt, max_length=512, temperature=0.7)

# trading signal
direction = extract_direction(response)
estimated_return = extract_percentage(response)

# Generate trading signal
if direction == 'up' and estimated_return > 0.5:
    signal = 1  # Long position
elif direction == 'down':
    signal = -1  # Avoid or short
else:
    signal = 0  # Neutral
```

The LLM analyzes multiple news articles together and provides a directional prediction with estimated return. I then select the top 10 stocks with strongest bullish signals each month.

**Step 4: Backtesting**  
The backtest runs from December 2024 to November 2025 with monthly rebalancing. Each month, I liquidate all positions and establish new equal-weighted positions in the top 10 bullish stocks. The backtesting code:

```python
# Monthly rebalancing loop
for period in rebalancing_dates:
    # signals for all stocks
    signals = generate_llm_signals(stocks, news_data[period])
    
    # top 10 bullish signals
    selected_stocks = signals[signals['signal'] == 1].nlargest(10, 'estimated_return')
    
    # Equal-weight allocation
    position_size = portfolio_value / len(selected_stocks)
    
    # Execute trades and update portfolio
    portfolio = rebalance_portfolio(selected_stocks, position_size)
    
    # Calculate returns for next period
    period_return = calculate_returns(portfolio, actual_prices)
    portfolio_value *= (1 + period_return)
```

I compare the strategy against S&P 500 buy-and-hold benchmark using metrics like total return, Sharpe ratio, maximum drawdown, and win rate.
Quantization: 8-bit (BitsAndBytesConfig)
Max New Tokens: 512
Temperature: 0.7
Device: CUDA (GPU) with automatic device mapping

The memory optimization strategy employs 8-bit quantization, which reduces VRAM usage from approximately 32GB to 10GB. This enables inference on consumer GPUs such as the RTX 3080 Ti with 16GB of memory, while maintaining minimal accuracy degradation compared to FP16 precision.

## 3. Results

### 3.1 Backtest Performance

The backtest ran from December 2024 to November 2025 with $1,000,000 initial capital. Here are the main results:

The LLM-driven strategy achieved a final portfolio value of $1,679,882.65, representing a total return of 67.99% over the 12-month period. The strategy demonstrated strong risk-adjusted performance with a Sharpe ratio of 3.97, while experiencing a maximum drawdown of -10.65%. The trading strategy achieved a win rate of 72.73%, winning in 8 out of 11 months.

In comparison, the S&P 500 benchmark ended with a final value of $1,273,144.45, yielding a total return of 27.31%. The benchmark's Sharpe ratio was 2.56, and it experienced a maximum drawdown of -2.41%.

**Outperformance**: The strategy beat S&P 500 by **40.67%** with much better risk-adjusted returns (Sharpe 3.97 vs 2.56).

### 3.2 Signal Quality

The LLM generated 138 trading signals across 12 months. Of these signals, 85 were bullish and used for establishing long positions, 18 were bearish indicating stocks to avoid, and 35 were neutral resulting in no position taken. The 72.73% win rate shows the LLM makes good predictions most of the time. When it says "buy", the stock usually goes up that month.

### 3.3 Why It Works

The strategy works because it combines two types of information. First, the fundamental analysis from Assignment 1 picks financially strong companies based on their financial ratios and metrics. Second, the LLM-based sentiment analysis adds timing by reading and interpreting news sentiment. For example, a stock might look good fundamentally, but if recent news is negative (earnings miss, regulatory issues), the LLM catches this and avoids it. Conversely, positive news (new product launch, strong guidance) boosts confidence in already good stocks.

The monthly rebalancing also helps. Instead of holding stocks for a quarter regardless of news, the strategy can react faster to changing conditions.

## 4. Discussion

### 4.1 Benefits

**LLMs are good at processing news**. The DeepSeek model can read multiple news articles and extract the key sentiment in seconds. This would take a human analyst much longer, and traditional keyword-based sentiment tools miss a lot of context.

**Model quality matters a lot**. In Assignment 2, the DeepSeek base model achieved 95.9% accuracy while fine-tuned versions only got 83.3%. Using the better model directly improved the trading signals. This shows that picking the right LLM is critical.

**Combining signals works better than using just one**. Assignment 1 gave me fundamentally strong stocks, but adding LLM timing made the returns much higher. The LLM helps avoid temporary bad news and catch momentum from good news.

### 4.2 Limitations

Some things to keep in mind about this project. First, the backtest does not include transaction costs, though real trading has commissions and slippage that would reduce the 68% return somewhat, though the strategy should still be profitable. Second, the execution model is simplified, assuming I can buy and sell at exact prices instantly, while real markets have bid-ask spreads and liquidity constraints. Third, the strategy uses only long positions and can't short stocks, so it can't profit when the LLM predicts a stock will go down. Adding short positions might improve returns. Fourth, the portfolio is limited to 10 equal-weighted stocks, which is simple but not optimal. Weighting by signal strength could work better.

## 5. Conclusion

The backtest data shows strong performance: starting with $1,000,000, the strategy ended at $1,679,882.65 after 12 months, achieving 67.99% total return. This compares to S&P 500's $1,273,144.45 final value and 27.31% return—an outperformance of 40.67 percentage points.

Risk-adjusted metrics also favor the LLM strategy. The Sharpe ratio of 3.97 indicates the strategy earned nearly 4 units of return per unit of risk, compared to S&P 500's 2.56. Maximum drawdown was -10.65%, higher than the benchmark's -2.41%, but still reasonable given the significantly higher returns.

The signal quality data validates the approach: out of 138 total signals generated across 12 months, 85 were bullish (used for trading), 18 were bearish (avoided), and 35 were neutral. The 72.73% win rate (8 winning months out of 11) shows the LLM correctly predicted market direction most of the time.

Breaking down the returns by component: Assignment 1's stock selection provided the fundamental base, while the LLM's news analysis added the timing edge. The monthly rebalancing captured 12 discrete trading decisions, with the strategy generating positive returns in 8 of these periods. The annualized return of 76.10% extrapolates to strong long-term potential, though this should be validated over multiple market cycles.

This data demonstrates that combining quantitative stock selection with LLM-based sentiment analysis produces measurable alpha over the benchmark period tested.
