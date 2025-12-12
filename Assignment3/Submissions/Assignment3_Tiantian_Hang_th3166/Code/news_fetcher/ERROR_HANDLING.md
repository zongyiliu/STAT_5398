# 新闻数据获取的异常处理策略

## 概述

在获取新闻数据时，可能会遇到多种异常情况。本文档详细说明了我们的异常处理策略。

## 异常情况分类

### 1. API 调用失败

**可能原因**：
- 网络连接问题
- API 速率限制
- API 密钥无效或过期
- API 服务暂时不可用

**处理方式**：
```python
try:
    weekly_news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
except Exception as e:
    print(f"Warning: Failed to fetch news for {symbol} from {start_date} to {end_date}: {e}")
    weekly_news = []  # 返回空列表，不中断流程
```

### 2. 没有新闻数据

**可能原因**：
- 该股票在指定时间段内确实没有新闻
- 股票代码无效或已退市
- 时间段太短或太早（历史数据不足）

**处理方式**：
```python
if not weekly_news or len(weekly_news) == 0:
    # 这是正常情况，不是错误
    weekly_news = []  # 返回空列表
```

**注意**：空新闻数据是允许的，不会中断处理流程。下游的 sentiment agent 需要能够处理空新闻的情况。

### 3. API 返回数据格式异常

**可能原因**：
- API 响应格式变化
- 某些字段缺失（如 `datetime`, `headline`, `summary`）

**处理方式**：
```python
try:
    weekly_news = [
        {
            "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
            "headline": n.get('headline', ''),  # 使用 .get() 提供默认值
            "summary": n.get('summary', ''),
        } for n in weekly_news if 'datetime' in n  # 只处理包含必要字段的新闻
    ]
except KeyError as e:
    print(f"Warning: Missing fields in news data: {e}")
    weekly_news = []
```

### 4. 股票代码映射失败

**可能原因**：
- gvkey 在价格数据中不存在
- 股票已退市或代码变更

**处理方式**：
```python
ticker = gvkey_to_ticker.get(gvkey)
if ticker is None:
    print(f"Warning: No ticker found for gvkey {gvkey}, skipping...")
    continue  # 跳过该股票，继续处理其他股票
```

### 5. 股票价格数据获取失败

**可能原因**：
- yfinance 无法获取该股票数据
- 股票代码无效
- 时间段内没有交易数据

**处理方式**：
```python
try:
    news_data = self.fetch_all_data(ticker, curday, n_weeks=n_weeks)
except ValueError as e:
    # 股票数据下载失败
    print(f"Warning: Stock data unavailable for {ticker}: {e}")
    continue  # 跳过该股票
```

## 异常处理层次

### 第一层：API 调用级别

在 `get_news()` 方法中：
- **捕获所有异常**：使用 `try-except Exception`
- **返回空数据**：异常时返回空列表 `[]`
- **记录警告**：打印警告信息，但不中断流程
- **继续处理**：即使某个时间段失败，继续处理其他时间段

### 第二层：数据获取级别

在 `fetch_all_data()` 方法中：
- **股票数据失败**：抛出 `ValueError`，由上层处理
- **新闻数据失败**：已在 `get_news()` 中处理，返回空列表

### 第三层：批量处理级别

在 `get_news_for_selected_stocks()` 方法中：
- **映射失败**：跳过该股票，继续处理其他股票
- **获取失败**：捕获异常，跳过该股票，继续处理其他股票
- **返回部分结果**：即使部分股票失败，也返回成功获取的数据

## 设计原则

### 1. 容错性（Fault Tolerance）

- **不中断流程**：单个股票或时间段失败不应影响其他股票的处理
- **返回部分结果**：即使部分失败，也返回成功获取的数据
- **优雅降级**：没有新闻数据时，返回空列表而不是抛出错误

### 2. 可观测性（Observability）

- **详细日志**：记录所有警告和错误信息
- **区分错误类型**：区分不同类型的异常（网络错误、数据缺失、格式错误等）
- **提供上下文**：日志中包含股票代码、日期范围等上下文信息

### 3. 数据完整性（Data Integrity）

- **统一格式**：即使没有新闻，也返回统一的数据格式（空 JSON 数组）
- **字段验证**：检查必要字段是否存在
- **类型安全**：使用 `.get()` 方法避免 KeyError

## 与原始代码的对比

### FinGPT_Forecaster 的处理方式

```python
# app.py 中的处理（较严格）
if len(weekly_news) == 0:
    raise gr.Error(f"No company news found for symbol {symbol} from finnhub!")
```

**问题**：
- 如果某个时间段没有新闻，会抛出错误，中断整个流程
- 不适合批量处理多个股票

### 我们的处理方式（更宽容）

```python
# 允许空新闻数据
if not weekly_news or len(weekly_news) == 0:
    weekly_news = []  # 正常情况，继续处理
```

**优势**：
- 适合批量处理
- 允许部分时间段没有新闻
- 下游可以处理空新闻的情况

## 下游处理建议

### Sentiment Agent

在 `sentiment_agent.py` 中，应该能够处理空新闻：

```python
def analyze_news(self, news_data):
    """
    分析新闻数据，生成情感信号
    
    Args:
        news_data: DataFrame with News column (可能包含空 JSON 数组)
    """
    for idx, row in news_data.iterrows():
        news_json = row['News']
        news = json.loads(news_json) if news_json else []
        
        if len(news) == 0:
            # 没有新闻时，可以：
            # 1. 使用技术指标作为主要信号
            # 2. 返回中性信号
            # 3. 使用历史平均情感
            return self._get_neutral_signal()
        else:
            # 正常处理新闻
            return self._process_news(news)
```

### Prompt Builder

在 `prompt_builder.py` 中，应该处理空新闻：

```python
def sample_news(news_list, max_news=5):
    """
    从新闻列表中采样
    
    Args:
        news_list: 新闻列表（可能为空）
    """
    if not news_list or len(news_list) == 0:
        return ["No relative news reported."]  # 返回占位文本
    
    # 正常采样逻辑
    ...
```

## 最佳实践

1. **始终检查数据**：在使用新闻数据前，检查是否为空
2. **提供默认值**：没有新闻时，提供合理的默认行为
3. **记录统计信息**：记录有多少股票/时间段没有新闻数据
4. **监控失败率**：如果失败率过高，可能需要检查 API 密钥或网络连接

## 示例：完整的错误处理流程

```python
# 1. 批量获取新闻
results = fetcher.get_news_for_selected_stocks(
    selected_stocks_df,
    price_data_path="...",
    trade_date=trade_date,
    n_weeks=3
)

# 2. 统计结果
total_stocks = len(selected_stocks_df.groupby(['trade_date', 'gvkey']))
successful_stocks = len(results)
failed_stocks = total_stocks - successful_stocks

print(f"Successfully fetched news for {successful_stocks}/{total_stocks} stocks")
print(f"Failed or skipped: {failed_stocks} stocks")

# 3. 检查空新闻
empty_news_count = 0
for (gvkey, td), result in results.items():
    news_data = result['news_data']
    for idx, row in news_data.iterrows():
        news = json.loads(row['News'])
        if len(news) == 0:
            empty_news_count += 1

print(f"Time periods with no news: {empty_news_count}")

# 4. 继续处理（即使有部分失败）
if len(results) > 0:
    # 传递给 sentiment agent
    sentiment_signals = sentiment_agent.process_batch(results)
else:
    print("Warning: No news data available for any stocks!")
```

## 总结

我们的异常处理策略遵循以下原则：

1. **宽容性**：允许空新闻数据，不将其视为错误
2. **容错性**：单个失败不影响整体流程
3. **可观测性**：详细记录所有异常情况
4. **一致性**：统一的数据格式，即使数据为空

这样的设计使得系统能够：
- 处理大量股票时不会因个别失败而中断
- 适应不同股票的数据可用性差异
- 提供清晰的错误信息用于调试和监控


