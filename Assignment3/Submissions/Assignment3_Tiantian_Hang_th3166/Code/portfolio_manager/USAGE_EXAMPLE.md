# Portfolio Manager 使用示例

## 新逻辑说明

新的 Portfolio Manager 实现了以下逻辑：

1. **Sentiment 作为过滤器**: 如果股票的 sentiment `predicted_return < -0.025`，则不投资该股票
2. **Technical 提供权重**: 对通过过滤的股票，使用 Technical Agent 的均值-方差优化权重
3. **无权重融合**: 不再使用 sentiment weight + technical weight 的融合方式

## 使用示例

### 示例 1: 基本使用

```python
import pandas as pd
from portfolio_manager.portfolio_manager import PortfolioManager
from signal_generator.technical_agent_optimized import TechnicalAgentOptimized

# 1. 初始化 Portfolio Manager
portfolio_manager = PortfolioManager(
    sentiment_threshold=-0.025,  # 过滤阈值
    use_optimization=True
)

# 2. 准备 Sentiment 数据（DataFrame 格式）
sentiment_df = pd.DataFrame({
    'gvkey': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    'predicted_return': [0.05, -0.03, 0.02, 0.01],  # AMZN 会被过滤（-0.03 < -0.025）
    'confidence': [0.8, 0.7, 0.9, 0.6]
})

# 3. 准备 Technical 数据（使用 TechnicalAgentOptimized）
technical_agent = TechnicalAgentOptimized(use_optimization=True)

# 准备技术指标数据
tech_data_dict = {
    'AAPL': {'rsi': 35, 'macd': 0.5, 'cci': -50, 'adx': 30},
    'MSFT': {'rsi': 65, 'macd': -0.2, 'cci': 80, 'adx': 20},
    'GOOGL': {'rsi': 45, 'macd': 0.1, 'cci': 10, 'adx': 25},
    'AMZN': {'rsi': 55, 'macd': 0.3, 'cci': 20, 'adx': 22}
}

# 准备历史收益率数据（至少 252 个交易日）
historical_returns = pd.DataFrame(
    # ... 从实际数据加载
    index=pd.date_range('2023-01-01', periods=252),
    columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
)

# 4. 使用 Technical Agent 生成优化权重
technical_result = technical_agent.analyze_portfolio(
    tech_data_dict, 
    historical_returns
)
# technical_result 包含:
# {
#     'weights': {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.4, 'AMZN': 0.1},
#     'confidence': 0.75,
#     'sharpe_ratio': 1.2,
#     ...
# }

# 5. 使用 Portfolio Manager 生成最终权重
result = portfolio_manager.make_decision(
    sentiment_result=sentiment_df,
    technical_result=technical_result
)

# 6. 获取结果
print("Final Weights:", result['weights'])
# 输出: {'AAPL': 0.35, 'MSFT': 0.0, 'GOOGL': 0.65}
# 注意: MSFT 被过滤（predicted_return = -0.03 < -0.025）
#       AMZN 也被过滤（predicted_return = -0.03 < -0.025）
#       权重已重新归一化

print("Filtered Stocks:", result['filtered_stocks'])
# 输出: ['MSFT', 'AMZN']

print("Sharpe Ratio:", result['sharpe_ratio'])
# 输出: 1.2

print("Reasoning:", result['reasoning'])
# 输出: "Filtered 2 stocks with sentiment predicted_return < -0.025. 
#        Applied mean-variance optimized weights to 2 stocks. 
#        Portfolio Sharpe Ratio: 1.200"
```

### 示例 2: 使用 Dict 格式的 Sentiment 数据

```python
# Sentiment 数据可以是 dict 格式
sentiment_dict = {
    'AAPL': {'predicted_return': 0.05, 'confidence': 0.8},
    'MSFT': {'predicted_return': -0.03, 'confidence': 0.7},
    'GOOGL': {'predicted_return': 0.02, 'confidence': 0.9}
}

result = portfolio_manager.make_decision(
    sentiment_result=sentiment_dict,
    technical_result=technical_result
)
```

### 示例 3: 获取权重和过滤股票

```python
# 直接获取权重
weights = portfolio_manager.get_portfolio_weights(
    sentiment_result=sentiment_df,
    technical_result=technical_result
)
# 返回: {'AAPL': 0.35, 'GOOGL': 0.65}

# 获取被过滤的股票
filtered = portfolio_manager.get_filtered_stocks(
    sentiment_result=sentiment_df,
    technical_result=technical_result
)
# 返回: ['MSFT', 'AMZN']
```

### 示例 4: 在回测中使用

```python
def run_backtest_with_new_logic(price_data, selected_stocks, sentiment_data):
    """
    在回测中使用新的 Portfolio Manager 逻辑
    """
    portfolio_manager = PortfolioManager(sentiment_threshold=-0.025)
    technical_agent = TechnicalAgentOptimized(use_optimization=True)
    
    results = []
    
    for trade_date in selected_stocks['trade_date'].unique():
        # 获取当前交易日的股票
        current_stocks = selected_stocks[selected_stocks['trade_date'] == trade_date]
        
        # 获取 Sentiment 数据
        sentiment_df = sentiment_data[sentiment_data['trade_date'] == trade_date]
        
        # 计算技术指标
        tech_data_dict = calculate_technical_indicators(price_data, current_stocks, trade_date)
        
        # 计算历史收益率
        historical_returns = calculate_historical_returns(price_data, current_stocks, trade_date)
        
        # 生成 Technical 权重
        technical_result = technical_agent.analyze_portfolio(tech_data_dict, historical_returns)
        
        # 生成最终权重
        portfolio_result = portfolio_manager.make_decision(
            sentiment_result=sentiment_df,
            technical_result=technical_result
        )
        
        # 保存结果
        for gvkey, weight in portfolio_result['weights'].items():
            results.append({
                'trade_date': trade_date,
                'gvkey': gvkey,
                'weight': weight,
                'filtered': gvkey in portfolio_result['filtered_stocks']
            })
    
    return pd.DataFrame(results)
```

## 数据格式要求

### Sentiment Result 格式

支持三种格式：

1. **DataFrame**:
```python
pd.DataFrame({
    'gvkey': ['AAPL', 'MSFT'],
    'predicted_return': [0.05, -0.03],
    'confidence': [0.8, 0.7]
})
```

2. **Dict (推荐)**:
```python
{
    'AAPL': {'predicted_return': 0.05, 'confidence': 0.8},
    'MSFT': {'predicted_return': -0.03, 'confidence': 0.7}
}
```

3. **Dict with 'stocks' key**:
```python
{
    'stocks': [
        {'gvkey': 'AAPL', 'predicted_return': 0.05, 'confidence': 0.8},
        {'gvkey': 'MSFT', 'predicted_return': -0.03, 'confidence': 0.7}
    ]
}
```

### Technical Result 格式

必须包含 `weights` 键：

```python
{
    'weights': {
        'AAPL': 0.3,
        'MSFT': 0.2,
        'GOOGL': 0.4,
        'AMZN': 0.1
    },
    'confidence': 0.75,
    'sharpe_ratio': 1.2,
    'expected_return': 0.15,
    'volatility': 0.12
}
```

## 关键变化

### 旧逻辑（已删除）

```python
# 旧逻辑：权重融合
w_sentiment = 0.35
w_technical = 0.65
weighted_score = sentiment_signal * w_sentiment + technical_signal * w_technical
```

### 新逻辑

```python
# 新逻辑：Sentiment 过滤 + Technical 权重
# 1. 过滤
valid_stocks = [s for s in stocks if sentiment[s]['predicted_return'] >= -0.025]

# 2. 应用 Technical 权重
weights = technical_result['weights']
final_weights = {s: weights[s] for s in valid_stocks}

# 3. 重新归一化
total = sum(final_weights.values())
final_weights = {s: w/total for s, w in final_weights.items()}
```

## 注意事项

1. **Sentiment Threshold**: 默认值为 -0.025，可以通过 `PortfolioManager(sentiment_threshold=...)` 修改

2. **权重归一化**: 过滤后的权重会自动重新归一化，确保总和为 1

3. **空结果处理**: 如果所有股票都被过滤，返回空的权重字典

4. **Technical 权重要求**: `technical_result` 必须包含 `weights` 键，使用 `TechnicalAgentOptimized.analyze_portfolio()` 生成

5. **数据对齐**: 确保 Sentiment 和 Technical 数据中的 `gvkey` 一致

## 错误处理

```python
try:
    result = portfolio_manager.make_decision(sentiment_result, technical_result)
except ValueError as e:
    print(f"Error: {e}")
    # 常见错误:
    # - technical_result 缺少 'weights' 键
    # - sentiment_result 格式不正确
    # - 缺少 'predicted_return' 字段
```
