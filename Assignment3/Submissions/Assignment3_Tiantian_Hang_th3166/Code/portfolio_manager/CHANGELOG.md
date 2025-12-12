# Portfolio Manager 改动说明

## 版本更新

### 新版本逻辑（已实施）

**日期**: 2024

**主要改动**:
1. ✅ 使用均值-方差优化计算权重（通过 TechnicalAgentOptimized）
2. ✅ 考虑风险（协方差矩阵）
3. ✅ 基于技术指标预测收益率（替代离散信号）
4. ✅ 新逻辑：Sentiment 作为过滤器，Technical 提供权重

---

## 详细改动

### 1. 删除的旧方法

以下方法已被删除（不再使用权重融合）：
- ❌ `_weighted_fusion()` - 旧的权重融合方法
- ❌ `_consensus_fusion()` - 共识融合方法
- ❌ `_adaptive_fusion()` - 自适应融合方法
- ❌ `_majority_fusion()` - 多数投票方法

### 2. 新增的核心方法

#### `make_decision()`
- **新逻辑**: 
  1. 使用 Sentiment 过滤股票（predicted_return < -0.025）
  2. 对剩余股票应用 Technical 的均值-方差优化权重
  3. 重新归一化权重

#### `_normalize_sentiment_result()`
- **功能**: 将不同格式的 Sentiment 数据统一为 dict 格式
- **支持格式**:
  - DataFrame: `pd.DataFrame({'gvkey': [...], 'predicted_return': [...]})`
  - Dict: `{gvkey: {'predicted_return': float, ...}}`
  - Dict with 'stocks': `{'stocks': [{gvkey: ..., predicted_return: ...}]}`

#### `_filter_by_sentiment()`
- **功能**: 根据 Sentiment predicted_return 阈值过滤股票
- **逻辑**: `predicted_return < -0.025` 的股票被排除

#### `_apply_technical_weights()`
- **功能**: 对通过过滤的股票应用 Technical 权重并重新归一化

### 3. 新增的辅助方法

#### `get_portfolio_weights()`
- **功能**: 直接获取最终权重字典
- **返回**: `{gvkey: weight}`

#### `get_filtered_stocks()`
- **功能**: 获取被过滤的股票列表
- **返回**: `[gvkey, ...]`

---

## 接口变化

### 旧接口（已废弃）

```python
# 旧接口
portfolio_manager = PortfolioManager(fusion_strategy='weighted')
result = portfolio_manager.make_decision(
    sentiment_result={'signal': 1, 'confidence': 0.8},
    technical_result={'signal': 0.5, 'confidence': 0.7}
)
# 返回: {'action': 'BUY', 'weights': {'sentiment': 0.35, 'technical': 0.65}}
```

### 新接口

```python
# 新接口
portfolio_manager = PortfolioManager(sentiment_threshold=-0.025)
result = portfolio_manager.make_decision(
    sentiment_result={
        'AAPL': {'predicted_return': 0.05},
        'MSFT': {'predicted_return': -0.03}  # 会被过滤
    },
    technical_result={
        'weights': {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.5},
        'sharpe_ratio': 1.2,
        'confidence': 0.75
    }
)
# 返回: {
#     'weights': {'AAPL': 0.375, 'GOOGL': 0.625},  # MSFT 被过滤，权重重新归一化
#     'filtered_stocks': ['MSFT'],
#     'sharpe_ratio': 1.2,
#     'confidence': 0.75
# }
```

---

## 数据格式要求

### Sentiment Result

**必须包含**: `predicted_return` 字段

**支持格式**:
1. DataFrame: `pd.DataFrame({'gvkey': [...], 'predicted_return': [...]})`
2. Dict: `{gvkey: {'predicted_return': float, ...}}`

### Technical Result

**必须包含**: `weights` 键（由 `TechnicalAgentOptimized.analyze_portfolio()` 生成）

**格式**:
```python
{
    'weights': {gvkey: weight, ...},
    'confidence': float,
    'sharpe_ratio': float,  # 可选
    'expected_return': float,  # 可选
    'volatility': float  # 可选
}
```

---

## 使用 TechnicalAgentOptimized

新的 Portfolio Manager 需要配合 `TechnicalAgentOptimized` 使用：

```python
from signal_generator.technical_agent_optimized import TechnicalAgentOptimized

# 初始化
technical_agent = TechnicalAgentOptimized(use_optimization=True)

# 准备数据
tech_data_dict = {
    'AAPL': {'rsi': 35, 'macd': 0.5, 'cci': -50, 'adx': 30},
    # ...
}

historical_returns = pd.DataFrame(...)  # 至少 252 个交易日

# 生成优化权重
technical_result = technical_agent.analyze_portfolio(
    tech_data_dict, 
    historical_returns
)
```

---

## 迁移指南

### 步骤 1: 更新导入

```python
# 旧代码
from portfolio_manager.portfolio_manager import PortfolioManager

# 新代码（相同，但用法不同）
from portfolio_manager.portfolio_manager import PortfolioManager
from signal_generator.technical_agent_optimized import TechnicalAgentOptimized
```

### 步骤 2: 更新初始化

```python
# 旧代码
portfolio_manager = PortfolioManager(fusion_strategy='weighted')

# 新代码
portfolio_manager = PortfolioManager(sentiment_threshold=-0.025)
```

### 步骤 3: 更新数据准备

```python
# 旧代码
sentiment_result = {'signal': 1, 'confidence': 0.8}
technical_result = {'signal': 0.5, 'confidence': 0.7}

# 新代码
sentiment_result = {
    'AAPL': {'predicted_return': 0.05},
    'MSFT': {'predicted_return': -0.03}
}

# 使用 TechnicalAgentOptimized 生成权重
technical_agent = TechnicalAgentOptimized(use_optimization=True)
technical_result = technical_agent.analyze_portfolio(
    tech_data_dict, 
    historical_returns
)
```

### 步骤 4: 更新结果处理

```python
# 旧代码
result = portfolio_manager.make_decision(sentiment_result, technical_result)
action = result['action']  # 'BUY', 'SELL', 'HOLD'
weights = result['weights']  # {'sentiment': 0.35, 'technical': 0.65}

# 新代码
result = portfolio_manager.make_decision(sentiment_result, technical_result)
weights = result['weights']  # {'AAPL': 0.375, 'GOOGL': 0.625}
filtered = result['filtered_stocks']  # ['MSFT']
sharpe = result['sharpe_ratio']  # 1.2
```

---

## 优势

### 1. 更科学的权重分配
- ✅ 使用均值-方差优化（Markowitz 理论）
- ✅ 考虑风险（协方差矩阵）
- ✅ 最大化 Sharpe Ratio

### 2. 更清晰的角色分工
- ✅ Sentiment: 作为过滤器，排除高风险股票
- ✅ Technical: 提供优化权重

### 3. 更好的风险控制
- ✅ 自动排除 sentiment predicted_return < -0.025 的股票
- ✅ 权重自动归一化，确保总和为 1

---

## 注意事项

1. **依赖关系**: 需要安装 `pypfopt` 库
   ```bash
   pip install pypfopt
   ```

2. **数据要求**: 
   - 历史收益率数据：至少 252 个交易日
   - Sentiment 数据：必须包含 `predicted_return` 字段

3. **性能**: 均值-方差优化需要计算协方差矩阵，可能较慢（几秒到几分钟）

4. **向后兼容**: 旧代码需要更新才能使用新逻辑

---

## 测试建议

1. **单元测试**: 测试过滤逻辑和权重归一化
2. **集成测试**: 测试与 TechnicalAgentOptimized 的集成
3. **回测对比**: 对比新旧逻辑的回测结果

---

## 相关文件

- `portfolio_manager.py` - 主要实现
- `technical_agent_optimized.py` - Technical Agent 优化版本
- `USAGE_EXAMPLE.md` - 使用示例
- `CHANGELOG.md` - 本文件
