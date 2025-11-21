# Reward Logic Documentation

本文档记录了强化学习智能体奖励函数的设计逻辑、演变历史及未来规划。

## Current Version: v1.0 (MVP)

### 核心逻辑
目前的奖励函数非常直接，旨在最大化**投资组合的对数收益率**，同时隐式包含了交易成本。

$$ R_t = \frac{V_{t+1} - V_t}{V_t} $$

其中 $V_t$ 是 $t$ 时刻的投资组合总价值（现金 + 持仓市值）。

### 详细计算步骤
1.  **动作执行**: 智能体在 $t$ 时刻决定目标仓位 $P_{target}$。
2.  **交易成本**: 计算从当前仓位调整到目标仓位所需的交易额，扣除手续费（默认 0.2%）。
    $$ Cost = |Position_{new} - Position_{old}| \times Price_t \times 0.002 $$
    $$ V_{after\_trade} = V_t - Cost $$
3.  **市值更新**: 持仓部分随股价变动到 $t+1$ 时刻。
    $$ V_{t+1} = Cash + Holdings \times Price_{t+1} $$
4.  **奖励计算**:
    $$ Reward = \frac{V_{t+1} - V_t}{V_t} $$

### 优缺点分析
*   **优点**: 简单直接，与最终目标（赚钱）完全一致。自动惩罚过度交易（因为交易成本会降低 $V$）。
*   **缺点**:
    *   **风险无感**: 没有惩罚巨大的回撤。智能体可能会为了 1% 的收益去冒 10% 的风险。
    *   **稀疏性**: 虽然每一步都有奖励，但对于长期持有的策略，奖励可能在震荡市中不够明确。

---

## Planned Version: v2.0 (Risk-Adjusted)

为了解决 v1.0 的风险无感问题，我们将引入 **波动率惩罚 (Volatility Penalty)**。

## Planned Version: v2.0 (Stable Trend & Risk-Adjusted)

### 核心思想
1.  **拉长视野**: 用户指出每日变动太快，建议看未来一周（5天）的数据。
### v2.0 奖励函数 (5日趋势 + 下行波动惩罚)

**目标**: 鼓励模型捕捉中期趋势，同时避免过度交易和下行风险。

**公式**:
```
R_total = R_trend - R_vol - R_cost

其中:
R_trend = position * (price_t+5 - price_t) / price_t  # 5日趋势奖励
R_vol = λ * σ_downside(recent_returns)                # 下行波动惩罚
R_cost = transaction_cost / portfolio_value           # 交易成本惩罚
```

**关键改进**:
1. **5日趋势奖励**: 不再只看当日涨跌，而是看未来5天的价格变化。如果模型持有多头仓位且未来5天上涨，则获得正奖励。
2. **下行波动惩罚 (Downside Deviation)**: 
   - 只计算**负收益**的标准差，类似于 Sortino Ratio 的思路
   - 如果最近10天全是正收益，波动率惩罚为 0
   - 如果有负收益，则根据下跌的剧烈程度进行惩罚
   - 这样可以鼓励模型在上涨趋势中保持仓位，而不是因为"涨得太快"而清仓
3. **交易成本**: 显式扣除，避免频繁换仓。

**实现位置**: `src/environment.py` 的 `step()` 方法。
*   **$Volatility$ (波动惩罚)**:
    $$ Volatility = \text{std}(Returns_{t-10...t}) $$
    *   惩罚近期的市场波动或策略波动。

### 实现细节
*   在 `HKStockSignalEnv` 中，`step()` 函数需要访问 `t+5` 的数据。
*   如果 `t+5` 超出边界，则使用剩余天数的收益或 0。
