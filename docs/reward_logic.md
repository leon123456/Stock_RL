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

为了解决 v1.0 的风险无感问题，计划引入夏普比率（Sharpe Ratio）或索提诺比率（Sortino Ratio）的变体作为奖励。

### 拟定公式
$$ R_t = Return_t - \lambda \times (Volatility_t) $$

或者使用 **差分夏普比率 (Differential Sharpe Ratio)**。

### 待办事项
- [ ] 在 `src/environment.py` 中实现波动率惩罚。
- [ ] 引入最大回撤（Max Drawdown）作为终止条件或强惩罚项。
