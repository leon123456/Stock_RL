# 部署与测试指南 (Deployment Guide)

本文档为团队成员提供完整的环境配置、模型训练、回测和分析流程。

---

## 📋 目录
1. [环境准备](#环境准备)
2. [单股票训练与回测](#单股票训练与回测)
3. [批量训练与回测](#批量训练与回测)
4. [决策分析](#决策分析)
5. [常见问题](#常见问题)

---

## 🛠️ 环境准备

### 1. 克隆项目
```bash
git clone <repository_url>
cd Stock_RL
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

**核心依赖**:
- `torch` (PyTorch)
- `gymnasium` (强化学习环境)
- `akshare` (港股数据获取)
- `pandas`, `numpy`, `matplotlib` (数据处理与可视化)
- `python-dotenv` (环境变量管理)

### 3. 配置环境变量 (可选)
如果需要使用新闻情绪分析（当前未启用），创建 `.env` 文件：
```bash
DASHSCOPE_API_KEY=your_qwen_api_key_here
```

> **注意**: 当前版本的模型**不依赖新闻数据**，可以跳过此步骤。

---

## 🎯 单股票训练与回测

### 训练模型
```bash
python3 train_real.py <股票代码>
```

**示例**:
```bash
python3 train_real.py 01024  # 训练快手 (01024)
```

**训练参数**:
- **训练数据**: 2021-01-01 至 2024-12-31
- **训练轮数**: 100 episodes (在 `train_real.py` 中可修改)
- **特征数量**: 13 (9 个基础技术指标 + 4 个显式技术指标)
- **奖励函数**: v2.1 (5 日趋势 + 下行波动惩罚)

**输出**:
- 模型文件: `models/ppo_actor_critic_<股票代码>.pth`
- 训练曲线: `training_rewards_<股票代码>.png`
- 数据缓存: `data/<股票代码>.csv`

### 回测模型
```bash
python3 backtest_detailed.py <股票代码>
```

**示例**:
```bash
python3 backtest_detailed.py 01024  # 回测快手 (01024)
```

**回测参数**:
- **回测数据**: 2025-01-01 至 2025-11-20 (当前日期)
- **初始资金**: 100,000 元
- **交易成本**: 0.2% (双边)

**输出**:
- 回测日志: `backtest_logs_<股票代码>_2025.csv`
- 回测图表: `backtest_result_<股票代码>_2025.png`
- 终端输出: 最终收益率、前 5 日交易记录

**日志字段说明**:
| 字段 | 说明 |
| :--- | :--- |
| `Date` | 交易日期 |
| `Price` | 收盘价 |
| `Action` | 模型决策 (0% Clear, 25% Light, 50% Medium, 75% Heavy, 100% Full) |
| `Portfolio Value` | 组合净值 |
| `Position` | 实际仓位比例 (0.0-1.0) |
| `Daily Reward` | 当日奖励值 (训练时使用，回测时仅供参考) |

---

## 📦 批量训练与回测

### 批量处理多只股票
```bash
python3 run_batch.py
```

**默认股票列表** (在 `run_batch.py` 中定义):
```python
symbols = ["00700", "09992", "01810", "01024", "03690", "09988"]
```

**流程**:
1. 依次训练每只股票 (100 episodes)
2. 自动回测每只股票 (2025 年数据)
3. 生成汇总表格

**输出示例**:
```
========================================
Batch Run Complete. Summary:
========================================
Symbol     | Return         
----------------------------
00700      | 47.51%         
09992      | 0.00%          
01810      | -18.32%        
03690      | -9.76%         
09988      | 69.15%         
01024      | 144.64%        
========================================
```

**自定义股票列表**:
编辑 `run_batch.py` 第 5-13 行：
```python
symbols = [
    "00700",  # 腾讯控股
    "01024",  # 快手
    # 添加更多股票代码...
]
```

---

## 🔍 决策分析

### 分析特定日期的决策因子
```bash
python3 analyze_decision.py <股票代码> <日期>
```

**示例**:
```bash
python3 analyze_decision.py 01024 2025-06-26
```

**输出**:
- 目标日期前后 5 天的技术指标
- 关键指标解读 (RSI、MACD、Price/MA 等)
- 5 日后价格预测 (用于理解模型的趋势判断)

**示例输出**:
```
================================================================================
Key Observations on 2025-06-26:
================================================================================

Close Price: 61.35
MA5: 58.63
MA20: 57.33
Price/MA5: 1.0463 (Above MA5)
Price/MA20: 1.0701 (Above MA20)
RSI: 56.54 (Neutral)
MACD Bullish: True (Golden Cross: Yes)
Volume/MA5_Vol: 1.6021 (High)

5-Day Future Price: 61.50 (+0.24%)
Trend Signal: Bullish
```

---

## ❓ 常见问题

### 1. 数据缓存机制
**问题**: 每次训练都要重新下载数据吗？

**答案**: 不需要。数据会缓存到 `data/<股票代码>.csv`。如果需要强制刷新数据，删除对应的 CSV 文件：
```bash
rm data/01024.csv
```

### 2. 训练时间过长
**问题**: 100 轮训练需要多久？

**答案**: 取决于 CPU 性能，通常 5-10 分钟/股票。如果需要加速，可以减少训练轮数：
```python
# train_real.py 第 46 行
num_episodes=50  # 从 100 改成 50
```

### 3. 回测结果为 0.00%
**问题**: 模型全程空仓，收益率为 0%。

**可能原因**:
- 模型过于保守（波动率惩罚过重）
- 技术指标冲突（如 RSI 超买 + MACD 金叉）
- 训练数据不足或过拟合

**解决方案**:
1. 增加训练轮数 (100 → 200)
2. 降低波动率惩罚系数 (`lambda_vol` 从 0.1 改成 0.05)
3. 简化特征（去掉部分显式技术指标）

### 4. 模型文件丢失
**问题**: 运行回测时提示 `Model file not found`。

**解决方案**:
```bash
# 先训练模型
python3 train_real.py 01024

# 再回测
python3 backtest_detailed.py 01024
```

### 5. 修改回测日期范围
**问题**: 如何回测其他时间段？

**解决方案**: 编辑 `backtest_detailed.py` 第 15-16 行：
```python
start_date = "20250101"  # 修改起始日期
end_date = "20251120"    # 修改结束日期
```

### 6. 查看训练日志
**问题**: 如何判断模型训练得好不好？

**答案**: 观察终端输出的 `Total Reward` 和 `Final Portfolio`:
- ✅ **好**: Reward 从负数逐渐变成正数，或者震荡上升
- ❌ **坏**: Reward 一直卡在很大的负数 (如 -100)，或者 Final Portfolio 每次都归零

**示例** (好的训练):
```
Episode 1/100, Total Reward: -59.44, Final Portfolio: 36782.61
Episode 50/100, Total Reward: -30.18, Final Portfolio: 10824.38
Episode 100/100, Total Reward: -25.37, Final Portfolio: 20252.15
```

---

## 📊 文件结构

```
Stock_RL/
├── src/
│   ├── data_loader.py       # 数据获取与技术指标计算
│   ├── data_processor.py    # 数据归一化与特征工程
│   ├── environment.py       # 强化学习环境 (Reward 函数在这里)
│   ├── model.py             # Transformer + Actor-Critic 模型
│   └── trainer.py           # PPO 训练器
├── docs/
│   ├── reward_logic.md      # 奖励函数设计文档
│   └── deploy.md            # 本文档
├── data/                    # 数据缓存目录 (自动生成)
├── models/                  # 模型文件目录 (自动生成)
├── train_real.py            # 单股票训练脚本
├── backtest_detailed.py     # 单股票回测脚本
├── run_batch.py             # 批量训练与回测脚本
├── analyze_decision.py      # 决策分析脚本
└── requirements.txt         # Python 依赖
```

---

## 🚀 快速开始 (3 步)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python3 train_real.py 01024

# 3. 回测模型
python3 backtest_detailed.py 01024
```

---

## 📞 联系与支持

如有问题，请联系项目负责人或查看 `docs/reward_logic.md` 了解更多技术细节。
