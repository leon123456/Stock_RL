# Model Architecture: FusionTransformerActorCritic

本文档描述了用于港股交易的强化学习模型架构。该模型旨在融合**数值型市场数据**（价格、成交量、指标）和**文本型新闻数据**（Qwen-max 分析结果）。

## 1. 整体架构图

```mermaid
graph TD
    subgraph Inputs
        Num[数值输入 (B, Seq, 5+N)] --> Linear1[线性层]
        Text[文本输入 (B, 1536)] --> Linear2[线性层]
    end

    subgraph Numerical Branch
        Linear1 --> Transformer[Transformer Encoder / LSTM]
        Transformer --> AvgPool[全局平均池化]
    end

    subgraph Text Branch
        Linear2 --> TextFeat[文本特征向量]
    end

    subgraph Fusion
        AvgPool --> Concat[拼接]
        TextFeat --> Concat
        Concat --> MLP[多层感知机 (Hidden Layer)]
    end

    subgraph Heads
        MLP --> Actor[Actor Head (Action Logits)]
        MLP --> Critic[Critic Head (Value)]
    end
```

## 2. 详细组件

### A. 数值分支 (Numerical Branch)
*   **输入**: 形状为 `(Batch, Seq_Len, Num_Features)` 的时间序列。
    *   `Seq_Len`: 滑动窗口长度（默认 10 天）。
    *   `Num_Features`: 包含开高低收、成交量、资金流、MACD、KDJ 等（默认 5，即将扩展）。
*   **处理**:
    *   通过线性层映射到高维空间（如 128 维）。
    *   通过 **Transformer Encoder** (或 LSTM) 提取时间序列特征，捕捉价格趋势。
    *   输出经过池化，得到一个固定长度的向量。

### B. 文本分支 (Text Branch)
*   **输入**: 形状为 `(Batch, Embedding_Dim)` 的向量。
    *   `Embedding_Dim`: 1536 (来自阿里云 text-embedding-v1)。
    *   来源：Qwen-max 分析新闻后生成的摘要+情绪，再经 Embedding 模型向量化。
*   **处理**:
    *   通过线性层降维/特征提取，使其与数值特征维度匹配。

### C. 融合与输出 (Fusion & Output)
*   **融合**: 将数值特征向量和文本特征向量拼接 (Concatenate)。
*   **Actor Head**: 输出 5 个动作的概率分布 (0%, 25%, 50%, 75%, 100% 持仓)。
*   **Critic Head**: 输出当前状态的价值估计 $V(s)$，用于 PPO 算法计算优势函数。

## 3. 算法
*   **PPO (Proximal Policy Optimization)**: 一种先进的策略梯度算法，通过截断（Clipping）限制策略更新幅度，保证训练稳定性。
