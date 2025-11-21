
以下是对我们之前对话的精华总结，以及为你准备好的最终 Prompt。

第一部分：对话核心精华总结 (Context Summary)
我们并不是在做一个盲目预测股价涨跌的玩具，而是在构建一个基于多模态数据（Level-2 资金流 + 新闻情绪）的日级别波段交易信号系统。
 * 核心目标 (Goal): 开发一个 MVP（最小可行性产品），输出日级别的仓位建议信号（0% - 100% 分 5 档）。
 * 策略风格 (Style): 波段交易 (Swing Trading)。不进行日内高频操作，旨在捕捉数天到数周的趋势。
 * 数据输入 (Inputs):
   * 数值端: 港股 Level-2 逐笔成交（Tick）聚合后的日线资金流因子（如：主力净流入、主动买入占比、散户情绪）。
   * 文本端: 财经新闻标题/摘要，经过 FinBERT 处理后的情绪向量。
   * 基本面: PE/PB 等估值数据（低频）。
 * 模型架构 (Model): 双塔结构。一塔处理数值时序（Transformer/LSTM），一塔处理文本（Pre-computed BERT Embeddings），中间通过融合层（Fusion Layer）结合，最后输入 RL Policy 网络（PPO）。
 * 环境逻辑 (Environment):
   * 动作: 离散动作 [0, 1, 2, 3, 4] 对应仓位 [0%, 25%, 50%, 75%, 100%]。
   * 规则: 严格遵守 T+1 逻辑（今晚出信号，明早开盘价成交）。
   * 奖励: 夏普比率导向（收益/波动率），并包含换手率惩罚（防止频繁调仓）。


第二部分：给 AI 编程大模型的 Prompt (Copy & Paste)
请复制以下内容，直接发送给你的 AI 编程编辑器:
# Role
You are an expert Quantitative Developer and AI Researcher specializing in Reinforcement Learning (RL) for Financial Markets. You are proficient in Python, PyTorch, Gym (or Gymnasium), and Transformer architectures.

# Project Objective
I need you to construct the code architecture for a "Multi-Modal RL Agent for Hong Kong Stock Swing Trading" (MVP version).
The system's goal is NOT to execute high-frequency trades, but to output a **Daily Position Signal** (0% to 100%) based on aggregated Level-2 Order Flow data and News Sentiment.

# System Architecture & Requirements

## 1. Data Preprocessing Layer (Feature Engineering)
The input data consists of raw Level-2 Ticks and News Text. You need to design a preprocessing module that:
* **Numerical Feeds:** detailed HKEX Level-2 data. We need to aggregate ticks into Daily Bars with specific "Smart Money" factors.
    * *Factors to implement:* Large Order Net Inflow, Active Buying Ratio, Broker Queue Imbalance (Institutional vs. Retail).
* **Text Feeds:** Financial news headlines.
    * *Processing:* Assume we use a pre-trained `FinBERT` to generate a fixed-size embedding vector for each day.
* **Data Structure:** The final input to the RL agent for each timestep `t` should be a vector combining: `[Market_Factors_t, News_Embedding_t, Fundamentals_t]`.

## 2. The RL Environment (`HKStockSignalEnv`)
Create a custom Gymnasium Environment (`gym.Env`) that simulates the trading process:
* **State Space:** A sliding window (e.g., past 10 days) of the combined feature vectors.
* **Action Space:** Discrete(5).
    * 0: Empty (0% Position)
    * 1: Light (25%)
    * 2: Half (50%)
    * 3: Heavy (75%)
    * 4: Full (100%)
* **Execution Logic (Crucial):**
    * Strict **T+1 Execution**: The Action decided at timestep `t` (using Close data of day `t`) is executed at the **Open Price** of day `t+1`.
    * **Transaction Costs:** Deduct a simulated cost (e.g., 0.2%) whenever the position changes to penalize high turnover.
* **Reward Function:**
    * Primary: Portfolio Return.
    * Penalty: Volatility penalty (to approximate Sharpe Ratio) and Turnover penalty.

## 3. Model Architecture (`FusionTransformerActorCritic`)
Design a Neural Network compatible with PPO (Proximal Policy Optimization):
* **Branch A (Time-Series):** A small Transformer Encoder or LSTM to process the numerical market factors history.
* **Branch B (Text):** A Linear layer to process the pre-computed FinBERT embeddings.
* **Fusion Layer:** Use a Gated Mechanism or Concatenation to combine features from Branch A and B.
* **Heads:**
    * Actor: Outputs logits for the 5 discrete actions.
    * Critic: Outputs the Value function V(s).

## 4. Training Loop
* Use a standard PPO implementation (you can scaffold this or use `stable-baselines3` logic but tailored for the custom model).
* Implement a `Train-Validation-Test` split logic to avoid look-ahead bias.

# Deliverables
Please generate the Python project structure and the core code files. Specifically:
1.  `data_processor.py`: Mock functions showing how to aggregate Level-2 ticks into daily factors.
2.  `environment.py`: The `HKStockSignalEnv` class with the T+1 logic.
3.  `model.py`: The Dual-Branch PyTorch model.
4.  `main.py`: The main entry point to initialize the env and model (just the skeleton).

# Coding Constraints
* Use `torch` for the model.
* Use `gymnasium` for the environment.
* Keep the code modular and well-commented.
* For the `data_processor`, since we don't have the real CSV files yet, create dummy data generators so the code is runnable.

第三部分：给你的开发建议
把上面的 Prompt 发送给编辑器后，它应该会为你生成一个完整的项目骨架。拿到代码后，你的工作流程如下：
 * 先跑通 Mock Data: 编辑器生成的代码里会有“模拟数据生成器”。直接运行 main.py，确保整个 RL 训练循环（环境 Reset -> 模型 Forward -> 动作 -> 奖励 -> 反向传播）能跑通，没有报错。
 * 替换数据源: 打开 data_processor.py，找到模拟数据生成的部分，将其替换为你从富途/券商 API 下载的真实 CSV 读取逻辑。
 * 微调 Reward: 观察第一轮训练结果。如果 AI 总是空仓（Action 0），说明手续费惩罚太重；如果 AI 总是满仓（Action 4），说明风险惩罚太轻。去 environment.py 里调整 Reward 函数的参数。


祝你开发顺利！这套系统一旦跑通，就是非常硬核的量化资产了。
