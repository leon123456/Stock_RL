import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HKStockSignalEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment simulates a swing trading strategy for HK stocks.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, dataset, initial_balance=100000, transaction_cost=0.002):
        super(HKStockSignalEnv, self).__init__()
        
        self.dataset = dataset
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Define Action Space: Discrete(5)
        # 0: 0%, 1: 25%, 2: 50%, 3: 75%, 4: 100%
        self.action_space = spaces.Discrete(5)
        self.position_map = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
        
        # Define Observation Space
        # We have two parts: Numerical (seq_len, n_features) and Text (embedding_dim)
        # Gym spaces usually prefer a single box or dict. Let's use Dict.
        
        # Infer shapes from dataset
        sample = dataset[0]
        self.num_shape = sample['numerical'].shape
        self.text_shape = sample['text'].shape
        
        self.observation_space = spaces.Dict({
            "numerical": spaces.Box(low=-np.inf, high=np.inf, shape=self.num_shape, dtype=np.float32),
            "text": spaces.Box(low=-np.inf, high=np.inf, shape=self.text_shape, dtype=np.float32)
        })
        
        self.current_step = 0
        self.current_position = 0.0 # Current holding percentage
        self.portfolio_value = initial_balance
        self.cash = initial_balance
        self.holdings = 0 # Number of shares
        
        # Track history for rendering/analysis
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_position = 0.0
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.holdings = 0
        self.history = []
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # 1. Get current data (Day t)
        current_data = self.dataset[self.current_step]
        # current_price = current_data['price'] # This is Close of Day t
        
        # 2. Execute Action (T+1 Logic)
        # The action decided at Day t is executed at Day t+1 Open.
        # For simplicity in this MVP, we will approximate:
        # We use Day t+1 Close to calculate PnL for the step, assuming we entered at Day t+1 Open.
        # Wait, standard RL usually steps from t to t+1.
        # Let's define:
        # State s_t -> Action a_t -> Reward r_t+1 -> State s_t+1
        
        # If we are at step t, we see window ending at t.
        # We take action a_t.
        # This action sets the target position for tomorrow.
        
        # Check if we are at the end
        if self.current_step >= len(self.dataset) - 1:
            terminated = True
            truncated = False
            return self._get_obs(), 0, terminated, truncated, self._get_info()
            
        # Move to t+1 to calculate reward
        next_step = self.current_step + 1
        next_data = self.dataset[next_step]
        
        # Price evolution
        # We assume we trade at the "price" provided in dataset.
        # If dataset['price'] is Close price:
        # We rebalance at this price (Simplification of T+1 Open, or assume we trade at Close t for T+0).
        # STRICT T+1 REQUIREMENT:
        # "Action decided at timestep t (using Close data of day t) is executed at the Open Price of day t+1."
        # Since our mock data only has 'close', let's assume Open_t+1 ~= Close_t for now, 
        # OR we just use the price of next step as the execution price.
        
        # Let's stick to the standard:
        # We change position to `target_pct` at `current_price` (Close t).
        # This effectively simulates T+0 or "Next Open is same as Current Close".
        # To be more precise with T+1, we would need Open prices.
        # Given the constraints, we will proceed with rebalancing at the current step's price 
        # but the PnL is realized in the next step.
        
        current_price = current_data['price']
        next_price = next_data['price']
        
        target_pct = self.position_map[action]
        
        # Calculate Transaction Cost
        # Change in position value
        target_value = self.portfolio_value * target_pct
        current_holding_value = self.holdings * current_price
        
        trade_value = abs(target_value - current_holding_value)
        cost = trade_value * self.transaction_cost
        
        # Update Portfolio
        # We are rebalancing to target_pct
        self.cash = self.portfolio_value - target_value - cost
        self.holdings = target_value / current_price
        
        # Update Portfolio Value for next step
        # Value at t+1 = Cash + Holdings * Price_t+1
        new_portfolio_value = self.cash + (self.holdings * next_price)
        
        # Calculate Reward v2.0 (Stable Trend + Volatility Penalty)
        
        # 1. Trend Reward (Lookahead 5 days)
        lookahead = 5
        if self.current_step + lookahead < len(self.dataset):
            future_data = self.dataset[self.current_step + lookahead]
            future_price = future_data['price']
            # Calculate percentage change over next 5 days
            trend_return = (future_price - current_price) / current_price
        else:
            # End of episode, fallback to daily return
            trend_return = (next_price - current_price) / current_price
            
        # Reward is proportional to our position alignment with the trend
        # If we are 100% long (1.0) and price goes up 5%, reward is 0.05
        # If we are 0% long (0.0) and price goes up 5%, reward is 0.0
        r_trend = self.current_position * trend_return
        
        # 2. Volatility Penalty
        # Calculate std of recent daily returns (e.g., last 10 days)
        # We need to track daily portfolio returns
        daily_ret = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.history.append(daily_ret)
        
        window_size = 10
        if len(self.history) > window_size:
            recent_vol = np.std(self.history[-window_size:])
        else:
            recent_vol = 0.0
            
        lambda_vol = 0.1 # Risk aversion coefficient
        r_vol = lambda_vol * recent_vol
        
        # 3. Transaction Cost Penalty (Already in PnL, but we can add explicit penalty if needed)
        # For now, let's rely on the fact that 'daily_ret' (used for history) includes cost.
        # But r_trend does NOT include cost.
        # Let's subtract cost from the reward explicitly to discourage churning.
        # Cost ratio relative to portfolio value
        cost_penalty = cost / self.portfolio_value
        
        # Total Reward
        # We combine the long-term trend signal with the immediate cost and volatility risk
        reward = r_trend - r_vol - cost_penalty
        
        # Scaling (Optional, PPO likes small rewards)
        reward = reward * 10 # Scale up slightly if returns are very small (e.g. 0.001)
        
        # Update state
        self.portfolio_value = new_portfolio_value
        self.current_position = target_pct
        self.current_step += 1
        
        terminated = False
        truncated = False
        if self.current_step >= len(self.dataset) - 1:
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Return the observation for the current step
        # If we are done, return the last valid one
        idx = min(self.current_step, len(self.dataset) - 1)
        data = self.dataset[idx]
        return {
            "numerical": data['numerical'].astype(np.float32),
            "text": data['text'].astype(np.float32)
        }

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "position": self.current_position,
            "step": self.current_step
        }
