import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_processor import DataProcessor
from src.environment import HKStockSignalEnv
from src.model import FusionTransformerActorCritic
import pandas as pd

def backtest():
    print("=== Backtesting on 2023 Data (00700) ===")
    
    # 1. Load Validation Data (2023)
    symbol = "00700"
    start_date = "20230101"
    end_date = "20231231"
    
    print(f"Loading validation data for {symbol} from {start_date} to {end_date}...")
    dp = DataProcessor(seq_len=10, embedding_dim=1536, n_features=9)
    df_norm, embeddings, raw_prices = dp.load_real_data(symbol, start_date, end_date)
    
    if df_norm.empty:
        print("Error: No data loaded.")
        return

    dataset = dp.create_dataset(df_norm, embeddings, raw_prices)
    print(f"Validation Dataset: {len(dataset)} samples.")
    
    # 2. Init Environment
    initial_balance = 100000
    env = HKStockSignalEnv(dataset, initial_balance=initial_balance)
    
    # 3. Load Model
    model = FusionTransformerActorCritic(
        num_features=9,
        seq_len=10,
        text_dim=1536,
        hidden_dim=64,
        action_dim=5
    )
    try:
        model.load_state_dict(torch.load("models/ppo_actor_critic_real.pth"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Please train first.")
        return
    
    # 4. Run Backtest
    state_obs, _ = env.reset()
    done = False
    
    portfolio_values = [initial_balance]
    actions = []
    buy_signals = [] # (index, price)
    sell_signals = [] # (index, price)
    
    step = 0
    while not done:
        # Prepare input
        num_in = torch.FloatTensor(state_obs['numerical']).unsqueeze(0)
        text_in = torch.FloatTensor(state_obs['text']).unsqueeze(0)
        
        # Inference (Deterministic for backtest?)
        # Usually we want deterministic action (argmax) for backtest, 
        # but our model outputs probabilities. Let's take the max prob action.
        with torch.no_grad():
            action_probs, _ = model(num_in, text_in)
            action = torch.argmax(action_probs).item()
            
        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state_obs = next_obs
        
        # Track
        current_val = info['portfolio_value']
        portfolio_values.append(current_val)
        actions.append(action)
        
        # Record Buy/Sell for plotting
        # Action 0=0%, 1=25%, 2=50%, 3=75%, 4=100%
        # This is target position. Change in action implies buy/sell.
        # For simplicity, let's just plot the action on a separate subplot.
        
        step += 1
        
    final_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    print(f"Backtest Complete.")
    print(f"Final Balance: {portfolio_values[-1]:.2f}")
    print(f"Return: {final_return:.2f}%")
    
    # 5. Visualize
    # Plot 1: Portfolio Value vs Benchmark (Buy & Hold)
    # Benchmark: Buy at start, hold to end.
    initial_price = raw_prices[10] # seq_len offset
    final_price = raw_prices[10 + len(portfolio_values) - 1]
    benchmark_return = (final_price - initial_price) / initial_price * 100
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values, label='RL Agent')
    # Create benchmark curve (approximate)
    # We need price series aligned with steps.
    # dataset raw_prices starts from t=seq_len
    price_series = raw_prices[10 : 10+len(portfolio_values)]
    # Normalize benchmark to start at initial_balance
    benchmark_curve = (price_series / price_series[0]) * initial_balance
    plt.plot(benchmark_curve, label='Buy & Hold (Benchmark)', linestyle='--')
    plt.title(f"Backtest 2023: Agent ({final_return:.2f}%) vs Benchmark ({benchmark_return:.2f}%)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(actions, marker='o', linestyle='-', markersize=2)
    plt.yticks([0, 1, 2, 3, 4], ['0%', '25%', '50%', '75%', '100%'])
    plt.title("Daily Target Position")
    plt.xlabel("Day")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("backtest_result_2023.png")
    print("Saved backtest_result_2023.png")

if __name__ == "__main__":
    backtest()
