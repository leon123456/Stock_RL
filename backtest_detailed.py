import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processor import DataProcessor
from src.environment import HKStockSignalEnv
from src.model import FusionTransformerActorCritic
import os

def backtest_detailed(symbol="00700"):
    print(f"=== Detailed Backtesting on 2025 Data ({symbol}) ===")
    
    # 1. Load Validation Data (2025)
    # symbol passed as arg
    start_date = "20250101"
    end_date = "20251120"
    
    print(f"Loading validation data for {symbol} from {start_date} to {end_date}...")
    dp = DataProcessor(seq_len=10, embedding_dim=1536, n_features=13)
    
    # We need to get the dates back. 
    # load_real_data returns (df_norm, embeddings, raw_prices)
    # But we also need the original DF to get dates.
    # Let's call DataLoader directly to get dates, or modify DataProcessor to return dates.
    # For now, let's just fetch dates separately to avoid changing DP interface too much if not needed.
    # Actually, DataProcessor.load_real_data filters dates.
    # Let's just re-fetch using DataLoader to get the date index.
    from src.data_loader import DataLoader
    loader = DataLoader(symbol)
    raw_df = loader.fetch_daily_data(start_date, end_date)
    
    # Ensure alignment
    # DP does: df = loader.fetch... then normalize.
    # So raw_df should align with dataset if we skip seq_len.
    
    df_norm, embeddings, raw_prices = dp.load_real_data(symbol, start_date, end_date)
    
    if df_norm.empty:
        print("Error: No data loaded.")
        return

    dataset = dp.create_dataset(df_norm, embeddings, raw_prices)
    print(f"Validation Dataset: {len(dataset)} samples.")
    
    # Align dates
    # dataset starts from index = seq_len
    # So dataset[0] corresponds to raw_df.iloc[seq_len]
    # Let's verify lengths
    # len(dataset) = len(raw_df) - seq_len
    
    valid_dates = raw_df['date'].iloc[dp.seq_len:].reset_index(drop=True)
    
    # 2. Init Environment
    initial_balance = 100000
    env = HKStockSignalEnv(dataset, initial_balance=initial_balance)
    
    # 3. Load Model
    model = FusionTransformerActorCritic(
        num_features=13,
        seq_len=10,
        text_dim=1536,
        hidden_dim=64,
        action_dim=5
    )
    
    model_path = f"models/ppo_actor_critic_{symbol}.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train first.")
        return
    
    # 4. Run Backtest
    state_obs, _ = env.reset()
    done = False
    
    logs = []
    
    step = 0
    while not done:
        # Prepare input
        num_in = torch.FloatTensor(state_obs['numerical']).unsqueeze(0)
        text_in = torch.FloatTensor(state_obs['text']).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            action_probs, _ = model(num_in, text_in)
            action = torch.argmax(action_probs).item()
            
        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state_obs = next_obs
        
        # Log details
        current_date = valid_dates.iloc[step] if step < len(valid_dates) else "End"
        current_price = dataset[step]['price']
        
        # Action map
        action_map = {0: "0% (Clear)", 1: "25% (Light)", 2: "50% (Medium)", 3: "75% (Heavy)", 4: "100% (Full)"}
        action_str = action_map.get(action, str(action))
        
        log_entry = {
            "Date": current_date,
            "Price": current_price,
            "Action": action_str,
            "Portfolio Value": info['portfolio_value'],
            "Position": info['position'],
            "Daily Reward": reward
        }
        logs.append(log_entry)
        
        step += 1
        
    # 5. Save Logs
    log_df = pd.DataFrame(logs)
    log_file = f"backtest_logs_{symbol}_2025.csv"
    log_df.to_csv(log_file, index=False)
    print(f"Detailed logs saved to {log_file}")
    
    # Print sample
    print("\n=== Log Sample (First 5 Days) ===")
    print(log_df.head())
    
    # Calculate Final Return
    final_balance = logs[-1]['Portfolio Value']
    ret = (final_balance - initial_balance) / initial_balance * 100
    print(f"\nFinal Return: {ret:.2f}%")
    
    # 6. Visualization
    print("Generating Backtest Plot...")
    dates = pd.to_datetime(log_df['Date'])
    portfolio_values = log_df['Portfolio Value']
    prices = log_df['Price']
    
    # Extract actions for plotting
    # Action string format: "0% (Clear)" -> 0
    # We need to map back or just use the raw action if we had it.
    # The log has "Action" column as string.
    # Let's reverse map or just parse.
    # Map: "0% (Clear)" -> 0, "25% (Light)" -> 1, etc.
    action_map_rev = {
        "0% (Clear)": 0, 
        "25% (Light)": 1, 
        "50% (Medium)": 2, 
        "75% (Heavy)": 3, 
        "100% (Full)": 4
    }
    # Handle potential raw numbers if any
    actions = log_df['Action'].apply(lambda x: action_map_rev.get(x, 0))
    
    # Normalize for comparison
    norm_portfolio = portfolio_values / initial_balance
    norm_price = prices / prices.iloc[0]
    
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Portfolio vs Benchmark
    plt.subplot(2, 1, 1)
    plt.plot(dates, norm_portfolio, label=f'Strategy (Return: {ret:.2f}%)', color='blue', linewidth=2)
    plt.plot(dates, norm_price, label=f'Benchmark (Buy & Hold)', color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Backtest Result: {symbol} ({start_date} - {end_date})")
    plt.ylabel("Normalized Value (Start=1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Daily Target Position
    plt.subplot(2, 1, 2)
    plt.plot(dates, actions, marker='o', linestyle='-', markersize=3, color='orange', label='Target Position')
    plt.yticks([0, 1, 2, 3, 4], ['0%', '25%', '50%', '75%', '100%'])
    plt.ylabel("Position")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    plot_file = f"backtest_result_{symbol}_2025.png"
    plt.savefig(plot_file)
    print(f"Backtest plot saved to {plot_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        backtest_detailed(sys.argv[1])
    else:
        backtest_detailed("00700")
