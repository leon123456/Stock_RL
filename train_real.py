import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_processor import DataProcessor
from src.environment import HKStockSignalEnv
from src.model import FusionTransformerActorCritic
from src.train import train_ppo
import os

def train_real_data(symbol="00700"):
    print(f"=== Training on Real Data ({symbol}) ===")
    
    # 1. Load Data (2021-2024)
    # symbol is passed as arg
    start_date = "20210101"
    end_date = "20241231"
    
    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    dp = DataProcessor(seq_len=10, embedding_dim=1536, n_features=13)
    
    # Note: We are not passing news_data here, so it will use zero embeddings.
    # In a production system, we would load historical news embeddings here.
    df_norm, embeddings, raw_prices = dp.load_real_data(symbol, start_date, end_date)
    
    if df_norm.empty:
        print("Error: No data loaded.")
        return

    dataset = dp.create_dataset(df_norm, embeddings, raw_prices)
    print(f"Training Dataset: {len(dataset)} samples.")
    
    # 2. Init Environment
    env = HKStockSignalEnv(dataset, initial_balance=100000)
    
    # 3. Init Model
    model = FusionTransformerActorCritic(
        num_features=13,
        seq_len=10,
        text_dim=1536,
        hidden_dim=64,
        action_dim=5
    )
    
    # 4. Train
    print("Starting PPO Training...")
    rewards = train_ppo(env, model, num_episodes=100, lr=1e-4)
    
    # 5. Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = f"models/ppo_actor_critic_{symbol}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 6. Plot Training Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f"Training Rewards ({symbol})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plot_path = f"training_rewards_{symbol}.png"
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_real_data(sys.argv[1])
    else:
        train_real_data("00700")
