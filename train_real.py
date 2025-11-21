import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_processor import DataProcessor
from src.environment import HKStockSignalEnv
from src.model import FusionTransformerActorCritic
from src.train import train_ppo
import os

def train_real_data():
    print("=== Training on Real Data (00700) ===")
    
    # 1. Load Data (2020-2022)
    symbol = "00700"
    start_date = "20210101"
    end_date = "20241231"
    
    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    dp = DataProcessor(seq_len=10, embedding_dim=1536, n_features=9)
    
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
        num_features=9,
        seq_len=10,
        text_dim=1536,
        hidden_dim=64,
        action_dim=5
    )
    
    # 4. Train
    print("Starting PPO Training...")
    rewards = train_ppo(env, model, num_episodes=50, lr=1e-4)
    
    # 5. Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/ppo_actor_critic_real.pth")
    print("Model saved to models/ppo_actor_critic_real.pth")
    
    # 6. Plot Training Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Training Rewards (Real Data)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_rewards_real.png")
    print("Saved training_rewards_real.png")

if __name__ == "__main__":
    train_real_data()
