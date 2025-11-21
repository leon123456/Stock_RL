import torch
from src.data_processor import DataProcessor
from src.environment import HKStockSignalEnv
from src.model import FusionTransformerActorCritic
from src.train import train_ppo
import matplotlib.pyplot as plt

def main():
    print("Initializing Stock RL System...")
    
    # 1. Data
    print("Generating Mock Data...")
    dp = DataProcessor(seq_len=10, embedding_dim=1536, n_features=9)
    df, embeddings = dp.generate_mock_data(num_days=200) # Small data for quick test
    dataset = dp.create_dataset(df, embeddings)
    print(f"Dataset created with {len(dataset)} samples.")
    
    # 2. Environment
    print("Creating Environment...")
    env = HKStockSignalEnv(dataset, initial_balance=100000)
    
    # 3. Model
    print("Initializing Model...")
    model = FusionTransformerActorCritic(
        num_features=9,
        seq_len=10,
        text_dim=1536,
        hidden_dim=64,
        action_dim=5
    )
    
    # 4. Train
    print("Starting Training...")
    rewards = train_ppo(env, model, num_episodes=10, lr=1e-3)
    
    print("Training Complete.")
    
    # Plot results
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig("training_rewards.png")
        print("Reward plot saved to training_rewards.png")
    except Exception as e:
        print(f"Could not plot: {e}")

if __name__ == "__main__":
    main()
