import torch
import torch.optim as optim
import numpy as np
from collections import deque

def train_ppo(env, model, num_episodes=50, max_steps=1000, gamma=0.99, lr=1e-3, clip_epsilon=0.2):
    """
    Simplified PPO Training Loop.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track metrics
    all_rewards = []
    
    for episode in range(num_episodes):
        state_obs, _ = env.reset()
        
        # Storage for trajectory
        states_num = []
        states_text = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        done = False
        total_reward = 0
        
        # --- 1. Collect Trajectory ---
        while not done:
            # Prepare input
            num_in = torch.FloatTensor(state_obs['numerical']).unsqueeze(0) # (1, S, F)
            text_in = torch.FloatTensor(state_obs['text']).unsqueeze(0)     # (1, D)
            
            # Get action
            action, action_probs, value = model.get_action(num_in, text_in)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action))
            
            # Step env
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store
            states_num.append(num_in)
            states_text.append(text_in)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            
            state_obs = next_obs
            total_reward += reward
            
        all_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.4f}, Final Portfolio: {info['portfolio_value']:.2f}")
        
        # --- 2. Compute Advantages (GAE or Simple) ---
        # For MVP, let's use simple discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # Normalize
        
        # --- 3. Update Policy ---
        # Convert lists to tensors
        # Note: In real PPO, we do multiple epochs over this batch.
        # Here we do 1 epoch for simplicity of the MVP skeleton.
        
        states_num_batch = torch.cat(states_num)
        states_text_batch = torch.cat(states_text)
        actions_batch = torch.tensor(actions)
        old_log_probs_batch = torch.cat(log_probs).detach()
        returns_batch = returns
        values_batch = torch.cat(values).squeeze()
        
        # Forward pass again
        action_probs, current_values = model(states_num_batch, states_text_batch)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()
        
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        
        # Advantage
        advantage = returns_batch - values_batch.detach()
        
        # Surrogate Loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic Loss
        critic_loss = (returns_batch - current_values.squeeze()).pow(2).mean()
        
        # Total Loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return all_rewards
