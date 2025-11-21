import torch
import torch.nn as nn
import numpy as np

class FusionTransformerActorCritic(nn.Module):
    """
    Dual-Branch Network for Multi-Modal RL.
    Branch A: Transformer for Numerical Time-Series.
    Branch B: Linear for Text Embeddings.
    Fusion: Concatenation -> Actor/Critic Heads.
    """
    def __init__(self, 
                 num_features=9, 
                 seq_len=10, 
                 text_dim=1536, 
                 hidden_dim=128, 
                 action_dim=5,
                 nhead=4,
                 num_layers=2):
        super(FusionTransformerActorCritic, self).__init__()
        
        # --- Branch A: Numerical Time-Series (Transformer) ---
        self.num_embedding = nn.Linear(num_features, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Flatten or Pool the transformer output
        # We will take the last time step's output or average
        self.num_head = nn.Linear(hidden_dim * seq_len, hidden_dim) 
        
        # --- Branch B: Text (Linear) ---
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # --- Fusion Layer ---
        # Concatenate [Num_Features, Text_Features]
        fusion_dim = hidden_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Actor Head (Policy) ---
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # --- Critic Head (Value) ---
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, numerical_input, text_input):
        """
        Args:
            numerical_input: (Batch, Seq_Len, Num_Features)
            text_input: (Batch, Text_Dim)
        """
        # Branch A
        x_num = self.num_embedding(numerical_input) # (B, S, H)
        x_num = self.transformer_encoder(x_num)     # (B, S, H)
        x_num = x_num.reshape(x_num.size(0), -1)    # Flatten (B, S*H)
        x_num = torch.relu(self.num_head(x_num))    # (B, H)
        
        # Branch B
        x_text = self.text_encoder(text_input)      # (B, H)
        
        # Fusion
        combined = torch.cat([x_num, x_text], dim=1) # (B, 2H)
        fused = self.fusion_layer(combined)          # (B, H)
        
        # Heads
        action_probs = self.actor(fused)
        value = self.critic(fused)
        
        return action_probs, value

    def get_action(self, numerical_input, text_input, deterministic=False):
        action_probs, value = self.forward(numerical_input, text_input)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
        return action.item(), action_probs, value
