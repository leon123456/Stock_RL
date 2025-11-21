import numpy as np
import pandas as pd
import torch
from src.data_loader import DataLoader
from src.news_processor import NewsProcessor

class DataProcessor:
    """
    Handles data loading and preprocessing for the RL agent.
    Supports both mock data and real data (AkShare + Qwen).
    """
    def __init__(self, seq_len=10, embedding_dim=1536, n_features=13):
        """
        Args:
            seq_len (int): Length of the sliding window.
            embedding_dim (int): Dimension of news embedding.
            n_features (int): Number of numerical features.
        """
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_features = n_features
        self.news_processor = NewsProcessor()

    def generate_mock_data(self, num_days=1000):
        """
        Generates dummy data for testing.
        """
        # ... (Keep existing mock logic for backward compatibility if needed, 
        # or just use the logic from previous version. 
        # For brevity, I will re-implement a simple one or just focus on real data)
        
        # Re-implementing simple mock for fallback
        returns = np.random.normal(0, 0.02, num_days)
        price = 100 * np.cumprod(1 + returns)
        volume = np.random.lognormal(10, 1, num_days)
        net_inflow = np.random.normal(0, 1000000, num_days)
        active_buy_ratio = np.random.beta(5, 5, num_days)
        retail_sentiment = np.random.uniform(-1, 1, num_days)
        
        # Mock new features
        macd = np.random.normal(0, 1, num_days)
        kdj_k = np.random.uniform(0, 100, num_days)
        kdj_d = np.random.uniform(0, 100, num_days)
        bb_pct_b = np.random.uniform(0, 1, num_days)
        
        data = {
            'close': price,
            'volume': volume,
            'net_inflow': net_inflow,
            'active_buy_ratio': active_buy_ratio,
            'retail_sentiment': retail_sentiment,
            'macd': macd,
            'kdj_k': kdj_k,
            'kdj_d': kdj_d,
            'bb_pct_b': bb_pct_b
        }
        df = pd.DataFrame(data)
        df_normalized = (df - df.mean()) / df.std()
        embeddings = np.random.normal(0, 1, (num_days, self.embedding_dim))
        
        return df_normalized, embeddings

    def load_real_data(self, symbol="00700", start_date="20200101", end_date="20231231", news_data=None):
        """
        Loads real data using DataLoader and processes news.
        
        Args:
            symbol (str): Stock code.
            start_date (str): Start date.
            end_date (str): End date.
            news_data (dict): Dictionary mapping date (str) to list of news headlines.
                              e.g. {'2023-01-01': ['News 1', 'News 2']}
                              If None, uses zero embeddings.
        """
        # 1. Load Numerical Data
        loader = DataLoader(symbol)
        df = loader.fetch_daily_data(start_date, end_date)
        
        if df.empty:
            print("No data found.")
            return pd.DataFrame(), np.array([]), np.array([])
            
        # Select features for the model
        # We need to map the loader's columns to what the model expects.
        # Model expects: [Close, Volume, Net_Inflow, Active_Buy_Ratio, Retail_Sentiment, MACD, KDJ_K, KDJ_D, BB_Pct_B]
        
        feature_df = pd.DataFrame()
        feature_df['close'] = df['close']
        feature_df['volume'] = df['volume']
        feature_df['net_inflow'] = df['money_flow_proxy']
        feature_df['active_buy_ratio'] = df['rsi']
        feature_df['retail_sentiment'] = df['atr'] # Using ATR as a feature
        
        # New Indicators
        feature_df['macd'] = df['macd_hist'] # Use Histogram
        feature_df['kdj_k'] = df['kdj_k']
        feature_df['kdj_d'] = df['kdj_d']
        feature_df['bb_pct_b'] = df['bb_pct_b']
        
        # New Features (User Request)
        # feature_df['hsi_return'] = df['hsi_return'] # Removed
        # feature_df['southbound_change'] = df['southbound_change'] # Removed
        feature_df['price_div_ma5'] = df['price_div_ma5']
        feature_df['price_div_ma20'] = df['price_div_ma20']
        feature_df['vol_div_ma5_vol'] = df['vol_div_ma5_vol']
        feature_df['macd_bullish'] = df['macd_bullish']
        
        # Total Features: 9 (Old) + 4 (New Technicals) = 13
        
        # Normalize
        df_normalized = (feature_df - feature_df.mean()) / feature_df.std()
        
        # 2. Process News
        num_days = len(df)
        embeddings = np.zeros((num_days, self.embedding_dim))
        
        if news_data:
            print("Processing news data...")
            dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            for i, date_str in enumerate(dates):
                if date_str in news_data:
                    news_list = news_data[date_str]
                    emb = self.news_processor.process_daily_news(news_list)
                    embeddings[i] = emb
                else:
                    # No news for this day, use zero or previous?
                    # Use zero for now
                    pass
        else:
            print("No news data provided, using zero embeddings.")
            
        return df_normalized, embeddings, df['close'].values # Return raw close for Env if needed

    def create_dataset(self, df_normalized, embeddings, raw_prices=None):
        """
        Prepares the data for the environment.
        """
        data = []
        # We need at least seq_len history
        # df_normalized is the FEATURES
        
        # If raw_prices is not provided, try to recover from df (but df is normalized)
        # Ideally raw_prices should be passed.
        
        for t in range(self.seq_len, len(df_normalized)):
            num_window = df_normalized.iloc[t-self.seq_len : t].values
            text_embed = embeddings[t-1]
            
            # Price for reward calculation
            if raw_prices is not None:
                current_price = raw_prices[t-1] # Close of t-1 (Action at t uses this info, executes at t+1 open/close)
                # Wait, Env logic:
                # Step t:
                # Obs: window [t-seq_len : t]
                # Action: a_t
                # Reward: based on price change from t to t+1
                # So we need price at t (execution) and t+1 (next step).
                # The Env uses `dataset[step]['price']`.
                # Let's store the price of the *current* step's day.
                # If we are at step t, we are looking at day t (index t in df).
                # The window ends at t (exclusive in iloc? No, iloc[start:end] excludes end).
                # So window is [t-seq_len, t-seq_len+1, ..., t-1].
                # This is data UP TO day t-1.
                # So we are making decision at start of day t (or end of day t-1).
                # Let's assume we are at day t.
                
                # Let's stick to:
                # Index t in the loop means we are predicting for day t.
                # Input: Data from t-seq_len to t-1.
                # Price: Price of day t.
                current_price = raw_prices[t] 
            else:
                # Fallback to normalized (bad for PnL but works for code)
                current_price = df_normalized.iloc[t]['close']
            
            data.append({
                'numerical': num_window,
                'text': text_embed,
                'price': current_price,
                'index': t
            })
            
        return data

if __name__ == "__main__":
    # Test Real Data Loading
    dp = DataProcessor()
    # Try fetching some real data
    print("Testing Real Data Loading...")
    df_norm, emb, raw_prices = dp.load_real_data(symbol="00700", start_date="20230101", end_date="20230131")
    
    if not df_norm.empty:
        dataset = dp.create_dataset(df_norm, emb, raw_prices)
        print(f"Generated {len(dataset)} samples from real data.")
        print(f"Numerical shape: {dataset[0]['numerical'].shape}")
    else:
        print("Failed to load real data.")
