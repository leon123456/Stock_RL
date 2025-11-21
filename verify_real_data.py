from src.data_processor import DataProcessor
import pandas as pd
import numpy as np

def verify_pipeline():
    print("=== Verifying Real Data Pipeline ===")
    
    dp = DataProcessor()
    
    # 1. Test AkShare Loading
    print("\n[Test 1] Loading Market Data for Tencent (00700)...")
    try:
        # Fetch a small range
        df_norm, emb, raw_prices = dp.load_real_data(
            symbol="00700", 
            start_date="20230101", 
            end_date="20230201"
        )
        
        if not df_norm.empty:
            print(f"SUCCESS: Loaded {len(df_norm)} days of data.")
            print(f"Features: {df_norm.columns.tolist()}")
            print(f"Sample Price: {raw_prices[0]}")
        else:
            print("FAILURE: No data loaded.")
            return
            
    except Exception as e:
        print(f"FAILURE: Error loading data: {e}")
        return

    # 2. Test Dataset Creation
    print("\n[Test 2] Creating Dataset...")
    try:
        dataset = dp.create_dataset(df_norm, emb, raw_prices)
        if len(dataset) > 0:
            print(f"SUCCESS: Created {len(dataset)} samples.")
            print(f"Numerical Input Shape: {dataset[0]['numerical'].shape}")
            print(f"Text Input Shape: {dataset[0]['text'].shape}")
        else:
            print("WARNING: Dataset empty (might be due to seq_len > data_len).")
    except Exception as e:
        print(f"FAILURE: Error creating dataset: {e}")

    # 3. Test News Processor (Mock check)
    print("\n[Test 3] News Processor Check...")
    # We expect a warning if no API key, but no crash
    try:
        # Dummy news
        news_data = {'2023-01-03': ["Tencent buys back shares.", "Tech rally continues."]}
        # This might fail if no API key is present and we try to call API
        # But let's see if it handles it gracefully or if we need to catch it.
        # The current implementation tries to call API.
        # We can skip this if we know we don't have a key, but let's try to see the behavior.
        print("Skipping actual API call to avoid error without key, but checking instantiation.")
        if dp.news_processor:
            print("SUCCESS: NewsProcessor initialized.")
    except Exception as e:
        print(f"FAILURE: NewsProcessor error: {e}")

if __name__ == "__main__":
    verify_pipeline()
