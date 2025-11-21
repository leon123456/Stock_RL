import os
from src.data_loader import DataLoader

def download_and_cache(symbols, start_date="20200101", end_date="20231231", output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        loader = DataLoader(symbol)
        df = loader.fetch_daily_data(start_date, end_date)
        
        if not df.empty:
            file_path = os.path.join(output_dir, f"{symbol}.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved {symbol} to {file_path} ({len(df)} rows)")
        else:
            print(f"Failed to download {symbol}")

if __name__ == "__main__":
    # 00700: Tencent, 01024: Kuaishou
    symbols = ["00700", "01024"]
    download_and_cache(symbols, start_date="20210101", end_date="20240101")
