import subprocess
import time

def run_batch():
    # List of stocks to test
    # 00700: Tencent
    # 09992: Pop Mart
    # 01810: Xiaomi
    # 03690: Meituan
    # 09988: Alibaba
    # 01024: Kuaishou
    symbols = ["00700", "09992", "01810", "03690", "09988", "01024"]
    
    print(f"Starting batch run for: {symbols}")
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"Processing {symbol}...")
        print(f"{'='*40}")
        
        # 1. Train
        print(f"Training {symbol}...")
        try:
            subprocess.run(["python3", "train_real.py", symbol], check=True)
        except subprocess.CalledProcessError:
            print(f"Training failed for {symbol}. Skipping...")
            results[symbol] = "Train Failed"
            continue
            
        # 2. Backtest
        print(f"Backtesting {symbol}...")
        try:
            # Capture output to get return
            result = subprocess.run(["python3", "backtest_detailed.py", symbol], capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Parse return from output (simple hack)
            lines = result.stdout.split('\n')
            ret = "N/A"
            for line in lines:
                if "Final Return:" in line:
                    ret = line.split("Final Return:")[1].strip()
            
            results[symbol] = ret
            
        except subprocess.CalledProcessError as e:
            print(f"Backtest failed for {symbol}.")
            print(e.stderr)
            results[symbol] = "Backtest Failed"
            
    print("\n" + "="*40)
    print("Batch Run Complete. Summary:")
    print("="*40)
    print(f"{'Symbol':<10} | {'Return':<15}")
    print("-" * 28)
    for sym, res in results.items():
        print(f"{sym:<10} | {res:<15}")
    print("="*40)

if __name__ == "__main__":
    run_batch()
