"""
Analyze the decision-making factors for a specific stock on a specific date.
This script loads the cached data and displays all features around the decision point.
"""
import pandas as pd
import sys

def analyze_decision(symbol, target_date):
    """
    Load cached data and display features around the target date.
    
    Args:
        symbol (str): Stock symbol (e.g., '01024')
        target_date (str): Target date in YYYY-MM-DD format (e.g., '2025-06-26')
    """
    cache_file = f"data/{symbol}.csv"
    
    try:
        df = pd.read_csv(cache_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the target date
        target_idx = df[df['date'] == target_date].index
        if len(target_idx) == 0:
            print(f"Date {target_date} not found in data.")
            return
        
        target_idx = target_idx[0]
        
        # Display data from 5 days before to 10 days after
        start_idx = max(0, target_idx - 5)
        end_idx = min(len(df), target_idx + 11)
        
        window_df = df.iloc[start_idx:end_idx].copy()
        
        # Select relevant columns
        display_cols = [
            'date', 'close', 'volume',
            'ma5', 'ma20', 'rsi', 'macd', 'macd_signal',
            'price_div_ma5', 'price_div_ma20', 'macd_bullish', 'vol_div_ma5_vol'
        ]
        
        # Filter to only existing columns
        display_cols = [col for col in display_cols if col in window_df.columns]
        
        print(f"\n{'='*80}")
        print(f"Decision Analysis for {symbol} on {target_date}")
        print(f"{'='*80}\n")
        
        # Set pandas display options for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print(window_df[display_cols].to_string(index=False))
        
        print(f"\n{'='*80}")
        print(f"Key Observations on {target_date}:")
        print(f"{'='*80}\n")
        
        target_row = window_df[window_df['date'] == target_date].iloc[0]
        
        print(f"Close Price: {target_row['close']:.2f}")
        if 'ma5' in target_row:
            print(f"MA5: {target_row['ma5']:.2f}")
        if 'ma20' in target_row:
            print(f"MA20: {target_row['ma20']:.2f}")
        if 'price_div_ma5' in target_row:
            print(f"Price/MA5: {target_row['price_div_ma5']:.4f} ({'Above' if target_row['price_div_ma5'] > 1 else 'Below'} MA5)")
        if 'price_div_ma20' in target_row:
            print(f"Price/MA20: {target_row['price_div_ma20']:.4f} ({'Above' if target_row['price_div_ma20'] > 1 else 'Below'} MA20)")
        if 'rsi' in target_row:
            rsi_val = target_row['rsi']
            rsi_status = 'Overbought' if rsi_val > 70 else ('Oversold' if rsi_val < 30 else 'Neutral')
            print(f"RSI: {rsi_val:.2f} ({rsi_status})")
        if 'macd_bullish' in target_row:
            print(f"MACD Bullish: {bool(target_row['macd_bullish'])} (Golden Cross: {'Yes' if target_row['macd_bullish'] else 'No'})")
        if 'vol_div_ma5_vol' in target_row:
            print(f"Volume/MA5_Vol: {target_row['vol_div_ma5_vol']:.4f} ({'High' if target_row['vol_div_ma5_vol'] > 1.5 else 'Normal'})")
        
        # Check next 5 days price movement (for trend reward)
        if target_idx + 5 < len(df):
            future_price = df.iloc[target_idx + 5]['close']
            price_change = (future_price - target_row['close']) / target_row['close']
            print(f"\n5-Day Future Price: {future_price:.2f} ({price_change:+.2%})")
            print(f"Trend Signal: {'Bullish' if price_change > 0 else 'Bearish'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_decision.py <symbol> <date>")
        print("Example: python analyze_decision.py 01024 2025-06-26")
        sys.exit(1)
    
    symbol = sys.argv[1]
    target_date = sys.argv[2]
    
    analyze_decision(symbol, target_date)
