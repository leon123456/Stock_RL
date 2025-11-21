import akshare as ak
import pandas as pd
import datetime

def test_hsi():
    print("Fetching HSI Index Data...")
    try:
        # HSI Index
        # ak.stock_hk_index_daily_sina(symbol="HSI")
        df = ak.stock_hk_index_daily_sina(symbol="HSI")
        print("HSI Data Sample:")
        print(df.tail())
        return True
    except Exception as e:
        print(f"Error fetching HSI: {e}")
        return False

def test_southbound(symbol="00700"):
    print(f"Fetching Southbound Data for {symbol}...")
    try:
        # Correct function seems to be stock_hsgt_hold_stock_em
        # It likely requires "market" and "stock" code.
        # Actually, for HK stocks in Southbound, the code is usually 5 digits.
        # Try stock_hsgt_individual_em
        df = ak.stock_hsgt_individual_em(symbol=symbol)
        print("Southbound Data Sample:")
        print(df.tail())
        return True
    except Exception as e:
        print(f"Error fetching Southbound: {e}")
        return False

if __name__ == "__main__":
    hsi_ok = test_hsi()
    sb_ok = test_southbound("00700")
    
    if hsi_ok and sb_ok:
        print("All tests passed!")
    else:
        print("Some tests failed.")
