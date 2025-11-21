import akshare as ak
import pandas as pd
import numpy as np

class DataLoader:
    """
    Fetches and preprocesses HK stock data using AkShare.
    """
    def __init__(self, symbol="00700"):
        """
        Args:
            symbol (str): HK stock code (e.g., "00700" for Tencent).
        """
        self.symbol = symbol

    def fetch_daily_data(self, start_date="20200101", end_date="20231231", use_cache=True):
        """
        Fetches daily historical data from AkShare or local cache.
        
        Returns:
            pd.DataFrame: DataFrame with columns [date, open, high, low, close, volume] and calculated indicators.
        """
        import os
        cache_dir = "data"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        cache_file = os.path.join(cache_dir, f"{self.symbol}.csv")
        
        df = pd.DataFrame()
        
        # Try loading from cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading {self.symbol} from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Check if cache covers the requested range (simple check: start and end)
            # If cache is insufficient, we might need to re-download.
            # For now, if cache exists, we assume it's the "master" copy. 
            # If user wants fresh data, they should delete cache or pass use_cache=False.
            # Or we can check if requested end_date is beyond cache end_date.
            
            cache_start = df['date'].min()
            cache_end = df['date'].max()
            req_start = pd.to_datetime(start_date)
            req_end = pd.to_datetime(end_date)
            
            if req_end > cache_end:
                print(f"Cache outdated (Ends {cache_end.date()}, Request {req_end.date()}). Re-fetching...")
                df = pd.DataFrame() # Trigger re-fetch
            else:
                # Filter by date
                mask = (df['date'] >= req_start) & (df['date'] <= req_end)
                df = df.loc[mask].copy()
                return df.sort_values('date').reset_index(drop=True)

        if df.empty:
            print(f"Fetching data for {self.symbol} from AkShare ({start_date} to {end_date})...")
            try:
                # AkShare interface for HK daily data
                # Note: adjust_flag="qfq" for forward adjusted prices
                df = ak.stock_hk_daily(symbol=self.symbol, adjust="qfq")
                
                # Filter by date
                df['date'] = pd.to_datetime(df['date'])
                
                # Save full history to cache before filtering? 
                # Better to save what we got. AkShare returns full history usually.
                # Let's save the full processed dataframe to cache.
                
                # Rename columns to match our system
                df = df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
                
                # Calculate Technical Indicators
                df = self._add_indicators(df)
                
                # Sort
                df = df.sort_values('date').reset_index(drop=True)
                
                # Save to cache
                print(f"Saving cache to {cache_file}")
                df.to_csv(cache_file, index=False)
                
                # Now filter for return
                mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
                df = df.loc[mask].copy()
                
                return df.reset_index(drop=True)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                return pd.DataFrame()

    def fetch_hsi_data(self, start_date, end_date):
        """
        Fetches HSI index data.
        """
        try:
            df = ak.stock_hk_index_daily_sina(symbol="HSI")
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            df = df.loc[mask].copy()
            df = df.sort_values('date').reset_index(drop=True)
            # Calculate HSI return
            df['hsi_return'] = df['close'].pct_change().fillna(0)
            return df[['date', 'hsi_return']]
        except Exception as e:
            print(f"Error fetching HSI: {e}")
            return pd.DataFrame()

    def fetch_southbound_data(self, start_date, end_date):
        """
        Fetches Southbound holding data for the stock.
        """
        try:
            df = ak.stock_hsgt_individual_em(symbol=self.symbol)
            # Columns: 持股日期, 当日收盘价, 当日涨跌幅, 持股数量, 持股市值, 持股数量占发行股百分比, ...
            # We want '持股数量' (Share Holding) or '持股数量占发行股百分比' (Holding Ratio)
            # Let's use '持股数量' as a raw measure of smart money accumulation.
            
            df['date'] = pd.to_datetime(df['持股日期'])
            df['southbound_holding'] = df['持股数量']
            
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            df = df.loc[mask].copy()
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate daily change in holding (Net Inflow proxy)
            df['southbound_change'] = df['southbound_holding'].diff().fillna(0)
            
            return df[['date', 'southbound_holding', 'southbound_change']]
        except Exception as e:
            print(f"Error fetching Southbound: {e}")
            return pd.DataFrame()

    def _add_indicators(self, df):
        """
        Adds technical indicators to the DataFrame.
        """
        # 1. Moving Averages
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # Explicit Price vs MA (User Request)
        # Ratio > 1 means price above MA
        df['price_div_ma5'] = df['close'] / df['ma5']
        df['price_div_ma20'] = df['close'] / df['ma20']
        
        # 2. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. Money Flow Proxy: (Close - Open) / (High - Low) * Volume
        denom = df['high'] - df['low']
        denom = denom.replace(0, 1) # Avoid division by zero
        df['money_flow_proxy'] = ((df['close'] - df['open']) / denom) * df['volume']
        
        # 4. Volatility (ATR - Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # 5. MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Explicit MACD Death Cross / Golden Cross state (User Request)
        # We can use macd_hist > 0 as a proxy for Golden Cross state
        # Or explicit boolean
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(float)

        # 6. Bollinger Bands (20, 2)
        df['bb_upper'] = df['ma20'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['ma20'] - 2 * df['close'].rolling(window=20).std()
        # %B Indicator: (Price - Lower) / (Upper - Lower)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 7. KDJ (9, 3, 3)
        low_list = df['low'].rolling(window=9, min_periods=9).min()
        high_list = df['high'].rolling(window=9, min_periods=9).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        
        k_list = []
        d_list = []
        k = 50
        d = 50
        for r in rsv:
            if np.isnan(r):
                k_list.append(np.nan)
                d_list.append(np.nan)
            else:
                k = (2/3) * k + (1/3) * r
                d = (2/3) * d + (1/3) * k
                k_list.append(k)
                d_list.append(d)
        
        df['kdj_k'] = k_list
        df['kdj_d'] = d_list
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # 8. Volume Change (User Request)
        # Volume vs MA5 Volume
        df['ma5_vol'] = df['volume'].rolling(window=5).mean()
        df['vol_div_ma5_vol'] = df['volume'] / df['ma5_vol']
        
        # Fill NaNs
        df = df.bfill().ffill()
        
        return df

if __name__ == "__main__":
    # Test
    loader = DataLoader(symbol="00700")
    df = loader.fetch_daily_data()
    print(df.head())
    print(df.tail())
    print(f"Columns: {df.columns}")
