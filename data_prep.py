import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_aligned_data(tickers=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]):
    # Fetch data
    data = yf.download(tickers, start="2020-01-01", end="2025-01-01")['Close']
    data = data.ffill().dropna() # Handle missing values
    
    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    return pd.DataFrame(scaled_data, columns=tickers, index=data.index), scaler

if __name__ == "__main__":
    df, _ = get_aligned_data()
    print("Data Prepared. Shape:", df.shape)
