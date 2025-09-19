import streamlit as st
import requests
import pandas as pd
import hashlib
import hmac
import time
from typing import Optional
from datetime import datetime

# ----------------- Delta Exchange API -----------------
class DeltaExchangeAPI:
    def __init__(self):
        self.base_url = "https://api.delta.exchange"

    def get_headers(self, method: str, path: str, body: str = "") -> dict:
        return {}

    def get_tickers(self) -> pd.DataFrame:
        path = "/v2/tickers"
        try:
            response = requests.get(f"{self.base_url}{path}", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success') and data.get('result'):
                return pd.DataFrame(data['result'])
            return pd.DataFrame()
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_products(self) -> pd.DataFrame:
        path = "/v2/products"
        try:
            response = requests.get(f"{self.base_url}{path}", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success') and data.get('result'):
                return pd.DataFrame(data['result'])
            return pd.DataFrame()
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_candles(self, symbol: str, resolution: str = "1m", limit: int = 3) -> pd.DataFrame:
        path = f"/v2/history/candles"
        params = {"symbol": symbol, "resolution": resolution, "limit": limit}
        try:
            response = requests.get(f"{self.base_url}{path}", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success') and data.get('result'):
                return pd.DataFrame(data['result'])
            else:
                return pd.DataFrame()
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_last3_candle_signal(self, symbol: str, resolution: str = "1m") -> str:
        df = self.get_candles(symbol, resolution=resolution, limit=3)
        if df.empty or len(df) < 3:
            return "NEUTRAL"
        
        colors = []
        for _, row in df.iterrows():
            if row['close'] > row['open']:
                colors.append("GREEN")
            elif row['close'] < row['open']:
                colors.append("RED")
            else:
                colors.append("NEUTRAL")

        if all(c == "GREEN" for c in colors):
            return "UP"
        elif all(c == "RED" for c in colors):
            return "DOWN"
        else:
            return "NEUTRAL"


# ----------------- Data Loader -----------------
@st.cache_data(ttl=60)
def load_data(resolution: str):
    client = DeltaExchangeAPI()
    tickers = client.get_tickers()
    products = client.get_products()

    if tickers.empty or products.empty:
        return pd.DataFrame()

    df = tickers.merge(products[['symbol', 'contract_type', 'underlying_asset']], 
                       left_on="symbol", right_on="symbol", how="left")

    df.rename(columns={
        "symbol": "Symbol",
        "contract_type": "Contract_Type",
        "underlying_asset": "Underlying_Asset",
        "close": "Last_Price",
        "change_percent": "24h_Change_%",
        "volume": "24h_Volume",
        "open_interest": "Open_Interest"
    }, inplace=True)

    # Add OI Value & Volume Display
    df["OI_Value"] = df["Open_Interest"] * df["Last_Price"]
    df["Volume_Display"] = df["24h_Volume"].apply(lambda x: f"{x:,.0f}")

    # Add last 3 candle signal
    client = DeltaExchangeAPI()
    df["Signal"] = df["Symbol"].apply(lambda s: client.get_last3_candle_signal(s, resolution=resolution))

    return df


# ----------------- Streamlit Dashboard -----------------
st.set_page_config(page_title="Delta Exchange Dashboard", layout="wide")
st.title("📊 Delta Exchange Market Dashboard")

# Sidebar Filters
st.sidebar.header("Filters")

# Timeframe selector
resolution = st.sidebar.selectbox("Candle Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)

# Load Data
df = load_data(resolution)

if not df.empty:
    # Symbol filter
    symbols = st.sidebar.multiselect("Select Symbols", options=df["Symbol"].unique(), default=[])
    if symbols:
        df = df[df["Symbol"].isin(symbols)]

    # Contract type filter
    contract_types = st.sidebar.multiselect("Select Contract Types", options=df["Contract_Type"].unique(), default=[])
    if contract_types:
        df = df[df["Contract_Type"].isin(contract_types)]

    # Underlying asset filter
    assets = st.sidebar.multiselect("Select Underlying Assets", options=df["Underlying_Asset"].unique(), default=[])
    if assets:
        df = df[df["Underlying_Asset"].isin(assets)]

    # 24h Change sort
    sort_order = st.sidebar.radio("Sort by 24h Change %", ["None", "Ascending", "Descending"], index=0)
    if sort_order == "Ascending":
        df = df.sort_values(by="24h_Change_%", ascending=True)
    elif sort_order == "Descending":
        df = df.sort_values(by="24h_Change_%", ascending=False)

    # 24h Volume Range Filter
    min_vol, max_vol = st.sidebar.slider("24h Volume Range", 
                                         min_value=int(df["24h_Volume"].min()), 
                                         max_value=int(df["24h_Volume"].max()), 
                                         value=(int(df["24h_Volume"].min()), int(df["24h_Volume"].max())))
    df = df[(df["24h_Volume"] >= min_vol) & (df["24h_Volume"] <= max_vol)]

    # Signal Filter (UP/DOWN/NEUTRAL)
    signals = st.sidebar.multiselect("Select Signals", options=["UP", "DOWN", "NEUTRAL"], default=[])
    if signals:
        df = df[df["Signal"].isin(signals)]

    # Show Table
    st.subheader("Market Data")
    st.dataframe(
        df[['Symbol','Contract_Type','Last_Price','24h_Change_%','24h_Volume',
            'Volume_Display','Open_Interest','OI_Value','Underlying_Asset','Signal']],
        height=600
    )
else:
    st.error("⚠️ Failed to load data from Delta Exchange API.")
