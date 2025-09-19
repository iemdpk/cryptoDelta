import streamlit as st
import pandas as pd
import requests
import hashlib
import hmac
import time
from typing import Optional, List, Dict

# Delta Exchange API Client (unchanged from original)
class DeltaExchangeAPI:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.base_url = "https://api.delta.exchange"
        self.api_key = api_key
        self.api_secret = api_secret
        self.valid_symbols = self._get_valid_symbols()

    def generate_signature(self, secret: str, message: str) -> str:
        message = bytes(message, 'utf-8')
        secret = bytes(secret, 'utf-8')
        hash_hmac = hmac.new(secret, message, hashlib.sha256)
        return hash_hmac.hexdigest()

    def get_headers(self, method: str, path: str, query_string: str = "", payload: str = "") -> Dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'python-delta-client'
        }
        if self.api_key and self.api_secret:
            timestamp = str(int(time.time()))
            signature_data = method + timestamp + path + query_string + payload
            signature = self.generate_signature(self.api_secret, signature_data)
            headers.update({
                'api-key': self.api_key,
                'timestamp': timestamp,
                'signature': signature
            })
        return headers

    def _get_valid_symbols(self) -> List[str]:
        path = "/v2/tickers"
        params = {"contract_types": "perpetual_futures"}
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            query_string = "?" + query_string
        headers = self.get_headers("GET", path, query_string)
        try:
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return [ticker.get('symbol', 'N/A') for ticker in data['result']]
            else:
                st.error(f"API Error fetching symbols: {data}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error fetching symbols: {e}")
            return []

    def get_tickers(self, contract_types: Optional[str] = None) -> pd.DataFrame:
        path = "/v2/tickers"
        params = {}
        if contract_types:
            params['contract_types'] = contract_types
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            query_string = "?" + query_string
        headers = self.get_headers("GET", path, query_string)
        try:
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return self._format_ticker_data(data['result'])
            else:
                st.error(f"API Error: {data}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error: {e}")
            return pd.DataFrame()

    def get_perpetual_data(self) -> pd.DataFrame:
        return self.get_tickers(contract_types="perpetual_futures")

    def get_candles(self, symbol: str, resolution: str = "5m", limit: int = 3) -> List[Dict]:
        if symbol not in self.valid_symbols:
            st.error(f"Invalid symbol: {symbol}. Valid symbols: {self.valid_symbols[:10]}...")
            return []
        path = "/v2/history/candles"
        end_time = int(time.time())
        start_time = end_time - (limit * 5 * 60)
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "limit": limit,
            "start": start_time,
            "end": end_time
        }
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        headers = self.get_headers("GET", path, query_string)
        try:
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data["result"]
            else:
                st.error(f"API Error for {symbol}: {data}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error for {symbol}: {e}")
            return []

    def get_trend_last3(self, symbol: str) -> str:
        candles = self.get_candles(symbol, resolution="5m", limit=3)
        if len(candles) < 3:
            return "NEUTRAL"
        greens = 0
        reds = 0
        for c in candles:
            close = self._safe_float(c.get("close"))
            open_price = self._safe_float(c.get("open"))
            if close > open_price:
                greens += 1
            elif close < open_price:
                reds += 1
        if greens == 3:
            return "UP"
        elif reds == 3:
            return "DOWN"
        else:
            return "NEUTRAL"

    def _format_ticker_data(self, tickers: List[Dict]) -> pd.DataFrame:
        formatted_data = []
        for ticker in tickers:
            try:
                symbol = ticker.get('symbol', 'N/A')
                close_price = self._safe_float(ticker.get('close', 0))
                open_price = self._safe_float(ticker.get('open', 0))
                volume = self._safe_float(ticker.get('volume', 0))
                if open_price:
                    change_pct = ((close_price - open_price) / open_price) * 100
                else:
                    change_pct = 0
                trend = self.get_trend_last3(symbol)
                time.sleep(0.5)
                formatted_data.append({
                    'Name': str(symbol),
                    'Last_Price': f"${close_price:,.4f}",
                    '24h_Change_%': f"{change_pct:+.2f}%",
                    '24h_Volume': f"${volume:,.0f}",
                    'Trend_3x3': trend
                })
            except Exception as e:
                st.error(f"Error processing ticker {symbol}: {e}")
                continue
        return pd.DataFrame(formatted_data)

    def _safe_float(self, value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

# Streamlit App
def main():
    st.title("Delta Exchange Perpetual Futures Data")
    
    # Initialize client
    client = DeltaExchangeAPI()
    
    # Fetch data
    with st.spinner("Fetching perpetual futures data..."):
        df = client.get_perpetual_data()
    
    if df.empty:
        st.error("No data available")
        return
    
    # Create filters
    st.sidebar.header("Filters")
    
    # Digit range filter
    min_digits = st.sidebar.number_input("Minimum Last Price ($)", min_value=0.0, value=0.0, step=0.0001)
    max_digits = st.sidebar.number_input("Maximum Last Price ($)", min_value=0.0, value=float('inf'), step=0.0001)
    
    # Sort order
    sort_order = st.sidebar.selectbox("Sort 24h Change %", ["Ascending", "Descending"])
    
    # Apply filters
    filtered_df = df.copy()
    
    # Convert Last_Price to float for filtering
    filtered_df['Last_Price_num'] = filtered_df['Last_Price'].str.replace('$', '').str.replace(',', '').astype(float)
    filtered_df = filtered_df[
        (filtered_df['Last_Price_num'] >= min_digits) & 
        (filtered_df['Last_Price_num'] <= max_digits)
    ]
    
    # Convert 24h_Change_% to float for sorting
    filtered_df['24h_Change_num'] = filtered_df['24h_Change_%'].str.replace('%', '').astype(float)
    
    # Apply sorting
    if sort_order == "Ascending":
        filtered_df = filtered_df.sort_values('24h_Change_num', ascending=True)
    else:
        filtered_df = filtered_df.sort_values('24h_Change_num', ascending=False)
    
    # Drop temporary numeric columns
    filtered_df = filtered_df.drop(['Last_Price_num', '24h_Change_num'], axis=1)
    
    # Display filtered data
    st.write("### Filtered Perpetual Futures Data")
    st.dataframe(
        filtered_df[['Name', 'Last_Price', '24h_Change_%', '24h_Volume', 'Trend_3x3']],
        use_container_width=True
    )
    
    st.write(f"Total records: {len(filtered_df)}")

if __name__ == "__main__":
    main()
