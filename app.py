import streamlit as st
import requests
import pandas as pd
import hashlib
import hmac
import time
from typing import Optional, List, Dict
from datetime import datetime

# ----------------- Delta Exchange API -----------------
class DeltaExchangeAPI:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.base_url = "https://api.india.delta.exchange"
        self.api_key = api_key
        self.api_secret = api_secret

    def generate_signature(self, secret: str, message: str) -> str:
        message = bytes(message, 'utf-8')
        secret = bytes(secret, 'utf-8')
        return hmac.new(secret, message, hashlib.sha256).hexdigest()

    def get_headers(self, method: str, path: str, query_string: str = "", payload: str = "") -> Dict[str, str]:
        headers = {'Content-Type': 'application/json', 'User-Agent': 'python-delta-client'}
        if self.api_key and self.api_secret:
            timestamp = str(int(time.time()))
            signature_data = method + timestamp + path + query_string + payload
            signature = self.generate_signature(self.api_secret, signature_data)
            headers.update({'api-key': self.api_key, 'timestamp': timestamp, 'signature': signature})
        return headers

    def get_tickers(self, contract_types: Optional[str] = None) -> pd.DataFrame:
        path = "/v2/tickers"
        params = {}
        if contract_types:
            params['contract_types'] = contract_types
        headers = self.get_headers("GET", path)
        try:
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=10)
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

    def get_products(self) -> pd.DataFrame:
        path = "/v2/products"
        headers = self.get_headers("GET", path)
        try:
            response = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return self._format_products_data(data['result'])
            else:
                st.error(f"API Error: {data}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error: {e}")
            return pd.DataFrame()

    def get_candles(self, symbol: str, resolution: str = "1m", limit: int = 3) -> pd.DataFrame:
        path = f"/v2/history/candles"
        params = {"symbol": symbol, "resolution": resolution, "limit": limit}
        headers = self.get_headers("GET", path)
        try:
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success') and data.get('result'):
                return pd.DataFrame(data['result'])
            else:
                return pd.DataFrame()
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_last3_candle_signal(self, symbol: str) -> str:
        df = self.get_candles(symbol, resolution="1m", limit=3)
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

    def _safe_float(self, value) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _format_ticker_data(self, tickers: List[Dict]) -> pd.DataFrame:
        formatted_data = []
        for ticker in tickers:
            close_price = self._safe_float(ticker.get('close', 0))
            open_price = self._safe_float(ticker.get('open', 0))
            volume = self._safe_float(ticker.get('volume', 0))
            oi = self._safe_float(ticker.get('oi', 0))
            oi_value = self._safe_float(ticker.get('oi_value', 0))
            high = self._safe_float(ticker.get('high', 0))
            low = self._safe_float(ticker.get('low', 0))
            mark_price = self._safe_float(ticker.get('mark_price', 0))
            change_pct = ((close_price - open_price)/open_price * 100) if open_price else 0
            change_abs = close_price - open_price if open_price else 0
            quotes = ticker.get('quotes', {}) or {}
            best_bid = self._safe_float(quotes.get('best_bid', 0))
            best_ask = self._safe_float(quotes.get('best_ask', 0))

            # Volume display with K, M, B
            if volume >= 1e9:
                vol_display = f"{volume/1e9:.2f}B"
            elif volume >= 1e6:
                vol_display = f"{volume/1e6:.2f}M"
            elif volume >= 1e3:
                vol_display = f"{volume/1e3:.2f}K"
            else:
                vol_display = f"{volume:.2f}"

            formatted_data.append({
                'Symbol': ticker.get('symbol', 'N/A'),
                'Contract_Type': ticker.get('contract_type', 'N/A'),
                'Last_Price': close_price,
                '24h_Change_%': change_pct,
                '24h_Change_Abs': change_abs,
                '24h_Volume': volume,
                'Volume_Display': vol_display,
                'Open_Interest': oi,
                'OI_Value': oi_value,
                'Mark_Price': mark_price,
                'High_24h': high,
                'Low_24h': low,
                'Open_Price': open_price,
                'Best_Bid': best_bid,
                'Best_Ask': best_ask,
                'Spread': best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0,
                'Underlying_Asset': ticker.get('underlying_asset', {}).get('symbol', 'N/A') if ticker.get('underlying_asset') else 'N/A'
            })
        return pd.DataFrame(formatted_data)

    def _format_products_data(self, products: List[Dict]) -> pd.DataFrame:
        formatted_data = []
        for product in products:
            maker_rate = self._safe_float(product.get('maker_commission_rate', 0))
            taker_rate = self._safe_float(product.get('taker_commission_rate', 0))
            formatted_data.append({
                'Product_ID': product.get('id'),
                'Symbol': product.get('symbol', 'N/A'),
                'Contract_Type': product.get('contract_type', 'N/A'),
                'State': product.get('state', 'N/A'),
                'Underlying_Asset': product.get('underlying_asset', {}).get('symbol', 'N/A') if product.get('underlying_asset') else 'N/A',
                'Maker_Commission_%': maker_rate*100,
                'Taker_Commission_%': taker_rate*100
            })
        return pd.DataFrame(formatted_data)

# ----------------- Streamlit Setup -----------------
st.set_page_config(page_title="Delta Exchange Dashboard", page_icon="📊", layout="wide")
st.title("📊 Delta Exchange Dashboard")

@st.cache_data(ttl=60)
def load_data():
    client = DeltaExchangeAPI()
    tickers = client.get_tickers()
    products = client.get_products()
    if not tickers.empty:
        tickers['Last_3_Candle'] = tickers['Symbol'].apply(lambda s: client.get_last3_candle_signal(s))
    return tickers, products

def apply_filters(df, filters):
    filtered_df = df.copy()
    for column, config in filters.items():
        if config['type'] == 'multiselect' and config['values']:
            filtered_df = filtered_df[filtered_df[column].isin(config['values'])]
        elif config['type'] == 'range':
            if config.get('min') is not None:
                filtered_df = filtered_df[filtered_df[column] >= config['min']]
            if config.get('max') is not None:
                filtered_df = filtered_df[filtered_df[column] <= config['max']]
        elif config['type'] == 'text' and config.get('value'):
            filtered_df = filtered_df[filtered_df[column].str.contains(config['value'], case=False, na=False)]
    return filtered_df


# ----------------- Load Data -----------------
tickers_df, products_df = load_data()

# ----------------- Sidebar Filters -----------------
st.sidebar.header("Filters")
symbol_filter = st.sidebar.text_input("Search Symbol")
contract_filter = st.sidebar.multiselect("Contract Type", tickers_df['Contract_Type'].unique() if not tickers_df.empty else [])
asset_filter = st.sidebar.multiselect("Underlying Asset", tickers_df['Underlying_Asset'].unique() if not tickers_df.empty else [])
price_min, price_max = st.sidebar.slider("Price Range ($)", float(tickers_df['Last_Price'].min()), float(tickers_df['Last_Price'].max()), (float(tickers_df['Last_Price'].min()), float(tickers_df['Last_Price'].max())))
volume_min = st.sidebar.number_input("Minimum Volume ($)", min_value=0.0, value=0.0, step=1000.0)

# Filter by number of digits in 24h_Volume
digit_filter_options = {"All": (0, float("inf")), "1K-10K": (1e3, 1e4), "10K-100K": (1e4, 1e5), "100K-1M": (1e5, 1e6), "1M-10M": (1e6, 1e7), "10M-100M": (1e7, 1e8), "100M-1B": (1e8, 1e9), ">=1B": (1e9, float("inf"))}
digit_filter = st.sidebar.selectbox("24h Volume Digit Range", list(digit_filter_options.keys()))

# Sort by 24h Change %
sort_order = st.sidebar.radio("Sort by 24h Change %", ["None", "Ascending", "Descending"])

# ----------------- Apply Filters -----------------
filters = {}
if symbol_filter:
    filters['Symbol'] = {'type':'text','value':symbol_filter}
if contract_filter:
    filters['Contract_Type'] = {'type':'multiselect','values':contract_filter}
if asset_filter:
    filters['Underlying_Asset'] = {'type':'multiselect','values':asset_filter}
filters['Last_Price'] = {'type':'range','min':price_min,'max':price_max}
if volume_min > 0:
    filters['24h_Volume'] = {'type':'range','min':volume_min,'max':None}

filtered_df = apply_filters(tickers_df, filters)

# Apply digit range filter
if digit_filter != "All":
    min_val, max_val = digit_filter_options[digit_filter]
    filtered_df = filtered_df[(filtered_df['24h_Volume'] >= min_val) & (filtered_df['24h_Volume'] < max_val)]

# Apply sorting
if sort_order == "Ascending":
    filtered_df = filtered_df.sort_values('24h_Change_%', ascending=True)
elif sort_order == "Descending":
    filtered_df = filtered_df.sort_values('24h_Change_%', ascending=False)

# ----------------- Main Data -----------------
st.header("Market Data")
st.dataframe(
    filtered_df[['Symbol','Contract_Type','Last_Price','24h_Change_%','24h_Volume',
                 'Volume_Display','Open_Interest','OI_Value','Underlying_Asset','Last_3_Candle']],
    height=500
)
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
