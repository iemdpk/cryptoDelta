import streamlit as st
import requests
import pandas as pd
import hashlib, hmac, time
from typing import Optional, List, Dict
from datetime import datetime

# ----------------- Delta Exchange API -----------------
class DeltaExchangeAPI:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.base_url = "https://api.india.delta.exchange"
        self.api_key = api_key
        self.api_secret = api_secret

    def generate_signature(self, secret: str, message: str) -> str:
        return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    def get_headers(self, method: str, path: str, query_string: str = "", payload: str = "") -> Dict[str, str]:
        headers = {'Content-Type': 'application/json', 'User-Agent': 'python-delta-client'}
        if self.api_key and self.api_secret:
            ts = str(int(time.time()))
            sig_data = method + ts + path + query_string + payload
            signature = self.generate_signature(self.api_secret, sig_data)
            headers.update({'api-key': self.api_key, 'timestamp': ts, 'signature': signature})
        return headers

    def get_tickers(self) -> pd.DataFrame:
        path = "/v2/tickers"
        headers = self.get_headers("GET", path)
        try:
            r = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                return self._format_tickers(data["result"])
        except Exception as e:
            st.error(f"Tickers Error: {e}")
        return pd.DataFrame()

    def get_candles(self, symbol: str, resolution: str = "1m", limit: int = 3) -> List[Dict]:
        path = "/v2/candles"
        params = {"symbol": symbol, "resolution": resolution, "limit": limit}
        try:
            r = requests.get(f"{self.base_url}{path}", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                return data["result"]
        except Exception as e:
            st.error(f"Candles Error ({symbol}): {e}")
        return []

    def _format_tickers(self, tickers: List[Dict]) -> pd.DataFrame:
        out = []
        for t in tickers:
            try:
                close_price = float(t.get("close", 0))
                open_price = float(t.get("open", 0))
                volume = float(t.get("volume", 0))
                change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0
                out.append({
                    "Symbol": t.get("symbol", "N/A"),
                    "Contract_Type": t.get("contract_type", "N/A"),
                    "Last_Price": close_price,
                    "24h_Change_%": change_pct,
                    "24h_Volume": volume,
                    "Underlying_Asset": t.get("underlying_asset", {}).get("symbol", "N/A"),
                })
            except:
                pass
        return pd.DataFrame(out)

# ----------------- Streamlit -----------------
st.set_page_config(page_title="Delta Exchange Dashboard", page_icon="📊", layout="wide")
st.title("📊 Delta Exchange Dashboard")

@st.cache_data(ttl=60)
def load_data():
    api = DeltaExchangeAPI()
    df = api.get_tickers()
    return df

df = load_data()

# Add Candle Trend column
api = DeltaExchangeAPI()
def get_trend(symbol: str) -> str:
    candles = api.get_candles(symbol, resolution="1m", limit=3)
    if len(candles) < 3:
        return "NEUTRAL"
    moves = []
    for c in candles:
        o, cl = float(c["open"]), float(c["close"])
        if cl > o:
            moves.append("UP")
        elif cl < o:
            moves.append("DOWN")
        else:
            moves.append("NEUTRAL")
    if all(m == "UP" for m in moves):
        return "UP"
    elif all(m == "DOWN" for m in moves):
        return "DOWN"
    else:
        return "NEUTRAL"

if not df.empty:
    df["Candle_Trend"] = df["Symbol"].apply(get_trend)

# ----------------- Sidebar Filters -----------------
st.sidebar.header("Filters")

# Volume filter (min/max)
vol_min, vol_max = st.sidebar.slider(
    "24h Volume Range",
    min_value=float(df["24h_Volume"].min()) if not df.empty else 0,
    max_value=float(df["24h_Volume"].max()) if not df.empty else 1_000_000,
    value=(
        float(df["24h_Volume"].min()) if not df.empty else 0,
        float(df["24h_Volume"].max()) if not df.empty else 1_000_000,
    ),
)

# Candle Trend filter
trend_filter = st.sidebar.multiselect("Candle Trend", ["UP", "DOWN", "NEUTRAL"])

# Apply filters
filtered = df.copy()
filtered = filtered[(filtered["24h_Volume"] >= vol_min) & (filtered["24h_Volume"] <= vol_max)]
if trend_filter:
    filtered = filtered[filtered["Candle_Trend"].isin(trend_filter)]

# ----------------- Display -----------------
st.subheader("Market Data")
st.dataframe(filtered, height=600)
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
