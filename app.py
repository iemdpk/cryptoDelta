import streamlit as st
import requests
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime

# ----------------- Delta Exchange API -----------------
class DeltaExchangeAPI:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: str = "https://api.india.delta.exchange"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret

    def get_headers(self, method: str = "GET", path: str = "", query_string: str = "", payload: str = "") -> Dict[str, str]:
        return {"Content-Type": "application/json", "User-Agent": "python-delta-client"}

    def _safe_float(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def get_tickers(self, contract_types: Optional[str] = None) -> pd.DataFrame:
        path = "/v2/tickers"
        params = {}
        if contract_types:
            params["contract_types"] = contract_types
        try:
            resp = requests.get(f"{self.base_url}{path}", params=params, headers=self.get_headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and data.get("result"):
                return self._format_ticker_data(data["result"])
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.warning(f"Ticker fetch error: {e}")
            return pd.DataFrame()

    def get_products(self) -> pd.DataFrame:
        path = "/v2/products"
        try:
            resp = requests.get(f"{self.base_url}{path}", headers=self.get_headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and data.get("result"):
                return pd.DataFrame(data["result"])
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.warning(f"Products fetch error: {e}")
            return pd.DataFrame()

    def _format_ticker_data(self, tickers: List[Dict]) -> pd.DataFrame:
        rows = []
        for t in tickers:
            close_price = self._safe_float(t.get("close", 0))
            open_price = self._safe_float(t.get("open", 0))
            volume = self._safe_float(t.get("volume", 0))
            oi = self._safe_float(t.get("oi", 0))
            oi_value = self._safe_float(t.get("oi_value", 0))
            high = self._safe_float(t.get("high", 0))
            low = self._safe_float(t.get("low", 0))
            mark_price = self._safe_float(t.get("mark_price", 0))
            change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0.0
            change_abs = close_price - open_price if open_price else 0.0
            quotes = t.get("quotes") or {}
            best_bid = self._safe_float(quotes.get("best_bid", 0))
            best_ask = self._safe_float(quotes.get("best_ask", 0))
            vol_display = (
                f"{volume/1e9:.2f}B" if volume >= 1e9 else
                f"{volume/1e6:.2f}M" if volume >= 1e6 else
                f"{volume/1e3:.2f}K" if volume >= 1e3 else
                f"{volume:.2f}"
            )
            underlying = None
            if isinstance(t.get("underlying_asset"), dict):
                underlying = t["underlying_asset"].get("symbol")
            rows.append({
                "Symbol": t.get("symbol", "N/A"),
                "Contract_Type": t.get("contract_type", "N/A"),
                "Last_Price": close_price,
                "24h_Change_%": change_pct,
                "24h_Change_Abs": change_abs,
                "24h_Volume": volume,
                "Volume_Display": vol_display,
                "Open_Interest": oi,
                "OI_Value": oi_value,
                "Mark_Price": mark_price,
                "High_24h": high,
                "Low_24h": low,
                "Open_Price": open_price,
                "Best_Bid": best_bid,
                "Best_Ask": best_ask,
                "Spread": (best_ask - best_bid) if best_ask > 0 and best_bid > 0 else 0.0,
                "Underlying_Asset": underlying or "N/A"
            })
        return pd.DataFrame(rows)

    def get_candles(self, symbol: str, resolution: str = "1m", limit: int = 3) -> pd.DataFrame:
        path = "/v2/history/candles"
        params = {"symbol": symbol, "resolution": resolution, "limit": limit}
        try:
            resp = requests.get(f"{self.base_url}{path}", params=params, headers=self.get_headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and data.get("result"):
                df = pd.DataFrame(data["result"])
                if df.empty:
                    return pd.DataFrame()
                # Rename shorthand columns
                rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "time": "timestamp"}
                df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            return pd.DataFrame()
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_last3_candle_signal(self, symbol: str, resolution: str = "1m") -> str:
        df = self.get_candles(symbol, resolution=resolution, limit=3)
        if df.empty or len(df) < 3:
            return "NEUTRAL"

        last3 = df.tail(3)
        colors = []
        for _, row in last3.iterrows():
            if row["close"] > row["open"]:
                colors.append("GREEN")
            elif row["close"] < row["open"]:
                colors.append("RED")
            else:
                colors.append("NEUTRAL")

        if all(c == "GREEN" for c in colors):
            return "UP"
        elif all(c == "RED" for c in colors):
            return "DOWN"
        else:
            return "NEUTRAL"

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Delta Exchange Dashboard", page_icon="📊", layout="wide")
st.title("📊 Delta Exchange Dashboard")

# Sidebar Controls
st.sidebar.header("Controls")
resolution = st.sidebar.selectbox("Candle Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)
only_perpetuals = st.sidebar.checkbox("Only Perpetual Futures", value=True)
contract_types_param = "perpetual_futures" if only_perpetuals else None

# Load Data
@st.cache_data(ttl=30)
def load_tickers_and_products(contract_types_param):
    client = DeltaExchangeAPI()
    tickers = client.get_tickers(contract_types=contract_types_param)
    products = client.get_products()
    return tickers, products

tickers_df, products_df = load_tickers_and_products(contract_types_param)

if tickers_df is None or tickers_df.empty:
    st.error("No tickers data available from Delta Exchange API.")
    st.stop()

# Sidebar Filters
symbol_filter_text = st.sidebar.text_input("Search Symbol")
contract_options = tickers_df["Contract_Type"].unique().tolist()
contract_filter = st.sidebar.multiselect("Contract Type", options=contract_options, default=[])
asset_options = tickers_df["Underlying_Asset"].unique().tolist()
asset_filter = st.sidebar.multiselect("Underlying Asset", options=asset_options, default=[])

# Price filter
price_min = float(tickers_df["Last_Price"].min())
price_max = float(tickers_df["Last_Price"].max())
price_range = st.sidebar.slider("Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))

# Volume filter
vol_min_total = float(tickers_df["24h_Volume"].min())
vol_max_total = float(tickers_df["24h_Volume"].max())
vol_range = st.sidebar.slider("24h Volume Range", min_value=vol_min_total, max_value=vol_max_total, value=(vol_min_total, vol_max_total))

digit_presets = {
    "All": (0.0, float("inf")),
    "1K - 10K": (1e3, 1e4),
    "10K - 100K": (1e4, 1e5),
    "100K - 1M": (1e5, 1e6),
    "1M - 10M": (1e6, 1e7),
    "10M - 100M": (1e7, 1e8),
    "100M - 1B": (1e8, 1e9),
    ">= 1B": (1e9, float("inf"))
}
digit_preset_choice = st.sidebar.selectbox("24h Volume Digit Preset", list(digit_presets.keys()), index=0)

# Signal filter
signal_filter = st.sidebar.multiselect("Signal Filter", ["UP", "DOWN", "NEUTRAL"], default=[])

# Sorting
sort_order = st.sidebar.radio("Sort by 24h Change %", ["None", "Ascending", "Descending"], index=0)

# Apply filters
df = tickers_df.copy()
if symbol_filter_text:
    df = df[df["Symbol"].str.contains(symbol_filter_text, case=False, na=False)]
if contract_filter:
    df = df[df["Contract_Type"].isin(contract_filter)]
if asset_filter:
    df = df[df["Underlying_Asset"].isin(asset_filter)]
df = df[(df["Last_Price"] >= price_range[0]) & (df["Last_Price"] <= price_range[1])]

preset_min, preset_max = digit_presets[digit_preset_choice]
df = df[(df["24h_Volume"] >= max(vol_range[0], preset_min)) & (df["24h_Volume"] <= min(vol_range[1], preset_max))]

if sort_order == "Ascending":
    df = df.sort_values("24h_Change_%", ascending=True)
elif sort_order == "Descending":
    df = df.sort_values("24h_Change_%", ascending=False)

# Fetch signals
client = DeltaExchangeAPI()
st.info("Fetching last-3-candle signals...")
signals = [client.get_last3_candle_signal(sym, resolution=resolution) for sym in df["Symbol"]]
df["Last_3_Candle"] = signals

# Apply signal filter
if signal_filter:
    df = df[df["Last_3_Candle"].isin(signal_filter)]

# Display
display_cols = ["Symbol", "Contract_Type", "Last_Price", "24h_Change_%", "24h_Volume",
                "Volume_Display", "Open_Interest", "OI_Value", "Underlying_Asset", "Last_3_Candle"]
df = df[display_cols]

# Color code signals
def highlight_signal(val):
    if val == "UP":
        return "background-color: lightgreen; color: black"
    elif val == "DOWN":
        return "background-color: salmon; color: black"
    else:
        return "background-color: lightgrey; color: black"

st.header("Market Data")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.dataframe(df.style.applymap(highlight_signal, subset=["Last_3_Candle"]), height=600)
