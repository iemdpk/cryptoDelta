import streamlit as st
import requests
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

# ----------------- Delta Exchange API -----------------
class DeltaExchangeAPI:
    def __init__(self, base_url: str = "https://api.india.delta.exchange"):
        self.base_url = base_url.rstrip("/")

    def get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "User-Agent": "streamlit-delta-client"}

    def _safe_float(self, v) -> float:
        try:
            return float(v)
        except:
            return 0.0

    def get_tickers(self, contract_types: Optional[str] = None) -> pd.DataFrame:
        path = "/v2/tickers"
        params = {}
        if contract_types:
            params["contract_types"] = contract_types
        try:
            r = requests.get(f"{self.base_url}{path}", params=params, headers=self.get_headers(), timeout=10)
            r.raise_for_status()
            js = r.json()
            if js.get("success") and js.get("result"):
                return self._format_ticker_data(js["result"])
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Ticker fetch error: {e}")
            return pd.DataFrame()

    def get_products(self) -> pd.DataFrame:
        try:
            r = requests.get(f"{self.base_url}/v2/products", headers=self.get_headers(), timeout=10)
            r.raise_for_status()
            js = r.json()
            if js.get("success") and js.get("result"):
                return pd.DataFrame(js["result"])
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Products fetch error: {e}")
            return pd.DataFrame()

    def _format_ticker_data(self, tickers: List[Dict]) -> pd.DataFrame:
        rows = []
        for t in tickers:
            close_price = self._safe_float(t.get("close"))
            open_price = self._safe_float(t.get("open"))
            volume = self._safe_float(t.get("volume"))
            oi = self._safe_float(t.get("oi"))
            oi_value = self._safe_float(t.get("oi_value"))
            high = self._safe_float(t.get("high"))
            low = self._safe_float(t.get("low"))
            mark_price = self._safe_float(t.get("mark_price"))
            change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0.0
            quotes = t.get("quotes") or {}
            best_bid = self._safe_float(quotes.get("best_bid"))
            best_ask = self._safe_float(quotes.get("best_ask"))
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
                "Symbol": t.get("symbol"),
                "Contract_Type": t.get("contract_type"),
                "Last_Price": close_price,
                "24h_Change_%": change_pct,
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
                "Spread": (best_ask - best_bid) if best_ask and best_bid else 0.0,
                "Underlying_Asset": underlying or "N/A"
            })
        return pd.DataFrame(rows)

    def get_candles(self, symbol: str, resolution: str = "1m", limit: int = 3) -> pd.DataFrame:
        url = f"{self.base_url}/v2/history/candles"
        params = {"symbol": symbol, "resolution": resolution, "limit": limit}
        try:
            r = requests.get(url, params=params, headers=self.get_headers(), timeout=10)
            r.raise_for_status()
            js = r.json()
            if js.get("success") and js.get("result"):
                return pd.DataFrame(js["result"])
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    def get_last3_candle_signal(self, symbol: str, resolution: str = "1m") -> str:
        df = self.get_candles(symbol, resolution=resolution, limit=3)
        if df.empty or not {"open", "close"}.issubset(df.columns.str.lower()):
            return "NEUTRAL"

        # normalize case
        df.columns = [c.lower() for c in df.columns]
        df = df.tail(3)

        greens = (df["close"] > df["open"]).sum()
        reds = (df["close"] < df["open"]).sum()

        if greens == 3:
            return "UP"
        elif reds == 3:
            return "DOWN"
        return "NEUTRAL"


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Delta Exchange Dashboard", layout="wide")
st.title("📊 Delta Exchange Dashboard")

st.sidebar.header("Controls")
resolution = st.sidebar.selectbox("Candle Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)
only_perpetuals = st.sidebar.checkbox("Only Perpetual Futures", value=True)
contract_types_param = "perpetual_futures" if only_perpetuals else None

client = DeltaExchangeAPI()

@st.cache_data(ttl=30)
def load_data(contract_types_param):
    return client.get_tickers(contract_types_param), client.get_products()

tickers_df, products_df = load_data(contract_types_param)

if tickers_df.empty:
    st.error("No tickers data available.")
    st.stop()

# Sidebar filters
symbol_filter = st.sidebar.text_input("Search Symbol")
contract_filter = st.sidebar.multiselect("Contract Type", tickers_df["Contract_Type"].unique())
asset_filter = st.sidebar.multiselect("Underlying Asset", tickers_df["Underlying_Asset"].unique())

# Volume filter
vol_min, vol_max = float(tickers_df["24h_Volume"].min()), float(tickers_df["24h_Volume"].max())
vol_range = st.sidebar.slider("24h Volume Range", min_value=vol_min, max_value=vol_max, value=(vol_min, vol_max))

digit_presets = {
    "All": (0, float("inf")),
    "1K-10K": (1e3, 1e4),
    "10K-100K": (1e4, 1e5),
    "100K-1M": (1e5, 1e6),
    "1M-10M": (1e6, 1e7),
    "10M-100M": (1e7, 1e8),
    "100M-1B": (1e8, 1e9),
    ">=1B": (1e9, float("inf"))
}
digit_choice = st.sidebar.selectbox("24h Volume Digit Preset", digit_presets.keys())
preset_min, preset_max = digit_presets[digit_choice]

signal_filter = st.sidebar.multiselect("Signal Filter", ["UP", "DOWN", "NEUTRAL"])
sort_order = st.sidebar.radio("Sort 24h Change %", ["None", "Ascending", "Descending"])

# Apply filters
df = tickers_df.copy()
if symbol_filter:
    df = df[df["Symbol"].str.contains(symbol_filter, case=False, na=False)]
if contract_filter:
    df = df[df["Contract_Type"].isin(contract_filter)]
if asset_filter:
    df = df[df["Underlying_Asset"].isin(asset_filter)]
df = df[(df["24h_Volume"] >= vol_range[0]) & (df["24h_Volume"] <= vol_range[1])]
if digit_choice != "All":
    df = df[(df["24h_Volume"] >= preset_min) & (df["24h_Volume"] <= preset_max)]

# Sort
if sort_order == "Ascending":
    df = df.sort_values("24h_Change_%", ascending=True)
elif sort_order == "Descending":
    df = df.sort_values("24h_Change_%", ascending=False)

# Fetch signals
st.info("Fetching signals (one API call per symbol)...")
signals = []
for sym in df["Symbol"]:
    sig = client.get_last3_candle_signal(sym, resolution=resolution)
    signals.append(sig)
df["Last_3_Candle"] = signals

# Signal filter
if signal_filter:
    df = df[df["Last_3_Candle"].isin(signal_filter)]

# Display
st.header("Market Data")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.dataframe(df[["Symbol", "Last_Price", "24h_Change_%", "24h_Volume", "Volume_Display", "Underlying_Asset", "Last_3_Candle"]], height=600)
