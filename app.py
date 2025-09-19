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
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: str = "https://api.india.delta.exchange"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret

    def get_headers(self, method: str = "GET", path: str = "", query_string: str = "", payload: str = "") -> Dict[str, str]:
        # Minimal headers for public endpoints
        headers = {"Content-Type": "application/json", "User-Agent": "python-delta-client"}
        # If you later want authenticated endpoints, generate signature here.
        return headers

    def _safe_float(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def get_tickers(self, contract_types: Optional[str] = None) -> pd.DataFrame:
        """Return formatted tickers DataFrame (public endpoint)."""
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
        """Return candles (open/high/low/close/volume) as DataFrame. Columns depend on API response keys."""
        path = "/v2/history/candles"
        params = {"symbol": symbol, "resolution": resolution, "limit": limit}
        try:
            resp = requests.get(f"{self.base_url}{path}", params=params, headers=self.get_headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and data.get("result"):
                # The API returns list-of-dicts – convert to DataFrame and keep needed fields
                df = pd.DataFrame(data["result"])
                # Ensure open/close columns exist (some APIs use lowercase names)
                # We'll try to normalize common names: 'open','close','high','low','volume'
                expected = ["open", "high", "low", "close", "volume", "timestamp"]
                df_cols = {c.lower(): c for c in df.columns}
                normalized = {}
                for col in expected:
                    if col in df_cols:
                        normalized[col] = df_cols[col]
                # if open/close not present, try alternatives
                if "open" not in normalized or "close" not in normalized:
                    # fallback: attempt dictionary items
                    for idx, row in df.iterrows():
                        # as last resort, try keys inside nested objects (rare)
                        pass
                # coerce numeric
                for k, actual in normalized.items():
                    df[actual] = pd.to_numeric(df[actual], errors="coerce")
                return df
            return pd.DataFrame()
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_last3_candle_signal(self, symbol: str, resolution: str = "1m") -> str:
        """
        Returns: "UP" if last 3 candles are all green (close > open),
                 "DOWN" if last 3 candles are all red (close < open),
                 "NEUTRAL" otherwise or on error.
        """
        try:
            df = self.get_candles(symbol, resolution=resolution, limit=3)
            if df.empty or len(df) < 3:
                return "NEUTRAL"
            # Try to find column names for open & close
            lower_cols = [c.lower() for c in df.columns]
            try:
                open_col = df.columns[lower_cols.index("open")]
                close_col = df.columns[lower_cols.index("close")]
            except ValueError:
                # if not found, attempt common alternatives
                if "o" in lower_cols and "c" in lower_cols:
                    open_col = df.columns[lower_cols.index("o")]
                    close_col = df.columns[lower_cols.index("c")]
                else:
                    return "NEUTRAL"
            opens = pd.to_numeric(df[open_col], errors="coerce").fillna(0.0)
            closes = pd.to_numeric(df[close_col], errors="coerce").fillna(0.0)
            greens = (closes > opens).sum()
            reds = (closes < opens).sum()
            if greens == 3:
                return "UP"
            elif reds == 3:
                return "DOWN"
            else:
                return "NEUTRAL"
        except Exception:
            return "NEUTRAL"


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Delta Exchange Dashboard", page_icon="📊", layout="wide")
st.title("📊 Delta Exchange Dashboard")

# Sidebar: allow timeframe and controls
st.sidebar.header("Controls")
resolution = st.sidebar.selectbox("Candle Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)
only_perpetuals = st.sidebar.checkbox("Only Perpetual Futures", value=True)
contract_types_param = "perpetual_futures" if only_perpetuals else None

# Load tickers and products (cached)
@st.cache_data(ttl=30)
def load_tickers_and_products(contract_types_param):
    client = DeltaExchangeAPI()
    tickers = client.get_tickers(contract_types=contract_types_param)
    products = client.get_products()
    return tickers, products

tickers_df, products_df = load_tickers_and_products(contract_types_param)

# If tickers DF is empty, stop early with explanation
if tickers_df is None or tickers_df.empty:
    st.error("No tickers data available from Delta Exchange API. Check network or API endpoint.")
    st.stop()

# Sidebar Filters (guard column access)
symbol_filter_text = st.sidebar.text_input("Search Symbol (text match)")

contract_options = tickers_df["Contract_Type"].unique().tolist() if "Contract_Type" in tickers_df.columns else []
contract_filter = st.sidebar.multiselect("Contract Type", options=contract_options, default=[])

asset_options = tickers_df["Underlying_Asset"].unique().tolist() if "Underlying_Asset" in tickers_df.columns else []
asset_filter = st.sidebar.multiselect("Underlying Asset", options=asset_options, default=[])

# Price slider bounds (safe)
if "Last_Price" in tickers_df.columns:
    price_min = float(tickers_df["Last_Price"].min())
    price_max = float(tickers_df["Last_Price"].max())
else:
    price_min, price_max = 0.0, 1.0
price_range = st.sidebar.slider("Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))

# 24h Volume range filter (min-max)
if "24h_Volume" in tickers_df.columns:
    vol_min_total = float(tickers_df["24h_Volume"].min())
    vol_max_total = float(tickers_df["24h_Volume"].max())
else:
    vol_min_total, vol_max_total = 0.0, 1.0
vol_range = st.sidebar.slider("24h Volume Range (min - max)", min_value=vol_min_total, max_value=vol_max_total, value=(vol_min_total, vol_max_total))

# Preset digit ranges (optional quick filter)
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
digit_preset_choice = st.sidebar.selectbox("24h Volume Digit Preset (optional)", list(digit_presets.keys()), index=0)

# Signal (UP/DOWN/NEUTRAL) filter - user explicitly requested
signal_filter = st.sidebar.multiselect("Signal Filter (Last 3 candles)", options=["UP", "DOWN", "NEUTRAL"], default=[])

# Sorting option
sort_order = st.sidebar.radio("Sort by 24h Change %", ["None", "Ascending", "Descending"], index=0)

# Button to refresh signals (since retrieving signals can be slow)
do_fetch_signals = st.sidebar.button("Fetch Last-3-Candle Signals (might be slow)")

# Apply basic filters (text / contract / asset / ranges)
df = tickers_df.copy()
if symbol_filter_text:
    df = df[df["Symbol"].str.contains(symbol_filter_text, case=False, na=False)]

if contract_filter:
    if "Contract_Type" in df.columns:
        df = df[df["Contract_Type"].isin(contract_filter)]

if asset_filter:
    if "Underlying_Asset" in df.columns:
        df = df[df["Underlying_Asset"].isin(asset_filter)]

# Price filter
if "Last_Price" in df.columns:
    df = df[(df["Last_Price"] >= price_range[0]) & (df["Last_Price"] <= price_range[1])]

# Volume presets
preset_min, preset_max = digit_presets[digit_preset_choice]
if "24h_Volume" in df.columns:
    # Apply preset
    if digit_preset_choice != "All":
        df = df[(df["24h_Volume"] >= preset_min) & (df["24h_Volume"] <= preset_max)]
    # Also apply explicit slider range
    df = df[(df["24h_Volume"] >= vol_range[0]) & (df["24h_Volume"] <= vol_range[1])]

# Sorting
if "24h_Change_%" in df.columns:
    if sort_order == "Ascending":
        df = df.sort_values("24h_Change_%", ascending=True)
    elif sort_order == "Descending":
        df = df.sort_values("24h_Change_%", ascending=False)

# Fetch Last-3-Candle signals if requested or if column not present
client = DeltaExchangeAPI()
if "Last_3_Candle" not in df.columns or do_fetch_signals:
    st.info("Fetching last-3-candle signals for visible symbols. This will make one API call per symbol (may be slow).")
    symbols = df["Symbol"].tolist()
    signals = []
    progress = st.progress(0)
    for i, sym in enumerate(symbols):
        try:
            sig = client.get_last3_candle_signal(sym, resolution=resolution)
        except Exception:
            sig = "NEUTRAL"
        signals.append(sig)
        progress.progress(int((i + 1) / max(1, len(symbols)) * 100))
    # attach signals back to df in same order
    df = df.reset_index(drop=True)
    df["Last_3_Candle"] = signals

# Apply signal filter if selected
if signal_filter:
    if "Last_3_Candle" in df.columns:
        df = df[df["Last_3_Candle"].isin(signal_filter)]
    else:
        st.warning("Signal column not available to filter.")

# Final display columns (only include columns that exist)
display_cols = ["Symbol", "Contract_Type", "Last_Price", "24h_Change_%", "24h_Volume", "Volume_Display", "Open_Interest", "OI_Value", "Underlying_Asset", "Last_3_Candle"]
display_cols = [c for c in display_cols if c in df.columns]

st.header("Market Data")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.dataframe(df[display_cols].reset_index(drop=True), height=600)

# Small diagnostics area for debugging missing columns (helpful if you previously had KeyError)
with st.expander("Debug / Diagnostics (helpful when errors occur)"):
    st.write("DF columns:", df.columns.tolist())
    if df.shape[0] > 0:
        st.write("Sample rows:")
        st.write(df[display_cols].head(5))
    else:
        st.write("DataFrame is empty after filtering.")
