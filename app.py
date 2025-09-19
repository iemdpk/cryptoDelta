import streamlit as st
import pandas as pd
import requests
import hashlib
import hmac
import time
from typing import Optional, List, Dict

# Delta Exchange API Client
class DeltaExchangeAPI:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.base_url = "https://api.delta.exchange"
        self.api_key = api_key
        self.api_secret = api_secret
        progress_bar = st.progress(0, text="Initializing: Fetching valid symbols...")
        self.valid_symbols = self._get_valid_symbols()
        progress_bar.progress(100, text="Valid symbols fetched successfully!")
        progress_bar.empty()

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
        progress_bar = st.progress(0, text="Fetching valid symbols...")
        try:
            progress_bar.progress(50, text="Sending request for symbols...")
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            progress_bar.progress(100, text="Symbols fetched successfully!")
            progress_bar.empty()
            if data.get('success'):
                return [ticker.get('symbol', 'N/A') for ticker in data['result']]
            else:
                st.error(f"API Error fetching symbols: {data}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error fetching symbols: {e}")
            progress_bar.empty()
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
        progress_bar = st.progress(0, text="Fetching ticker data...")
        try:
            progress_bar.progress(50, text="Sending request for tickers...")
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            progress_bar.progress(100, text="Ticker data fetched successfully!")
            progress_bar.empty()
            if data.get('success'):
                return self._format_ticker_data(data['result'])
            else:
                st.error(f"API Error: {data}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error: {e}")
            progress_bar.empty()
            return pd.DataFrame()

    def get_perpetual_data(self) -> pd.DataFrame:
        return self.get_tickers(contract_types="perpetual_futures")

    def get_candles(self, symbol: str, resolution: str = "5m", limit: int = 10) -> List[Dict]:
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
        progress_bar = st.progress(0, text=f"Fetching candles for {symbol}...")
        try:
            progress_bar.progress(50, text=f"Sending candle request for {symbol}...")
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            progress_bar.progress(100, text=f"Candles for {symbol} fetched successfully!")
            progress_bar.empty()
            if data.get("success"):
                return data["result"]
            else:
                st.error(f"API Error for {symbol}: {data}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error for {symbol}: {e}")
            progress_bar.empty()
            return []

    def get_trend_and_sma(self, symbol: str) -> Dict[str, str]:
        candles = self.get_candles(symbol, resolution="5m", limit=10)
        if len(candles) < 5:
            st.warning(f"Insufficient candles for {symbol}: {len(candles)} received")
            return {"trend": "NEUTRAL", "sma_5": "0.0000", "sma_10": "0.0000", "sma_signal": "NEUTRAL"}
        
        # Extract close prices
        close_prices = [self._safe_float(c.get("close")) for c in candles]
        current_price = close_prices[-1] if close_prices else 0
        
        # Calculate SMAs
        sma_5 = sum(close_prices[-5:]) / 5 if len(close_prices) >= 5 else 0
        sma_10 = sum(close_prices) / len(close_prices) if close_prices else 0
        
        # SMA Signal Analysis
        sma_signal = "NEUTRAL"
        if current_price > 0 and sma_5 > 0 and sma_10 > 0:
            if current_price > sma_5 > sma_10:
                sma_signal = "BULLISH"
            elif current_price < sma_5 < sma_10:
                sma_signal = "BEARISH"
            elif current_price > sma_5 and sma_5 < sma_10:
                sma_signal = "MIXED_UP"
            elif current_price < sma_5 and sma_5 > sma_10:
                sma_signal = "MIXED_DOWN"
        
        # Use last 5 candles for trend analysis
        candles = candles[-5:]
        greens = 0
        reds = 0
        for c in candles:
            close = self._safe_float(c.get("close"))
            open_price = self._safe_float(c.get("open"))
            if close > open_price:
                greens += 1
            elif close < open_price:
                reds += 1
        
        # More detailed trend analysis
        trend = "NEUTRAL"
        if greens >= 4:
            trend = "STRONG_UP"
        elif greens == 3 and reds <= 2:
            trend = "UP"
        elif reds >= 4:
            trend = "STRONG_DOWN"
        elif reds == 3 and greens <= 2:
            trend = "DOWN"
        
        return {
            "trend": trend,
            "sma_5": f"{sma_5:.4f}",
            "sma_10": f"{sma_10:.4f}",
            "sma_signal": sma_signal
        }

    def _format_volume_short(self, volume: float) -> str:
        """Convert volume to short format with k, m, b suffixes."""
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.1f}b"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}m"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}k"
        else:
            return f"{volume:.0f}"

    def _format_ticker_data(self, tickers: List[Dict]) -> pd.DataFrame:
        formatted_data = []
        total_tickers = len(tickers)
        progress_bar = st.progress(0, text="Formatting ticker data...")
        for i, ticker in enumerate(tickers):
            try:
                symbol = ticker.get('symbol', 'N/A')
                close_price = self._safe_float(ticker.get('close', 0))
                open_price = self._safe_float(ticker.get('open', 0))
                high_price = self._safe_float(ticker.get('high', 0))
                low_price = self._safe_float(ticker.get('low', 0))
                volume = self._safe_float(ticker.get('volume', 0))
                
                if open_price:
                    change_pct = ((close_price - open_price) / open_price) * 100
                else:
                    change_pct = 0
                
                # Calculate price volatility (high-low range as % of close)
                if close_price:
                    volatility = ((high_price - low_price) / close_price) * 100
                else:
                    volatility = 0
                
                # Get trend and SMA data
                trend_sma_data = self.get_trend_and_sma(symbol)
                time.sleep(0.5)
                
                formatted_data.append({
                    'Name': str(symbol),
                    'Last_Price': f"{close_price:.4f}",
                    'SMA_5': trend_sma_data['sma_5'],
                    'SMA_10': trend_sma_data['sma_10'],
                    'SMA_Signal': trend_sma_data['sma_signal'],
                    'High_24h': f"{high_price:.4f}",
                    'Low_24h': f"{low_price:.4f}",
                    '24h_Change': f"{change_pct:+.2f}",
                    'Volatility_24h': f"{volatility:.2f}",
                    '24h_Volume': f"{volume:.0f}",
                    '24h_Volume_Short': self._format_volume_short(volume),
                    'Trend_5x5': trend_sma_data['trend']
                })
                # Update progress bar
                progress_value = int((i + 1) / total_tickers * 100)
                progress_bar.progress(progress_value, text=f"Formatting ticker {i + 1}/{total_tickers}...")
            except Exception as e:
                st.error(f"Error processing ticker {symbol}: {e}")
                continue
        progress_bar.progress(100, text="Ticker data formatting complete!")
        progress_bar.empty()
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
    st.title("Delta Exchange Perpetual Futures Data - Enhanced")
    st.write("*Now analyzing trends based on last 5 candles from 10 fetched candles*")
    
    # Initialize client
    client = DeltaExchangeAPI()
    
    # Fetch data
    progress_bar = st.progress(0, text="Fetching perpetual futures data...")
    progress_bar.progress(50, text="Processing perpetual futures data...")
    df = client.get_perpetual_data()
    progress_bar.progress(100, text="Perpetual futures data fetched successfully!")
    progress_bar.empty()
    
    if df.empty:
        st.error("No data available")
        return
    
    # Create filters
    st.sidebar.header("Filters")
    
    # Price range filter
    min_price = st.sidebar.number_input("Minimum Last Price", min_value=0.0, value=0.0, step=0.0001)
    max_price = st.sidebar.number_input("Maximum Last Price", min_value=0.0, value=1000000.0, step=0.0001)
    
    # 24h Change filter
    min_change = st.sidebar.number_input("Minimum 24h Change (%)", value=-100.0, step=0.1)
    max_change = st.sidebar.number_input("Maximum 24h Change (%)", value=100.0, step=0.1)
    
    # Volume filter
    min_volume = st.sidebar.number_input("Minimum 24h Volume", min_value=0.0, value=0.0, step=1000.0)
    max_volume = st.sidebar.number_input("Maximum 24h Volume", min_value=0.0, value=1000000000.0, step=1000.0)
    
    # Volatility filter
    min_volatility = st.sidebar.number_input("Minimum Volatility (%)", min_value=0.0, value=0.0, step=0.1)
    max_volatility = st.sidebar.number_input("Maximum Volatility (%)", min_value=0.0, value=100.0, step=0.1)
    
    # Trend filter
    trend_options = st.sidebar.multiselect(
        "Filter by Trend (5-candle analysis)",
        ["STRONG_UP", "UP", "NEUTRAL", "DOWN", "STRONG_DOWN"],
        default=["STRONG_UP", "UP", "NEUTRAL", "DOWN", "STRONG_DOWN"]
    )
    
    # SMA Signal filter
    sma_signal_options = st.sidebar.multiselect(
        "Filter by SMA Signal",
        ["BULLISH", "BEARISH", "MIXED_UP", "MIXED_DOWN", "NEUTRAL"],
        default=["BULLISH", "BEARISH", "MIXED_UP", "MIXED_DOWN", "NEUTRAL"]
    )
    
    # Symbol name filter
    symbol_filter = st.sidebar.text_input("Filter by Symbol (contains)", value="")
    
    # Sort options
    sort_by = st.sidebar.selectbox(
        "Sort by", 
        ["24h_Change", "Last_Price", "24h_Volume", "Volatility_24h", "SMA_5", "SMA_10", "Name"]
    )
    sort_order = st.sidebar.selectbox("Sort Order", ["Descending", "Ascending"])
    
    # Apply filters
    filtered_df = df.copy()
    
    # Convert columns to numeric for filtering
    try:
        filtered_df['Last_Price_num'] = filtered_df['Last_Price'].astype(float)
        filtered_df['24h_Change_num'] = filtered_df['24h_Change'].astype(float)
        filtered_df['24h_Volume_num'] = filtered_df['24h_Volume'].astype(float)
        filtered_df['Volatility_24h_num'] = filtered_df['Volatility_24h'].astype(float)
        filtered_df['SMA_5_num'] = filtered_df['SMA_5'].astype(float)
        filtered_df['SMA_10_num'] = filtered_df['SMA_10'].astype(float)
    except ValueError as e:
        st.error(f"Error converting columns to numeric: {e}")
        return
    
    # Apply all filters
    filtered_df = filtered_df[
        (filtered_df['Last_Price_num'] >= min_price) & 
        (filtered_df['Last_Price_num'] <= max_price) &
        (filtered_df['24h_Change_num'] >= min_change) &
        (filtered_df['24h_Change_num'] <= max_change) &
        (filtered_df['24h_Volume_num'] >= min_volume) &
        (filtered_df['24h_Volume_num'] <= max_volume) &
        (filtered_df['Volatility_24h_num'] >= min_volatility) &
        (filtered_df['Volatility_24h_num'] <= max_volatility) &
        (filtered_df['Trend_5x5'].isin(trend_options)) &
        (filtered_df['SMA_Signal'].isin(sma_signal_options))
    ]
    
    # Symbol name filter
    if symbol_filter:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(symbol_filter, case=False, na=False)]
    
    # Apply sorting
    sort_column = f"{sort_by}_num" if f"{sort_by}_num" in filtered_df.columns else sort_by
    if sort_column == "Name_num":
        sort_column = "Name"
    
    ascending = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)
    
    # Drop temporary numeric columns
    numeric_cols_to_drop = ['Last_Price_num', '24h_Change_num', '24h_Volume_num', 'Volatility_24h_num', 'SMA_5_num', 'SMA_10_num']
    filtered_df = filtered_df.drop([col for col in numeric_cols_to_drop if col in filtered_df.columns], axis=1)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        bullish_count = len(filtered_df[filtered_df['SMA_Signal'] == 'BULLISH'])
        st.metric("Bullish SMA Signals", bullish_count)
    with col3:
        bearish_count = len(filtered_df[filtered_df['SMA_Signal'] == 'BEARISH'])
        st.metric("Bearish SMA Signals", bearish_count)
    with col4:
        if len(filtered_df) > 0:
            avg_change = filtered_df['24h_Change'].astype(float).mean()
            st.metric("Avg 24h Change", f"{avg_change:+.2f}%")
    
    # Display filtered data
    st.write("### Filtered Perpetual Futures Data")
    
    # Color code trends and SMA signals for better visualization
    def color_trend(val):
        if val == 'STRONG_UP':
            return ''  # Light green
        elif val == 'UP':
            return ''  # Pale green
        elif val == 'STRONG_DOWN':
            return ''  # Light pink
        elif val == 'DOWN':
            return ''  # Pink
        else:
            return ''  # Light gray
    
    def color_sma_signal(val):
        if val == 'BULLISH':
            return ''  # Light green
        elif val == 'BEARISH':
            return ''  # Light pink
        elif val == 'MIXED_UP':
            return ''  # Light yellow
        elif val == 'MIXED_DOWN':
            return ''  # Moccasin
        else:
            return ''  # Light gray
    
    # Apply styling
    styled_df = filtered_df.style.applymap(color_trend, subset=['Trend_5x5']).applymap(color_sma_signal, subset=['SMA_Signal'])
    selected_columns = ["Name", "Last_Price", "SMA_Signal", "Trend_5x5", "24h_Change", "24h_Volume", "24h_Volume_Short"]    
    st.dataframe(
        styled_df[selected_columns],
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn("Symbol", width="medium"),
            "Last_Price": st.column_config.NumberColumn("Last Price", format="%.4f"),
            "SMA_Signal": st.column_config.TextColumn("SMA Signal", width="small"),
            "Trend_5x5": st.column_config.TextColumn("5-Candle Trend", width="medium"),
            "24h_Change": st.column_config.NumberColumn("24h Change (%)", format="%+.2f"),
            "24h_Volume": st.column_config.NumberColumn("24h Volume", format="%.0f"),
            "24h_Volume_Short": st.column_config.TextColumn("Volume (Short)", width="small"),
        }
    )
    
    # Display trend and SMA signal distributions
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Trend Distribution")
            trend_counts = filtered_df['Trend_5x5'].value_counts()
            st.bar_chart(trend_counts)
        
        with col2:
            st.write("### SMA Signal Distribution")
            sma_counts = filtered_df['SMA_Signal'].value_counts()
            st.bar_chart(sma_counts)
    
    st.write("---")
    st.write("**Legend:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Candle Trends:**")
        st.write("- **STRONG_UP**: 4+ green candles out of 5")
        st.write("- **UP**: 3 green candles, ≤2 red candles")
        st.write("- **STRONG_DOWN**: 4+ red candles out of 5")
        st.write("- **DOWN**: 3 red candles, ≤2 green candles")
        st.write("- **NEUTRAL**: Mixed pattern")
    
    with col2:
        st.write("**SMA Signals:**")
        st.write("- **BULLISH**: Price > SMA(5) > SMA(10)")
        st.write("- **BEARISH**: Price < SMA(5) < SMA(10)")
        st.write("- **MIXED_UP**: Price > SMA(5), but SMA(5) < SMA(10)")
        st.write("- **MIXED_DOWN**: Price < SMA(5), but SMA(5) > SMA(10)")
        st.write("- **NEUTRAL**: Other combinations")
    
    st.write("**Technical Details:**")
    st.write("- **SMA(5)**: Simple Moving Average of last 5 candles")
    st.write("- **SMA(10)**: Simple Moving Average of all 10 candles")
    st.write("- **Volatility**: (High - Low) / Close Price × 100%")

if __name__ == "__main__":
    main()
