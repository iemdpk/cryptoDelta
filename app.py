import requests
import pandas as pd

BASE_URL = "https://api.india.delta.exchange"

def get_candles(symbol: str, resolution: str = "1m", limit: int = 3) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/history/candles"
    params = {"symbol": symbol, "resolution": resolution, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"\n[DEBUG] GET {r.url} -> {r.status_code}")
        r.raise_for_status()
        js = r.json()
        print(f"[DEBUG] Response keys: {list(js.keys())}")
        if js.get("success") and js.get("result"):
            df = pd.DataFrame(js["result"])
            print(f"[DEBUG] Raw columns: {df.columns.tolist()}")
            # Rename if API uses short names
            rename_map = {}
            if "o" in df.columns: rename_map["o"] = "open"
            if "c" in df.columns: rename_map["c"] = "close"
            if "h" in df.columns: rename_map["h"] = "high"
            if "l" in df.columns: rename_map["l"] = "low"
            if "v" in df.columns: rename_map["v"] = "volume"
            if rename_map:
                df = df.rename(columns=rename_map)
                print(f"[DEBUG] Applied rename: {rename_map}")
            print(f"[DEBUG] Final columns: {df.columns.tolist()}")
            return df
        else:
            print("[DEBUG] No result in response")
            return pd.DataFrame()
    except Exception as e:
        print(f"[DEBUG] ERROR in get_candles: {e}")
        return pd.DataFrame()

def get_last3_candle_signal(symbol: str, resolution: str = "1m") -> str:
    print(f"\n[DEBUG] Checking last3 signal for {symbol} ({resolution})")
    df = get_candles(symbol, resolution=resolution, limit=3)

    if df.empty or not {"open", "close"}.issubset(df.columns.str.lower()):
        print("[DEBUG] Empty or missing open/close columns")
        return "NEUTRAL"

    # Normalize case
    df.columns = [c.lower() for c in df.columns]
    df = df.tail(3)
    print("[DEBUG] Last 3 rows:\n", df[["open", "close"]])

    greens = (df["close"] > df["open"]).sum()
    reds = (df["close"] < df["open"]).sum()
    print(f"[DEBUG] greens={greens}, reds={reds}")

    if greens == 3:
        return "UP"
    elif reds == 3:
        return "DOWN"
    return "NEUTRAL"


# --- Test run ---
if __name__ == "__main__":
    symbols = ["BTCUSD", "ETHUSD", "XRPUSD"]
    for sym in symbols:
        sig = get_last3_candle_signal(sym, "1m")
        print(f"Signal for {sym}: {sig}")
