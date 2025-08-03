import os
import requests
import pandas as pd
import numpy as np
import talib
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
SYMBOL = os.getenv("BOT_SYMBOL", "BTCUSD")
INTERVAL = os.getenv("BOT_INTERVAL", "240")


def fetch_ohlcv(symbol=SYMBOL, interval=INTERVAL):
    """
    Fetch OHLCV data from Kraken's public API.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": symbol, "interval": interval}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    json_data = response.json()["result"]
    # The result dictionary contains a single key for the pair; extract it
    pair_key = next(iter(json_data))
    data = json_data[pair_key]
    df = pd.DataFrame(
        data, columns=["time","open","high","low","close","vwap","volume","count"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    # Convert numeric columns to floats
    for col in ["open","high","low","close","vwap","volume"]:
        df[col] = df[col].astype(float)
    return df


def calculate_indicators(df):
    """
    Calculate Bollinger Bands and ADX indicators.
    """
    upper, middle, lower = talib.BBANDS(
        df["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
    )
    adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    df["upper_bb"] = upper
    df["middle_bb"] = middle
    df["lower_bb"] = lower
    df["adx"] = adx
    return df


def generate_signal(df):
    """
    Generate a simple trading signal.
    Buy when price closes above upper Bollinger Band and ADX indicates a strong trend;
    sell when price closes below lower Bollinger Band and ADX indicates a strong trend.
    """
    last_row = df.iloc[-1]
    if last_row["close"] > last_row["upper_bb"] and last_row["adx"] > 25:
        return "buy"
    elif last_row["close"] < last_row["lower_bb"] and last_row["adx"] > 25:
        return "sell"
    return None


def place_order(signal):
    """
    Placeholder for order execution.
    Replace this with calls to Kraken's private API (AddOrder endpoint).
    """
    print(f"Would place {signal} order for {SYMBOL}")


def main():
    df = fetch_ohlcv()
    df = calculate_indicators(df)
    signal = generate_signal(df)
    if signal:
        place_order(signal)
    else:
        print("No trade signal at this time.")


if __name__ == "__main__":
    main()
