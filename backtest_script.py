"""
backtest_script.py
====================

This module contains a very simple back‑testing harness for a Bollinger Band
breakout strategy with an Average Directional Index (ADX) filter.  The goal
of this script is not to produce a production‑ready system but to
demonstrate, in a self‑contained way, how one might evaluate a trading
strategy on historical data when external libraries or data sources are
unavailable.  The data used here are a small sample of Bitcoin price
history taken from the `backtest/data/BTC‑6h‑1000wks‑data.csv` file in the
MoonDev repository.  Only the first few dozen rows are included inline as a
CSV string to avoid relying on external network calls.  In a real
research/workflow you would download full data files using the GitHub API
or another reliable source and load them with ``pandas.read_csv``.

The strategy implemented below goes long when the closing price closes
above the upper Bollinger band and short when it closes below the lower
Bollinger band, provided the ADX is above a threshold (default 25).  It
exits positions when the price returns to the middle band.  The
performance of the strategy is summarised by the total return and the
number of winning and losing trades.  This simple evaluation should be
extended with more robust metrics (Sharpe ratio, drawdown, etc.) for
production use.
"""

from __future__ import annotations

import io
import pandas as pd
import numpy as np

def load_sample_data() -> pd.DataFrame:
    """Load a small sample of BTC price data embedded as a CSV string.

    The original dataset resides in MoonDev's ``backtest/data`` folder.  Due to
    network restrictions in this environment we embed a subset of lines
    directly in the script.  Each row contains the timestamp, open, high,
    low, close and volume for a 6‑hour candle.

    Returns
    -------
    pandas.DataFrame
        Data frame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close`` and ``volume``.  The ``timestamp`` column is parsed as
        ``datetime64[ns]`` and set as the index.
    """
    csv_data = """datetime,open,high,low,close,volume
    2015-07-20 18:00:00,277.98,280.00,277.37,280.00,782.88341959
    2015-07-21 00:00:00,279.96,281.27,279.38,280.81,1480.19472074
    2015-07-21 06:00:00,280.81,280.89,278.76,279.40,602.33046995
    2015-07-21 12:00:00,279.38,280.00,278.25,279.76,1177.27234242
    2015-07-21 18:00:00,279.76,280.00,276.85,277.32,1683.76190120
    2015-07-22 00:00:00,277.33,278.30,275.51,275.81,1247.91026529
    2015-07-22 06:00:00,275.74,277.11,275.01,277.11,811.93510005
    2015-07-22 12:00:00,276.93,278.54,276.00,277.81,1168.00671797
    2015-07-22 18:00:00,277.81,277.96,276.00,277.89,1460.05730000
    2015-07-23 00:00:00,277.96,279.75,277.33,277.63,1306.86177323
    2015-07-23 06:00:00,277.63,278.19,277.14,277.31,700.57674403
    2015-07-23 12:00:00,277.36,278.25,276.36,276.88,1638.37599316
    2015-07-23 18:00:00,276.88,277.80,276.28,277.39,1661.10506489
    2015-07-24 00:00:00,277.23,277.62,276.43,277.62,1110.44746833
    2015-07-24 06:00:00,277.62,286.72,277.60,285.10,1736.71818546
    2015-07-24 12:00:00,285.10,288.99,283.75,288.26,2205.24517242
    2015-07-24 18:00:00,288.49,291.52,287.79,289.12,2310.05825674
    2015-07-25 00:00:00,289.12,289.99,286.82,288.24,1515.75599918
    2015-07-25 06:00:00,288.23,291.25,288.02,290.40,445.09194556
    2015-07-25 12:00:00,290.45,291.67,288.59,288.59,1086.77199060
    2015-07-25 18:00:00,288.60,289.70,287.62,289.70,1054.83302474
    2015-07-26 00:00:00,289.68,289.99,288.65,288.96,664.58793698
    2015-07-26 06:00:00,288.95,293.44,288.65,293.43,539.23742278
    2015-07-26 12:00:00,293.44,293.87,291.92,292.74,1376.39790748
    2015-07-26 18:00:00,292.75,294.49,292.03,293.89,1155.02333531
    2015-07-27 00:00:00,293.88,294.96,292.37,292.42,1333.31355158
    2015-07-27 06:00:00,292.43,292.50,287.24,288.40,858.73775909
    2015-07-27 12:00:00,288.39,289.92,288.11,288.75,1626.77670500
    2015-07-27 18:00:00,288.75,297.00,288.74,294.21,2642.20511923
    2015-07-28 00:00:00,294.22,298.00,293.65,297.04,1723.73927510
    2015-07-28 06:00:00,296.98,297.25,293.69,294.69,948.54294452
    2015-07-28 12:00:00,294.76,297.13,294.27,296.19,1595.72368163
    2015-07-28 18:00:00,296.15,296.55,295.50,295.76,1493.86371645
    2015-07-29 00:00:00,295.74,296.58,291.50,291.50,1523.74555647
    2015-07-29 06:00:00,291.52,291.80,289.01,290.24,939.97438577
    2015-07-29 12:00:00,290.33,292.19,290.27,290.30,1282.87982941
    2015-07-29 18:00:00,290.31,293.00,289.12,290.19,1476.90739899
    2015-07-30 00:00:00,290.26,290.98,288.24,290.69,1223.86310751
    2015-07-30 06:00:00,290.78,291.56,286.56,286.70,942.91220157
    2015-07-30 12:00:00,286.70,289.40,286.70,288.71,1481.61562783
    2015-07-30 18:00:00,288.71,289.27,287.71,288.49,1490.59061165
    2015-07-31 00:00:00,288.49,290.00,285.00,285.28,1389.22682912
    2015-07-31 06:00:00,285.07,285.37,282.79,284.96,996.01092572
    2015-07-31 12:00:00,284.92,286.25,283.68,285.50,1604.47963558
    2015-07-31 18:00:00,285.57,287.54,284.28,285.19,1686.90106851
    2015-08-01 00:00:00,285.20,285.60,282.25,282.34,990.06916822
    2015-08-01 06:00:00,282.30,282.30,277.82,277.82,602.54920253
    2015-08-01 12:00:00,278.09,281.66,277.26,280.95,1101.13309297
    2015-08-01 18:00:00,281.06,282.83,280.75,281.53,1238.69828773
    2015-08-02 00:00:00,281.53,281.63,277.51,278.96,866.00117402
    2015-08-02 06:00:00,278.88,280.69,277.33,280.42,695.70327547
    """

    df = pd.read_csv(io.StringIO(csv_data), parse_dates=["datetime"])
    df.rename(columns={"datetime": "timestamp"}, inplace=True)
    df.set_index("timestamp", inplace=True)
    return df

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands for a price series.

    Parameters
    ----------
    series : pandas.Series
        The price series (closing prices).
    window : int
        Length of the moving average window.
    num_std : float
        Number of standard deviations for the upper and lower bands.

    Returns
    -------
    tuple[pandas.Series, pandas.Series, pandas.Series]
        A triple of (middle band, upper band, lower band).
    """
    sma = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute the Average Directional Index (ADX).

    This implementation follows the common textbook calculation.  It
    calculates True Range (TR), the positive and negative directional
    movements (DM+ and DM−), converts these into Directional Indicators
    (DI+ and DI−), computes the Directional Movement Index (DX) and
    finally smooths DX to obtain the ADX.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns ``high``, ``low`` and ``close``.
    window : int
        Number of periods over which to smooth the indicators.

    Returns
    -------
    pandas.Series
        The ADX values.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate directional movements
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()

    # True range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smooth the values
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window).sum() / atr)

    # Avoid division by zero
    di_sum = plus_di + minus_di
    di_sum[di_sum == 0] = np.nan
    dx = (abs(plus_di - minus_di) / di_sum) * 100
    adx = dx.rolling(window).mean()
    return adx.fillna(0)

def backtest_strategy(df: pd.DataFrame, adx_threshold: float = 25.0) -> dict:
    """Run a simple Bollinger Band breakout strategy on the provided data.

    The strategy opens a long position when the closing price crosses above
    the upper Bollinger band and the ADX is above ``adx_threshold``.  It
    opens a short position when the close crosses below the lower band and
    ADX is above the threshold.  Positions are closed when the price
    returns to the middle band.  The function iterates through the
    data chronologically, keeping track of trades and calculating a
    cumulative return.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with columns ``open``, ``high``, ``low``, ``close``.
    adx_threshold : float
        Minimum ADX value required to take a trade.

    Returns
    -------
    dict
        Summary statistics including number of trades, number of winning
        and losing trades, and total return.
    """
    close = df['close']
    mid, upper, lower = bollinger_bands(close)
    adx = compute_adx(df)

    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    cumulative_return = 1.0
    wins = 0
    losses = 0

    for idx in range(1, len(df)):
        price = close.iloc[idx]
        prev_price = close.iloc[idx - 1]
        adx_value = adx.iloc[idx]
        mid_band = mid.iloc[idx]
        upper_band = upper.iloc[idx]
        lower_band = lower.iloc[idx]

        # Open long
        if position == 0 and price > upper_band and adx_value >= adx_threshold:
            position = 1
            entry_price = price
        # Open short
        elif position == 0 and price < lower_band and adx_value >= adx_threshold:
            position = -1
            entry_price = price
        # Close long
        elif position == 1 and price < mid_band:
            # Calculate return on exit
            pct_change = (price - entry_price) / entry_price
            cumulative_return *= (1 + pct_change)
            if pct_change > 0:
                wins += 1
            else:
                losses += 1
            position = 0
        # Close short
        elif position == -1 and price > mid_band:
            pct_change = (entry_price - price) / entry_price
            cumulative_return *= (1 + pct_change)
            if pct_change > 0:
                wins += 1
            else:
                losses += 1
            position = 0

    # If a position remains open at the end, close it at the last price
    if position != 0:
        price = close.iloc[-1]
        if position == 1:
            pct_change = (price - entry_price) / entry_price
        else:
            pct_change = (entry_price - price) / entry_price
        cumulative_return *= (1 + pct_change)
        if pct_change > 0:
            wins += 1
        else:
            losses += 1

    return {
        'trades': wins + losses,
        'wins': wins,
        'losses': losses,
        'total_return': cumulative_return - 1,
    }

def main() -> None:
    df = load_sample_data()
    stats = backtest_strategy(df)
    print("Strategy performance on sample data:")
    print(f"  Trades executed: {stats['trades']}")
    print(f"  Winning trades: {stats['wins']}")
    print(f"  Losing trades: {stats['losses']}")
    print(f"  Total return: {stats['total_return']:.2%}")


if __name__ == '__main__':
    main()