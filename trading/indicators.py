"""Technical indicators implemented without external dependencies.

This module defines a collection of functions to compute common
technical indicators used in algorithmic trading.  The goal of
implementing these indicators here rather than relying on third party
packages is twofold: to avoid network constraints when installing
packages like :mod:`pandas_ta`, and to make the computations explicit
for educational purposes.  Each function accepts pandas Series or
DataFrame columns and returns a pandas Series aligned to the input
index.  Some indicators will return multiple Series packaged in a
dictionary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict



def sma(series: pd.Series, window: int) -> pd.Series:
    """Compute the simple moving average (SMA) of a series.

    Parameters
    ----------
    series : pandas.Series
        Input time series.
    window : int
        Number of periods over which to average.

    Returns
    -------
    pandas.Series
        SMA values with the same index as ``series``.  The first
        ``window - 1`` observations will be ``NaN``.
    """
    return series.rolling(window).mean()



def ema(series: pd.Series, window: int) -> pd.Series:
    """Compute the exponential moving average (EMA) of a series.

    The EMA assigns greater weight to more recent observations.  It is
    computed using the standard recursive formula.

    Parameters
    ----------
    series : pandas.Series
        Input time series.
    window : int
        Smoothing window (span) for the EMA.

    Returns
    -------
    pandas.Series
        EMA values aligned to the input series.  The first values will
        match the input until enough data points are available.
    """
    return series.ewm(span=window, adjust=False).mean()



def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.  Values range from 0 to 100.

    Parameters
    ----------
    series : pandas.Series
        Input time series of prices.
    window : int, optional
        Lookback period for computing average gains and losses.  Default
        is 14.

    Returns
    -------
    pandas.Series
        RSI values between 0 and 100.  The first ``window`` points
        will be ``NaN``.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Use exponential moving average for smoother RSI (Wilder's method)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi



def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands for a series.

    Bollinger Bands consist of a moving average (the middle band) and
    upper/lower bands offset by a number of standard deviations.

    Parameters
    ----------
    series : pandas.Series
        Input time series of prices.
    window : int, optional
        Number of periods for the moving average and standard
        deviation.  Default is 20.
    num_std : float, optional
        Number of standard deviations to set the width of the bands.
        Default is 2.0.

    Returns
    -------
    tuple of pandas.Series
        A three‑tuple ``(middle, upper, lower)`` of the moving
        average and upper/lower bands.
    """
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower



def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Compute the Moving Average Convergence Divergence (MACD).

    The MACD is the difference between a fast and slow exponential
    moving average.  A signal line (EMA of the MACD) and histogram
    (difference between MACD and signal) are also returned.

    Parameters
    ----------
    series : pandas.Series
        Input time series of prices.
    fast : int, optional
        Span for the fast EMA.  Default is 12.
    slow : int, optional
        Span for the slow EMA.  Default is 26.
    signal : int, optional
        Span for the signal line.  Default is 9.

    Returns
    -------
    dict
        Dictionary with keys ``"macd"``, ``"signal"`` and ``"hist"``.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}



def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of
    an asset price for a given period.  It is the moving average of
    the true range (TR).

    Parameters
    ----------
    high : pandas.Series
        High prices.
    low : pandas.Series
        Low prices.
    close : pandas.Series
        Close prices.
    window : int, optional
        Number of periods over which to average the true range.  Default
        is 14.

    Returns
    -------
    pandas.Series
        ATR values.
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr



def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Average Directional Index (ADX).

    ADX quantifies the strength of a trend.  It is derived from the
    positive and negative directional movement indicators (DI+ and DI-)
    which in turn rely on the true range (TR).

    Parameters
    ----------
    high : pandas.Series
        High prices.
    low : pandas.Series
        Low prices.
    close : pandas.Series
        Close prices.
    window : int, optional
        Number of periods used to smooth the directional indicators.  Default
        is 14.

    Returns
    -------
    pandas.Series
        ADX values, where higher values indicate stronger trends.
    """
    # Directional movement
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
    atr = tr.rolling(window).mean()
    # Directional indices
    plus_di = 100 * (plus_dm.rolling(window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window).sum() / atr)
    # Avoid division by zero
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / di_sum) * 100
    adx = dx.rolling(window).mean()
    return adx.fillna(0)
