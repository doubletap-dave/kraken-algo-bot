"""Collection of trading strategies for the backtester.

Each function defined in this module implements a complete trading
strategy.  Strategies accept a ``pandas.DataFrame`` of OHLCV data and
optional parameters for indicator windows or thresholds.  They return
a dictionary of simple performance metrics: number of trades, number
of winning and losing trades and the total compounded return.  These
metrics allow the strategies to be compared on equal footing.  New
strategies can be added by following the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from . import indicators


@dataclass
class BacktestResult:
    """Simple container for backtest statistics."""

    trades: int
    wins: int
    losses: int
    total_return: float



def bollinger_adx_strategy(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    adx_window: int = 14,
    adx_threshold: float = 25.0,
) -> BacktestResult:
    """Bollinger Band breakout strategy with ADX filter.

    This strategy goes long when the closing price breaks above the upper
    Bollinger Band and the ADX exceeds ``adx_threshold``.  It goes short
    when the price breaks below the lower band under the same ADX
    condition.  Positions are exited when the price reverts to the
    middle band.  Only one position is held at any time.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``open``, ``high``, ``low`` and ``close`` columns.
    bb_window : int, optional
        Window length for the Bollinger Bands moving average.  Default
        is 20.
    bb_std : float, optional
        Number of standard deviations for the Bollinger band width.  Default
        is 2.0.
    adx_window : int, optional
        Window length for the ADX calculation.  Default is 14.
    adx_threshold : float, optional
        Minimum ADX value to consider a breakout valid.  Default is 25.0.

    Returns
    -------
    BacktestResult
        Summary statistics of the strategy on the provided data.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    mid, upper, lower = indicators.bollinger_bands(close, window=bb_window, num_std=bb_std)
    adx_vals = indicators.adx(high, low, close, window=adx_window)
    position = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    cumulative_return = 1.0
    wins = 0
    losses = 0
    trades = 0
    for ts in range(1, len(df)):
        price = close.iloc[ts]
        mid_band = mid.iloc[ts]
        upper_band = upper.iloc[ts]
        lower_band = lower.iloc[ts]
        adx_val = adx_vals.iloc[ts]
        # Check entry conditions only when flat
        if position == 0:
            if price > upper_band and adx_val >= adx_threshold:
                position = 1
                entry_price = price
                trades += 1
                continue
            if price < lower_band and adx_val >= adx_threshold:
                position = -1
                entry_price = price
                trades += 1
                continue
        # Check exit conditions when in a position
        if position == 1 and price < mid_band:
            pct_change = (price - entry_price) / entry_price
            cumulative_return *= (1 + pct_change)
            if pct_change > 0:
                wins += 1
            else:
                losses += 1
            position = 0
        elif position == -1 and price > mid_band:
            pct_change = (entry_price - price) / entry_price
            cumulative_return *= (1 + pct_change)
            if pct_change > 0:
                wins += 1
            else:
                losses += 1
            position = 0
    # Close any open position at last price
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
    return BacktestResult(trades=trades, wins=wins, losses=losses, total_return=cumulative_return - 1)



def moving_average_crossover_strategy(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    long_only: bool = False,
) -> BacktestResult:
    """Simple moving average crossover strategy.

    This strategy computes two simple moving averages with different
    lengths.  When the fast SMA crosses above the slow SMA, a long
    position is taken; when it crosses below, a short position is taken
    (unless ``long_only`` is true, in which case short positions are
    ignored and the position is simply closed).  Positions are closed
    on the opposite crossover.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a ``close`` column.
    fast : int, optional
        Window length for the fast SMA.  Default is 20.
    slow : int, optional
        Window length for the slow SMA.  Default is 50.
    long_only : bool, optional
        If True, do not open short positions.  Default is False.

    Returns
    -------
    BacktestResult
        Summary of the strategy's performance.
    """
    close = df["close"]
    sma_fast = indicators.sma(close, fast)
    sma_slow = indicators.sma(close, slow)
    position = 0
    entry_price = 0.0
    cumulative_return = 1.0
    wins = 0
    losses = 0
    trades = 0
    # Determine crossovers
    for ts in range(1, len(df)):
        fast_prev, slow_prev = sma_fast.iloc[ts - 1], sma_slow.iloc[ts - 1]
        fast_cur, slow_cur = sma_fast.iloc[ts], sma_slow.iloc[ts]
        price = close.iloc[ts]
        # Skip if either average is NaN
        if pd.isna(fast_prev) or pd.isna(slow_prev) or pd.isna(fast_cur) or pd.isna(slow_cur):
            continue
        # Golden cross: fast crosses above slow
        if position == 0 and fast_prev < slow_prev and fast_cur >= slow_cur:
            position = 1
            entry_price = price
            trades += 1
            continue
        # Death cross: fast crosses below slow
        if position == 0 and not long_only and fast_prev > slow_prev and fast_cur <= slow_cur:
            position = -1
            entry_price = price
            trades += 1
            continue
        # Exit long on death cross
        if position == 1 and fast_prev > slow_prev and fast_cur <= slow_cur:
            pct_change = (price - entry_price) / entry_price
            cumulative_return *= (1 + pct_change)
            wins += int(pct_change > 0)
            losses += int(pct_change <= 0)
            position = 0
            entry_price = 0.0
            continue
        # Exit short on golden cross
        if position == -1 and fast_prev < slow_prev and fast_cur >= slow_cur:
            pct_change = (entry_price - price) / entry_price
            cumulative_return *= (1 + pct_change)
            wins += int(pct_change > 0)
            losses += int(pct_change <= 0)
            position = 0
            entry_price = 0.0
            continue
    # Close any open position at last price
    if position != 0:
        price = close.iloc[-1]
        pct_change = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        cumulative_return *= (1 + pct_change)
        wins += int(pct_change > 0)
        losses += int(pct_change <= 0)
    return BacktestResult(trades=trades, wins=wins, losses=losses, total_return=cumulative_return - 1)



def rsi_mean_reversion_strategy(
    df: pd.DataFrame,
    window: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
    exit_level: float = 50.0,
) -> BacktestResult:
    """RSI mean reversion strategy.

    This strategy attempts to profit from short‑term overbought and
    oversold conditions.  It goes long when the RSI drops below
    ``lower`` and goes short when the RSI rises above ``upper``.  The
    position is closed when the RSI reverts to ``exit_level``.  Only
    one position is held at any time.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``close`` column.
    window : int, optional
        Lookback window for RSI calculation.  Default is 14.
    lower : float, optional
        Threshold below which the market is considered oversold.  Default
        is 30.
    upper : float, optional
        Threshold above which the market is considered overbought.  Default
        is 70.
    exit_level : float, optional
        RSI level at which to close positions.  Default is 50.

    Returns
    -------
    BacktestResult
        Performance summary.
    """
    close = df["close"]
    rsi_vals = indicators.rsi(close, window=window)
    position = 0
    entry_price = 0.0
    cumulative_return = 1.0
    wins = 0
    losses = 0
    trades = 0
    for ts in range(1, len(df)):
        price = close.iloc[ts]
        rsi_prev = rsi_vals.iloc[ts - 1]
        rsi_cur = rsi_vals.iloc[ts]
        # Skip if NaN
        if pd.isna(rsi_prev) or pd.isna(rsi_cur):
            continue
        if position == 0:
            # Enter long
            if rsi_prev >= lower and rsi_cur < lower:
                position = 1
                entry_price = price
                trades += 1
                continue
            # Enter short
            if rsi_prev <= upper and rsi_cur > upper:
                position = -1
                entry_price = price
                trades += 1
                continue
        else:
            # Exit long when RSI reverts above exit_level
            if position == 1 and rsi_prev <= exit_level and rsi_cur > exit_level:
                pct_change = (price - entry_price) / entry_price
                cumulative_return *= (1 + pct_change)
                wins += int(pct_change > 0)
                losses += int(pct_change <= 0)
                position = 0
                entry_price = 0.0
                continue
            # Exit short when RSI reverts below exit_level
            if position == -1 and rsi_prev >= exit_level and rsi_cur < exit_level:
                pct_change = (entry_price - price) / entry_price
                cumulative_return *= (1 + pct_change)
                wins += int(pct_change > 0)
                losses += int(pct_change <= 0)
                position = 0
                entry_price = 0.0
                continue
    # Close any open position at last price
    if position != 0:
        price = close.iloc[-1]
        pct_change = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        cumulative_return *= (1 + pct_change)
        wins += int(pct_change > 0)
        losses += int(pct_change <= 0)
    return BacktestResult(trades=trades, wins=wins, losses=losses, total_return=cumulative_return - 1)



def macd_crossover_strategy(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    long_only: bool = False,
) -> BacktestResult:
    """MACD signal line crossover strategy.

    This strategy computes the MACD line and signal line.  It goes long
    when the MACD crosses above the signal line and goes short when it
    crosses below (unless ``long_only`` is true, in which case short
    entries are ignored).  Positions are exited on the opposite cross.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``close`` column.
    fast : int, optional
        Span for the fast EMA used in MACD.  Default is 12.
    slow : int, optional
        Span for the slow EMA used in MACD.  Default is 26.
    signal : int, optional
        Span for the signal line EMA.  Default is 9.
    long_only : bool, optional
        If True, ignore short entries and close longs when MACD falls below
        the signal line.  Default is False.

    Returns
    -------
    BacktestResult
        Summary of the strategy's performance.
    """
    close = df["close"]
    macd_dict = indicators.macd(close, fast=fast, slow=slow, signal=signal)
    macd_line = macd_dict["macd"]
    signal_line = macd_dict["signal"]
    position = 0
    entry_price = 0.0
    cumulative_return = 1.0
    wins = 0
    losses = 0
    trades = 0
    for ts in range(1, len(df)):
        macd_prev, sig_prev = macd_line.iloc[ts - 1], signal_line.iloc[ts - 1]
        macd_cur, sig_cur = macd_line.iloc[ts], signal_line.iloc[ts]
        price = close.iloc[ts]
        # Skip until both lines defined
        if pd.isna(macd_prev) or pd.isna(sig_prev) or pd.isna(macd_cur) or pd.isna(sig_cur):
            continue
        # Entry conditions
        if position == 0:
            # Bullish cross
            if macd_prev < sig_prev and macd_cur >= sig_cur:
                position = 1
                entry_price = price
                trades += 1
                continue
            # Bearish cross
            if not long_only and macd_prev > sig_prev and macd_cur <= sig_cur:
                position = -1
                entry_price = price
                trades += 1
                continue
        else:
            # Exit long on bearish cross
            if position == 1 and macd_prev > sig_prev and macd_cur <= sig_cur:
                pct_change = (price - entry_price) / entry_price
                cumulative_return *= (1 + pct_change)
                wins += int(pct_change > 0)
                losses += int(pct_change <= 0)
                position = 0
                entry_price = 0.0
                continue
            # Exit short on bullish cross
            if position == -1 and macd_prev < sig_prev and macd_cur >= sig_cur:
                pct_change = (entry_price - price) / entry_price
                cumulative_return *= (1 + pct_change)
                wins += int(pct_change > 0)
                losses += int(pct_change <= 0)
                position = 0
                entry_price = 0.0
                continue
    # Close any open position at final price
    if position != 0:
        price = close.iloc[-1]
        pct_change = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        cumulative_return *= (1 + pct_change)
        wins += int(pct_change > 0)
        losses += int(pct_change <= 0)
    return BacktestResult(trades=trades, wins=wins, losses=losses, total_return=cumulative_return - 1)



def zscore_mean_reversion_strategy(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 1.5,
    exit_level: float = 0.0,
) -> BacktestResult:
    """Z‑score based mean reversion strategy.

    This strategy computes a rolling mean and standard deviation of the
    closing prices and then derives a Z‑score for the current price.
    When the Z‑score drops below ``-threshold`` the market is
    considered oversold and a long position is taken.  When the
    Z‑score rises above ``threshold`` it is considered overbought and a
    short position is taken.  Positions are closed when the Z‑score
    crosses ``exit_level`` (typically zero).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``close`` column.
    window : int, optional
        Lookback window for the rolling mean and standard deviation.  Default
        is 20.
    threshold : float, optional
        Number of standard deviations to trigger entry.  Default is 1.5.
    exit_level : float, optional
        Z‑score level at which to close positions.  Default is 0.0.

    Returns
    -------
    BacktestResult
        Summary of the strategy's performance.
    """
    close = df["close"]
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std(ddof=0)
    zscore = (close - rolling_mean) / rolling_std
    position = 0
    entry_price = 0.0
    cumulative_return = 1.0
    wins = 0
    losses = 0
    trades = 0
    for ts in range(1, len(df)):
        z_prev, z_cur = zscore.iloc[ts - 1], zscore.iloc[ts]
        price = close.iloc[ts]
        if pd.isna(z_prev) or pd.isna(z_cur):
            continue
        if position == 0:
            if z_prev >= -threshold and z_cur < -threshold:
                position = 1
                entry_price = price
                trades += 1
                continue
            if z_prev <= threshold and z_cur > threshold:
                position = -1
                entry_price = price
                trades += 1
                continue
        else:
            # Exit long when z-score reverts above exit_level
            if position == 1 and z_prev <= exit_level and z_cur > exit_level:
                pct_change = (price - entry_price) / entry_price
                cumulative_return *= (1 + pct_change)
                wins += int(pct_change > 0)
                losses += int(pct_change <= 0)
                position = 0
                entry_price = 0.0
                continue
            # Exit short when z-score reverts below exit_level
            if position == -1 and z_prev >= exit_level and z_cur < exit_level:
                pct_change = (entry_price - price) / entry_price
                cumulative_return *= (1 + pct_change)
                wins += int(pct_change > 0)
                losses += int(pct_change <= 0)
                position = 0
                entry_price = 0.0
                continue
    # Close open position at end
    if position != 0:
        price = close.iloc[-1]
        pct_change = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        cumulative_return *= (1 + pct_change)
        wins += int(pct_change > 0)
        losses += int(pct_change <= 0)
    return BacktestResult(trades=trades, wins=wins, losses=losses, total_return=cumulative_return - 1)



def atr_trailing_stop_strategy(
    df: pd.DataFrame,
    ma_window: int = 50,
    atr_window: int = 14,
    atr_multiplier: float = 3.0,
    long_only: bool = False,
) -> BacktestResult:
    """Trend following strategy with ATR trailing stops.

    This strategy enters a long position when the closing price rises
    above a moving average and enters a short position when it falls
    below (unless ``long_only``).  Once a position is opened a
    trailing stop is maintained based on the Average True Range (ATR)
    multiplied by ``atr_multiplier``.  For long positions the stop is
    set to ``max(previous_stop, price - atr_multiplier * ATR)`` and
    the position is closed when the closing price falls below the
    trailing stop.  For shorts it is symmetric.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``open``, ``high``, ``low`` and ``close``.
    ma_window : int, optional
        Window for the moving average used to determine trend direction.
    atr_window : int, optional
        Window for the ATR calculation.
    atr_multiplier : float, optional
        Multiplier applied to the ATR to set the trailing stop distance.
    long_only : bool, optional
        If True, do not open short positions.

    Returns
    -------
    BacktestResult
        Summary performance metrics.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    moving_avg = indicators.sma(close, ma_window)
    atr_vals = indicators.atr(high, low, close, window=atr_window)
    position = 0
    entry_price = 0.0
    trailing_stop = None
    cumulative_return = 1.0
    wins = 0
    losses = 0
    trades = 0
    for ts in range(1, len(df)):
        ma_prev, ma_cur = moving_avg.iloc[ts - 1], moving_avg.iloc[ts]
        price = close.iloc[ts]
        atr = atr_vals.iloc[ts]
        # Skip if moving average or ATR undefined
        if pd.isna(ma_prev) or pd.isna(ma_cur) or pd.isna(atr):
            continue
        # If not in position, check entry signals
        if position == 0:
            # Long entry: price crosses above moving average
            if price > ma_cur and close.iloc[ts - 1] <= ma_prev:
                position = 1
                entry_price = price
                trailing_stop = price - atr_multiplier * atr
                trades += 1
                continue
            # Short entry: price crosses below moving average
            if not long_only and price < ma_cur and close.iloc[ts - 1] >= ma_prev:
                position = -1
                entry_price = price
                trailing_stop = price + atr_multiplier * atr
                trades += 1
                continue
        else:
            # Update trailing stop
            if position == 1:
                new_stop = price - atr_multiplier * atr
                if trailing_stop is None or new_stop > trailing_stop:
                    trailing_stop = new_stop
                # Exit if price falls below trailing stop
                if price < trailing_stop:
                    pct_change = (price - entry_price) / entry_price
                    cumulative_return *= (1 + pct_change)
                    wins += int(pct_change > 0)
                    losses += int(pct_change <= 0)
                    position = 0
                    entry_price = 0.0
                    trailing_stop = None
                continue
            if position == -1:
                new_stop = price + atr_multiplier * atr
                if trailing_stop is None or new_stop < trailing_stop:
                    trailing_stop = new_stop
                # Exit if price rises above trailing stop
                if price > trailing_stop:
                    pct_change = (entry_price - price) / entry_price
                    cumulative_return *= (1 + pct_change)
                    wins += int(pct_change > 0)
                    losses += int(pct_change <= 0)
                    position = 0
                    entry_price = 0.0
                    trailing_stop = None
                continue
    # Close any remaining position at final price
    if position != 0:
        price = close.iloc[-1]
        pct_change = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        cumulative_return *= (1 + pct_change)
        wins += int(pct_change > 0)
        losses += int(pct_change <= 0)
    return BacktestResult(trades=trades, wins=wins, losses=losses, total_return=cumulative_return - 1)
