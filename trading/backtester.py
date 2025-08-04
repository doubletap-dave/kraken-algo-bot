"""Driver for running backtests on trading strategies.

The backtester orchestrates loading data and applying strategy
functions.  It provides a simple ``run_backtest`` function that
accepts a DataFrame of OHLCV data and a strategy function.  Any
additional keyword arguments are passed through to the strategy.  The
strategy must return an instance of :class:`trading.strategies.BacktestResult`.
"""
from __future__ import annotations

from typing import Callable, Any
import pandas as pd
from .strategies import BacktestResult



def run_backtest(
    data: pd.DataFrame,
    strategy: Callable[..., BacktestResult],
    **kwargs: Any,
) -> BacktestResult:
    """Run a trading strategy on a dataset and return the result.

    Parameters
    ----------
    data : pandas.DataFrame
        OHLCV data on which to run the strategy.  Must include ``open``,
        ``high``, ``low`` and ``close`` columns at a minimum.
    strategy : callable
        A function that accepts the data and keyword arguments and
        returns a :class:`BacktestResult` object.
    **kwargs : any
        Additional keyword arguments passed directly to the strategy
        function.

    Returns
    -------
    BacktestResult
        Summary statistics describing strategy performance.
    """
    if not callable(strategy):
        raise TypeError("strategy must be a callable")
    result = strategy(data, **kwargs)
    if not isinstance(result, BacktestResult):
        raise TypeError("strategy must return a BacktestResult instance")
    return result
