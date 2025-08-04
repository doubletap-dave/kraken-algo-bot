"""Trading package for backtesting crypto strategies.

This package contains modules for loading data, computing technical
indicators, implementing trading strategies and running backtests.  It
is intentionally self contained so that it does not depend on any
external libraries beyond pandas and numpy.  Most common technical
indicators are re‑implemented here to avoid the need for third party
packages such as `pandas_ta`.
"""

__all__ = [
    "data_loader",
    "indicators",
    "strategies",
    "backtester",
]
