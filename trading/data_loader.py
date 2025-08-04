"""Data loading utilities for the trading backtester.

This module provides functions to load OHLCV data from CSV files.  The
MoonDev repository includes several large Bitcoin datasets under the
``data`` folder.  The :func:`load_csv` function reads such a CSV into a
pandas DataFrame, parses the timestamp column and sets it as the index
so that downstream indicator calculations can operate on time series
data.  The loader is deliberately simple—there is no caching or
database support—to keep it easy to test and extend.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional



def load_csv(path: str | Path, timestamp_col: str = "datetime") -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing columns for date/time, open, high,
        low, close and volume.  Additional columns will be ignored.
    timestamp_col : str, optional
        Name of the column containing timestamp strings.  Defaults to
        ``"datetime"``.  The column will be renamed to ``"timestamp"`` and
        set as the index of the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by timestamp with numeric columns for OHLCV.

    Notes
    -----
    The returned index is converted to pandas datetime objects using
    ``pandas.to_datetime`` with default settings.  Callers should
    ensure the timestamp column is parseable by pandas.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp column '{timestamp_col}' not found in CSV")
    # Standardise column names
    df = df.rename(columns={timestamp_col: "timestamp"})
    # Parse timestamp strings into datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    # Ensure numeric columns are floats
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def load_sample(n: int = 200) -> pd.DataFrame:
    """Load a sample of the Bitcoin dataset for testing purposes.

    This helper returns the first ``n`` rows from the BTC dataset
    ``BTC-6h-1000wks-data.csv`` located in the project root.  It is
    useful for unit tests or quick experimentation where loading the
    entire dataset is unnecessary.

    Parameters
    ----------
    n : int, optional
        Number of rows to return from the beginning of the file.  Defaults to
        200.

    Returns
    -------
    pandas.DataFrame
        Sample DataFrame with ``n`` rows.
    """
    sample_path = Path(__file__).resolve().parent.parent / "BTC-6h-1000wks-data.csv"
    df = load_csv(sample_path)
    return df.head(n).copy()
