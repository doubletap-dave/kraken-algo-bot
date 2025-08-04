import unittest
import numpy as np
import pandas as pd
from trading import indicators


class TestIndicators(unittest.TestCase):
    def test_sma(self):
        data = pd.Series([1, 2, 3, 4, 5])
        sma_values = indicators.sma(data, window=2)
        self.assertEqual(len(sma_values), len(data))
        self.assertTrue(np.isnan(sma_values.iloc[0]))
        # second value should be average of first two
        self.assertAlmostEqual(sma_values.iloc[1], (1 + 2) / 2)

    def test_bollinger_bands_shape(self):
        data = pd.Series(range(1, 31))
        mid, upper, lower = indicators.bollinger_bands(data, window=5, num_std=2)
        self.assertEqual(len(mid), len(data))
        self.assertEqual(len(upper), len(data))
        self.assertEqual(len(lower), len(data))

    def test_rsi_range(self):
        data = pd.Series(np.random.randn(100).cumsum())
        rsi_vals = indicators.rsi(data)
        # rsi outputs between 0 and 100 inclusive
        self.assertTrue((rsi_vals >= 0).all())
        self.assertTrue((rsi_vals <= 100).all())

    def test_macd_keys(self):
        data = pd.Series(range(100))
        macd_line, signal, hist = indicators.macd(data)
        self.assertEqual(len(macd_line), len(data))
        self.assertEqual(len(signal), len(data))
        self.assertEqual(len(hist), len(data))

    def test_atr_positive(self):
        df = pd.DataFrame({
            'high': [2, 3, 4, 5, 6],
            'low': [1, 2, 3, 3, 4],
            'close': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        atr_vals = indicators.atr(df, window=2)
        # first values may be NaN; check last value is positive
        self.assertTrue(atr_vals.dropna().gt(0).all())

    def test_adx_positive(self):
        df = pd.DataFrame({
            'high': [2, 3, 4, 5, 6, 7],
            'low': [1, 2, 3, 4, 5, 6],
            'close': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        })
        adx_vals = indicators.adx(df, window=2)
        self.assertEqual(len(adx_vals), len(df))
        self.assertTrue((adx_vals >= 0).all())


if __name__ == '__main__':
    unittest.main()
