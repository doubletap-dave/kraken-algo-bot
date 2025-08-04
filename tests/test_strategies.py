import unittest
from trading import data_loader, strategies


class TestStrategies(unittest.TestCase):
    def setUp(self):
        # load sample data for testing strategies
        self.data = data_loader.load_sample()

    def check_result(self, result):
        # helper to validate a BacktestResult
        self.assertIsInstance(result, strategies.BacktestResult)
        self.assertEqual(result.trades, result.wins + result.losses)

    def test_bollinger_adx_strategy(self):
        res = strategies.bollinger_adx_strategy(self.data)
        self.check_result(res)

    def test_moving_average_crossover_strategy(self):
        res = strategies.moving_average_crossover_strategy(self.data)
        self.check_result(res)

    def test_rsi_mean_reversion_strategy(self):
        res = strategies.rsi_mean_reversion_strategy(self.data)
        self.check_result(res)

    def test_macd_crossover_strategy(self):
        res = strategies.macd_crossover_strategy(self.data)
        self.check_result(res)

    def test_zscore_mean_reversion_strategy(self):
        res = strategies.zscore_mean_reversion_strategy(self.data)
        self.check_result(res)

    def test_atr_trailing_stop_strategy(self):
        res = strategies.atr_trailing_stop_strategy(self.data)
        self.check_result(res)


if __name__ == '__main__':
    unittest.main()
