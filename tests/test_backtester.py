import unittest
from trading.strategies import BacktestResult
from trading import backtester


def dummy_strategy(*args, **kwargs) -> BacktestResult:
    # return a simple backtest result
    return BacktestResult(trades=2, wins=1, losses=1, total_return=0.1)


class TestBacktester(unittest.TestCase):
    def test_run_backtest_valid(self):
        result = backtester.run_backtest(dummy_strategy)
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.trades, result.wins + result.losses)

    def test_run_backtest_invalid_type(self):
        def bad_strategy():
            return "not a backtest result"
        with self.assertRaises(TypeError):
            backtester.run_backtest(bad_strategy)


if __name__ == '__main__':
    unittest.main()
