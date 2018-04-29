#!/usr/bin/env python
# coding=utf-8

import unittest

import os
from path import Path
import pandas as pd
from poor_trader import trading
from poor_trader import pse


MAIN_PATH = Path(os.path.dirname(__file__)).parent

TESTS_PATH = MAIN_PATH / 'tests'

TESTS_RESOURCES_PATH = TESTS_PATH / 'resources'


class TestBroker(unittest.TestCase):
    def test_min_requirements_to_init_trading_platform(self):
        market = trading.Market(historical_data=pd.read_csv(
            TESTS_RESOURCES_PATH / 'historical_data.csv',
            parse_dates=True, index_col=0),
            symbols=['2GO', 'JFC', 'SM', 'ABSP', 'ALI'])
        self.assertEqual(len(market.historical_data.index.values), 5)

        broker = trading.Broker()
        portfolio = trading.Portfolio(starting_value=1000000)

        position_sizing = trading.PositionSizing()
        def _calculate_quantity(trading_period, symbol, portfolio):
            return 0
        position_sizing.calculate_quantity = _calculate_quantity

        trading_system = trading.TradingSystem()
        def _is_long(trading_period, symbol):
            return True
        def _is_short(trading_period, symbol):
            return False
        trading_system.is_long = _is_long
        trading_system.is_short = _is_short

        trading_platform = trading.TradingPlatform(
            market, portfolio, broker, position_sizing, trading_system)
        self.assertIsInstance(trading_platform, trading.TradingPlatform)
        trading_platform.run()


if __name__ == '__main__':
    unittest.main()
