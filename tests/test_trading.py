#!/usr/bin/env python
# coding=utf-8

import unittest

import os
from path import Path
import pandas as pd
from poor_trader import config
from poor_trader import trading
from poor_trader import pse


class TestBroker(unittest.TestCase):
    def test_min_requirements_to_init_trading_platform(self):
        print('- - - - - - - - - - - - - - START test_min_requirements_to_init_trading_platform - - - - - - - - - - - - - -')
        market_name = 'TestMarket'
        market = trading.Market(name=market_name, historical_data=pd.read_csv(
            config.RESOURCES_PATH / 'historical_data.csv',
            parse_dates=True, index_col=0),
            symbols=['2GO', 'JFC', 'SM', 'ABSP', 'ALI'])
        self.assertEqual(len(market.historical_data.index.values), 5)

        broker = trading.Broker(name='TestBroker')

        position_sizing = trading.PositionSizing(name='TestPositionSizing')
        def _calculate_quantity(trading_period, symbol, portfolio):
            return 0
        position_sizing.calculate_quantity = _calculate_quantity

        portfolio = trading.Portfolio(name='TestPortfolio', starting_value=1000000, market=market, broker=broker, position_sizing=position_sizing)

        trading_system = trading.TradingSystem(name='TestTradingSystem')
        def _is_long(trading_period, symbol):
            return True
        def _is_short(trading_period, symbol):
            return False
        trading_system.is_long = _is_long
        trading_system.is_short = _is_short

        trading_platform = trading.TradingPlatform(portfolio, trading_system)
        trading_platform.run()
        print('- - - - - - - - - - - - - - END test_min_requirements_to_init_trading_platform - - - - - - - - - - - - - -')

    def test_pse(self):
        print()
        print()
        print('- - - - - - - - - - - - - - START test_pse - - - - - - - - - - - - - -')
        market = pse.market
        broker = pse.broker
        portfolio = pse.portfolio
        position_sizing = pse.position_sizing
        trading_system = pse.trading_system

        trading_platform = trading.TradingPlatform(portfolio, trading_system)
        trading_platform.run(start_date='2010-01-01')
        print('- - - - - - - - - - - - - - END test_pse - - - - - - - - - - - - - -')


if __name__ == '__main__':
    unittest.main()
