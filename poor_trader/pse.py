#!/usr/bin/env python
# coding=utf-8

import os
import pandas as pd
import numpy as np
from path import Path
from poor_trader import config
from poor_trader import trading
from poor_trader import indicators
from poor_trader import systems
from poor_trader.sample import brokers
from poor_trader.sample import trading_systems
from poor_trader.sample import position_sizing_methods


market = trading.Market(name='PSEMarket', historical_data=pd.read_pickle(config.HISTORICAL_DATA_PATH))
broker = brokers.PSEBroker(market)
position_sizing = position_sizing_methods.EquityPctBased(market)
portfolio = trading.Portfolio(name='PSEPortfolio', starting_value=100000, market=market, broker=broker, position_sizing=position_sizing)

class PSEATRChannel(trading_systems.CombinedIndicators, trading.CutLoss):
    def __init__(self, portfolio, name='PSEATRChannelStopPrice', stop_price_pct=0.2):
        super(PSEATRChannel, self).__init__(name=name, portfolio=portfolio,
                                            systems_method_list=[systems.run_atr_channel_breakout_sma(market.symbols, market.historical_data)])
        trading.CutLoss.__init__(self, portfolio=portfolio, stop_price_pct=stop_price_pct)

    def is_short(self, trading_period, symbol):
        '''
        NotImplemented in PSE
        '''
        return False

    def is_close(self, trading_period, symbol, open_trades):
        if not super(PSEATRChannel, self).is_close(trading_period, symbol, open_trades):
            for index in open_trades.index.values:
                open_trade = open_trades.loc[index]
                last_price = self.portfolio.df_group_quotes.loc[trading_period]['{}_Open'.format(symbol)]
                if pd.isnull(last_price):
                    return False
                else:
                    return last_price <= self.calculate_stop_price(open_trade.Price, open_trade.Date, trading_period, open_trade.Symbol)
        else:
            return True

trading_system = PSEATRChannel(portfolio, stop_price_pct=position_sizing.unit_risk_pct * 2)
