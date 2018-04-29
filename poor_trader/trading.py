#!/usr/bin/env python
# coding=utf-8

import abc
from enum import Enum
import pandas as pd


class Market(object):
    def __init__(self, historical_data, symbols=None):
        self.historical_data = historical_data
        self.symbols = symbols or sorted(list([_[:-5] for _ in self.historical_data.filter(like='Date').columns]))

    def iter_trading_periods(self):
        def _filter_symbols(i):
            _df = pd.DataFrame({'Date': self.historical_data.filter(regex='^({})_Date'.format('|'.join(self.symbols))).loc[i]})
            return [_[:-5] for _ in _df['Date'].dropna().index.values]
        return [(i, _filter_symbols(i)) for i in self.historical_data.index.values]

    def get_price(self, trading_period, symbol, field='Close'):
        return self.historical_data.loc[trading_period]['{}_{}'.format(symbol, field)]

    def get_boardlot(self, trading_period, symbol):
        return self.historical_data.loc[trading_period]['{}_BoardLot'.foramt(symbol)]


class Portfolio(object):
    def __init__(self, starting_value):
        self.starting_value = starting_value
        self.buying_power = starting_value
        self.cash = starting_value
        self.transactions = pd.DataFrame()
        self.trades = pd.DataFrame()

    def filter_open_symbols(self, symbols):
        if self.trades.empty:
            return []
        else:
            open_trades = self.trades.loc[self.trades['EndDate'].dropna().index.values][self.trades['Symbol'] in symbols]
            return open_trades['Symbol'].values

    def filter_active_symbols(self, symbols):
        return symbols


class Direction(Enum):
    LONG = 1
    SHORT = -1


class Position(object):
    def __init__(self, symbol, quantity, direction):
        self.id = symbol
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.transactions = pd.DataFrame()

    def add_buy_transaction(trading_period, symbol, price, quantity, value_with_fees):
        _df = pd.DataFrame({'StartDate': trading_period,
                      'Symbol': symbol,
                      'BuyPrice': price,
                      'Shares': quantity,
                      'BuyValue': value_with_fees})
        self.transactions = self.transactions.append(_df).reset_index()

    def add_sell_transaction(trading_period, symbol, price, quantity, value_with_fees):
        _df = pd.DataFrame({'EndDate': trading_period,
                      'Symbol': symbol,
                      'SellPrice': price,
                      'Shares': quantity,
                      'SellValue': value_with_fees})
        self.transactions = self.transactions.append(_df).reset_index()

    def add_sell_short_transaction(trading_period, symbol, price, quantity, value_with_fees):
        raise NotImplementedError


class Broker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='Broker'):
        self.name = name

    @abc.abstractmethod
    def trade(self):
        raise NotImplementedError


class PositionSizing(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='PositionSizing'):
        self.name = name

    @abc.abstractmethod
    def calculate_quantity(self, trading_period, symbol, portfolio):
        raise NotImplementedError


class Action(Enum):
    OPEN_LONG = 0
    OPEN_SHORT = 1
    HOLD = 2
    CLOSE = 3


class TradingSystem(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='TradingSystem'):
        self.name = name

    def filter_long_symbols(self, trading_period, symbols):
        return [_ for _ in symbols if self.is_long(trading_period, _)]

    def filter_short_symbols(self, trading_period, symbols):
        return [_ for _ in symbols if self.is_long(trading_period, _)]

    @abc.abstractmethod
    def is_long(self, trading_period, symbol):
        raise NotImplementedError

    @abc.abstractmethod
    def is_short(self, trading_period, symbol):
        raise NotImplementedError


class TradingPlatform():
    def __init__(self, market, portfolio, broker, position_sizing, trading_system):
        self.market = market
        self.portfolio = portfolio
        self.broker = broker
        self.position_sizing = position_sizing
        self.trading_system = trading_system

    def run(self):
        for trading_period, symbols in self.market.iter_trading_periods():
            open_symbols = self.portfolio.filter_open_symbols(symbols)
            for symbol in open_symbols:
                open_position = portfolio.get_open_position(symbol)
                if trading_system.is_close(trading_period, symbol, open_position):
                    portfolio.close_position(open_position)

            active_symbols = self.portfolio.filter_active_symbols(symbols)
            long_symbols = self.trading_system.filter_long_symbols(trading_period, active_symbols)
            short_symbols = self.trading_system.filter_short_symbols(trading_period, active_symbols)

            for symbol in long_symbols:
                quantity = self.position_sizing.calculate_quantity(trading_period, symbol, self.portfolio)
                if quantity > 0:
                    position = self.broker.trade(trading_period, symbol, quantity, direction)
                    self.portfolio.add_position(position)
            for symbol in short_symbols:
                quantity = self.position_sizing.calculate_quantity(trading_period, symbol, self.portfolio)
                if quantity > 0:
                    position = self.broker.trade(trading_period, symbol, quantity, direction)
                    self.portfolio.add_position(position)
