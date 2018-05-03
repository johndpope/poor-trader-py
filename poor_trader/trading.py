#!/usr/bin/env python
# coding=utf-8

import os
import abc
import warnings
import pandas as pd
import numpy as np
from path import Path
from enum import Enum
from poor_trader import reports
from poor_trader import utils
from poor_trader import config

np.seterr(divide='ignore', invalid='ignore')

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)


class Market(object):
    def __init__(self, historical_data, name='Market', symbols=None):
        self.name = name
        self.historical_data = historical_data
        self.symbols = symbols or sorted(list([_[:-5] for _ in self.historical_data.filter(like='Date').columns]))

    def iter_trading_periods(self, start_date=None, end_date=None):
        _historical_data = self.historical_data.copy()
        if start_date:
            _historical_data = _historical_data.loc[pd.to_datetime(start_date):]
        if end_date:
            _historical_data = _historical_data.loc[:pd.to_datetime(end_date)]
        def _filter_symbols(i):
            _df = pd.DataFrame({'Date': _historical_data.filter(regex='^({})_Date'.format('|'.join(self.symbols))).loc[i]})
            return [_[:-5] for _ in _df['Date'].dropna().index.values]
        return [(i, _filter_symbols(i)) for i in _historical_data.index.values]

    def get_price(self, trading_period, symbol, field='Close'):
        return self.historical_data.loc[trading_period]['{}_{}'.format(symbol, field)]

    def get_volume(self, trading_period, symbol, field='Volume'):
        return self.historical_data.loc[trading_period]['{}_{}'.format(symbol, field)]

    def get_boardlot(self, trading_period, symbol):
        return self.historical_data.loc[trading_period]['{}_BoardLot'.format(symbol)]


class Portfolio(object):
    def __init__(self, starting_value, market, broker, position_sizing, name='Portfolio'):
        self.name = name
        self.fname = '{}.csv'.format(self.name)
        self.dirpath = config.PORTFOLIO_PATH / self.name
        self.fpath = self.dirpath / self.fname
        self.market = market
        self.df_group_quotes = self.market.historical_data
        self.broker = broker
        self.position_sizing = position_sizing
        self.equity = starting_value
        self.starting_value = starting_value
        self.buying_power = starting_value
        self.cash = starting_value
        self.positions = pd.DataFrame(columns=['Symbol', 'Direction', 'Quantity', 'LastPrice', 'LastValue', 'LastRecordDate'])
        self.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Price', 'Shares', 'Value', 'Action', 'Indicator'])
        self.trades = pd.DataFrame(columns=['BarsHeld', 'StartDate', 'EndDate', 'Symbol', 'BuyPrice', 'SellPrice', 'Shares',
                                            'BuyValue', 'SellValue', 'TotalRisk', 'PnL', 'RMultiple',
                                            'LastRecordDate', 'LastPrice', 'LastValue', 'LastPnL', 'LastRMultiple', 'OpenIndicator', 'CloseIndicator'])

    def filter_open_symbols(self, symbols):
        if self.positions.empty:
            return []
        else:
            open_positions = self.positions[self.positions['Quantity'] > 0]
            return open_positions['Symbol'].values

    def filter_active_symbols(self, symbols):
        return symbols

    def translate_transactions_to_trades(self):
        df = self.trades.copy()
        close_transactions = self.transactions.loc[self.transactions['Action'] == Action.CLOSE]
        df['EndDate'] = close_transactions['Date']
        df['Symbol'] = close_transactions['Symbol']
        df['SellPrice'] = close_transactions['Price']
        df['Shares'] = close_transactions['Shares']
        df['SellValue'] = close_transactions['Value']
        df['CloseIndicator'] = close_transactions['Indicator']
        df['LastRecordDate'] = close_transactions['Date']
        df['LastPrice'] = close_transactions['Price']
        df['LastValue'] = close_transactions['Value']

        for index in close_transactions.index.values:
            _df = self.transactions.copy()
            close_transaction = close_transactions.loc[index]
            open_transaction = _df.loc[_df['Shares'] == close_transaction.Shares].loc[_df['Date'] <= close_transaction.Date].loc[_df['Symbol'] == close_transaction.Symbol].loc[_df['Action'] != Action.CLOSE].iloc[-1]
            df_index = df.loc[df['EndDate'] == close_transaction.Date].loc[df['Symbol'] == close_transaction.Symbol].index.values[-1]
            df.loc[df_index, 'StartDate'] = pd.to_datetime(open_transaction.Date).strftime('%Y-%m-%d')
            df.loc[df_index, 'BuyPrice'] = open_transaction.Price
            df.loc[df_index, 'BuyValue'] = open_transaction.Value
            df.loc[df_index, 'OpenIndicator'] = open_transaction.Indicator
            df.loc[df_index, 'TotalRisk'] = self.position_sizing.calculate_total_risk(open_transaction.Price, close_transaction.Shares)

        for index in self.positions.loc[self.positions.Quantity > 0].index.values:
            _df = self.transactions.copy()
            position = self.positions.loc[index]
            open_transaction = _df.loc[_df['Shares'] == position.Quantity].loc[_df['Symbol'] == position.Symbol].loc[_df['Action'] != Action.CLOSE].iloc[-1]
            df = df.append(pd.DataFrame({'StartDate':pd.to_datetime(open_transaction.Date).strftime('%Y-%m-%d'),
                                         'BuyPrice':open_transaction.Price,
                                         'BuyValue':open_transaction.Value,
                                         'OpenIndicator':open_transaction.Indicator,
                                         'TotalRisk':self.position_sizing.calculate_total_risk(open_transaction.Price, open_transaction.Shares),
                                         'Symbol':position.Symbol,
                                         'Shares':open_transaction.Shares,
                                         'LastRecordDate':position.LastRecordDate,
                                         'LastPrice':position.LastPrice,
                                         'LastValue':position.LastValue}, index=[0]), ignore_index=True)

        df = df.sort_values(['StartDate'])
        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        try:
            df['BarsHeld'] = df.apply(lambda trade: len(self.df_group_quotes.loc[pd.to_datetime(trade['StartDate']):pd.to_datetime(trade['LastRecordDate'])].filter(regex='^{}_Close'.format(trade.Symbol)).dropna().values), axis=1)
        except:
            pass
        df['PnL'] = np.subtract(df['SellValue'].values, np.absolute(df['BuyValue'].values))
        df['RMultiple'] = np.divide(df['PnL'].values, df['TotalRisk'])
        df['LastPnL'] = np.subtract(df['LastValue'].values, np.absolute(df['BuyValue'].values))
        df['LastRMultiple'] = np.divide(df['LastPnL'].values, df['TotalRisk'])
        df = utils.round_df(df)

        df['RMultiple'] = utils._round(df['RMultiple'], places=2)
        df['LastRMultiple'] = utils._round(df['LastRMultiple'], places=2)

        columns = self.trades.columns
        self.trades = df.copy()[columns]

    def get_open_trades(self, symbol):
        return self.transactions.loc[self.transactions['Symbol'] == symbol][-1:]

    def get_open_position(self, symbol):
        return self.positions.loc[self.positions['Symbol'] == symbol].loc[self.positions['Quantity'] > 0]

    def open_position(self, position):
        buy_values = position.transactions[position.transactions.Action == Action.OPEN_LONG]['Value'].values.sum()
        if self.buying_power >= buy_values:
            df = self.get_open_position(position.symbol)
            df = df.loc[df['Direction'] == position.direction]
            if df.empty:
                _df = pd.DataFrame({'Symbol': position.symbol,
                                    'Direction': position.direction,
                                    'Quantity': position.quantity,
                                    'LastPrice': position.last_price,
                                    'LastValue': position.last_value,
                                    'LastRecordDate': position.last_record_date}, index=[0])
                self.positions = self.positions.append(_df, ignore_index=True)
            else:
                index = df.index.values[-1]
                self.positions.loc[index, 'Quantity'] = position.quantity + self.positions.loc[index]['Quantity']
            self.transactions = self.transactions.append(position.transactions, ignore_index=True)
            self.buying_power = self.buying_power - buy_values

    def close_position(self, position):
        sell_values = position.transactions[position.transactions.Action == Action.CLOSE]['Value'].values.sum()
        df = self.get_open_position(position.symbol)
        df = df.loc[df['Direction'] == position.direction]
        index = df.index.values[-1]
        self.positions.loc[index, 'Quantity'] = self.positions.loc[index]['Quantity'] - position.quantity
        self.transactions = self.transactions.append(position.transactions, ignore_index=True)
        self.buying_power = self.buying_power + sell_values

    def update_positions_values(self, trading_period):
        market_quotes = self.market.historical_data.loc[:trading_period]
        for index in self.positions.index.values:
            position = self.positions.loc[index]
            last_price = market_quotes['{}_Close'.format(position.Symbol)].dropna().values[-1]
            last_value = self.broker.calculate_sell_value(last_price, position.Quantity)
            self.positions.loc[index, 'LastPrice'] = last_price
            self.positions.loc[index, 'LastValue'] = last_value
            self.positions.loc[index, 'LastRecordDate'] = trading_period

    def update(self, trading_period):
        self.update_positions_values(trading_period)
        buy_transactions = self.transactions.loc[self.transactions.Action == Action.OPEN_LONG].Value.sum()
        sell_transactions = self.transactions.loc[self.transactions.Action == Action.CLOSE].Value.sum()
        positions_value = self.positions.LastValue.sum()
        self.equity = self.starting_value - buy_transactions + sell_transactions + positions_value
        self.buying_power = self.starting_value - buy_transactions + sell_transactions

    def save_portfolio(self, directory_name=None):
        self.translate_transactions_to_trades()

        directory_name = directory_name or pd.to_datetime('today').strftime('%Y-%m-%d')
        save_path = self.dirpath / directory_name
        utils.makedirs(save_path)

        print('Saving', save_path / 'trades.csv')
        self.trades.to_csv(save_path / 'trades.csv')

        print('Saving', self.fpath)
        self.transactions.to_csv(save_path / 'transactions.csv')

        reports.generate_report(self.trades, self.starting_value, self.df_group_quotes, output_dir_path=save_path, calculate_selling_fees_method=self.broker.calculate_selling_fees)


class Direction(Enum):
    LONG = 1
    SHORT = -1


class Position(object):
    def __init__(self, symbol, quantity, direction):
        self.id = symbol
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.last_price = 0
        self.last_value = 0
        self.last_record_date = None
        self.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Price', 'Shares', 'Value', 'Action', 'Indicator'])

    def update_indicator_column(self, indicator_name):
        for index in self.transactions.index.values:
            if pd.isnull(self.transactions.loc[index]['Indicator']):
                self.transactions.loc[index, 'Indicator'] = indicator_name

    def add_open_long_transaction(self, trading_period, symbol, price, quantity, value_with_fees):
        _df = pd.DataFrame({'Date': trading_period,
                            'Symbol': symbol,
                            'Price': price,
                            'Shares': quantity,
                            'Value': value_with_fees,
                            'Action': Action.OPEN_LONG}, index=[0])
        self.transactions = self.transactions.append(_df, ignore_index=True)

    def add_close_transaction(self, trading_period, symbol, price, quantity, value_with_fees):
        _df = pd.DataFrame({'Date': trading_period,
                            'Symbol': symbol,
                            'Price': price,
                            'Shares': quantity,
                            'Value': value_with_fees,
                            'Action': Action.CLOSE}, index=[0])
        self.transactions = self.transactions.append(_df, ignore_index=True)

    def add_open_short_transaction(self, trading_period, symbol, price, quantity, value_with_fees):
        _df = pd.DataFrame({'Date': trading_period,
                            'Symbol': symbol,
                            'Price': price,
                            'Shares': quantity,
                            'Value': value_with_fees,
                            'Action': Action.OPEN_SHORT}, index=[0])
        self.transactions = self.transactions.append(_df, ignore_index=True)


class Broker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='Broker'):
        self.name = name

    def calculate_commission(self, price, shares, commission=0.0025):
        value = price * shares
        com = value * commission
        return com if com > 20 else 20

    def calculate_buying_fees(self, price, shares, commission=0.0025, vat_on_commission=0.12, pse_trans_fee=0.00005, sccp=0.0001):
        value = price * shares
        com = self.calculate_commission(price, shares, commission=commission)
        vat_com = com * vat_on_commission
        trans = value * pse_trans_fee
        sccp_fee = value * sccp
        return com + vat_com + trans + sccp_fee

    def calculate_selling_fees(self, price, shares, sales_tax=0.006):
        tax = price * shares * sales_tax
        return self.calculate_buying_fees(price, shares) + tax

    def calculate_buy_value(self, price, shares):
        if shares <= 0:
            return 0
        return price * shares + self.calculate_buying_fees(price, shares)

    def calculate_sell_value(self, price, shares):
        if shares <= 0:
            return 0
        return price * shares - self.calculate_selling_fees(price, shares)

    @abc.abstractmethod
    def trade(self, trading_period, symbol, quantity, action):
        raise NotImplementedError


class PositionSizing(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='PositionSizing'):
        self.name = name

    @abc.abstractmethod
    def calculate_quantity(self, trading_period, symbol, portfolio):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_total_risk(self, price, shares):
        raise NotImplementedError


class Action(Enum):
    OPEN_LONG = 0
    OPEN_SHORT = 1
    HOLD = 2
    CLOSE = 3


class Indicator(object):
    def __init__(self, name, df_positions):
        self.name = name
        self.df_positions = df_positions


class CutLoss(object):
    def __init__(self, portfolio, stop_price_pct=0.1, name='CutLoss'):
        self.name = name
        self.portfolio = portfolio
        self.stop_price_pct = stop_price_pct

    def calculate_stop_price(self, open_price, start_date, trading_period, symbol):
        highs = self.portfolio.df_group_quotes[start_date:trading_period]['{}_High'.format(symbol)]
        peak = open_price
        if len(highs.values) > 2:
            peak = np.max(highs.values[1:-1])
            peak = max(peak, open_price)
        return peak - (peak * self.stop_price_pct)


class TradingSystem(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='TradingSystem'):
        self.name = name

    def filter_symbols(self, trading_period, symbols, direction):
        if direction == Direction.LONG:
            return self.filter_long_symbols(trading_period, symbols)
        elif direction == Direction.SHORT:
            return self.filter_short_symbols(trading_period, symbols)
        else:
            raise NotImplementedError

    def filter_long_symbols(self, trading_period, symbols):
        return [_ for _ in symbols if self.is_long(trading_period, _)]

    def filter_short_symbols(self, trading_period, symbols):
        return [_ for _ in symbols if self.is_short(trading_period, _)]

    @abc.abstractmethod
    def get_indicator_name(self, trading_period, symbol, direction):
        raise NotImplementedError

    @abc.abstractmethod
    def get_close_indicator_name(self, trading_period, symbol, direction):
        raise NotImplementedError

    @abc.abstractmethod
    def is_long(self, trading_period, symbol):
        raise NotImplementedError

    @abc.abstractmethod
    def is_short(self, trading_period, symbol):
        raise NotImplementedError

    @abc.abstractmethod
    def is_close(self, trading_period, symbol):
        raise NotImplementedError


class TradingPlatform():
    def __init__(self, portfolio, trading_system):
        self.portfolio = portfolio
        self.market = self.portfolio.market
        self.broker = self.portfolio.broker
        self.position_sizing = self.portfolio.position_sizing
        self.trading_system = trading_system

    def _run_close(self, trading_period, symbols):
        open_symbols = self.portfolio.filter_open_symbols(symbols)
        for symbol in open_symbols:
            open_position = self.portfolio.get_open_position(symbol)
            open_trades = self.portfolio.get_open_trades(symbol)
            open_direction = Direction.LONG
            if self.trading_system.is_close(trading_period, symbol, open_trades):
                position_to_close = self.broker.trade(trading_period, symbol, open_position.Quantity.values[-1], Action.CLOSE)
                position_to_close.update_indicator_column(self.trading_system.get_close_indicator_name(trading_period, symbol, open_direction=open_direction))
                self.portfolio.close_position(position_to_close)

    def _run_open(self, trading_period, symbols, direction):
        active_symbols = self.portfolio.filter_active_symbols(symbols)
        active_symbols = self.trading_system.filter_symbols(trading_period, active_symbols, direction)
        for symbol in active_symbols:
            quantity = self.position_sizing.calculate_quantity(trading_period, symbol, self.portfolio)
            if quantity > 0:
                action = Action.OPEN_LONG if direction == Direction.LONG else Action.OPEN_SHORT
                position = self.broker.trade(trading_period, symbol, quantity, action)
                if position is not None:
                    position.update_indicator_column(self.trading_system.get_indicator_name(trading_period, symbol, direction))
                    self.portfolio.open_position(position)

    def run(self, start_date=None, end_date=None):
        trading_period_list = self.market.iter_trading_periods(start_date, end_date)
        end_date = end_date or pd.to_datetime(trading_period_list[-1][0]).strftime('%Y-%m-%d')
        start_date = start_date or pd.to_datetime(trading_period_list[0][0]).strftime('%Y-%m-%d')
        for trading_period, symbols in trading_period_list:
            self._run_close(trading_period, symbols)
            self._run_open(trading_period, symbols, Direction.LONG)
            self._run_open(trading_period, symbols, Direction.SHORT)
            self.portfolio.update(trading_period)
            open_symbols = self.portfolio.filter_open_symbols(symbols)
            print(pd.to_datetime(trading_period).strftime('%Y-%m-%d'), '/', end_date, np.round(self.portfolio.buying_power, 1), np.round(self.portfolio.equity, 1), ' '.join(open_symbols))

        print(self.portfolio.positions)
        self.portfolio.save_portfolio(directory_name='{}_{}'.format(start_date, end_date))
        print(self.portfolio.trades)
        print()
