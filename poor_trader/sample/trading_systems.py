
import os
import pandas as pd
from poor_trader import config
from poor_trader import trading
from poor_trader import systems


class CombinedIndicators(trading.TradingSystem):
    def __init__(self, portfolio, systems_method_list, name='CombinedIndicators'):
        super(CombinedIndicators, self).__init__(name=name)
        self.portfolio = portfolio
        self.market = self.portfolio.market
        self.systems_method_list = systems_method_list
        self.fpath = config.TRADING_SYSTEMS_PATH / '{}.pkl'.format(self.name)
        self.df_indicators = pd.DataFrame()
        self.init_indicators()

    def init_indicators(self):
        if os.path.exists(self.fpath):
            self.df_indicators = pd.read_pickle(self.fpath)
        else:
            symbols = self.market.symbols
            df_group_quotes = self.market.historical_data
            df = pd.DataFrame()
            for fname, df_positions in self.systems_method_list:
                df_positions.columns = ['{}_{}'.format(col, fname) for col in df_positions.columns]
                df = df.join(df_positions, how='outer')
            self.df_indicators = df.copy()
            self.df_indicators.to_pickle(self.fpath)

    def get_indicators(self, trading_period, symbol, direction):
        df = self.df_indicators.filter(regex='^{}_'.format(symbol))
        df.columns = [col.replace('{}_'.format(symbol), '') for col in df.columns]
        positions = df.loc[:trading_period].dropna().shift(1).iloc[-1]
        df = pd.DataFrame()
        df['Position'] = positions
        direction_str = 'LONG' if direction == trading.Direction.LONG else 'SHORT'
        return df[df['Position'] == direction_str]

    def get_indicator_name(self, trading_period, symbol, direction):
        return '_'.join(self.get_indicators(trading_period, symbol, direction).index.values)

    def get_close_indicator_name(self, trading_period, symbol, open_direction):
        close_direction = trading.Direction.LONG if open_direction == trading.Direction.SHORT else trading.Direction.SHORT
        return self.get_indicator_name(trading_period, symbol, close_direction)

    def is_long(self, trading_period, symbol):
        open_position = self.portfolio.get_open_position(symbol)
        if open_position.empty:
            return len(self.get_indicators(trading_period, symbol, trading.Direction.LONG).index.values) > 0
        return False

    def is_short(self, trading_period, symbol):
        open_position = self.portfolio.get_open_position(symbol)
        if open_position.empty:
            return len(self.get_indicators(trading_period, symbol, trading.Direction.SHORT).index.values) > 0
        return False

    def is_close(self, trading_period, symbol, open_trades):
        short_indicators = self.get_indicator_name(trading_period, symbol, trading.Direction.SHORT)
        if len(open_trades.index.values) > 1:
            print(open_trades)
            raise NotImplementedError
        for index in open_trades.index.values:
            open_indicators = open_trades.loc[index]['Indicator'].split('_')
            close_indicators = short_indicators.split('_')
            remaining_indicators = [_ for _ in open_indicators if _ not in close_indicators]
            return len(remaining_indicators) <= 0


class Turtle(CombinedIndicators):
    def __init__(self, portfolio, name='Turtle'):
        symbols = portfolio.market.symbols
        df_group_quotes = portfolio.df_group_quotes
        super(Turtle, self).__init__(portfolio,
                                     [systems.run_atr_channel_breakout(symbols, df_group_quotes),
                                      systems.run_dcsma(symbols, df_group_quotes),
                                      systems.run_slsma(symbols, df_group_quotes)],
                                     name=name)

