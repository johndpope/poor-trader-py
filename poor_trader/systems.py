
import os
import pandas as pd
import numpy as np
from path import Path
from poor_trader import indicators
from poor_trader.config import SYSTEMS_PATH


def _trim_quotes(symbol, df_group_quotes):
    df_quotes = df_group_quotes.filter(regex='^{}_'.format(symbol))
    df_quotes.columns = [_.replace(symbol + '_', '') for _ in df_quotes.columns]
    df_quotes = df_quotes.loc[df_quotes['Date'].dropna().index]
    return df_quotes

def run_atr_channel_breakout(symbols, df_group_quotes, prefix='ATRChannel', top=7, bottom=3, sma=120):
    fname = '{}{}|{}|{}'.format(prefix, top, bottom, sma)
    fpath = SYSTEMS_PATH / '{}.pkl'.format(fname)
    if os.path.exists(fpath):
        return fname, pd.read_pickle(fpath)
    else:
        df_positions = pd.DataFrame()
        for symbol in symbols:
            print('Running', symbol)
            df_quotes = _trim_quotes(symbol, df_group_quotes)
            df_atr_channel = indicators.atr_channel(df_quotes, top=top, bottom=bottom, sma=sma, symbol=symbol)
            df = pd.DataFrame(index=df_quotes.index)
            long_condition = np.logical_and(df_quotes.Close > df_atr_channel.top, df_quotes.Close.shift(1) < df_atr_channel.top.shift(1))
            short_condition = np.logical_or(df_quotes.Close < df_atr_channel.bottom, df_quotes.Close < df_atr_channel.mid)
            df[symbol] = np.where(long_condition, 'LONG', np.where(short_condition, 'SHORT', 'HOLD'))
            df_positions = df_positions.join(df, how='outer')
        df_positions.to_pickle(fpath)
        return fname, df_positions


def run_dcsma(symbols, df_group_quotes, prefix='DonchianSMA', high=50, low=50, fast=100, slow=150):
    fname = '{}{}|{}|{}|{}'.format(prefix, high, low, fast, slow)
    fpath = SYSTEMS_PATH / '{}.pkl'.format(fname)
    if os.path.exists(fpath):
        return fname, pd.read_pickle(fpath)
    else:
        df_positions = pd.DataFrame()
        for symbol in symbols:
            print('Running', symbol)
            df_quotes = _trim_quotes(symbol, df_group_quotes)
            df_donchian = indicators.donchian_channel(df_quotes, high=high, low=low, symbol=symbol)
            df_sma = indicators.SMA_cross(df_quotes, fast=fast, slow=slow, symbol=symbol)
            df = pd.DataFrame(index=df_quotes.index)
            long_condition = np.logical_and(np.logical_and(df_sma.FastSMA > df_sma.SlowSMA, df_quotes.Close > df_sma.FastSMA),
                                                np.logical_and(df_donchian.high.shift(1) < df_donchian.high, df_donchian.low.shift(1) <= df_donchian.low))
            short_condition = np.logical_and(df_donchian.low.shift(1) > df_donchian.low, df_donchian.high.shift(1) >= df_donchian.high)
            df[symbol] = np.where(long_condition, 'LONG', np.where(short_condition, 'SHORT', 'HOLD'))
            df_positions = df_positions.join(df, how='outer')
        df_positions.to_pickle(fpath)
        return fname, df_positions

def run_slsma(symbols, df_group_quotes, prefix='SLSMA', st_fast=5, st_slow=10, s_fast=40, s_slow=60, l_fast=100, l_slow=120):
    fname = '{}{}|{}|{}|{}|{}|{}'.format(prefix, st_fast, st_slow, s_fast, s_slow, l_fast, l_slow)
    fpath = SYSTEMS_PATH / '{}.pkl'.format(fname)
    if os.path.exists(fpath):
        return fname, pd.read_pickle(fpath)
    else:
        df_positions = pd.DataFrame()
        for symbol in symbols:
            print('Running', symbol)
            df_quotes = _trim_quotes(symbol, df_group_quotes)

            shortest_sma = indicators.SMA_cross(df_quotes, fast=st_fast, slow=st_slow, symbol=symbol)
            short_sma = indicators.SMA_cross(df_quotes, fast=s_fast, slow=s_slow, symbol=symbol)
            long_sma = indicators.SMA_cross(df_quotes, fast=l_fast, slow=l_slow, symbol=symbol)

            df = pd.DataFrame(index=df_quotes.index)
            long_condition = np.logical_and(np.logical_and(long_sma.FastSMA > long_sma.SlowSMA, short_sma.FastSMA > long_sma.FastSMA),
                                            np.logical_or(long_sma.FastCrossoverSlow == 1,
                                                          np.logical_or(short_sma.FastCrossoverSlow == 1,
                                                                        np.logical_and(short_sma.FastSMA > short_sma.SlowSMA,
                                                                                       shortest_sma.FastCrossoverSlow == 1))))
            short_condition = short_sma.FastSMA < long_sma.FastSMA
            df[symbol] = np.where(long_condition, 'LONG', np.where(short_condition, 'SHORT', 'HOLD'))
            df_positions = df_positions.join(df, how='outer')

        df_positions.to_pickle(fpath)
        return fname, df_positions
