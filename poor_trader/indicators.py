import os
from os import listdir
from os.path import isfile, join
import re
from path import Path

import numpy as np
import pandas as pd

from poor_trader import utils
from poor_trader.utils import quotes_range
from poor_trader.config import INDICATORS_OUTPUT_PATH


def _true_range(df_quotes, indices):
    cur = df_quotes.iloc[indices[1]]
    prev = df_quotes.iloc[indices[0]]
    high, low, prev_close = cur.High, cur.Low, prev.Close
    a = utils.roundn(high - low, 4)
    b = utils.roundn(abs(high - prev_close), 4)
    c = utils.roundn(abs(low - prev_close), 4)
    return max(a, b, c)


def true_range(df_quotes):
    df = pd.DataFrame(index=df_quotes.index)
    df['n_index'] = range(len(df_quotes))
    _trf = lambda x: _true_range(df_quotes, [int(i) for i in x])
    df['true_range'] = df.n_index.rolling(2).apply(_trf)
    return df.filter(like='true_range')


def SMA(df_quotes, period, field='Close'):
    df = pd.DataFrame(index=df_quotes.index)
    df['SMA'] = df_quotes[field].rolling(period).mean()
    df = utils.round_df(df)
    return df


def STDEV(df_quotes, period, field='Close'):
    df = pd.DataFrame(index=df_quotes.index)
    df['STDEV'] = df_quotes[field].rolling(period).std()
    df = utils.round_df(df)
    return df


def _ema(i, df_quotes, df_ema, period, field='Close'):
    i = [int(_) for _ in i]
    prev_ema, price = df_ema.iloc[i[0]], df_quotes.iloc[i[1]]
    if pd.isnull(prev_ema.EMA):
        return prev_ema.EMA
    else:
        c = 2. / (period + 1.)
        return c * price[field] + (1. - c) * prev_ema.EMA


def EMA(df_quotes, period, field='Close'):
    c = 2./(period + 1.)
    df = pd.DataFrame(columns=['EMA'], index=df_quotes.index)
    sma = SMA(df_quotes, period, field)
    _sma = sma.dropna()
    if len(_sma.index.values) == 0:
        print('ts')
    df.loc[_sma.index.values[0], 'EMA'] = _sma.SMA.values[0]
    for i in range(1, len(df_quotes)):
        prev_ema = df.iloc[i-1]
        if pd.isnull(prev_ema.EMA): continue
        price = df_quotes.iloc[i]
        ema_value = c * price[field] + (1. - c) * prev_ema.EMA
        df.loc[df_quotes.index.values[i], 'EMA'] = ema_value

    df = utils.round_df(df)
    return df


def ATR(df_quotes, period=10, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_ATR_{}.pkl'.format(symbol, quotes_range(df_quotes), period)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(columns=['ATR'], index=df_quotes.index)
    df_true_range = true_range(df_quotes)
    for i in range(1+len(df_quotes)-period):
        if pd.isnull(df_true_range.iloc[i].true_range): continue
        start = i
        end = i+period
        last_index = end - 1
        trs = df_true_range[start:end]
        prev_atr = df.iloc[last_index-1].ATR
        if pd.isnull(prev_atr):
            atr = np.mean([tr for tr in trs.true_range.values])
        else:
            atr = (prev_atr * (period-1) + df_true_range.iloc[last_index].true_range) / period
        df.loc[df_quotes.index.values[last_index], 'ATR'] = atr

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return utils.round_df(df)


def atr_channel(df_quotes, top=7, bottom=3, sma=150, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_atr_channel_{}_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), top, bottom, sma)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)
    df_top_atr = ATR(df_quotes, period=top, symbol=symbol)
    df_bottom_atr = ATR(df_quotes, period=bottom, symbol=symbol)
    df_sma = SMA(df_quotes, period=sma)

    df = pd.DataFrame(columns=['top', 'mid', 'bottom'], index=df_quotes.index)
    df['mid'] = df_sma.SMA
    df['top'] = df.mid + df_top_atr.ATR
    df['bottom'] = df.mid - df_bottom_atr.ATR
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def trailing_stops(df_quotes, multiplier=4, period=10, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_trailing_stops_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), period, multiplier)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(columns=['BuyStops', 'SellStops'], index=df_quotes.index)
    df_atr = ATR(df_quotes, period=period, symbol=symbol)
    sign = -1 # SellStops: -1, BuyStops: 1
    for i in range(len(df_quotes)-1):
        if pd.isnull(df_atr.iloc[i].ATR): continue
        start = i - period
        end = i
        quotes = df_quotes.iloc[start+1:end+1]
        cur_quote = df_quotes.iloc[i]
        next_quote = df_quotes.iloc[i + 1]
        _atr = df_atr.iloc[i].ATR

        # close_price = next_quote.Close
        # trend_dir_sign = -1 if close_price > _atr else 1

        max_price = quotes.Close.max()
        min_price = quotes.Close.min()

        sell = max_price + sign * (multiplier * _atr)
        buy = min_price + sign * (multiplier * _atr)

        sell = [sell, df.iloc[i].SellStops]
        buy = [buy, df.iloc[i].BuyStops]

        try:
            sell = np.max([x for x in sell if not pd.isnull(x)])
            buy = np.min([x for x in buy if not pd.isnull(x)])
        except:
            print(sell)

        if sign < 0:
            df.set_value(index=df_quotes.index.values[i+1], col='SellStops', value=sell)
            if next_quote.Close <= sell:
                sign = 1
        else:
            df.set_value(index=df_quotes.index.values[i+1], col='BuyStops', value=buy)
            if next_quote.Close >= buy:
                sign = -1
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def donchian_channel(df_quotes, high=50, low=50, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_donchian_channel_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), high, low)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(columns=['high', 'mid', 'low'], index=df_quotes.index)
    df['high'] = df_quotes.High.rolling(window=high).max()
    df['low'] = df_quotes.Low.rolling(window=low).min()
    df['mid'] = (df.high + df.low)/2
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def MACD(df_quotes, fast=12, slow=26, signal=9, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_MACD_{}_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), fast, slow, signal)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(index=df_quotes.index)
    fast_ema = EMA(df_quotes, fast)
    slow_ema = EMA(df_quotes, slow)
    df['MACD'] = fast_ema.EMA - slow_ema.EMA
    signal_ema = EMA(df, signal, field='MACD')
    df['Signal'] = signal_ema.EMA
    df['MACDCrossoverSignal'] = np.where(np.logical_and(df.MACD > df.Signal, df.MACD.shift(1) <= df.Signal.shift(1)), 1, 0)
    df['SignalCrossoverMACD'] = np.where(np.logical_and(df.MACD < df.Signal, df.Signal.shift(1) <= df.MACD.shift(1)), 1, 0)
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def SMA_cross(df_quotes, fast=40, slow=60, symbol=None, field='Close'):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_MA_cross_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), fast, slow)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(index=df_quotes.index)
    fast_sma = SMA(df_quotes, fast, field=field)
    slow_sma = SMA(df_quotes, slow, field=field)
    df['FastSMA'] = fast_sma.SMA
    df['SlowSMA'] = slow_sma.SMA
    df['SlowCrossoverFast'] = np.where(np.logical_and(df.FastSMA <= df.SlowSMA, df.FastSMA.shift(1) > df.SlowSMA.shift(1)), 1, 0)
    df['FastCrossoverSlow'] = np.where(np.logical_and(df.FastSMA >= df.SlowSMA, df.SlowSMA.shift(1) > df.FastSMA.shift(1)), 1, 0)
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def SLSMA(df_quotes, s_fast=40, s_slow=60, l_fast=100, l_slow=150, field='Close'):
    # For charting...
    df = pd.DataFrame(index=df_quotes.index)
    s_fast_sma = SMA(df_quotes, s_fast, field=field)
    s_slow_sma = SMA(df_quotes, s_slow, field=field)
    l_fast_sma = SMA(df_quotes, l_fast, field=field)
    l_slow_sma = SMA(df_quotes, l_slow, field=field)
    df['S_FastSMA'] = s_fast_sma.SMA
    df['S_SlowSMA'] = s_slow_sma.SMA
    df['L_FastSMA'] = l_fast_sma.SMA
    df['L_SlowSMA'] = l_slow_sma.SMA
    df = utils.round_df(df)
    return df


def volume(df_quotes, period=20):
    df = pd.DataFrame(index=df_quotes.index)
    ema = EMA(df_quotes, period=period, field='Volume')
    df['Volume'] = df_quotes.Volume
    df['EMA'] = ema.EMA
    df = utils.round_df(df)
    return df


def trend_strength_indicator(df_quotes, start=40, end=150, step=5, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_trend_strength_indicator_{}_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), start, end, step)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(index=df_quotes.index)
    columns = [x for x in range(start, end, step)]
    columns += [end]
    for col in columns:
        df['SMA{}'.format(col)] = SMA(df_quotes, col)
    col_size = len(columns)
    df_comparison = df.lt(df_quotes.Close, axis=0)
    df_comparison['CountSMABelowPrice'] = round(100 * (df_comparison.filter(like='SMA') == True).astype(int).sum(axis=1) / col_size)
    df_comparison['CountSMAAbovePrice'] = round(100 * -(df_comparison.filter(like='SMA') == False).astype(int).sum(axis=1) / col_size)
    df['TrendStrength'] = df_comparison.CountSMABelowPrice + df_comparison.CountSMAAbovePrice
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def bollinger_band(df_quotes, period=60, stdev=1.2, symbol=None):
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_bollinger_band_{}_{}.pkl'.format(symbol, quotes_range(df_quotes), period, stdev)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    df = pd.DataFrame(index=df_quotes.index)
    df_sma = SMA(df_quotes, period)
    df_stdev = STDEV(df_quotes, period)
    df['UP'] = df_sma.SMA + (df_stdev.STDEV * stdev)
    df['MID'] = df_sma.SMA
    df['LOW'] = df_sma.SMA - (df_stdev.STDEV * stdev)
    df = utils.round_df(df)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def SMMA(series, window=14):
    """ get smoothed moving average.

    :param df: data
    :param windows: range
    :return: result series
    """
    smma = series.ewm(
        ignore_na=False, alpha=1.0 / window,
        min_periods=0, adjust=True).mean()
    return smma


def RSI(df_quotes, period=20, symbol=None, field='Close'):
    """
    Relative Strength Index
    :param df_quotes:
    :param period:
    :return:
    """
    if symbol:
        outpath = INDICATORS_OUTPUT_PATH / '{}/{}_RSI_{}.pkl'.format(symbol, quotes_range(df_quotes), period)
        if os.path.exists(outpath):
            return pd.read_pickle(outpath)

    d = df_quotes[field].diff()

    df = pd.DataFrame()

    p_ema = SMMA((d + d.abs()) / 2, window=period)
    n_ema = SMMA((-d + d.abs()) / 2, window=period)

    df['RS'] = rs = p_ema / n_ema
    df['RSI'] = 100 - 100 / (1.0 + rs)

    if symbol:
        if not os.path.exists(outpath.parent):
            os.makedirs(outpath.parent)
        df.to_pickle(outpath)
    return df


def PPSR(data):
    PP = pd.Series((data['High'] + data['Low'] + data['Close']) / 3)
    R1 = pd.Series(2 * PP - data['Low'])
    S1 = pd.Series(2 * PP - data['High'])
    R2 = pd.Series(PP + data['High'] - data['Low'])
    S2 = pd.Series(PP - data['High'] + data['Low'])
    R3 = pd.Series(data['High'] + 2 * (PP - data['Low']))
    S3 = pd.Series(data['Low'] - 2 * (data['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    data= data.join(PSR)
    return data


def stochastic(df, period=20, field='RSI'):
    stoch = ((df[field] - df[field].rolling(period).min()) / (df[field].rolling(period).max() - df[field].rolling(period).min()))
    return stoch


def rm_indicators_file(re_str=None):
    for symbol_dir in listdir(INDICATORS_OUTPUT_PATH):
        symbol_dir_path = INDICATORS_OUTPUT_PATH / symbol_dir
        for fn in listdir(symbol_dir_path):
            m = re.match(re_str, fn)
            if m:
                fpath = symbol_dir_path / fn
                print('Removing', symbol_dir, fn)
                os.remove(fpath)
