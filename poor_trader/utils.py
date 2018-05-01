import os
import re
import numpy as np
import pandas as pd
import traceback


def load_trades(fpath=None):
    columns = ['StartDate', 'EndDate', 'Symbol', 'BuyPrice', 'SellPrice',
               'Shares', 'BuyValue', 'SellValue', 'TotalRisk', 'PnL', 'RMultiple',
               'LastRecordDate', 'LastPrice', 'LastValue', 'LastPnL', 'LastRMultiple', 'OpenIndicator']
    if fpath is None or not os.path.exists(fpath):
        return pd.DataFrame(columns=columns)
    else:
        try:
            df = pd.read_csv(fpath, index_col=0)
            df['StartDate'] = pd.to_datetime(df['StartDate'])
            df['EndDate'] = pd.to_datetime(df['EndDate'])
            if 'CloseIndicator' not in df.columns:
                df['CloseIndicator'] = ''
            return df
        except:
            print(traceback.print_exc())
            return pd.DataFrame(columns=columns)


def load_equity_table(fpath):
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        return df


def load_quotes(symbol):
    df = price_loader.load_price(symbol)
    f_boardlot = lambda price : utils.boardlot(price)
    df['BoardLot'] = df.Close.map(f_boardlot)
    df = df.drop_duplicates(['Date'], keep='first')
    return df


def roundn(n, places=4):
    try:
        return float('%.{}f'.format(places) % n)
    except:
        return n


def _round(nseries, places=4):
    try:
        return pd.Series([roundn(n, places) for n in nseries], nseries.index)
    except:
        return nseries


def round_df(df, places=4):
    return df.apply(lambda x : _round(x, places))


def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1


def historical_volatility(df_quotes):
    logreturns = np.log(df_quotes.Close / df_quotes.Close.shift(1))
    return np.round(np.sqrt(252 * logreturns.var()), 1)


def quotes_range(df_quotes):
    if len(df_quotes.index.values) == 0:
        return 'None'
    start = df_quotes.index.values[0]
    end = df_quotes.index.values[-1]
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        dateformat = '%Y%m%d'
        return '{}_to_{}'.format(start.strftime(dateformat), end.strftime(dateformat))
    except:
        return '{}_to_{}'.format(start, end)


