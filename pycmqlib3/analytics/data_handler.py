# -*- coding: utf-8 -*-
import datetime

import numpy as np
import math
from numpy.lib.recfunctions import append_fields
from collections import OrderedDict
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

def response_curve(y, response='linear', param=1):
    ''' response curve to apply to a signal, either string or a 1D function f(x)'''
    if not isinstance(response, str):  # 1D interpolation function
        out = response(y)
    elif response == 'reverting':
        scale = (1 + 2 / param ** 2) ** 0.75
        out = scale * y * np.exp(-0.5 * (y / param) ** 2)  # min/max on param
    elif response == 'absorbing':
        scale = 0.258198 * (1 + 6 / param ** 2) ** 1.75
        out = scale * y ** 3 * np.exp(-1.5 * (y / param) ** 2)
    elif response == 'sigmoid':
        # no closed form as a function of the parameter for the 2 below?
        # out = y*0+scale*(erf(y/param/np.sqrt(2))) # y*0 to maintain pandas shape through scipy
        # out = y*0+scale*(2/(1+np.exp(-y/param/np.sqrt(2)))-1) # y*0 to maintain pandas shape through scipy
        scale = 1 / np.sqrt(1 - np.sqrt(np.pi / 2) * param * np.exp(param ** 2 / 2) * math.erfc(param / np.sqrt(2)))
        out = scale * y / np.sqrt(param ** 2 + y ** 2)
    elif response == 'linear':
        out = y
    elif response == 'sign':
        out = 1.0 if y >= 0 else -1.0 
    elif response == 'semilinear':
        scale = 1 / np.sqrt(
            param ** 2 + (1 - param ** 2) * math.erf(param / np.sqrt(2)) - 0.797885 * param * np.exp(-0.5 * param ** 2))
        out = scale * np.minimum(param, np.maximum(-param, y))
    elif response == 'buffer':
        scale = 1 / np.sqrt(2 * (-param * stats.norm.pdf(param) + (1 + param ** 2) * stats.norm.cdf(-param)))
        out = scale * (np.maximum(y - param, 0) + np.minimum(y + param, 0))
    elif response == 'band':
        scale = 1 / np.sqrt(1 - math.erf(param / np.sqrt(2)) + 0.797885 * param * np.exp(-0.5 * param ** 2))
        out = y * (np.abs(y) > param)
        out = out * scale
    else:
        raise Exception('unknown response curve')
    return out

def conv_date(d):
    if type(d).__name__ == 'datetime64':
        d = pd.to_datetime(str(d)).date()
    return d

def date_datetime64(d):
    if type(d).__name__ == 'datetime64':
        return d
    dt = d
    if type(d).__name__ == 'date':
        dt = datetime.datetime.combine(d, datetime.time(0,0,0))
    return np.datetime64(dt)

class DynamicRecArray(object):
    def __init__(self, dtype = [], dataframe = None, size_ratio = 1.5, nlen = 100):
        self.size_ratio = size_ratio
        if isinstance(dataframe, pd.DataFrame) and (len(dataframe) > 0):
            self.create_from_df(dataframe)
        else:
            self.dtype = np.dtype(dtype)
            self.length = 0
            self.size = int(nlen)
            self._data = np.empty(self.size, dtype=self.dtype)

    def __len__(self):
        return self.length

    def append(self, rec):
        if self.length == self.size:
            self.size = int(1.5*self.size) if self.size > 20 else 30
            self._data = np.resize(self._data, self.size)        
        self._data[self.length] = rec
        self.length += 1

    def append_by_dict(self, data_dict):
        if self.length == self.size:
            self.size = int(1.5*self.size) if self.size > 20 else 30
            self._data = np.resize(self._data, self.size)        
        for name in self.dtype.names:
            try:
                self._data[name][self.length] = data_dict[name]
            except:
                continue
        self.length += 1

    def append_by_obj(self, data_obj):
        if self.length == self.size:
            self.size = int(1.5*self.size) if self.size > 20 else 30
            self._data = np.resize(self._data, self.size)
        for name in self.dtype.names:
            try:
                self._data[name][self.length] = getattr(data_obj, name)
            except:
                continue
        self.length += 1

    def remove_lastn(self, n):
        self.length -= n

    def shift_lastn(self, n, forced = False):
        min_array_size = 2 * n
        shifted = 0
        if forced:
            min_array_size = n
        nlen = self.length
        if nlen > min_array_size:
            self._data[:n] = self._data[(nlen-n):(nlen)]
            shifted = nlen - n
            self.length = n
        return shifted

    def extend(self, recs):
        for rec in recs:
            self.append(rec)
    
    def extend_from_df(self, df):
        df_len = len(df)
        if (self.size - self.length) <= df_len * self.size_ratio:
            self.size = self.length + int(self.size_ratio * df_len)
            self._data = np.resize(self._data, self.size)
        s_idx = self.length
        e_idx = self.length + df_len
        for name in self.dtype.names:
            if name in df.columns:
                self._data[name][s_idx:e_idx] = df[name].values

    def create_from_df(self, df, need_index = False):
        df_len = len(df)
        self.size = int(self.size_ratio * df_len) if df_len > 20 else 30
        self._data = np.resize(np.array(df.to_records(index = need_index)), self.size)
        self.dtype = self._data.dtype
        self.length = df_len

    def append_field(self, field, field_value = None, field_type = np.float64):
        if field_value == None:
            field_value = np.zeros(self.size, dtype = field_type)
        self._data = append_fields(self._data, field, field_value, usemask = False)

    @property
    def data(self):
        return self._data[:self.length]
        
def ohlcsum(df):
    return pd.Series([df.index[0], df['open'][0], df['high'].max(), df['low'].min(), df['close'][-1], df['volume'].sum()],
                  index = ['datetime', 'open','high','low','close','volume'])

def min_freq_group(mdf, freq = 5, index_col = 'datetime'):
    if index_col == None:
        mdf = mdf.set_index('datetime')
    min_cnt = (mdf['min_id']/100).astype(int)*60 + (mdf['min_id'] % 100)
    mdf['min_idx'] = (min_cnt/freq).astype(int)
    mdf['date_idx'] = mdf.index.date
    xdf = mdf.groupby([mdf['date_idx'], mdf['min_idx']]).apply(ohlcsum).reset_index()
    if index_col != None:
        xdf = xdf.set_index('datetime')
    return xdf

def min2daily(df, extra_cols = []):
    ts = [df.index[0], df['min_id'][-1], df['open'][0], df['high'].max(), df['low'].min(), df['close'][-1], df['volume'].sum(), df['openInterest'][-1]]
    col_idx = ['datetime', 'min_id', 'open','high','low','close','volume', 'openInterest']
    for col in extra_cols:
        ts.append(df[col][-1])
        col_idx.append(col)
    return pd.Series(ts, index = col_idx)

def day_split(mdf, minlist = [1500], index_col = 'datetime', extra_cols = [], group_func = min2daily):
    min_func = lambda df: group_func(df, extra_cols)
    if index_col == None:
        mdf = mdf.set_index('datetime')
    mdf['min_idx'] = 0
    for idx, mid in enumerate(minlist):
        mdf.loc[mdf['min_id']>=mid, 'min_idx'] = idx + 1
    xdf = mdf.groupby([mdf['date'], mdf['min_idx']]).apply(min_func).reset_index()
    xdf.drop('min_idx', axis = 1, inplace=True)
    if index_col != None:
        xdf = xdf.set_index('datetime')
    #xdf = xdf.sort_values(by=['date', 'min_id'])
    return xdf

def day_split1(df, minlist = [1500], index_col = 'datetime'):
    if index_col != None:
        df = df.reset_index()
    func_dict = OrderedDict()
    for col_name in df.columns:
        if type(col_name).__name__ in ['str', 'unicode']:
            col = str(col_name.strip())
        else:
            col = col_name
        if ('date' == col):
            pass
        elif ('close' in col) or ('openInterest' in col) or ('oi' in col) or ('min_id' in col):
            func_dict[col] = 'last'
        elif ('open' in col) or ('date' in col):
            func_dict[col] = 'first'
        elif 'high' in col:
            func_dict[col] = 'max'
        elif 'low' in col:
            func_dict[col] = 'min'
        elif 'volume' in col:
            func_dict[col] = 'sum'
        else:
            func_dict[col] = 'last'
    df['min_idx'] = 0
    for idx, mid in enumerate(minlist):
        df.loc[df['min_id']>=mid, 'min_idx'] = idx + 1
    xdf = df.groupby([df['date'], df['min_idx']]).aggregate(func_dict)
    if 'min_idx' in xdf.columns:
        xdf.drop('min_idx', axis = 1, inplace=True)
    xdf = xdf.reset_index()
    if index_col != None:
        xdf = xdf.set_index(index_col)
    #xdf = xdf.sort_values(by=['date', 'min_id'])
    return xdf

def index_slice(df, slice_index = ['date', 'min_id'], index_col = 'datetime'):
    if index_col != None:
        df = df.reset_index()
    func_dict = OrderedDict()
    for col_name in df.columns:
        if type(col_name).__name__ in ['str', 'unicode']:
            col = str(col_name.strip())
        else:
            col = col_name
        if ('date' == col) or (col in slice_index):
            pass
        elif ('close' in col) or ('openInterest' in col) or ('oi' in col) or ('min_id' in col):
            func_dict[col] = 'last'
        elif ('open' in col) or ('date' in col):
            func_dict[col] = 'first'
        elif 'high' in col:
            func_dict[col] = 'max'
        elif 'low' in col:
            func_dict[col] = 'min'
        elif 'volume' in col:
            func_dict[col] = 'sum'
        else:
            func_dict[col] = 'last'
    xdf = df.groupby(by = slice_index).aggregate(func_dict)
    #xdf.drop(slice_index, inplace = True)
    xdf = xdf.reset_index()
    if index_col != None:
        xdf = xdf.set_index(index_col)
    #xdf = xdf.sort_values(by=['date', 'min_id'])
    return xdf

def array_split_by_bar(darr, split_list = [300, 1500, 2100], field = 'min_id'):
    s_idx = 0
    sparr = DynamicRecArray(dtype = darr.dtype)
    ind = np.zeros(len(darr))
    for i in range(1, len(split_list)-1):
        ind[(darr[field]>=split_list[i]) & (darr[field]<split_list[i+1])] = i
    for i in range(len(darr)):
        if (i == len(darr)-1) or (darr['date'][s_idx] != darr['date'][i+1]) or (ind[s_idx] != ind[i+1]):
            tmp = darr[s_idx:(i+1)]
            data_dict = {'datetime': tmp['datetime'][0], 'date': tmp['date'][0], 'open': tmp['open'][0], \
                         'high': tmp['high'].max(), 'low': tmp['low'].min(), 'close': tmp['close'][-1], \
                         'volume': tmp['volume'].sum(), 'openInterest': tmp['openInterest'][-1], 'min_id': tmp['min_id'][-1]}
            sparr.append_by_dict(data_dict)
            s_idx = i+1
    return sparr

def bar_conv_func(min_ts, bar_shift = []):
    if type(min_ts).__name__ == 'Series':
        bar_ts = (min_ts/100).astype('int') * 60 + min_ts % 100
        for pair in bar_shift:
            bar_ts[min_ts >= pair[0]] += pair[1]
        return bar_ts
    else:
        bar_id = int(min_ts/100)*60 + min_ts % 100
        for pair in bar_shift:
            if min_ts >= pair[0]:
                bar_id += pair[1]
        return bar_id

def bar_conv_func2(min_ts):
    if type(min_ts).__name__ == 'Series':
        bar_ts = (min_ts/100).astype('int') * 60 + min_ts % 100
        return bar_ts
    else:
        bar_id = int(min_ts/100) * 60 + min_ts % 100
        return bar_id

def conv_ohlc_freq(mdf, freq, index_col = 'datetime', bar_func = bar_conv_func2, extra_cols = [], group_func = min2daily):
    df = mdf
    min_func = lambda df: group_func(df, extra_cols)
    if index_col == None:
        df = df.set_index('datetime')
    if freq in ['d', 'D']:
        res = df.groupby([df['date']]).apply(min_func).reset_index().set_index(['date'])
    else:
        if freq[-3:] in ['min', 'Min']:
            f = int(freq[:-3])
        elif freq[-1:] in ['m', 'M']:
            f = int(freq[:-1])
        df['grp_id'] = pd.Series((bar_func(df['min_id'])/f).astype('int'), name = 'grp_id')
        res = df.groupby([df['date'], df['grp_id']]).apply(min_func).reset_index()
        res.drop('grp_id', axis = 1, inplace=True)
        #res = res.sort_values(by = ['date', 'min_id'])
        if index_col == 'datetime':
            res.set_index(index_col, inplace = True)
    return res

def conv_ohlc_freq1(mdf, freq, index_col = 'datetime', bar_func = bar_conv_func2):
    df = mdf
    if index_col != None:
        df = df.reset_index()
    func_dict = OrderedDict()
    for col_name in df.columns:
        if type(col_name).__name__ in ['str', 'unicode']:
            col = str(col_name.strip())
        else:
            col = col_name
        if ('date' == col):
            pass
        elif ('close' in col) or ('openInterest' in col) or ('oi' in col) or ('min_id' in col):
            func_dict[col] = 'last'
        elif ('open' in col) or ('date' in col):
            func_dict[col] = 'first'
        elif 'high' in col:
            func_dict[col] = 'max'
        elif 'low' in col:
            func_dict[col] = 'min'
        elif 'volume' in col:
            func_dict[col] = 'sum'
        else:
            func_dict[col] = 'last'
    if freq in ['d', 'D']:
        res = df.groupby([df['date']]).aggregate(func_dict).reset_index()
    else:
        if freq[-3:] in ['min', 'Min']:
            f = int(freq[:-3])
        elif freq[-1:] in ['m', 'M']:
            f = int(freq[:-1])
        grp_id = pd.Series((bar_func(df['min_id'])/f).astype('int'), name = 'grp_id')
        res = df.groupby([df['date'], grp_id]).aggregate(func_dict).reset_index()
        res.drop('grp_id', axis = 1, inplace=True)
        #res = res.sort_values(by = ['date', 'min_id'])
    if index_col == 'datetime':
        res.set_index(index_col, inplace = True)
    return res

def crossover(ts, value = 0, direction = 1):
    return ((ts[-1] - value)*direction>0) and ((ts[-2] - value)*direction<0)

def CROSSOVER(ts, value = 0, direction = 1):
    return ((ts - value)*direction > 0) & ((ts.shift(1) - value)*direction < 0)

def crossover2(ts1, ts2, value = 0, direction = 1):
    return ((ts1[-1] - ts2[-1] - value) * direction>0) and ((ts1[-2] - ts2[-2] - value) * direction<0)

def CROSSOVER2(ts1, ts2, value = 0, direction = 1):
    return ((ts1 - ts2 - value) * direction > 0) & ((ts1.shift(1) - ts2.shift(1) - value) * direction < 0)

def TR(df, prefix = ''):
    tr_df = pd.concat([df[prefix + 'high'] - df[prefix + 'close'], \
                       abs(df[prefix + 'high'] - df[prefix + 'close'].shift(1)), \
                       abs(df[prefix + 'low'] - df[prefix + 'close'].shift(1))], \
                      sort = False, join='outer', axis=1)
    ts_tr = pd.Series(tr_df.max(1), name=prefix + 'TR')
    return ts_tr

def tr(df, prefix = ''):
    if np.isnan(df[prefix + 'TR'][-1]):
        df[prefix + 'TR'][-1] = max(df[prefix + 'high'][-1]-df[prefix + 'low'][-1], \
                                    abs(df[prefix + 'high'][-1] - df[prefix + 'close'][-2]), \
                                    abs(df[prefix + 'low'][-1] - df[prefix + 'close'][-2]))

def CMI(df, n, prefix = ''):
    ts = pd.Series(abs(float(df[prefix + 'close'] - df[prefix + 'close'].shift(n)))/\
                   (df[prefix + 'high'].rolling(n).max() - df[prefix + 'low'].rolling(n).min())*100, name=prefix + 'CMI'+str(n))
    return ts

def cmi(df, n, prefix = ''):
    key = prefix + 'CMI'+str(n)
    if (len(df) >= n):
        df[key][-1] = abs(float(df[prefix + 'close'][-1] - df[prefix + 'close'][-n]))/\
                      (max(df[prefix + 'high'][-n:]) - min(df[prefix + 'low'][-n:]))*100
    else:
        df[key][-1] = np.nan

def EFF_RATIO(df, n, field = 'close', prefix = ''):
    er = pd.Series(abs(float(df[prefix + field] - df[prefix + field].shift(n)))/\
                   (abs(df[prefix + field] - df[prefix + field].shift(1)).rolling(n).sum()) * 100.0, \
                   name = prefix + 'ER_' + str(n))
    return er

def PRICE_DENSITY(df, n, prefix = ''):
    price_den = pd.Series(float((df[prefix + 'high']-df[prefix + 'low']).rolling(n).sum())/\
                          (df[prefix + 'high'].rolling(n).max()-df[prefix + 'low'].rolling(n).min()),\
        name = prefix + 'PRICEDEN_' + str(n))
    return price_den

def ATR(df, n = 20, prefix = ''):
    tr = TR(df, prefix = prefix)
    ts_atr = tr.ewm(span=n,  min_periods = n-1, adjust = False).mean()
    ts_atr.name = prefix + 'ATR'+str(n)
    return ts_atr

def atr(df, n = 20, prefix = ''):
    new_tr = max(df[prefix + 'high'][-1]-df[prefix + 'low'][-1], \
                 abs(df[prefix + 'high'][-1] - df[prefix + 'close'][-2]), \
                 abs(df[prefix + 'low'][-1] - df[prefix + 'close'][-2]))
    alpha = 2.0/(n+1)
    df[prefix + 'ATR'+str(n)][-1] = df[prefix + 'ATR'+str(n)][-2] * (1-alpha) + alpha * new_tr

def MA(df, n, field = 'close', prefix = ''):
    return pd.Series(df[prefix + field].rolling(n).mean(), name = prefix + 'MA_' + field.upper() + '_' + str(n), index = df.index)

def ma(df, n, field = 'close', prefix = ''):
    key = prefix + 'MA_' + field.upper() + '_' + str(n)
    df[key][-1] = df[key][-2] + float(df[prefix + field][-1] - df[prefix + field][-1-n])/n

def STDEV(df, n, field = 'close', prefix = ''):
    return pd.Series(df[prefix + field].rolling(n).std(), name = prefix + 'STDEV_' + field.upper() + '_' + str(n))

def stdev(df, n, field = 'close', prefix = ''):
    df[prefix + 'STDEV_' + field.upper() + '_' + str(n)][-1] = np.std(df[field][-n:])

def BSTDEV(df, n, field = 'close', prefix = ''):
    return pd.Series((df[prefix + field].rolling(n).var() \
        + df[prefix + field].rolling(n).mean()**2)**(1/2), \
        name = prefix + 'BSTDEV_' + field.upper() + '_' + str(n))

def SMAVAR(df, n, field = 'close', prefix = ''):
    ma_ts = MA(df, n, field, prefix = prefix)
    var_ts = pd.Series((df[prefix + field]**2).rolling(n).mean() - ma_ts**2, name = prefix + 'SVAR_' + field.upper() + '_' + str(n))
    return pd.concat([ma_ts, var_ts], sort = False, join='outer', axis=1)

def smavar(df, n, field = 'close', prefix = ''):
    ma(df, n, field, prefix = prefix)
    key_var = prefix + 'SVAR_' + field.upper() + '_' + str(n)
    key_ma = prefix + 'MA_' + field.upper() + '_' + str(n)
    df[key_var][-1] = df[key_var][-2] + df[key_ma][-1-n]**2 - df[key_ma][-1]**2 \
                      + float(df[prefix + field][-1]**2 - df[prefix + field][-1-n]**2)/n

#Exponential Moving Average
def EMA(df, n, field = 'close', prefix = ''):
    return pd.Series(df[prefix + field].ewm(span = n).mean(), \
                     name = prefix + 'EMA_' + field.upper() + '_' + str(n), index = df.index)

def ema(df, n, field =  'close', prefix = ''):
    key = prefix + 'EMA_' + field.upper() + '_' + str(n)
    alpha = 2.0/(n+1)
    df[key][-1] = df[key][-2] * (1-alpha) + df[prefix + field][-1] * alpha

def EMAVAR(df, n, field = 'close', prefix = ''):
    ema_ts = EMA(df, n, field, prefix = prefix)
    alpha = 2.0 / (n + 1)
    var_adj = (1-alpha) * (df[prefix + field] - ema_ts.shift(1).fillna(0))**2
    evar_ts = pd.Series(var_adj.ewm(span = n).mean(), \
                        name = prefix + 'EVAR_' + field.upper() + '_' + str(n), index = df.index)
    return pd.concat([ema_ts, evar_ts], sort = False, join='outer', axis=1)

def emavar(df, n ,field = 'close', prefix = ''):
    ema(df, n, field, prefix = prefix)
    alpha = 2.0 / (n + 1)
    key_var = prefix + 'EVAR_' + field.upper() + '_' + str(n)
    key_ema = prefix + 'EMA_' + field.upper() + '_' + str(n)
    df[key_var][-1] = (1-alpha) * ( df[key_var][-2] \
                    + alpha * ((df[prefix + field][-1] - df[key_ema][-2])**2))

#Momentum
def MOM(df, n, prefix = ''):
    return pd.Series(df[prefix + 'close'].diff(n), name = prefix + 'Momentum' + str(n))#Rate of Change

def ROC(df, n, field = 'close', prefix = ''):
    return pd.Series(df[prefix + field].astype('float')/df[prefix + field].shift(n) - 1, \
                     name = 'ROC_' + field.upper()+ '_'  + str(n))

def roc(df, n, field = 'close', prefix = ''):
    key = prefix + 'ROC_' + field.upper() + '_' + str(n)
    df[key][-1] = float(df[prefix + field][-1])/df[prefix + field][-1-n] - 1.0

#Bollinger Bands
def BBANDS(df, n, k = 2, field = 'close', prefix = ''):
    MA = pd.Series(df[prefix + field].rolling(n).mean(), name = prefix + 'MA_' + field.upper() + '_' + str(n))
    MSD = pd.Series(df[prefix + field].rolling(n).std())
    b1 = 2 * k * MSD / MA
    B1 = pd.Series(b1, name = prefix + 'BollingerB' + str(n))
    b2 = (df[field] - MA + k * MSD) / (2 * k * MSD)
    B2 = pd.Series(b2, name = prefix + 'Bollingerb' + str(n))
    UB = pd.Series(MA + k * MSD, name = prefix + 'BollingerU_' + str(n))
    LB = pd.Series(MA - k * MSD, name = prefix + 'BollingerL_' + str(n))
    return pd.concat([B1, B2, MA, UB, LB], sort = False, join='outer', axis=1)

#Pivot Points, Supports and Resistances
def PPSR(df, prefix = ''):
    PP = pd.Series((df[prefix + 'high'] + df[prefix + 'low'] + df[prefix + 'close']) / 3)
    R1 = pd.Series(2 * PP - df[prefix + 'low'])
    S1 = pd.Series(2 * PP - df[prefix + 'high'])
    R2 = pd.Series(PP + df[prefix + 'high'] - df[prefix + 'low'])
    S2 = pd.Series(PP - df[prefix + 'high'] + df[prefix + 'low'])
    R3 = pd.Series(df[prefix + 'high'] + 2 * (PP - df[prefix + 'low']))
    S3 = pd.Series(df[prefix + 'low'] - 2 * (df[prefix + 'high'] - PP))
    psr = {prefix + 'PP':PP, prefix + 'R1':R1, prefix + 'S1':S1, prefix + 'R2':R2, prefix + 'S2':S2, prefix + 'R3':R3, prefix + 'S3':S3}
    PSR = pd.DataFrame(psr)
    return PSR

#Stochastic oscillator %K    
def STOCH(df, n = 14, slowk_period = 3, slowd_period = 3, prefix = ''):
    fastk = float(df[prefix + 'close'] - df[prefix + 'low'].rolling(n).min())\
            /(df[prefix + 'high'].rolling(n).max() - df[prefix + 'low'].rolling(n).min()) * 100.0
    slowk = fastk.rolling(slowk_period).mean()
    slowd = slowk.rolling(slowd_period).mean()
    fk = pd.Series(fastk, index = df.index, name = prefix + "STOCHFK_%s_%s_%s" % (str(n), str(slowk_period), str(slowd_period)))
    sk = pd.Series(slowk, index = df.index, name = prefix + "STOCHSK_%s_%s_%s" % (str(n), str(slowk_period), str(slowd_period)))
    sd = pd.Series(slowd, index = df.index, name = prefix + "STOCHSD_%s_%s_%s" % (str(n), str(slowk_period), str(slowd_period)))
    return pd.concat([fk, sk, sd], sort = False, join='outer', axis=1)

def stoch(df, n=14, slowk_period=3, slowd_period=3, prefix = ''):
    key1 = prefix + "STOCHFK_%s_%s_%s" % (str(n), str(slowk_period), str(slowd_period))
    df[key1][-1] = float(df[prefix + 'close'][-1] - min(df[prefix + 'low'][-n:])) \
                   / (max(df[prefix + 'high'][-n:]) - min(df[prefix + 'low'][-n:])) * 100
    key2 = prefix + "STOCHSK_%s_%s_%s" % (str(n), str(slowk_period), str(slowd_period))
    df[key2][-1] = df[key2][-2] + (df[key1][-1] - df[key1][-1-slowk_period])/float(slowk_period)
    key3 = prefix + "STOCHSD_%s_%s_%s" % (str(n), str(slowk_period), str(slowd_period))
    df[key3][-1] = df[key3][-2] + (df[key2][-1] - df[key2][-1 - slowd_period]) / float(slowd_period)

def STOCHRSI(df, n=14, fastk_period=5, fastd_period=3, prefix = ''):
    RSI = RSI_F(df, n, prefix = prefix)
    fastk = float(RSI - RSI.rolling(fastk_period).min()) / (RSI.rolling(fastk_period).max() - RSI.rolling(fastk_period).min()) * 100.0
    fastd = fastk.rolling(fastd_period).mean()
    fk = pd.Series(fastk, index = df.index, name = prefix + "STOCRSI_FK_%s" % (str(n)))
    fd = pd.Series(fastd, index = df.index, name = prefix + "STOCRSI_FD_%s" % (str(n)))
    return pd.concat([fk,fd], sort = False, join='outer', axis=1)
    
    #Trix
def TRIX(df, n, prefix = ''):
    EX1 = df[prefix + 'close'].ewm(span = n, min_periods = n - 1, adjust = False).mean()
    EX2 = EX1.ewm(span = n, min_periods = n - 1, adjust = False).mean()
    EX3 = EX2.ewm(span = n, min_periods = n - 1, adjust = False).mean()
    return pd.Series(EX3/EX3.shift(1) - 1, name = prefix + 'Trix' + str(n))

#Average Directional Movement Index
def ADX(df, n, prefix = ''):
    UpMove = df[prefix + 'high'] - df[prefix + 'high'].shift(1)
    DoMove = df[prefix + 'low'].shift(1) - df[prefix + 'low']
    UpD = pd.Series(UpMove)
    DoD = pd.Series(DoMove)
    UpD[(UpMove<=DoMove)|(UpMove <= 0)] = 0
    DoD[(DoMove<=UpMove)|(DoMove <= 0)] = 0
    ATRs = ATR(df, n, prefix = prefix)
    PosDI = pd.Series(UpD.ewm(span = n, min_periods = n - 1) / ATRs).mean()
    NegDI = pd.Series(DoD.ewm(span = n, min_periods = n - 1) / ATRs).mean()
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span = n, min_periods = n - 1), \
                    name = prefix + 'ADX' + str(n) + '_' + str(n)).mean()
    return ADX

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow, n_signal, prefix = ''):
    EMAfast = pd.Series(df[prefix + 'close'].ewm(span = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = pd.Series(df[prefix + 'close'].ewm(span = n_slow, min_periods = n_slow - 1).mean())
    MACD = pd.Series(EMAfast - EMAslow, name = prefix + 'MACD' + str(n_fast) + '_' + str(n_slow) + '_' + str(n_signal))
    MACDsig = pd.Series(MACD.ewm(span = n_signal, min_periods = n_signal - 1).mean(), \
                        name = prefix + 'MACDsig' + str(n_fast) + '_' + str(n_slow) + '_' + str(n_signal))
    MACDhist = pd.Series(MACD - MACDsig, \
                        name = prefix + 'MACDhist' + str(n_fast) + '_' + str(n_slow) + '_' + str(n_signal))
    return pd.concat([MACD, MACDsig, MACDhist], sort = False, join='outer', axis=1)

#Mass Index
def MassI(df, prefix = ''):
    Range = df[prefix + 'high'] - df[prefix + 'low']
    EX1 = Range.ewm(span = 9, min_periods = 8).mean()
    EX2 = EX1.ewm(span = 9, min_periods = 8).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(25).sum(), name = prefix + 'MassIndex')
    return MassI

#Vortex Indicator
def Vortex(df, n, prefix = ''):
    tr = TR(df, prefix = prefix)
    vm = abs(df[prefix + 'high'] - df[prefix + 'low'].shift(1)) - abs(df[prefix + 'low']-df[prefix + 'high'].shift(1))
    VI = pd.Series(vm.astype('float').rolling(n).sum() / tr.rolling(n).sum(), name = prefix + 'Vortex' + str(n))
    return VI

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4, prefix = ''):
    M = df[prefix + 'close'].diff(r1 - 1)
    N = df[prefix + 'close'].shift(r1 - 1)
    ROC1 = M.astype('float') / N
    M = df[prefix + 'close'].diff(r2 - 1)
    N = df[prefix + 'close'].shift(r2 - 1)
    ROC2 = M.astype('float') / N
    M = df[prefix + 'close'].diff(r3 - 1)
    N = df[prefix + 'close'].shift(r3 - 1)
    ROC3 = M.astype('float') / N
    M = df[prefix + 'close'].diff(r4 - 1)
    N = df[prefix + 'close'].shift(r4 - 1)
    ROC4 = M.astype('float') / N
    KST = pd.Series(ROC1.rolling(n1).sum() + ROC2.rolling(n2).sum() * 2 + ROC3.rolling(n3).sum() * 3 \
                    + ROC4.rolling(n4).sum() * 4, name = prefix + 'KST' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    return KST

def RSI_F(df, n, field='close', prefix = ''):
    UpMove = df[prefix + field] - df[prefix + field].shift(1)
    DoMove = df[prefix + field].shift(1) - df[prefix + field]
    UpD = pd.Series(UpMove)
    DoD = pd.Series(DoMove)
    UpD[(UpMove <= 0)] = 0
    DoD[(DoMove <= 0)] = 0
    PosDI = pd.Series(UpD.ewm(com = n-1).mean(), name = prefix + "RSI"+str(n)+'_UP')
    NegDI = pd.Series(DoD.ewm(com = n-1).mean(), name = prefix + "RSI"+str(n)+'_DN')
    RSI = pd.Series(PosDI / (PosDI + NegDI) * 100, name = prefix + 'RSI' + str(n))
    return pd.concat([RSI, PosDI, NegDI], sort = False, join='outer', axis=1)

def rsi_f(df, n, field = 'close', prefix = ''):
    RSI_key = prefix + 'RSI%s' % str(n)
    dx = df[prefix + field][-1] - df[prefix + field][-2]
    alpha = 1.0/n
    if dx > 0:
        upx = dx
        dnx = 0
    else:
        upx = 0
        dnx = -dx
    udi = df[RSI_key + '_UP'][-1] = df[RSI_key + '_UP'][-2] * (1 - alpha) + upx * alpha
    ddi = df[RSI_key + '_DN'][-1] = df[RSI_key + '_DN'][-2] * (1 - alpha) + dnx * alpha
    df[RSI_key][-1] = udi/(udi + ddi) * 100.0
    
#True Strength Index
def TSI(df, r, s, prefix = ''):
    M = pd.Series(df[prefix + 'close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm(span = r, min_periods = r - 1).mean())
    aEMA1 = pd.Series(aM.ewm(span = r, min_periods = r - 1).mean())
    EMA2 = pd.Series(EMA1.ewm(span = s, min_periods = s - 1).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span = s, min_periods = s - 1).mean())
    TSI = pd.Series(EMA2 / aEMA2, name = prefix + 'TSI' + str(r) + '_' + str(s))
    return TSI

#Accumulation/Distribution
def ACCDIST(df, n, prefix = ''):
    ad = float(2 * df[prefix + 'close'] - df[prefix + 'high'] - df[prefix + 'low']) \
         / (df[prefix + 'high'] - df[prefix + 'low']) * df[prefix + 'volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = prefix + 'AccDist_ROC' + str(n))
    return AD

#Chaikin Oscillator
def Chaikin(df, prefix = ''):
    ad = float(2 * df[prefix + 'close'] - df[prefix + 'high'] - df[prefix + 'low']) \
         / (df[prefix + 'high'] - df[prefix + 'low']) * df[prefix + 'volume']
    Chaikin = pd.Series(ad.ewm(span = 3, min_periods = 2).mean() - ad.ewm(span = 10, min_periods = 9).mean(), \
                        name = prefix + 'Chaikin')
    return Chaikin

#Money Flow Index and Ratio
def MFI(df, n, prefix = ''):
    PP = (df[prefix + 'high'] + df[prefix + 'low'] + df[prefix + 'close']) / 3.0
    PP = PP.shift(1)
    PosMF = pd.Series(PP)
    PosMF[PosMF <= PosMF.shift(1)] = 0
    PosMF = PosMF * df[prefix + 'volume']
    TotMF = PP * df[prefix + 'volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(MFR.rolling(n).mean(), name = prefix + 'MFI' + str(n))
    return MFI

#On-balance Volume
def OBV(df, n, prefix = ''):
    PosVol = pd.Series(df[prefix + 'volume'])
    NegVol = pd.Series(-df[prefix + 'volume'])
    PosVol[df[prefix + 'close'] <= df[prefix + 'close'].shift(1)] = 0
    NegVol[df[prefix + 'close'] >= df[prefix + 'close'].shift(1)] = 0
    OBV = pd.Series((PosVol + NegVol).rolling(n).mean(), name = prefix + 'OBV' + str(n))
    return OBV

#Force Index
def FORCE(df, n, prefix = ''):
    F = pd.Series(df[prefix + 'close'].diff(n) * df[prefix + 'volume'].diff(n), name = prefix + 'Force' + str(n))
    return F

#Ease of Movement
def EOM(df, n, prefix = ''):
    EoM = (df[prefix + 'high'].diff(1) + df[prefix + 'low'].diff(1)) * float(df[prefix + 'high'] - df[prefix + 'low']) \
          / (2 * df[prefix + 'volume'])
    Eom_ma = pd.Series(EoM.rolling(n).mean(), name = prefix + 'EoM' + str(n))
    return Eom_ma

#Coppock Curve
def COPP(df, n, prefix = ''):
    M = df[prefix + 'close'].diff(int(n * 11 / 10) - 1)
    N = df[prefix + 'close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M.astype('float') / N
    M = df[prefix + 'close'].diff(int(n * 14 / 10) - 1)
    N = df[prefix + 'close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M.astype('float') / N
    Copp = pd.Series((ROC1 + ROC2).ewm(span = n, min_periods = n).mean(), name = prefix + 'Copp' + str(n))
    return Copp

#Keltner Channel
def KELCH(df, n, prefix = ''):
    KelChM = pd.Series(((df[prefix + 'high'] + df[prefix + 'low'] + df[prefix + 'close']) / 3.0).rolling(n).mean(), name = prefix + 'KelChM' + str(n))
    KelChU = pd.Series(((4 * df[prefix + 'high'] - 2 * df[prefix + 'low'] + df[prefix + 'close']) / 3.0).rolling(n).mean(), name = prefix + 'KelChU' + str(n))
    KelChD = pd.Series(((-2 * df[prefix + 'high'] + 4 * df[prefix + 'low'] + df[prefix + 'close']) / 3.0).rolling(n).mean(), name = prefix + 'KelChD' + str(n))
    return pd.concat([KelChM, KelChU, KelChD], sort = False, join='outer', axis=1)

#Ultimate Oscillator
def ULTOSC(df, prefix = ''):
    TR_l = TR(df, prefix = prefix)
    BP_l = df[prefix + 'close'] - pd.concat([df[prefix + 'low'], df[prefix + 'close'].shift(1)], \
                    sort = False, axis=1).min(axis=1)
    UltO = pd.Series((4.0 * BP_l.rolling(7).sum() / TR_l.rolling(7).sum()) + (2.0 * BP_l.rolling(14).sum() \
                    / TR_l.rolling(14).sum()) + (BP_l.rolling(28).sum() / TR_l.rolling(28).sum()), \
                    name = prefix + 'UltOsc')
    return UltO

def DONCH_IDX(df, n, prefix = ''):
    high = pd.Series(df[prefix + 'high'].rolling(n).max(), name = prefix + 'DONCH_H'+ str(n))
    low  = pd.Series(df[prefix + 'low'].rolling(n).min(), name = prefix + 'DONCH_L'+ str(n))
    maxidx = pd.Series(index=df.index, name = prefix + 'DONIDX_H%s' % str(n))
    minidx = pd.Series(index=df.index, name = prefix + 'DONIDX_L%s' % str(n))
    for idx, dateidx in enumerate(high.index):
        if idx >= (n-1):
            highlist = list(df.iloc[(idx-n+1):(idx+1)][prefix + 'high'])[::-1]
            maxidx[idx] = highlist.index(high[idx])
            lowlist = list(df.iloc[(idx-n+1):(idx+1)][prefix + 'low'])[::-1]
            minidx[idx] = lowlist.index(low[idx])
    return pd.concat([high,low, maxidx, minidx], sort = False, join='outer', axis=1)

def CHENOW_PLUNGER(df, n, atr_n = 40, prefix = ''):
    atr = ATR(df, atr_n, prefix = prefix)
    high = pd.Series((df[prefix + 'high'].rolling(n).max() - df[prefix + 'close'])/atr, \
                     name = prefix + 'CPLUNGER_H'+ str(n))
    low  = pd.Series((df[prefix + 'close'] - df[prefix + 'low'].rolling(n).min())/atr, \
                     name = prefix + 'CPLUNGER_L'+ str(n))
    return pd.concat([high,low], sort = False, join='outer', axis=1)

#Donchian Channel
def DONCH_H(df, n, field = 'high', prefix = ''):
    DC_H = df[prefix + field].rolling(n).max()
    return pd.Series(DC_H, name = prefix + 'DONCH_H' + field[0].upper() + str(n))

def DONCH_L(df, n, field = 'low', prefix = ''):
    DC_L = df[prefix + field].rolling(n).min()
    return pd.Series(DC_L, name = prefix + 'DONCH_L'+ field[0].upper() + str(n))

def donch_h(df, n, field = 'high', prefix = ''):
    key = prefix + 'DONCH_H'+ field[0].upper() + str(n)
    df[key][-1] = max(df[prefix + field][-n:])
 
def donch_l(df, n, field = 'low', prefix = ''):
    key = prefix + 'DONCH_L'+ field[0].upper() + str(n)
    df[key][-1] = min(df[prefix + field][-n:])
    
def HEIKEN_ASHI(df, period1, prefix = ''):
    SM_O = df[prefix + 'open'].rolling(period1).mean()
    SM_H = df[prefix + 'high'].rolling(period1).mean()
    SM_L = df[prefix + 'low'].rolling(period1).mean()
    SM_C = df[prefix + 'close'].rolling(period1).mean()
    HA_C = pd.Series((SM_O + SM_H + SM_L + SM_C)/4.0, name = prefix + 'HAclose')
    HA_O = pd.Series(SM_O, name = prefix + 'HAopen')
    HA_H = pd.Series(SM_H, name = prefix + 'HAhigh')
    HA_L = pd.Series(SM_L, name = prefix + 'HAlow')
    for idx, dateidx in enumerate(HA_C.index):
        if idx >= (period1):
            HA_O[idx] = (HA_O[idx-1] + HA_C[idx-1])/2.0
        HA_H[idx] = max(SM_H[idx], HA_O[idx], HA_C[idx])
        HA_L[idx] = min(SM_L[idx], HA_O[idx], HA_C[idx])
    return pd.concat([HA_O, HA_H, HA_L, HA_C], sort = False, join='outer', axis=1)
    
def heiken_ashi(df, period, prefix = ''):
    ma_o = sum(df[prefix + 'open'][-period:])/float(period)
    ma_c = sum(df['close'][-period:])/float(period)
    ma_h = sum(df[prefix + 'high'][-period:])/float(period)
    ma_l = sum(df[prefix + 'low'][-period:])/float(period)
    df[prefix + 'HAclose'][-1] = (ma_o + ma_c + ma_h + ma_l)/4.0
    df[prefix + 'HAopen'][-1] = (df[prefix + 'HAopen'][-2] + df[prefix + 'HAclose'][-2])/2.0
    df[prefix + 'HAhigh'][-1] = max(ma_h, df[prefix + 'HAopen'][-1], df[prefix + 'HAclose'][-1])
    df[prefix + 'HAlow'][-1] = min(ma_l, df[prefix + 'HAopen'][-1], df[prefix + 'HAclose'][-1])

def BBANDS_STOP(df, n, nstd, prefix = ''):
    MA = pd.Series(df[prefix + 'close'].rolling(n).mean())
    MSD = pd.Series(df[prefix + 'close'].rolling(n).std())
    Upper = pd.Series(MA + MSD * nstd, name = prefix + 'BBSTOP_upper')
    Lower = pd.Series(MA - MSD * nstd, name = prefix + 'BBSTOP_lower')
    Trend = pd.Series(0, index = Lower.index, name = prefix + 'BBSTOP_trend')
    for idx, dateidx in enumerate(Upper.index):
        if idx >= n:
            Trend[idx] = Trend[idx-1]
            if (df.close[idx] > Upper[idx-1]):
                Trend[idx] = 1
            if (df.close[idx] < Lower[idx-1]):
                Trend[idx] = -1                
            if (Trend[idx]==1) and (Lower[idx] < Lower[idx-1]):
                Lower[idx] = Lower[idx-1]
            elif (Trend[idx]==-1) and (Upper[idx] > Upper[idx-1]):
                Upper[idx] = Upper[idx-1]
    return pd.concat([Upper,Lower, Trend], sort = False, join='outer', axis=1)

def bbands_stop(df, n, nstd, prefix = ''):
    ma = df[prefix + 'close'][-n:].mean()
    msd = df[prefix + 'close'][-n:].std()
    df[prefix + 'BBSTOP_upper'][-1] = ma + nstd * msd
    df[prefix + 'BBSTOP_lower'][-1] = ma - nstd * msd
    df[prefix + 'BBSTOP_trend'][-1] = df[prefix + 'BBSTOP_trend'][-2]
    if df[prefix + 'close'][-1] > df[prefix + 'BBSTOP_upper'][-2]:
        df[prefix + 'BBSTOP_trend'][-1] = 1
    if df[prefix + 'close'][-1] < df[prefix + 'BBSTOP_lower'][-2]:
        df[prefix + 'BBSTOP_trend'][-1] = -1
    if (df[prefix + 'BBSTOP_trend'][-1] == 1) and (df[prefix + 'BBSTOP_lower'][-1] < df[prefix + 'BBSTOP_lower'][-2]):
        df[prefix + 'BBSTOP_lower'][-1] = df[prefix + 'BBSTOP_lower'][-2]
    if (df[prefix + 'BBSTOP_trend'][-1] == -1) and (df[prefix + 'BBSTOP_upper'][-1] > df[prefix + 'BBSTOP_upper'][-2]):
        df[prefix + 'BBSTOP_upper'][-1] = df[prefix + 'BBSTOP_upper'][-2]

def FISHER(df, n, smooth_p = 0.7, smooth_i = 0.7, prefix = ''):
    roll_high = df[prefix + 'high'].rolling(n).max()
    roll_low  = df[prefix + 'low'].rolling(n).min()
    price_loc = (df[prefix + 'close'] - roll_low)/(roll_high - roll_low) * 2.0 - 1
    sm_price = pd.Series(price_loc.ewm(com = 1.0/smooth_p - 1, adjust = False).mean(), name = prefix + 'FISHER_P')
    fisher_ind = 0.5 * np.log((1 + sm_price)/(1 - sm_price))
    sm_fisher = pd.Series(fisher_ind.ewm(com = 1.0/smooth_i - 1, adjust = False).mean(), name = prefix + 'FISHER_I')
    return pd.concat([sm_price, sm_fisher], sort = False, join='outer', axis=1)

def fisher(df, n, smooth_p = 0.7, smooth_i = 0.7, prefix = ''):
    roll_high = max(df[prefix + 'high'][-n:])
    roll_low  = min(df[prefix + 'low'][-n:])
    price_loc = (df[prefix + 'close'][-1] - roll_low)*2.0/(roll_high - roll_low) - 1
    df[prefix + 'FISHER_P'][-1] = df[prefix + 'FISHER_P'][-2] * (1 - smooth_p) + smooth_p * price_loc
    fisher_ind = 0.5 * np.log((1 + df[prefix + 'FISHER_P'][-1])/(1 - df[prefix + 'FISHER_P'][-1]))
    df[prefix + 'FISHER_I'][-1] = df[prefix + 'FISHER_I'][-2] * (1 - smooth_i) + smooth_i * fisher_ind

def PCT_CHANNEL(df, n = 20, pct = 50, field = 'close', prefix = ''):
    out = pd.Series(index=df.index, name = prefix + 'PCT%sCH%s' % (pct, n))
    for idx, d in enumerate(df.index):
        if idx >= n:
            out[d] = np.percentile(df[prefix + field].iloc[max(idx-n,0):idx], pct)
    return out

def pct_channel(df, n = 20, pct = 50, field = 'close', prefix = ''):
    key =  prefix + 'PCT%sCH%s' % (pct, n)
    df[key][-1] = np.percentile(df[prefix + field][-n:], pct)

def COND_PCT_CHAN(df, n = 20, pct = 50, field = 'close', direction=1, prefix = ''):
    out = pd.Series(index=df.index, name = prefix + 'C_CH%s_PCT%s' % (n, pct))
    for idx, d in enumerate(df.index):
        if idx >= n:
            ts = df[prefix + field].iloc[max(idx-n,0):idx]
            cutoff = np.percentile(ts, pct)
            ind = (ts*direction>=cutoff*direction)
            filtered = ts[ind]
            ranks = filtered.rank(ascending=False)
            tot_s = sum([filtered[dt] * ranks[dt] * (seq + 1) for seq, dt in enumerate(filtered.index)])
            tot_w = sum([ranks[dt] * (seq + 1) for seq, dt in enumerate(filtered.index)])    
            out[d] = tot_s/tot_w
    return out
   
def VCI(df, n, rng = 8, prefix = ''):
    if n > 7:
        varA = df[prefix + 'high'].rolling(rng).max() - df[prefix + 'low'].rolling(rng).min()
        varB = varA.shift(rng)
        varC = varA.shift(rng*2)
        varD = varA.shift(rng*3)
        varE = varA.shift(rng*4)
        avg_tr = (varA+varB+varC+varD+varE)/25.0
    else:
        tr = pd.concat([df[prefix + 'high'] - df['low'], abs(df[prefix + 'close'] - df[prefix + 'close'].shift(1))], \
                       sort = False, join='outer', axis=1).max(1)
        avg_tr = tr.rolling(n).mean() * 0.16
    avg_pr = (df[prefix + 'high'].rolling(n).mean() + df[prefix + 'low'].rolling(n).mean())/2.0
    VO = pd.Series((df[prefix + 'open'] - avg_pr)/avg_tr, name = prefix + 'VCIO')
    VH = pd.Series((df[prefix + 'high'] - avg_pr)/avg_tr, name = prefix + 'VCIH')
    VL = pd.Series((df[prefix + 'low'] - avg_pr)/avg_tr, name = prefix + 'VCIL')
    VC = pd.Series((df[prefix + 'close'] - avg_pr)/avg_tr, name = prefix + 'VCIC')
    return pd.concat([VO, VH, VL, VC], sort = False, join='outer', axis=1)

def TEMA(ts, n, prefix = ''):
    n = int(n)
    ts_ema1 = pd.Series( ts.ewm(span = n, adjust = False).mean(), name = prefix + 'EMA' + str(n) )
    ts_ema2 = pd.Series( ts_ema1.ewm(span = n, adjust = False).mean(), name = prefix + 'EMA2' + str(n) )
    ts_ema3 = pd.Series( ts_ema2.ewm(span = n, adjust = False).mean(), name = prefix + 'EMA3' + str(n) )
    ts_tema = pd.Series( 3 * ts_ema1 - 3 * ts_ema2 + ts_ema3, name = prefix + 'TEMA' + str(n) )
    return ts_tema
    
def SVAPO(df, period = 8, cutoff = 1, stdev_h = 1.5, stdev_l = 1.3, stdev_period = 100, prefix = ''):
    HA = HEIKEN_ASHI(df, 1, prefix = prefix)
    haCl = (HA[prefix + 'HAopen'] + HA[prefix + 'HAclose'] + HA[prefix + 'HAhigh'] + HA[prefix + 'HAlow'])/4.0
    haC = TEMA( haCl, 0.625 * period, prefix = prefix)
    vave = MA(df, 5 * period, field = 'volume', prefix = prefix).shift(1)
    vc = pd.concat([df[prefix + 'volume'], vave*2], sort = False, axis=1).min(axis=1)
    vtrend = TEMA(LINEAR_REG_SLOPE(df['volume'], period), period, prefix = prefix)
    UpD = pd.Series(vc)
    DoD = pd.Series(-vc)
    UpD[(haC<=haC.shift(1)*(1+cutoff/1000.0))|(vtrend < vtrend.shift(1))] = 0
    DoD[(haC>=haC.shift(1)*(1-cutoff/1000.0))|(vtrend > vtrend.shift(1))] = 0
    delta_sum = (UpD + DoD).rolling(period).sum()/(vave+1)
    svapo = pd.Series(TEMA(delta_sum, period, prefix = prefix), name = prefix + 'SVAPO_%s' % period)
    svapo_std = svapo.rolling(stdev_period).std()
    svapo_ub = pd.Series(svapo_std * stdev_h, name = prefix + 'SVAPO_UB%s' % period)
    svapo_lb = pd.Series(-svapo_std * stdev_l, name = prefix + 'SVAPO_LB%s' % period)
    return pd.concat([svapo, svapo_ub, svapo_lb], sort = False, join='outer', axis=1)

def LINEAR_REG_SLOPE(ts, n):
    sumbars = n*(n-1)*0.5
    sumsqrbars = (n-1)*n*(2*n-1)/6.0
    lrs = pd.Series(index = ts.index, name = 'LINREGSLOPE_%s' % n)
    for idx, d in enumerate(ts.index):
        if idx >= n-1:
            y_array = ts[idx-n+1:idx+1].values
            x_array = np.arange(n-1,-1,-1)
            lrs[idx] = (n * np.dot(x_array, y_array) - sumbars * y_array.sum())/(sumbars*sumbars-n*sumsqrbars)
    return lrs

def DVO(df, w = [0.5, 0.5, 0, 0], N = 2, s = [0.5, 0.5], M = 252, prefix = ''):
    ratio = df[prefix + 'close']/(df[prefix + 'high'] * w[0] + df[prefix + 'low'] * w[1] \
                                  + df[prefix + 'open'] * w[2] + df[prefix + 'close'] * w[3])
    theta = pd.Series(index = df.index)
    dvo = pd.Series(index = df.index, name=prefix + 'DV%s_%s' % (N, M))
    ss = np.array(list(reversed(s)))
    for idx, d in enumerate(ratio.index):
        if idx >= N-1:
            y = ratio[idx-N+1:idx+1].values
            theta[idx] = np.dot(y, ss)
        if idx >= M+N-2:
            ts = theta[idx-(M-1):idx+1]
            dvo[idx] = stats.percentileofscore(ts.values, theta[idx])
    return dvo

def PSAR(df, iaf = 0.02, maxaf = 0.2, incr = 0, prefix = ''):
    if incr == 0:
        incr = iaf
    psar = pd.Series(index = df.index, name=prefix + 'PSAR_VAL')
    direction = pd.Series(index = df.index, name=prefix + 'PSAR_DIR')
    bull = True
    ep = df[prefix + 'low'][0]
    hp = df[prefix + 'high'][0]
    lp = df[prefix + 'low'][0]
    af = iaf
    for idx, d in enumerate(df.index):
        if idx == 0:
            continue
        if bull:
            psar[idx] = psar[idx - 1] + af * (hp - psar[idx - 1])
        else:
            psar[idx] = psar[idx - 1] + af * (lp - psar[idx - 1])
        reverse = False
        if bull:
            if df.low[idx] < psar[idx]:
                bull = False
                reverse = True
                psar[idx] = hp
                lp = df[prefix + 'low'][idx]
                af = iaf
        else:
            if df.high[idx] > psar[idx]:
                bull = True
                reverse = True
                psar[idx] = lp
                hp = df[prefix + 'high'][idx]
                af = iaf
        if not reverse:
            if bull:
                if df[prefix + 'high'][idx] > hp:
                    hp = df[prefix + 'high'][idx]
                    af = min(af + incr, maxaf)
                psar[idx] = min(psar[idx], df['low'][idx - 1], df[prefix + 'low'][idx - 2])
            else:
                if df.low[idx] < lp:
                    lp = df[prefix + 'low'][idx]
                    af = min(af + incr, maxaf)
                psar[idx] = max(psar[idx], df[prefix + 'high'][idx - 1], df[prefix + 'high'][idx - 2])
                direction[idx] = -1
        if bull:
            direction[idx] = 1
        else:
            direction[idx] = -1
    return pd.concat([psar, direction], sort = False, join='outer', axis=1)

def SPBFILTER(df, n1 = 40, n2 = 60, n3 = 0, field = 'close', prefix = ''):
    if n3 == 0:
        n3 = int((n1 + n2)/2)
    a1 = 5.0/n1
    a2 = 5.0/n2
    B = [a1-a2, a2-a1]
    A = [1, (1-a1)+(1-a2), -(1-a1)*(1-a2)]
    PB = pd.Series(signal.lfilter(B, A, df[prefix + field]), name = prefix + 'SPB_%s_%s' % (n1, n2))
    RMS = pd.Series((PB*PB).rolling(n3).mean()**0.5, name = prefix + 'SPBRMS__%s_%s' % (n1, n2))
    return pd.concat([PB, RMS], sort = False, join='outer', axis=1)

def spbfilter(df, n1 = 40, n2 = 60, n3 = 0, field = 'close', prefix = ''):
    if n3 == 0:
        n3 = int((n1 + n2)/2)
    a1 = 5.0/n1
    a2 = 5.0/n2
    SPB_key = prefix + 'SPB_%s_%s' % (n1, n2)
    RMS_key = prefix + 'SPBRMS_%s_%s' % (n1, n2)
    df[SPB_key][-1] = df[prefix + field][-1]*(a1-a2) + df[prefix + field][-2]*(a2-a1) \
                    + df[SPB_key][-2]*(2-a1-a2) - df[SPB_key][-2]*(1-a1)*(1-a2)
    df[RMS_key][-1] = np.sqrt((df[SPB_key][(-n3):]**2).mean())
    
def MA_RIBBON(df, ma_series, prefix = ''):
    ma_array = np.zeros([len(df), len(ma_series)])
    ema_list = []
    for idx, ma_len in enumerate(ma_series):
        ema_i = EMA(df, n = ma_len, field = 'close', prefix = prefix)
        ma_array[:, idx] = ema_i
        ema_list.append(ema_i)
    corr = np.empty([len(df)])
    pval = np.empty([len(df)])
    dist = np.empty([len(df)])
    corr[:] = np.NAN
    pval[:] = np.NAN
    dist[:] = np.NAN
    max_n = max(ma_series)
    for idy in range(len(df)):
        if idy >= max_n - 1:
            corr[idy], pval[idy] = stats.spearmanr(ma_array[idy,:], list(range(len(ma_series), 0, -1)))
            dist[idy] = max(ma_array[idy,:]) - min(ma_array[idy,:])
    corr_ts = pd.Series(corr*100, index = df.index, name = prefix + "MARIBBON_CORR")
    pval_ts = pd.Series(pval*100, index = df.index, name = prefix + "MARIBBON_PVAL")
    dist_ts = pd.Series(dist, index = df.index, name = prefix + "MARIBBON_DIST")
    return pd.concat([corr_ts, pval_ts, dist_ts] + ema_list, sort = False, join='outer', axis=1)
    
def ma_ribbon(df, ma_series, prefix = ''):
    ma_array = np.zeros([len(df)])
    for idx, ma_len in enumerate(ma_series):
        key = prefix + 'EMA_CLOSE_' + str(ma_len)
        ema(df, ma_len, field = 'close', prefix = prefix)
        ma_array[idx] = df[key][-1]
    corr, pval = stats.spearmanr(ma_array, list(range(len(ma_series), 0, -1)))
    dist = max(ma_array) - min(ma_array)
    df[prefix + "MARIBBON_CORR"][-1] = corr * 100
    df[prefix + "MARIBBON_PVAL"][-1] = pval * 100
    df[prefix + "MARIBBON_DIST"][-1] = dist

def DT_RNG(df, win = 2, ratio = 0.7, prefix = ''):
    if win == 0:
        tr_ts = pd.concat([(df[prefix + 'high'].rolling(2).max() - df[prefix + 'close'].rolling(2).min())*0.5,
                        (df[prefix + 'close'].rolling(2).max() - df[prefix + 'low'].rolling(2).min())*0.5,
                        df[prefix + 'high'] - df[prefix + 'close'],
                        df[prefix + 'close'] - df[prefix + 'low']],
                        sort = False, join='outer', axis=1).max(axis=1)
    else:
        tr_ts = pd.concat([df[prefix + 'high'].rolling(win).max() - df[prefix + 'close'].rolling(win).min(),
                           df[prefix + 'close'].rolling(win).max() - df[prefix + 'low'].rolling(win).min()],
                       sort = False, join='outer', axis=1).max(axis=1)
    return pd.Series(tr_ts, name = prefix + 'DTRNG%s_%s' % (win, ratio))

def dt_rng(df, win = 2, ratio = 0.7, prefix = ''):
    key = prefix + 'DTRNG%s_%s' % (win, ratio)
    if win > 0:
        df[key][-1] = max(max(df[prefix + 'high'][-win:]) - min(df[prefix + 'close'][-win:]),
                                max(df[prefix + 'close'][-win:]) - min(df[prefix + 'low'][-win:]))
    elif win == 0:
        df[key][-1] = max(max(df[prefix + 'high'][-2:]) - min(df[prefix + 'close'][-2:]),
                                max(df[prefix + 'close'][-2:]) - min(df[prefix + 'low'][-2:]))
        df[key][-1] = max(df[key][-1] * 0.5, df[prefix + 'high'][-1] - df[prefix + 'close'][-1],
                                df[prefix + 'close'][-1] - df[prefix + 'low'][-1])

def KUMO_CLOUD(df, n = 26, short_ratio = 0.35, long_ratio = 2.0, prefix = ''):
    short_win = int(short_ratio * n)
    long_win = int(long_ratio * n)
    conv_line = pd.Series((df[prefix + 'high'].rolling(short_win).max() \
                           + df[prefix + 'low'].rolling(short_win).min())/2.0, \
                          index = df.index, name = prefix + "KUMO_TK_%s" % str(n))
    base_line = pd.Series((df[prefix + 'high'].rolling(n).max()  \
                           + df[prefix + 'low'].rolling(n).min()) / 2.0, \
                          index=df.index, name = prefix + "KUMO_KJ_%s" % str(n))
    lspan_a = pd.Series((conv_line + base_line)/2.0, index = df.index, name = prefix + "KUMO_SKA_%s" % str(n))
    lspan_b = pd.Series((df[prefix + 'high'].rolling(long_win).max()  \
                         + df[prefix + 'low'].rolling(long_win).min())/2.0, \
                        index = df.index, name = prefix + "KUMO_SKB_%s" % str(n))
    return pd.concat([conv_line, base_line, lspan_a, lspan_b], sort = False, join='outer', axis=1)

def kumo_cloud(df, n = 26, short_ratio = 0.35, long_ratio = 2.0, prefix = ''):
    short_win = int(short_ratio * n)
    long_win = int(long_ratio * n)
    df[prefix + 'KUMO_TK_' % str(n)][-1] = (max(df[prefix + 'high'][(-short_win):]) \
                                             + min(df[prefix + 'low'][(-short_win):]))/2.0
    df[prefix + 'KUMO_KJ_' % str(n)][-1] = (max(df[prefix + 'high'][(-n):]) \
                                             + min(df[prefix + 'low'][(-n):]))/2.0
    df[prefix + 'KUMO_SKA_' % str(n)][-1] = (df[prefix + 'KUMO_TK_' % str(n)][-1] \
                                              + df[prefix + 'KUMO_KJ_' % str(n)][-1])/2.0
    df[prefix + 'KUMO_SKB_' % str(n)][-1] = (max(df[prefix + 'high'][(-long_win):]) \
                                              + min(df[prefix + 'low'][(-long_win):]))/2.0
