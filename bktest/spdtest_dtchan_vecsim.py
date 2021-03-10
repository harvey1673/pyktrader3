import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from bktest_dtchan_vecsim import *

class SpdDTChanSim(DTChanSim):
    def __init__(self, config):
        super(SpdDTChanSim, self).__init__(config)

    def process_config(self, config):
        super(SpdDTChanSim, self).process_config(config)
        self.weights = config.get('weights', [1.0, -1.0])

    def process_data(self, df):
        for field in ['open', 'close']:
            df[field] = 0
            for asset, w in zip(self.assets, self.weights):
                df[field] += w * (df[(asset, field)])
        df['high'] = df[['open', 'close']].max(axis=1)
        df['low'] = df[['open', 'close']].min(axis=1)
        for field in ['volume', 'openInterest']:
            df[field] = 0.0
            # df[field] = pd.concat([abs(w) * df[(asset, field)] for asset, w in zip(self.assets, self.weights)],
            #                       sort = False, axis=1).min(axis=1)
        if self.freq in self.data_store:
            xdf = self.data_store[self.freq]
        else:
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq(df, self.freq, extra_cols=['contract'])
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                xdf = dh.day_split(df, sp, extra_cols=['contract'])
            else:
                print("uncovered data case")
                self.data_store[self.freq] = df
            self.data_store[self.freq] = xdf
        xdf = xdf.copy()
        if self.win == 0:
            tr= pd.concat([xdf.high - xdf.low, abs(xdf.close - xdf.close.shift(1))],
                          join='outer', axis=1).max(axis=1)
        elif self.win < -1:
            abs_win = abs(self.win)
            tr = pd.concat([xdf.high - xdf.close, xdf.close - xdf.low], join='outer', axis=1).max(axis=1)
            for win in range(2, abs_win + 1):
                fact = np.sqrt(1.0/win)
                tr = pd.concat([(xdf.high.rolling(win).max() - xdf.close.rolling(win).min()) * fact, \
                            (xdf.close.rolling(win).max() - xdf.low.rolling(win).min()) * fact, tr], join='outer', axis=1).max(axis=1)
        else:
            tr= pd.concat([xdf.high.rolling(self.win).max() - xdf.close.rolling(self.win).min(),
                            xdf.close.rolling(self.win).max() - xdf.low.rolling(self.win).min()],
                            join='outer', axis=1).max(axis=1)/np.sqrt(self.win)
        xdf['tr'] = tr
        xdf['last_min'] = 2059
        xdf['chan_h'] = self.chan_high(xdf, self.chan, **self.chan_func['high']['args'])
        xdf['chan_l'] = self.chan_low(xdf, self.chan, **self.chan_func['low']['args'])
        xdf['ATR'] = dh.ATR(xdf, 20)
        xdf['ma'] = xdf.close.rolling(max(self.machan, 1)).mean()
        xdf['rng'] = pd.DataFrame([self.min_rng * xdf['ATR'], self.k * xdf['tr']]).max()
        xdf['upper'] = xdf['open'] + xdf['rng'].shift(1) * (1 + \
                        (((xdf['open'] - xdf['ma'].shift(1)) * misc.sign(self.f) < - self.ma_width * xdf['ATR'].shift(1)) & self.use_chan) *self.f)
        xdf['lower'] = xdf['open'] - xdf['rng'].shift(1) * (1 + \
                        (((xdf['open'] - xdf['ma'].shift(1)) * misc.sign(self.f) > self.ma_width * xdf['ATR'].shift(1)) & self.use_chan) *self.f)
        xdata = pd.concat([xdf['upper'], xdf['lower'], xdf['chan_h'].shift(1), xdf['chan_l'].shift(1),
                           xdf['open'], xdf['last_min'], xdf['ATR'].shift(1)], axis=1,
                          keys=['upper','lower', 'chan_h', 'chan_l', 'xopen', 'last_min', 'ATR'])
        self.df = df.join(xdata, how='left').fillna(method='ffill').dropna()
        self.df.reset_index(inplace=True)
        self.df['closeout'] = 0.0
        self.df['cost'] = 0.0
        self.df['pos'] = 0.0
        self.df['traded_price'] = self.df['open']
        self.trade_cost = sum([ abs(w) * offset for offset, w in zip(self.offset, self.weights)])
