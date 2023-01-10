import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from bktest_chanbreak import *

class SpdChanBreakSim(ChanBreakSim):
    def __init__(self, config):
        super(SpdChanBreakSim, self).__init__(config)

    def process_config(self, config):
        super(SpdChanBreakSim, self).process_config(config)
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
        for i in range(len(self.channel)):
            xdf['chan_h' + str(i)] = self.chan_high(xdf, self.channel[i], **self.chan_func['high']['args']).shift(1)
            xdf['chan_l' + str(i)] = self.chan_low(xdf, self.channel[i], **self.chan_func['low']['args']).shift(1)
        xdf['ATR'] = dh.STDEV(xdf, n=self.SL_win).shift(1)
        xdf['chan_h0'] = xdf['chan_h0'] + self.entry_buf * xdf['ATR']
        xdf['chan_l0'] = xdf['chan_l0'] - self.entry_buf * xdf['ATR']
        xdf['tr_stop_h'] = xdf['chan_h0'] - self.SL * xdf['ATR']
        xdf['tr_stop_l'] = xdf['chan_l0'] + self.SL * xdf['ATR']
        xdf['last_min'] = 2059
        if (self.data_freq == 'm') and (self.freq != 'm1'):
            xdf = df.join(xdf[['tr_stop_h', 'tr_stop_l', 'chan_h0', 'chan_l0', \
                               'chan_h1', 'chan_l1', 'ATR', 'last_min']], how='left').fillna(method='ffill')
        xdf = xdf.dropna()
        for field in ['chan_h0', 'chan_l0', 'tr_stop_h', 'tr_stop_l']:
            xdf[field] = (xdf[field]/self.tick_base[0]).astype('int') * self.tick_base[0]
        xdf['long_exit'] = pd.concat([xdf['chan_l1'], xdf['tr_stop_h']], sort = False, join='outer', axis=1).max(axis=1)
        xdf['short_exit'] = pd.concat([xdf['chan_h1'], xdf['tr_stop_l']], sort = False, join='outer', axis=1).min(axis=1)
        xdf['closeout'] = 0.0
        xdf['cost'] = 0.0
        xdf['pos'] = 0.0
        xdf['traded_price'] = xdf['open']
        self.df = xdf
        self.trade_cost = sum([ abs(w) * offset for offset, w in zip(self.offset, self.weights)])

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class'] = 'bktest.spdtest_chanbreak.ChanBreakSim'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['sim_freq'] = 'm'
    sim_config['scen_keys'] = ['stoploss_win', 'stoploss', 'channel']
    sim_config['sim_name'] = 'chanbreak_181028'
    sim_config['products'] = ['rb', 'hc', 'i', 'j', 'jm']
    sim_config['start_date'] = '20150901'
    sim_config['end_date'] = '20181028'
    sim_config['stoploss'] = [0.5, 1.0, 2.0, 3.0]
    sim_config['stoploss_win'] = [10, 20]
    sim_config['channel'] = [[10, 3], [10, 5], [15, 3], [15, 5], [15, 10], [20, 5], [20, 10], [25, 5], [25, 10],
                             [25, 15]]
    sim_config['pos_class'] = 'trade_position.TradePos'
    chan_func = {'high': {'func': 'dh.DONCH_H', 'args': {}},
                 'low': {'func': 'dh.DONCH_L', 'args': {}},
                 }

    sim_config['pos_class'] = 'trade_position.TradePos'
    sim_config['offset'] = 1
    config = {'capital': 10000,
              'trans_cost': 0.0,
              'unit': 1,
              'freq': 1,
              'stoploss': 0.0,
              'close_daily': False,
              'chan_func': chan_func,
              'trading_mode': 'match'
              }
    sim_config['config'] = config
    with open(filename, 'w') as outfile:
        json.dump(sim_config, outfile)
    return sim_config

if __name__=="__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("need to input a file name for config file")
    else:
        gen_config_file(args[0])
    pass