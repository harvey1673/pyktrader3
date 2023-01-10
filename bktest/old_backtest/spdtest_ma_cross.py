import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from bktest_ma_cross import *

class SpdMACrossSim(MACrossSim):
    def __init__(self, config):        
        super(SpdMACrossSim, self).__init__(config)

    def process_config(self, config):
        super(SpdMACrossSim, self).process_config(config)        
        self.weights = config.get('weights', [1.0, -1.0])

    def process_data(self, df):
        if self.freq in self.data_store:
            xdf = self.data_store[self.freq]
        else:
            for field in ['open', 'close']:
                df[field] = 0
                for asset, w in zip(self.assets, self.weights):
                    df[field] += w * (df[(asset, field)])
            df['high'] = df[['open', 'close']].max(axis=1)
            df['low'] = df[['open', 'close']].min(axis=1)
            for field in ['volume', 'openInterest']:
                df[field] = 0.0
                #for asset, w in zip(self.assets, self.weights):
                df[field] = pd.concat([abs(w) * df[(asset, field)] for asset, w in zip(self.assets, self.weights)], \
                                      sort = False, axis=1).min(axis=1)
            xdf = df
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq(df, self.freq, extra_cols=['contract'])
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                xdf = dh.day_split(df, sp, extra_cols=['contract'])
            else:
                print("uncovered data case")
                self.data_store[self.freq] = df
            self.data_store[self.freq] = xdf
        self.df = xdf.copy()
        for idx, win in enumerate(self.win_list):
            self.df['MA' + str(idx + 1)] = self.ma_func(self.df, win).shift(1)
        if self.use_chan:
            self.df['chan_h'] = self.chan_high(self.df, self.channel).shift(2)
            self.df['chan_l'] = self.chan_low(self.df, self.channel).shift(2)
        else:
            self.df['chan_h'] = pd.Series(0, index = self.df.index)
            self.df['chan_l'] = pd.Series(0, index = self.df.index)
        self.df = self.df.dropna()
        self.df['closeout'] = 0.0
        self.df['cost'] = 0.0
        self.df['pos'] = 0.0
        self.df['traded_price'] = self.df['open']
        self.trade_cost = sum([ abs(w) * offset for offset, w in zip(self.offset, self.weights)])

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_func']  = 'run_vec_sim'
    sim_config['sim_class'] = 'bktest.spdtest_ma_cross.SpdMACrossSim'
    sim_config['scen_keys'] = ['freq', 'win_list', 'channel_ratio']
    sim_config['sim_freq'] = 'm'
    sim_config['sim_name']   = 'MA3_sp_190201'
    sim_config['products']   = ['rb', 'hc', 'i', 'j','jm']
    sim_config['start_date'] = '20150701'
    sim_config['end_date']   = '20190201'
    sim_config['win_list'] =[[5, 10, 20], [5, 10, 40], [5, 20, 40], [5, 20, 80], [5, 30, 60], [5, 30, 90], \
                             [10, 20, 40], [10, 20, 60], [10, 20, 80], [10, 20, 120], [10, 40, 80], [10, 40, 120], \
                             [10, 40, 160], [10, 60, 120]]
    sim_config['channel_ratio'] = [0.0, 0.25, 0.5]
    sim_config['freq'] = ['s2', 's3', 's4']
    sim_config['offset']    = 1
    config = {'capital': 10000,
              'trans_cost': 0.0,
              'unit': 1,
              'stoploss': 0.0,
              'ma_func': 'dh.EMA',
              'channel_func': ['dh.DONCH_H', 'dh.DONCH_L'],
              'channel_args': [{}, {}],
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