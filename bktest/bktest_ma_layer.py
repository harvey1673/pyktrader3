import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import data_handler as dh
from backtest import *
import trade_position

day_split_dict = {'s1': [300, 2115],
                  's2': [300, 1500, 2115],
                  's3': [300, 1500, 1900, 2115],
                  's4': [300, 1500, 1630, 1900, 2115],}

class MALayerSim(StratSim):
    def __init__(self, config):
        super(MALayerSim, self).__init__(config)

    def process_config(self, config):
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.upper_win = config['upper_win']
        self.lower_win = config['lower_win']
        self.ma_func = eval(config['ma_func'])
        self.corr_th = config['corr_th']
        self.close_daily = config['close_daily']
        trade_freq = config['trading_freq']
        self.upper_freq = trade_freq[0]
        self.lower_freq = trade_freq[1]
        self.tcost = config['trans_cost']
        #self.SL = config['stoploss']
        self.unit = config['unit']
        self.chan_ratio = config['channel_ratio']
        if self.chan_ratio > 0:
            self.use_chan = True
            self.channel = int(self.chan_ratio * self.upper_win[-1])
        else:
            self.use_chan = False
            self.channel = 0

    def process_data(self, df):
        df_list = []
        for idx, freq in enumerate([self.upper_freq, self.lower_freq]):
            if ('m' in freq) and (int(freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq(df, freq, extra_cols = ['contract'])
            elif ('s' in freq):
                sp = day_split_dict[freq]
                xdf = dh.day_split(df, sp, extra_cols = ['contract'])
            else:
                xdf = df
            df_list.append(xdf)
        xdf, self.df = df_list
        for idx, win in enumerate(self.lower_win):
            self.df['MA' + str(idx + 1)] = self.ma_func(self.df, win).shift(1).fillna(method = 'bfill')
        ma_ribbon = dh.MA_RIBBON(xdf, self.upper_win)
        xdf['RIBBON_CORR'] = ma_ribbon['MARIBBON_CORR']
        xdf['RIBBON_PVAL'] = ma_ribbon['MARIBBON_PVAL']
        xdf['filter'] = (xdf['RIBBON_CORR'] > self.corr_th) * 1.0 - (xdf['RIBBON_CORR'] < -self.corr_th) * 1.0
        if self.use_chan:
            xdf['CHAN_H'] = dh.DONCH_H(xdf, self.channel, field='high')
            xdf['CHAN_L'] = dh.DONCH_L(xdf, self.channel, field='low')
            xdf['ATR'] = dh.ATR(xdf, self.channel)
        else:
            xdf['CHAN_H'] = pd.Series(-10000, index=xdf.index)
            xdf['CHAN_L'] = pd.Series(10000, index=xdf.index)
            xdf['ATR'] = pd.Series(10000, index=xdf.index)
        #xdf['tr_stop_h'] = xdf['CHAN_H'] - (self.df['ATR'] * self.SL / self.tick_base).astype('int') * self.tick_base
        #xdf['tr_stop_l'] = xdf['CHAN_L'] + (self.df['ATR'] * self.SL / self.tick_base).astype('int') * self.tick_base
        xdata = pd.concat([xdf['filter'].shift(1),
                        #xdf['CHAN_H'].shift(1), xdf['CHAN_L'].shift(1), xdf['ATR'].shift(1).fillna(method='bfill') \
                        ], axis=1, keys = ['filter', \
                        #'chan_h', 'chan_l', 'atr'\
                        ])
        self.df = self.df.join(xdata, how='left').fillna(method='ffill').dropna()

    def run_vec_sim(self):
        xdf = self.df.copy()
        long_signal = pd.Series(np.nan, index = xdf.index)
        last = len(self.lower_win)
        long_flag = (xdf['filter'] > 0) & (xdf['MA1'] > xdf['MA2']) & (xdf['MA1'] > xdf['MA'+str(last)])
        long_signal[long_flag] = 1
        cover_flag = (xdf['MA1'] <= xdf['MA'+str(last)])
        long_signal[cover_flag] = 0
        long_signal = long_signal.fillna(method='ffill').fillna(0)
        short_signal = pd.Series(np.nan, index = xdf.index)
        short_flag = (xdf['filter'] < 0) & (xdf['MA1'] <= xdf['MA2']) & (xdf['MA1'] <= xdf['MA'+str(last)])
        short_signal[short_flag] = -1
        cover_flag = (xdf['MA1'] > xdf['MA'+str(last)])
        short_signal[cover_flag] = 0
        short_signal = short_signal.fillna(method='ffill').fillna(0)
        if len(xdf[(long_signal>0) & (short_signal<0)])>0:
            print(xdf[(long_signal > 0) & (short_signal < 0)])
            print("something wrong with the position as long signal and short signal happen the same time")
        xdf['pos'] = long_signal + short_signal
        xdf.ix[-1, 'pos'] = 0.0
        xdf['cost'] = abs(xdf['pos'] - xdf['pos'].shift(1)) * (self.offset + xdf['open'] * self.tcost)
        xdf['cost'] = xdf['cost'].fillna(0.0)
        xdf['traded_price'] = xdf.open
        closed_trades = simdf_to_trades1(xdf, slippage = self.offset )
        return ([xdf], closed_trades)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_func']  = 'run_vec_sim'
    sim_config['sim_class'] = 'bktest.bktest_ma_layer.MALayerSim'
    sim_config['scen_keys'] = ['trading_freq', 'lower_win', 'corr_th']
    sim_config['sim_freq'] = 'm'
    sim_config['sim_name']   = 'MALayer_190125'
    sim_config['products']   = ['rb', 'hc', 'i', 'j','jm']
    sim_config['start_date'] = '20150701'
    sim_config['end_date']   = '20190125'
    sim_config['lower_win'] = [[5, 10], [5, 20], [5, 40], [10, 20], [10, 40], [10, 80]]
    sim_config['corr_th'] = [0.9, 0.8, 0.6, 0.4]
    sim_config['trading_freq'] = [['s2', '30m'], ['s2', '15m'], ['s2', '5m'], ['s3', '30m'], ['s3', '15m'], ['s3', '5m'], \
                                  ['60m', '15m'], ['60m', '5m'], ['30m', '5m']]
    sim_config['offset'] = 1
    config = {'capital': 10000,
              'trans_cost': 0.0,
              'unit': 1,
              'ma_func': 'dh.EMA',
              'upper_win': list(range(10, 90, 10)),
              'channel_ratio': 0.5,
              'close_daily': False,
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