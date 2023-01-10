import sys
import json
import pandas as pd
import numpy as np
import pycmqlib3.analytics.data_handler as dh
from pycmqlib3.utility.misc import sign, day_split_dict
from pycmqlib3.core.trade_position import TradePos, TargetTrailTradePos
from . backtest import StratSim, simdf_to_trades1, simdf_to_trades2

class MACrossSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(MACrossSim, self).__init__(config)

    def process_config(self, config):
        self.assets = config['assets']
        self.tick_base = config['tick_base']
        self.offset = config['offset']
        self.win_list = config['win_list']
        self.ma_func = eval(config['ma_func'])
        self.freq = config['freq']
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.chan_ratio = config['channel_ratio']
        if self.chan_ratio > 0:
            self.use_chan = True
            self.channel = int(self.chan_ratio * self.win_list[-1])
        else:
            self.use_chan = False
            self.channel = 0
        self.chan_func = config['channel_func']
        self.chan_high = eval(self.chan_func[0])
        self.chan_low  = eval(self.chan_func[1])
        self.high_args = config['channel_args'][0]
        self.low_args = config['channel_args'][1]
        self.weights = [1.0]

    def process_data(self, df):
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
        self.trade_cost = self.offset

    def run_vec_sim(self):
        xdf = self.df
        long_signal = pd.Series(np.nan, index = xdf.index)
        last = len(self.win_list)
        long_flag = (xdf['MA1'] > xdf['MA2']) & (xdf['MA1'] > xdf['MA'+str(last)])
        if self.use_chan:
            long_flag = long_flag & (xdf['open'] >= xdf['chan_h'])
        long_signal[long_flag] = 1
        cover_flag = (xdf['MA1'] <= xdf['MA'+str(last)])
        if self.use_chan:
            cover_flag = cover_flag | (xdf['open'] < xdf['chan_l'])
        long_signal[cover_flag] = 0
        long_signal = long_signal.fillna(method='ffill').fillna(0)
        short_signal = pd.Series(np.nan, index = xdf.index)
        short_flag = (xdf['MA1'] <= xdf['MA2']) & (xdf['MA1'] <= xdf['MA'+str(last)])
        if self.use_chan:
            short_flag = short_flag & (xdf['open'] <= xdf['chan_l'])
        short_signal[short_flag] = -1
        cover_flag = (xdf['MA1'] > xdf['MA'+str(last)])
        if self.use_chan:
            cover_flag = cover_flag | (xdf['open'] > xdf['chan_h'])
        short_signal[cover_flag] = 0
        short_signal = short_signal.fillna(method='ffill').fillna(0)
        if len(xdf[(long_signal>0) & (short_signal<0)])>0:
            print(xdf[(long_signal > 0) & (short_signal < 0)])
            print("something wrong with the position as long signal and short signal happen the same time")
        xdf['pos'] = long_signal + short_signal
        xdf.ix[-1, 'pos'] = 0.0
        xdf['cost'] = abs(xdf['pos'] - xdf['pos'].shift(1)) * self.trade_cost
        xdf['cost'] = xdf['cost'].fillna(0.0)
        xdf['traded_price'] = xdf.open
        closed_trades = simdf_to_trades1(xdf, slippage = self.offset )
        return ([xdf], closed_trades)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_func']  = 'run_vec_sim'
    sim_config['sim_class'] = 'bktest.bktest_ma_cross.MACrossSim'
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