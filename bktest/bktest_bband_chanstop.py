import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import json
import misc
import data_handler as dh
import pandas as pd
import numpy as np
import datetime
from backtest import *

class BBChanStop(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(BBChanStop, self).__init__(config)

    def process_config(self, config):
        self.close_daily = config['close_daily']
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.freq = config['freq']
        self.data_freq = config['data_freq']
        self.band_ratio = config['band_ratio']
        self.ma_func = eval(config.get('ma_func', 'dh.MA'))
        self.band_func = eval(config.get('band_func', 'dh.ATR'))
        self.boll_len = config['boll_len']
        self.filter_func = eval(config.get('filter_func', 'dh.CCI'))
        self.filter_len = int(config.get('filter_ratio', 0.5) * self.boll_len)
        self.filter_thres = config.get('filter_thres', [0, 0])
        self.channel = int(config.get('chan_ratio', 0.5) * self.boll_len)
        self.pos_class = config['pos_class']
        self.pos_args  = config['pos_args']        
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', [1])
        self.SL = config['stoploss'][0] * self.band_ratio
        self.chan_entry = config['stoploss'][1] * self.band_ratio if config['stoploss'][1]>=0 else -1
        self.channel_mode = config.get('channel_mode', 'HL')
     
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
        xdata = xdf.copy()
        xdata['band_wth'] = self.band_func(xdata, n = self.boll_len).shift(1)
        xdata['band_mid'] = self.ma_func(xdata, n=self.boll_len).shift(1)
        xdata['band_up'] = xdata['band_mid'] + xdata['band_wth'] * self.band_ratio
        xdata['band_dn'] = xdata['band_mid'] - xdata['band_wth'] * self.band_ratio
        #xdata['filter_ind'] = self.filter_func(xdata, self.filter_len)
        if self.channel_mode == "HL":
            chan_mode = ['high', 'low']
        else:
            chan_mode = ['close', 'close']
        xdata['chan_h'] = dh.DONCH_H(xdata, self.channel, field = chan_mode[0]).shift(1)
        xdata['chan_l'] = dh.DONCH_L(xdata, self.channel, field = chan_mode[1]).shift(1)
        if self.chan_entry >= 0:
            xdata['chan_entry_l'] = xdata['chan_h'] - xdata['band_wth'] * self.chan_entry
            xdata['chan_entry_s'] = xdata['chan_l'] + xdata['band_wth'] * self.chan_entry
        else:
            xdata['chan_entry_l'] = xdata['band_up']
            xdata['chan_entry_s'] = xdata['band_dn']
        xdata['chan_stop_l'] = xdata['chan_h'] - xdata['band_wth'] * self.SL
        xdata['chan_stop_s'] = xdata['chan_l'] + xdata['band_wth'] * self.SL
        xdata['prev_high'] = xdata['high'].shift(1)
        xdata['prev_low'] = xdata['low'].shift(1)
        xdata['prev_close'] = xdata['close'].shift(1)
        self.df = xdata.dropna()
        #df.join(xdata, how = 'left').fillna(method='ffill').dropna()

    def run_vec_sim(self):
        if self.channel_mode == "HL":
            chan_mode = ['high', 'low']
        else:
            chan_mode = ['close', 'close']
        if self.data_freq == 'm':
            columns = ['open', 'close', 'contract', 'date', 'min_id']
        else:
            columns = ['open', 'close', 'contract']
        long_df = self.df[columns].copy()
        short_df = self.df[columns].copy()
        long_df['pos'] = np.nan
        long_df['traded_price'] = long_df['open']
        long_df.ix[ (self.df['open'] >= self.df['band_up']) \
                    & (self.df[chan_mode[0]].shift(1) >= self.df['chan_entry_l'].shift(1))\
                    & (self.df['open'] >= self.df['prev_low']) \
                    #& (self.df['filter_ind']>self.filter_thres[0]) \
                    , 'pos'] = 1.0
        long_df.ix[(self.df['open'] < self.df['chan_stop_l']) \
                       | (self.df['prev_close'] <= self.df['band_mid']), 'pos'] = 0.0
        long_df.ix[-1, 'pos'] = 0
        long_df['pos'] = long_df['pos'].fillna(method='ffill').fillna(0.0)
        #flag = (self.df['high'] >= self.df['long_entry']) & (self.df['open'] < self.df['long_entry']) \
        #       & (long_df['pos'] > long_df['pos'].shift(1).fillna(0.0))
        #long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'long_entry']
        #flag = (self.df['low'] <= self.df['long_exit']) & (self.df['open'] > self.df['long_exit']) \
        #       & (long_df['pos'] < long_df['pos'].shift(1).fillna(0.0))
        #long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'long_exit']
        long_df['cost'] = abs(long_df['pos'] - long_df['pos'].shift(1)) * (self.offset + long_df['close'] * self.tcost)
        long_df['cost'].fillna(0.0, inplace=True)
        long_trades = simdf_to_trades1(long_df, slippage=self.offset)

        short_df['pos'] = np.nan
        short_df['traded_price'] = short_df['open']
        short_df.ix[(self.df['open'] <= self.df['band_dn']) \
                    & (self.df[chan_mode[1]].shift(1) <= self.df['chan_entry_s'].shift(1)) \
                    & (self.df['open'] <= self.df['prev_high']) \
                    #& (self.df['filter_ind']<self.filter_thres[1]) \
                    , 'pos'] = -1.0
        short_df.ix[(self.df['open'] > self.df['chan_stop_s'])\
                    | (self.df['prev_close'] >= self.df['band_mid']), 'pos'] = 0.0
        short_df.ix[-1, 'pos'] = 0
        short_df['pos'] = short_df['pos'].fillna(method='ffill').fillna(0.0)
        #flag = (self.df['low'] <= self.df['short_entry']) & (self.df['open'] > self.df['short_entry']) \
        #       & (short_df['pos'] < short_df['pos'].shift(1).fillna(0.0))
        #short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'short_entry']
        #flag = (self.df['high'] >= self.df['short_exit']) & (self.df['open'] < self.df['short_exit']) \
        #       & (short_df['pos'] > short_df['pos'].shift(1).fillna(0.0))
        #short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'short_exit']
        short_df['cost'] = abs(short_df['pos'] - short_df['pos'].shift(1)) * (self.offset + short_df['close'] * self.tcost)
        short_df['cost'].fillna(0.0, inplace = True)
        short_trades = simdf_to_trades1(short_df, slippage=self.offset)
        closed_trades = long_trades + short_trades
        return ([long_df, short_df], closed_trades)

    def daily_initialize(self, sim_data, n):
        pass

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['band_up'][n]) or np.isnan(sim_data['chan_h'][n]) or np.isnan(sim_data['chan_l'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = round((self.SL * sim_data['band_wth'][n-1]) / float(self.tick_base)) * self.tick_base
        return gap

    def on_bar(self, sim_data, n):
        self.pos_args = {'reset_margin': 0}
        curr_pos = 0
        next_pos = (sim_data['high'][n-1] >= max(sim_data['band_up'][n-1], sim_data['chan_h'][n-1])) * (sim_data['filter'] > self.filter_thres[0]) * 1.0 - \
                   (sim_data['low'][n-1] <= min(sim_data['band_dn'][n-1], sim_data['chan_l'][n-1])) * (sim_data['filter'] < self.filter_thres[1]) * 1.0
        if len(self.positions)>0:
            curr_pos = self.positions[0].pos
            need_close = (self.close_daily or (self.scur_day == sim_data['date'][-1])) and (sim_data['min_id'][n-1] >= self.exit_min)
            if need_close or (curr_pos * next_pos < 0):
                for tradepos in self.positions:
                    self.close_tradepos(tradepos, sim_data['open'][n])
                self.positions = []
                curr_pos = 0
                if need_close:
                    return
            else:
                curr_pos = self.positions[0].pos
        if (curr_pos == 0) and next_pos != 0:
            self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], next_pos)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class']  = 'bktest.bktest_bband_chanstop.BBChanStop'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['scen_keys'] = ['boll_len', 'band_ratio', 'stoploss']
    sim_config['sim_name']   = 'bband_chanstop'
    sim_config['products']   = ['rb', 'hc', 'i', 'j', 'jm']
    sim_config['start_date'] = '20151201'
    sim_config['end_date']   = '20190328'
    sim_config['boll_len'] = [20, 30, 40, 50, 60]
    sim_config['band_ratio'] = [0.5, 1.0, 1.5, 2.0]
    sim_config['stoploss'] = [[0.5, 0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0], [1.0, 0.5], [1.5, 1.0], [1.5, 0.5], [2.0, 0.5], [2.0, 1.0]]
    sim_config['pos_class'] = 'trade_position.TradePos'
    sim_config['offset']    = 1

    config = {'capital': 10000,
              'trans_cost': 0.0,
              'close_daily': False,
              'filter_func': 'dh.CCI',
              'filter_len': 10,
              'ma_func': 'dh.MA',
              'band_func': 'dh.ATR',
              'filter_args': [0.0, 0.0],
              'unit': 1,                           
              'pos_args': {},
              'freq': '15m',
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
