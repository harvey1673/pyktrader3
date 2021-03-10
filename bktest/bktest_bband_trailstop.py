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

class BBTrailStop(StratSim):
    def __init__(self, config):
        super(BBTrailStop, self).__init__(config)

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
        self.filter_len = config.get('filter_len', 10)
        self.filter_thres = config.get('filter_args', [0, 0])
        self.channel = config.get('channel', 30)
        self.pos_class = config['pos_class']
        self.pos_args  = config['pos_args']        
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', [1])
        self.SL = config['stoploss']
        self.trading_mode = config.get('trading_mode', 'close')
     
    def process_data(self, df):
        if self.freq == 1:
            xdf = df
        else:
            freq_str = str(self.freq) + "min"
            xdf = dh.conv_ohlc_freq(df, freq_str, extra_cols = ['contract'])
        xdf['band_wth'] = self.band_func(xdf, n = self.boll_len).fillna(method = 'bfill')
        xdf['band_mid'] = self.ma_func(xdf, n=self.boll_len)
        xdf['band_up'] = xdf['band_mid'] + xdf['band_wth'] * self.band_ratio
        xdf['band_dn'] = xdf['band_mid'] - xdf['band_wth'] * self.band_ratio
        xdf['filter_ind'] = self.filter_func(xdf, self.filter_len)
        xdf['chan_h'] = dh.DONCH_H(xdf, self.channel)
        xdf['chan_l'] = dh.DONCH_L(xdf, self.channel)
        xdf['tr_stop_h'] = xdf['chan_h'] - (xdf['band_wth'] * self.SL / self.tick_base).astype('int') * self.tick_base
        xdf['tr_stop_l'] = xdf['chan_l'] + (xdf['band_wth'] * self.SL / self.tick_base).astype('int') * self.tick_base
        xdata = pd.concat([xdf['band_up'].shift(1), xdf['band_dn'].shift(1), \
                           xdf['tr_stop_h'].shift(1), xdf['tr_stop_l'].shift(1), xdf['filter_ind'].shift(1)], axis=1, \
                           keys=['band_up','band_dn', 'tr_stop_h', 'tr_stop_l', 'filter_ind'])
        self.df = df.join(xdata, how = 'left').fillna(method='ffill').dropna()
        self.df['cost'] = 0.0
        self.df['pos'] = 0.0
        self.df['closeout'] = 0.0
        self.df['traded_price'] = self.df['open']

    def daily_initialize(self, sim_data, n):
        pass

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['band_up'][n]) or np.isnan(sim_data['chan_h'][n]) or np.isnan(sim_data['chan_l'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = (int((self.SL * sim_data['band_wth'][n-1]) / float(self.tick_base)) + 1) * float(self.tick_base)
        return gap

    def run_vec_sim(self):
        self.df['long_entry_pr'] = pd.concat([self.df['band_up'], self.df['open']], join='outer', axis=1).max(axis=1)
        self.df['long_exit_pr'] = pd.concat([self.df['tr_stop_h'], self.df['open']], join='outer', axis=1).min(axis=1)
        self.df['short_entry_pr'] = pd.concat([self.df['band_dn'], self.df['open']], join='outer', axis=1).min(axis=1)
        self.df['short_exit_pr'] = pd.concat([self.df['tr_stop_l'], self.df['open']], join='outer', axis=1).max(axis=1)
        if self.data_freq == 'm':
            columns = ['open', 'close', 'contract', 'date', 'min_id']
        else:
            columns = ['open', 'close', 'contract']
        long_df = self.df[columns].copy()
        short_df = self.df[columns].copy()
        if self.trading_mode in ['match']:
            long_df['traded_price'] = long_df['open']
            short_df['traded_price'] = short_df['open']
        else:
            long_df['traded_price'] = long_df['close']
            short_df['traded_price'] = short_df['close']
        long_df['pos'] = pd.Series(np.nan, index = self.df.index)
        long_df.ix[(self.df['high'] > self.df['band_up']) & (self.df['filter_ind']>self.filter_thres[0]), 'pos'] = 1.0
        long_df.ix[(self.df['low'] <= self.df['tr_stop_h']), 'pos'] = 0.0
        long_df['pos'] = long_df['pos'].fillna(method='ffill').fillna(0.0)
        if self.trading_mode == 'match':
            flag = (self.df['high'] > self.df['band_up']) & (self.df['filter_ind']>self.filter_thres[0]) \
                   & (self.df['open'] < self.df['band_up']) & (long_df['pos'] > long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'long_entry_pr']
            flag = (self.df['high'] > self.df['band_up']) & (self.df['filter_ind']>self.filter_thres[0]) \
                   & (self.df['open'] >= self.df['band_up']) & (long_df['pos'] > long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
            flag = (self.df['low'] <= self.df['tr_stop_h']) & (self.df['open'] > self.df['tr_stop_h']) \
                   & (long_df['pos'] < long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'long_exit_pr']
            flag = (self.df['low'] <= self.df['tr_stop_h']) & (self.df['open'] <= self.df['tr_stop_h']) \
                   & (long_df['pos'] < long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
        long_df.ix[-1, 'pos'] = 0
        long_df['cost'] = abs(long_df['pos'] - long_df['pos'].shift(1)) * (self.offset + long_df['close'] * self.tcost)
        long_df['cost'].fillna(0.0, inplace = True)
        long_trades = simdf_to_trades1(long_df, slippage=self.offset)

        short_df['pos'] = pd.Series(np.nan, index = self.df.index)
        short_df.ix[(self.df['low'] < self.df['band_dn']) & (self.df['filter_ind']<self.filter_thres[1]), 'pos'] = -1.0
        short_df.ix[(self.df['high'] >= self.df['tr_stop_l']), 'pos'] = 0.0
        short_df['pos'] = short_df['pos'].fillna(method='ffill').fillna(0.0)
        if self.trading_mode == 'match':
            flag = (self.df['low'] < self.df['band_dn']) & (self.df['filter_ind']<self.filter_thres[1]) \
                   & (self.df['open'] > self.df['band_dn']) & (short_df['pos'] < short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'short_entry_pr']
            flag = (self.df['low'] < self.df['band_dn']) & (self.df['filter_ind']<self.filter_thres[1]) \
                   & (self.df['open'] <= self.df['band_dn']) & (short_df['pos'] < short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
            flag = (self.df['high'] >= self.df['tr_stop_l']) & (self.df['open'] < self.df['tr_stop_l']) \
                   & (short_df['pos'] > short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'short_exit_pr']
            flag = (self.df['high'] >= self.df['tr_stop_l']) & (self.df['open'] >= self.df['tr_stop_l']) \
                   & (short_df['pos'] > short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
        short_df.ix[-1, 'pos'] = 0
        short_df['cost'] = abs(short_df['pos'] - short_df['pos'].shift(1)) * (self.offset + short_df['close'] * self.tcost)
        short_df['cost'].fillna(0.0, inplace = True)
        short_trades = simdf_to_trades1(short_df, slippage=self.offset)
        closed_trades = long_trades + short_trades
        return ([long_df, short_df], closed_trades)

    def on_bar(self, sim_data, n):
        self.pos_args = {'reset_margin': 0}
        curr_pos = 0
        next_pos = (sim_data['high'][n] >= max(sim_data['band_up'][n], sim_data['chan_h'][n])) * (sim_data['filter'] > self.filter_thres[0]) * 1.0 - \
                   (sim_data['low'][n] <= min(sim_data['band_dn'][n], sim_data['chan_l'][n])) * (sim_data['filter'] < self.filter_thres[1]) * 1.0
        if len(self.positions)>0:
            curr_pos = self.positions[0].pos
            need_close = (self.close_daily or (self.scur_day == sim_data['date'][-1])) and (sim_data['min_id'][n] >= self.exit_min)
            if need_close or (curr_pos * next_pos < 0):
                for tradepos in self.positions:
                    self.close_tradepos(tradepos, sim_data['open'][n+1])
                self.positions = []
                curr_pos = 0
                if need_close:
                    return
            else:
                curr_pos = self.positions[0].pos
        if (curr_pos == 0) and next_pos != 0:
            self.open_tradepos([sim_data['contract'][n]], sim_data['open'][n+1], next_pos)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class']  = 'bktest.bktest_bband_trailstop.BBTrailStop'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['scen_keys'] = ['boll_len', 'band_ratio', 'filter_args', 'stoploss']
    sim_config['filter_len'] = [20, 40, 60, 80]
    sim_config['sim_name']   = 'band_trailstop'
    sim_config['products']   = ['m', 'RM', 'y', 'p', 'a', 'rb', 'SR', 'TA', 'MA', 'i', 'ru', 'j' ]
    sim_config['start_date'] = '20150701'
    sim_config['end_date']   = '20181028'
    sim_config['boll_len'] = [20, 30, 40, 50, 60]
    sim_config['band_ratio'] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    sim_config['stoploss'] = [0.5, 1.0, 1.5, 2.0]
    sim_config['pos_class'] = 'trade_position.TargetTrailTradePos'
    sim_config['offset']    = 1

    config = {'capital': 10000,
              'trans_cost': 0.0,
              'close_daily': False,
              'chan_ratio': 2.0,
              'filter_func': 'dh.CCI',
              'ma_func': 'dh.MA',
              'band_func': 'dh.ATR',
              'filter_args': [0.0, 0.0],
              'unit': 1,                           
              'pos_args': {},
              'freq': 15,
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
