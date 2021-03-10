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

class ChanBreakSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(ChanBreakSim, self).__init__(config)

    def process_config(self, config):
        self.assets = config['assets']
        self.close_daily = config['close_daily']
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.freq = config['freq']
        self.data_freq = config['data_freq']
        self.channel = config['channel']
        self.SL = config['stoploss']
        self.SL_win = config['stoploss_win']
        self.entry_buf = config.get('entry_buf', 0.0)
        self.reset_margin = config.get('reset_margin', 0)
        self.chan_func = config['chan_func']
        self.chan_high = eval(self.chan_func['high']['func'])
        self.chan_low = eval(self.chan_func['low']['func'])
        self.pos_class = config['pos_class']
        self.trading_mode = config.get('trading_mode', 'close')
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', [1])

    def process_data(self, df):
        if self.freq in self.data_store:
            xdf = self.data_store[self.freq].copy()
        else:
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq(df, self.freq, extra_cols=['contract'])
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                xdf = dh.day_split(df, sp, extra_cols=['contract'])
            else:
                print("uncovered data case")
            self.data_store[self.freq] = xdf.copy()
        for i in range(len(self.channel)):
            xdf['chan_h' + str(i)] = self.chan_high(xdf, self.channel[i], **self.chan_func['high']['args']).shift(1)
            xdf['chan_l' + str(i)] = self.chan_low(xdf, self.channel[i], **self.chan_func['low']['args']).shift(1)
        xdf['ATR'] = dh.ATR(xdf, n=self.SL_win).shift(1)
        xdf['chan_h0'] = xdf['chan_h0'] + self.entry_buf * xdf['ATR']
        xdf['chan_l0'] = xdf['chan_l0'] - self.entry_buf * xdf['ATR']
        xdf['tr_stop_h'] = xdf['chan_h0'] - self.SL * xdf['ATR']
        xdf['tr_stop_l'] = xdf['chan_l0'] + self.SL * xdf['ATR']
        xdf['last_min'] = 2059
        if (self.data_freq == 'm') and (self.freq != 'm1'):
            xdf = df.join(xdf[['tr_stop_h', 'tr_stop_l', 'chan_h0', 'chan_l0', \
                               'chan_h1', 'chan_l1', 'ATR', 'last_min']], how='left').fillna(method='ffill')
        self.df = xdf.dropna()
        for field in ['chan_h0', 'chan_l0', 'tr_stop_h', 'tr_stop_l']:
            self.df[field] = (self.df[field]/self.tick_base).astype('int') * self.tick_base
        self.df['long_exit'] = pd.concat([self.df['chan_l1'], self.df['tr_stop_h']], join='outer', axis=1).max(axis=1)
        self.df['short_exit'] = pd.concat([self.df['chan_h1'], self.df['tr_stop_l']], join='outer', axis=1).min(axis=1)
        self.df['datetime'] = self.df.index
        self.df['closeout'] = 0.0
        self.df['cost'] = 0.0
        self.df['pos'] = 0.0
        self.df['traded_price'] = self.df['open']
        self.trade_cost = self.offset

    def run_vec_sim(self):
        self.df['long_entry_pr'] = pd.concat([self.df['chan_h0'], self.df['open']], sort = False, join='outer', axis=1).max(
            axis=1)
        self.df['long_exit_pr'] = pd.concat([self.df['long_exit'], self.df['open']], sort = False, join='outer', axis=1).min(
            axis=1)
        self.df['short_entry_pr'] = pd.concat([self.df['chan_l0'], self.df['open']], sort = False, join='outer', axis=1).min(
            axis=1)
        self.df['short_exit_pr'] = pd.concat([self.df['short_exit'], self.df['open']], sort = False, join='outer', axis=1).max(
            axis=1)
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
        long_df.ix[(self.df['high'] > self.df['chan_h0']), 'pos'] = 1.0
        long_df.ix[(self.df['low'] <= self.df['long_exit']), 'pos'] = 0.0
        long_df['pos'] = long_df['pos'].fillna(method='ffill').fillna(0.0)
        if self.trading_mode == 'match':
            flag = (self.df['high'] > self.df['chan_h0']) & (self.df['open'] < self.df['chan_h0']) \
                   & (long_df['pos'] > long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'long_entry_pr']
            flag = (self.df['high'] > self.df['chan_h0']) & (self.df['open'] >= self.df['chan_h0']) \
                   & (long_df['pos'] > long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
            flag = (self.df['low'] <= self.df['long_exit']) & (self.df['open'] > self.df['long_exit']) \
                   & (long_df['pos'] < long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'long_exit_pr']
            flag = (self.df['low'] <= self.df['long_exit']) & (self.df['open'] <= self.df['long_exit']) \
                   & (long_df['pos'] < long_df['pos'].shift(1).fillna(0.0))
            long_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
        long_df.ix[-1, 'pos'] = 0
        long_df['cost'] = abs(long_df['pos'] - long_df['pos'].shift(1)) * self.trade_cost
        long_df['cost'].fillna(0.0, inplace = True)
        long_trades = simdf_to_trades1(long_df, slippage=self.offset)

        short_df['pos'] = pd.Series(np.nan, index = self.df.index)
        short_df.ix[(self.df['low'] < self.df['chan_l0']), 'pos'] = -1.0
        short_df.ix[(self.df['high'] >= self.df['short_exit']), 'pos'] = 0.0
        short_df['pos'] = short_df['pos'].fillna(method='ffill').fillna(0.0)
        if self.trading_mode == 'match':
            flag = (self.df['low'] < self.df['chan_l0']) & (self.df['open'] > self.df['chan_l0']) \
                   & (short_df['pos'] < short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'short_entry_pr']
            flag = (self.df['low'] < self.df['chan_l0']) & (self.df['open'] <= self.df['chan_l0']) \
                   & (short_df['pos'] < short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
            flag = (self.df['high'] >= self.df['short_exit']) & (self.df['open'] < self.df['short_exit']) \
                   & (short_df['pos'] > short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'short_exit_pr']
            flag = (self.df['high'] >= self.df['short_exit']) & (self.df['open'] >= self.df['short_exit']) \
                   & (short_df['pos'] > short_df['pos'].shift(1).fillna(0.0))
            short_df.ix[flag, 'traded_price'] = self.df.ix[flag, 'open']
        short_df.ix[-1, 'pos'] = 0
        short_df['cost'] = abs(short_df['pos'] - short_df['pos'].shift(1)) * self.trade_cost
        short_df['cost'].fillna(0.0, inplace = True)
        short_trades = simdf_to_trades1(short_df, slippage=self.offset)
        closed_trades = long_trades + short_trades
        return ([long_df, short_df], closed_trades)

    def daily_initialize(self, sim_data, n):
        pass

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['ATR'][n]) or np.isnan(sim_data['chan_h0'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = round((self.SL * sim_data['ATR'][n-1]) / float(self.tick_base)) * self.tick_base
        return gap

    def on_bar(self, sim_data, n):
        self.pos_args = {'reset_margin': self.reset_margin * sim_data['ATR'][n-1]}
        curr_pos = 0
        next_pos = (sim_data['high'][n-1] > sim_data['chan_h0'][n-1]) * 1 - (sim_data['low'][n-1] < sim_data['chan_l0'][n-1]) * 1
        if len(self.positions)>0:
            curr_pos = self.positions[0].pos
            need_close = (self.close_daily or (self.scur_day == sim_data['date'][-1])) \
                         and (sim_data['min_id'][n-1] >= sim_data['last_min'][n-1] - 1)
            need_close = need_close or ((sim_data['low'][n-1] < sim_data['chan_l1'][n-1]) and (curr_pos > 0)) or \
                         ((sim_data['high'][n-1] > sim_data['chan_h1'][n-1]) and (curr_pos < 0))
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
    sim_config['sim_class']  = 'bktest.bktest_chanbreak.ChanBreakSim'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['sim_freq'] = 'm'
    sim_config['scen_keys'] = ['stoploss_win', 'stoploss', 'channel']
    sim_config['sim_name']   = 'chanbreak_181028'
    sim_config['products']   = ['rb', 'hc', 'i', 'j', 'jm']
    sim_config['start_date'] = '20150901'
    sim_config['end_date']   = '20181028'
    sim_config['stoploss'] = [0.5, 1.0, 2.0, 3.0]
    sim_config['stoploss_win'] = [10, 20]
    sim_config['channel'] = [[10, 3], [10, 5], [15, 3], [15, 5], [15, 10], [20, 5], [20,10], [25, 5], [25, 10], [25, 15]]
    sim_config['pos_class'] = 'trade_position.TargetTrailTradePos'
    chan_func = {'high': {'func': 'dh.DONCH_H', 'args': {}},
                 'low': {'func': 'dh.DONCH_L', 'args': {}},
                 }
    sim_config['offset']    = 1
    config = {'capital': 10000,
              'trans_cost': 0.0,
              'unit': 1,
              'freq': 1,
              'stoploss': 0.0,
              'close_daily': False,
              'pos_update': True,
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