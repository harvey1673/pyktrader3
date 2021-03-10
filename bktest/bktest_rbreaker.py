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

class RBreakerSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(RBreakerSim, self).__init__(config)

    def process_config(self, config):
        self.assets = config['assets']
        self.freq = config['freq']
        self.data_freq = config['data_freq']
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.close_daily = config['close_daily']
        self.a = config['param'][0]
        self.b = config['param'][1]
        self.c = config['param'][2]
        self.pos_update = config['pos_update']
        self.pos_class = config['pos_class']
        self.pos_args  = config['pos_args']
        self.SL = config.get('stoploss', 1.0)
        self.SL_win = config.get('stoploss_win', 20)
        self.lookback_win = config.get('lookback_win', 30)
        self.bar_win = config.get('bar_win', 1)
        self.trade_limit = config.get('trade_limit', 3)
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.breakout_position = config.get('breakout_position', 1.0)
        self.reversal_position = config.get('reversal_position', 1.0)
        self.min_range = config.get('min_range', 0.7)
        self.weights = config.get('weights', [1])
        self.start_min = config.get('start_min', 301)
        self.exit_min = config.get('exit_min', 2058)
        self.session_start = 300
        self.session_end = 2059
        self.session_idx = 0
        self.rev_flag = True

    def process_data(self, mdf):
        if self.freq in self.data_store:
            xdf = self.data_store[self.freq]
        else:
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq1(mdf, self.freq)
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                xdf = dh.day_split(mdf, sp)
            else:
                print("uncovered data case")
                self.data_store[self.freq] = mdf
            self.data_store[self.freq] = xdf
        xdf = xdf.copy()
        xdf['range'] = xdf['high'].rolling(self.bar_win).max() - xdf['low'].rolling(self.bar_win).min()
        xdf['ssetup'] = xdf['high'].rolling(self.bar_win).max() + self.a * (xdf['close'] - xdf['low'].rolling(self.bar_win).min())
        xdf['bsetup'] = xdf['low'].rolling(self.bar_win).min()  - self.a * (xdf['high'].rolling(self.bar_win).max() - xdf['close'])
        xdf['senter'] = (1 + self.b) * (xdf['high'].rolling(self.bar_win).max() + xdf['close'])/2.0 \
                        - self.b * xdf['low'].rolling(self.bar_win).min()
        xdf['benter'] = (1 + self.b) * (xdf['low'].rolling(self.bar_win).min() + xdf['close'])/2.0 \
                        - self.b * xdf['high'].rolling(self.bar_win).max()
        xdf['bbreak'] = xdf.ssetup + self.c * (xdf.ssetup - xdf.bsetup)
        xdf['sbreak'] = xdf.bsetup - self.c * (xdf.ssetup - xdf.bsetup)
        xdf['ATR'] = dh.ATR(xdf, n=self.SL_win)
        tdf = xdf[['range', 'ssetup', 'bsetup', 'senter', 'benter', 'bbreak', 'sbreak', 'ATR']].shift(1)
        tdf['last_min'] = xdf['min_id']
        if (self.data_freq == 'm') and (self.freq != 'm1'):
            xdf = mdf.join(tdf, how='left').fillna(method='ffill')
        self.df = xdf.dropna()
        self.df['datetime'] = self.df.index
        self.df.loc[:, 'closeout'] = 0.0
        self.df.loc[:, 'cost'] = 0.0
        self.df.loc[:, 'pos'] = 0.0
        self.df.loc[:, 'traded_price'] = self.df.loc[:, 'open']
        self.trade_cost = self.offset
        self.num_trades = 0
        self.day_open = 0.0
        self.day_high = -1e+8
        self.day_low = 1e+8
        #self.bbreak = self.day_high
        #self.sbreak = self.day_low
        self.rev_flag = True

    def daily_initialize(self, sim_data, n):
        self.session_initialize(sim_data, n)

    def session_initialize(self, sim_data, n):
        self.num_trades = 0
        self.session_idx = n
        self.day_open = sim_data['open'][n]
        self.bbreak = sim_data['bbreak'][n]
        self.sbreak = sim_data['sbreak'][n]
        self.session_start = sim_data['min_id'][n]
        self.session_end = sim_data['last_min'][n]
        if (sim_data['range'][n] < sim_data['ATR'][n] * self.min_range):
            self.rev_flag = False
        else:
            self.rev_flag = True

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['ATR'][n]) or np.isnan(sim_data['sbreak'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = round((self.SL * sim_data['ATR'][n-1]) / float(self.tick_base)) * self.tick_base
        return gap

    def on_bar(self, sim_data, n):
        if (sim_data['date'][n]==sim_data['date'][n-1]) and (sim_data['last_min'][n]!=sim_data['last_min'][n-1]):
            self.session_initialize(sim_data, n)
        self.pos_args = {'reset_margin': 0}
        #print self.session_start, self.session_end, self.session_idx, self.scur_day, self.bbreak, self.sbreak
        if (self.close_daily or (self.scur_day == sim_data['date'][-1])) \
                         and (sim_data['min_id'][n] >= self.session_end - 1):
            for tradepos in self.positions:
                self.close_tradepos(tradepos, sim_data['open'][n])
            self.positions = []
        else:
            if (sim_data['min_id'][n] >= self.session_start + 1) and (self.num_trades <= self.trade_limit):
                t_high = max(sim_data['high'][max(self.session_idx, n - self.lookback_win):n])
                t_low = min(sim_data['low'][max(self.session_idx, n - self.lookback_win):n])
                if len(self.positions) == 0:
                    next_pos = ((sim_data['open'][n] >= self.bbreak)) * 1.0 - (sim_data['open'][n] <= self.sbreak) * 1.0
                    if next_pos != 0:
                        self.open_tradepos([sim_data['contract'][n]], sim_data['open'][n], next_pos * self.breakout_position)
                        self.bbreak = max(self.bbreak, sim_data['high'][n])
                        self.sbreak = min(self.sbreak, sim_data['low'][n])
                        self.num_trades += 1
                    elif self.rev_flag:
                        next_pos = ((t_low <= sim_data['bsetup'][n]) and (sim_data['open'][n] >= sim_data['benter'][n])) * 1.0 - \
                                   ((t_high >= sim_data['ssetup'][n]) and (sim_data['open'][n] <= sim_data['senter'][n])) * 1.0
                        if next_pos != 0:
                            self.open_tradepos([sim_data['contract'][n]], sim_data['open'][n], next_pos * self.reversal_position)
                            self.num_trades += 1
                else:
                    curr_pos = self.positions[0].pos
                    if (self.num_trades <= self.trade_limit) and (self.rev_flag):
                        next_pos = 1.0 * ((t_low <= sim_data['bsetup'][n]) and (sim_data['open'][n] >= sim_data['benter'][n]) and (curr_pos < 0)) - \
                                   ((t_high >= sim_data['ssetup'][n]) and (sim_data['open'][n] <= sim_data['senter'][n]) and (curr_pos > 0))
                        if next_pos != 0:
                            for tradepos in self.positions:
                                self.close_tradepos(tradepos, sim_data['open'][n])
                            self.open_tradepos([sim_data['contract'][n]], sim_data['open'][n], self.reversal_position * next_pos)
                            self.num_trades += 1

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class']  = 'bktest.bktest_rbreaker.RBreakerSim'
    sim_config['sim_func'] = 'run_loop_sim'
    sim_config['scen_keys'] = ['params', 'min_range', 'stoploss']
    sim_config['sim_name']   = 'rbreaker_test_191029'
    sim_config['products']   = ['y', 'p', 'l', 'pp', 'rb', 'SR', 'TA', 'MA', 'i', 'j', 'ru']
    sim_config['start_date'] = '20161001'
    sim_config['end_date']   = '20191029'
    sim_config['min_range']  =  [0.6, 0.8, 1.0, 1.2]
    sim_config['stoploss'] = [0.2, 0.4, 0.6, 0.8]
    sim_config['param'] = [(0.25, 0.07, 0.2)]
    sim_config['pos_class'] = 'trade_position.TargetTrailTradePos'
    sim_config['offset']    = 1

    config = {'capital': 10000,
              'trans_cost': 0.0,
              'close_daily': True,
              'unit': 1,
              'pos_update': True,
              'pos_args': {},
              'freq': 's1',
              'data_freq': 'm1',
              'stoploss_win': 20,
              'bar_win': 1,
              'trade_limit': 3,
              'breakout_position': 1,
              'reversal_position': 1,
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
