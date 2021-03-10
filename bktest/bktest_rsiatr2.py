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

class RSIATRSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(RSIATRSim, self).__init__(config)

    def process_config(self, config): 
        self.close_daily = config['close_daily']
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.freq = config['freq']
        self.atr_len = config['atr_period']
        self.atrma_len = config['atrma_period']
        self.rsi_len = config['rsi_len']
        self.rsi_trigger = config['rsi_trigger']
        self.pos_update = config['pos_update']
        self.pos_class = config['pos_class']
        self.pos_args  = config['pos_args']        
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', [1])
        self.SL = config['stoploss']
        self.no_trade_set = config['no_trade_set']
        self.pos_freq = config.get('pos_freq', 1)
        self.exit_min = 2058
        self.buy_trig = 0.0
        self.sell_trig = 0.0
     
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
        xdf['ATR'] = dh.ATR(xdf, n = self.atr_len)
        xdf['ATRMA'] = dh.MA(xdf, n=self.atrma_len, field = 'ATR')
        adf = dh.RSI_F(xdf, n = self.rsi_len)
        xdf['RSI'] = adf['RSI'+ str(self.rsi_len)]
        xdf['datetime'] = xdf.index
        xdf['closeout'] = 0.0
        xdf['cost'] = 0.0
        xdf['pos'] = 0.0
        xdf['traded_price'] = xdf['open']
        self.df = xdf.dropna()

    def daily_initialize(self, sim_data, n):
        pass

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['ATR'][n]) or np.isnan(sim_data['ATRMA'][n]) or np.isnan(sim_data['RSI'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = round((self.SL * sim_data['ATRMA'][n-1]) / float(self.tick_base)) * self.tick_base
        return gap

    def on_bar(self, sim_data, n):
        self.pos_args = {'reset_margin': 0}
        curr_pos = 0
        next_pos = (sim_data['RSI'][n-1] > 50.0 + self.rsi_trigger) * 1 - (sim_data['RSI'][n-1] < 50.0 - self.rsi_trigger) * 1
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
        target_pos = next_pos * (sim_data['ATR'][n-1] > sim_data['ATRMA'][n-1])
        if (curr_pos == 0) and target_pos != 0:
            self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], target_pos)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class']  = 'bktest.bktest_rsiatr2.RSIATRSim'
    sim_config['sim_func'] = 'run_loop_sim'
    sim_config['scen_keys'] = ['freq', 'stoploss']
    sim_config['sim_name']   = 'RSI_ATR'
    sim_config['products']   = ['m', 'RM', 'y', 'p', 'a', 'rb', 'SR', 'TA', 'MA', 'i', 'ru', 'j' ]
    sim_config['start_date'] = '20160102'
    sim_config['end_date']   = '20170607'
    sim_config['stoploss'] = [3.0, 6.0, 9.0]
    sim_config['freq'] = [3, 5, 15]
    sim_config['pos_class'] = 'strat.TargetTrailTradePos'
    sim_config['offset']    = 1

    config = {'capital': 10000,
              'trans_cost': 0.0,
              'close_daily': False,
              'unit': 1,                           
              'pos_args': {},
              'pos_update': True,
              'atr_period': 22,
              'atrma_period': 10, 
              'rsi_len': 5,
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
