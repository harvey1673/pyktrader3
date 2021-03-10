import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import misc
import data_handler as dh
import pandas as pd
import numpy as np
import datetime
from backtest import *


class ATRBreakSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(ATRBreakSim, self).__init__(config)

    def process_config(self, config):
        self.close_daily = config['close_daily']
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', [1])
        self.freq = config['freq']
        self.ref_mode = config.get('ref_mode', 'xopen')
        self.price_mode = config.get('price_mode', 'CL')
        self.atr_win = config['atr_win']
        self.band_ratio = config['band_ratio']
        self.exit_min = config.get('exit_min', 2059)
        self.pos_update = config['pos_update']
        self.pos_class = config['pos_class']
        self.pos_args = config['pos_args']
        self.reset_margin = config.get('reset_margin', 0.0)

    def process_data(self, mdf):
        if self.freq in self.data_store:
            xdf = self.data_store[self.freq].copy()
        else:
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq(mdf, self.freq, extra_cols=['contract'])
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                xdf = dh.day_split(mdf, sp, extra_cols=['contract'])
            else:
                print("uncovered data case")
            self.data_store[self.freq] = xdf.copy()
        xdf['ATR'] = dh.ATR(xdf, self.atr_win).shift(1)
        xdf['xma'] = dh.MA(xdf, self.atr_win).shift(1)
        xdf['xclose'] = xdf['close'].shift(1)
        xdf['xopen'] = xdf['open']
        self.df = mdf.join(xdf[['ATR', 'xclose', 'xopen', 'xma']], how='left').fillna(method='ffill').dropna()
        self.df['closeout'] = 0.0
        self.df['cost'] = 0.0
        self.df['pos'] = 0.0
        self.df['traded_price'] = self.df['open']
        self.trade_cost = self.offset

    def run_vec_sim(self):
        if self.price_mode == "HL":
            up_price = self.df['high']
            dn_price = self.df['low']
        elif self.price_mode == "TP":
            up_price = (self.df['high'] + self.df['low'] + self.df['close'])/3.0
            dn_price = up_price
        elif self.price_mode == "CL":
            up_price = self.df['close']
            dn_price = self.df['close']
        else:
            print("unsupported price mode")
        self.df['upper'] = self.df[self.ref_mode] + self.df['ATR'] * self.band_ratio
        self.df['upper'] = (self.df['upper']/self.tick_base).astype('int') * self.tick_base
        self.df['lower'] = self.df[self.ref_mode] - self.df['ATR'] * self.band_ratio
        self.df['lower'] = (self.df['lower'] / self.tick_base).astype('int') * self.tick_base
        self.df['pos']  = np.nan
        self.df['traded_price'] = self.df['open']
        self.df.ix[dh.CROSSOVER2(up_price, self.df['upper'], value = 0, direction = 1), 'pos'] = 1.0
        self.df.ix[dh.CROSSOVER2(dn_price, self.df['lower'], value = 0, direction = -1), 'pos'] = -1.0
        if self.close_daily:
            self.df.ix[self.df['min_id'] >= self.exit_min, 'pos'] = 0
        self.df['pos'][-2:] = 0
        self.df['pos'] = self.df['pos'].shift(1).fillna(method='ffill')
        self.df['pos'] = self.df['pos'].fillna(0)
        self.df['cost'] = abs(self.df['pos'] - self.df['pos'].shift(1)) * (self.offset + self.df['open'] * self.tcost)
        self.df['cost'] = self.df['cost'].fillna(0.0)
        self.closed_trades = simdf_to_trades1(self.df, slippage=self.offset)
        return ([self.df], self.closed_trades)

    def daily_initialize(self, sim_data, n):
        pass

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['ATR'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = (int((self.SL * sim_data['ATR'][n-1]) / float(self.tick_base)) + 1) * float(self.tick_base)
        return gap

    def on_bar(self, sim_data, n):
        self.pos_args = {'reset_margin': self.reset_margin * sim_data['ATR'][n-1]}
        if self.price_mode == "HL":
            up_price = sim_data['high'][n-1]
            dn_price = sim_data['low'][n-1]
        elif self.price_mode == "TP":
            up_price = (sim_data['high'][n-1] + sim_data['low'][n-1] + sim_data['close'][n-1])/3.0
            dn_price = up_price
        elif self.price_mode == "CL":
            up_price = sim_data['close'][n-1]
            dn_price = sim_data['close'][n-1]
        upper = sim_data[self.ref_mode][n-1] + round(self.band_ratio * sim_data['ATR'][n-1]/float(self.tick_base)) * self.tick_base
        lower = sim_data[self.ref_mode][n-1] - round(self.band_ratio * sim_data['ATR'][n-1]/float(self.tick_base)) * self.tick_base
        if len(self.positions)>0:
            curr_pos = self.positions[0].pos
        else:
            curr_pos = 0
        if (curr_pos != 0) and (self.close_daily or (self.scur_day == sim_data['date'][-1])) and \
                (sim_data['min_id'][n-1] >= sim_data['last_min'][n-1] - 1):
            for tradepos in self.positions:
                self.close_tradepos(tradepos, sim_data['open'][n])
            self.positions = []
            return
        if (curr_pos >= 0) and (dn_price < lower):
            if curr_pos > 0:
                for tradepos in self.positions:
                    self.close_tradepos(tradepos, sim_data['open'][n])
                self.positions = []
            self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], -1.0)
        elif (curr_pos <= 0) and (up_price > upper):
            if curr_pos < 0:
                for tradepos in self.positions:
                    self.close_tradepos(tradepos, sim_data['open'][n])
                self.positions = []
            self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], 1.0)

def gen_config_file(filename):
    sim_config = {}
    sim_config = {}
    sim_config['sim_class'] = 'bktest.bktest_atrbreak.ATRBreakSim'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['sim_freq'] = 'm'
    sim_config['scen_keys'] = ['freq', 'ref_mode','band_ratio', 'atr_win', 'close_daily']
    sim_config['sim_name'] = 'ATRBrk_190925'
    sim_config['products'] = ['rb', 'hc', 'i', 'j', 'jm']
    sim_config['start_date'] = '20151201'
    sim_config['end_date'] = '20190925'
    sim_config['freq'] = ['s1', 's2', 's3', 's4', '60m', '30m']
    sim_config['ref_mode'] = ['xopen', 'xclose']
    sim_config['band_ratio'] = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    sim_config['atr_win'] = [20, 40, 60]
    sim_config['close_daily'] = [False, True]
    sim_config['pos_class'] = 'trade_position.TradePos'
    sim_config['offset'] = 1

    config = {'capital': 10000,
              'trans_cost': 0.0,
              'unit': 1,
              'price_mode': 'CL',
              }
    sim_config['config'] = config
    with open(filename, 'w') as outfile:
        json.dump(sim_config, outfile)
    return sim_config

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("need to input a file name for config file")
    else:
        gen_config_file(args[0])
    pass
