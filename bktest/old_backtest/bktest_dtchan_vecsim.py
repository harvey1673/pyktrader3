import sys
import json
import pandas as pd
import numpy as np
import datetime
import pycmqlib3.analytics.data_handler as dh
from pycmqlib3.utility.misc import sign, day_split_dict
from pycmqlib3.core.trade_position import TradePos, TargetTrailTradePos
from . backtest import StratSim, simdf_to_trades1, simdf_to_trades2

class DTChanSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(DTChanSim, self).__init__(config)

    def process_config(self, config):
        self.assets = config['assets']
        self.close_daily = config['close_daily']
        self.freq = config['freq']
        self.offset = config['offset']
        self.tick_base = config['tick_base']
        self.k = config['multi']
        self.win = config['rng_win']
        self.f = config.get('trend_factor', 0.5)
        self.ma_width = config.get('ma_width', 1.0)
        self.price_mode = config.get('mode','TP')
        self.pos_update = config['pos_update']
        self.pos_class = config['pos_class']
        self.pos_args  = config['pos_args']
        self.reset_margin = config.get('reset_margin', 0.0)
        self.chan_func = config['chan_func']
        self.trading_mode = config['trading_mode']
        self.chan_high = eval(self.chan_func['high']['func'])
        self.chan_low  = eval(self.chan_func['low']['func'])
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', [1])
        self.SL = config['stoploss']
        self.min_rng = config['min_range']
        self.chan = config['chan']
        self.machan = config['machan']
        self.use_chan = (self.machan > 0)
        self.no_trade_set = config['no_trade_set']
        self.pos_freq = config.get('pos_freq', 1)
        self.combo_signal = config.get('combo_signal', True)
     
    def process_data(self, mdf):
        if self.freq in self.data_store:
            xdf = self.data_store[self.freq]
        else:
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                xdf = dh.conv_ohlc_freq(mdf, self.freq, extra_cols=['contract'])
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                xdf = dh.day_split(mdf, sp, extra_cols=['contract'])
            else:
                print("uncovered data case")
                self.data_store[self.freq] = mdf
            self.data_store[self.freq] = xdf
        xdf = xdf.copy()
        if self.win == 0:
            tr= pd.concat([xdf.high - xdf.low, abs(xdf.close - xdf.close.shift(1))],
                          join='outer', axis=1).max(axis=1)
        elif self.win < -1:
            abs_win = abs(self.win)
            tr = pd.concat([xdf.high - xdf.close, xdf.close - xdf.low], join='outer', axis=1).max(axis=1)
            for win in range(2, abs_win + 1):
                fact = np.sqrt(1.0/win)
                tr = pd.concat([(xdf.high.rolling(win).max() - xdf.close.rolling(win).min()) * fact, \
                            (xdf.close.rolling(win).max() - xdf.low.rolling(win).min()) * fact, tr], join='outer', axis=1).max(axis=1)
        else:
            tr= pd.concat([xdf.high.rolling(self.win).max() - xdf.close.rolling(self.win).min(),
                            xdf.close.rolling(self.win).max() - xdf.low.rolling(self.win).min()],
                            join='outer', axis=1).max(axis=1)/np.sqrt(self.win)
        xdf['tr'] = tr
        xdf['last_min'] = 2059
        xdf['chan_h'] = self.chan_high(xdf, self.chan, **self.chan_func['high']['args'])
        xdf['chan_l'] = self.chan_low(xdf, self.chan, **self.chan_func['low']['args'])
        xdf['ATR'] = dh.ATR(xdf, 20)
        xdf['ma'] = xdf.close.rolling(max(self.machan, 1)).mean()
        xdf['rng'] = pd.DataFrame([self.min_rng * xdf['ATR'], self.k * xdf['tr']]).max()
        xdf['upper'] = xdf['open'] + xdf['rng'].shift(1) * (1 + \
                        (((xdf['open'] - xdf['ma'].shift(1)) * sign(self.f) < - self.ma_width * xdf['ATR'].shift(1)) & self.use_chan) *self.f)
        xdf['lower'] = xdf['open'] - xdf['rng'].shift(1) * (1 + \
                        (((xdf['open'] - xdf['ma'].shift(1)) * sign(self.f) > self.ma_width * xdf['ATR'].shift(1)) & self.use_chan) *self.f)
        xdata = pd.concat([xdf['upper'], xdf['lower'], xdf['chan_h'].shift(1), xdf['chan_l'].shift(1),
                           xdf['open'], xdf['last_min'], xdf['ATR'].shift(1)], axis=1,
                          keys=['upper','lower', 'chan_h', 'chan_l', 'xopen', 'last_min', 'ATR'])
        self.df = mdf.join(xdata, how='left').fillna(method='ffill').dropna()
        self.df['upper'] = (self.df['upper']/self.tick_base + 1).astype('int') * self.tick_base
        self.df['lower'] = (self.df['lower']/self.tick_base).astype('int') * self.tick_base
        self.df.reset_index(inplace = True)
        self.df['closeout'] = 0.0
        self.df['cost'] = 0.0
        self.df['pos'] = 0.0
        self.df['traded_price'] = self.df['open']
        self.trade_cost = self.offset
    
    def run_vec_sim(self):
        self.df.set_index("datetime", inplace = True)
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
        up_level1 = self.df[['upper', 'chan_h']].max(axis=1)
        up_level2 = self.df[['upper', 'chan_h']].min(axis=1)
        dn_level1 = self.df[['lower', 'chan_l']].min(axis=1)
        dn_level2 = self.df[['lower', 'chan_l']].max(axis=1)
        long_df = self.df.copy()
        short_df = self.df.copy()
        long_df['pos'] = np.nan
        short_df['pos'] = np.nan
        if self.combo_signal:
            long_df.ix[(long_df['high'] >= long_df['chan_h']) & (up_price >= long_df['upper']) & (long_df['min_id'] < long_df['last_min']), 'pos'] = 1
            long_df.ix[(long_df['low'] <= long_df['chan_l']) | (dn_price <= long_df['lower']), 'pos'] = 0
            short_df.ix[(short_df['low'] <= short_df['chan_l']) & (dn_price <= short_df['lower']) & (short_df['min_id'] < short_df['last_min']), 'pos'] = -1
            short_df.ix[(short_df['high'] >= short_df['chan_h']) | (up_price >= short_df['upper']), 'pos'] = 0
        else:
            long_df.ix[(up_price >= long_df['upper']) & (long_df['min_id'] < long_df['last_min']), 'pos'] = 1
            long_df.ix[dn_price <= long_df['lower'], 'pos'] = 0
            short_df.ix[(dn_price <= short_df['lower']) & (short_df['min_id'] < short_df['last_min']), 'pos'] = -1
            short_df.ix[up_price >= short_df['upper'], 'pos'] = 0
        if self.trading_mode == 'match':
            long_df['pos'] = long_df['pos'].fillna(method='ffill')
            short_df['pos'] = short_df['pos'].fillna(method='ffill')
        else:
            long_df['pos'] = long_df['pos'].shift(1).fillna(method='ffill')
            short_df['pos'] = short_df['pos'].shift(1).fillna(method='ffill')
        if self.close_daily:
            long_df.ix[ long_df['min_id'] >= long_df['last_min'] - 1, 'pos'] = 0
            short_df.ix[short_df['min_id'] >= short_df['last_min'] - 1, 'pos'] = 0
        long_df.ix[-2:, 'pos'] = 0
        short_df.ix[-2:, 'pos'] = 0
        long_df['pos'] = long_df['pos'].fillna(0.0)
        short_df['pos'] = short_df['pos'].fillna(0.0)
        long_df['traded_price'] = long_df['open']
        short_df['traded_price'] = short_df['open']
        if self.trading_mode == 'match':
            if self.combo_signal:
                flag = (long_df['pos'] > long_df['pos'].shift(1)) & (long_df['high'] > up_level1) & (long_df['open'] < up_level1)
                long_df.ix[flag, 'traded_price'] = up_level1
                flag = (long_df['pos'] < long_df['pos'].shift(1)) & (long_df['low'] < dn_level2) & (long_df['open'] > dn_level2)
                long_df.ix[flag, 'traded_price'] = dn_level2
                flag = (short_df['pos'] < short_df['pos'].shift(1)) & (short_df['low'] < dn_level1) & (short_df['open'] > dn_level1)
                short_df.ix[flag, 'traded_price'] = dn_level1
                flag = (short_df['pos'] > short_df['pos'].shift(1)) & (short_df['high'] > up_level2) & (short_df['open'] < up_level2)
                short_df.ix[flag, 'traded_price'] = up_level2
            else:
                flag = (long_df['pos'] > long_df['pos'].shift(1)) & (long_df['high'] > long_df['upper']) & (long_df['open'] < long_df['upper'])
                long_df.ix[flag, 'traded_price'] = long_df['upper']
                flag = (long_df['pos'] < long_df['pos'].shift(1)) & (long_df['low'] < long_df['lower']) & (long_df['open'] > long_df['lower'])
                long_df.ix[flag, 'traded_price'] = long_df['lower']
                flag = (short_df['pos'] < short_df['pos'].shift(1)) & (short_df['low']<short_df['lower']) & (short_df['open']>short_df['lower'])
                short_df.ix[flag, 'traded_price'] = long_df['lower']
                flag = (short_df['pos'] > short_df['pos'].shift(1)) & (short_df['high'] > short_df['upper']) & (short_df['open']<short_df['upper'])
                short_df.ix[flag, 'traded_price'] = short_df['upper']
        long_df['cost'] = abs(long_df['pos'] - long_df['pos'].shift(1)) * self.trade_cost
        short_df['cost'] = abs(short_df['pos'] - short_df['pos'].shift(1)) * self.trade_cost
        long_df['cost'] = long_df['cost'].fillna(0.0)
        short_df['cost'] = short_df['cost'].fillna(0.0)
        long_trades = simdf_to_trades1(long_df, slippage = self.offset)
        short_trades = simdf_to_trades1(short_df, slippage=self.offset)
        closed_trades = long_trades + short_trades
        return ([long_df, short_df], closed_trades)

    def daily_initialize(self, sim_data, n):
        pass

    def check_data_invalid(self, sim_data, n):
        return np.isnan(sim_data['ATR'][n])
        # or (sim_data['date'][n] != sim_data['date'][n + 1])

    def get_tradepos_exit(self, tradepos, sim_data, n):
        gap = round((self.SL * sim_data['ATR'][n-1]) / float(self.tick_base)) * self.tick_base
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
        if self.combo_signal:
            if ((curr_pos >= 0) and ((dn_price < sim_data['lower'][n-1]) or (sim_data['low'][n-1] <= sim_data['chan_l'][n-1]))):
                if curr_pos > 0:
                    for tradepos in self.positions:
                        self.close_tradepos(tradepos, sim_data['open'][n])
                    self.positions = []
                if (dn_price < sim_data['lower'][n-1]) and (sim_data['low'][n-1] <= sim_data['chan_l'][n-1]):
                    self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], -1.0)
            elif ((curr_pos <= 0) and ((up_price > sim_data['upper'][n-1]) or (sim_data['high'][n-1] >= sim_data['chan_h'][n-1]))):
                if curr_pos < 0:
                    for tradepos in self.positions:
                        self.close_tradepos(tradepos, sim_data['open'][n])
                    self.positions = []
                if (up_price > sim_data['upper'][n-1]) and (sim_data['high'][n-1] >= sim_data['chan_h'][n-1]):
                    self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], 1.0)
        else:
            if (curr_pos >= 0) and (dn_price < sim_data['lower'][n-1]):
                if curr_pos > 0:
                    for tradepos in self.positions:
                        self.close_tradepos(tradepos, sim_data['open'][n])
                    self.positions = []
                self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], -1.0)
            elif (curr_pos <= 0) and (up_price > sim_data['upper'][n-1]):
                if curr_pos < 0:
                    for tradepos in self.positions:
                        self.close_tradepos(tradepos, sim_data['open'][n])
                    self.positions = []
                self.open_tradepos([sim_data['contract'][n-1]], sim_data['open'][n], 1.0)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class']  = 'bktest_dtchan_vecsim.DTChanSim'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['scen_keys'] = ['param', 'chan']
    sim_config['sim_name']   = 'DTChan_VecSim'
    sim_config['products']   = ['m', 'RM', 'y', 'p', 'a', 'rb', 'SR', 'TA', 'MA', 'i', 'ru', 'j' ]
    sim_config['start_date'] = '20150102'
    sim_config['end_date']   = '20170428'
    sim_config['param']  =  [
            (0.5, 0, 0.5, 0.0), (0.6, 0, 0.5, 0.0), (0.7, 0, 0.5, 0.0), (0.8, 0, 0.5, 0.0), \
            (0.9, 0, 0.5, 0.0), (1.0, 0, 0.5, 0.0), (1.1, 0, 0.5, 0.0), \
            (0.5, 1, 0.5, 0.0), (0.6, 1, 0.5, 0.0), (0.7, 1, 0.5, 0.0), (0.8, 1, 0.5, 0.0), \
            (0.9, 1, 0.5, 0.0), (1.0, 1, 0.5, 0.0), (1.1, 1, 0.5, 0.0), \
            (0.2, 2, 0.5, 0.0), (0.25,2, 0.5, 0.0), (0.3, 2, 0.5, 0.0), (0.35, 2, 0.5, 0.0),\
            (0.4, 2, 0.5, 0.0), (0.45, 2, 0.5, 0.0),(0.5, 2, 0.5, 0.0), \
            (0.2, 4, 0.5, 0.0), (0.25, 4, 0.5, 0.0),(0.3, 4, 0.5, 0.0), (0.35, 4, 0.5, 0.0),\
            (0.4, 4, 0.5, 0.0), (0.45, 4, 0.5, 0.0),(0.5, 4, 0.5, 0.0),\
            ]
    sim_config['chan'] = [3, 5, 10, 15, 20]
    sim_config['pos_class'] = 'trade_position.TradePos'
    sim_config['offset']    = 1
    chan_func = {'high': {'func': 'dh.DONCH_H', 'args':{}},
                 'low':  {'func': 'dh.DONCH_L', 'args':{}},
                 }
    config = {'capital': 10000,
              'trans_cost': 0.0,
              'close_daily': False,
              'unit': 1,              
              'min_range': 0.35,
              'chan_func': chan_func,
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
