#-*- coding:utf-8 -*-
from base import *
from misc import *
from strategy import *

class BulkTrade(Strategy):
    common_params =  dict({'trade_file': "bulktrade_test"}, **Strategy.common_params)
    asset_params = dict({'limit_price': 0.0, 'direction': 1, 'tick_num': 1, 'run_flag': 1, \
                         'max_vol': 10, 'time_period': 50, 'price_type': OrderType.LIMIT.value, \
                         'exec_args': {'max_vol': 10, 'time_period': 50, 'price_type': OrderType.LIMIT.value, \
                                  'tick_num': 1, 'order_type': '', 'order_offset': True, 'inst_order': None},},\
                        **Strategy.asset_params)
    def __init__(self, config, agent = None):
        Strategy.__init__(self, config, agent)
        numAssets = len(self.underliers)
        self.tick_base = [0.0] * numAssets

    def set_exec_args(self, idx, direction):
        for key in ['max_vol', 'time_period', 'price_type', 'tick_num']:
            self.exec_args[idx][key] = getattr(self, key)[idx]

    def register_bar_freq(self):
        for idx, under_obj in enumerate(self.underlying):
            inst = under_obj.name
            self.agent.inst2strat[inst][self.name] = ''

    def load_trades(self):
        filename = self.folder + self.trade_file + ".json"
        try:
            pos_dict = json.load(open(filename, "r"))
            print('input file is loaded for bulk trade')
        except:
            print('default input is loaded for bulk trade')
            pos_dict = {}
        for idx, under in enumerate(self.underliers):
            inst = '_'.join(under)
            if inst in pos_dict:
                self.alloc_w[idx] = abs(pos_dict[inst]['position'])
                self.direction[idx] = 1 if pos_dict[inst]['position'] > 0 else -1
                if 'price' in pos_dict[inst]:
                    self.limit_price[idx] = pos_dict[inst]['price']
                else:
                    self.limit_price[idx] = 0.0
            else:
                self.alloc_w[idx] = 0
                self.direction[idx] = 1
                self.limit_price[idx] = 0.0
        self.update_trade_unit()

    def bulk_trades(self):
        for idx, under in enumerate(self.underliers):
            if self.trade_unit[idx] != 0:
                self.set_exec_args(idx, self.direction[idx])
                if self.limit_price[idx] == 0:
                    self.limit_price[idx] = self.curr_prices[idx]
                self.open_tradepos(idx, self.direction[idx], self.limit_price[idx], int(self.trade_unit[idx]))

    def initialize(self):
        super(BulkTrade, self).initialize()
        for idx in self.positions:
            self.curr_pos[idx] = sum([tradepos.pos for tradepos in self.positions[idx]])
            if len(self.positions[idx]) > 1:
                if self.curr_pos[idx] == 0:
                    self.positions[idx] = []
                else:
                    tradepos = self.positions[idx][-1]
                    tradepos.pos = self.curr_pos[idx]
                    tradepos.target_pos = self.curr_pos[idx]
                    tradepos.direction = 1 if self.curr_pos[idx] > 0 else -1
                    self.positions[idx] = [tradepos]

