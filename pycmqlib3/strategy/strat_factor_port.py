# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import logging
from pycmqlib3.utility.base import BaseObject, fcustom
from pycmqlib3.utility.misc import day_shift, check_trading_range
from pycmqlib3.utility.dbaccess import load_factor_data
import pycmqlib3.analytics.data_handler as dh
from . strategy import Strategy

# table format for factor data
# create table fut_fact_data (product_code varchar(10), roll_label varchar(20), exch varchar(10),
# fact_name varchar(100), freq varchar(10), date date, serial_no int, serial_key varchar(30),
# fact_val double, PRIMARY KEY(product_code, roll_label, exch, fact_name, freq, date, serial_no));

class FactorPortTrader(Strategy):
    common_params = dict(Strategy.common_params, \
                         **{'freq': 's1',
                            'factor_repo': {'lr_sum_20': {'name': 'lr_sum_20','type': 'ts', 'param': [], 'weight': 1.0, \
                                            'rebal': 5, 'threshold': 0.0}}, \
                            'factor_data': {}, \
                            'factor_pos': {}, \
                            'pos_scaler': 1000000.0, 'vol_win': 20, \
                            'exec_bar_list': [305], \
                            'fact_db_table': 'fut_fact_data', \
                            'hist_fact_lookback': 250, })
    asset_params = dict({'roll_label': 'CAL_30b',}, **Strategy.asset_params)

    def __init__(self, config, agent=None):
        Strategy.__init__(self, config, agent)
        numAssets = len(self.underliers)
        self.prod_list = [''] * numAssets
        self.target_pos = [0] * numAssets
        self.vol_weight = [1.0] * numAssets
        self.bar_update_record = set()
        self.fact_src = 'db.prod_fact'
        self.fact_data = {}
        self.pos_summary = pd.DataFrame()
        self.tick_base = [0.0] * numAssets
        self.threshold = int(numAssets * 0.8)

    def save_local_variables(self, file_writer):
        if len(self.pos_summary) > 0:
            header = 'target_pos'
            for idx in self.pos_summary.index:
                row = [header, idx] + self.pos_summary.loc[idx].tolist()
                file_writer.writerow(row)            

    def load_local_variables(self, row):
        pass

    def set_agent(self, agent):
        super(FactorPortTrader, self).set_agent(agent)
        self.prod_list = [under.product for under in self.underlying]

    def load_fact_data(self):
        if 'db' in self.fact_src:
            end_date = self.agent.scur_day
            start_date = day_shift(end_date, '-%sb' % (str(self.hist_fact_lookback)))
            df = pd.DataFrame()
            fact_list = list(set([self.factor_repo[fact]['name'] for fact in self.factor_repo.keys()]))
            for roll_rule in set(self.roll_label):
                prod_list = [prod for prod, roll in zip(self.prod_list, self.roll_label) if roll == roll_rule]                
                adf = load_factor_data(prod_list, \
                             factor_list = fact_list,\
                             roll_label = roll_rule,\
                             start = start_date, \
                             end = end_date, \
                             freq = self.freq, db_table = self.fact_db_table)
                df = df.append(adf)
            for fact in self.factor_repo:
                xdf = pd.pivot_table(df[df['fact_name'] == self.factor_repo[fact]['name']], values = 'fact_val', \
                                     index = ['date', 'serial_key'], columns = ['product_code'],\
                                     aggfunc = 'last')
                xdf = xdf[self.prod_list]
                self.fact_data[fact] = xdf
        else:
            print("Update from the strategy is not implemented yet.")
        self.update_target_pos()

    def update_target_pos(self):
        net_pos = pd.Series(dtype = 'float')
        self.pos_summary = pd.DataFrame()
        for fact in self.factor_repo:
            rebal_freq = self.factor_repo[fact]['rebal']
            weight =  self.factor_repo[fact]['weight']
            self.factor_pos[fact] = pd.DataFrame(index = self.fact_data[fact].index, \
                                                 columns = self.fact_data[fact].columns)
            if self.factor_repo[fact]['type'] == 'pos':
                self.factor_pos[fact] = self.fact_data[fact].copy()
            elif self.factor_repo[fact]['type'] == 'ts':
                rebal_ts = pd.Series(range(len(self.fact_data[fact].index)), index = self.fact_data[fact].index)
                for rebal_idx in range(rebal_freq):
                    flag = rebal_ts % rebal_freq == rebal_idx
                    long_pos  = pd.Series(np.nan, index = self.fact_data[fact].index)
                    short_pos = pd.Series(np.nan, index = self.fact_data[fact].index)
                    for asset in self.prod_list:
                        pflag = (self.fact_data[fact][asset] >= 0.0)
                        nflag = (self.fact_data[fact][asset] <= 0.0)
                        long_pos[flag & pflag] = self.fact_data[fact][asset][flag & pflag]
                        long_pos[flag & (~pflag)] = 0.0
                        long_pos[flag] = long_pos[flag].fillna(method = 'ffill').fillna(0.0)
                        short_pos[flag & nflag] = self.fact_data[fact][asset][flag & nflag]
                        short_pos[flag & (~nflag)] = 0.0
                        short_pos[flag] = short_pos[flag].fillna(method='ffill').fillna(0.0)
                        self.factor_pos[fact].loc[flag, asset] = long_pos[flag] + short_pos[flag]
            elif self.factor_repo[fact]['type'] == 'xs':
                lower_rank = int(len(self.prod_list) * self.factor_repo[fact]['threshold']) + 1
                upper_rank = len(self.prod_list) - int(len(self.prod_list) * self.factor_repo[fact]['threshold'])
                rank_df = self.fact_data[fact].rank(axis = 1)
                rebal_ts = pd.Series(range(len(self.fact_data[fact].index)), index = self.fact_data[fact].index)
                for rebal_idx in range(rebal_freq):
                    flag = rebal_ts % rebal_freq == rebal_idx
                    long_pos  = pd.Series(np.nan, index = self.fact_data[fact].index)
                    short_pos = pd.Series(np.nan, index = self.fact_data[fact].index)
                    for asset in self.prod_list:
                        pflag = (rank_df[asset] > upper_rank)
                        nflag = (rank_df[asset] < lower_rank)
                        long_pos[flag & pflag] = 1.0
                        long_pos[flag & (~pflag)] = 0.0
                        long_pos[flag] = long_pos[flag].fillna(method = 'ffill').fillna(0.0)
                        short_pos[flag & nflag] = -1.0
                        short_pos[flag & (~nflag)] = 0.0
                        short_pos[flag] = short_pos[flag].fillna(method='ffill').fillna(0.0)
                        self.factor_pos[fact].loc[flag, asset] = long_pos[flag] + short_pos[flag]
                self.factor_pos[fact] = self.factor_pos[fact].fillna(0.0)
            fact_pos = pd.Series(self.factor_pos[fact].iloc[-rebal_freq:,:].sum()/rebal_freq * weight, name = fact)
            self.pos_summary = self.pos_summary.append(fact_pos)              
        self.pos_summary = self.pos_summary[self.prod_list].round(2)
        net_pos = self.pos_summary.sum()           
        for idx, (under, prodcode) in enumerate(zip(self.underliers, self.prod_list)):            
            self.target_pos[idx] = int(net_pos[prodcode] * self.vol_weight[idx] + (0.5 if net_pos[prodcode]>0 else -0.5))
        self.save_state()        

    def register_func_freq(self):
        for idx, under in enumerate(self.underliers):
            inst = under[0]
            if ('s' in self.freq) or ('m' in self.freq):
                self.agent.register_data_func(inst, self.freq, None)
        vol_name = 'ATR' + str(self.vol_win)
        self.scaling_field = vol_name
        for idx, under_obj in enumerate(self.underlying):
            atr_fobj = BaseObject(name = vol_name,
                            sfunc = fcustom(dh.ATR, n=self.vol_win),
                            rfunc=fcustom(dh.atr, n=self.vol_win))
            self.agent.register_data_func(under_obj.name, 'd', atr_fobj)

    def register_bar_freq(self):
        for idx, (under, prev_under) in enumerate(zip(self.underlying, self.prev_underlying)):
            self.agent.inst2strat[under.name][self.name] = 'm1'
            if prev_under:
                self.agent.inst2strat[prev_under.name][self.name] = 'm1'

    def initialize(self):
        for idx, underlier in enumerate(self.underliers):
            self.tick_base[idx] = max([self.agent.instruments[inst].tick_base for inst in underlier])
            xdata = self.agent.day_data[self.underlying[idx].name].data
            self.vol_weight[idx] = self.pos_scaler/xdata[self.scaling_field][-1]/self.conv_f[underlier[0]]
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
        self.bar_update_record = set()        
        self.load_fact_data()

    def on_bar(self, idx, freq_list):
        save_status = False
        if (self.agent.tick_id // 1000) in self.exec_bar_list:
            self.bar_update_record.add(idx)
        if len(self.bar_update_record) >= self.threshold:
            self.on_log("running strat = %s for tick_id = %d" % (self.name, self.agent.tick_id))
            # self.load_fact_data()
            save_status = self.trade_target_pos()
            self.bar_update_record = set()
        return save_status

    def on_tick(self, idx, ctick):
        return False

    def trade_target_pos(self):
        save_status = False
        for idx, underlier in enumerate(self.underliers):
            if not check_trading_range(self.agent.tick_id, self.underlying[idx].product, self.underlying[idx].exchange):
                continue
            curr_pos = 0
            curr_positions = self.positions[idx]
            new_pos = self.target_pos[idx]
            if len(curr_positions) > 0:
                if (curr_positions[0].insts != underlier):
                    msg = '%s to close position for inst = %s, pos=%s' % \
                          (self.name, curr_positions[0].insts, curr_positions[0].pos)
                    self.close_tradepos(idx, curr_positions[0], self.prev_underlying[idx].mid_price)
                    curr_pos = 0
                    self.status_notifier(underlier, msg)
                    save_status = True
                else:
                    curr_pos = curr_positions[0].target_pos
            if (curr_pos == 0) and (new_pos != 0):
                msg = '%s to open position for inst = %s, pos=%s' % \
                      (self.name, underlier, new_pos)
                dir = 1 if new_pos > 0 else -1
                self.open_tradepos(idx, dir, self.curr_prices[idx], volume = dir * new_pos)
                self.status_notifier(underlier, msg)
                save_status = True
            else:
                trade_volume = new_pos - curr_pos
                if trade_volume != 0:
                    msg = '%s to modify position for inst = %s, old_pos=%s, new_pos = %s' % \
                          (self.name, underlier, curr_pos, new_pos)
                    self.update_tradepos(idx, self.curr_prices[idx], trade_volume)
                    self.status_notifier(underlier, msg)
                    save_status = True
        return save_status

