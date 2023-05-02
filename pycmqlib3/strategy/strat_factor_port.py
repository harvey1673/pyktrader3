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
                            'repo_type': 'asset',
                            'factor_repo': {'lr_sum_20': {'name': 'lr_sum_20','type': 'ts', 'param': [], 'weight': 1.0, \
                                            'rebal': 5, 'threshold': 0.0}},
                            'factor_data': {},
                            'factor_pos': {},
                            'pos_scaler': 1000000.0, 'vol_win': 20,
                            'exec_bar_list': [305],
                            'fact_db_table': 'fut_fact_data',
                            'roll_label': 'CAL_30b',
                            'vol_key': 'atr',
                            'hist_fact_lookback': 250, })
    asset_params = dict({}, **Strategy.asset_params)

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
        self.min_trade_size = [1] * numAssets
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
            fact_list = list(set([self.factor_repo[fact]['name'] for fact in self.factor_repo.keys()]))
            vol_df = load_factor_data(self.prod_list,
                                      factor_list=[self.vol_key],
                                      roll_label=self.roll_label,
                                      start=start_date,
                                      end=end_date,
                                      freq=self.freq,
                                      db_table=self.fact_db_table)
            vol_df = pd.pivot_table(vol_df[vol_df['fact_name'] == self.vol_key], values='fact_val',
                                    index=['date', 'serial_key'],
                                    columns=['product_code'],
                                    aggfunc='last')
            for idx, underlier in enumerate(self.underliers):
                self.vol_weight[idx] = self.pos_scaler/vol_df[self.prod_list[idx]].iloc[-1]/self.conv_f[underlier[0]]
            if self.repo_type == 'port':
                df = load_factor_data([],
                                      factor_list=fact_list,
                                      roll_label=self.roll_label,
                                      start=start_date,
                                      end=end_date,
                                      freq=self.freq,
                                      db_table=self.fact_db_table)
                xdf = pd.pivot_table(df, values='fact_val', index=['date', 'serial_key'],
                                     columns=['fact_name'], aggfunc='last')
                for fact in self.factor_repo:
                    self.fact_data[fact] = pd.concat([xdf[fact]] * len(self.prod_list), axis=1)
                    self.fact_data[fact].columns = self.prod_list
            else:
                df = load_factor_data(self.prod_list,
                                      factor_list=fact_list,
                                      roll_label=self.roll_label,
                                      start=start_date,
                                      end=end_date,
                                      freq=self.freq,
                                      db_table=self.fact_db_table)
                for fact in self.factor_repo:
                    xdf = pd.pivot_table(df[df['fact_name'] == self.factor_repo[fact]['name']], values='fact_val',
                                         index=['date', 'serial_key'], columns=['product_code'],
                                         aggfunc='last')
                    for prod in self.prod_list:
                        if prod not in xdf.columns:
                            xdf[prod] = np.nan
                    self.fact_data[fact] = xdf[self.prod_list]
        else:
            print("Update from the strategy is not implemented yet.")
        self.update_target_pos()

    def update_target_pos(self):
        self.pos_summary = pd.DataFrame()
        for fact in self.factor_repo:
            rebal_freq = self.factor_repo[fact]['rebal']
            weight =  self.factor_repo[fact]['weight']
            self.factor_pos[fact] = pd.DataFrame(index = self.fact_data[fact].index, \
                                                 columns = self.fact_data[fact].columns)
            self.factor_pos[fact] = self.fact_data[fact].copy()
            if self.factor_repo[fact]['type'] != 'pos':
                if 'xs' in self.factor_repo[fact]['type']:
                    xs_split = self.factor_repo[fact]['type'].split('-')
                    if len(xs_split) <= 1:
                        xs_signal = 'rank_cutoff'
                    else:
                        xs_signal = xs_split[1]
                    if xs_signal == 'rank_cutoff':
                        cutoff = self.factor_repo[fact]['threshold']
                        lower_rank = int(len(self.prod_list) * cutoff) + 1
                        upper_rank = len(self.prod_list) - int(len(self.prod_list) * cutoff)
                        rank_df = self.factor_pos[fact].rank(axis=1)
                        self.factor_pos[fact] = rank_df.gt(upper_rank, axis=0) * 1.0 - rank_df.lt(lower_rank, axis=0) * 1.0
                    elif xs_signal == 'demedian':
                        median_ts = self.factor_pos[fact].quantile(0.5, axis=1)
                        self.factor_pos[fact] = self.factor_pos[fact].sub(median_ts, axis=0)
                    elif xs_signal == 'demean':
                        mean_ts = self.factor_pos[fact].mean(axis=1)
                        self.factor_pos[fact] = self.factor_pos[fact].sub(mean_ts, axis=0)
                    elif xs_signal == 'rank':
                        rank_df = self.factor_pos[fact].rank(axis=1)
                        median_ts = rank_df.quantile(0.5, axis=1)
                        self.factor_pos[fact] = rank_df.sub(median_ts, axis=0) / len(self.prod_list) * 2.0
                    elif len(xs_signal) > 0:
                        print('unsupported xs signal types')
                self.factor_pos[fact] = self.factor_pos[fact].rolling(rebal_freq).mean().fillna(0.0)
            fact_pos = pd.Series(self.factor_pos[fact].iloc[-1] * weight, name=fact)
            self.pos_summary = self.pos_summary.append(fact_pos)
        self.pos_summary = self.pos_summary[self.prod_list].round(2)
        net_pos = self.pos_summary.sum()           
        for idx, (under, prodcode) in enumerate(zip(self.underliers, self.prod_list)):            
            if prodcode == 'CJ':
                self.target_pos[idx] = int((self.alloc_w[idx] * net_pos[prodcode] * self.vol_weight[idx]/4 + (0.5 if net_pos[prodcode]>0 else -0.5)))*4
            elif prodcode == 'ZC':
                self.target_pos[idx] = int((self.alloc_w[idx] * net_pos[prodcode] * self.vol_weight[idx]/2 + (0.5 if net_pos[prodcode]>0 else -0.5)))*2
            else:
                self.target_pos[idx] = int(self.alloc_w[idx] * net_pos[prodcode] * self.vol_weight[idx] + (0.5 if net_pos[prodcode]>0 else -0.5))
        self.save_state()        

    def register_func_freq(self):
        for idx, under in enumerate(self.underliers):
            inst = under[0]
            if ('s' in self.freq) or ('m' in self.freq):
                self.agent.register_data_func(inst, self.freq, None)

    def register_bar_freq(self):
        for idx, (under, prev_under) in enumerate(zip(self.underlying, self.prev_underlying)):
            self.agent.inst2strat[under.name][self.name] = 'm1'
            if prev_under:
                self.agent.inst2strat[prev_under.name][self.name] = 'm1'

    def initialize(self):
        for idx, underlier in enumerate(self.underliers):
            self.tick_base[idx] = max([self.agent.instruments[inst].tick_base for inst in underlier])            
            if self.prod_list[idx] in ['CJ']:
                self.min_trade_size[idx] = 4
            elif self.prod_list[idx] in ['ZC']:
                self.min_trade_size[idx] = 2
            else:
                self.min_trade_size[idx] = 1
            # xdata = self.agent.day_data[self.underlying[idx].name].data
            # self.vol_weight[idx] = self.pos_scaler/xdata[self.scaling_field][-1]/self.conv_f[underlier[0]]
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
            if (curr_pos == 0) and (abs(new_pos)>=self.min_trade_size[idx]):
                msg = '%s to open position for inst = %s, pos=%s' % \
                      (self.name, underlier, new_pos)
                dir = 1 if new_pos > 0 else -1
                self.open_tradepos(idx, dir, self.curr_prices[idx], volume = dir * new_pos)
                self.status_notifier(underlier, msg)
                save_status = True
            else:
                trade_volume = new_pos - curr_pos
                if (abs(trade_volume)>=self.min_trade_size[idx]):
                    msg = '%s to modify position for inst = %s, old_pos=%s, new_pos = %s' % \
                          (self.name, underlier, curr_pos, new_pos)
                    self.update_tradepos(idx, self.curr_prices[idx], trade_volume)
                    self.status_notifier(underlier, msg)
                    save_status = True
        return save_status

