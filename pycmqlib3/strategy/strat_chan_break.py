# -*- coding:utf-8 -*-
import numpy as np
from pycmqlib3.analytics.data_handler as dh
from pycmqlib3.utility.base import BaseObject, fcustom
from . strategy import Strategy

class ChanBreak(Strategy):
    common_params = dict({'daily_close_buffer': 0, 'price_limit_buffer': 5}, **Strategy.common_params)
    asset_params = dict({'split_mode': 's1', 'atr_win': 10, 'stoploss': 1.0, 'entry_chan': 20, 'exit_chan': 10}, **Strategy.asset_params)

    def __init__(self, config, agent=None):
        Strategy.__init__(self, config, agent)
        numAssets = len(self.underliers)
        self.entry_high = [0.0] * numAssets
        self.entry_low = [0.0] * numAssets
        self.exit_high = [0.0] * numAssets
        self.exit_low = [0.0] * numAssets
        self.atr = [0.0] * numAssets
        self.long_exit = [0.0] * numAssets
        self.short_exit = [0.0] * numAssets
        self.long_entry = [0.0] * numAssets
        self.short_entry = [0.0] * numAssets
        self.tick_base = [0.0] * numAssets

    def register_bar_freq(self):
        for idx, under_obj in enumerate(self.underlying):
            inst = under_obj.name
            self.agent.inst2strat[inst][self.name] = 'm1'


    def register_func_freq(self):
        for idx, under_obj in enumerate(self.underlying):
            name = "ATR"
            sfunc = eval("dh.ATR")
            rfunc = eval("dh.atr")
            fobj = BaseObject(name = name + str(self.atr_win[idx]),
                              sfunc=fcustom(sfunc, n = self.atr_win[idx]),
                              rfunc=fcustom(rfunc, n = self.atr_win[idx]))
            self.agent.register_data_func(under_obj.name, self.split_mode[idx], fobj)

    def initialize(self):
        self.load_state()
        for idx, underlier in enumerate(self.underliers):
            self.tick_base[idx] = max([self.agent.instruments[inst].tick_base for inst in underlier])
            inst = underlier[0]
            under_obj = self.underlying[idx]
            min_id = self.agent.instruments[inst].last_tick_id / 1000
            min_id = int(min_id / 100) * 60 + min_id % 100 - self.daily_close_buffer
            self.last_min_id[idx] = int(min_id / 60) * 100 + min_id % 60
            ddf = self.agent.day_data[inst].data
            mdf = self.agent.bar_factory[under_obj.name][self.split_mode[idx]].data
            min_date = mdf['date'][-1]
            last_date = ddf['date'][-1]
            nshift = -1 if last_date < min_date else 0
            self.recalc_rng(idx, shift = nshift)
        self.update_trade_unit()
        self.save_state()

    def recalc_rng(self, idx, shift = 0):
        under_obj = self.underlying[idx]
        #up_limit = self.agent.instruments[inst].up_limit - self.price_limit_buffer * self.tick_base[idx]
        #dn_limit = self.agent.instruments[inst].down_limit + self.price_limit_buffer * self.tick_base[idx]
        df = self.agent.bar_factory[under_obj.name][self.split_mode[idx]].data
        df = df[:(len(df)+shift)]
        self.entry_high[idx] = max(df['high'][-self.entry_chan[idx]:])
        self.entry_low[idx] = min(df['low'][-self.entry_chan[idx]:])
        self.exit_high[idx] = max(df['high'][-self.exit_chan[idx]:])
        self.exit_low[idx] = min(df['low'][-self.exit_chan[idx]:])
        self.atr[idx] = df['ATR'+str(self.atr_win[idx])][-1]
        self.long_exit[idx] = max(self.exit_low[idx], self.entry_high[idx] - self.atr[idx] * self.stoploss[idx])
        self.short_exit[idx] = min(self.exit_high[idx], self.entry_low[idx] + self.atr[idx] * self.stoploss[idx])
        self.long_entry[idx] = self.entry_high[idx]
        self.short_entry[idx] = self.entry_low[idx]

    def save_local_variables(self, file_writer):
        pass

    def load_local_variables(self, row):
        pass

    def on_bar(self, idx, freq_list):
        status = False
        if self.split_mode[idx] in freq_list:
            self.recalc_rng(idx)
            status = self.check_trigger(idx, self.curr_prices[idx], self.curr_prices[idx])
        return status

    def on_tick(self, idx, ctick):
        return self.check_trigger(idx, self.curr_prices[idx], self.curr_prices[idx])

    def check_trigger(self, idx, buy_price, sell_price):
        save_status = False
        if len(self.submitted_trades[idx]) > 0:
            return save_status
        inst = self.underliers[idx][0]
        min_id = int(self.agent.instruments[inst].last_update / 1000.0)
        under = self.underlying[idx].name
        num_pos = len(self.positions[idx])
        buysell = 0
        if num_pos >= 1:
            buysell = self.positions[idx][0].direction
        if (min_id >= self.last_min_id[idx]):
            if (buysell != 0) and (self.close_tday[idx]):
                msg = 'Chanbreak to close position before EOD for inst = %s, direction=%s, current min_id = %s' \
                      % (under, buysell, min_id)
                for tp in self.positions[idx]:
                    self.close_tradepos(idx, tp, self.curr_prices[idx] - buysell * self.tick_base[idx])
                self.status_notifier(under, msg)
                save_status = True
            return save_status
        if ((buy_price >= self.short_exit[idx]) and (buysell < 0)) or ((sell_price <= self.long_exit[idx]) and (buysell > 0)):
            msg = '%s to close position for inst = %s, long_exit=%s, short_exit=%s, buy_price= %s, sell_price= %s, direction=%s, num_pos=%s' \
                  % (self.name, under, self.long_exit[idx], self.short_exit[idx], buy_price, sell_price, buysell, num_pos)
            for tp in self.positions[idx]:
                self.close_tradepos(idx, tp, self.curr_prices[idx] - buysell * self.tick_base[idx])
            self.status_notifier(inst, msg)
            save_status = True
            num_pos = 0
        if (self.trade_unit[idx] <= 0):
            return save_status
        if (buy_price >= self.long_entry[idx]):
            buysell = 1
        elif (sell_price <= self.short_entry[idx]):
            buysell = -1
        else:
            buysell = 0
        if (buysell != 0) and (num_pos == 0):
            new_vol = int(self.trade_unit[idx])
            msg = '%s to open position for inst = %s, long_entry=%s, short_entry=%s, buy_price= %s, sell_price= %s, direction=%s, volume=%s' \
                  % (self.name, under, self.long_entry[idx], self.short_entry[idx], buy_price, sell_price, buysell, new_vol)
            self.open_tradepos(idx, buysell, self.curr_prices[idx] + buysell * self.tick_base[idx], new_vol)
            self.status_notifier(under, msg)
            save_status = True
        return save_status
