#-*- coding:utf-8 -*-
import numpy as np
import logging
from pycmqlib3.analytics.data_handler as dh
from pycmqlib3.utility.base import BaseObject, fcustom
from . strategy import Strategy
 
class DTSplitChan(Strategy):
    common_params =  dict({'daily_close_buffer': 0, 'price_limit_buffer': 5}, \
                          **Strategy.common_params)
    asset_params = dict({'lookbacks': 1, 'ratios': 1.0, 'freq': 1, 'channels': 20, 'ma_chan': 0, 'trend_factor': 0.0, \
                         'vol_ratio': [1.0, 0.0], 'split_mode': 's1', 'channel_type': 0, 'price_mode': 'HL', 'min_rng': 0.004}, **Strategy.asset_params)
    def __init__(self, config, agent = None):
        Strategy.__init__(self, config, agent)
        numAssets = len(self.underliers)
        self.cur_rng = [0.0] * numAssets
        self.chan_high = [-1000000.0] * numAssets
        self.chan_low  = [1000000.0] * numAssets
        self.tday_open = [0.0] * numAssets
        self.ma_level = [0.0] * numAssets
        self.buy_trig = [0.0] * numAssets
        self.sell_trig = [0.0] * numAssets
        self.tick_base = [0.0] * numAssets
        self.max_pos = [1] * numAssets
        nfunc = int(len(self.data_func)/2)
        self.high_func = []
        self.low_func = []
        self.high_field = []
        self.low_field = []
        for i in range(nfunc):
            hfunc = fcustom(eval(self.data_func[i*2][0]), **self.data_func[i*2][2])
            self.high_func.append(hfunc)
            hfield = self.data_func[i*2][1]
            self.high_field.append(hfield)
            lfunc = fcustom(eval(self.data_func[i*2+1][0]), **self.data_func[i*2+1][2])
            self.low_func.append(lfunc)
            lfield = self.data_func[i*2+1][1]
            self.low_field.append(lfield)

    def register_func_freq(self):
        for idx, under in enumerate(self.underliers):
            inst = under[0]
            if ('s' in self.split_mode[idx]) or ('m' in self.split_mode[idx]):
                self.agent.register_data_func(inst, self.split_mode[idx], None)

    def register_bar_freq(self):
        for idx, under in enumerate(self.underliers):
            inst = under[0]
            self.agent.inst2strat[inst][self.name] = 'm1'

    def initialize(self):
        self.load_state()
        for idx, underlier in enumerate(self.underliers):
            inst = underlier[0]
            self.max_pos[idx] = sum(v > 0.0 for v in self.vol_ratio[idx])
            self.tick_base[idx] = self.agent.instruments[inst].tick_base
            min_id = self.agent.instruments[inst].last_tick_id/1000
            min_id = int(min_id/100)*60 + min_id % 100 - self.daily_close_buffer
            self.last_min_id[idx] = int(min_id/60)*100 + min_id % 60
            ddf = self.agent.day_data[inst].data
            mdf = self.agent.bar_factory[inst][self.split_mode[idx]].data
            min_date = mdf['date'][-1]
            last_date = ddf['date'][-1]
            if last_date < min_date:
                self.tday_open[idx] = mdf['open'][-1]
                self.recalc_rng(idx, -1)
            else:
                self.tday_open[idx] = mdf['close'][-1]
                self.recalc_rng(idx, 0)
        self.update_trade_unit()
        self.save_state()

    def recalc_rng(self, idx, shift = 0):
        win = int(self.lookbacks[idx])
        inst = self.underliers[idx][0]
        ddf = self.agent.bar_factory[inst][self.split_mode[idx]].data
        ddf = ddf[:(len(ddf) + shift)]
        if self.channels[idx] > 0:
            mode = self.channel_type[idx]
            self.chan_high[idx] = self.high_func[mode](ddf[self.high_field[mode]][-self.channels[idx]:])
            self.chan_low[idx] = self.low_func[mode](ddf[self.low_field[mode]][-self.channels[idx]:])
        if self.ma_chan[idx] > 0:
            self.ma_level[idx] = np.mean(ddf['close'][-self.ma_chan[idx]:])
        if win > 0:
            self.cur_rng[idx] = max(max(ddf['high'][-win:])- min(ddf['close'][-win:]), \
                                    max(ddf['close'][-win:]) - min(ddf['low'][-win:]))
        elif win == 0:
            self.cur_rng[idx] = max(max(ddf['high'][-2:])- min(ddf['close'][-2:]), \
                                    max(ddf['close'][-2:]) - min(ddf['low'][-2:]))
            self.cur_rng[idx] = max(self.cur_rng[idx] * 0.5, ddf['high'][-1]-ddf['close'][-1], \
                                    ddf['close'][-1]-ddf['low'][-1])
        else:
            self.cur_rng[idx] = max(ddf['high'][-1] - ddf['low'][-1], abs(ddf['close'][-1] - ddf['close'][-2]))
        rng = max(self.cur_rng[idx] * self.ratios[idx], self.tday_open[idx] * self.min_rng[idx])
        up_fact = 1.0
        dn_fact = 1.0
        if (self.ma_chan[idx] > 0):
            if (self.tday_open[idx] < self.ma_level[idx]):
                up_fact += self.trend_factor[idx]
            else:
                dn_fact += self.trend_factor[idx]
        self.buy_trig[idx] = min(self.tday_open[idx] + up_fact * rng,
                       self.agent.instruments[inst].up_limit - self.price_limit_buffer * self.tick_base[idx])
        self.sell_trig[idx] = max(self.tday_open[idx] - dn_fact * rng,
                        self.agent.instruments[inst].down_limit + self.price_limit_buffer * self.tick_base[idx])

    def save_local_variables(self, file_writer):
        pass
    
    def load_local_variables(self, row):
        pass

    def on_bar(self, idx, freq_list):
        inst = self.underliers[idx][0]
        if self.split_mode[idx] in freq_list:
            self.tday_open[idx] = self.agent.instruments[inst].price
            self.recalc_rng(idx)
            return True
        if (self.freq[idx]>0):
            inst = self.underliers[idx][0]
            min_data = self.agent.bar_factory[inst]['m1'].bar_array.data
            if self.price_mode[idx] == 'HL':
                buy_p = min_data['high'][-1]
                sell_p = min_data['low'][-1]
            elif self.price_mode[idx] == 'C':
                buy_p = min_data['close'][-1]
                sell_p = buy_p
            elif self.price_mode[idx] == 'TP':
                buy_p = (min_data['high'][-1] + min_data['low'][-1] + min_data['close'][-1])/3.0
                sell_p = buy_p
            else:
                self.on_log('Unsupported price type for strat=%s inst=%s' % (self.name, inst), level = logging.WARNING)
            save_status = self.check_trigger(idx, buy_p, sell_p)
            return save_status

    def on_tick(self, idx, ctick):
        if self.freq[idx] == 0:
            self.check_trigger(idx, self.curr_prices[idx], self.curr_prices[idx])

    def check_trigger(self, idx, buy_price, sell_price):
        save_status = False
        if len(self.submitted_trades[idx]) > 0:
            return save_status
        inst = self.underliers[idx][0]
        min_id = int(self.agent.instruments[inst].last_update/1000.0)
        num_pos = len(self.positions[idx])
        buysell = 0
        if num_pos > self.max_pos[idx]:
            self.on_log('something wrong - number of tradepos is more than max_pos=%s' % self.max_pos[idx], level = logging.WARNING)
            return save_status
        elif num_pos >= 1:
            buysell = self.positions[idx][0].direction
        if (min_id >= self.last_min_id[idx]):
            if (buysell!=0) and (self.close_tday[idx]):
                msg = '%s to close position before EOD for inst = %s, direction=%s, num_pos=%s, current min_id = %s' \
                        % (self.name, inst, buysell, num_pos, min_id)
                for tp in self.positions[idx]:
                    self.close_tradepos(idx, tp, self.curr_prices[idx] - buysell * self.tick_base[idx])
                self.status_notifier(inst, msg)
                save_status = True
            return save_status
        if ((buy_price >= self.buy_trig[idx]) and (buysell <0)) or \
                ((sell_price <= self.sell_trig[idx]) and (buysell > 0)):
            msg = '%s to close position for inst = %s, open= %s, buy_trig=%s, sell_trig=%s, buy_price= %s, sell_price= %s, direction=%s, num_pos=%s' \
                    % (self.name, inst, self.tday_open[idx], self.buy_trig[idx], self.sell_trig[idx], buy_price, sell_price, buysell, num_pos)
            for tp in self.positions[idx]:
                self.close_tradepos(idx, tp, self.curr_prices[idx] - buysell * self.tick_base[idx])
            self.status_notifier(inst, msg)
            save_status = True
            num_pos = 0
        if (self.trade_unit[idx] <= 0):
            return save_status
        if  (buy_price >= self.buy_trig[idx]):
            buysell = 1
        elif (sell_price <= self.sell_trig[idx]):
            buysell = -1
        else:
            buysell = 0
        if (buysell!=0) and (self.vol_ratio[idx][0]>0) and (num_pos == 0):
            new_vol = int(self.trade_unit[idx] * self.vol_ratio[idx][0])
            msg = '%s to open position for inst = %s, open= %s, buy_trig=%s, sell_trig=%s, buy_price= %s, sell_price= %s, direction=%s, volume=%s' \
                    % (self.name, inst, self.tday_open[idx], self.buy_trig[idx], self.sell_trig[idx], buy_price, sell_price, buysell, new_vol)
            self.open_tradepos(idx, buysell, self.curr_prices[idx] + buysell * self.tick_base[idx], new_vol)
            self.status_notifier(inst, msg)
            save_status = True
            num_pos = 1
        if (num_pos < self.max_pos[idx]) and (self.vol_ratio[idx][1]>0) and (((buysell > 0) and (buy_price >= self.chan_high[idx])) or ((buysell < 0) and (sell_price <= self.chan_low[idx]))):
            addon_vol = int(self.vol_ratio[idx][1]*self.trade_unit[idx])
            msg = '%s to add position for inst = %s, high=%s, low=%s, buy= %s, sell= %s, direction=%s, volume=%s' \
                    % (self.name, inst, self.chan_high[idx], self.chan_low[idx], buy_price, sell_price, buysell, addon_vol)
            self.open_tradepos(idx, buysell, self.curr_prices[idx] + buysell * self.tick_base[idx], addon_vol)
            self.status_notifier(inst, msg)
            save_status = True
        return save_status
