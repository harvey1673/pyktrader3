#-*- coding:utf-8 -*-
from base import *
from misc import *
import data_handler as dh
import copy
from strategy import *
 
class BbandPChanTrader(Strategy):
    common_params =  dict( Strategy.common_params, **{'channel_keys': ['DONCH_HC', 'DONCH_LC'], 'band_keys': ['MA_CLOSE', 'STDEV_CLOSE'], \
                                                      'price_limit_buffer': 5, 'daily_close_buffer': 0, \
                                                      'data_func': [["DONCH_HC", "dh.DONCH_H", "dh.donch_h", {'field':'close'}], ["DONCH_LC", "dh.DONCH_L", "dh.donch_l", {'field':'close'}], \
                                                                      ['MA_CLOSE_', 'dh.MA', 'dh.ma'], ['STDEV_CLOSE_', 'dh.STDEV', 'dh.stdev']]})
    asset_params = dict({'band_win': 40, 'ratios': 1.0, 'freq': 'm30', 'channels': 20, 'daily_close': False, }, **Strategy.asset_params)
    def __init__(self, config, agent = None):
        Strategy.__init__(self, config, agent)
        numAssets = len(self.underliers)
        self.upper_band = [0.0] * numAssets
        self.mid_band   = [0.0] * numAssets
        self.lower_band = [0.0] * numAssets
        self.chan_high = [0.0] * numAssets
        self.chan_low  = [0.0] * numAssets
        self.tick_base = [0.0] * numAssets

    def register_func_freq(self):
        for idx, under in enumerate(self.underliers):
            for idy, infunc in enumerate(self.data_func):
                name  = infunc[0]
                sfunc = eval(infunc[1])
                rfunc = eval(infunc[2])
                if len(infunc) > 3:
                    fargs = infunc[3]
                else:
                    fargs = {}
                if idy < 2:
                    chan = self.channels[idx]
                else:
                    chan = self.band_win[idx]
                fobj = BaseObject(name = name + str(chan), sfunc = fcustom(sfunc, n = chan, **fargs), rfunc = fcustom(rfunc, n = chan, **fargs))
                self.agent.register_data_func(under[0], self.freq[idx], fobj)

    def register_bar_freq(self):
        for idx, under in enumerate(self.underliers):
            inst = under[0]
            self.agent.inst2strat[inst][self.name] = self.freq[idx]

    def initialize(self):
        self.load_state()
        for idx, underlier in enumerate(self.underliers):
            inst = underlier[0]
            self.tick_base[idx] = self.agent.instruments[inst].tick_base
            min_id = self.agent.instruments[inst].last_tick_id/1000
            min_id = int(min_id/100)*60 + min_id % 100 - self.daily_close_buffer
            self.last_min_id[idx] = int(min_id/60)*100 + min_id % 60
            self.update_mkt_state(idx)
        self.update_trade_unit()

    def update_mkt_state(self, idx):
        instID = self.underliers[idx][0]
        xdf = self.agent.bar_factory[instID][self.freq[idx]].data
        key = self.channel_keys[0] + str(self.channels[idx])
        self.chan_high[idx] = xdf[key][-2]
        key = self.channel_keys[1] + str(self.channels[idx])
        self.chan_low[idx]  = xdf[key][-2]
        key = self.band_keys[0] + '_' + str(self.band_win[idx])
        self.mid_band[idx] = xdf[key][-1]
        key = self.band_keys[1] + '_' + str(self.band_win[idx])
        stdev = xdf[key][-1]
        self.upper_band[idx] = self.mid_band[idx] + self.ratios[idx] * stdev
        self.lower_band[idx] = self.mid_band[idx] - self.ratios[idx] * stdev

    def on_bar(self, idx):
        inst = self.underliers[idx][0]
        self.update_mkt_state(idx)
        if len(self.positions[idx]) > 0:
            self.positions[idx][0].set_exit(self.mid_band[idx])
        self.check_trigger(idx)

    def on_tick(self, idx, ctick):
        pass

    def check_trigger(self, idx):
        if len(self.submitted_trades[idx]) > 0:
            return
        inst = self.underliers[idx][0]
        curr_p = self.curr_prices[idx]
        min_id = self.agent.tick_id/1000.0
        num_pos = len(self.positions[idx])
        buysell = 0
        if num_pos > 0:
            buysell = self.positions[idx][0].direction
        tick_base = self.tick_base[idx]
        save_status = False
        if (min_id >= self.last_min_id[idx]):
            if (buysell!=0) and (self.close_tday[idx]):
                msg = 'BbandPchanTrader to close position before EOD for inst = %s, direction=%s, volume=%s, current tick_id = %s' \
                        % (inst, buysell, self.trade_unit[idx], min_id)
                self.close_tradepos(idx, self.positions[idx][0], self.curr_prices[idx] - buysell * tick_base)
                self.status_notifier(inst, msg)
                save_status = True
            return save_status
        if (buysell != 0) and (self.positions[idx][0].check_exit(curr_p, 0)):
            msg = 'BbandPchanTrader to close position after hitting MA line for inst = %s, direction=%s, current price = %s, exit target = %s' \
                    % (inst, buysell, curr_p, self.positions[idx][0].exit_target)
            self.close_tradepos(idx, self.positions[idx][0], self.curr_prices[idx] - buysell * tick_base)
            self.status_notifier(inst, msg)
            save_status = True
            buysell = 0
        if (self.trade_unit[idx] > 0) and (buysell == 0):
            if curr_p >= max(self.chan_high[idx], self.upper_band[idx]):
                buysell = 1
            elif curr_p <= min(self.chan_low[idx], self.lower_band[idx]):
                buysell = -1
            if buysell != 0:
                msg = 'BbandPchTrader to open position for inst = %s, chan_high=%s, chan_low=%s, upper_band=%s, lower_band=%s, curr_price= %s, direction=%s, volume=%s' \
                                        % (inst, self.chan_high[idx], self.chan_low[idx], self.upper_band[idx], self.lower_band[idx], self.curr_prices[idx], buysell, self.trade_unit[idx])
                self.open_tradepos(idx, buysell, self.curr_prices[idx] + buysell * tick_base)
                self.status_notifier(inst, msg)
                save_status = True
        return save_status
