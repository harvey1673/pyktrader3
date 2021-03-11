# -*- coding:utf-8 -*-
import numpy as np
from pycmqlib3.analytics.data_handler as dh
from pycmqlib3.utility.base import BaseObject, fcustom
from strategy import Strategy


class XSMOMRetTrader(Strategy):
    common_params = dict(Strategy.common_params, **{'data_func': [['ROC', 'dh.ROC', 'dh.roc'], \
                                                                    ["MA", "dh.MA", "dh.ma"]], \
                                                    'batch_setup': {'f1': ['m58', 9, 3, 0.2, 1],}, \
                                                    'pos_scaler': 300000.0,\
                                                    'vol_win': 20})
    asset_params = dict({'batch': 'f1', 'ret': 0.0, 'ret_name': "", \
                         "scaling_field": "close"}, **Strategy.asset_params)

    def __init__(self, config, agent=None):
        Strategy.__init__(self, config, agent)
        numAssets = len(self.underliers)
        self.run_idx = {}
        self.run_time = {}
        self.update_time = [(datetime.date(1970, 1,1), 0)] * numAssets
        self.run_freq = {}
        self.lookback = {}
        self.hold_period = {}
        self.cutoff = {}
        self.ma_win = {}
        self.pos_mode = {}
        self.product_indices = {}
        self.bar_update = {}
        index_list = []
        self.pos_table = {}
        self.tick_base = [0.0] * numAssets
        for idx, underlier in enumerate(self.underliers):
            if self.batch[idx] not in self.product_indices:
                self.product_indices[self.batch[idx]] = []
            self.product_indices[self.batch[idx]].append(idx)
        for batch in self.batch_setup:
            self.run_idx[batch] = 0
            self.run_freq[batch] = self.batch_setup[batch][0]
            self.lookback[batch] = self.batch_setup[batch][1]
            self.hold_period[batch] = self.batch_setup[batch][2]
            self.cutoff[batch] = self.batch_setup[batch][3]
            self.ma_win[batch] = self.batch_setup[batch][4]
            self.pos_mode[batch] = self.batch_setup[batch][5] if len(self.batch_setup[batch]) > 5 else 0
            self.bar_update[batch] = []
            index_list = index_list + [(batch, i) for i in range(self.hold_period[batch])]
            for i in range(self.hold_period[batch]):
                index_list.append((batch, i))
                self.run_time[(batch, i)] = (datetime.date(1970,1,1), 0)
            dtypes = [(self.underliers[idx][0], 'i8') for idx in self.product_indices[batch]]
            self.pos_table[batch] = np.array(np.zeros(self.hold_period[batch]), dtype = dtypes)

    def load_state(self):
        super(XSMOMRetTrader, self).load_state()
        self.refresh_pos_table()

    def save_local_variables(self, file_writer):
        for idx in self.run_time:
            row = ['run_time', idx[0], idx[1], self.run_time[idx][0], self.run_time[idx][1]]
            file_writer.writerow(row)

    def load_local_variables(self, row):
        if row[0] == 'run_time':
            batch = row[1]
            idx = int(row[2])
            run_date = datetime.datetime.strptime(row[3], '%Y-%m-%d').date()
            bar_id = int(row[4])
            self.run_time[(batch, idx)] = (run_date, bar_id)

    def refresh_pos_table(self):
        for idx in self.positions:
            under = self.underlying[idx]
            for trade_pos in self.positions[idx]:
                str_split = trade_pos.tag.split('_')
                batch = str(str_split[0])
                s = int(str_split[1])
                self.pos_table[batch][under.name][s] = trade_pos.pos

    def refresh_run_idx(self):
        for batch in self.batch_setup:
            self.run_idx[batch], (last_d, last_bar) = max(enumerate([self.run_time[(batch, idx)] for idx in range(self.hold_period[batch])]), key=operator.itemgetter(1))
            self.run_idx[batch] = (self.run_idx[batch] + 1) % self.hold_period[batch]

    def register_func_freq(self):
        for idx, under in enumerate(self.underliers):
            inst = under[0]
            freq = self.run_freq[self.batch[idx]]
            if ('s' in freq) or ('m' in freq):
                self.agent.register_data_func(inst, freq, None)
        for idx, under_obj in enumerate(self.underlying):
            batch = self.batch[idx]
            freq = self.run_freq[batch]
            roc_win = self.lookback[batch]
            roc_name = '%s_CLOSE_%s' % (self.data_func[0][0], roc_win)
            roc_sfunc = eval(self.data_func[0][1])
            roc_rfunc = eval(self.data_func[0][2])
            roc_fobj = BaseObject(name = roc_name, sfunc = fcustom(roc_sfunc, n = roc_win), rfunc = fcustom(roc_rfunc, n = roc_win))
            self.agent.register_data_func(under_obj.name, freq, roc_fobj)
            ma_win = self.ma_win[batch]
            ma_name =  '%s_%s_%s' % (self.data_func[1][0], roc_name, ma_win)
            self.ret_name[idx] = ma_name
            ma_sfunc = eval(self.data_func[1][1])
            ma_rfunc = eval(self.data_func[1][2])
            ma_fobj = BaseObject(name = ma_name, sfunc = fcustom(ma_sfunc, n = ma_win, field = roc_name), rfunc = fcustom(ma_rfunc, n = ma_win, field = roc_name))
            self.agent.register_data_func(under_obj.name, freq, ma_fobj)
            if self.pos_mode[batch] > 0 and (len(self.data_func) > 2):
                vol_name = self.data_func[2][0] + str(self.vol_win)
                vol_sfunc = eval(self.data_func[2][1])
                vol_rfunc = eval(self.data_func[2][2])
                atr_fobj = BaseObject(name = vol_name,
                              sfunc=fcustom(vol_sfunc, n=self.vol_win),
                              rfunc=fcustom(vol_rfunc, n=self.vol_win))
                self.scaling_field[idx] = vol_name
                self.agent.register_data_func(under_obj.name, 'd', atr_fobj)
            else:
                self.scaling_field[idx] = 'close'

    def register_bar_freq(self):
        for idx, under in enumerate(self.underlying):
            self.agent.inst2strat[under.name][self.name] = self.run_freq[self.batch[idx]]

    def update_trade_unit(self):
        for idx, underlier in enumerate(self.underliers):
            instID = underlier[0]
            hold_period = self.hold_period[self.batch[idx]]
            xdata = self.agent.day_data[instID].data
            vol_factor = xdata[self.scaling_field[idx]][-1]
            self.trade_unit[idx] = int(self.pos_scaler * self.alloc_w[idx] \
                                       / (self.conv_f[instID] * vol_factor * hold_period) + 0.5)

    def initialize(self):
        for idx, underlier in enumerate(self.underliers):
            self.tick_base[idx] = max([self.agent.instruments[inst].tick_base for inst in underlier])
            self.update_mkt_state(idx)
        self.update_trade_unit()

    def update_mkt_state(self, idx):
        freq = self.run_freq[self.batch[idx]]
        xdata = self.agent.bar_factory[self.underlying[idx].name][freq].data
        nlen = self.lookback[self.batch[idx]]
        self.ret[idx] = xdata[self.ret_name[idx]][-1] * 100.0
        self.update_time[idx] = (xdata['date'][-1], xdata['bar_id'][-1])

    def on_bar(self, idx, freq_list):
        batch = self.batch[idx]
        freq = self.run_freq[batch]
        self.update_mkt_state(idx)
        save_status = False
        run_idx = self.run_idx[batch]
        if (self.update_time[idx] > self.run_time[(batch, run_idx)]) and (idx in self.product_indices[batch]):
            self.bar_update[batch].append(idx)
            if len(self.bar_update[batch]) >= len(self.product_indices[batch]):
                save_status = self.run_xs_batch(batch)
                print("prev run_time = %s, curr = %s" % (self.run_time[(batch, run_idx)], self.update_time[idx]))
                self.run_time[(batch, run_idx)] = self.update_time[idx]
                self.bar_update[batch] = []
        return save_status

    def on_tick(self, idx, ctick):
        pass

    def run_xs_batch(self, batch):
        save_status = False
        # find current holding period index
        idy = self.run_idx[batch]
        # ranking the performance and find the long and short
        kcut = min(int(self.cutoff[batch] * len(self.product_indices[batch])+0.5), 1)
        perf_array = [self.ret[k] for k in self.product_indices[batch]]
        sort_index = np.argsort(perf_array)
        curr_pos = list(self.pos_table[batch][idy])
        target_dir = np.zeros(len(curr_pos))
        target_dir[sort_index[:kcut]] = -1
        target_dir[sort_index[-kcut:]] = 1
        for k, (pos_old, new_dir) in enumerate(zip(curr_pos, target_dir)):
            idx = self.product_indices[batch][k]
            instID = self.underliers[idx][0]
            tag_key = batch + '_' + str(idy)
            new_pos = new_dir * self.trade_unit[idx]
            if (pos_old != 0) and (pos_old * new_dir <= 0):
                for tpos in self.positions[idx]:
                    if tag_key == tpos.tag:
                        msg = '%s to close position for inst = %s, pos=%s, batch = %s, run_idx = %s' % \
                              (self.name, instID, tpos.pos, batch, idy)
                        self.close_tradepos(idx, tpos, self.curr_prices[idx] - sign(pos_old) * self.tick_base[idx])
                        self.status_notifier(instID, msg)
                        save_status = True
                        self.pos_table[batch][instID][idy] = 0
            if (new_dir !=0) and (pos_old * new_dir <= 0):
                msg = '%s to open position for inst = %s, pos=%s, batch = %s, run_idx = %s' % \
                      (self.name, instID, new_pos, batch, idy)
                self.open_tradepos(idx, int(new_dir), self.curr_prices[idx] + new_dir * self.tick_base[idx], tag = tag_key)
                self.status_notifier(instID, msg)
                save_status = True
                self.pos_table[batch][instID][idy] = new_pos
        self.run_idx[batch] = (self.run_idx[batch] + 1) % self.hold_period[batch]
        return save_status

