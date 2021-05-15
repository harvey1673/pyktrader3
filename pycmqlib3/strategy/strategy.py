#-*- coding:utf-8 -*-
import datetime
import csv
import json
import os
import shutil
import logging
from pycmqlib3.core.event_type import EVENT_LOG, EVENT_MAIL
from pycmqlib3.core.event_engine import Event
from pycmqlib3.core.trade_position import tradepos2dict, tradepos_header, TradePos, TargetTrailTradePos
from pycmqlib3.core.trade import XTrade
from pycmqlib3.core.trade_exec_algo import ExecAlgo1DFixT, ExecAlgoFixTimer
from pycmqlib3.core.trading_const import OrderType, TradeStatus
from pycmqlib3.utility.misc import NO_ENTRY_TIME

class Strategy(object):
    common_params = {'name': 'test_strat', 'email_notify':{}, 'data_func': [], 'pos_scaler': 1.0, \
                     'daily_close_buffer': 0, 'pos_class': 'TradePos', 'pos_args': {}, 'is_disabled': False}
    asset_params = {'underliers': [], 'volumes': [], 'trade_unit': 1,  'alloc_w': 0.0, 'price_unit': None, \
                    'close_tday': False, 'last_min_id': 2059, 'trail_loss': 0, \
                    'prev_underliers': [], 'run_flag': 1, 'exec_class': 'ExecAlgoFixTimer', \
                    'exec_args': {'max_vol': 50, 'time_period': 200, 'price_type': OrderType.LIMIT.value, \
                                  'tick_num': 0, 'price_mode': -1}}
    def __init__(self, config, agent = None):
        self.load_config(config)
        num_assets = len(self.underliers)
        self.instIDs = self.dep_instIDs()
        self.underlying = [None] * num_assets
        self.prev_underlying = [None] * num_assets
        self.positions  = dict([(idx, []) for idx in range(num_assets)])
        self.submitted_trades = dict([(idx, []) for idx in range(num_assets)])
        self.agent = agent
        self.folder = ''
        self.inst2idx = dict([(inst, []) for inst in self.instIDs])
        self.under2idx = {}
        self.num_entries = [0] * num_assets
        self.num_exits   = [0] * num_assets
        self.curr_pos = [0] * num_assets
        self.curr_prices = [0.0] * num_assets
        self.conv_f = {}
        email_dict = {}
        for key in self.email_notify:
            for prod in self.email_notify[key]:
                email_dict[str(prod)] = str(key)
        self.email_notify = email_dict

    def save_config(self):
        config = {}
        d = self.__dict__
        for key in self.common_params:
            config[key] = d[key]
        config['assets'] = []
        for idx, under in enumerate(self.underliers):
            asset = {}
            for key in self.asset_params:
                asset[key] = d[key][idx]
            config['assets'].append(asset)
        fname = self.folder + 'config.json'
        with open(fname, 'w') as ofile:
            json.dump(config, ofile)        
    
    def load_config(self, config):
        d = self.__dict__
        for key in self.common_params:
            d[key] = config.get(key, self.common_params[key])
        for key in self.asset_params:
            d[key] = []
        for asset in config['assets']:
            for key in self.asset_params:
                d[key].append(asset.get(key, self.asset_params[key]))
        
    def dep_instIDs(self):
        return list(set().union(*self.underliers).union(*self.prev_underliers))

    def set_agent(self, agent):
        self.agent = agent
        self.folder = self.agent.folder + self.name + '_'
        for idx, (under, vol) in enumerate(zip(self.underliers, self.volumes)):
            self.underlying[idx] = self.agent.get_underlying(under, self.volumes[idx], self.price_unit[idx])
            self.under2idx[self.underlying[idx].name] = idx
            for inst in under:
                self.inst2idx[inst].append(idx)
            if len(self.prev_underliers[idx]) > 0:
                self.prev_underlying[idx] = self.agent.get_underlying(self.prev_underliers[idx], self.volumes[idx], self.price_unit[idx])
                self.under2idx[self.prev_underlying[idx].name] = idx
                for inst in self.prev_underliers[idx]:
                    self.inst2idx[inst].append(idx)
        self.conv_f = dict([(inst, self.agent.instruments[inst].multiple) for inst in self.instIDs])
        self.register_func_freq()
        self.register_bar_freq()
        self.load_state()

    def register_func_freq(self):
        pass

    def register_bar_freq(self):
        pass

    def initialize(self):
        self.update_trade_unit()
    
    def on_log(self, text, level = logging.INFO, args = {}):    
        event = Event(type = EVENT_LOG, priority = 1000)
        event.dict['data'] = text
        event.dict['owner'] = "strategy_" + self.name
        event.dict['level'] = level
        self.agent.event_engine.put(event)
        
    def on_trade(self, xtrade):
        under_key = xtrade.underlying.name
        idx = self.under2idx[under_key]
        entry_ids = [ tp.entry_tradeid for tp in self.positions[idx]]
        exit_ids = [tp.exit_tradeid for tp in self.positions[idx]]
        i = 0
        if xtrade.id in entry_ids:
            i = entry_ids.index(xtrade.id)
            is_entry = True
        elif xtrade.id in exit_ids:
            i = exit_ids.index(xtrade.id)
            is_entry = False
        else:
            print("entry_id = %s, exit_id = %s" % (entry_ids, exit_ids))
            self.on_log('the trade %s is in status = %s but not found in the strat=%s tradepos table' % (xtrade.id, xtrade.status, self.name), \
                                                                            level = logging.WARNING)
            xtrade.status = TradeStatus.StratConfirm
            return
        tradepos = self.positions[idx][i]        
        traded_price = xtrade.filled_price
        if is_entry:
            if xtrade.filled_vol != 0:
                pos = tradepos.pos
                if pos != 0:
                    tradepos.update(traded_price, xtrade.filled_vol, datetime.datetime.now())
                else:
                    tradepos.open( traded_price, xtrade.filled_vol, datetime.datetime.now())
                self.on_log('strat %s successfully opened a position on %s after tradeid=%s is done, trade status is changed to confirmed' %
                            (self.name, '_'.join(tradepos.insts), xtrade.id), level = logging.INFO)
                self.num_entries[idx] += 1
            else:
                tradepos.cancel_open()
                self.on_log('strat %s cancelled an open position on %s after tradeid=%s is cancelled. Both trade and position will be removed.' %
                                (self.name, '_'.join(tradepos.insts), xtrade.id), level = logging.INFO)
            xtrade.status = TradeStatus.StratConfirm
        else:
            save_pos = tradepos.close( traded_price, datetime.datetime.now(), xtrade.filled_vol)
            if save_pos != None:
                self.save_closed_pos(save_pos)
                self.on_log('strat %s closed a position on %s after tradeid=%s (filled = %s, full = %s) is done, the closed trade position is saved' %
                            (self.name, '_'.join(tradepos.insts), xtrade.id, xtrade.filled_vol, xtrade.vol), level = logging.INFO)
            xtrade.status = TradeStatus.StratConfirm
            self.num_exits[idx] += 1
        self.positions[idx] = [ tradepos for tradepos in self.positions[idx] if not tradepos.is_closed]
        self.curr_pos[idx] = sum([tradepos.pos for tradepos in self.positions[idx]])
        self.submitted_trades[idx] = [xtrade for xtrade in self.submitted_trades[idx] if xtrade.status!= TradeStatus.StratConfirm]
        self.save_state()

    def liquidate_tradepos(self, idx):
        save_status = False
        if len(self.positions[idx]) > 0:
            for pos in self.positions[idx]:
                if (pos.entry_time > NO_ENTRY_TIME) and (pos.exit_tradeid == 0):
                    msg = 'strat=%s is liquidating underliers = %s' % ( self.name,   '_'.join(pos.insts))
                    self.status_notifier(pos.insts[0], msg)
                    self.close_tradepos(idx, pos, self.curr_prices[idx])
                    save_status = True
        return save_status

    def add_live_trades(self, xtrade):
        trade_key = xtrade.underlying.name
        idx = self.under2idx[trade_key]
        for cur_trade in self.submitted_trades[idx]:
            if xtrade.id == cur_trade.id:
                self.on_log('trade_id = %s is already in the strategy= %s list' % (xtrade.id, self.name), level = logging.DEBUG)
                return False
        self.on_log('trade_id = %s is added to the strategy= %s list' % (xtrade.id, self.name), level= logging.INFO)
        self.submitted_trades[idx].append(xtrade)
        return True

    def day_finalize(self):
        for idx in self.positions:
            for tradepos in self.positions[idx]:
                if tradepos.exit_time == NO_ENTRY_TIME:
                    tradepos.close(0, NO_ENTRY_TIME, 0)
                if tradepos.entry_time == NO_ENTRY_TIME:
                    tradepos.cancel_open()
        self.on_log('strat %s is finalizing the day - update trade unit, save state' % self.name, level = logging.INFO)
        self.update_trade_unit()
        self.num_entries = [0] * len(self.underliers)
        self.num_exits = [0] * len(self.underliers)
        self.save_state()
        srcname = self.folder + 'strat_status.csv'
        dstname = self.folder + self.agent.scur_day.strftime("%Y-%m%d") + '.csv' 
        shutil.copyfile(srcname, dstname)
        self.initialize()

    def calc_curr_price(self, idx):
        self.curr_prices[idx] = self.underlying[idx].mid_price

    def run_tick(self, ctick):
        if self.is_disabled: return
        save_status = False
        inst = ctick.instID
        idx_list = self.inst2idx[inst]
        for idx in idx_list:
            self.calc_curr_price(idx)
            if self.run_flag[idx] == 1:
                save_status = save_status or self.on_tick(idx, ctick)
            elif self.run_flag[idx] == 2:
                save_status = save_status or self.liquidate_tradepos(idx)
        if save_status:
            self.save_state()

    def run_min(self, under, freq_list):
        if self.is_disabled: return
        save_status = False
        idx = self.under2idx[under]
        if self.run_flag[idx] == 1:
            self.calc_curr_price(idx)
            save_status = save_status or self.on_bar(idx, freq_list)
        if save_status:
            self.save_state()

    def on_tick(self, idx, ctick):
        return False

    def on_bar(self, idx, freq_list):
        return False

    def open_tradepos(self, idx, direction, price, volume = 0, tag = ''):
        tunit = self.trade_unit[idx] if volume == 0 else volume
        start_time = self.agent.tick_id
        xtrade = XTrade( instIDs = self.underliers[idx], units = self.volumes[idx], \
                               vol = direction * tunit, limit_price = price, \
                               price_unit = self.price_unit[idx], start_time = start_time, \
                               strategy=self.name, book= str(idx))
        xtrade.set_agent(self.agent)
        exec_algo = eval(self.exec_class[idx])(xtrade, **self.exec_args[idx])
        xtrade.set_algo(exec_algo)
        tradepos = eval(self.pos_class)(insts = self.underliers[idx], \
                        volumes = self.volumes[idx], target_pos = direction * tunit, \
                        entry_target = price, exit_target = price, \
                        multiple = self.underlying[idx].multiple, tag = tag, **self.pos_args)
        tradepos.entry_tradeid = xtrade.id
        self.submit_trade(idx, xtrade)
        self.positions[idx].append(tradepos)        

    def update_tradepos(self, idx, price, volume, tag = ''):
        start_time = self.agent.tick_id
        trade_volume = volume
        if len(self.positions[idx]) == 0:
            tradepos = eval(self.pos_class)(insts=self.underliers[idx], \
                                        volumes=self.volumes[idx], target_pos = volume, \
                                        entry_target = price, exit_target = price, \
                                        multiple = self.underlying[idx].multiple, tag=tag, **self.pos_args)
            self.positions[idx].append(tradepos)
        else:
            tradepos = self.positions[idx][0]
            tradepos.target_pos += volume
            trade_volume = tradepos.target_pos - tradepos.pos
        xtrade = XTrade(instIDs=self.underliers[idx], units=self.volumes[idx], \
                              vol = trade_volume, limit_price = price, \
                              price_unit = self.price_unit[idx], start_time=start_time, \
                              strategy=self.name, book=str(idx))
        xtrade.set_agent(self.agent)
        exec_algo = eval(self.exec_class[idx])(xtrade, **self.exec_args[idx])
        xtrade.set_algo(exec_algo)
        tradepos.entry_tradeid = xtrade.id
        self.submit_trade(idx, xtrade)

    def submit_trade(self, idx, xtrade):
        xtrade.book = str(idx)
        self.submitted_trades[idx].append(xtrade)
        self.agent.submit_trade(xtrade)

    def close_tradepos(self, idx, tradepos, price):
        start_time = self.agent.tick_id
        xtrade = XTrade( instIDs = tradepos.insts, units = tradepos.volumes, \
                               vol = -tradepos.pos, limit_price = price, \
                               price_unit = self.price_unit[idx], start_time = start_time, \
                               strategy = self.name, book = str(idx))
        xtrade.set_agent(self.agent)
        exec_algo = eval(self.exec_class[idx])(xtrade, **self.exec_args[idx])
        xtrade.set_algo(exec_algo)
        tradepos.exit_tradeid = xtrade.id
        self.submit_trade(idx, xtrade)

    def update_trade_unit(self):
        self.trade_unit = [ int(self.pos_scaler * self.alloc_w[idx] + 0.5) for idx in range(len(self.underliers))]

    def status_notifier(self, inst, msg):
        self.on_log(msg, level = logging.INFO)
        if len(self.email_notify) > 0:
            recepient = self.email_notify[inst]
            if len(recepient) > 0:
                event = Event(type = EVENT_MAIL)
                event.dict['sender'] = self.name
                event.dict['body'] = msg
                event.dict['recepient'] = recepient
                event.dict['subject'] = inst
                self.agent.event_engine.put(event)

    def save_local_variables(self, file_writer):
        pass

    def load_local_variables(self, row):
        pass

    def save_state(self):
        filename = self.folder + 'strat_status.csv'
        self.on_log('save state for strat = %s' % self.name, level = logging.DEBUG)
        with open(filename,'w', newline='') as log_file:
            file_writer = csv.writer(log_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for key in sorted(self.positions.keys()):
                header = 'tradepos'
                for tradepos in self.positions[key]:
                    tradedict = tradepos2dict(tradepos)
                    row = [header] + [tradedict[itm] for itm in tradepos_header]
                    file_writer.writerow(row)
            self.save_local_variables(file_writer)

    def load_state(self):
        logfile = self.folder + 'strat_status.csv'
        positions  = dict([(idx, []) for idx in range(len(self.underliers))])
        if not os.path.isfile(logfile):
            self.positions  = positions
            return
        self.on_log('load state for strat = %s' % self.name, level = logging.DEBUG)
        with open(logfile, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0]  == 'tradepos':
                    if len(row) == 19:
                        shift = 0
                    else:
                        print("unsupported tradepos format")
                    insts = row[1].split(' ')
                    vols = [ int(n) for n in row[2].split(' ')]
                    target_pos = int(float(row[4]))
                    pos = int(float(row[3]))
                    # hack for old trade position loading
                    if (target_pos != pos) and (abs(target_pos) == 1):
                        target_pos = pos
                    entry_target = float(row[7])
                    exit_target = float(row[11])
                    multiple = float(row[15])
                    reset_margin = float(row[16])
                    trailing = True if float(row[17]) > 0 else False
                    tag = str(row[18])
                    tradepos = eval(self.pos_class)(insts = insts, volumes = vols, pos = pos, target_pos = target_pos, \
                                    entry_target = entry_target, exit_target = exit_target, \
                                    multiple = multiple, reset_margin = reset_margin, **self.pos_args)
                    if row[6] in ['', '19700101 00:00:00 000000']:
                        entry_time = NO_ENTRY_TIME
                        entry_price = 0
                    else:
                        entry_time = datetime.datetime.strptime(row[6], '%Y%m%d %H:%M:%S %f')
                        entry_price = float(row[5])
                        tradepos.open(entry_price, pos, entry_time)
                    tradepos.entry_tradeid = int(row[8])
                    tradepos.trailing = trailing
                    tradepos.exit_tradeid = int(row[12])
                    tradepos.tag = tag
                    if row[10] in ['', '19700101 00:00:00 000000']:
                        exit_time = NO_ENTRY_TIME
                        exit_price = 0
                    else:
                        exit_time = datetime.datetime.strptime(row[10], '%Y%m%d %H:%M:%S %f')
                        exit_price = float(row[9])
                        tradepos.close(exit_price, exit_time)
                    is_added = False
                    for i, under in enumerate(self.underliers):
                        if (set(under) == set(insts)) or (set(self.prev_underliers[i]) == set(insts)):
                            idx = i
                            is_added = True
                            break
                    if not is_added:
                        self.on_log('underlying = %s is missing in strategy=%s, need to check' % (insts, self.name), level = logging.WARNING)
                    positions[idx].append(tradepos)
                else:
                    self.load_local_variables(row)
        self.positions = positions
        for key in sorted(self.positions.keys()):
            self.curr_pos[key] = sum([tp.pos for tp in self.positions[key]])

    def save_closed_pos(self, tradepos):
        logfile = self.folder + 'hist_tradepos.csv'
        with open(logfile,'a') as log_file:
            file_writer = csv.writer(log_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            tradedict = tradepos2dict(tradepos)
            file_writer.writerow([tradedict[itm] for itm in tradepos_header])

    def risk_agg(self, risk_list):
        sum_risk = {}
        inst_risk = {}
        for inst in self.instIDs:
            inst_risk[inst] = dict([(risk, 0) for risk in risk_list])
            sum_risk[inst] = dict([(risk, 0) for risk in risk_list])
            for risk in risk_list:
                try:
                    prisk = risk[1:]
                    inst_risk[inst][risk] = getattr(self.agent.instruments[inst], prisk)
                except:
                    continue
        for idx, under in enumerate(self.underliers):
            for tp in self.positions[idx]:
                for instID, v in zip(tp.insts, tp.volumes):
                    for risk in risk_list:
                        sum_risk[instID][risk] += tp.pos * v * inst_risk[instID][risk]
        return sum_risk
