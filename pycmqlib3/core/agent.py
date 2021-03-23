#-*- coding:utf-8 -*-
import workdays
import json
import datetime
import logging
import bisect
import pandas as pd
import os
from pycmqlib3.utility import dbaccess
from pycmqlib3.analytics import data_handler
from pycmqlib3.utility.misc import get_obj_by_name, CHN_Holidays, cleanup_mindata, day_split_dict
from . order import Order
from . instrument import Future, FutOptionInst, Stock, StockOptionInst, SpreadInst
from . event_engine import Event, PriEventEngine
from . trade_manager import TradeManager
from . bar_manager import BarManager, DaySplitBarManager
from . trading_object import OrderStatus, SubscribeRequest
from . trading_const import TradeStatus
from . agent_conf import BAR_BUF_SHIFT_SIZE, TICK_BUF_SHIFT_SIZE, tick_data_list, \
    min_data_list, day_data_list, dtype_map
from . event_type import EVENT_LOG, EVENT_TICK, EVENT_ETRADEUPDATE, EVENT_DAYSWITCH, EVENT_TIMER, \
    EVENT_DB_WRITE, EVENT_MKTDATA_EOD, EVENT_MIN_BAR

class MktDataMixin(object):
    def __init__(self, config):
        self.tick_data  = {}
        self.day_data  = {}
        self.cur_min = {}
        self.cur_day = {}
        self.bar_factory = {}
        self.day_data_func = {}
        self.daily_data_days = config.get('daily_data_days', 25)
        self.min_data_days = config.get('min_data_days', 1)
        self.tick_data_days = config.get('tick_data_days', 0)
        self.save_tick_idx = {}
        self.db_conn = None
        if 'min_func' in config:
            self.get_min_id = eval(config['min_func'])
        else:
            self.get_min_id = lambda x: int(x/1000)
        if 'bar_func' in config:
            self.conv_bar_id = eval(config['bar_func'])
        else:
            self.conv_bar_id = data_handler.bar_conv_func2            
        self.tick_db_table = config.get('tick_db_table', 'fut_tick')
        self.min_db_table  = config.get('min_db_table', 'fut_min')
        self.daily_db_table = config.get('daily_db_table', 'fut_daily')
        self.calc_func_dict = {}

    def add_underlying(self, name):
        dtypes = [(field, dtype_map[field]) for field in tick_data_list]
        self.tick_data[name] = data_handler.DynamicRecArray(dtype = dtypes, nlen = 2000)
        self.save_tick_idx[name] = 0
        dtypes = [(field, dtype_map[field]) for field in day_data_list]
        self.day_data[name]  = data_handler.DynamicRecArray(dtype = dtypes)
        self.day_data_func[name] = []
        dtypes = [(field, dtype_map[field]) for field in min_data_list]
        self.bar_factory[name] = {'m1': BarManager(freq = 'm1', data_args={'dtype': dtypes, 'nlen': 1000})}
        self.cur_day[name]   = dict([(item, None) for item in day_data_list])
        self.cur_min[name]   = dict([(item, None) for item in min_data_list])
        self.cur_min[name]['datetime'] = datetime.datetime.fromordinal(self.scur_day.toordinal())
        self.cur_min[name]['volume'] = 0
        self.cur_min[name]['tick_min'] = 0
        self.cur_min[name]['date'] = self.scur_day
        self.cur_day[name]['date'] = self.scur_day
        self.cur_day[name]['volume'] = 0

    def register_data_func(self, under, freq, fobj):
        if under not in self.day_data_func:
            self.day_data_func[under] = []
        if (fobj != None) and (fobj.name not in self.calc_func_dict):
            self.calc_func_dict[fobj.name] = fobj
        if freq == 'd':
            for func in self.day_data_func[under]:
                if fobj.name == func.name:
                    return False
            self.day_data_func[under].append(self.calc_func_dict[fobj.name])
            return True
        elif ('m' in freq) or ('s' in freq):
            if freq not in self.bar_factory[under]:
                if 's' in freq:
                    self.bar_factory[under][freq] = DaySplitBarManager(freq, bar_func = self.conv_bar_id)
                else:
                    self.bar_factory[under][freq] = BarManager(freq)
            if fobj:
                if self.bar_factory[under][freq].add_func(self.calc_func_dict[fobj.name]):
                    return True
            return False
        else:
            self.logger.info("Unknown freq = %s for inst = %s" % (freq, under))
            return False

    def mkt_data_sod(self, tday):
        for instID in self.instruments:
            self.instruments[instID].reset_mkt_data()
        for under in self.bar_factory:
            self.cur_min[under] = dict([(item, None) for item in min_data_list])
            self.cur_day[under] = dict([(item, None) for item in day_data_list])
            self.cur_day[under]['date'] = tday
            self.cur_day[under]['volume'] = 0
            self.cur_min[under]['datetime'] = datetime.datetime.fromordinal(tday.toordinal())
            self.cur_min[under]['volume'] = 0
            self.cur_min[under]['tick_min'] = 0
            if under not in self.instruments:
                self.spread_data[under].reset_mkt_data()

    def mkt_data_eod(self):
        for under in self.bar_factory:
            if len(self.tick_data[under]) > 0:
                tick_data = self.tick_data[under].data
                self.cur_min[under]['volume'] = int(tick_data['volume'][-1] - self.cur_min[under]['volume'])
                self.cur_min[under]['openInterest'] = int(tick_data['openInterest'][-1])
                self.min_switch(under, True)
            for freq in self.bar_factory[under]:
                self.bar_factory[under][freq].bar_array.shift_lastn(BAR_BUF_SHIFT_SIZE)
            self.save_tick_idx[under] = max(0, self.save_tick_idx[under] - self.tick_data[under].shift_lastn(TICK_BUF_SHIFT_SIZE))
            new_day_data = { key: self.cur_day[under][key] for key in day_data_list }
            self.day_data[under].append_by_dict(new_day_data)
            for fobj in self.day_data_func[under]:
                fobj.rfunc(self.day_data[under].data)
            if (under in self.instruments) and self.live_save:
                if self.cur_day[under]['open'] != None:
                    event = Event(type=EVENT_DB_WRITE, priority = 500)
                    event.dict['data'] = self.cur_day[under]
                    event.dict['type'] = EVENT_MKTDATA_EOD
                    event.dict['instID'] = under
                    self.event_engine.put(event)


class Agent(MktDataMixin):
    def __init__(self, config = {}, tday = datetime.date.today()):
        self.tick_id = 0
        self.timer_count = 0
        self.name = config.get('name', 'test_agent')
        self.sched_commands = []
        self.folder = str(config.get('folder', self.name + os.path.sep))
        self.live_save = config.get('live_save', False)
        self.logger = logging.getLogger()
        self.eod_flag = False
        self.scur_day = tday
        super(Agent, self).__init__(config)
        self.event_period = config.get('event_period', 1.0)
        self.event_engine = PriEventEngine(self.event_period)
        self.instruments = {}
        self.positions = {}
        self.gateways = {}
        gateway_dict = config.get('gateway', {})
        for gateway_name in gateway_dict:
            gateway_class = get_obj_by_name(gateway_dict[gateway_name]['class'])
            gateway_args = gateway_dict[gateway_name].get('args', {})
            self.add_gateway(gateway_class, gateway_name = gateway_name, gateway_args = gateway_args)
        self.gateway_map = config.get('gateway_map', {})
        self.inst2gateway = {}
        self.inst2strat = {}
        self.spread_data = {}
        self.inst2spread = {}
        self.strat_list = []
        self.strategies = {}
        self.trade_manager = TradeManager(self)
        self.ref2order = {}
        strat_files = config.get('strat_files', [])
        for sfile in strat_files:
            with open(sfile, 'r') as fp:
                strat_conf = json.load(fp)
            strat_args  = strat_conf.get('config', {})
            strat = get_obj_by_name(strat_conf['class'])(strat_args, self)
            self.add_strategy(strat)
        self.init_init()    #init in init, to be used by child class

    def register_event_handler(self):
        for key in self.gateways:
            gateway = self.gateways[key]
            gateway.register_event_handler()
        self.event_engine.register(EVENT_LOG, self.log_handler)
        self.event_engine.register(EVENT_TICK, self.run_tick)
        self.event_engine.register(EVENT_ETRADEUPDATE, self.trade_update)
        self.event_engine.register(EVENT_DAYSWITCH, self.day_switch)
        self.event_engine.register(EVENT_TIMER, self.check_commands)

    def put_command(self, timestamp, command, arg = {} ): # insert command by timestamp
        stamps = [tstamp for (tstamp,cmd, fargs) in self.sched_commands]
        ii = bisect.bisect(stamps, timestamp)
        self.sched_commands.insert(ii,(timestamp, command, arg))

    def check_commands(self, event):
        l = len(self.sched_commands)
        curr_time = datetime.datetime.now()
        i = 0
        while(i<l and curr_time >= self.sched_commands[i][0]):
            logging.info('exec command:,i=%s,time=%s,command[i][1]=%s' % (i, curr_time, self.sched_commands[i][1].__name__))
            arg = self.sched_commands[i][2]
            self.sched_commands[i][1](**arg)
            i += 1
        if i>0:
            del self.sched_commands[0:i]

    def map_gateway(self, instID):
        exch = self.instruments[instID].exchange
        key = None
        if (instID in self.gateway_map):
            key = instID
        elif exch in self.gateway_map:
            key = exch
        if key:
            gateway = self.gateways[self.gateway_map[key]]
            return gateway
        else:
            for key in self.gateways:
                gateway = self.gateways[key]
                gway_class = type(gateway).__name__
                if ('ctp' in gway_class) or ('Ctp' in gway_class):
                    return gateway
            return None

    def add_instrument(self, name):
        if name not in self.instruments:
            if name.isdigit():
                if len(name) == 8:
                    self.instruments[name] = StockOptionInst(name)
                else:
                    self.instruments[name] = Stock(name)
            else:
                if len(name) > 8:
                    self.instruments[name] = FutOptionInst(name)
                else:
                    self.instruments[name] = Future(name)
            self.instruments[name].update_param(self.scur_day)
            if name not in self.inst2strat:
                self.inst2strat[name] = {}
            if name not in self.inst2gateway:
                gateway = self.map_gateway(name)
                if gateway != None:
                    self.inst2gateway[name] = gateway
                    subreq = SubscribeRequest(name, self.instruments[name].exchange)
                    #subreq.symbol = name
                    #subreq.exchange = self.instruments[name].exchange
                    #subreq.productClass = self.instruments[name].ptype
                    #subreq.currency = self.instruments[name].ccy
                    #subreq.expiry = self.instruments[name].expiry
                    #subreq.strikePrice = self.instruments[name].strike if hasattr(self.instruments[name], 'strike') else 0
                    #subreq.optionType = self.instruments[name].otype if hasattr(self.instruments[name], 'otype') else ''
                    #subreq.contMonth = self.instruments[name].cont_mth if hasattr(self.instruments[name], 'cont_mth') else 205012
                    #subreq.multiple = self.instruments[name].multiple
                    gateway.subscribe(subreq)
                else:
                    self.logger.warning("No Gateway is assigned to instID = %s" % name)
            self.add_underlying(name)

    def add_spread(self, instIDs, weights, multiple = None):        
        key = '_'.join([str(s) for s in instIDs + weights])
        self.spread_data[key] = SpreadInst(self.instruments, instIDs, weights, multiple)
        for inst in instIDs:
            if inst not in self.inst2spread:
                self.inst2spread[inst] = []
            self.inst2spread[inst].append(key)
        self.inst2strat[key] = {}
        self.add_underlying(key)
        return self.spread_data[key]
                
    def get_underlying(self, instIDs, weights, multiple = None):
        if len(instIDs) == 1:
            key = instIDs[0]
            return self.instruments[key]
        else:
            key = '_'.join([str(s) for s in instIDs + weights])           
            if key not in self.spread_data:
                self.add_spread(instIDs, weights, multiple)
            return self.spread_data[key]

    def add_strategy(self, strat):
        if strat.name not in self.strat_list:
            self.strat_list.append(strat.name)
            self.strategies[strat.name] = strat
            for instID in strat.dep_instIDs():
                self.add_instrument(instID)
            strat.set_agent(self)

    def add_gateway(self, gateway, gateway_name=None, gateway_args = {}):
        """create a gateway acccess"""
        if gateway_name not in self.gateways:
            self.gateways[gateway_name] = gateway(self, gateway_name, **gateway_args)

    def connect(self, gateway_name):
        """connect to gateway by name"""
        if gateway_name in self.gateways:
            gateway = self.gateways[gateway_name]
            gateway.connect()
        else:
            self.logger.warning('Gateway does not exist：%s' % gateway_name)
        
    
    def subscribe(self, subscribeReq, gateway_name):
        """subscribe symbol in gateway"""
        if gateway_name in self.gateways:
            gateway = self.gateways[gateway_name]
            gateway.subscribe(subscribeReq)
        else:
            self.logger.warning('Gateway does not exist：%s' %gateway_name)
        
    
    def send_order(self, iorder, urgent = 1):
        """send order to gateway"""
        gateway = self.inst2gateway[iorder.instrument]
        gateway.add_order(iorder)
        if urgent:
            gateway.sendOrder(iorder)

    
    def cancel_order(self, iorder):
        """cancel order from gateway"""
        if iorder.status in [OrderStatus.Ready]:
            iorder.on_cancel()
        elif iorder.status  == OrderStatus.Sent:
            if iorder.gateway != None:
                iorder.gateway.cancel_order(iorder)
            else:
                self.logger.warning('no gateway is associated with the order_ref = %s' % iorder.order_ref)
        elif not iorder.is_closed():
            self.logger.warning('order_ref = %s is in status = %s, but is not closed' % (iorder.order_ref, iorder.status))


    def submit_trade(self, xtrade):
        self.trade_manager.add_trade(xtrade)

    def remove_trade(self, xtrade):
        self.trade_manager.remove_trade(xtrade)

    def log_handler(self, event):
        lvl = event.dict['level']
        self.logger.log(lvl, event.dict['data'])

    def get_eod_positions(self):
        for name in self.gateways:
            self.gateways[name].load_local_positions(self.scur_day)

    def get_all_orders(self):
        self.ref2order = {}
        if self.eod_flag:
            return
        for name in self.gateways:
            gway = self.gateways[name]
            gway.load_order_list(self.scur_day)
            order_dict = gway.id2order
            for local_id in order_dict:
                iorder = order_dict[local_id]
                iorder.gateway = gway
                self.ref2order[iorder.order_ref] = iorder

    def risk_by_strats(self, risk_list = ['ppos']):
        # position = lots, delta, gamma, vega, theta in price
        risk_dict = {}
        sum_risk = dict([(inst, dict([(risk, 0) for risk in risk_list])) for inst in self.instruments])
        for strat_name in self.strat_list:
            strat = self.strategies[strat_name]
            risk_dict[strat_name] = strat.risk_agg(risk_list)
            for inst in risk_dict[strat_name]:
                for risk in risk_list:
                    sum_risk[inst][risk] += risk_dict[strat_name][inst][risk]
        return sum_risk, risk_dict

    def prepare_data_env(self, mid_day = True):
        daily_store = {}
        min_store = {}
        self.db_conn = dbaccess.connect(**dbaccess.dbconfig)
        data_insts = [inst for inst in self.instruments if self.instruments[inst].update_bar]
        for inst in data_insts:
            if self.daily_data_days > 0 or mid_day:
                d_start = workdays.workday(self.scur_day, -self.daily_data_days, CHN_Holidays)
                d_end = self.scur_day
                ddf = dbaccess.load_daily_data_to_df(self.db_conn, 'fut_daily', inst, d_start, d_end, index_col = None)
                daily_store[inst] = ddf
                if len(ddf) > 0:
                    self.instruments[inst].price = self.instruments[inst].mid_price = float(ddf['close'].iloc[-1])
                    self.instruments[inst].bid_price1 = self.instruments[inst].ask_price1 = self.instruments[inst].price
                    self.instruments[inst].last_update = 0
                    self.instruments[inst].prev_close = float(ddf['close'].iloc[-1])
            if self.min_data_days > 0 or mid_day:
                min_start = int(self.instruments[inst].start_tick_id / 1000)
                min_end = int(self.instruments[inst].last_tick_id / 1000) + 1
                d_start = workdays.workday(self.scur_day, -self.min_data_days, CHN_Holidays)
                d_end = self.scur_day
                mdf = dbaccess.load_min_data_to_df(self.db_conn, 'fut_min', inst, d_start, d_end, minid_start=min_start, minid_end=min_end, index_col = None)
                mdf = cleanup_mindata(mdf, self.instruments[inst].product, index_col = None)
                mdf['bar_id'] = self.conv_bar_id(mdf['min_id'])
                min_store[inst] = mdf
                if len(mdf)>0:
                    min_date = mdf['date'].iloc[-1]
                    if (len(self.day_data[inst])==0) or (min_date > self.day_data[inst].data['date'][-1]):
                        ddf = data_handler.conv_ohlc_freq(mdf, 'd', index_col = None)
                        self.cur_day[inst]['open'] = float(ddf.open[-1])
                        self.cur_day[inst]['close'] = float(ddf.close[-1])
                        self.cur_day[inst]['high'] = float(ddf.high[-1])
                        self.cur_day[inst]['low'] = float(ddf.low[-1])
                        self.cur_day[inst]['volume'] = int(ddf.volume[-1])
                        self.cur_day[inst]['openInterest'] = int(ddf.openInterest[-1])
                        self.cur_min[inst]['datetime'] = mdf['datetime'].iloc[-1].to_pydatetime()
                        self.cur_min[inst]['date'] = mdf['date'].iloc[-1]
                        self.cur_min[inst]['open'] = float(mdf['open'].iloc[-1])
                        self.cur_min[inst]['close'] = float(mdf['close'].iloc[-1])
                        self.cur_min[inst]['high'] = float(mdf['high'].iloc[-1])
                        self.cur_min[inst]['low'] = float(mdf['low'].iloc[-1])
                        self.cur_min[inst]['volume'] = int(self.cur_day[inst]['volume'])
                        self.cur_min[inst]['openInterest'] = int(self.cur_day[inst]['openInterest'])
                        self.cur_min[inst]['min_id'] = int(mdf['min_id'].iloc[-1])
                        self.cur_min[inst]['bar_id'] = self.conv_bar_id(self.cur_min[inst]['min_id'])
                        self.instruments[inst].price = self.instruments[inst].mid_price = float(mdf['close'].iloc[-1])
                        self.instruments[inst].bid_price1 = self.instruments[inst].ask_price1 = self.instruments[inst].price
                        self.instruments[inst].last_update = 0
            if self.tick_data_days > 0:
                d_start = workdays.workday(self.scur_day, -(self.tick_data_days - 1), CHN_Holidays)
                d_end = self.scur_day
                tdf = dbaccess.load_tick_data_to_df(self.db_conn, 'fut_tick', inst, d_start, d_end, index_col=None)
                self.tick_data[inst] = data_handler.DynamicRecArray(dataframe = tdf)
        self.db_conn.close()
        func_keys = ['open', 'high', 'low', 'close', 'volume', 'openInterest']
        inst_keys = data_insts + list(self.spread_data.keys())
        for idx, data_dict in enumerate([daily_store, min_store]):
            for key in inst_keys:
                if key in self.spread_data:
                    self.spread_data[key].update()
                    if not self.spread_data[key].update_bar:
                        continue
                    if idx == 0:
                        on_key = ['date']
                        index_keys = ['date']
                    else:
                        on_key = ['date', 'min_id']
                        index_keys = ['datetime', 'date', 'min_id', 'bar_id']
                    for idy, (inst, vol, cf) in enumerate(zip(self.spread_data[key].instIDs, \
                                                              self.spread_data[key].weights, \
                                                              self.spread_data[key].conv_factor)):
                        curr_w = vol * cf / self.spread_data[key].multiple
                        if idy == 0:
                            xdf = data_dict[inst].copy()
                            xdf[['open', 'close']] = xdf[['open', 'close']] * curr_w
                            prev_suffix = '_last'
                        elif idy == 1:
                            xdf = pd.merge(xdf, data_dict[inst][on_key + ['open', 'high', 'low', 'close']], on=on_key, how='inner', sort=False,
                                           suffixes=[prev_suffix, '_' + inst])
                            xdf['open_last'] = xdf['open_last'] + xdf['open_' + inst] * curr_w
                            xdf['close_last'] = xdf['close_last'] + xdf['close_' + inst] * curr_w
                            prev_suffix = ''
                    xdf.rename(columns = {'open_last': 'open', 'close_last': 'close'}, inplace = True)
                    xdf['high'] = xdf[['open', 'close']].max(axis=1)
                    xdf['low'] = xdf[['open', 'close']].min(axis=1)
                    xdf['volume'] = 0
                    xdf['openInterest'] = 0
                    key_list = index_keys + func_keys
                    xdf = xdf[key_list]
                    self.spread_data[key].price = self.spread_data[key].mid_price = float(xdf['close'].iloc[-1])
                    self.spread_data[key].bid_price1 = self.spread_data[key].ask_price1 = self.spread_data[key].price
                    self.spread_data[key].last_update = 0
                    self.spread_data[key].prev_close = float(xdf['close'].iloc[-1])
                else:
                    xdf = data_dict[key]
                if len(xdf) == 0:
                    continue
                if idx == 1:
                    for freq in self.bar_factory[key]:
                        if freq == 'm1':
                            mdf_m = xdf
                        elif 'm' in freq:
                            m = int(freq[1:])
                            mdf_m = data_handler.conv_ohlc_freq(xdf, str(m) + 'min', index_col=None,
                                                                bar_func=self.conv_bar_id, extra_cols=['bar_id'])
                        elif 's' in freq:
                            min_split = day_split_dict[freq]
                            mdf_m = data_handler.day_split(xdf, minlist=min_split, index_col=None, extra_cols=['bar_id'])
                        else:
                            self.logger.info("unsupported freq, %s, %s, use default m1" % (freq, key))
                            mdf_m = xdf
                        self.bar_factory[key][freq].run_init_func(mdf_m)
                        self.bar_factory[key][freq].set_bar_array(data_args={'dataframe': mdf_m})
                elif idx == 0:
                    for fobj in self.day_data_func[key]:
                        ts = fobj.sfunc(xdf)
                        if type(ts).__name__ == 'Series':
                            if ts.name in xdf.columns:
                                self.logger.info(
                                    'TimeSeries name %s is already in the columns for inst = %s' % (ts.name, key))
                            else:
                                xdf[ts.name] = ts
                        elif type(ts).__name__ == 'DataFrame':
                            for col_name in ts.columns:
                                if col_name in xdf.columns:
                                    self.logger.info(
                                        'TimeSeries name %s is already in the columns for inst = %s' % (col_name, key))
                                else:
                                    xdf[col_name] = ts[col_name]
                    self.day_data[key] = data_handler.DynamicRecArray(dataframe = xdf)

    def restart(self):
        self.logger.debug('Prepare trade environment for %s' % self.scur_day.strftime('%y%m%d'))
        self.prepare_data_env(mid_day = True)
        self.get_eod_positions()
        self.get_all_orders()
        self.trade_manager.initialize()
        for strat_name in self.strat_list:
            strat = self.strategies[strat_name]
            strat.initialize()
            strat_trades = self.trade_manager.get_trades_by_strat(strat.name)
            for xtrade in strat_trades:
                if xtrade.status != TradeStatus.StratConfirm:
                    strat.add_live_trades(xtrade)
        for gway in self.gateways:
            gateway = self.gateways[gway]
            if not self.eod_flag:
                for inst in gateway.positions:
                    gateway.positions[inst].re_calc()
            gateway.calc_margin()
            gateway.connect()
        self.event_engine.start()

    def save_state(self):
        if not self.eod_flag:
            self.logger.debug('save agent state ...')
            for gway in self.gateways:
                self.gateways[gway].save_order_list(self.scur_day)
            self.trade_manager.save_trade_list(self.scur_day, self.trade_manager.ref2trade, self.folder)
    
    def run_eod(self):
        if self.eod_flag:
            return
        self.logger.info('run EOD process')
        self.mkt_data_eod()
        if len(self.strat_list) == 0:
            self.eod_flag = True
            return
        self.trade_manager.day_finalize(self.scur_day, self.folder)
        for strat_name in self.strat_list:
            strat = self.strategies[strat_name]
            strat.day_finalize()
        for name in self.gateways:
            self.gateways[name].day_finalize(self.scur_day)
        self.eod_flag = True
        self.ref2order = {}
        for inst in self.instruments:
            self.instruments[inst].prev_close = self.cur_day[inst]['close']

    def day_switch(self, event):
        newday = event.dict['date']
        if newday <= self.scur_day:
            return
        self.logger.info('switching the trading day from %s to %s, reset tick_id=%s to 0' % (self.scur_day, newday, self.tick_id))
        if not self.eod_flag:
            self.run_eod()
        self.scur_day = newday
        self.tick_id = 0
        self.timer_count = 0
        super(Agent, self).mkt_data_sod(newday)
        self.eod_flag = False
        eod_time = datetime.datetime.combine(newday, datetime.time(15, 20, 0))
        self.put_command(eod_time, self.run_eod)
                
    def init_init(self): 
        self.register_event_handler()

    def min_switch(self, under_key, forced = False):
        ra = self.bar_factory[under_key]['m1'].bar_array
        bar_id = self.conv_bar_id(self.cur_min[under_key]['tick_min'])
        switched_freq = []
        if self.cur_min[under_key]['open'] == None:
            return switched_freq
        if (self.cur_min[under_key]['min_id'] > 0):
            ra.append_by_dict(self.cur_min[under_key])
        switched_freq.append('m1')
        for freq in self.bar_factory[under_key]:
            bhandler = self.bar_factory[under_key][freq]
            if freq != 'm1':
                if bhandler.update(ra.data, self.cur_day[under_key]['date'], bar_id, forced):
                    switched_freq.append(freq)
            bhandler.run_update_func()
        self.save_tick_idx[under_key] = max(0, self.save_tick_idx[under_key] - self.tick_data[under_key].shift_lastn(TICK_BUF_SHIFT_SIZE))
        if self.live_save and (under_key in self.instruments):
            event1 = Event(type=EVENT_DB_WRITE, priority = 500)
            nlen = len(self.tick_data[under_key].data)
            if nlen > self.save_tick_idx[under_key]:
                event1.dict['data'] = self.tick_data[under_key].data[self.save_tick_idx[under_key]:nlen]
                event1.dict['type'] = EVENT_TICK
                event1.dict['instID'] = under_key
                self.event_engine.put(event1)
                self.save_tick_idx[under_key] = nlen
            if (self.cur_min[under_key]['volume'] > 0):
                event2 = Event(type=EVENT_DB_WRITE, priority = 500)
                event2.dict['data'] = self.cur_min[under_key]
                event2.dict['type'] = EVENT_MIN_BAR
                event2.dict['instID'] = under_key
                self.event_engine.put(event2)
        return switched_freq

    def update_underlying_bar(self, underlying, tick):
        key = underlying.name
        tick_min = self.get_min_id(underlying.last_update)
        self.cur_min[key]['tick_min'] = tick_min
        if (self.cur_day[key]['open'] == None):
            self.cur_day[key]['open'] = self.cur_day[key]['high'] = self.cur_day[key]['low'] = underlying.price
        self.cur_day[key]['close'] = underlying.price
        self.cur_day[key]['openInterest'] = underlying.openInterest
        self.cur_day[key]['volume'] = underlying.volume
        self.cur_day[key]['close'] = underlying.price
        self.cur_day[key]['high'] = max(self.cur_day[key]['high'], underlying.price)
        self.cur_day[key]['low'] = min(self.cur_day[key]['low'], underlying.price)
        self.cur_day[key]['date'] = self.scur_day
        if (tick_min == self.cur_min[key]['min_id']):
            self.tick_data[key].append_by_obj(tick)
            self.cur_min[key]['close'] = underlying.price
            self.cur_min[key]['high'] = max(self.cur_min[key]['high'], underlying.price)
            self.cur_min[key]['low'] = min(self.cur_min[key]['low'], underlying.price)
        else:
            last_vol = self.cur_min[key]['volume']
            if (key in self.instruments) and (len(self.tick_data[key]) > self.save_tick_idx[key]):
                self.cur_min[key]['volume'] = int(self.tick_data[key].data['volume'][-1] - self.cur_min[key]['volume'])
                self.cur_min[key]['openInterest'] = int(self.tick_data[key].data['openInterest'][-1])
                last_vol = int(self.tick_data[key].data['volume'][-1])
            else:
                self.cur_min[key]['volume'] = 0
            freq_list = self.min_switch(key, False)
            self.run_min(key, freq_list)
            self.cur_min[key] = {}
            self.cur_min[key]['open'] = self.cur_min[key]['close'] = self.cur_min[key]['high'] = self.cur_min[key]['low'] = float(underlying.price)
            self.cur_min[key]['min_id'] = self.cur_min[key]['tick_min'] = tick_min
            self.cur_min[key]['bar_id'] = self.conv_bar_id(tick_min)
            self.cur_min[key]['volume'] = last_vol
            self.cur_min[key]['openInterest'] = underlying.openInterest
            self.cur_min[key]['datetime'] = tick.timestamp.replace(second=0, microsecond=0)
            self.cur_min[key]['date'] = self.scur_day
            if tick_min > 0:
                self.tick_data[key].append_by_obj(tick)

    def run_tick(self, event):# the main loop for event engine
        tick = event.dict['data']
        inst = tick.instID
        curr_tick = tick.tick_id
        self.tick_id = max(curr_tick, self.tick_id)
        if not self.instruments[inst].update(tick):
            return
        self.update_underlying_bar(self.instruments[inst], tick)
        if inst in self.inst2spread:
            for key in self.inst2spread[inst]:
                if self.spread_data[key].update(tick):
                    self.update_underlying_bar(self.spread_data[key], tick)
                    self.trade_manager.check_pending_trades(key)
        for strat_name in self.inst2strat[inst]:
            self.strategies[strat_name].run_tick(tick)
        self.trade_manager.check_pending_trades(inst)
        if inst in self.inst2spread:
            for key in self.inst2spread[inst]:
                self.trade_manager.execute_trades(key)
        self.trade_manager.execute_trades(inst)
        gway = self.inst2gateway[inst]
        if gway.process_flag:
            gway.send_queued_orders()

    def run_min(self, under, freq_list):
        for strat_name in self.inst2strat[under]:
            if self.inst2strat[under][strat_name] in freq_list:
                self.strategies[strat_name].run_min(under, freq_list)

    def trade_update(self, event):
        trade_ref = event.dict['trade_ref']
        mytrade = self.trade_manager.get_trade(trade_ref)
        if mytrade == None:
            self.logger.warning("get trade update for trade_id = %s, but it is not in the trade list" % trade_ref)
            return
        mytrade.refresh()
        mytrade.execute()
        self.save_state()

    def run_gway_service(self, gway, service, args):
        if gway in self.gateways:
            gateway = self.gateways[gway]
            svc_func = service
            if hasattr(gateway, svc_func):
                ts = datetime.datetime.now()
                self.put_command(ts, getattr(gateway, svc_func), args)
            else:
                self.logger.info("no such service = %s for %s" % (service, gway))
        else:
            self.logger.info("no such a gateway %s" % gway)

    def exit(self):
        """exit the agent"""
        # stop the event engine and save the states
        self.event_engine.stop()
        self.logger.info('stopped the engine, exiting the agent ...')
        self.save_state()
        for strat_name in self.strat_list:
            strat = self.strategies[strat_name]
            strat.save_state()
        for name in self.gateways:
            gateway = self.gateways[name]
            gateway.close()
            gateway.mdApi = None
            gateway.tdApi = None

if __name__=="__main__":
    pass
