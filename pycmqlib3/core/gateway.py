# -*- coding: utf-8 -*-
import pandas as pd
import time
import datetime
import os
import csv
import workdays
import logging
from pycmqlib3.utility.misc import inst2exch, CHN_Holidays, spreadinst2underlying, day_shift
from . position import Position, GrossPosition, SHFEPosition
from . order import Order, SpreadOrder
from . event_type import *
from . event_engine import Event
from . trading_const import Alive_Order_Status, Direction, Offset, \
    OrderStatus, OrderType, Option_ProductTypes


class Gateway(object):
    def __init__(self, agent, gateway_name = 'Gateway'):
        """Constructor"""
        self.gateway_name = gateway_name
        self.agent = agent
        self.event_engine = agent.event_engine
        self.file_prefix = agent.folder + gateway_name + '_'
        self.qry_acct_data = {}
        self.qry_pos = {}
        self.id2order = {}
        self.positions = {}
        self.working_order_stats = {}
        self.instruments = []      # subscribed instruments
        self.working_orders = []
        self.process_flag = False
        self.eod_report = True
        self.account_info = {'available': 0,
                            'locked_margin': 0,
                            'used_margin': 0,
                            'curr_capital': 1000000,
                            'prev_capital': 1000000,
                            'pnl_total': 0,
                            'yday_pnl': 0,
                            'tday_pnl': 0,
                            }
        self.pl_by_product = {}
        self.order_stats = {'total_submit': 0, 'total_failure': 0, 'total_cancel':0 }
        self.order_constraints = {	'total_submit': 2000, 'total_cancel': 2000, 'total_failure':500, \
                                    'submit_limit': 200,  'cancel_limit': 200,  'failure_limit': 200 }

    def initialize(self):
        self.id2order  = {}
        self.working_orders = []
        self.order_stats = {'total_submit': 0, 'total_failure': 0, 'total_cancel':0 }
        self.account_info['prev_capital'] = self.account_info['curr_capital']
        self.check_connection()
    
    def check_connection(self):
        pass

    def process_eod_report(self, tday):
        df = pd.DataFrame.from_dict( self.pl_by_product, orient='index' )
        cols = ['yday_mark', 'tday_mark', 'yday_pos', 'yday_pnl', 'new_long', 'new_long_avg', 'new_short', 'new_short_avg', 'tday_pnl', 'used_margin', 'locked_margin']
        df = df[cols]
        logfile = self.file_prefix + 'PNL_attribution_' + tday.strftime('%y%m%d')+'.csv'
        if os.path.isfile(logfile):
            return False
        else:
            df.to_csv(logfile, sep=',')
            return True

    def day_finalize(self, tday):
        self.save_local_positions(tday)
        if self.eod_report:
            self.process_eod_report(tday)
        eod_pos = {}
        for inst in self.positions:
            pos = self.positions[inst]
            eod_pos[inst] = pos.curr_pos

        self.positions = {}
        for inst in self.instruments:
            self.order_stats[inst] = {'submit': 0, 'cancel':0, 'failure': 0, 'status': True }
            (pos_cls, pos_args) = self.get_pos_class(self.agent.instruments[inst])
            self.positions[inst] = pos_cls(self.agent.instruments[inst], self, **pos_args)
            self.positions[inst].pos_yday = eod_pos[inst]
            self.positions[inst].re_calc()

        if day_shift(self.agent.scur_day, '1b') in CHN_Holidays:
            next_run = datetime.datetime.combine(day_shift(self.agent.scur_day, '1b', CHN_Holidays), \
                                                    datetime.time(8, 0, 0))
        else:
            next_run = datetime.datetime.combine(self.agent.scur_day, datetime.time(20, 0, 0))
        self.agent.put_command(next_run, self.initialize)

    def get_pos_class(self, inst):
        return (Position, {})

    def add_instrument(self, instID):
        if instID not in self.instruments:
            self.instruments.append(instID)
        if instID not in self.positions:
            (pos_cls, pos_args) = self.get_pos_class(self.agent.instruments[instID])
            self.positions[instID] = pos_cls(self.agent.instruments[instID], self, **pos_args)
        if instID not in self.order_stats:
            self.order_stats[instID] = {'submit': 0, 'cancel':0, 'failure': 0, 'status': True }
        if instID not in self.qry_pos:
            self.qry_pos[instID]   = {'tday': [0, 0], 'yday': [0, 0]}

    def on_tick(self, tick):
        """market data tick """
        # generic tick event
        event1 = Event(type=EVENT_TICK)
        event1.dict['data'] = tick
        self.event_engine.put(event1)
        # specific contract tick
        event2 = Event(type=EVENT_TICK + tick.instID)
        event2.dict['data'] = tick
        self.event_engine.put(event2)

    def on_trade(self, trade):
        """trade event"""
        # generic trade event
        event1 = Event(type=EVENT_TRADE)
        event1.dict['data'] = trade
        self.event_engine.put(event1)
        # specific contract  
        event2 = Event(type=EVENT_TRADE + trade.order_ref)
        event2.dict['data'] = trade
        self.event_engine.put(event2)

    def on_order(self, order):
        """order event"""
        # generic order event
        event1 = Event(type=EVENT_ORDER)
        event1.dict['data'] = order
        self.event_engine.put(event1)
        # specific order 
        event2 = Event(type=EVENT_ORDER + order.order_ref)
        event2.dict['data'] = order
        self.event_engine.put(event2)

    def on_position(self, position):
        """position event"""
        # generic position
        event1 = Event(type=EVENT_POSITION)
        event1.dict['data'] = position
        self.event_engine.put(event1)
        # event on specific position 
        event2 = Event(type=EVENT_POSITION+position.instID)
        event2.dict['data'] = position
        self.event_engine.put(event2)

    def on_account(self, account):
        """account event"""
        event1 = Event(type=EVENT_ACCOUNT)
        event1.dict['data'] = account
        self.event_engine.put(event1)

        event2 = Event(type=EVENT_ACCOUNT+account.vtAccountID)
        event2.dict['data'] = account
        self.event_engine.put(event2)

    def on_log(self, log_content, level = logging.DEBUG):
        """log event"""
        event = Event(type=EVENT_LOG)
        event.dict['data'] = log_content
        event.dict['owner'] = "gateway_" + self.gateway_name
        event.dict['level'] = level
        self.event_engine.put(event)

    def on_contract(self, contract):
        """contract data event"""        
        event1 = Event(type=EVENT_CONTRACT)
        event1.dict['data'] = contract
        self.event_engine.put(event1)

    def load_order_list(self, tday):
        logfile = self.file_prefix + 'order_' + tday.strftime('%y%m%d') + '.csv'
        if not os.path.isfile(logfile):
            return {}
        self.id2order = {}
        with open(logfile, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx > 0:
                    instIDs = row[3].strip().split('.')
                    inst = instIDs[0]
                    if len(instIDs) > 1:
                        exchange = instIDs[1]
                    else:
                        exchange = inst2exch(inst)
                    order_class = eval(str(row[14]))
                    filled_orders = {}
                    if ':' in row[7]:
                        filled_str = row[7].split('|')
                        for fstr in filled_str:
                            if (':' not in fstr) or ('_' not in fstr):
                                continue
                            forder = fstr.split(':')
                            pair_str = forder[1].split('_')
                            filled_orders[forder[0]] = [float(pair_str[0]), int(float(pair_str[1]))]
                    iorder = order_class(instID = inst, exchange = exchange, limit_price = float(row[11]), \
                            volume = int(float(row[4])), order_time = int(float(row[12])), \
                            action_type = Offset(row[8]), direction = Direction(row[9]), price_type = OrderType(row[10]), \
                            trade_ref = int(float(row[15])), order_ref = int(row[0]), \
                            sys_id = row[2].strip(), status = OrderStatus(int(row[13])), \
                            local_id = row[1], filled_orders = filled_orders, \
                            filled_volume = int(float(row[5])), filled_price = float(row[6]))
                    self.add_order(iorder)

    def save_order_list(self, tday):
        orders = list(self.id2order.keys())
        if len(self.id2order) > 1:
            orders.sort()
        order_list = [self.id2order[key] for key in orders]
        filename = self.file_prefix + 'order_' + tday.strftime('%y%m%d') + '.csv'
        with open(filename, 'w', newline='') as log_file:
            file_writer = csv.writer(log_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL);
            file_writer.writerow(
                ['order_ref', 'local_id', 'sysID', 'inst', 'volume',
                 'filledvolume', 'filledprice', 'filledorders',
                 'action_type', 'direction', 'price_type',
                 'limitprice', 'order_time', 'status', 'order_class', 'trade_ref'])
            for iorder in order_list:
                forders = [ str(key) + ':' + '_'.join([str(s) for s in iorder.filled_orders[key]]) for key in iorder.filled_orders if len(str(key))>0 ]
                filled_str = '|'.join(forders)
                instID = '.'.join([iorder.instrument, iorder.exchange])
                file_writer.writerow(
                    [iorder.order_ref, iorder.local_id, iorder.sys_id, instID, iorder.volume,
                     iorder.filled_volume, iorder.filled_price, filled_str,
                     iorder.action_type.value, iorder.direction.value, iorder.price_type.value,
                     iorder.limit_price, iorder.start_tick, iorder.status.value, iorder.type, iorder.trade_ref])

    def add_orders(self, orders):
        for iorder in orders:
            self.add_order(iorder)

    def set_order_id(self, iorder):
        iorder.local_id = str(iorder.order_ref)

    def add_order(self, iorder):
        self.set_order_id(iorder)
        iorder.set_gateway(self)
        iorder.add_pos()
        self.id2order[iorder.local_id] = iorder
        if (iorder.status in Alive_Order_Status) and (iorder.local_id not in self.working_orders):
            self.working_orders.append(iorder.local_id)
            self.process_flag = True

    def update_working_orders(self):
        self.working_orders = [ local_id for local_id in self.working_orders if not self.id2order[local_id].is_closed()]
        self.working_order_stats = {}
        for local_id in self.working_orders:
            iorder = self.id2order[local_id]
            key = (iorder.instrument, iorder.direction)
            if (key not in self.working_order_stats):
                self.working_order_stats[key] = []
            data = (local_id, iorder.limit_price, iorder.unfilled())
            self.working_order_stats[key].append(data)

    def check_order_permission(self, instID, direction):
        if direction == Direction.LONG:
            rev_dir = Direction.SHORT
        else:
            rev_dir = Direction.LONG
        key =  (instID, rev_dir)
        if (key in self.working_order_stats):
            return False
        else:
            return self.order_stats[instID]['status']

    def send_queued_orders(self):
        self.update_working_orders()
        for local_id in self.working_orders:
            iorder = self.id2order[local_id]
            if (iorder.status == OrderStatus.Ready) and self.check_order_permission(iorder.instrument, iorder.direction):
                self.send_order(iorder)
        self.process_flag = False

    def remove_order(self, iorder):
        iorder.remove_pos()
        self.id2order.pop(iorder.local_id, None)

    def load_local_positions(self, tday):
        pos_date = tday
        logfile = self.file_prefix + 'EODPos_' + pos_date.strftime('%y%m%d')+'.csv'
        if not os.path.isfile(logfile):
            pos_date = workdays.workday(pos_date, -1, CHN_Holidays)
            logfile = self.file_prefix + 'EODPos_' + pos_date.strftime('%y%m%d')+'.csv'
            if not os.path.isfile(logfile):
                logContent = "no prior position file is found"
                self.on_log(logContent, level = logging.INFO)
                return False
        else:
            self.agent.eod_flag = True
        with open(logfile, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if row[0] == 'capital':
                    self.account_info['prev_capital'] = float(row[1])
                elif row[0] == 'pos':
                    inst = row[1]
                    if inst in self.instruments:
                        if inst not in self.positions:
                            (pos_cls, pos_args) = self.get_pos_class(self.agent.instruments[inst])
                            self.positions[inst] = pos_cls(self.agent.instruments[inst], self, **pos_args)
                        self.positions[inst].update_pos('pos_yday', row[2:])
        return True

    def save_local_positions(self, tday):
        file_prefix = self.file_prefix
        logfile = file_prefix + 'EODPos_' + tday.strftime('%y%m%d')+'.csv'
        if os.path.isfile(logfile):
            return False
        else:
            with open(logfile,'w', newline='') as log_file:
                file_writer = csv.writer(log_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL);
                for inst in self.positions:
                    pos = self.positions[inst]
                    pos.re_calc()
                self.calc_margin()
                file_writer.writerow(['capital', self.account_info['curr_capital']])
                for inst in self.positions:
                    pos = self.positions[inst]
                    if sum([ abs(ppos) for ppos in pos.curr_pos]) > 0:
                        file_writer.writerow(['pos', inst] + pos.curr_pos)
            return True

    def get_order_offset(self, instID, volume, order_num = 1):
        return [(Offset.OPEN, volume)]

    def book_order(self, instID, exchange, volume, price_type, limit_price, trade_ref = 0, order_num = 1):
        direction = Direction.LONG if volume > 0 else Direction.SHORT
        self.update_working_orders()
        # avoid sending orders when the other direction is in the market
        if not self.check_order_permission(instID, direction):
            return []
        order_offsets = self.get_order_offset(instID, volume, order_num)
        new_orders = [Order(instID = instID, exchange = exchange, limit_price = limit_price, volume = v, \
                            order_time = self.agent.tick_id, action_type = action_type, \
                            direction = direction, price_type = price_type, trade_ref = trade_ref) \
                            for (action_type, v) in order_offsets]
        self.add_orders(new_orders)
        self.positions[instID].re_calc()
        return new_orders

    def calc_margin(self):
        for instID in self.positions:
            inst = self.agent.instruments[instID]
            pos = self.positions[instID]
            under_price = 0.0
            if (inst.ptype in Option_ProductTypes):
                under_price = self.agent.instruments[inst.underlying].price
            self.pl_by_product[instID] = {}
            self.pl_by_product[instID]['yday_mark'] = inst.prev_close
            self.pl_by_product[instID]['tday_mark'] = inst.price
            self.pl_by_product[instID]['new_long']  = pos.tday_pos[0]
            self.pl_by_product[instID]['new_short'] = pos.tday_pos[1]
            self.pl_by_product[instID]['new_long_avg'] = pos.tday_avp[0]
            self.pl_by_product[instID]['new_short_avg'] = pos.tday_avp[1]
            self.pl_by_product[instID]['locked_margin'] =  abs(pos.locked_pos[0]) \
                                                        * inst.calc_margin_amount(Direction.LONG if pos.locked_pos[0] > 0 else Direction.SHORT, under_price)
            self.pl_by_product[instID]['used_margin'] =  abs(pos.curr_pos[0]) \
                                                        * inst.calc_margin_amount(Direction.LONG if pos.curr_pos[0] > 0 else Direction.SHORT, under_price)
            self.pl_by_product[instID]['yday_pos'] = pos.pos_yday[0]
            self.pl_by_product[instID]['yday_pnl'] = pos.pos_yday[0] * (inst.price - inst.prev_close) * inst.multiple
            self.pl_by_product[instID]['tday_pnl'] =  pos.tday_pos[0] * (inst.price-pos.tday_avp[0]) * inst.multiple
            self.pl_by_product[instID]['tday_pnl'] -= pos.tday_pos[1] * (inst.price-pos.tday_avp[1]) * inst.multiple
        for key in ['locked_margin', 'used_margin', 'yday_pnl', 'tday_pnl']:
            self.account_info[key] = sum([self.pl_by_product[instID][key] for instID in self.positions])
        self.account_info['pnl_total'] = self.account_info['yday_pnl'] + self.account_info['tday_pnl']
        self.account_info['curr_capital'] = self.account_info['prev_capital'] + self.account_info['pnl_total']
        self.account_info['available'] = self.account_info['curr_capital'] - self.account_info['locked_margin']

    def connect(self):
        """connectgateway"""
        pass

    def subscribe(self, subscribeReq):
        """subscribe market data"""
        pass

    def send_order(self, iorder):
        instID = iorder.instrument
        self.order_stats[instID]['submit'] += 1
        self.order_stats['total_submit'] += 1
        if (self.order_stats[instID]['submit'] >= self.order_constraints['submit_limit']) and (self.order_stats[instID]['status']):
            self.order_stats[instID]['status'] = False
            logContent = 'instrument = %s is disabled for trading after the new submitted order due to position control' % (instID)
            self.on_log( logContent, level = logging.WARNING)

    def cancel_order(self, iorder):
        """cancel order"""
        instID = iorder.instrument
        self.order_stats[instID]['cancel'] += 1
        if (self.order_stats[instID]['cancel'] >= self.order_constraints['cancel_limit']) and self.order_stats[instID]['status']:
            self.order_stats[instID]['status'] = False
            self.on_log( 'Number of Cancel on instID=%s is more than 100: OrderRef=%s, OrderSysID=%s, volume=%s, filled=%s, cancelled=%s' % (iorder.instrument, \
                            iorder.local_id, iorder.sys_id, iorder.volume, iorder.filled_volume, iorder.cancelled_volume), level = logging.WARNING)

        self.order_stats['total_cancel'] += 1
        self.on_log( 'A_CC:cancel order: OrderRef=%s, OrderSysID=%s, instID=%s, volume=%s, filled=%s, cancelled=%s' % (iorder.local_id, \
                            iorder.sys_id, iorder.instrument, iorder.volume, iorder.filled_volume, iorder.cancelled_volume), level = logging.DEBUG)

    def qry_account(self):
        """query account """
        pass

    def qry_position(self):
        """query position"""
        pass

    def qry_instrument(self):
        pass

    def qry_order(self):
        pass

    def qry_trade(self):
        pass

    def close(self):
        """close connection"""
        pass

    def register_event_handler(self):
        pass


class GrossGateway(Gateway):
    def __init__(self, agent, gateway_name = 'Gateway'):
        super(GrossGateway, self).__init__(agent, gateway_name)

    def get_order_offset(self, instID, volume, order_num = 1):
        direction = Direction.LONG if volume > 0 else Direction.SHORT
        vol = abs(volume)
        pos = self.positions[instID]
        if volume > 0:
            can_close = pos.can_close[0]
            can_yclose = pos.can_yclose[0]
        else:
            can_close = pos.can_close[1]
            can_yclose = pos.can_yclose[1]
        is_shfe = (pos.instrument.exchange == 'SHFE')
        n_orders = order_num
        res = []
        if (can_close > 0) and (vol > 0) and ((n_orders > 1) or (can_close >= vol)):
            trade_vol = min(vol, can_close)
            offset = Offset.CLOSETDAY if is_shfe else Offset.CLOSE
            res.append((offset, trade_vol))
            vol -= trade_vol
            n_orders -= 1
        if (can_yclose > 0) and (vol > 0) and is_shfe and ((n_orders > 1) or (can_yclose >= vol)):
            trade_vol = min(vol, can_yclose)
            res.append((Offset.CLOSEYDAY, trade_vol))
            vol -= trade_vol
            n_orders -= 1
        if vol > 0:
            res.append((Offset.OPEN, vol))
        return res

    def book_spd_orders(self, instID, exchange, volume, price_type, limit_price, trade_ref = 0):
        direction = Direction.LONG if volume > 0 else Direction.SHORT
        self.update_working_orders()
        # avoid sending orders when the other direction is in the market
        if not self.check_order_permission(instID, direction):
            return []
        vol = abs(volume)
        instIDs, units = spreadinst2underlying(instID)
        res = [self.get_order_offset(inst, u * volume, 1) for inst, u in zip(instIDs, units)]
        action_type = ''.join([offset[0][0] for offset in res])
        new_order = SpreadOrder(inst = instID, exchange = exchange, limit_price = limit_price, volume = vol, \
                            order_time = self.agent.tick_id, action_type = action_type, \
                            direction = direction, price_type = price_type, \
                            trade_ref = trade_ref)
        self.add_order(new_order)
        for inst in instIDs:
            self.positions[inst].re_calc()
        return [new_order]

    def calc_margin(self):
        for instID in self.positions:
            inst = self.agent.instruments[instID]
            pos = self.positions[instID]
            under_price = 0.0
            if (inst.ptype in Option_ProductTypes):
                under_price = self.agent.instruments[inst.underlying].price
            self.pl_by_product[instID] = {}
            self.pl_by_product[instID]['yday_mark'] = inst.prev_close
            self.pl_by_product[instID]['tday_mark'] = inst.price
            self.pl_by_product[instID]['new_long']  = pos.tday_pos[0]
            self.pl_by_product[instID]['new_short'] = pos.tday_pos[1]
            self.pl_by_product[instID]['new_long_avg'] = pos.tday_avp[0]
            self.pl_by_product[instID]['new_short_avg'] = pos.tday_avp[1]
            self.pl_by_product[instID]['locked_margin'] =  pos.locked_pos[0] * inst.calc_margin_amount(Direction.LONG, under_price)
            self.pl_by_product[instID]['locked_margin'] += pos.locked_pos[1] * inst.calc_margin_amount(Direction.SHORT, under_price)
            self.pl_by_product[instID]['used_margin'] =  pos.curr_pos[0] * inst.calc_margin_amount(Direction.LONG, under_price)
            self.pl_by_product[instID]['used_margin'] += pos.curr_pos[1] * inst.calc_margin_amount(Direction.SHORT, under_price)
            self.pl_by_product[instID]['yday_pos'] = pos.pos_yday[0] - pos.pos_yday[1]
            self.pl_by_product[instID]['yday_pnl'] = (pos.pos_yday[0] - pos.pos_yday[1]) * (inst.price - inst.prev_close) * inst.multiple
            self.pl_by_product[instID]['tday_pnl'] =  pos.tday_pos[0] * (inst.price-pos.tday_avp[0]) * inst.multiple
            self.pl_by_product[instID]['tday_pnl'] -= pos.tday_pos[1] * (inst.price-pos.tday_avp[1]) * inst.multiple
        for key in ['locked_margin', 'used_margin', 'yday_pnl', 'tday_pnl']:
            self.account_info[key] = sum([self.pl_by_product[instID][key] for instID in self.positions])
        self.account_info['pnl_total'] = self.account_info['yday_pnl'] + self.account_info['tday_pnl']
        self.account_info['curr_capital'] = self.account_info['prev_capital'] + self.account_info['pnl_total']
        self.account_info['available'] = self.account_info['curr_capital'] - self.account_info['locked_margin']
