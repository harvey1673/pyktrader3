# -*- coding: utf-8 -*-
import os.path
import csv
import json
from . order import Order
from . trade import XTrade
from . trading_const import Alive_Trade_Status, Active_Trade_Status
from . trade_exec_algo import *
from pycmqlib3.utility.misc import sign


class TradeBook(object):
    def __init__(self, underlying):
        self.name = underlying.name
        self.bids = []
        self.asks = []
        self.underlying = underlying

    def get_all_trades(self):
        self.filter_alive_trades()
        return [xtrade.id for xtrade in self.bids + self.asks if xtrade.status in Alive_Trade_Status]

    def remove_trade(self, xtrade):
        if xtrade.vol > 0:
            self.bids = [x for x in self.bids if x.id != xtrade.id]
        else:
            self.asks = [x for x in self.asks if x.id != xtrade.id]

    def filter_alive_trades(self):
        self.bids = [xtrade for xtrade in self.bids if xtrade.status in Alive_Trade_Status]
        self.asks = [xtrade for xtrade in self.asks if xtrade.status in Alive_Trade_Status]

    def add_trade(self, xtrade):
        if xtrade.vol > 0:
            self.bids.append(xtrade)
        else:
            self.asks.append(xtrade)

    def match_trades(self):
        nbid = len(self.bids)
        nask = len(self.asks)
        n = 0
        m = 0
        traded_price = self.underlying.mid_price
        while (n < nbid) and (m < nask):
            bid_trade = self.bids[n]
            ask_trade = self.asks[m]
            if bid_trade.remaining_vol == 0:
                n += 1
            elif ask_trade.remaining_vol == 0:
                m += 1
            else:
                if abs(bid_trade.remaining_vol) <= abs(ask_trade.remaining_vol):
                    ask_trade.on_trade(traded_price, -bid_trade.remaining_vol)
                    ask_trade.remaining_vol += bid_trade.remaining_vol
                    ask_trade.refresh()
                    bid_trade.on_trade(traded_price, bid_trade.remaining_vol)
                    bid_trade.remaining_vol = 0
                    bid_trade.refresh()
                    n += 1
                else:
                    ask_trade.on_trade(traded_price, ask_trade.remaining_vol)
                    ask_trade.remaining_vol = 0
                    ask_trade.refresh()
                    bid_trade.on_trade(traded_price, -ask_trade.remaining_vol)
                    bid_trade.remaining_vol += ask_trade.remaining_vol
                    bid_trade.refresh()
                    m += 1

class SimpleTradeBook(object):
    def __init__(self, xtrade):
        self.underlying = xtrade.underlying
        self.instIDs = xtrade.instIDs
        self.inst_objs = [xtrade.agent.instruments[inst] for inst in self.instIDs]
        self.units = xtrade.units
        self.name = xtrade.underlying.name
        self.bids = []
        self.asks = []
    
    def get_all_trades(self):
        self.filter_alive_trades()
        return [xtrade.id for xtrade in self.bids + self.asks if xtrade.status in Alive_Trade_Status]
        
    def remove_trade(self, xtrade):
        if xtrade.vol > 0:
            self.bids = [ x for x in self.bids if x.id != xtrade.id ]
        else:
            self.asks = [ x for x in self.asks if x.id != xtrade.id ]

    def filter_alive_trades(self):
        self.bids = [xtrade for xtrade in self.bids if xtrade.status in Alive_Trade_Status]
        self.asks = [xtrade for xtrade in self.asks if xtrade.status in Alive_Trade_Status]

    def add_trade(self, xtrade):
        if xtrade.vol > 0 and (xtrade not in self.bids):
            self.bids.append(xtrade)
        elif xtrade.vol < 0 and (xtrade not in self.asks):
            self.asks.append(xtrade)
    
    def match_trade(self, xtrade):
        direction = int(sign(xtrade.vol))
        trade_list = self.asks if direction > 0 else self.bids
        traded_price = self.underlying.mid_price
        for mtrade in trade_list:
            if xtrade.remaining_vol == 0:
                return
            mvol = min(abs(mtrade.remaining_vol), abs(xtrade.remaining_vol))
            if mvol > 0:
                mtrade.on_trade(traded_price, - direction * mvol)
                mtrade.remaining_vol += direction * mvol
                mtrade.refresh()
                xtrade.on_trade(traded_price, direction * mvol)
                xtrade.remaining_vol -= direction * mvol
                xtrade.refresh()

    def match_leg(self, xtrade, leg_idx):
        inst = xtrade.instIDs[leg_idx]
        total_vol = vol = abs(xtrade.remaining_vol * xtrade.units[leg_idx])
        if vol== 0:
            return
        leg_direction = int(sign(xtrade.units[leg_idx] * xtrade.remaining_vol))
        idy = self.instIDs.index(inst)
        inst_obj = self.inst_objs[idy]
        inst_dir =  int(sign(self.units[idy]))
        trade_list = self.asks if leg_direction * inst_dir > 0 else self.bids
        for mtrade in trade_list:
            mvol = min(abs(mtrade.units[idy] * mtrade.remaining_vol), vol)
            if mvol > 0:
                direction = Direction.LONG if leg_direction > 0 else Direction.SHORT
                traded_price = inst_obj.mid_price
                iorder = Order(instID = inst, exchange = inst_obj.exchange, limit_price = traded_price, \
                        volume = mvol, direction = direction, trade_ref = xtrade.id, \
                        status = OrderStatus.Done, filled_volume = mvol, filled_price = traded_price)
                xtrade.order_dict[inst].append(iorder)
                vol -= mvol
                direction = Direction.LONG if leg_direction < 0 else Direction.SHORT
                iorder = Order(instID = inst, exchange = inst_obj.exchange, limit_price = traded_price, \
                        volume = mvol, direction = direction, trade_ref = xtrade.id, \
                        status = OrderStatus.Done, filled_volume = mvol, filled_price = traded_price)
                mtrade.order_dict[inst].append(iorder)
                xvol = mvol//abs(self.units[idy]) + (mvol % abs(self.units[idy]) > 0)
                mtrade.remaining_vol -= xvol * int(sign(mtrade.vol))
                mtrade.working_vol += xvol * int(sign(mtrade.vol))
                mtrade.calc_order_stats()
        mvol = total_vol - vol
        if mvol > 0:
            xvol = mvol//abs(xtrade.units[leg_idx]) + (mvol % abs(xtrade.units[leg_idx]) > 0)
            xtrade.remaining_vol -= xvol * int(sign(xtrade.vol))
            xtrade.working_vol += xvol * int(sign(xtrade.vol))
            xtrade.calc_order_stats()


class TradeManager(object):
    def __init__(self, agent):
        self.agent = agent        
        self.tradebooks = {}
        self.pending_trades = {}
        self.ref2trade = {}

    def initialize(self):
        if self.agent.eod_flag:
            return
        trade_list = self.load_trade_list(self.agent.scur_day, self.agent.folder)
        for xtrade in trade_list:
            self.add_trade(xtrade)
        for trade_id in self.ref2trade:
            xtrade = self.ref2trade[trade_id]
            xtrade.refresh()

    def day_finalize(self, scur_day, file_prefix):
        for trade_id in self.ref2trade:
            xtrade = self.ref2trade[trade_id]
            if xtrade.status == TradeStatus.StratConfirm:
                continue
            if xtrade.status == TradeStatus.Pending:
                xtrade.status = TradeStatus.Ready
            xtrade.cancel_remaining()
            xtrade.refresh()
            if xtrade.working_vol != 0:
                filled = [ inst + ":" + str((abs(xtrade.working_vol * unit) - unsent)*sign(xtrade.working_vol*unit)) \
                          for inst, unsent, unit in zip(xtrade.instIDs, xtrade.unsent_volumes, xtrade.units)]
                self.agent.logger.warning("trade_id = %s is forced done, though there are working orders: %s" % (xtrade.id, ','.join(filled)))
            if xtrade.status in Active_Trade_Status:
                xtrade.set_done()
        self.save_trade_list(scur_day, self.ref2trade, file_prefix)
        self.tradebooks = {}
        self.pending_trades = {}
        self.ref2trade = {}

    def get_trade(self, trade_id):
        return self.ref2trade[trade_id] if trade_id in self.ref2trade else None

    def get_trades_by_strat(self, strat_name):
        return [xtrade for xtrade in list(self.ref2trade.values()) if xtrade.strategy == strat_name]

    def add_trade(self, xtrade):
        self.ref2trade[xtrade.id] = xtrade
        key = xtrade.underlying.name
        if xtrade.status == TradeStatus.Pending:
            if key not in self.pending_trades:
                self.pending_trades[key] = []
            self.pending_trades[key].append(xtrade)
        elif xtrade.status in Alive_Trade_Status:
            if key not in self.tradebooks:
                self.tradebooks[key] = SimpleTradeBook(xtrade)
            self.tradebooks[key].match_trade(xtrade)
            if len(xtrade.instIDs) > 1:
                for idx, inst in enumerate(xtrade.instIDs):
                    inst_key = self.agent.instruments[inst].name
                    if inst_key in self.tradebooks:
                        self.tradebooks[inst_key].match_leg(xtrade, idx)
            self.tradebooks[key].add_trade(xtrade)

    def remove_trade(self, xtrade):
        key = xtrade.name
        if xtrade.status == TradeStatus.Pending:
            self.pending_trades[key].remove(xtrade, None)
        elif xtrade.status in Alive_Trade_Status:
            self.tradebooks[key].remove_trade(xtrade)

    def check_pending_trades(self, key):
        alive_trades = []
        if key not in self.pending_trades:
            return
        for xtrade in self.pending_trades[key]:
            curr_price = xtrade.underlying.ask_price1 if xtrade.vol > 0 else xtrade.underlying.bid_price1 
            if (curr_price - xtrade.limit_price) * xtrade.vol >= 0:
                xtrade.status = TradeStatus.Ready
                alive_trades.append(xtrade)
        self.pending_trades[key] = [xtrade for xtrade in self.pending_trades[key] if xtrade.status == TradeStatus.Pending]
        [self.add_trade(xtrade) for xtrade in alive_trades]

    def execute_trades(self, key):
        if key not in self.tradebooks:
            return
        for trade_id in self.tradebooks[key].get_all_trades():
            xtrade = self.ref2trade[trade_id]
            xtrade.execute()

    def save_trade_list(self, curr_date, trade_list, file_prefix):
        filename = file_prefix + 'trade_' + curr_date.strftime('%y%m%d')+'.csv'
        with open(filename,'w', newline='') as log_file:
            file_writer = csv.writer(log_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL);
            file_writer.writerow(['id', 'insts', 'units', 'price_unit', 'vol', 'limitprice',
                                  'filledvol', 'filledprice', 'order_dict', 'aggressive',
                                  'start_time', 'end_time', 'strategy','book', 'status', 'exec_class', 'exec_args'])
            for xtrade in list(trade_list.values()):
                insts = ' '.join(xtrade.instIDs)
                units = ' '.join([str(i) for i in xtrade.units])
                if len(xtrade.order_dict)>0:
                    order_dict = ' '.join([inst +':'+'_'.join([str(o.order_ref) for o in xtrade.order_dict[inst] if o.volume > 0])
                                        for inst in xtrade.order_dict])
                else:
                    order_dict = ''
                if xtrade.algo:
                    exec_class = str(xtrade.algo.__class__.__name__)
                    exec_args = json.dumps(xtrade.algo.exec_args, separators=('#',':'))
                else:
                    exec_class = ''
                    exec_args = ''
                file_writer.writerow([xtrade.id, insts, units, xtrade.price_unit, int(xtrade.vol), xtrade.limit_price,
                                      int(xtrade.filled_vol), xtrade.filled_price, order_dict, xtrade.aggressive_level,
                                      xtrade.start_time, xtrade.end_time, xtrade.strategy, xtrade.book, xtrade.status.value,
                                      exec_class, exec_args])

    def load_trade_list(self, curr_date, file_prefix):
        logfile = file_prefix + 'trade_' + curr_date.strftime('%y%m%d')+'.csv'
        if not os.path.isfile(logfile):
            return []
        trade_list = []
        with open(logfile, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx > 0:
                    instIDs = row[1].split(' ')
                    units = [ int(n) for n in row[2].split(' ')]
                    price_unit = None if len(row[3]) == 0 else float(row[3])
                    vol = int(float(row[4]))
                    limit_price = float(row[5])
                    filled_vol = int(float(row[6]))
                    filled_price = float(row[7])
                    aggressiveness = float(row[9])
                    start_time = int(float(row[10]))
                    end_time = int(float(row[11]))
                    order_dict = {}
                    if ':' in row[8]:
                        str_dict =  dict([tuple(s.split(':')) for s in row[8].split(' ')])
                        for inst in str_dict:
                            if len(str_dict[inst])>0:
                                order_dict[inst] = [int(o_id) for o_id in str_dict[inst].split('_')]
                    strategy = row[12]
                    book = row[13]
                    status = int(row[14])
                    xtrade = XTrade(instIDs = instIDs, units = units, vol = vol, \
                                    limit_price = limit_price, price_unit = price_unit, \
                                    strategy = strategy, book = book, \
                                    filled_vol = filled_vol, filled_price = filled_price, \
                                    start_time = start_time, end_time = end_time, aggressiveness = aggressiveness, \
                                    tradeid = int(row[0]), status = TradeStatus(status), order_dict = order_dict)
                    xtrade.set_agent(self.agent)
                    trade_list.append(xtrade)
                    if xtrade.status in Active_Trade_Status:
                        if len(row)>16 and (len(row[16]) > 0):
                            exec_args = json.loads(row[16].replace('|', '').replace('#', ','))
                            exec_args = dict([(str(key), exec_args[key]) for key in exec_args])
                        else:
                            exec_args = {}
                        if len(row)>15 and (len(row[15]) > 0):
                            exec_class = str(row[15])
                        else:
                            if len(instIDs) == 1:
                                exec_class = 'ExecAlgo1DFixT'
                            else:
                                exec_class = 'ExecAlgoFixTimer'
                        algo = eval(exec_class)(xtrade, **exec_args)
                        xtrade.set_algo(algo)
        return trade_list
