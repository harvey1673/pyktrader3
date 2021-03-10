#-*- coding:utf-8 -*-
import itertools
import datetime
import json
from pycmqlib3.core.trading_const import TradeStatus
from pycmqlib3.utility.misc import sign

class XTrade(object):
    # instances = weakref.WeakSet()
    id_generator = itertools.count(int(datetime.datetime.strftime(datetime.datetime.now(), '%d%H%M%S')))
    def __init__(self, **kwargs):
        self.id = kwargs.get('tradeid', next(self.id_generator))
        self.instIDs = kwargs['instIDs']
        self.units = kwargs['units']
        self.vol = kwargs['vol']
        self.filled_vol = kwargs.get('filled_vol', 0)
        self.filled_price = kwargs.get('filled_price', 0.0)
        self.limit_price = kwargs['limit_price']
        self.price_unit = kwargs.get('price_unit', None)
        self.strategy = kwargs.get('strategy', 'dummy')
        self.book = kwargs.get('book', '0')
        self.status = TradeStatus(kwargs.get('status', TradeStatus.Ready.value))
        self.order_dict = kwargs.get('order_dict', dict([(inst, []) for inst in self.instIDs]))
        self.aggressive_level = kwargs.get('aggressiveness', 1.0)
        self.start_time = kwargs.get('start_time', 300000)
        self.end_time = kwargs.get('end_time', 2059300)
        self.algo = None
        self.agent = None
        self.underlying = None
        self.working_vol = 0
        self.remaining_vol = self.vol - self.filled_vol - self.working_vol
        self.unsent_volumes = [0] * len(self.units)
        self.working_order_list = []

    def set_agent(self, agent):
        self.agent = agent
        self.underlying = agent.get_underlying(self.instIDs, self.units, self.price_unit)
        self.price_unit = self.underlying.multiple
        for inst in self.instIDs:
            if inst not in self.order_dict:
                self.order_dict[inst] = []
            else:
                self.order_dict[inst] = [ self.agent.ref2order[order_ref] for order_ref in self.order_dict[inst] ]
        res = self.calc_order_stats()
        self.remaining_vol = self.vol - self.filled_vol - self.working_vol
        self.unsent_volumes = [abs(self.working_vol * unit) - full for unit, full in zip(self.units, res['full_vols'])]

    def set_algo(self, algo):
        self.algo = algo
        self.algo.set_agent(self.agent)

    def add_working_vol(self, vol):
        self.working_vol += vol
        self.unsent_volumes = [abs(vol * unit) + uv for uv, unit in zip(self.unsent_volumes, self.units)]
        self.remaining_vol = self.vol - self.filled_vol - self.working_vol

    def save(self):
        return json.dumps(self, skipkeys = True)

    def cancel_working_orders(self):
        for iorder in self.working_order_list:
            self.agent.cancel_order(iorder)

    def calc_order_stats(self):
        fill_vols = [0] * len(self.units)
        full_vols = [0] * len(self.units)
        filled_prices = [0.0] * len(self.units)
        self.working_order_list = []
        work_vol = 0
        for idx, (instID, unit) in enumerate(zip(self.instIDs, self.units)):
            self.order_dict[instID] = [ o for o in self.order_dict[instID] if o.volume > 0 ]
            fill_vols[idx] = sum([o.filled_volume for o in self.order_dict[instID]])
            full_vols[idx] = sum([o.volume for o in self.order_dict[instID]])
            if fill_vols[idx] > 0:
                filled_prices[idx] = sum([o.filled_price * o.filled_volume for o in self.order_dict[instID]])/fill_vols[idx]
            order_list = [ o for o in self.order_dict[instID] if not o.is_closed()]
            self.working_order_list += order_list
            work_vol = max(work_vol, abs(full_vols[idx]/unit))
        work_vol = int(sign(self.vol) * work_vol)
        if self.working_vol == 0:
            self.working_vol = work_vol
        stats = {'fill_vols': fill_vols, 'full_vols': full_vols, 'filled_prices': filled_prices}
        self.unsent_volumes = [abs(self.working_vol * unit) - full for unit, full in zip(self.units, full_vols)]
        return stats
            
    def refresh(self):
        if (self.status == TradeStatus.StratConfirm) or (self.status == TradeStatus.Pending):
            return
        elif (self.status == TradeStatus.Done):
            self.set_done()
            return
        order_stats = self.calc_order_stats()
        if (len(self.working_order_list) > 0):
            self.status = TradeStatus.OrderSent
        elif (self.working_vol != 0):
            self.status = TradeStatus.PFilled
            if (sum(self.unsent_volumes) == 0):
                self.order_dict = dict([(inst, []) for inst in self.instIDs])
                avg_price = self.underlying.calc_price(prices= order_stats['filled_prices'])
                self.on_trade(avg_price, self.working_vol)
                self.remaining_vol = self.vol - self.filled_vol
                self.working_vol = 0
        if (self.working_vol == 0) and (self.remaining_vol == 0):
            self.set_done()

    def cancel_remaining(self):
        self.vol = self.vol - self.remaining_vol
        self.remaining_vol = 0
        if self.filled_vol == self.vol:
            self.status = TradeStatus.Done

    def on_trade(self, price, volume):
        if volume == 0:
            return
        sum_price = self.filled_vol * self.filled_price
        self.filled_vol += volume
        self.filled_price = (sum_price + price * volume)/self.filled_vol
    
    def set_done(self):
        self.status = TradeStatus.Done
        self.vol = self.filled_vol
        self.remaining_vol = 0
        self.working_vol = 0
        self.working_order_list = []
        self.unsent_volumes = [0] * len(self.units)
        self.order_dict = dict([(inst, []) for inst in self.instIDs])
        self.algo = None
        self.update_strat()

    def update_strat(self):
        strat = self.agent.strategies[self.strategy]
        strat.on_trade(self)

    def execute(self):
        if (self.status in [TradeStatus.Pending, TradeStatus.StratConfirm]):
            return
        elif self.status == TradeStatus.Done:
             self.update_strat()
        elif self.algo:
            self.algo.execute()
