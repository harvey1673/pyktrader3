#-*- coding:utf-8 -*-
from pycmqlib3.utility.misc import sign
from trading_const import OrderStatus, OrderType, TradeStatus

class ExecAlgoBase(object):
    def __init__(self, xtrade, **kwargs):
        self.xtrade = xtrade
        self.inst_order = kwargs.get('inst_order', list(range(len(xtrade.instIDs))))
        self.max_vol = kwargs.get('max_vol', 40)
        self.stop_price = kwargs.get('stop_price', None)
        self.price_mode = kwargs.get('price_mode', -1)
        self.tick_num = kwargs.get('tick_num', 0)
        self.exec_args = kwargs
        self.set_agent(xtrade.agent)
        self.book_func = 'book_order'
        self.book_args = {'order_num': 3}

    def set_agent(self, agent):
        self.agent = agent
        if self.agent!= None:
            self.inst_objs = [ self.agent.instruments[instID] for instID in self.xtrade.instIDs]
        else:
            self.inst_objs = [None] * len(self.xtrade.instIDs)

    def execute(self):
        pass

class ExecAlgo1DFixT(ExecAlgoBase):
    def __init__(self, xtrade, **kwargs):
        super(ExecAlgo1DFixT, self).__init__(xtrade, **kwargs)
        if len(xtrade.instIDs) > 1:
            print("error on ExecAlgo1DFixT for asset %s" % xtrade.instIDs)
        self.timer_period = kwargs.get('time_period', 100)
        self.next_timer = 0
        self.price_type = kwargs.get('price_type', OrderType.LIMIT.value)

    def set_agent(self, agent):
        super(ExecAlgo1DFixT, self).set_agent(agent)
        self.next_timer = self.agent.tick_id - 1

    def execute(self):
        direction = int(sign(self.xtrade.vol))
        if (self.agent.tick_id > self.xtrade.end_time):
            self.xtrade.cancel_remaining()
        if self.xtrade.status == TradeStatus.OrderSent:
            if (self.agent.tick_id > self.next_timer):
                self.xtrade.cancel_working_orders()
            return
        inst = self.xtrade.instIDs[0]
        if (self.agent.tick_id > self.next_timer) and (self.xtrade.remaining_vol != 0):
            vol = min(self.max_vol, abs(self.xtrade.remaining_vol)) * direction
            self.xtrade.add_working_vol(vol)
            self.next_timer += self.timer_period
        if (self.xtrade.unsent_volumes[0] > 0):
            next_vol = min(self.xtrade.unsent_volumes[0], self.max_vol) * direction
            inst_obj = self.xtrade.underlying
            next_price = inst_obj.shift_price(self.price_mode * next_vol, self.price_mode * self.tick_num)
            gway = self.agent.map_gateway(inst)
            new_orders = getattr(gway, self.book_func)(inst, inst_obj.exchange, next_vol, OrderType(self.price_type), next_price,
                                                       trade_ref=self.xtrade.id, **self.book_args)
            if len(new_orders) > 0:
                self.xtrade.order_dict[inst] += new_orders
                self.xtrade.working_order_list += new_orders
                self.next_timer = self.agent.tick_id + self.timer_period
                self.xtrade.status = TradeStatus.OrderSent
        elif (self.xtrade.remaining_vol == 0) and (self.xtrade.working_vol == 0):
            self.xtrade.status = TradeStatus.Done

        
class ExecAlgoFixTimer(ExecAlgoBase):
    '''send out order by fixed period, cancel the trade when hit the stop price '''
    def __init__(self, xtrade, **kwargs):
        super(ExecAlgoFixTimer, self).__init__(xtrade, **kwargs)
        self.timer_period = kwargs.get('time_period', 600)
        self.next_timer = 0
        self.price_type = kwargs.get('price_type', OrderType.LIMIT.value)
        self.order_num = 3 if kwargs.get('order_offset', True) else 1

    def set_agent(self, agent):
        super(ExecAlgoFixTimer, self).set_agent(agent)
        self.next_timer = self.agent.tick_id - 1

    def execute(self):
        if (self.agent.tick_id > self.xtrade.end_time):
            self.xtrade.cancel_remaining()
        if self.xtrade.status == TradeStatus.OrderSent:
            if (self.agent.tick_id > self.next_timer):
                self.xtrade.cancel_working_orders()
            return
        if (self.agent.tick_id > self.next_timer) and (self.xtrade.remaining_vol != 0):
            vol = min(self.max_vol, abs(self.xtrade.remaining_vol)) * int(sign(self.xtrade.vol))
            self.xtrade.add_working_vol(vol)
            self.next_timer += self.timer_period
        for idx in self.inst_order:
            if self.xtrade.unsent_volumes[idx] > 0:
                direction = int(sign(self.xtrade.working_vol * self.xtrade.units[idx]))
                next_vol = min(self.xtrade.unsent_volumes[idx], self.max_vol * abs(self.xtrade.units[idx])) * direction
                inst_obj = self.inst_objs[idx]
                inst_name = self.xtrade.instIDs[idx]
                next_price = inst_obj.shift_price(self.price_mode * next_vol, self.price_mode * self.tick_num)
                gway = self.agent.map_gateway(inst_name)
                new_orders = getattr(gway, self.book_func)(inst_name, inst_obj.exchange, next_vol, OrderType(self.price_type), next_price,
                                                           trade_ref=self.xtrade.id, **self.book_args)
                if len(new_orders) > 0:
                    self.xtrade.order_dict[inst_name] += new_orders
                    self.xtrade.working_order_list += new_orders
                    self.next_timer = self.agent.tick_id + self.timer_period
                    self.xtrade.status = TradeStatus.OrderSent
                return
        if self.xtrade.remaining_vol == 0:
            self.xtrade.status = TradeStatus.Done
        else:
            self.xtrade.status = TradeStatus.PFilled


class ExecAlgoPriceStd(ExecAlgoBase):
    '''send out order by fixed period, cancel the trade when hit the stop price '''
    def __init__(self, xtrade, **kwargs):
        super(ExecAlgoPriceStd, self).__init__(xtrade, **kwargs)
        self.price_level = kwargs.get('price_level', 0.0)
        self.price_std = kwargs.get('price_std', 0.0)
        self.num_std = kwargs.get('num_std', 0.0)
        self.timer_period = kwargs.get('time_period', 600)
        self.next_timer = 0
        self.price_type = kwargs.get('price_type', OrderType.LIMIT.value)
        self.order_num = 3 if kwargs.get('order_offset', True) else 1

    def set_agent(self, agent):
        super(ExecAlgoPriceStd, self).set_agent(agent)
        self.next_timer = self.agent.tick_id - 1

    def set_target_price(self, price_level, price_std = 0.0, num_std = 0):
        self.price_level = price_level
        self.price_std = price_std
        self.num_std = num_std

    def price_favorable(self, direction = 1):
        under_obj = self.xtrade.underlying
        if direction > 0:
            if (under_obj.ask_price1 <= self.price_level - self.num_std * self.price_std):
                return True
        else:
            if (under_obj.bid_price1 >= self.price_level + self.num_std * self.price_std):
                return True
        return False

    def execute(self):
        direction = int(sign(self.xtrade.vol))
        if (self.agent.tick_id > self.xtrade.end_time):
            self.xtrade.cancel_remaining()
        if self.xtrade.status == TradeStatus.OrderSent:
            if (self.agent.tick_id > self.next_timer):
                self.xtrade.cancel_working_orders()
            return
        elif (self.xtrade.status in [TradeStatus.PFilled, TradeStatus.Ready]) \
                and (sum(self.xtrade.unsent_volumes) == 0) and self.price_favorable(direction):
            vol = min(self.max_vol, abs(self.xtrade.remaining_vol)) * direction
            self.xtrade.add_working_vol(vol)
        inst_obj = None
        inst_name = ''
        for idx in self.inst_order:
            if self.xtrade.unsent_volumes[idx] > 0:
                leg_direction = int(sign(self.xtrade.working_vol * self.xtrade.units[idx]))
                next_vol = min(self.xtrade.unsent_volumes[idx], self.max_vol * abs(self.xtrade.units[idx])) * leg_direction
                inst_obj = self.inst_objs[idx]
                inst_name = self.xtrade.instIDs[idx]
                break
        if inst_obj:
            next_price = inst_obj.shift_price(self.price_mode * next_vol, self.price_mode * self.tick_num)
            gway = self.agent.map_gateway(inst_name)
            new_orders = getattr(gway, self.book_func)(inst_name, inst_obj.exchange, next_vol, OrderType(self.price_type), next_price,
                                                           trade_ref=self.xtrade.id, **self.book_args)
            if len(new_orders) > 0: # in case there are
                self.xtrade.order_dict[inst_name] += new_orders
                self.xtrade.working_order_list += new_orders
                self.next_timer = self.agent.tick_id + self.timer_period
                self.xtrade.status = TradeStatus.OrderSent

class ExecAlgoTWAP(ExecAlgoPriceStd):
    '''send out order by fixed period, cancel the trade when hit the stop price '''
    def __init__(self, xtrade, **kwargs):
        super(ExecAlgoTWAP, self).__init__(xtrade, **kwargs)

