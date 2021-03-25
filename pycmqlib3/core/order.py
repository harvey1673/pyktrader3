#-*- coding:utf-8 -*-
import logging
import itertools
import datetime
import csv
import json
from . trading_const import OrderStatus, Direction, OrderType, Offset
from pycmqlib3.utility.base import BaseObject
from pycmqlib3.utility.misc import spreadinst2underlying

class Order(object):
    id_generator = itertools.count(int(datetime.datetime.strftime(datetime.datetime.now(),'%d%H%M%S')))
    def __init__(self, **kwargs):
        self.instrument = kwargs['instID']
        self.exchange = kwargs.get('exchange', '')
        self.type = self.__class__.__name__
        self.instIDs = kwargs.get('instIDs', [self.instrument])
        self.units = kwargs.get('units', [1])
        self.limit_price = kwargs.get('limit_price', 0.0)
        self.start_tick  = kwargs.get('start_tick', 0)
        self.order_ref = kwargs.get('order_ref', next(self.id_generator))
        self.local_id  = kwargs.get('local_id', str(self.order_ref))
        self.sys_id = kwargs.get('sys_id', '')
        self.trade_ref = kwargs.get('trade_ref', 0)
        self.direction = Direction(kwargs.get('direction', Direction.LONG.value)) # ORDER_BUY, ORDER_SELL
        self.action_type = kwargs.get('action_type', Offset.OPEN) # OF_CLOSE_TDAY, OF_CLOSE, OF_OPEN
        self.price_type = OrderType(kwargs.get('price_type', OrderType.LIMIT.value))
        self.volume = int(kwargs['volume']) 
        self.filled_volume = int(kwargs.get('filled_volume', 0))
        self.filled_price  = kwargs.get('filled_price', 0.0)
        self.cancelled_volume = int(kwargs.get('cancelled_volume', 0))
        self.filled_orders = kwargs.get('filled_orders', {})
        self.positions = []
        self.status = OrderStatus(kwargs.get('status', OrderStatus.Ready.value))
        self.gateway = None

    def set_gateway(self, gateway):
        self.gateway = gateway
        self.positions = [ self.gateway.positions[inst] for inst in self.instIDs]
        self.exchange = gateway.agent.instruments[self.instIDs[0]].exchange

    def recalc_pos(self):
        for pos in self.positions:
            pos.re_calc()

    def add_pos(self):
        for pos in self.positions:
            pos.orders.append(self)
            
    def remove_pos(self):
        for pos in self.positions:
            pos.orders.remove(self)

    def on_trade(self, price, volume, trade_id):
        ''' return traded volume
        '''
        if self.status == OrderStatus.Done:
            return True
        id_key = str(trade_id)
        if id_key in self.filled_orders:
            return False
        self.filled_orders[id_key] = [price, int(volume)]
        self.update()
        if (self.filled_volume == self.volume):
            self.status = OrderStatus.Done
        logging.debug('order traded:price=%s,volume=%s,filled_vol=%s' % (price,volume,self.filled_volume))
        self.recalc_pos()
        return self.is_closed()

    def update(self):
        self.filled_volume = sum([v for p, v in list(self.filled_orders.values())])
        self.filled_price = sum([p * v for p, v in list(self.filled_orders.values())])/self.filled_volume

    def on_order(self, sys_id, price = 0, volume = 0):
        self.sys_id = sys_id
        if volume > self.filled_volume:
            self.filled_price = price
            self.filled_volume = int(volume)
            if self.filled_volume == self.volume:
                self.status = OrderStatus.Done
                self.recalc_pos()
                return True
        return False

    def on_cancel(self): 
        if (self.status != OrderStatus.Cancelled) and (self.volume > self.filled_volume):
            self.status = OrderStatus.Cancelled
            self.cancelled_volume = max(self.volume - self.filled_volume, 0)
            self.volume = self.filled_volume 
            logging.debug('cancel order: OrderRef=%s, instID=%s, volume=%s, filled=%s, cancelled=%s' \
                % (self.order_ref, self.instrument, self.volume, self.filled_volume, self.cancelled_volume))
            self.recalc_pos()

    def is_closed(self): 
        return (self.filled_volume == self.volume) 

    def unfilled(self):
        return (self.volume - self.filled_volume)

    def __unicode__(self):
        return 'Order_A: Order_ref = %s, InstID=%s,Direction=%s,Target vol=%s,Vol=%s,Status=%s' % (self.order_ref, \
                    self.instrument, self.direction.value,
                    self.volume, self.filled_volume, self.status, )

    def __str__(self):
        return str(self).encode('utf-8')

    def to_json(self):
        return json.dumps(self, skipkeys = True)

class SpreadOrder(Order):
    def __init__(self, **kwargs):
        super(SpreadOrder, self).__init__(**kwargs)
        self.instIDs, self.units = spreadinst2underlying(self.instrument)
        self.sub_orders = []
        for idx, (inst, unit) in enumerate(zip(self.instIDs, self.units)):
            sorder = BaseObject(action_type = self.action_type[idx],
                                   direction = self.direction if unit > 0 else reverse_direction(self.direction),
                                   filled_volume = 0,
                                   volume = abs(unit * self.vol),
                                   filled_price = 0)
            self.sub_orders.append(sorder)

    def add_pos(self):
        for pos, sorder in zip(self.positions, self.sub_orders):
            pos.orders.append(sorder)

    def remove_pos(self):
        for pos, sorder in zip(self.positions, self.sub_orders):
            pos.orders.remove(sorder)        

    def update(self):
        super(SpreadOrder, self).update()
        curr_p = 0
        for pos, sorder, unit in zip(self.positions, self.sub_orders, self.units):
            p = pos.instrument.mid_price
            sorder.filled_volume = self.filled_volume * abs(unit)
            sorder.filled_price = p
            curr_p += p * unit
        self.sub_orders[0].filled_price -= (curr_p - self.filled_price)/self.units[0]
