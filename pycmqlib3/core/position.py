#-*- coding:utf-8 -*-
from . trading_const import Direction, Offset

class Position(object):
    def __init__(self, instrument, gateway = 'CTP'):
        self.instrument = instrument
        self.gateway = gateway
        self.tday_pos = [0, 0]
        self.tday_avp = [0.0, 0.0]
        self.pos_tday = [0]
        self.pos_yday = [0]  # yday's overnight position
        self.curr_pos = [0]
        self.locked_pos = [0]
        self.orders = []    # a list of Orders

    def re_calc(self):
        tday_opened = [0, 0]
        tday_o_locked = [0, 0]
        self.orders = [o for o in self.orders if o.volume != 0]
        for mo in self.orders:
            if mo.direction == Direction.LONG:
                tday_opened[0] += mo.filled_volume
                tday_o_locked[0] += mo.volume
            elif mo.direction == Direction.SHORT:
                tday_opened[1] += mo.filled_volume
                tday_o_locked[1] += mo.volume
        self.tday_pos  = tday_opened
        if self.tday_pos[0] > 0:
            self.tday_avp[0] = sum([o.filled_price*o.filled_volume for o in self.orders if o.direction == Direction.LONG])/self.tday_pos[0]
        else:
            self.tday_avp[0] = 0.0
        if self.tday_pos[1] > 0:
            self.tday_avp[1] = sum([o.filled_price*o.filled_volume for o in self.orders if o.direction == Direction.SHORT])/self.tday_pos[1]
        else:
            self.tday_avp[1] = 0.0

    def update_pos(self, key, value):
        setattr(self, key, int(value[0]))

    def __str__(self):
        return str(self).encode('utf-8')
    
    def __unicode__(self):
        return '%s' % (self.instrument.name)
        
class GrossPosition(Position):
    def __init__(self, instrument, gateway = 'CTP', intraday_close_ratio = 1):
        super(GrossPosition, self).__init__(instrument, gateway)
        self.pos_tday = [0, 0]
        self.pos_yday = [0, 0] # yday's overnight position
        self.curr_pos = [0, 0]
        self.locked_pos = [0, 0]
        self.can_yclose = [0, 0]
        self.can_close  = [0, 0]
        self.can_open = [0, 0]
        self.intraday_close_ratio = intraday_close_ratio

    def update_pos(self, key, value):
        setattr(self, key, [int(value[0]), int(value[1])])

    def set_intraday_close_ratio(self, ratio):
        self.intraday_close_ratio = ratio
    
    def update_can_close(self, tday_opened, tday_c_locked, yday_c_locked):
        self.can_yclose = [0, 0]
        self.can_close[0]  = max(self.pos_yday[1] + int(tday_opened[1] * self.intraday_close_ratio) - tday_c_locked[0], 0)
        self.can_close[1] = max(self.pos_yday[0]  + int(tday_opened[0] * self.intraday_close_ratio) - tday_c_locked[1], 0)
        
    def re_calc(self):
        tday_opened = [0, 0]
        tday_o_locked = [0, 0]
        tday_closed = [0, 0]
        tday_c_locked = [0, 0]
        yday_closed = [0, 0]
        yday_c_locked = [0, 0]

        for mo in self.orders:
            if mo.action_type == Offset.OPEN:
                if mo.direction == Direction.LONG:
                    tday_opened[0] += mo.filled_volume
                    tday_o_locked[0] += mo.volume
                else:
                    tday_opened[1] += mo.filled_volume
                    tday_o_locked[1] += mo.volume
            elif (mo.action_type == Offset.CLOSE) or (mo.action_type == Offset.CLOSETDAY):
                if mo.direction == Direction.LONG:
                    tday_closed[0]  += mo.filled_volume
                    tday_c_locked[0] += mo.volume
                else: 
                    tday_closed[1] += mo.filled_volume
                    tday_c_locked[1] += mo.volume
            elif mo.action_type == Offset.CLOSEYDAY:
                if mo.direction == Direction.LONG:
                    yday_closed[0]  += mo.filled_volume
                    yday_c_locked[0] += mo.volume
                else:
                    yday_closed[1] += mo.filled_volume
                    yday_c_locked[1] += mo.volume

        self.update_can_close(tday_opened, tday_c_locked, yday_c_locked)
        self.tday_pos[0]  = tday_opened[0] + tday_closed[0] + yday_closed[0]
        self.tday_pos[1] = tday_opened[1] + tday_closed[1] + yday_closed[1]

        if self.tday_pos[0] > 0:
            self.tday_avp[0] = sum([o.filled_price*o.filled_volume for o in self.orders if o.direction == Direction.LONG])/self.tday_pos[0]
        else:
            self.tday_avp[0] = 0.0
        if self.tday_pos[1] > 0:
            self.tday_avp[1] = sum([o.filled_price*o.filled_volume for o in self.orders if o.direction == Direction.SHORT])/self.tday_pos[1]
        else:
            self.tday_avp[1] = 0.0

        self.curr_pos[0] = tday_opened[0] - tday_closed[1] + self.pos_yday[0] - yday_closed[1]
        self.curr_pos[1] = tday_opened[1] - tday_closed[0] + self.pos_yday[1] - yday_closed[0]
        self.locked_pos[0] = self.pos_yday[0] - yday_closed[1] + tday_o_locked[0] - tday_closed[1]
        self.locked_pos[1] = self.pos_yday[1] - yday_closed[0] + tday_o_locked[1] - tday_closed[0]
        self.can_open[0] = max(self.instrument.max_holding[0] - self.locked_pos[0], 0)
        self.can_open[1] = max(self.instrument.max_holding[1] - self.locked_pos[1], 0)

    def get_open_volume(self):
        return self.can_open
    
    def get_close_volume(self):
        return self.can_close
    
    def get_yclose_volume(self):
        return self.can_yclose


class SHFEPosition(GrossPosition):
    def __init__(self, instrument, gateway = 'CTP', intraday_close_ratio = 1):
        super(SHFEPosition, self).__init__(instrument, gateway, intraday_close_ratio)
    
    def update_can_close(self, tday_opened, tday_c_locked, yday_c_locked):
        self.can_yclose[0] = max(self.pos_yday[1] - yday_c_locked[0], 0)
        self.can_yclose[1] = max(self.pos_yday[0]  - yday_c_locked[1], 0)
        self.can_close[0] = max(int(tday_opened[1] * self.intraday_close_ratio) - tday_c_locked[0], 0)
        self.can_close[1] = max(int(tday_opened[0] * self.intraday_close_ratio) - tday_c_locked[1], 0)
