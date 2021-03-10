#-*- coding:utf-8 -*-
import datetime
import copy
from pycmqlib3.cmqlib import cmqlib
from pycmqlib3.utility.misc import BDAYS_PER_YEAR, sign, inst2product, product_code, get_opt_expiry,\
    night_session_markets, CHN_Stock_Exch, AMERICAN_OPTION_STEPS
from pycmqlib3.utility.dbaccess import load_product_info, load_inst_marginrate, load_stockopt_info
from pycmqlib3.core.trading_const import OptionType, ProductType, Direction

class VolGrid(object):
    def __init__(self, name, accrual = 'COM', tday = datetime.date.today(), is_spot = False, ccy = 'CNY'):
        self.name = name
        self.accrual = accrual
        self.ccy = ccy
        self.df = {}
        self.fwd = {}
        self.last_update = {}
        self.volnode = {}
        self.volparam = {}
        self.underlier = {}
        self.t2expiry = {}
        self.main_cont = ''
        self.option_insts = {}
        self.spot_model = is_spot

def copy_volgrid(vg):
    volgrid = VolGrid(vg.name, accrual = vg.accrual, is_spot = vg.spot_model, ccy = vg.ccy)
    volgrid.main_cont = vg.main_cont
    for expiry in vg.option_insts:
        volgrid.df[expiry] = vg.df[expiry]
        volgrid.fwd[expiry] = vg.fwd[expiry]
        volgrid.last_update[expiry] = vg.last_update[expiry]
        volgrid.volnode[expiry] = cmqlib.Delta5VolNode(vg.t2expiry[expiry]/BDAYS_PER_YEAR,
                                                          vg.fwd[expiry],
                                                          vg.volparam[expiry][0],
                                                          vg.volparam[expiry][1],
                                                          vg.volparam[expiry][2],
                                                          vg.volparam[expiry][3],
                                                          vg.volparam[expiry][4],
                                                          vg.accrual)
        volgrid.volparam[expiry] = copy.copy(vg.volparam[expiry])
        volgrid.underlier[expiry] = copy.copy(vg.underlier[expiry])
        volgrid.t2expiry[expiry] = vg.t2expiry[expiry]
        volgrid.option_insts[expiry] = copy.copy(vg.option_insts[expiry])
    return volgrid

class Instrument(object):
    def __init__(self,name):
        self.name = name
        self.exchange = 'CFFEX'
        self.ptype = ProductType.Future
        self.product = 'IF'
        self.ccy = 'CNY'
        self.broker_fee = 0.0
        self.marginrate = (0,0) 
        self.multiple = 0
        self.tick_base = 0  
        self.start_tick_id = 0
        self.last_tick_id = 0
        self.max_holding = (500, 500)
        self.cont_mth = 205012  # only used by option and future
        self.expiry = datetime.datetime(2050, 12, 31, 15, 0, 0)
        self.update_bar = True
        self.prev_close = 0.0
        self.pos = 1
        # market snapshot
        self.price = 0.0
        self.openInterest = 0
        self.ask_price1 = 0.0
        self.ask_vol1 = 0
        self.bid_price1 = 0.0
        self.bid_vol1 = 0
        self.ask_price2 = 0.0
        self.ask_vol2 = 0
        self.bid_price2 = 0.0
        self.bid_vol2 = 0
        self.ask_price3 = 0.0
        self.ask_vol3 = 0
        self.bid_price3 = 0.0
        self.bid_vol3 = 0
        self.ask_price4 = 0.0
        self.ask_vol4 = 0
        self.bid_price4 = 0.0
        self.bid_vol4 = 0
        self.ask_price5 = 0.0
        self.ask_vol5 = 0
        self.bid_price5 = 0.0
        self.bid_vol5 = 0
        self.up_limit = 1e10
        self.down_limit = -1e10
        self.mid_price = 0.0
        self.reset_mkt_data()

    def reset_mkt_data(self):
        self.last_update = 0
        self.last_traded = 0
        self.volume = 0
        self.day_finalized = False
    
    def shift_price(self, direction, tick_num = 0, price_level = '1'):
        price_str = 'bid_price' + str(price_level) if direction > 0 else 'ask_price' + str(price_level)
        base_price = getattr(self, price_str)
        if direction > 0:
            return min(base_price + tick_num * self.tick_base, self.up_limit)
        else:
            return max(base_price - tick_num * self.tick_base, self.down_limit)

    def check_price_limit(self, num_tick = 0):        
        tick_base = self.tick_base
        if (self.ask_price1 >= self.up_limit - num_tick * tick_base) or (self.bid_price1 <= self.down_limit + num_tick * tick_base):
            return True
        else:
            return False
            
    def fair_price(self):
        self.mid_price = (self.ask_price1 + self.bid_price1)/2.0
        return self.mid_price

    def calc_price(self, direction = 'mid', prices = None):
        if prices == None:
            if direction == 'bid':
                price = self.bid_price1
            elif direction == 'ask':
                price = self.ask_price1
            else:
                price = self.mid_price
        else:
            price = prices[0]
        return price

    def initialize(self):
        pass
    
    def update_param(self, tday):
        pass

    def update(self, tick):
        curr_tick = tick.tick_id
        if (self.exchange == 'CZCE') and (self.last_update == tick.tick_id) and \
                ((self.volume < tick.volume) or (self.ask_vol1 != tick.ask_vol1) or (self.bid_vol1 != tick.bid_vol1)):
                if tick.tick_id % 10 < 5:
                    tick.tick_id += 5
                    tick.timestamp = tick.timestamp + datetime.timedelta(milliseconds=500)
        if tick.tick_id <= self.last_update:
            return False
        self.up_limit = min(tick.up_limit, self.up_limit)
        self.down_limit = max(tick.down_limit, self.down_limit)
        self.last_update = curr_tick
        for att in ['bid_price1', 'ask_price1', 'bid_vol1', 'ask_vol1', 'openInterest']:
            setattr(self, att, getattr(tick, att))
        self.ask_price1 = min(self.ask_price1, self.up_limit)
        self.bid_price1 = max(self.bid_price1, self.down_limit)
        self.mid_price = (tick.ask_price1 + tick.bid_price1)/2.0
        if (self.mid_price > self.up_limit) or (self.mid_price < self.down_limit):
            return False
        if tick.volume > self.volume:
            self.price  = tick.price
            self.volume = tick.volume
            self.last_traded = curr_tick
        return True

    def calc_margin_amount(self, direction, price = 0.0):
        my_marginrate = self.marginrate[0] if direction == Direction.LONG else self.marginrate[1]
        return self.price * self.multiple * my_marginrate

class SpreadInst(object):
    def __init__(self, inst_data, instIDs, weights, multiple = None):
        self.instIDs = instIDs
        self.ptype = ProductType.Spread
        self.name = '_'.join([str(s) for s in instIDs + weights])
        self.inst_objs = [inst_data[inst] for inst in instIDs]
        self.weights = weights
        self.conv_factor = [ inst_obj.multiple for inst_obj in self.inst_objs ]
        self.tick_base = [inst_obj.tick_base for inst_obj in self.inst_objs]
        self.multiple = multiple if multiple != None else self.conv_factor[-1]
        self.update_bar = True
        for inst_obj in self.inst_objs:
            self.update_bar = self.update_bar and inst_obj.update_bar
        self.prev_close = 0
        self.ask_price1 = 0.0
        self.ask_vol1 = 0
        self.bid_price1 = 0.0
        self.bid_vol1 = 0
        self.mid_price = 0
        self.price = 0
        self.openInterest = 0
        self.up_limit = 1e10
        self.down_limit = -1e10
        self.last_tick_id = 0
        # self.marginrate = (0,0)
        self.reset_mkt_data()


    def reset_mkt_data(self):
        self.last_update = min([inst_obj.last_update for inst_obj in self.inst_objs])
        self.last_traded = min([inst_obj.last_traded for inst_obj in self.inst_objs])
        self.last_tick_id = min([inst_obj.last_tick_id for inst_obj in self.inst_objs])
        self.volume = 0
        self.day_finalized = False

    def update(self, tick = None):
        leg_update = min([inst_obj.last_update for inst_obj in self.inst_objs])
        if tick and (self.last_update >= leg_update):
            return False
        self.last_update = leg_update
        self.bid_price1 = self.calc_price('bid')
        self.ask_price1 = self.calc_price('ask')
        self.mid_price = (self.ask_price1 + self.bid_price1) / 2.0
        self.price = self.calc_price('last')
        return True

    def shift_price(self, direction, tick_num = 0, price_level = '1'):
        price_str = 'bid_price' + str(price_level) if direction > 0 else 'ask_price' + str(price_level)
        base_price = getattr(self, price_str)
        return base_price + sign(direction) * tick_num * sum([abs(tb) for tb in self.tick_base])

    def calc_price(self, direction = 'mid', prices = None):
        if prices == None:
            if direction == 'bid':
                fields = ['bid_price1', 'ask_price1']
            elif direction == 'ask':
                fields = ['ask_price1', 'bid_price1']
            elif direction == 'last':
                fields = ['price', 'price']
            else:
                fields = ['mid_price', 'mid_price']
            prices = [getattr(inst_obj, fields[0]) if w>0 else getattr(inst_obj, fields[1]) for inst_obj, w in zip(self.inst_objs, self.weights)]
        return sum([ p * w * cf for (p, w, cf) in zip(prices, self.weights, self.conv_factor)])/self.multiple
        
class Stock(Instrument):
    def __init__(self,name):
        Instrument.__init__(self, name)
        self.initialize()
        
    def initialize(self):
        self.product = self.name
        self.ptype = ProductType.Equity
        self.start_tick_id = 1530000
        self.last_tick_id  = 2130000
        self.multiple = 1
        self.tick_base = 0.01
        self.broker_fee = 0    
        self.marginrate = (1,0)
        if self.name in CHN_Stock_Exch['SZE']:
            self.exchange = 'SZE'
        else:
            self.exchange = 'SSE'
        return

class Future(Instrument):
    def __init__(self,name):
        Instrument.__init__(self, name)
        self.initialize()
        
    def initialize(self):
        self.ptype = ProductType.Future
        self.product = inst2product(self.name)
        prod_info = load_product_info(self.product)
        self.exchange = prod_info['exch']
        if self.exchange == 'CZCE':
            self.cont_mth = int(self.name[-3:]) + 201000
        else:
            self.cont_mth = int(self.name[-4:]) + 200000
        self.start_tick_id =  1500000
        if self.product in night_session_markets:
            self.start_tick_id = 300000
        elif (self.product in ['T', 'TF', 'TS']):
            self.start_tick_id =  1515000
        elif (self.product in ['IF', 'IO', 'IH', 'IC']):
            self.start_tick_id =  1530000
        self.last_tick_id =  prod_info['end_min'] * 1000     
        self.multiple = prod_info['lot_size']
        self.tick_base = prod_info['tick_size']
        self.broker_fee = prod_info['broker_fee']
        return
    
    def update_param(self, tday):
        self.marginrate = load_inst_marginrate(self.name)
        
class OptionInst(Instrument):
    Greek_Map = {'pv': 'price', 'delta': 'delta', 'gamma': 'gamma', \
                 'theta': 'theta', 'vega': 'vega'}
    def __init__(self, name):
        self.strike = 0.0 # only used by option
        self.otype = OptionType.CALL   # only used by option
        self.underlying = ''   # only used by option
        Instrument.__init__(self, name)
        self.pricer = None
        self.pricer_func = cmqlib.BlackPricer
        self.pricer_param = []
        self.pv = 0.0
        self.delta = 1
        self.theta = 0.0
        self.gamma = 0.0
        self.vega = 0.0
        self.risk_price = self.price
        self.risk_updated = 0.0
        self.margin_param = [0.15, 0.1]
        self.update_bar = False
        self.initialize()

    def approx_pv(self, curr_price):
        dp = (curr_price - self.risk_price)
        return self.pv + self.delta * dp + self.gamma * dp * dp / 2.0

    def approx_delta(self, curr_price):
        return self.delta + (curr_price - self.risk_price) * self.gamma

    def initialize(self):
        pass

    def update_param(self, tday):
        pass
    
    def set_pricer(self, vg, irate):
        expiry = self.expiry
        t2exp = vg.t2expiry[expiry]/BDAYS_PER_YEAR
        param = [t2exp, vg.fwd[expiry], vg.volnode[expiry], self.strike, irate, self.otype.value] + self.pricer_param
        self.pricer = self.pricer_func(*param)

    def update_greeks(self, last_updated,  greeks = ['pv', 'delta', 'gamma', 'vega']):
        if self.pricer == None:
            return None
        for attr in greeks:
            setattr(self, attr, getattr(self.pricer, self.Greek_Map[attr])())
        self.risk_price = self.pricer.fwd_()
        self.risk_updated = last_updated
       
    def calc_margin_amount(self, direction, price = 0.0):
        my_margin = self.price
        if direction == Direction.SHORT:
            a = self.margin_param[0]
            b = self.margin_param[1]
            if price == 0.0:
                price = self.strike
            if self.otype == OptionType.CALL:
                my_margin += max(price * a - max(self.strike-price, 0), price * b)
            else:
                my_margin += max(price * a - max(price - self.strike, 0), self.strike * b)
        return my_margin * self.multiple
        
class StockOptionInst(OptionInst):
    def __init__(self,name):    
        OptionInst.__init__(self, name)
        self.margin_param = [0.12, 0.07]
        self.initialize()
        
    def initialize(self):
        self.ptype = ProductType.EqOpt
        prod_info = load_stockopt_info(self.name)
        self.exchange = prod_info['exch']
        self.multiple = prod_info['lot_size']
        self.tick_base = prod_info['tick_size']
        self.strike = prod_info['strike']
        self.otype = OptionType(prod_info['otype'])
        self.underlying = prod_info['underlying']
        self.product = self.underlying
        self.cont_mth = prod_info['cont_mth']
        self.expiry = get_opt_expiry(self.underlying, self.cont_mth)
        
class FutOptionInst(OptionInst):
    def __init__(self,name):    
        OptionInst.__init__(self, name)
        if self.exchange not in ['CFFEX', 'SHFE']:
            self.pricer_func = cmqlib.AmericanFutPricer
            self.pricer_param = [AMERICAN_OPTION_STEPS]
            self.margin_param = [0.15, 0.1]
        else:
            self.pricer_func = cmqlib.BlackPricer
            self.pricer_param = []
            self.margin_param = [0.15, 0.1]            
        self.initialize()
        
    def initialize(self):
        self.ptype = ProductType.FutOpt
        self.product = inst2product(self.name)
        if self.product in product_code['CZCE'] + product_code['SHFE']:
            if (self.product in product_code['CZCE']):
                idx = 5
            else:
                idx = 6
            self.underlying = self.name[:idx]
            self.otype = OptionType(str(self.name[idx]))
            if idx == 5:
                cmth = int(self.underlying[-3:])
                if cmth < 800:
                    cmth = cmth + 202000
                else:
                    cmth = cmth + 201000
            else:
                cmth = int(self.underlying[-4:]) + 200000
            self.cont_mth = cmth
            self.strike = float(self.name[(idx+1):])
            self.product = self.name[:2]
            self.expiry = get_opt_expiry(self.underlying, self.cont_mth)
        else:
            sep_name = self.name.split('-')
            if self.product == 'IO_Opt':
                self.underlying = sep_name[0].replace('IO','IF')
                self.product = 'IO'
            else:
                self.underlying = sep_name[0]
                self.product = str(self.product[:-4])
            self.strike = float(sep_name[2])
            self.otype = OptionType(str(sep_name[1]))
            self.cont_mth = int(self.underlying[-4:]) + 200000
            self.expiry = get_opt_expiry(self.underlying, self.cont_mth)
        prod_info = load_product_info(self.product)
        self.exchange = prod_info['exch']
        self.start_tick_id =  1500000
        if self.product in night_session_markets:
            self.start_tick_id = 300000
        elif (self.product in ['T', 'TF', 'TS']):
            self.start_tick_id =  1515000
        elif (self.product in ['IF', 'IO', 'IH', 'IC']):
            self.start_tick_id =  1530000
        self.last_tick_id =  prod_info['end_min'] * 1000     
        self.multiple = prod_info['lot_size']
        self.tick_base = prod_info['tick_size']
        self.broker_fee = prod_info['broker_fee']
