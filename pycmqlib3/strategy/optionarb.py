#-*- coding:utf-8 -*-
from strategy import Strategy
from pycmqlib3.core.trading_const import OrderType, Direction, Option_ProductTypes
from pycmqlib3.utility.misc import inst2exch

def get_option_map(underliers, cont_mths, strikes, exch = ''):
    opt_map = {}
    for under, cmth, ks in zip(underliers, cont_mths, strikes):
        exch = inst2exch(under)
        for otype in ['C', 'P']:
            for strike in ks:
                #cont_mth = int(under[-4:]) + 200000
                key = (str(under), otype, strike)
                instID = under
                if instID[:2] == "IF":
                    instID = instID.replace('IF', 'IO')
                if exch in ['CZCE', 'SHFE']:
                    instID = instID + otype + str(strike)
                else:
                    instID = instID + '-' + otype + '-' + str(strike)
                opt_map[key] = instID
    return opt_map
    
class OptionArbStrat(Strategy):
    common_params =  dict({'future_conts': [], 'strikes': [], 'cont_mths': [], 'exit_ratio': 0.1, 'profit_ratio': 0.01, 'fut_unit':1, 'opt_unit': 1, 'scaler':10}, **Strategy.common_params)
    asset_params = dict({'bid_prices': 0, 'ask_prices': 0, }, **Strategy.asset_params)
    def __init__(self, config, agent = None):
        self.future_conts = config['future_conts']
        self.cont_mths = config['cont_mths']
        self.strikes = config['strikes']
        self.fut_unit = config.get('fut_unit', 1)
        self.opt_unit = config.get('opt_unit', 1)
        self.scaler = config.get('scaler',10)
        self.option_map = get_option_map(self.future_conts, self.cont_mths, self.strikes, )
        underliers = []
        volumes = []
        trade_units = []
        self.value_range = []
        idx = 0
        for fut, strike_list in zip(self.future_conts, self.strikes):
            slen = len(strike_list)
            for i, strike in enumerate(strike_list):
                call_key = (fut, 'C', strike)
                put_key  = (fut, 'P', strike)
                underliers.append([self.option_map[call_key], self.option_map[put_key], fut])
                volumes.append([-self.opt_unit, self.opt_unit, self.fut_unit])
                trade_units.append(1)
                v_range = {'lower': -strike, 'upper':-strike, 'scaler': self.scaler }
                self.value_range.append(v_range)
                idx += 1
                if (i < slen - 1):
                    next_call = (fut, 'C', strike_list[i+1])
                    underliers.append([self.option_map[call_key], self.option_map[next_call]])
                    volumes.append([1, -1])
                    trade_units.append(1)
                    v_range = {'lower':0, 'upper': strike_list[i+1] - strike, 'scaler': self.scaler }
                    self.value_range.append(v_range)
                    idx += 1
                    next_put = (fut, 'P', strike_list[i+1])
                    underliers.append([self.option_map[next_put], self.option_map[put_key]])
                    volumes.append([1, -1])
                    trade_units.append(1)
                    v_range = {'lower':0, 'upper': strike_list[i+1] - strike, 'scaler': self.scaler  }
                    self.value_range.append(v_range)
                    idx += 1
                    if i > 0:
                        prev_call = (fut, 'C', strike_list[i-1])
                        underliers.append([self.option_map[prev_call], self.option_map[call_key], self.option_map[next_call]])
                        volumes.append([1, -2, 1])
                        trade_units.append(1)
                        v_range = {'lower':0, 'upper': None, 'scaler': self.scaler}
                        self.value_range.append(v_range)
                        idx += 1
                        prev_put = (fut, 'P', strike_list[i-1])
                        underliers.append([self.option_map[prev_put], self.option_map[put_key], self.option_map[next_put]])
                        volumes.append([1, -2, 1])
                        trade_units.append(1)
                        v_range = {'lower':0, 'upper': None, 'scaler': self.scaler}
                        self.value_range.append(v_range)
                        idx += 1
        config['assets'] = []
        for under, v, tunit in zip(underliers, volumes, trade_units):
            data = {'underliers': under, 'volumes': v, 'trade_unit': tunit}
            config['assets'].append(data)
        Strategy.__init__(self, config, agent)
        self.order_type = OrderType.LIMIT.value
        self.is_initialized = False
        self.trade_margin = [[0.0, 0.0]] * len(underliers)
        self.inst_margin = dict([(inst, [0.0,0.0]) for inst in self.instIDs])
        self.days_to_expiry = [1.0] * len(underliers)
    
    def initialize(self):
        self.load_state()
        self.update_margin()
        for idx, under in enumerate(self.underliers):
            inst = under[-1]
            conv_f = self.agent.instruments[inst].multiple
            self.value_range[idx]['scaler'] = conv_f
        return
    
    def update_margin(self):
        for instID in self.instIDs:
            inst = self.agent.instruments[instID]
            ins_p = inst.price
            if inst.ptype in Option_ProductTypes:
                ins_p = self.agent.instruments[inst.underlying].price
            self.inst_margin[instID] = [inst.calc_margin_amount(Direction.LONG, ins_p), inst.calc_margin_amount(Direction.SHORT, ins_p)]
        for idx, under in enumerate(self.underliers):
            expiry = self.agent.instruments[under[0]].expiry
            prod = self.agent.instruments[under[0]].product
            self.days_to_expiry[idx] = self.agent.volgrids[prod].t2expiry[expiry]
            margin_l = sum([v*self.inst_margin[ins][0] for v, ins in zip(self.volumes[idx], under) if v > 0])
            margin_l -= sum([ v * self.inst_margin[ins][1] for v, ins in zip(self.volumes[idx], under) if v < 0])
            margin_s = sum([  v * self.inst_margin[ins][1] for v, ins in zip(self.volumes[idx], under) if v > 0])            
            margin_s -= sum([ v * self.inst_margin[ins][0] for v, ins in zip(self.volumes[idx], under) if v < 0])
            self.trade_margin[idx] = [margin_l, margin_s]               
        return
    
    def load_local_variables(self, row):
        pass
    
    def save_local_variables(self, file_writer):
        pass

    def calc_curr_price(self, idx):
        conv_f = [ self.agent.instruments[inst].multiple for inst in self.underliers[idx]]
        ask1 = [ self.agent.instruments[inst].ask_price1 for inst in self.underliers[idx]]
        bid1 = [ self.agent.instruments[inst].bid_price1 for inst in self.underliers[idx]]
        volumes = self.volumes[idx]
        bid_p = sum([p*v*cf for p, v, cf in zip(bid1, volumes, conv_f) if v > 0])
        bid_p += sum([p*v*cf for p, v, cf in zip(ask1, volumes, conv_f) if v < 0])
        self.bid_prices[idx] = bid_p/conv_f[-1]        
        ask_p = sum([p*v*cf for p, v, cf in zip(bid1, volumes, conv_f) if v < 0])
        ask_p += sum([p*v*cf for p, v, cf in zip(ask1, volumes, conv_f) if v > 0])
        self.ask_prices[idx] = ask_p/conv_f[-1]
        return
            
    def on_tick(self, idx, ctick):
        need_save = False
        if len(self.submitted_trades[idx]) > 0:
            return
        if len(self.positions[idx]) > 0:
            return
        bound = self.value_range[idx]
        b_scaler = self.trade_margin[idx][0]*self.days_to_expiry[idx]/bound['scaler']
        s_scaler = self.trade_margin[idx][1]*self.days_to_expiry[idx]/bound['scaler']
        for tradepos in self.positions[idx]:
            buysell = tradepos.direction
            if ((buysell > 0) and (bound['lower'] != None) and 
                (bound['lower'] < self.bid_prices[idx] + self.exit_ratio * b_scaler))   \
                or ((buysell < 0) and (bound['lower'] != None) and                  \
                    (bound['upper'] > self.ask_prices[idx] - self.exit_ratio * s_scaler)):
                if buysell > 0:
                    order_price = self.bid_prices[idx]
                else:
                    order_price = self.ask_prices[idx]
                self.close_tradepos(idx, tradepos, order_price)
                #self.status_notifier(msg)
                need_save = True
        if (bound['lower']!= None) and (bound['lower'] > self.ask_prices[idx] + self.profit_ratio * b_scaler):
            self.open_tradepos(idx, 1, self.ask_prices[idx])
            need_save = True
        elif (bound['upper']!= None) and (bound['upper'] < self.bid_prices[idx] - self.profit_ratio * s_scaler): 
            self.open_tradepos(idx, -1, self.bid_prices[idx])
            need_save = True                
        if need_save:
            self.save_state()
        return

    def update_trade_unit(self):
        pass
