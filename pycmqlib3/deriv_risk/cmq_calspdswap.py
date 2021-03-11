import datetime
import copy
from . cmq_inst import *
from . cmq_calendarswap import *
from . cmq_calendarswap import *

class CMQCalSpdSwap(CMQInstrument):
    class_params = dict(CMQInstrument.class_params, **{ 'strike': 0.0,
                                                        'start1': datetime.date.today() + datetime.timedelta(days = 2),
                                                        'end1': datetime.date.today() + datetime.timedelta(days=2),
                                                        'start2': datetime.date.today() + datetime.timedelta(days=60),
                                                        'end2': datetime.date.today() + datetime.timedelta(days=60),
                                                        'fwd_index': 'SGXIRO',
                                                        'need_disc': True})
    inst_key = ['fwd_index', 'strike', 'start1', 'end1', 'start2', 'end2', 'ccy', 'need_disc', 'volume']

    def __init__(self, trade_data, market_data = {}, model_settings = {}):
        super(CMQCalSpdSwap, self).__init__(trade_data, market_data, model_settings)

    def set_trade_data(self, trade_data):
        super(CMQCalSpdSwap, self).set_trade_data(trade_data)
        tdata_a = copy.deepcopy(trade_data)
        tdata_a['strike'] = 0.0
        tdata_a['start'] = trade_data['start1']
        tdata_a['end'] = trade_data['end1']
        tdata_a['settled_flag'] = False
        tdata_b = copy.deepcopy(trade_data)
        tdata_b['strike'] = 0.0
        tdata_b['start'] = trade_data['start2']
        tdata_b['end'] = trade_data['end2']
        tdata_b['settled_flag'] = False
        self.swap_a = CMQCalSwapFuture(tdata_a)
        self.swap_b = CMQCalSwapFuture(tdata_b)
        self.mkt_deps = {}
        self.mkt_deps['COMFix'] = copy.deepcopy(self.swap_a.mkt_deps['COMFix'])
        for spot_id in self.swap_b.mkt_deps['COMFix']:
            if spot_id not in self.mkt_deps['COMFix']:
                self.mkt_deps['COMFix'][spot_id] = []
            self.mkt_deps['COMFix'][spot_id] = list(set(self.mkt_deps['COMFix'][spot_id]).union(self.swap_b.mkt_deps['COMFix'][spot_id]))
            self.mkt_deps['COMFix'][spot_id].sort()
        self.mkt_deps['COMFwd'] = { self.fwd_index:
                                        list(set(self.swap_a.mkt_deps['COMFwd'][self.fwd_index]).union(
                                            self.swap_b.mkt_deps['COMFwd'][self.fwd_index])) }
        self.mkt_deps['COMFwd'][self.fwd_index].sort()
        if self.need_disc:
            self.mkt_deps['IRCurve'] = { self.ccy.lower() + '_disc': ['ALL'] }

    def set_market_data(self, market_data):
        if len(market_data) == 0:
            return
        self.swap_a.set_market_data(market_data)
        self.swap_b.set_market_data(market_data)
        super(CMQCalSpdSwap, self).set_market_data(market_data)
        if self.need_disc:
            self.df = disc_factor(self.value_date, max(self.end1, self.end2, self.end), market_data['IRCurve'][self.ccy.lower() + '_disc'])

    def clean_price(self):
        if self.settled_flag and (self.value_date > max(self.end1, self.end2)):
            return 0.0
        else:
            return (self.swap_a.clean_price() - self.swap_b.clean_price() - self.strike) * self.df