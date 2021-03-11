import datetime
import pandas as pd
import numpy as np
from . cmq_inst import disc_factor, CMQInstrument
from . import cmq_curve, cmq_crv_defn
from pycmqlib3.utility import misc

class CMQFXForward(CMQInstrument):
    class_params = dict(CMQInstrument.class_params, **{  'ccypair':'USD/CNY',
                                                         'strike': 0.0,
                                                        'need_disc': True})
    inst_key = ['ccypair', 'strike', 'end', 'ccy', 'volume']

    def __init__(self, trade_data, market_data = {}, model_settings = {}):
        super(CMQFXForward, self).__init__(trade_data, market_data, model_settings)

    def set_trade_data(self, trade_data):
        super(CMQFXForward, self).set_trade_data(trade_data)
        self.mkt_deps['FXFwd'] = { self.ccypair: [ 'ALL'] }
        if self.need_disc:
            self.mkt_deps['IRCurve'] = { self.ccy.lower() + '_disc': ['ALL'] }

    def set_market_data(self, market_data):
        super(CMQFXForward, self).set_market_data(market_data)
        if len(market_data) == 0:
            self.fx_curve = None
            self.fx_fwd = 1.0
            self.df = 1.0
            return
        fx_quotes = market_data['FXFwd'][self.ccypair]
        fx_tenors = [ (self.value_date - quote[1]).days for quote in fx_quotes]
        fx_prices = [ quote[2] for quote in fx_quotes]
        mode = cmq_curve.ForwardCurve.InterpMode.LinearLog
        self.fx_curve = cmq_curve.ForwardCurve.from_array(fx_tenors, fx_prices, interp_mode = mode)
        self.fx_fwd = self.fx_curve((self.value_date - self.end).days)
        if self.need_disc:
            self.df = disc_factor(self.value_date, self.end, market_data['IRCurve'][self.ccy.lower() + '_disc'])
        else:
            self.df = 1.0

    def clean_price(self):
        return (self.fx_fwd - self.strike)/self.fx_fwd * self.df
