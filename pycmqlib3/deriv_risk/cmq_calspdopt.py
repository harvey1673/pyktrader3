import datetime
import copy
from . cmq_calspdswap import CMQCalSpdSwap
from . cmq_inst import disc_factor
from . import cmq_crv_defn
from pycmqlib3.analytics import exotic_opt

class CMQCalSpdOpt(CMQCalSpdSwap):
    class_params = dict(CMQCalSpdSwap.class_params, **{'otype': 'C',
                                                    'accrual': 'act252',
                                                    'expiry': datetime.date.today()})
    inst_key = ['fwd_index', 'otype', 'strike', 'expiry', 'start1', 'end1', 'start2', 'end2', 'ccy', 'volume']

    def __init__(self, trade_data, market_data = {}, model_settings = {}):
        super(CMQCalSpdOpt, self).__init__(trade_data, market_data, model_settings)

    def set_trade_data(self, trade_data):
        super(CMQCalSpdOpt, self).set_trade_data(trade_data)
        self.mkt_deps['COMVolATM'] = {}
        self.mkt_deps['COMVolATM'][self.fwd_index] = copy.deepcopy(self.mkt_deps['COMFwd'][self.fwd_index])

    def set_model_settings(self, model_settings):
        super(CMQCalSpdOpt, self).set_model_settings(model_settings)
        self.corr = model_settings.get('corr', 1.0)

    def set_market_data(self, market_data):
        super(CMQCalSpdOpt, self).set_market_data(market_data)
        if len(market_data) == 0:
            self.vol_a = 0.0
            self.vol_b = 0.0
            self.ir = 0.0
        else:
            self.vol_a = cmq_crv_defn.lookup_vol_mark(self.fwd_index, market_data, self.end1, \
                                                    vol_fields= ['COMVolATM'])['COMVolATM']
            self.vol_b = cmq_crv_defn.lookup_vol_mark(self.fwd_index, market_data, self.end2, \
                                                      vol_fields=['COMVolATM'])['COMVolATM']
            self.ir = disc_factor(self.value_date, self.end, market_data['IRCurve'][self.ccy.lower() + '_disc'],
                                  mode='IR')

    def clean_price(self):
        iscall = False
        if self.otype.lower in ['c', 'call',]:
            iscall = True
        fix_a = [[d, fix] for d, fix in zip(self.swap_a.fixing_dates, self.swap_a.past_fix + self.swap_a.fwd_fix)]
        fix_b = [[d, fix] for d, fix in zip(self.swap_b.fixing_dates, self.swap_b.past_fix + self.swap_b.fwd_fix)]
        res = exotic_opt.CalSpdAsianOption(iscall, self.swap_a.fwd_avg, self.swap_b.fwd_avg, self.strike,
                                     self.vol_a, self.vol_b, self.corr, fix_a, fix_b,
                                     self.value_date, self.expiry, self.ir,
                                     accr = self.accrual,
                                     eod_flag = self.eod_flag)
        return res

def test_case(tday = datetime.date.today() - datetime.timedelta(days = 1)):
    from . import cmq_book
    from . import cmq_market_data
    def create_test_book():
        fwd_index = 'SGXIRO'
        accrual = 'act252'
        need_disc = True
        otype = 'P'
        start1_dates = [datetime.date(2018, 12, 1), datetime.date(2018, 12, 1), datetime.date(2019, 1, 1)]
        start2_dates = [datetime.date(2019, 1, 1), datetime.date(2019, 2, 1), datetime.date(2019, 2, 1)]
        edate_dict = { datetime.date(2018, 12, 1): datetime.date(2018, 12,31),
                       datetime.date(2019, 1, 1): datetime.date(2019, 1, 31),
                       datetime.date(2019, 2, 1): datetime.date(2019, 2, 8),
                       }
        strat_data = [['SPD12', 'CALSPOPT_Z8F9'],
                      ['SPD13', 'CALSPOPT_Z8G9'],
                      ['SPD23', 'CALSPOPT_F9G9'],]

        expiry = datetime.date(2019,2,8)
        test_book = cmq_book.CMQBook({})
        for (sdate1, sdate2, deal_data) in zip(start1_dates, start2_dates, strat_data):
            deal_data = {'positions': [],
                         'strategy': deal_data[0],
                         'internal_id': deal_data[1],
                         'external_id': deal_data[1],
                         'product': fwd_index, }
            deal = cmq_book.CMQDeal(deal_data)
            trade_data = {'inst_type': "ComCalSpdOpt",
                          'strike': 0.0,
                          'fwd_index': fwd_index,
                          'accrual': accrual,
                          'otype': otype,
                          'start1': sdate1,
                          'end1': edate_dict[sdate1],
                          'start2': sdate2,
                          'end2': edate_dict[sdate2],
                          'expiry': expiry,
                          'need_disc': need_disc,
                          'volume': 1.0}
            deal.add_instrument(trade_data, 1)
            test_book.book_deal(deal)
        return test_book
    book = create_test_book()
    mkt_data = cmq_market_data.load_market_data(book.mkt_deps, value_date=tday, region='EOD', is_eod=True)
    return book, mkt_data
