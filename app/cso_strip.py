import datetime
import numpy as np
import misc
import cmq_market_data
import cmq_crv_defn
import cmq_book
from dateutil.relativedelta import relativedelta
import workdays

def generate_cso_strip(value_date, fwd_index, storage_cost, start_idx = 0, end_idx = 2):
    crv_info = cmq_crv_defn.COM_Curve_Map[fwd_index]
    hols = getattr(misc, crv_info['calendar'] + '_Holidays')
    market_date = workdays.workday(value_date, -1, hols)
    otype = 'P'
    accrual =  'act252'
    need_disc = True
    market_data = {'value_date': value_date, 'market_date': market_date, 'COMFwd': {}, \
                   'COMDV1': {}, 'COMDV2': {}, 'IRCurve': {}, }
    market_data['COMFwd'][fwd_index] = cmq_market_data.comfwd_db_loader(market_data, fwd_index)
    market_data['IRCurve']['usd_disc'] = cmq_market_data.ircurve_db_loader(market_data, 'usd_disc')
    if fwd_index == 'SGXIRO':
        market_data['COMDV1'][fwd_index] = [[datetime.date(2018, 6, 1), datetime.date(2018, 6, 30), 3.2],\
                                            [datetime.date(2018, 7, 1), datetime.date(2018, 7, 31), 3.0],\
                                            [datetime.date(2018, 8, 1),  datetime.date(2018, 8, 31), 2.8],\
                                            [datetime.date(2018, 9, 1),  datetime.date(2018, 9, 30), 2.6],\
                                            [datetime.date(2018, 10, 1),  datetime.date(2018, 10, 31), 2.4],\
                                            [datetime.date(2018, 11, 1),  datetime.date(2018, 11, 30), 2.2],\
                                            [datetime.date(2018, 12, 1),  datetime.date(2018, 12, 31), 2.0],\
                                            [datetime.date(2019, 1, 1),  datetime.date(2018, 1, 31), 1.8],\
                                            [datetime.date(2019, 2, 1),  datetime.date(2018, 2, 28), 1.6], \
                                            [datetime.date(2019, 3, 1), datetime.date(2018, 3, 31), 1.4], \
                                            [datetime.date(2019, 4, 1), datetime.date(2018, 4, 30), 1.2], \
                                            [datetime.date(2019, 5, 1), datetime.date(2018, 5, 31), 1.0], \
                                            [datetime.date(2019, 6, 1),   datetime.date(2019, 6, 30), 1.0]]
    else:
        market_data['COMDV1'][fwd_index] = [[datetime.date(2018,9,1), datetime.date(2018,9,15), 64.0], \
                                        [datetime.date(2019,1,1), datetime.date(2019,1,15), 32.0]]
    cont_list = [quote[0] for quote in market_data['COMFwd'][fwd_index]]
    test_book = cmq_book.CMQBook({})
    for mth in range(start_idx, end_idx):
        deal = cmq_book.CMQDeal({'positions': []})
        leg_a = cont_list[mth]
        leg_b = cont_list[mth + 1]
        expiry = workdays.workday(leg_a, -5, hols)
        trade_data = {'inst_type': "ComDVolCSO",
                    'strike': -storage_cost,
                    'fwd_index': fwd_index,
                    'leg_a': leg_a,
                    'leg_b': leg_b,
                    'leg_diff': 1,
                    'accrual': accrual,
                    'otype': otype,
                    'end': expiry,
                    'need_disc': need_disc}
        deal.add_instrument(trade_data, 1)
        test_book.book_deal(deal)
    return test_book, market_data


