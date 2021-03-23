import datetime
import numpy as np
import cmq_market_data
import cmq_book
from dateutil.relativedelta import relativedelta

def generate_ee_strip():
    market_date = datetime.date(2018,1, 2)
    value_date = datetime.date(2018, 1, 2)
    otype = 'P'
    strike = 50.0
    fwd_index = 'SGXIRO'
    accrual =  'act252'
    need_disc = True
    flat_vol = True
    start_cont = datetime.date(2018,3,1)
    market_data = {'value_date': value_date, 'market_date': market_date, 'COMFwd': {}, \
                   'COMVolATM': {}, 'COMVolV10': {}, 'COMVolV25': {}, 'COMVolV75': {}, 'COMVolV90': {}, \
                   'IRCurve': {}, }
    market_data['COMFwd'][fwd_index] = cmq_market_data.comfwd_db_loader(market_data, fwd_index)
    vol_dict = cmq_market_data.comvol_db_loader(market_data, fwd_index)
    back_vol = {'COMVolATM': 0.25, 'COMVolV10': 0.0,  'COMVolV25': 0.0, 'COMVolV75': 0.0, 'COMVolV90': 0.0}
    first_days = 365.0
    for vol_field in ['COMVolATM', 'COMVolV10', 'COMVolV25', 'COMVolV75', 'COMVolV90']:
        market_data[vol_field][fwd_index] = vol_dict[vol_field]
        last_q = market_data[vol_field][fwd_index][-1]
        for i in range(24):
            cont_mth = last_q[0] + relativedelta(months = i)
            cont_exp = cont_mth + relativedelta(months = 1) - datetime.timedelta(days = 1)
            curr_days = float((cont_exp - value_date).days)
            if i == 0:
                first_days = curr_days
            if flat_vol:
                curr_days = first_days
            market_data[vol_field][fwd_index].append([cont_mth, cont_exp, back_vol[vol_field] * np.sqrt(first_days/curr_days)])
    market_data['IRCurve']['usd_disc'] = cmq_market_data.ircurve_db_loader(market_data, 'usd_disc')
    model_settings = {'alpha': 1.8, 'beta': 1.2}
    test_book = cmq_book.CMQBook({})
    for mth in range(24):
        deal = cmq_book.CMQDeal({'positions': []})
        contract = start_cont + relativedelta(months=mth)
        expiry = contract - relativedelta(months=3)
        trade_data = {'inst_type': "ComEuroOption",
                    'strike': strike,
                    'fwd_index': fwd_index,
                    'contract': contract,
                    'accrual': accrual,
                    'otype': otype,
                    'end': expiry,
                    'need_disc': need_disc}
        deal.add_instrument(trade_data, 1)
        deal.positions[0][0].set_model_settings(model_settings)
        test_book.book_deal(deal)
    return test_book, market_data


