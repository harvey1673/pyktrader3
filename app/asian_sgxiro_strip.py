import datetime
import numpy as np
import cmq_market_data
import cmq_book
import cmq_risk_engine
import pandas as pd
from dateutil.relativedelta import relativedelta

def generate_strip(strike):
    market_input = 'C:\\Users\\H464717\\Documents\\jupyter\\data\\SGX_curves.csv'
    df = pd.read_csv(market_input, parse_dates = True).dropna()
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y").date())
    df['expiry'] = df['expiry'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y").date())
    market_date = datetime.date(2017,12, 28)
    value_date = datetime.date(2018, 1, 2)
    otype = 'P'
    fwd_index = 'SGXIRO'
    accrual =  'act252'
    need_disc = True
    start_cont = datetime.date(2018,3,1)
    market_data = {'value_date': value_date, 'market_date': market_date, 'COMFwd': {}, \
                   'COMVolATM': {}, 'COMVolV10': {}, 'COMVolV25': {}, 'COMVolV75': {}, 'COMVolV90': {}, \
                   'IRCurve': {}, }

    for field in ['COMFwd', 'COMVolATM', 'COMVolV10', 'COMVolV25', 'COMVolV75', 'COMVolV90']:
        market_data[field][fwd_index] = df[['date', 'expiry', field]].values.tolist()
    market_data['IRCurve']['usd_disc'] = cmq_market_data.ircurve_db_loader(market_data, 'usd_disc')
    test_book = cmq_book.CMQBook({})
    for mth in range(12):
        deal = cmq_book.CMQDeal({'positions': []})
        start = start_cont + relativedelta(months=mth)
        end = start + relativedelta(months=1, days=-1)
        trade_data = {'inst_type': "ComMthAsian",
                    'strike': strike,
                    'fwd_index': fwd_index,
                    'accrual': accrual,
                    'otype': otype,
                    'start': start,
                    'end': end,
                    'need_disc': need_disc,
                    'volume': 1.0}
        deal.add_instrument(trade_data, 1)
        test_book.book_deal(deal)
    return test_book, market_data

def run_scenarios(book_obj, mkt_data, risks):
    df_list = []
    for strike in [50.0, 55.0, 60.0]:
        book_obj, mkt_data = generate_strip(strike)
        re = cmq_risk_engine.CMQRiskEngine(book_obj, mkt_data, ['pv'])
        re.run_risks()
        df = pd.DataFrame.from_dict(re.deal_risks, orient='index')
        df = df.reset_index()
        df_list.append(df['pv'])
    xdf = pd.concat(df_list, axis = 1)
    return xdf


