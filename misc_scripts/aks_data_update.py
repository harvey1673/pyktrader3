import akshare as ak
import datetime
import json
import pandas as pd
from sqlalchemy import create_engine
from pycmqlib3.utility.dbaccess import dbconfig, mysql_replace_into, connect
from pycmqlib3.utility.misc import CHN_Holidays, day_shift, is_workday, product_code, instID_adjust

def generate_calendar_json(start_date, end_date, filename, hols = CHN_Holidays, date_for = "%Y%m%d"):
    d = start_date
    dlist = []
    while d <= end_date:
        if is_workday(d, calendar = 'CHN'):
            dlist.append(datetime.datetime.strftime(d, date_for))
        d = d + datetime.timedelta(days=1)    
    outfile = open(filename, "w")
    outfile.write(json.dumps(dlist, indent=4))
    outfile.close()
    
def save_data(dbtable, df, flavor = 'mysql'):
    if flavor == 'mysql':
        conn = create_engine('mysql+mysqlconnector://{user}:{passwd}@{host}/{dbase}'.format( \
                    user = dbconfig['user'], \
                    passwd = dbconfig['password'],\
                    host = dbconfig['host'],\
                    dbase = dbconfig['database']), echo=False)
        func = mysql_replace_into
    else:
        conn = connect(**dbconfig)
        func = None
    df.to_sql(dbtable, con = conn, if_exists='append', index=False, method=func)

def update_spot_daily(start_date = datetime.date.today(), end_date = datetime.date.today(), flavor = 'mysql'):
    dce_mkt = ak.futures.cons.market_exchange_symbols['dce']
    shfe_mkt = ak.futures.cons.market_exchange_symbols['shfe']
    df = ak.futures_spot_price_daily(start_date, end_date)
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d").date())
    df['symbol'] = df['symbol'].apply(lambda x: x.lower() if x in dce_mkt + shfe_mkt else x)
    df['source'] = '100ppi'
    df['spotID']=df[['symbol','source']].agg('_'.join,axis=1)
    df['close'] = df['spot_price']
    save_data('spot_daily', df[['spotID', 'date', 'close']], flavor = flavor)

def update_hist_fut_daily(start_date = datetime.date.today(), \
                          end_date = datetime.date.today(), \
                          exchanges = ['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE'], \
                          flavor = 'mysql', fut_table = 'fut_daily'):
    exl_list = []
    while start_date <= end_date:
        for exch in exchanges:
            df = ak.get_futures_daily(start_date=end_date, end_date=end_date, market = exch, index_bar=False)
            if (df is not None) and (len(df) > 0):
                df = df[df['close'].apply(lambda x: pd.api.types.is_number(x))]
                df = df[df['open'].apply(lambda x: pd.api.types.is_number(x))]
                df = df[df['volume'].apply(lambda x: pd.api.types.is_number(x))]
                df = df[df['volume'] > 0]
                df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d").date())
                if exch in ['DCE', 'SHFE', 'INE']:
                    df['symbol'] = df['symbol'].apply(lambda x: x.lower())
                    df['variety'] = df['variety'].apply(lambda x: x.lower())
                if exch == 'SHFE':
                    df = df[~df['variety'].apply(lambda x: x in product_code['INE'])]
                df['exch'] = exch
                df.rename(columns = {'symbol': 'instID', 'open_interest': 'openInterest'}, inplace=True)
                df['instID'] = df['instID'].apply(lambda x: instID_adjust(x, exch, end_date))
                xdf = df[['instID', 'exch', 'date', 'open', 'high', 'low', 'close', 'settle', 'volume', 'openInterest']]
                save_data(fut_table, xdf, flavor = flavor)
            else:
                print('no data for exch = %s, date = %s' % (exch, end_date))
                exl_list.append((exch, end_date))
                continue
        end_date = day_shift(end_date, '-1b', CHN_Holidays)
    return exl_list

def update_sgx_daily(start_date = datetime.date.today(), end_date = datetime.date.today(), flavor = 'mysql', freq = 1, dbtable = 'fut_daily'):
    exl_list = []
    while start_date <= end_date:
        df = ak.futures_sgx_daily(trade_date = start_date.strftime("%Y-%m-%d"), recent_day = freq)
        save_data(dbtable, df, flavor=flavor)
        start_date = day_shift(start_date, str(freq) + 'b')

