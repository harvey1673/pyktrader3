import akshare as ak
import warnings
import datetime
import json
import pandas as pd
from sqlalchemy import create_engine
from pycmqlib3.utility.dbaccess import dbconfig, mysql_replace_into, connect
from pycmqlib3.utility.misc import CHN_Holidays, day_shift, is_workday, product_code, instID_adjust
from akshare.futures.cot import get_dce_rank_table, \
                                get_czce_rank_table, get_shfe_rank_table, get_cffex_rank_table
from akshare.futures.symbol_var import symbol_varieties
from akshare.futures import cons
chn_calendar = cons.get_calendar()

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
            print("exch = %s, date=%s" % (exch, end_date))
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

def update_rank_table(start_date = datetime.date.today(), end_date = datetime.date.today(), product_list = cons.contract_symbols, flavor = 'mysql'):    
    records = pd.DataFrame()
    dce_var = [i for i in product_list if i in cons.market_exchange_symbols['dce']]
    shfe_var = [i for i in product_list if i in cons.market_exchange_symbols['shfe']]
    czce_var = [i for i in product_list if i in cons.market_exchange_symbols['czce']]
    cffex_var = [i for i in product_list if i in cons.market_exchange_symbols['cffex']]
    while start_date <= end_date:
        big_dict = {}
        print(start_date)
        if start_date.strftime('%Y%m%d') in chn_calendar:
            adf = pd.DataFrame()            
            for exch, rank_table_func in [('DCE', get_dce_rank_table), ('SHFE', get_shfe_rank_table), ('INE', get_shfe_rank_table), \
                               ('CZCE', get_czce_rank_table), ('CFFEX', get_cffex_rank_table)]:
                var_list = [ prod.upper() for prod in product_code[exch]]
                data = rank_table_func(start_date, var_list)
                print(data)
                if data is not False:
                    big_dict.update(data)
                    for key in data:
                        data[key].rename(columns = {'variety': 'prod', 'var': 'prod',  'symbol': 'instID'}, inplace=True)
                        if 'instID' not in data[key].columns:
                            data[key]['instID'] = key
                        if exch in ['DCE', 'SHFE']:
                            data[key]['prod'] = data[key]['prod'].str.lower()
                            data[key]['instID'] = data[key]['instID'].str.lower()
                        data[key]['date'] = start_date                        
                        adf = adf.append(data[key])                        
            save_data('chn_fut_broker_rank', adf, flavor = flavor)
            for symbol, table in big_dict.items():
                table = table.applymap(lambda x: 0 if x == '' else x)
                for symbol_inner in set(table['instID']):
                    var = symbol_varieties(symbol_inner)
                    if var in product_list:
                        if var in czce_var:
                            for col in [item for item in table.columns if item.find('open_interest') > -1] + ['vol', 'vol_chg']:
                                table[col] = [float(value.replace(',', '')) if value != '-' else 0.0 for value in table[col]]

                        table_cut = table[table['instID'] == symbol_inner]
                        table_cut['rank'] = table_cut['rank'].astype('float')
                        table_cut_top5 = table_cut[table_cut['rank'] <= 5]
                        table_cut_top10 = table_cut[table_cut['rank'] <= 10]
                        table_cut_top15 = table_cut[table_cut['rank'] <= 15]
                        table_cut_top20 = table_cut[table_cut['rank'] <= 20]

                        big_dict = {'instID': symbol_inner,
                                    'prod': var,
                                    'vol_top5': table_cut_top5['vol'].sum(), 'vol_chg_top5': table_cut_top5['vol_chg'].sum(),
                                    'long_open_interest_top5': table_cut_top5['long_open_interest'].sum(),
                                    'long_open_interest_chg_top5': table_cut_top5['long_open_interest_chg'].sum(),
                                    'short_open_interest_top5': table_cut_top5['short_open_interest'].sum(),
                                    'short_open_interest_chg_top5': table_cut_top5['short_open_interest_chg'].sum(),

                                    'vol_top10': table_cut_top10['vol'].sum(),
                                    'vol_chg_top10': table_cut_top10['vol_chg'].sum(),
                                    'long_open_interest_top10': table_cut_top10['long_open_interest'].sum(),
                                    'long_open_interest_chg_top10': table_cut_top10['long_open_interest_chg'].sum(),
                                    'short_open_interest_top10': table_cut_top10['short_open_interest'].sum(),
                                    'short_open_interest_chg_top10': table_cut_top10['short_open_interest_chg'].sum(),

                                    'vol_top15': table_cut_top15['vol'].sum(),
                                    'vol_chg_top15': table_cut_top15['vol_chg'].sum(),
                                    'long_open_interest_top15': table_cut_top15['long_open_interest'].sum(),
                                    'long_open_interest_chg_top15': table_cut_top15['long_open_interest_chg'].sum(),
                                    'short_open_interest_top15': table_cut_top15['short_open_interest'].sum(),
                                    'short_open_interest_chg_top15': table_cut_top15['short_open_interest_chg'].sum(),

                                    'vol_top20': table_cut_top20['vol'].sum(),
                                    'vol_chg_top20': table_cut_top20['vol_chg'].sum(),
                                    'long_open_interest_top20': table_cut_top20['long_open_interest'].sum(),
                                    'long_open_interest_chg_top20': table_cut_top20['long_open_interest_chg'].sum(),
                                    'short_open_interest_top20': table_cut_top20['short_open_interest'].sum(),
                                    'short_open_interest_chg_top20': table_cut_top20['short_open_interest_chg'].sum(),
                                    'date': start_date
                                    }
                        records = records.append(pd.DataFrame(big_dict, index=[0]))

            if len(big_dict.items()) > 0:
                add_vars = [i for i in cons.market_exchange_symbols['dce'] + cons.market_exchange_symbols['shfe'] +
                            cons.market_exchange_symbols['cffex'] if
                            i in records['prod'].tolist()]
                for var in add_vars:
                    records_cut = records[records['prod'] == var]
                    var_record = pd.DataFrame(records_cut.sum()).T
                    var_record['date'] = start_date
                    var_record.loc[:, ['prod', 'instID']] = var
                    records = records.append(var_record)
        else:
            warnings.warn(f"{start_date.strftime('%Y%m%d')}非交易日")
        start_date += datetime.timedelta(days=1)
    return records.reset_index(drop=True)
