import akshare as ak
import warnings
import datetime
import json
import pandas as pd
import numpy as np
import itertools
from pycmqlib3.utility.dbaccess import save_data, get_fut_daily_from_db
from pycmqlib3.utility.misc import CHN_Holidays, day_shift, is_workday, product_code, instID_adjust, inst2product
from akshare.futures.cot import get_dce_rank_table, \
                                get_czce_rank_table, get_shfe_rank_table, get_cffex_rank_table
from akshare.futures import cons, symbol_var
chn_calendar = cons.get_calendar()


def generate_calendar_json(start_date, end_date, filename, date_for="%Y%m%d"):
    d = start_date
    dlist = []
    while d <= end_date:
        if is_workday(d, calendar='CHN'):
            dlist.append(datetime.datetime.strftime(d, date_for))
        d = d + datetime.timedelta(days=1)    
    outfile = open(filename, "w")
    outfile.write(json.dumps(dlist, indent=4))
    outfile.close()


def update_spot_daily(start_date=datetime.date.today(), end_date=datetime.date.today(), flavor='mysql'):
    dce_mkt = cons.market_exchange_symbols['dce']
    shfe_mkt = cons.market_exchange_symbols['shfe']
    gfex_mkt = cons.market_exchange_symbols['gfex']
    df = ak.futures_spot_price_daily(start_date, end_date)
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d").date())
    df['symbol'] = df['symbol'].apply(lambda x: x.lower() if x in dce_mkt + shfe_mkt + gfex_mkt else x)
    df['source'] = '100ppi'
    df['spotID'] = df[['symbol', 'source']].agg('_'.join, axis=1)
    df['close'] = df['spot_price']
    save_data('spot_daily', df[['spotID', 'date', 'close']], flavor=flavor)


def update_hist_fut_daily(start_date=datetime.date.today(),
                          end_date=datetime.date.today(),
                          exchanges=['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE', 'GFEX'],
                          flavor='mysql',
                          fut_table='fut_daily'):
    exl_list = []
    while start_date <= end_date:
        for exch in exchanges:
            print("exch = %s, date=%s" % (exch, end_date))
            df = ak.get_futures_daily(start_date=end_date, end_date=end_date, market=exch)
            if (df is not None) and (len(df) > 0):
                df = df[df['close'].apply(lambda x: pd.api.types.is_number(x))]
                df = df[df['open'].apply(lambda x: pd.api.types.is_number(x))]
                df = df[df['volume'].apply(lambda x: pd.api.types.is_number(x))]
                df = df[df['volume'] > 0]
                df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d").date())
                if exch in ['DCE', 'SHFE', 'INE', 'GFEX']:
                    df['symbol'] = df['symbol'].apply(lambda x: x.lower())
                    df['variety'] = df['variety'].apply(lambda x: x.lower())
                #if exch == 'SHFE':
                df = df[df['variety'].apply(lambda x: x in product_code[exch])]
                df['exch'] = exch
                df.rename(columns = {'symbol': 'instID', 'open_interest': 'openInterest'}, inplace=True)
                df['instID'] = df['instID'].apply(lambda x: instID_adjust(x, exch, end_date))
                xdf = df[['instID', 'exch', 'date', 'open', 'high', 'low', 'close', 'settle', 'volume', 'openInterest']]
                save_data(fut_table, xdf, flavor=flavor)
            else:
                print('no data for exch = %s, date = %s' % (exch, end_date))
                exl_list.append((exch, end_date))
                continue
        end_date = day_shift(end_date, '-1b', CHN_Holidays)
    return exl_list


def get_fut_daily_from_web(curr_date, exchanges=['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE', 'GFEX']):
    xdf = pd.DataFrame()
    for exch in exchanges:
        df = ak.get_futures_daily(start_date=curr_date, end_date=curr_date, market=exch)
        if (df is not None) and (len(df) > 0):
            df['exchg'] = exch
            df = df[df['close'].apply(lambda x: pd.api.types.is_number(x))]
            df = df[df['open'].apply(lambda x: pd.api.types.is_number(x))]
            df = df[df['volume'].apply(lambda x: pd.api.types.is_number(x))]
            df = df[(df['volume'] > 0) & (df['symbol'].str.len() < 8)]
            if exch in ['DCE', 'SHFE', 'INE', 'GFEX', ]:
                df['symbol'] = df['symbol'].apply(lambda x: x.lower())
                df['variety'] = df['variety'].apply(lambda x: x.lower())
            xdf = xdf.append(df)
    xdf['upperlimit'] = 0
    xdf['lowerlimit'] = 0
    xdf = xdf.rename(columns={'open_interest': 'openinterest', 'symbol': 'code', 'variety': 'product'})
    return xdf


def save_market_snapshot(start_date=datetime.date.today(),
                    end_date=datetime.date.today(),
                    exchanges=['DCE', 'SHFE', 'CZCE', 'CFFEX', 'INE', 'GFEX'],
                    folder='C:/dev/wtdev/storage/his/snapshot',
                    source='db'):
    cols = ['date', 'exchg', 'code', 'open', 'high', 'low', 'close', 'settle', 'volume', 'turnover',
            'openinterest', 'upperlimit', 'lowerlimit', 'preclose', 'presettle', 'preinterest']
    sdate = day_shift(start_date, '-1b', CHN_Holidays)
    while sdate <= end_date:
        print('updating for date=%s' % sdate.strftime("%Y%m%d"))
        if source == 'web':
            xdf = get_fut_daily_from_web(sdate, exchanges=exchanges)
        elif source == 'db':
            xdf = get_fut_daily_from_db(sdate, exchanges=exchanges)
        else:
            print('unsupported source %s, should be db or web.' % source)
            return
        prev_xdf = xdf[['code', 'close', 'settle', 'openinterest']].rename(
                            columns={'close': 'preclose',
                                     'settle': 'presettle',
                                     'openinterest': 'preinterest', })
        if sdate >= start_date:
            xdf = xdf.merge(prev_xdf, on=['code'], how='left')
            xdf[['preclose', 'presettle', 'preinterest']].fillna(0, inplace=True)
            sdate_str = sdate.strftime("%Y%m%d")
            xdf[cols].to_csv(f'{folder}/{sdate_str}.csv', index=False)
        sdate = day_shift(sdate, '1b', CHN_Holidays)


def update_sgx_daily(start_date=datetime.date.today(), end_date=datetime.date.today(),
                     flavor='mysql', freq=1, dbtable='fut_daily'):
    while start_date <= end_date:
        df = ak.futures_sgx_daily(trade_date = start_date.strftime("%Y-%m-%d"), recent_day = freq)
        save_data(dbtable, df, flavor=flavor)
        start_date = day_shift(start_date, str(freq) + 'b')


def update_rank_table(start_date=datetime.date.today(), end_date=datetime.date.today(),
                        exch_list=['DCE', 'CZCE', 'SHFE', 'CFFEX'], flavor='mysql'):
    var_dict = {}
    exch_func = {}
    product_list = []
    for exch in exch_list:
        var_dict[exch] = [i.upper() for i in cons.market_exchange_symbols[exch.lower()]]
        product_list = product_list + product_code[exch]
        if exch == 'DCE':
            exch_func[exch] = get_dce_rank_table
        elif exch == 'SHFE':
            exch_func[exch] = get_shfe_rank_table
        elif exch == 'CZCE':
            exch_func[exch] = get_czce_rank_table
        elif exch == 'CFFEX':
            exch_func[exch] = get_cffex_rank_table
        else:
            print("exch = %s is not supported, use shfe for ine" % (exch))
            return []
    col_list = ['date', 'prod', 'instID', 'ranknum', \
                'vol_party_name', 'vol', 'vol_chg', \
                'long_party_name', 'long_open_interest', 'long_open_interest_chg', \
                'short_party_name', 'short_open_interest', 'short_open_interest_chg', ]
    run_d = end_date
    excl_dates = []
    while run_d >= start_date:
        records = pd.DataFrame()
        big_dict = {}
        print(run_d)
        if run_d.strftime('%Y%m%d') in chn_calendar:
            adf = pd.DataFrame()

            for exch in exch_list:                
                data = exch_func[exch](run_d, var_dict[exch])
                if data is not False:
                    for key in data:
                        data[key].rename(columns = {'variety': 'prod', 'var': 'prod',  'symbol': 'instID', 'rank': 'ranknum'}, inplace=True)
                        data[key] = data[key].applymap(lambda x: 0 if x == '' else x)                        
                        if 'instID' not in data[key].columns:
                            data[key]['instID'] = key
                        if exch in ['DCE', 'SHFE']:
                            data[key]['prod'] = data[key]['prod'].str.lower()
                            data[key]['instID'] = data[key]['instID'].str.lower()
                        elif exch in ['CZCE']:
                            for col in [item for item in data[key].columns if item.find('open_interest') > -1] + ['vol', 'vol_chg']:
                                data[key][col] = [float(value.replace(',', '')) if value != '-' else 0.0 for value in data[key][col]]                            
                        data[key]['date'] = run_d  
                        data[key] = data[key][col_list]
                        adf = adf.append(data[key])
                    big_dict.update(data)
                else:
                    print("date=%s, exch=%s is missing data" % (str(run_d), exch))
                    excl_dates.append([run_d, exch])
            if len(adf) > 0:
                save_data('chn_fut_broker_rank', adf, flavor = flavor)
            else:
                print("date=%s, exch=%s datafram is empty" % (str(run_d), exch))
            for symbol, table in big_dict.items():
                for symbol_inner in set(table['instID']):
                    if symbol_inner in product_list:
                        var = symbol_inner
                    elif symbol_inner == "PTA":
                        var = "TA"
                    else:
                        var = inst2product(symbol_inner)
                    if var in product_list:
                        table_cut = table[table['instID'] == symbol_inner]
                        table_cut['ranknum'] = table_cut['ranknum'].astype('float')
                        table_cut_top5 = table_cut[table_cut['ranknum'] <= 5]
                        table_cut_top10 = table_cut[table_cut['ranknum'] <= 10]
                        table_cut_top15 = table_cut[table_cut['ranknum'] <= 15]
                        table_cut_top20 = table_cut[table_cut['ranknum'] <= 20]

                        big_dict = {'instID': symbol_inner,
                                    'prod': var,
                                    'vol_top5': int(table_cut_top5['vol'].sum()), 'vol_chg_top5': int(table_cut_top5['vol_chg'].sum()),
                                    'long_open_interest_top5': int(table_cut_top5['long_open_interest'].sum()),
                                    'long_open_interest_chg_top5': int(table_cut_top5['long_open_interest_chg'].sum()),
                                    'short_open_interest_top5': int(table_cut_top5['short_open_interest'].sum()),
                                    'short_open_interest_chg_top5': int(table_cut_top5['short_open_interest_chg'].sum()),

                                    'vol_top10': int(table_cut_top10['vol'].sum()),
                                    'vol_chg_top10': int(table_cut_top10['vol_chg'].sum()),
                                    'long_open_interest_top10': int(table_cut_top10['long_open_interest'].sum()),
                                    'long_open_interest_chg_top10': int(table_cut_top10['long_open_interest_chg'].sum()),
                                    'short_open_interest_top10': int(table_cut_top10['short_open_interest'].sum()),
                                    'short_open_interest_chg_top10': int(table_cut_top10['short_open_interest_chg'].sum()),

                                    'vol_top15': int(table_cut_top15['vol'].sum()),
                                    'vol_chg_top15': int(table_cut_top15['vol_chg'].sum()),
                                    'long_open_interest_top15': int(table_cut_top15['long_open_interest'].sum()),
                                    'long_open_interest_chg_top15': int(table_cut_top15['long_open_interest_chg'].sum()),
                                    'short_open_interest_top15': int(table_cut_top15['short_open_interest'].sum()),
                                    'short_open_interest_chg_top15': int(table_cut_top15['short_open_interest_chg'].sum()),

                                    'vol_top20': int(table_cut_top20['vol'].sum()),
                                    'vol_chg_top20': int(table_cut_top20['vol_chg'].sum()),
                                    'long_open_interest_top20': int(table_cut_top20['long_open_interest'].sum()),
                                    'long_open_interest_chg_top20': int(table_cut_top20['long_open_interest_chg'].sum()),
                                    'short_open_interest_top20': int(table_cut_top20['short_open_interest'].sum()),
                                    'short_open_interest_chg_top20': int(table_cut_top20['short_open_interest_chg'].sum()),
                                    'date': run_d
                                    }
                        records = records.append(pd.DataFrame(big_dict, index=[0]))

            if len(big_dict.items()) > 0:
                add_prods = [prod for prod in product_code['DCE'] + product_code['SHFE'] \
                    + product_code['INE'] + product_code['CFFEX'] if prod in records['prod'].tolist()]
                for prod in add_prods:
                    records_cut = records[records['prod'] == prod]                    
                    var_record = pd.DataFrame(records_cut.sum(numeric_only = True)).T                    
                    var_record['date'] = run_d
                    var_record.loc[:, 'prod'] = prod
                    var_record.loc[:, 'instID'] = prod
                    records = records.append(var_record)                
            records = records.reset_index(drop=True)
            ordered_cols = ['%s%s_%s' % (s2, s3, s1) for (s1, s2, s3) in itertools.product(['top5','top10','top15','top20'], ['vol','long_open_interest','short_open_interest'], ['','_chg'], )]
            for col in ordered_cols:
                records[col] = records[col].astype('int32')
            records = records[ ['date', 'prod', 'instID'] + ordered_cols]
            save_data('chn_broker_rank_sum', records, flavor = flavor)
        else:
            warnings.warn(f"{run_d.strftime('%Y%m%d')}非交易日")
        run_d -= datetime.timedelta(days=1)
    return excl_dates


def update_exch_receipt_table(start_date, end_date, flavor='mysql'):
    run_d = end_date
    excl_dates = []
    while run_d >= start_date:        
        if run_d.strftime('%Y%m%d') in chn_calendar:
            df = ak.get_receipt(start_day=run_d, end_day=run_d)
            if len(df) == 0:
                print("no data for %s" % str(run_d))
                excl_dates.append(run_d)
            else:
                df['receipt'] = df['receipt'].fillna(0)
                flag = df['receipt'].fillna(0).astype('str').str.isnumeric()
                df.loc[~flag, 'receipt'] = 0
                df['exch'] = df['var'].apply(lambda x: symbol_var.symbol_market(x).upper())
                df['prod'] = df['var']
                flag = df['exch'].isin(['DCE', 'SHFE', 'GFEX', 'INE'])
                df.loc[flag, 'prod'] = df.loc[flag, 'prod'].str.lower()
                flag = df['prod'].isin(product_code['INE'])
                df['exch'][flag] = 'INE'
                df['rcpt_label'] = df['prod']
                df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
                df['rcpt_num'] = df['receipt'].astype('int32')
                df = df[['date', 'exch', 'prod', 'rcpt_label', 'rcpt_num']]
                save_data('exch_receipts', df, flavor=flavor)
        else:
            warnings.warn(f"{run_d.strftime('%Y%m%d')}非交易日")
        run_d -= datetime.timedelta(days=1)
    return excl_dates


def update_exch_inv_table(start_date, end_date, flavor='mysql'):
    excl_dates = []
    s_date = end_date - datetime.timedelta(days = end_date.weekday())    
    while s_date >= start_date - datetime.timedelta(days = start_date.weekday()):
        e_date = s_date + datetime.timedelta(days = 4)
        df = ak.get_shfe_inv(start_day = s_date, end_day = e_date)
        if len(df) == 0:
            print("no data for %s - %s" % (str(s_date), str(e_date)))
            excl_dates.append((s_date, e_date))     
        else:
            df['inv_label'] = df['var_label']
            df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
            data_cols = ['spot_inventory', 'warrant_inventory', 'warehouse_stocks'] 
            for col in data_cols:
                df[col] = df[col].astype('int32')
            df['exch'] = 'SHFE'
            df['prod'] = df['var'].str.lower()
            flag = df['prod'].isin(product_code['INE'])
            df['exch'][flag] = 'INE'
            df = df[['date', 'exch', 'prod', 'inv_label'] + data_cols]
            print("save data for the week %s - %s" % (str(s_date), str(e_date)))
            save_data('exch_inventory', df, flavor = flavor)
        s_date -= datetime.timedelta(days=7)
    return excl_dates