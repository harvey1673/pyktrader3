# -*- coding:gbk -*-
import os
import csv
from sqlalchemy import create_engine
import datetime,time
import pandas as pd
from WindPy import w
from . wind_data_config import *
from pycmqlib3.utility.dbaccess import dbconfig, misc_dbconfig, connect, mysql_replace_into, prod_main_cont_exch
from pycmqlib3.utility.misc import contract_expiry, contract_range, process_min_id, inst2exch

wind_exch_map = {'SHF': 'SHFE', 'CZC': 'CZCE', 'DCE': 'DCE', 'CFE': 'CFFEX', "INE":"INE"}
w.start()

def load_symbol(symbol, fields, start_date, end_date, freq = 'd',
                  data_struct = {'oi': 'openInterest'}):
    field_list = fields.split(',')
    df = pd.DataFrame()
    if freq == 'd':
        func = w.wsd
        data_struct['Times'] ='date'
    elif freq == 'm':
        func = w.wsi
        data_struct['Times'] = 'datetime'
    else:
        print("unsupported freq, d or m")
        return df
    out = func(symbol, ','.join(field_list), start_date, end_date)
    if out.ErrorCode < 0:
        return df
    df['Times'] = out.Times
    for f, data in zip(field_list, out.Data):
        df[f] = data
    df.rename(columns = data_struct, inplace = True)
    df = df.dropna()
    return df

def save_hist_data(start_date, end_date, index_list = [], product_codes = [], spot_list = [],
                   fut_list = [], inv_list = [], min_bar = 0, flavor = 'mysql', adj_datetime = True):
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
    tday = datetime.date.today()
    for symbol, cmd_idx, desc in index_list:
        df = load_symbol(symbol, 'open,high,low,close,volume', start_date, end_date)
        if len(df)> 0:
            df['instID'] = cmd_idx
            exch = symbol.split('.')[-1]
            df['exch'] = exch
            print("saving daily data for instID = %s with number of data pts = %s" % (cmd_idx, len(df)))
            df.to_sql('fut_daily', con = conn, if_exists='append', index=False, method = func)
    exch2wind_dict = dict([(v, k) for k, v in wind_exch_map.items()])

    for prodcode in product_codes:
        cont_mth, exch = prod_main_cont_exch(prodcode)
        cont_list, _ = contract_range(prodcode, exch, cont_mth, start_date, end_date)
        exp_dates = [contract_expiry(cont) for cont in cont_list]
        for cont, exp in zip(cont_list, exp_dates):
            if exp >= start_date:
                ex = exch2wind_dict[exch]
                symbol = cont + '.' + ex
                if min_bar in [0, 2]:
                    ddf = load_symbol(symbol, 'open,high,low,close,volume,oi', max(exp - datetime.timedelta(days = 400), start_date), min(exp, tday), freq = 'd')
                    if len(ddf) > 0:
                        print("saving daily data for instID = %s with number of data pts = %s" % (cont, len(ddf)))
                        ddf['instID'] = cont
                        ddf['exch'] = exch
                        ddf.to_sql('fut_daily', con = conn, if_exists='append', index=False, method = func)
                if min_bar in [1, 2]:
                    s_time = max(exp - datetime.timedelta(days = 400), start_date - datetime.timedelta(days = 1))
                    e_time = min(exp, tday + datetime.timedelta(days = 1))
                    mdf = load_symbol(symbol, 'open,high,low,close,volume,oi', s_time, e_time, freq='m')
                    if len(mdf) > 0:
                        print("saving min data for instID = %s with number of data pts = %s" % (cont, len(mdf)))
                        mdf['instID'] = cont
                        mdf['exch'] = exch
                        mdf = process_min_id(mdf, adj_datetime)
                        mdf.to_sql('fut_min', con = conn, if_exists='append', index=False, method = func)

    for symbol, spotID, desc in spot_list:
        df = load_symbol(symbol, 'close', start_date, end_date)
        if len(df)> 0:
            df['spotID'] = spotID
            print("saving daily data for spotID = %s with number of data pts = %s" % (spotID, len(df)))
            df.to_sql('spot_daily', con = conn, if_exists='append', index=False, method = func)

    for cont in fut_list:
        exch = inst2exch(cont)
        ex = exch2wind_dict[exch]
        symbol = cont + '.' + ex
        ddf = load_symbol(symbol, 'open,high,low,close,volume,oi', start_date, end_date + datetime.timedelta(days = 1), freq='d')
        print("loading data from %s, with len = %s" % (cont, len(ddf)))
        if len(ddf) > 0:
            print("saving daily data for instID = %s with number of data pts = %s" % (cont, len(ddf)))
            ddf['instID'] = cont
            ddf['exch'] = exch
            ddf.to_sql('fut_daily', con = conn, if_exists='append', index=False, method = func)
        if min_bar:
            mdf = load_symbol(symbol, 'open,high,low,close,volume,oi', start_date, end_date, freq='m')
            if len(mdf) > 0:
                print("saving min data for instID = %s with number of data pts = %s" % (cont, len(mdf)))
                mdf['instID'] = cont
                mdf['exch'] = exch
                mdf = process_min_id(mdf, adj_datetime)
                mdf.to_sql('fut_min', con = conn, if_exists='append', index=False, method = func)

    for prod, symbol, invID, freq, unit, start_time in inv_list:
        conn = connect()
        if flavor == 'mysql':
            conn = create_engine('mysql+mysqlconnector://{user}:{passwd}@{host}/{dbase}'.format( \
                user=dbconfig['user'], \
                passwd=dbconfig['password'], \
                host=dbconfig['host'], \
                dbase=dbconfig['database']), echo=False)
            func = mysql_replace_into
        else:
            conn = connect(**misc_dbconfig)
            func = None
        df = load_symbol(symbol, 'close', start_date, end_date)
        if len(df)> 0:
            df['invID'] = invID
            df['prod'] = prod
            print("saving data for invID = %s with number of data pts = %s" % (invID, len(df)))
            df.to_sql('inv', con = conn, if_exists='append', index=False, method = func)

if __name__ == "__main__":
    pass
