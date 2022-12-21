# -*- coding: utf-8 -*-
import datetime
import numpy as np
import mysql.connector as sqlconn
from sqlalchemy import create_engine
import copy
import csv
import os.path
import pandas as pd
from . import sec_bits, misc
from pycmqlib3.core import agent_conf
USE_DB_TYPE = 'mysql'
DB_QUOTE = "%s"

dbconfig = sec_bits.dbconfig
hist_dbconfig = sec_bits.hist_dbconfig
bktest_dbconfig = sec_bits.bktest_dbconfig
misc_dbconfig = sec_bits.misc_dbconfig

#dbconfig = {'database': "C:\\dev\\pycmqlib\\data\\market_data.db"}
#hist_dbconfig = {'database': "C:\\dev\\pycmqlib\\data\\market_data.db"}
#bktest_dbconfig = {'database': "C:\\dev\\pycmqlib\\data\\bktest_data.db"}
#misc_dbconfig = {'database': "C:\\dev\\pycmqlib\\data\\misc_data.db"}

trade_dbconfig = {'database': 'C:\\dev\\pycmqlib\\data\\deal_data.db'}
mktsnap_dbconfig = {'database': "C:\\dev\\pycmqlib\\data\\market_snapshot.db"}
pos_dbconfig = {'database': "C:\\dev\\pycmqlib\\data\\hist_position.db"}

fut_tick_columns = ['instID', 'date', 'tick_id', 'hour', 'min', 'sec', 'msec', 'openInterest', 'volume', 'price',
                    'high', 'low', 'bidPrice1', 'bidVol1', 'askPrice1', 'askVol1']
ss_tick_columns = ['instID', 'date', 'tick_id', 'hour', 'min', 'sec', 'msec', 'openInterest', 'volume', 'price', 'high',
                   'low', 'bidPrice1', 'bidVol1', 'askPrice1', 'askVol1']
min_columns = ['datetime', 'date', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'min_id']
daily_columns = ['date', 'open', 'high', 'low', 'close', 'settle', 'volume', 'openInterest']
fx_columns = ['date', 'tenor', 'rate']
ir_columns = ['date', 'tenor', 'rate']
spot_columns = ['date', 'close']
vol_columns = ['date', 'expiry', 'atm', 'v90', 'v75', 'v25', 'v10']
cmvol_columns = ['date', 'tenor_label', 'expiry_date', 'delta', 'vol']
cmdv_columns = ['date', 'expiry', 'vol']
price_fields = { 'instID': daily_columns, 'spotID': spot_columns, 'vol_index': vol_columns, 'cmvol': cmvol_columns, \
                 'cmdv': cmdv_columns, 'ccy': fx_columns, 'ir_index': ir_columns, }
deal_columns = ['status', 'internal_id', 'external_id', 'cpty', 'positions', \
                'strategy', 'book', 'external_src', 'last_updated', \
                'trader', 'sales', 'desk', 'business', 'portfolio', 'premium', 'product', 'reporting_ccy', \
                'enter_date', 'last_date', 'commission', 'day1_comments']


def get_proxy_server():
    user = sec_bits.PROXY_CREDENTIALS['user']
    passwd = sec_bits.PROXY_CREDENTIALS['passwd']
    server_dict = {'http':'http://%s:%s@10.252.22.102:4200' % (user, passwd),
                'https':'https://%s:%s@10.252.22.102:4200' % (user, passwd)}
    return server_dict


def connect(**args):
    conn = sqlconn.connect(**args)
    if USE_DB_TYPE == 'sqlite3':
        conn.text_factory = sqlconn.OptimizedUnicode
    return conn


def mysql_replace_into(table, conn, keys, data_iter):
    from sqlalchemy.dialects.mysql import insert
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql.expression import Insert

    @compiles(Insert)
    def replace_string(insert, compiler, **kw):
        s = compiler.visit_insert(insert, **kw)
        s = s.replace("INSERT INTO", "REPLACE INTO")
        return s

    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(table.table.insert(replace_string=""), data)


tick_field_rename_map = {
    'bid_price1': 'bidPrice1',
    'bid_vol1': 'bidVol1',
    'ask_price1': 'askPrice1',
    'ask_vol1': 'askVol1',
    'bid_price2': 'bidPrice2',
    'bid_vol2': 'bidVol2',
    'ask_price2': 'askPrice2',
    'ask_vol2': 'askVol2',
    'bid_price3': 'bidPrice3',
    'bid_vol3': 'bidVol3',
    'ask_price3': 'askPrice3',
    'ask_vol3': 'askVol3',
    'bid_price4': 'bidPrice4',
    'bid_vol4': 'bidVol4',
    'ask_price4': 'askPrice4',
    'ask_vol4': 'askVol4',
    'bid_price5': 'bidPrice5',
    'bid_vol5': 'bidVol5',
    'ask_price5': 'askPrice5',
    'ask_vol5': 'askVol5',
}


def tick2dict(tick):
    tick_dict = {}
    for idx, field in enumerate(agent_conf.tick_data_list):
        afield = tick_field_rename_map.get(field, field)
        tick_dict[afield] = tick[idx].astype(object)
    #tick_dict['timestamp'] = tick_dict['timestamp'].astype(object)
    tick_dict['hour'] = tick_dict['timestamp'].hour
    tick_dict['min'] = tick_dict['timestamp'].minute
    tick_dict['sec'] = tick_dict['timestamp'].second
    tick_dict['msec'] = tick_dict['timestamp'].microsecond / 1000
    tick_dict['date'] =  tick_dict['timestamp'].date()
    return tick_dict


def insert_tick_data(cnx, inst, tick, dbtable='fut_tick'):
    tick_columns = fut_tick_columns
    if inst.isdigit():
        tick_columns = ss_tick_columns
    cursor = cnx.cursor()
    value_format = ",".join([DB_QUOTE] * len(tick_columns))
    stmt = "INSERT IGNORE INTO {table} ({variables}) VALUES ({value_format})".format(
        table=dbtable, variables=','.join(tick_columns), value_format = value_format)
    tick_dict = tick2dict(tick)
    tick_dict['instID'] = inst
    args = tuple([tick_dict[col] for col in tick_columns])
    cursor.execute(stmt, args)
    cnx.commit()


def bulkinsert_tick_data(cnx, inst, tick_data, dbtable='fut_tick'):
    if len(tick_data) == 0:
        return
    tick_columns = fut_tick_columns
    if inst.isdigit():
        tick_columns = ss_tick_columns
    cursor = cnx.cursor()
    value_format = ",".join([DB_QUOTE] * len(tick_columns))
    stmt = "INSERT IGNORE INTO {table} ({variables}) VALUES ({value_format})".format(
        table=dbtable, variables=','.join(tick_columns), value_format = value_format)
    args = []
    for n in range(len(tick_data)):
        tick_dict = tick2dict(tick_data[n])
        tick_dict['instID'] = inst
        args.append(tuple([tick_dict[col] for col in tick_columns]))
    # args = [tuple([getattr(tick,col) for col in tick_columns]) for tick in ticks]
    cursor.executemany(stmt, args)
    cnx.commit()


def insert_min_data(cnx, inst, min_data, dbtable='fut_min', option='IGNORE'):
    cursor = cnx.cursor()
    exch = misc.inst2exch(inst)
    if (min_data['high'] == min_data['low']) and (min_data['volume'] == 0):
        return
    min_data['date'] = min_data['datetime'].date()
    value_format = ",".join([DB_QUOTE] * (len(min_columns) + 2))
    stmt = "INSERT {opt} INTO {table} (instID,exch,{variables}) VALUES ({value_format})".format(
        opt=option, table=dbtable, variables=','.join(min_columns), value_format = value_format)
    args = tuple([inst, exch] + [min_data[col] for col in min_columns])
    cursor.execute(stmt, args)
    cnx.commit()


def bulkinsert_min_data(cnx, inst, exch, rec_array, ldate, dbtable='fut_min', is_replace=False):
    cursor = cnx.cursor()
    if is_replace:
        if USE_DB_TYPE == 'mysql':
            cmd = "REPLACE"
        else:
            cmd = "INSERT OR REPLACE"
    else:
        if USE_DB_TYPE == 'mysql':
            cmd = "INSERT IGNORE"
        else:
            cmd = "INSERT OR IGNORE"
    all_min_columns = ['instID', 'exch'] + min_columns
    value_format = ",".join([DB_QUOTE] * len(all_min_columns))
    stmt = "{cmd} INTO {table} ({variables}) VALUES ({value_format})".format(\
        cmd=cmd, table=dbtable, variables=','.join(all_min_columns), value_format = value_format)
    args = []
    for idx in range(len(rec_array), 0, -1):
        if rec_array['date'][idx-1] < ldate:
            break
        if (rec_array['high'][idx-1] == rec_array['low'][idx-1]) and (rec_array['volume'][idx-1]==0):
            continue
        data_list = [inst, exch]
        for col in min_columns:
            if col in ['datetime']:
                data = rec_array[col][idx - 1].astype('M8[ms]').astype('O')
            elif col in ['date']:
                data = rec_array[col][idx - 1]
            else:
                data = rec_array[col][idx-1].item()
            data_list.append(data)
        args.append(tuple(data_list))
    if len(args) > 0:
        cursor.executemany(stmt, args)
    cnx.commit()


def bulkinsert_daily_data(cnx, inst, exch, rec_array, ldate, dbtable='fut_daily', is_replace=False):
    cursor = cnx.cursor()
    if is_replace:
        if USE_DB_TYPE == 'mysql':
            cmd = "REPLACE"
        else:
            cmd = "INSERT OR REPLACE"
    else:
        if USE_DB_TYPE == 'mysql':
            cmd = "INSERT IGNORE"
        else:
            cmd = "INSERT OR IGNORE"
    all_daily_columns = ['instID', 'exch'] + daily_columns
    value_format = ",".join([DB_QUOTE] * len(all_daily_columns))
    stmt = "{cmd} INTO {table} ({variables}) VALUES ({value_format})".format(\
        cmd=cmd, table=dbtable, variables=','.join(all_daily_columns), value_format = value_format)
    args = []
    for idx in range(len(rec_array), 0, -1):
        if rec_array['date'][idx-1] < ldate:
            break
        args.append(tuple([inst, exch]+[rec_array[col][idx-1] if col in ['date'] else rec_array[col][idx-1].item()
                           for col in daily_columns]))
    if len(args) > 0:
        cursor.executemany(stmt, args)
    cnx.commit()


def bulkinsert_spot_data(cnx, df, is_replace = True, dbtable = 'spot_daily'):
    cursor = cnx.cursor()
    col_list = df.columns
    if is_replace:
        if USE_DB_TYPE == 'mysql':
            cmd = "REPLACE"
        else:
            cmd = "INSERT OR REPLACE"
    else:
        if USE_DB_TYPE == 'mysql':
            cmd = "INSERT IGNORE"
        else:
            cmd = "INSERT OR IGNORE"
    value_format = ",".join([DB_QUOTE] * len(col_list))
    stmt = "{cmd} INTO {table} ({variables}) VALUES ({value_format})".format(\
        cmd=cmd, table=dbtable, variables=','.join(col_list), value_format = value_format)
    args = df.to_records(index=False)
    if len(args) > 0:
        cursor.executemany(stmt, args)
    cnx.commit()


def insert_daily_data(cnx, inst, daily_data, is_replace=False, dbtable='fut_daily'):
    cursor = cnx.cursor()
    col_list = list(daily_data.keys())
    exch = misc.inst2exch(inst)
    if is_replace:
        if USE_DB_TYPE == 'mysql':
            cmd = "REPLACE"
        else:
            cmd = "INSERT OR REPLACE"
    else:
        if USE_DB_TYPE == 'mysql':
            cmd = "INSERT IGNORE"
        else:
            cmd = "INSERT OR IGNORE"
    value_format = ",".join([DB_QUOTE] * (len(col_list)+2))
    stmt = "{commd} INTO {table} (instID,exch,{variables}) VALUES ({value_format})".format(\
                    commd=cmd, table=dbtable, variables=','.join(col_list), \
                    value_format = value_format)
    args = tuple([inst, exch] + [daily_data[col] for col in col_list])
    try:
        cursor.execute(stmt, args)
        cnx.commit()
    except:
        print([inst, exch] + [(daily_data[col], type(daily_data[col])) for col in col_list])


def insert_row_by_dict(cnx, dbtable, rowdict, is_replace=False):
    cursor = cnx.cursor()
    if USE_DB_TYPE in ['sqlite3']:
        cmd = "PRAGMA table_info(%s)"
        idx = 1
    else:
        cmd = "describe " + DB_QUOTE
        idx = 0
    cursor.execute(cmd % dbtable)
    allowed_keys = set(row[idx] for row in cursor.fetchall())
    keys = allowed_keys.intersection(rowdict)
    columns = ",".join(keys)
    values_template = ",".join([DB_QUOTE] * len(keys))
    if is_replace:
        if USE_DB_TYPE == 'mysql':
            cmd = "REPLACE"
        else:
            cmd = "INSERT OR REPLACE"
    else:
        if USE_DB_TYPE == 'mysql':
            cmd = "INSERT IGNORE"
        else:
            cmd = "INSERT OR IGNORE"
    stmt = "{cmd} into {table} ({variables}) values ({format})".format(cmd = cmd, table = dbtable, variables = columns, \
                                                                       format = values_template)
    values = tuple(float(rowdict[key]) if type(rowdict[key]).__name__ in 'float64' else rowdict[key] for key in keys)
    cursor.execute(stmt, values)
    cnx.commit()


def import_tick_from_file(dbtable, conn = None):
    inst_list = ['IF1406', 'IO1406-C-2300', 'IO1406-P-2300', 'IO1406-C-2250',
                 'IO1406-P-2250', 'IO1406-C-2200', 'IO1406-P-2200', 'IO1406-C-2150',
                 'IO1406-P-2150', 'IO1406-C-2100', 'IO1406-P-2100', 'IO1406-C-2050',
                 'IO1406-P-2050', 'IO1406-C-2000', 'IO1406-P-2000', 'IO1407-C-2300',
                 'IO1407-P-2300', 'IO1407-C-2250', 'IO1407-P-2250', 'IO1407-C-2200',
                 'IO1407-P-2200', 'IO1407-C-2150', 'IO1407-P-2150', 'IO1407-C-2100',
                 'IO1407-P-2100', 'IO1407-C-2050', 'IO1407-P-2050', 'IO1407-C-2000',
                 'IO1407-P-2000', 'IF1406']
    date_list = ['20140603', '20140604', '20140605', '20140606']
    main_path = 'C:/dev/data/'
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    for inst in inst_list:
        for date in date_list:
            path = main_path + inst + '/' + date + '_tick.txt'
            if os.path.isfile(path):
                stmt = "load data infile '{path}' replace into table {table} fields terminated by ',' lines terminated by '\n' (instID, date, @var1, sec, msec, openInterest, volume, price, high, low, bidPrice1, bidVol1, askPrice1, askVol1) set hour=(@var1 div 100), min=(@var1 % 100)".format(
                    path=path, table=dbtable)
                cursor.execute(stmt)
                cnx.commit()
    if conn == None:
        cnx.close()


def insert_cont_data(cont, conn = None, is_replace = True):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    col_list = list(cont.keys())
    value_format = ",".join([DB_QUOTE] * len(col_list))
    if is_replace:
        if USE_DB_TYPE == 'mysql':
            cmd = "REPLACE"
        else:
            cmd = "INSERT OR REPLACE"
    else:
        if USE_DB_TYPE == 'mysql':
            cmd = "INSERT IGNORE"
        else:
            cmd = "INSERT OR IGNORE"
    stmt = "{cmd} INTO {table} ({variables}) VALUES ({value_format}) ".format(cmd = cmd, table='contract_list', \
            variables=','.join(col_list), value_format = value_format)
    args = tuple([cont[col] for col in col_list])
    cursor.execute(stmt, args)
    cnx.commit()
    if conn == None:
        cnx.close()


def prod_main_cont_exch(prodcode, conn = None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    stmt = "select exchange, contract from trade_products where product_code='{prod}' ".format(prod=prodcode)
    cursor.execute(stmt)
    out = [(exchange, contract) for (exchange, contract) in cursor]
    exch = str(out[0][0])
    cont = str(out[0][1])
    cont_mth = [misc.month_code_map[c] for c in cont]
    if conn == None:
        cnx.close()
    return cont_mth, exch


def load_product_info(prod, conn = None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    stmt = "select exchange, lot_size, tick_size, start_min, end_min, broker_fee from trade_products where product_code='{product}' ".format(
        product=prod)
    cursor.execute(stmt)
    out = {}
    for (exchange, lot_size, tick_size, start_min, end_min, broker_fee) in cursor:
        out = {'exch': str(exchange),
               'lot_size': lot_size,
               'tick_size': float(tick_size),
               'start_min': start_min,
               'end_min': end_min,
               'broker_fee': float(broker_fee)
               }
    if conn == None:
        cnx.close()
    return out


def load_cont_by_prod(prod, conn=None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    exch = misc.prod2exch(prod)
    qry_key = f'{prod}____'
    if exch in ['CZCE']:
        qry_key = f'{prod}%'
    stmt = f"select distinct(instID) from fut_daily where instID like '{qry_key}' and exch='{exch}'"
    cursor.execute(stmt)
    out = []
    for (instID,) in cursor:
        out.append(instID)
    return out


def load_stockopt_info(inst, conn = None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    stmt = "select underlying, opt_mth, otype, exchange, strike, strike_scale, lot_size, tick_base from stock_opt_map where instID='{product}' ".format(
        product=inst)
    cursor.execute(stmt)
    out = {}
    for (underlying, opt_mth, otype, exchange, strike, strike_scale, lot_size, tick_size) in cursor:
        out = {'exch': str(exchange),
               'lot_size': int(lot_size),
               'tick_size': float(tick_size),
               'strike': float(strike) / float(strike_scale),
               'cont_mth': opt_mth,
               'otype': str(otype),
               'underlying': str(underlying)
               }
    if conn == None:
        cnx.close()
    return out


def get_stockopt_map(underlying, cont_mths, strikes, conn = None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    stmt = "select underlying, opt_mth, otype, strike, strike_scale, instID from stock_opt_map where underlying='{under}' and opt_mth in ({opt_mth_str}) and strike in ({strikes}) ".format(
        under=underlying,
        opt_mth_str=','.join([str(mth) for mth in cont_mths]), strikes=','.join([str(s) for s in strikes]))
    cursor.execute(stmt)
    out = {}
    for (underlying, opt_mth, otype, strike, strike_scale, instID) in cursor:
        key = (str(underlying), int(opt_mth), str(otype), float(strike) / float(strike_scale))
        out[key] = instID
    if conn == None:
        cnx.close()
    return out


def update_contract_list_table(sdate, exch = ['DCE', 'CZCE', 'SHFE', 'CFFEX', 'INE', 'GFEX'], default_margin = 0.08):
    conn = connect(**dbconfig)
    product_table = pd.read_sql("select * from trade_products where is_active=1", conn)
    product_table = product_table[product_table.exchange.isin(exch)]
    for pc, ex, list_cont in zip(product_table['product_code'], \
                                 product_table['exchange'], product_table['list_cont']):
        cont_mth = [misc.month_code_map[c] for c in list_cont]
        cont_list, tenor_list = misc.contract_range(pc, ex, cont_mth, sdate, sdate)
        for cont, tenor in zip(cont_list, tenor_list):
            exp = misc.cont_date_expiry(tenor, pc, ex)
            if (exp <= misc.day_shift(sdate, '2y')) and (exp >= sdate):
                cont_data = {}
                cont_data['instID'] = cont
                cont_data['start_date'] = misc.day_shift(exp, '-1y')
                cont_data['expiry'] = exp
                cont_data['product_code'] = pc
                cont_data['margin_l'] = default_margin
                cont_data['margin_s'] = default_margin
                insert_cont_data(cont_data, conn, is_replace = False)
    conn.close()


def load_alive_cont(sdate, conn = None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    stmt = "select instID, product_code from contract_list where expiry >= " + DB_QUOTE
    args = tuple([sdate])
    cursor.execute(stmt, args)
    cont = []
    pc = []
    for line in cursor:
        cont.append(str(line[0]))
        prod = str(line[1])
        if prod not in pc:
            pc.append(prod)
    if conn == None:
        cnx.close()
    return cont, pc


def load_inst_marginrate(instID, conn = None):
    if conn == None:
        cnx = connect(**dbconfig)
    else:
        cnx = conn
    cursor = cnx.cursor()
    stmt = "select margin_l, margin_s from contract_list where instID='{inst}' ".format(inst=instID)
    cursor.execute(stmt)
    out = (0, 0)
    for (margin_l, margin_s) in cursor:
        out = (float(margin_l), float(margin_s))
    if conn == None:
        cnx.close()
    return out


def load_min_data_to_df(cnx, dbtable, inst, d_start = None, d_end = None, minid_start=1, minid_end=2359, \
                        index_col='datetime', fields = 'open,high,low,close,volume,openInterest', shift_datetime = False):
    inst_list = inst.split(',')
    field_list = ['instID', 'exch', 'datetime', 'date', 'min_id'] + fields.split(',')
    stmt = "select {variables} from {table} where instID in ('{instID}') ".format(variables=','.join(field_list),
                                                                             table=dbtable, instID="','".join(inst_list))
    if minid_start:
        stmt = stmt + "and min_id >= %s " % minid_start
    if minid_end:
        stmt = stmt + "and min_id <= %s " % minid_end
    if d_start:
        stmt = stmt + "and date >= '%s' " % d_start.strftime('%Y-%m-%d')
    if d_end:
        stmt = stmt + "and date <= '%s' " % d_end.strftime('%Y-%m-%d')
    stmt = stmt + "order by instID, date, min_id"
    df = pd.io.sql.read_sql(stmt, cnx)
    col_name = 'datetime'
    if (len(df) > 0):
        if (isinstance(df[col_name][0], str)):
            if shift_datetime:
                df[col_name] = df.apply(lambda x: datetime.datetime.strptime(x['date'] + x['datetime'][-9:], "%Y-%m-%d %H:%M:%S"),1)
            else:
                df[col_name] = df[col_name].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
        elif shift_datetime:
            df[col_name] = df.apply(lambda x: datetime.datetime.combine(x['date'], x['datetime'].time()), 1)
    if index_col:
        df = df.set_index(index_col)
    return df


def load_daily_data_to_df(cnx, dbtable, inst, d_start, d_end, index_col='date', field = 'instID', date_as_str = False):
    if 'fut_daily' in dbtable:
        inst_field = 'instID'
    elif 'spot_daily' in dbtable:
        inst_field = 'spotID'
    elif 'fx_daily' in dbtable:
        inst_field = 'ccy'
    elif 'ir_daily' in dbtable:
        inst_field = 'ir_index'
    else:
        print("unknown =")
    stmt = "select {variables} from {table} where {field} like '{instID}' ".format( \
                                    variables=','.join(price_fields[field]),
                                    table=dbtable, field = inst_field, instID=inst)
    stmt = stmt + "and date >= '%s' " % d_start.strftime('%Y-%m-%d')
    stmt = stmt + "and date <= '%s' " % d_end.strftime('%Y-%m-%d')
    stmt = stmt + "order by date"
    df = pd.io.sql.read_sql(stmt, cnx)
    col_name = 'date'
    if (len(df) > 0) and (isinstance(df[col_name][0], str)) and (date_as_str == False):
        df[col_name] = df[col_name].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
    if index_col:
        df = df.set_index(index_col)
    return df


def query_data_to_df(cnx, dbtable, filter, fields):
    stmt = "select {variables} from {table} where {filter} order by date".format(variables = fields,
                                                                                 table = dbtable,
                                                                                 filter = filter)
    df = pd.io.sql.read_sql(stmt, cnx)
    return df


def load_fut_curve(cnx, prod_code, ref_date, dbtable = 'fut_daily', field = 'instID'):
    qry_str = '%'
    stmt = "select {variables} from {table} where {field} like '{prod}{qry}' ".format( \
                                    variables=','.join([field] + price_fields[field]), qry = qry_str,
                                    table=dbtable, field = field, prod = prod_code)
    stmt = stmt + "and date like '{refdate}%' ".format(refdate = ref_date)
    stmt = stmt + "order by {field}".format(field = field)
    df = pd.io.sql.read_sql(stmt, cnx)
    if field == 'instID':
        df['prod_code'] = df['instID'].apply(lambda x: misc.inst2product(x))
        df = df[df['prod_code'] == prod_code]
    return df


def load_deal_data(cnx, dbtable = 'deal', book = 'BOF', strategy = '', deal_status = [2]):
    stmt = "select {variables} from {table} where book like '{book}' and strategy like '{strat}' ".format(\
                        table=dbtable, variables=','.join(deal_columns), \
                        book = book if len(book)> 0 else '%', \
                        strat = strategy if len(strategy)>0 else '%')
    if len(deal_status) == 1:
        stmt = stmt + "and status = {deal_status} ".format(deal_status = deal_status[0])
    else:
        stmt = stmt + "and status in {deal_status} ".format(deal_status = tuple(deal_status))
    stmt = stmt + "order by internal_id"
    df = pd.io.sql.read_sql(stmt, cnx)
    return df


def load_cmvol_curve(cnx, prod_code, ref_date, dbtable = 'cmvol_daily', field = 'cmvol'):
    stmt = "select {variables} from {table} where product_code like '{prod}%' ".format( \
                                    variables=','.join(price_fields[field]),
                                    table = dbtable, prod = prod_code)
    stmt = stmt + "and date like '{refdate}%' ".format(refdate = ref_date)
    stmt = stmt + "order by expiry_date".format(field = field)
    df = pd.io.sql.read_sql(stmt, cnx)
    for col in ['tenor_label','expiry_date']:
        df[col] = df[col].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
    if len(df) > 0:
        df['delta'] = ((df['delta'].astype(float)+1)*100).astype(int) % 100
        vol_tbl = df.pivot_table(columns = ['delta'], index = ['tenor_label', 'expiry_date'], values = ['vol'], aggfunc = np.mean)
        atm_delta = 50
        for delta in [10, 25, 75, 90]:
            vol_tbl[('vol', delta)] = vol_tbl[('vol', delta)] - vol_tbl[('vol', atm_delta)]
    else:
        vol_tbl = pd.DataFrame()
    vol_tbl = vol_tbl.reset_index()
    vol_tbl.columns = [''.join([str(e) for e in col]).strip() for col in vol_tbl.columns.values]
    vol_tbl.rename(columns={'vol10': 'COMVolV10', \
                            'vol25': 'COMVolV25', \
                            'vol50': 'COMVolATM', \
                            'vol75': 'COMVolV75', \
                            'vol90': 'COMVolV90', }, inplace=True)
    return vol_tbl


def load_cmdv_curve(cnx, fwd_index, spd_key, ref_date, dbtable = 'cmspdvol_daily', field = 'cmdv'):
    stmt = "select {variables} from {table} where fwd_index like '{fwd_index}%' and spd_key='{spd_key}' ".format( \
                                    variables=','.join(price_fields[field]), spd_key = spd_key, \
                                    table = dbtable, fwd_index = fwd_index)
    stmt = stmt + "and date like '{refdate}%' order by expiry".format( refdate = ref_date)
    df = pd.io.sql.read_sql(stmt, cnx)
    return df


def load_tick_to_df(cnx, dbtable, inst, d_start, d_end, start_tick=1500000, end_tick=2115000):
    tick_columns = fut_tick_columns
    if dbtable == 'stock_tick':
        tick_columns = ss_tick_columns
    stmt = "select {variables} from {table} where instID='{instID}' ".format(variables=','.join(tick_columns),
                                                                             table=dbtable, instID=inst)
    stmt = stmt + "and tick_id >= %s " % start_tick
    stmt = stmt + "and tick_id <= %s " % end_tick
    stmt = stmt + "and date >='%s' " % d_start.strftime('%Y-%m-%d')
    stmt = stmt + "and date <='%s' " % d_end.strftime('%Y-%m-%d')
    stmt = stmt + "order by date, tick_id"
    df = pd.io.sql.read_sql(stmt, cnx)
    return df


def load_tick_data(cnx, dbtable, insts, d_start, d_end):
    cursor = cnx.cursor()
    tick_columns = fut_tick_columns
    if dbtable == 'stock_tick':
        tick_columns = ss_tick_columns
    stmt = "select {variables} from {table} where instID in ('{instIDs}') ".format(variables=','.join(tick_columns),
                                                                                   table=dbtable,
                                                                                   instIDs="','".join(insts))
    stmt = stmt + "and date >= '%s' " % d_start.strftime('%Y-%m-%d')
    stmt = stmt + "and date <= '%s' " % d_end.strftime('%Y-%m-%d')
    stmt = stmt + "order by date, tick_id"
    cursor.execute(stmt)
    all_ticks = []
    for line in cursor:
        tick = dict([(key, val) for (key, val) in zip(tick_columns, line)])
        tick['timestamp'] = datetime.datetime.combine(tick['date'], datetime.time(hour=tick['hour'], minute=tick['min'],
                                                                                  second=tick['sec'],
                                                                                  microsecond=tick['msec'] * 1000))
        all_ticks.append(tick)
    return all_ticks


def insert_min_data_to_df(df, min_data):
    new_data = {key: min_data[key] for key in min_columns[1:]}
    df.loc[min_data['datetime']] = pd.Series(new_data)


def insert_new_min_to_df(df, idx, min_data):
    need_update = True
    col_list = min_columns + ['bar_id']
    new_min = {key: min_data[key] for key in col_list}
    if idx > 0:
        idy = idx - 1
        if min_data['datetime'] < df.at[idy, 'datetime']:
            need_update = False
        elif min_data['datetime'] > df.at[idy, 'datetime']:
            idy = idx
    else:
        idy = 0
    if need_update:
        df.loc[idy] = pd.Series(new_min)
    return idy + 1


def insert_daily_data_to_df(df, daily_data):
    if (daily_data['date'] not in df.index):
        new_data = {key: daily_data[key] for key in daily_columns[1:]}
        df.loc[daily_data['date']] = pd.Series(new_data)


def get_daily_by_tick(inst, cur_date, start_tick=1500000, end_tick=2100000):
    df = load_tick_to_df('fut_tick', inst, cur_date, cur_date, start_tick, end_tick)
    ddata = {}
    ddata['date'] = cur_date
    if len(df) > 0:
        ddata['open'] = float(df.iloc[0].price)
        ddata['close'] = float(df.iloc[-1].price)
        ddata['high'] = float(df.iloc[-1].high)
        ddata['low'] = float(df.iloc[-1].low)
        ddata['volume'] = int(df.iloc[-1].volume)
        ddata['openInterest'] = int(df.iloc[-1].openInterest)
    else:
        ddata['open'] = 0.0
        ddata['close'] = 0.0
        ddata['high'] = 0.0
        ddata['low'] = 0.0
        ddata['volume'] = 0
        ddata['openInterest'] = 0
    return ddata


def load_agg_pos_from_db(start = datetime.date.today(), end = datetime.date.today(), strategy = ('SQP', 'SQP'), \
                         out_cols = ['position', 'pos_val', 'pos_bias', 'var95_30d'], db_table = "daily_position"):
    cnx = connect(**pos_dbconfig)
    columns = ['date', 'product', 'contract', 'exchange', 'strategy', 'sub_strategy'] + out_cols
    stmt = "select {variables} from {table} where strategy ='{strat}' and sub_strategy ='{substrat}'".format(variables=','.join(columns),
                                                                             table=db_table, strat = strategy[0], substrat = strategy[1])
    stmt = stmt + "and date >='%s' " % start.strftime('%Y-%m-%d')
    stmt = stmt + "and date <='%s' " % end.strftime('%Y-%m-%d')
    stmt = stmt + "order by date"
    df = pd.io.sql.read_sql(stmt, cnx)
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
    df['lot_size'] = df['product'].apply(lambda x: misc.product_lotsize[x])
    df['exchange'] = df['product'].apply(lambda x: misc.prod2exch(x))
    return df


def load_factor_data(product_list, \
                     factor_list = [],\
                     roll_label = 'CAL-1',\
                     start = datetime.date.today(), \
                     end = datetime.date.today(), \
                     freq = 'd', db_table = 'fut_fact_data'):
    cnx = connect(**dbconfig)
    columns = ['product_code', 'date', 'serial_no', 'serial_key', 'fact_name', 'fact_val']
    stmt = "select {variables} from {table} where fact_name in ('{fact_list}') and product_code in ('{prod_list}') ".format(\
                        fact_list = "','".join(factor_list), prod_list = "','".join(product_list), \
                        variables=','.join(columns), table = db_table)
    stmt = stmt + "and date >='%s' " % start.strftime('%Y-%m-%d')
    stmt = stmt + "and date <='%s' " % end.strftime('%Y-%m-%d')
    stmt = stmt + "and freq = '%s' and roll_label = '%s'" % (freq, roll_label)
    stmt = stmt + "order by date, serial_no"
    df = pd.io.sql.read_sql(stmt, cnx)
    return df


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


def load_fut_by_product(product, exch, start_date, end_date, freq = 'd'):
    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365)
    if freq == 'd':
        db_table = 'fut_daily'
        columns = ['instID', 'date', 'open', 'high', 'low', 'close', 'volume', 'openInterest']
        order_fields = ['instID', 'date',]
    elif freq == 'm':
        db_table = 'fut_min'
        columns = ['instID', 'datetime', 'date', 'min_id', 'open', 'high', 'low', 'close', 'volume', 'openInterest']
        order_fields = ['instID', 'date', 'min_id',]
    if product == 'MA':
        prod_keys = ['ME', 'MA']
    elif product == 'ZC':
        prod_keys = ['TC', 'ZC']
    else:
        prod_keys = [product]
    out_df = pd.DataFrame()
    for prod in prod_keys:
        if exch in ['CZCE']:
            prod_key = f'{prod}%'
        else:
            prod_key = f"{prod}____"
        cnx = connect(**dbconfig)        
        stmt = "select {variables} from {table} where instID like '{prod_key}'  ".format(\
                            prod_key = prod_key,
                            variables=','.join(columns), table = db_table)
        stmt = stmt + "and date >='%s' " % start_date.strftime('%Y-%m-%d')
        stmt = stmt + "and date <='%s' " % end_date.strftime('%Y-%m-%d')
        stmt = stmt + "and exch = '%s' " % (exch)
        stmt = stmt + "order by %s" % (','.join(order_fields))
        df = pd.io.sql.read_sql(stmt, cnx)
        out_df = out_df.append(df)
    if product == 'MA':        
        out_df = out_df[out_df.instID != 'ME505']
        out_df['instID'] = out_df['instID'].replace(['MA506'], 'MA505')
    return out_df