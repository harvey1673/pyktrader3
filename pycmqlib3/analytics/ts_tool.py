# -*- coding: utf-8 -*-
import datetime
from pycmqlib3.utility import misc
from pycmqlib3.utility import dbaccess
from . import data_handler as dh


def get_data(spotID, start, end, spot_table = 'spot_daily', name = None, index_col = 'date', fx_pair = None, field = 'spotID', args = None):
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    if args:
        args['start_date'] = start
        args['end_date'] = end
        df = misc.nearby(spotID, **args)
        df = df.reset_index()
    else:
        df = dbaccess.load_daily_data_to_df(cnx, spot_table, spotID, start, end, index_col = None, field = field)
    if isinstance(df[index_col][0], str):
        if len(df[index_col][0])> 12:
            df[index_col] = df[index_col].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
        else:
            df[index_col] = df[index_col].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
    df = df.set_index(index_col)
    if name:
        col_name = name
    else:
        col_name = spotID
    if field == 'ccy':
        df = df[df.tenor=='0W']
        data_field = 'rate'
    elif field == 'spotID':
        data_field = 'close'
    elif field == 'instID':
        data_field = 'close'
    df = df[[data_field]]
    df.rename(columns = {data_field: col_name}, inplace = True)
    if fx_pair:
        fx = fx_pair.split('/')
        direction = misc.get_mkt_fxpair(fx[0], fx[1])
        if direction < 0:
            mkt_pair = fx[1] + fx[0] + '_fx'
        else:
            mkt_pair = fx[0] + fx[1] + '_fx'
        fx = dbaccess.load_daily_data_to_df(cnx, 'spot_daily', mkt_pair, start, end, index_col = None, field = 'spotID')
        if isinstance(fx[index_col][0], str):
            fx[index_col] = fx[index_col].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
        fx = fx.set_index(index_col)
        if direction >= 0:
            df[col_name] = df[col_name]/fx['close']
        else:
            df[col_name] = df[col_name]*fx['close']
    return df


def get_multiple_data(inst_list, start, end):
    df_list = []
    for (instID, db_table, field, args) in inst_list:
        df = get_data(instID, start, end, spot_table = db_table, name = instID, field = field, args = args)
        df_list.append(df)
    xdf = merge_df(df_list)
    return xdf


def merge_df(df_list):
    if len(df_list) == 0:
        return None
    xdf = df_list[0]
    for i in range(1, len(df_list)):
        xdf = xdf.merge(df_list[i], left_index = True, right_index = True, how = 'outer')
        #xdf.rename(columns={ col_name: "x"+str(i)}, inplace=True )
    return xdf


def get_cont_data(asset, start_date, end_date, freq = '1m', nearby = 1, rollrule = '-10b'):
    cnx = dbaccess.connect(**dbaccess.hist_dbconfig)
    if nearby == 0:
        mdf = dbaccess.load_min_data_to_df(cnx, 'fut_min', asset, start_date, end_date, minid_start = 300, minid_end = 2114)
        mdf['contract'] = asset
    else:
        mdf = misc.nearby(asset, nearby, start_date, end_date, rollrule, 'm', shift_mode=True)
    mdf = misc.cleanup_mindata(mdf, asset)
    xdf = dh.conv_ohlc_freq(mdf, freq, extra_cols = ['contract'], bar_func = dh.bar_conv_func2)
    return xdf


def validate_db_data(tday, filter = False):
    all_insts = misc.filter_main_cont(tday, filter)
    data_count = {}
    inst_list = {'min': [], 'daily': [] }
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    for instID in all_insts:
        df = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', instID, tday, tday)
        if len(df) <= 0:
            inst_list['daily'].append(instID)
        elif (df.close[-1] == 0) or (df.high[-1] == 0) or (df.low[-1] == 0) or df.open[-1] == 0:
            inst_list['daily'].append(instID)
        df = dbaccess.load_min_data_to_df(cnx, 'fut_min', instID, tday, tday, minid_start=300, minid_end=2115)
        if len(df) <= 100:
            output = instID + ':' + str(len(df))
            inst_list['min'].append(output)
        elif df.min_id < 2055:
            output = instID + ': end earlier'
            inst_list['min'].append(output)        
    print(inst_list)

