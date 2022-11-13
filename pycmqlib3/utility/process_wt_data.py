import numpy as np
import pathlib
import math
import zipfile
import shutil
import pandas as pd
from typing import Union, List
import datetime
from os import listdir
from os.path import isfile, join
from pycmqlib3.utility import misc
from pycmqlib3.utility.dbaccess import *
import pycmqlib3.analytics.data_handler as dh
from wtpy.wrapper import WtDataHelper
from wtpy.WtCoreDefs import WTSBarStruct, WTSTickStruct


def assign(procession, buffer):
    tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))


def wt_time_to_min_id(wt_time: int = 0):
    hour = wt_time // 100
    minute = wt_time % 100
    min_id = (hour + 6) % 24 * 100 + minute
    return min_id


def convert_wt_data(df, cont, freq='d'):
    df['date'] = df['date'].apply(lambda x: datetime.date(x//10000, (x % 10000)//100, x % 100))
    df = df.rename(columns = {'hold': 'openInterest'})
    df['instID'] = cont
    if freq == 'd':
        col_list = ['instID', 'date', 'open', 'high', 'low', 'close', 'volume', 'openInterest']
    else:
        num_m = 1
        if len(freq)>1:
            num_m = int(freq[1:])
        df['datetime'] = df['bartime'].astype('str').apply(lambda s: datetime.datetime.strptime(s, '%Y%m%d%H%M') -
                                                                     datetime.timedelta(minutes=num_m))
        df['min_id'] = df['datetime'].apply(misc.get_min_id)
        col_list = ['instID', 'datetime', 'date', 'min_id', 'open', 'high', 'low', 'close', 'volume', 'openInterest']
    df = df[col_list]
    return df


def load_fut_by_product(product, exch, start_date, end_date, freq = 'd', folder_loc='C:/dev/wtdev/storage/his'):
    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365)
    period = 'day'
    if freq in ['m', 'm1']:
        freq = 'm1'
        period = 'min1'
    elif freq in ['m5']:
        period = 'min5'
    if product == 'MA':
        prod_keys = ['ME', 'MA']
    elif product == 'ZC':
        prod_keys = ['TC', 'ZC']
    else:
        prod_keys = [product]
    dtHelper = WtDataHelper()
    out_df = pd.DataFrame()
    src_path = '%s/%s/%s' % (folder_loc, period, exch)
    file_list = [f for f in listdir(src_path) if isfile(join(src_path, f))]
    for file in file_list:
        cont = file.split('.')[0]
        prod = cont[:2] if exch in ['CZCE', ] else cont[:-4]
        if prod not in prod_keys:
            continue
        dst_df = dtHelper.read_dsb_bars(f'{src_path}/{file}')
        if dst_df:
            dst_df = dst_df.to_df()
            dst_df = dst_df.rename(columns={'hold': 'openInterest'})
        else:
            continue
        dst_df = convert_wt_data(dst_df, cont, freq=freq)
        out_df = out_df.append(dst_df)
    out_df = out_df[(out_df['date']>=start_date)&(out_df['date']<=end_date)].reset_index(drop=True)
    return out_df


def load_bars_to_df(inst, d_start=None, d_end=None,
                    index_col='datetime',
                    freq='d',
                    folder_loc='C:/dev/wtdev/storage/his'):
    prod_code = misc.inst2product(inst)
    exch = misc.prod2exch(prod_code)
    dtHelper = WtDataHelper()
    period = 'day'
    if freq in ['m', 'm1']:
        period = 'min1'
    elif freq in ['m5']:
        period = 'min5'
    mdf = dtHelper.read_dsb_bars(f'{folder_loc}/{period}/{exch}/{inst}.dsb')
    if mdf:
        mdf = mdf.to_df()
        mdf = convert_wt_data(mdf, inst, freq=freq)
        if d_start:
            mdf = mdf[mdf['date']>=d_start]
        if d_end:
            mdf = mdf[mdf['date']<=d_end]
        mdf = mdf.reset_index(drop=True)
        if index_col:
            mdf = mdf.set_index(index_col)
    return mdf


def conv_min_data(mdf):
    mdf['cal_date'] = mdf['date']
    flag = mdf['min_id']<1400
    mdf.loc[flag, 'cal_date'] = mdf.loc[flag, 'date'].apply(lambda d: misc.day_shift(d, '-1b', misc.CHN_Holidays))
    flag = (mdf['min_id']<1400) & (mdf['min_id']>=600)
    mdf.loc[flag, 'cal_date'] = mdf.loc[flag, 'cal_date'].apply(lambda d: d + datetime.timedelta(days=1))
    mdf['date'] = mdf['date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')
    mdf['time'] = (mdf['cal_date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')-19900000)*10000 + (mdf['datetime'] + pd.Timedelta(minutes=1)).dt.strftime('%H%M%S').str[:-2].astype('int64')
    mdf['hold'] = mdf['openInterest']
    mdf['diff'] = mdf['hold'].diff().fillna(0)
    mdf['money'] = mdf['close'] * mdf['volume'] * mdf['multiple']
    mdf['vol'] = mdf['volume']
    return mdf


def save_bars_to_dsb(df, contract, folder_loc = '.', period = 'm1'):
    dtHelper = WtDataHelper()
    BUFFER = WTSBarStruct*len(df)
    buffer = BUFFER()
    df.apply(assign, buffer=buffer)
    pathlib.Path(folder_loc).mkdir(parents=True, exist_ok=True)
    dtHelper.store_bars(barFile=f'{folder_loc}/{contract}.dsb', firstBar=buffer, count=len(df), period=period)


def save_ticks_to_dsb(tdf, tick_settings):
    dtHelper = WtDataHelper()
    BUFFER = WTSTickStruct * len(tdf)
    buffer = BUFFER()
    exch = tick_settings['exchange']
    exchg = exch.encode('utf-8')
    cont = tick_settings['contract']
    code = cont.encode('utf-8')
    folder_loc = tick_settings['folder_loc']
    tdf = tdf.fillna(0)
    for i in range(len(tdf)):
        curTick = buffer[i]
        curTick.exchg = exchg
        curTick.code = code
        curTick.price = float(tdf["price"].iloc[i])
        curTick.open = float(tdf["open"].iloc[i])
        curTick.high = float(tdf["high"].iloc[i])
        curTick.low = float(tdf["low"].iloc[i])
        curTick.settle_price = float(tdf["settle_price"].iloc[i])
        curTick.total_volume = float(tdf["total_volume"].iloc[i])
        curTick.total_turnover = float(tdf["total_turnover"].iloc[i])
        curTick.open_interest = float(tdf["open_interest"].iloc[i])
        curTick.trading_date = int(tdf["trading_date"].iloc[i])
        curTick.action_date = int(tdf["action_date"].iloc[i])
        curTick.action_time = int(tdf["action_time"].iloc[i])
        curTick.pre_close = float(tdf["pre_close"].iloc[i])
        curTick.pre_settle = float(tdf["pre_settle"].iloc[i])
        curTick.pre_interest = float(tdf["pre_interest"].iloc[i])
        curTick.bid_price_0 = float(tdf["bid_price_0"].iloc[i])
        curTick.bid_qty_0 = float(tdf["bid_qty_0"].iloc[i])
        curTick.ask_price_0 = float(tdf["ask_price_0"].iloc[i])
        curTick.ask_qty_0 = float(tdf["ask_qty_0"].iloc[i])
    pathlib.Path(folder_loc).mkdir(parents=True, exist_ok=True)
    dtHelper.store_ticks(tickFile=f'{folder_loc}/{cont}.dsb', firstTick=buffer, count=len(tdf))


def save_bars_to_wt_store(exchange_list=['DCE', 'CZCE', 'SHFE', 'INE', 'CFFEX'],
                        start_date=datetime.date(2000, 1, 1),
                        end_date=misc.day_shift(datetime.date.today(), '-1b', misc.CHN_Holidays),
                        cutoff_date=None,
                        process_min=True,
                        process_day=True,
                        src_folder='../storage/his',
                        dst_folder='../storage/his',
                        exclusion_list=[]):
    mcol_list = ['date', 'time', 'open', 'high', 'low', 'close', 'money', 'vol', 'hold', 'diff']
    dcol_list = ['date', 'time', 'open', 'high', 'low', 'close', 'settle', 'money', 'vol', 'hold', 'diff']
    dtHelper = WtDataHelper()
    cnx = connect(**dbconfig)
    for exch in exchange_list:
        for prod in misc.product_code[exch]:
            if ('_Opt' in prod) or (prod in exclusion_list):
                continue
            print('product = %s\n' % prod)
            contlist, _ = misc.contract_range(prod, exch, range(1, 13), start_date, misc.day_shift(end_date, '1y'))
            multiple = misc.product_lotsize[prod]
            for cont in contlist:
                if process_min:
                    mdf = load_min_data_to_df(cnx, 'fut_min', cont, start_date, end_date, index_col=None, shift_datetime=False)
                    if len(mdf) > 0:
                        print('minute data for contract = %s\n' % cont)
                        m5df = dh.conv_ohlc_freq(mdf, '5m', index_col=None)

                        mdf['multiple'] = multiple
                        mdf = conv_min_data(mdf)
                        mdf = mdf[mcol_list]
                        period = 'min1'
                        filename = '%s/%s/%s/%s.dsb' % (src_folder, period, exch, cont)
                        if cutoff_date:
                            curr_df = dtHelper.read_dsb_bars(filename)
                            if curr_df:
                                curr_df = curr_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                                curr_df['time'] = curr_df['time'] - 199000000000
                                curr_df = curr_df[curr_df['date'] >= cutoff_date]
                                mdf = mdf[mdf['date'] < cutoff_date]
                                mdf = mdf.append(curr_df)
                        mdf['time'] = mdf['time'].astype('int64')
                        save_bars_to_dsb(mdf[mcol_list], contract=cont, folder_loc=f'{dst_folder}/{period}/{exch}',
                                         period='m1')

                        m5df['multiple'] = multiple
                        m5df = conv_min_data(m5df)
                        m5df['time_d'] = m5df['time'] // 10000
                        m5df['time_t'] = m5df['time'] % 10000
                        m5df['time_h'] = m5df['time_t'] // 100
                        m5df['time_m'] = ((m5df['time_t'] % 100) / 5).apply(math.ceil) * 5
                        m5df['time_h'] = (m5df['time_h'] + m5df['time_m'] // 60) % 24
                        m5df['time_m'] = m5df['time_m'] % 60
                        m5df['time'] = m5df['time_d'] * 10000 + m5df['time_h'] * 100 + m5df['time_m']
                        m5df = m5df[mcol_list]
                        period = 'min5'
                        filename = '%s/%s/%s/%s.dsb' % (src_folder, period, exch, cont)
                        if cutoff_date:
                            curr_df = dtHelper.read_dsb_bars(filename)
                            if curr_df:
                                curr_df = curr_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                                curr_df['time'] = curr_df['time'] - 199000000000
                                curr_df = curr_df[curr_df['date'] >= cutoff_date]
                                m5df = m5df[m5df['date'] < cutoff_date]
                                m5df = m5df.append(curr_df)
                        m5df['time'] = m5df['time'].astype('int64')
                        save_bars_to_dsb(m5df[mcol_list], contract=cont, folder_loc=f'{dst_folder}/{period}/{exch}',
                                         period='m5')

                if process_day:
                    ddf = load_daily_data_to_df(cnx, 'fut_daily', cont, start_date, end_date, index_col=None)
                    ddf['settle'] = ddf['settle'].fillna(method='ffill')
                    ddf = ddf.dropna(subset=['close', 'volume', 'settle'])
                    if len(ddf) > 0:
                        print('daily data for contract = %s\n' % cont)
                        ddf['date'] = ddf['date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')
                        ddf['time'] = 0
                        ddf['hold'] = ddf['openInterest']
                        ddf['diff'] = ddf['hold'].diff().fillna(0)
                        ddf['money'] = ddf['settle'] * ddf['volume'] * multiple
                        ddf['vol'] = ddf['volume']
                        period = 'day'
                        ddf = ddf[dcol_list]
                        filename = '%s/%s/%s/%s.dsb' % (src_folder, period, exch, cont)
                        if cutoff_date:
                            curr_df = dtHelper.read_dsb_bars(filename)
                            if curr_df:
                                curr_df = curr_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                                curr_df['time'] = curr_df['time'] - 199000000000
                                curr_df = curr_df[curr_df['date'] >= cutoff_date]
                                ddf = ddf[ddf['date'] < cutoff_date]
                                ddf = ddf.append(curr_df)
                        ddf['time'] = ddf['time'].astype('int64')
                        save_bars_to_dsb(ddf[dcol_list], contract=cont, folder_loc=f'{dst_folder}/{period}/{exch}',
                                         period='d')


def save_ticks_to_wt_store(
        exchange_list=['DCE', 'CZCE', 'SHFE', 'INE', 'CFFEX'],
        start_date=datetime.date(2000, 1, 1),
        end_date=misc.day_shift(datetime.date.today(), '-1b', misc.CHN_Holidays),
        dst_folder='../storage/his',
        exclusion_list=[]):
    cnx = connect(**dbconfig)
    for exch in exchange_list:
        for prod in misc.product_code[exch]:
            if ('_Opt' in prod) or (prod in exclusion_list):
                continue
            print('product = %s\n' % prod)
            contlist, _ = misc.contract_range(prod, exch, range(1, 13), start_date, misc.day_shift(end_date, '1y'))
            multiple = misc.product_lotsize[prod]
            for cont in contlist:
                tdf = load_tick_to_df(cnx, 'fut_tick', cont, start_date, end_date, start_tick=0)
                ddf = load_daily_data_to_df(cnx, 'fut_daily', cont, start_date, end_date, index_col='date', field='instID',
                                            date_as_str=False)
                if len(ddf) == 0:
                    continue
                print(f'contract = {cont}')
                ddf = ddf.rename(columns={'openInterest': 'open_interest'}).reset_index()
                ddf['pre_settle'] = ddf['settle'].shift(1)
                ddf['pre_interest'] = ddf['open_interest'].shift(1)
                ddf['pre_close'] = ddf['close'].shift(1)
                ddf['date'] = ddf['date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')
                ddf = ddf.set_index('date')
                tdf['trading_date'] = tdf['date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')
                tdf['action_time'] = tdf['hour'] * 10000000 + tdf['min'] * 100000 + tdf['sec'] * 1000 + tdf['msec']
                tdf['action_date'] = tdf['date']
                flag = tdf['tick_id'] < 1400000
                tdf.loc[flag, 'action_date'] = tdf.loc[flag, 'date'].apply(lambda d: misc.day_shift(d, '-1b', misc.CHN_Holidays))
                flag = (tdf['tick_id'] < 1400000) & (tdf['tick_id'] >= 600000)
                tdf.loc[flag, 'action_date'] = tdf.loc[flag, 'action_date'].apply(lambda d: d + datetime.timedelta(days=1))
                tdf['action_date'] = tdf['action_date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')
                tdf['time'] = tdf['action_date'] * 1000000000 + tdf['action_time']
                tdf['exchg'] = exch
                tdf.rename({'instID': 'code',
                            'volume': 'total_volume',
                            'openInterest': 'open_interest',
                            'bidPrice1': 'bid_price_0',
                            'bidVol1': 'bid_qty_0',
                            'askPrice1': 'ask_price_0',
                            'askVol1': 'ask_qty_0',
                            }, axis='columns', inplace=True)
                tdf['diff_interest'] = tdf['open_interest'].diff().fillna(0)
                tdf['settle_price'] = 0.0

                grouped = tdf.groupby('trading_date')
                tick_cols = [
                    'time', 'exchg', 'code', 'price', 'open', 'high', 'low', 'settle_price',
                    'total_volume', 'volume', 'total_turnover', 'turn_over', 'open_interest', 'diff_interest',
                    'trading_date', 'action_date', 'action_time', 'pre_close', 'pre_settle', 'pre_interest',
                    'bid_price_0', 'ask_price_0', 'bid_qty_0', 'ask_qty_0',
                ]
                for tdate, grp_df in grouped:
                    grp_df['volume'] = grp_df['total_volume'].diff()
                    grp_df.at[grp_df.index[0], 'volume'] = grp_df['total_volume'].iloc[0]
                    grp_df['turn_over'] = grp_df['volume'] * grp_df['price'] * multiple
                    grp_df['total_turnover'] = grp_df['turn_over'].cumsum()
                    if tdate in ddf.index:
                        grp_df['open'] = ddf.loc[tdate, 'open']
                        grp_df['pre_close'] = ddf.loc[tdate, 'pre_close']
                        grp_df['pre_settle'] = ddf.loc[tdate, 'pre_settle']
                        grp_df['pre_interest'] = ddf.loc[tdate, 'pre_interest']
                        grp_df.at[grp_df.index[-1], 'settle_price'] = ddf.loc[tdate, 'settle']
                        grp_df['settle_price'] = grp_df['settle_price'].fillna(0.0)
                    else:
                        grp_df['open'] = 0
                        grp_df['pre_close'] = 0
                        grp_df['pre_settle'] = 0
                        grp_df['pre_interest'] = 0
                        grp_df['settle_price'] = 0
                    grp_df = grp_df[tick_cols]
                    folder_loc = f'{dst_folder}/{exch}/{tdate}/'
                    tick_settings = {
                        'exchange': exch,
                        'contract': cont,
                        'folder_loc': folder_loc,
                    }
                    save_ticks_to_dsb(grp_df, tick_settings)


def combine_bars_wt_store(src_folder, dst_folder, target_folder, cutoff=None):
    dtHelper = WtDataHelper()
    period_map = {'day': 'd', 'min1': 'm1', 'min5': 'm5'}
    for period in ['day', 'min1', 'min5', ]:
        for exch in ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE', ]:
            print(f'{period}-{exch}')
            src_path = '%s/%s/%s' % (src_folder, period, exch)
            file_list = [f for f in listdir(src_path) if isfile(join(src_path, f))]
            for file in file_list:
                cont = file.split('.')[0]
                src_df = dtHelper.read_dsb_bars(f'{src_path}/{file}')
                src_df = src_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                src_df['time'] = src_df['time'] - 199000000000
                src_df = src_df[src_df['vol']>0]
                dst_path = '%s/%s/%s' % (dst_folder, period, exch)
                dst_df = dtHelper.read_dsb_bars(f'{dst_path}/{file}')
                if dst_df:
                    dst_df = dst_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                    dst_df['time'] = dst_df['time'] - 199000000000
                    dst_df = dst_df[dst_df['vol'] > 0]
                    if cutoff:
                        src_df = src_df[src_df['date'] < cutoff]
                        dst_df = dst_df[dst_df['date'] >= cutoff]
                        dst_df = src_df.append(dst_df)
                else:
                    dst_df = src_df
                dst_df['time'] = dst_df['time'].astype('int64')
                save_bars_to_dsb(dst_df, contract=cont, folder_loc=f'{target_folder}/{period}/{exch}',
                                 period=period_map[period])


def zip_wt_dir(path, filename, cutoff=None, file_type='.dsb'):
    with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(file_type):
                    time_mod = os.path.getmtime(f'{root}/{file}')
                    time_mod = datetime.datetime.fromtimestamp(time_mod)
                    if cutoff and (time_mod.date() < cutoff):
                        continue
                    ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


def update_wt_store(src_folder, dst_folder, cutoff=None):
    combine_bars_wt_store(dst_folder, src_folder, dst_folder, cutoff=cutoff)
    shutil.copytree(src_folder + '/ticks/', dst_folder + '/ticks/', dirs_exist_ok=True)


if __name__ == "__main__":
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2022, 10, 23)
    src_folder = 'c:/dev/data/his'
    dst_folder = 'c:/dev/data/his_new'
    # save_ticks_to_wt_store(
    #     exchange_list=['DCE'],
    #     start_date=start_date,
    #     end_date=end_date,
    #     dst_folder=dst_folder)

    cutoff_date = 20221001
    # save_bars_to_wt_store(exchange_list=['DCE', 'CZCE', 'SHFE', 'INE', 'CFFEX'],
    #                         start_date=start_date,
    #                         end_date=end_date,
    #                         cutoff_date=cutoff_date,
    #                         process_min=True,
    #                         process_day=True,
    #                         src_folder=src_folder,
    #                         dst_folder=dst_folder)

    cutoff = datetime.date(2022, 11, 5)
    zipdir('c:/dev/wtdev/storage/his/', 'C:/dev/data/test_zip.zip', cutoff=cutoff)
