import numpy as np
import json
import os
import pathlib
import math
import zipfile
import shutil
import pandas as pd
import datetime
from os import listdir
from os.path import isfile, join
from pycmqlib3.utility import misc
from pycmqlib3.utility import dbaccess
import pycmqlib3.analytics.data_handler as dh
from wtpy.wrapper import WtDataHelper
from wtpy import WtDtServo
from wtpy.WtCoreDefs import WTSBarStruct, WTSTickStruct


def assign(procession, buffer):
    tuple(map(lambda x: setattr(buffer[x[0]], procession.name, x[1]), enumerate(procession)))


def wtcode_to_code(wtcode):
    code_split = wtcode.split('.')
    if (len(code_split) == 0) or (len(code_split) > 3):
        raise ValueError(f"wtpy stdcode={wtcode} has a wrong format.")
    elif len(code_split) == 1:
        code = wtcode
        inst = code_split[1]
    else:
        if code_split[0] in ['CZCE']:
            if int(code_split[2]) > 1500:
                code_split[2] = str(int(code_split[2]) % 1000)
        inst = ''.join(code_split[1:])
        code = '.'.join([code_split[0], inst])
    return code, inst

def code_to_inst(code):
    code_split = code.split('.')
    exch = code_split[0]
    inst = ''.join(code_split[1:])
    return exch, inst


def wt_time_to_min_id(wt_time: int = 0):
    hour = wt_time // 100
    minute = wt_time % 100
    min_id = ((hour + 6) % 24) * 100 + minute
    return min_id


def roll_df_to_list(roll_df: pd.DataFrame, field_name: str='instID') -> list:
    roll_df = roll_df.reset_index()
    inst = roll_df[field_name].iloc[-1]
    exch = misc.inst2exch(inst)
    roll_df['prev_inst'] = roll_df[field_name].shift(1).fillna('')
    row_list = []
    for idx, row in enumerate(roll_df.to_dict('records')):
        prev_price = curr_price = 0.0
        if len(row['prev_inst']) > 0:
            prev_df = load_bars_to_df(exch+'.'+row['prev_inst'],
                                      period='d1',
                                      start_time=row['date'],
                                      end_time=row['date'])
            if len(prev_df) > 0:
                prev_price = prev_df['close'].iloc[-1]
            else:
                print("no prev price for %s on %s on roll date, set it as 0.0" % (exch+'.'+row['prev_inst'], row['date']))
        curr_df = load_bars_to_df(exch+'.'+row[field_name],
                                  period='d1',
                                  start_time=row['date'],
                                  end_time=row['date'])
        if len(curr_df) > 0:
            curr_price = curr_df['close'].iloc[-1]
        else:
            print("no curr price for %s on %s on roll date, set it as 0.0" % (exch + '.' + row[field_name], row['date']))
        roll_date = misc.day_shift(row['date'].date(), roll_rule='1b', hols=misc.CHN_Holidays)
        row_list.append({
            'date': int(roll_date.strftime("%Y%m%d")),
            'from': row['prev_inst'],
            'oldclose': prev_price,
            'to': row[field_name],
            'newclose': curr_price,
        })
    return row_list


def roll_list_to_df(roll_list: list, field_name: str='instID') -> pd.DataFrame:
    roll_df = pd.DataFrame.from_dict(roll_list).rename(columns={'to': field_name}).drop(columns=['from', 'oldclose', 'newclose'])
    roll_df['date'] = pd.to_datetime(roll_df['date'].astype(str), format='%Y%m%d')
    roll_df['date'] = roll_df['date'].apply(lambda d: misc.day_shift(d.date(), '-1b', misc.CHN_Holidays))
    roll_df = roll_df.set_index('date')
    return roll_df


def convert_wt_data(df, cont, freq='d'):
    df['date'] = df['date'].apply(lambda x: datetime.date(x//10000, (x % 10000)//100, x % 100))
    df = df.rename(columns={'hold': 'openInterest'})
    df['instID'] = cont    
    if freq in ['d', 'day', 'd1']:
        col_list = ['instID', 'date', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'diff_oi', 'settle']
    else:
        num_m = 1
        if len(freq) > 1:
            num_m = int(freq[1:])
        df['datetime'] = df['bartime'].astype('str').apply(lambda s: datetime.datetime.strptime(s, '%Y%m%d%H%M') -
                                                                     datetime.timedelta(minutes=num_m))
        df['min_id'] = df['datetime'].apply(misc.get_min_id)
        col_list = ['instID', 'datetime', 'date', 'min_id', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'diff_oi']
    df = df[col_list]
    return df


def load_fut_by_product(code, start_date, end_date, freq='d', folder_loc='C:/dev/wtdev/storage/his'):
    exch, product = code.split('.')
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
            dst_df = dst_df.rename(columns={'hold': 'openInterest', 'diff': 'diff_oi'})
        else:
            continue
        dst_df = convert_wt_data(dst_df, cont, freq=freq)
        out_df = pd.concat([out_df, dst_df])
    out_df = out_df[(out_df['date']>=start_date)&(out_df['date']<=end_date)].reset_index(drop=True)
    out_df['expiry'] = out_df.apply(lambda x: misc.contract_expiry(x["instID"], curr_dt=x["date"], hols=misc.CHN_Holidays), axis=1)
    out_df = out_df.sort_values(["date", "expiry"])
    return out_df


def load_hist_bars_to_df(code, start_date=None, end_date=None,
                    index_col='date',
                    freq='d',
                    folder_loc='C:/dev/wtdev/storage/his'):
    exch, instID = code.split('.')
    dtHelper = WtDataHelper()
    period = 'day'
    if freq in ['m', 'm1']:
        period = 'min1'
    elif freq in ['m5']:
        period = 'min5'
    mdf = dtHelper.read_dsb_bars(f'{folder_loc}/{period}/{exch}/{instID}.dsb')
    if mdf:
        mdf = mdf.to_df().rename(columns={'hold': 'openInterest', 'diff': 'diff_oi'})
        mdf = convert_wt_data(mdf, instID, freq=freq)
        if start_date:
            mdf = mdf[mdf['date'] >= start_date]
        if end_date:
            mdf = mdf[mdf['date'] <= end_date]
        mdf = mdf.reset_index(drop=True)
        if index_col:
            mdf = mdf.set_index(index_col)
    return mdf


def load_bars_by_code(code, start_date=None, end_date=None,
                    index_col=None,
                    freq='d',
                    folder_loc='C:/dev/wtdev/storage/his'):
    exch, instID = code.split('.')
    dtHelper = WtDataHelper()
    period = 'day'
    if freq == ['d', 'd1']:
        period = 'day'
    elif freq in ['m', 'm1']:
        period = 'min1'
    elif freq in ['m5']:
        period = 'min5'
    mdf = dtHelper.read_dsb_bars(f'{folder_loc}/{period}/{exch}/{instID}.dsb')
    if mdf:
        mdf = mdf.to_df().rename(columns={'hold': 'openInterest', 'diff': 'diff_oi'})
        mdf = mdf[(mdf['openInterest'] > 0) & (mdf['date'] < 20990000)]
        mdf = convert_wt_data(mdf, instID, freq=freq)
        if 'm' in freq:
            mdf = mdf.drop_duplicates(subset=['date', 'min_id'])
        if start_date:
            mdf = mdf[mdf['date'] >= start_date]
        if end_date:
            mdf = mdf[mdf['date'] <= end_date]
        mdf = mdf.reset_index(drop=True)
        if index_col:
            mdf = mdf.set_index(index_col)
    return mdf


def date_to_int(cur_date=None):
    if cur_date is None:
        cur_date = datetime.date.today()
    if isinstance(cur_date, (int, float)):
        cur_date = str(int(cur_date))
    cur_date = pd.to_datetime(cur_date)
    cur_date = int(cur_date.strftime("%Y%m%d"))
    return cur_date


def conv_dt_to_int(dt=None, period='d1', mode='s', exch='SHFE'):
    if dt is None:
        dt = datetime.date.today()
    if isinstance(dt, (float, int)):
        dt = str(int(dt))
    dt = pd.to_datetime(dt)
    out_dt = int(dt.strftime("%Y%m%d%H%M"))
    hols = misc.get_hols_by_exch(exch)
    if 'm' in period:
        if mode in ['S', 's']:
            bday = misc.day_shift(dt.date(), '-1b', hols)
            out_dt = int(bday.strftime("%Y%m%d"))*10000 + 1800
        elif out_dt % 10000 == 0:
            out_dt += 1700
    return out_dt


def load_bars_to_df(code, period='d1', start_time=None, end_time=None,
                    index_col=None,
                    folder_loc='C:/dev/wtdev'):
    exch, instID = code_to_inst(code)
    start_time = conv_dt_to_int(start_time, period, mode='s', exch=exch)
    end_time = conv_dt_to_int(end_time, period, mode='e', exch=exch)
    dtServo = WtDtServo()
    dtServo.setBasefiles(folder=f"{folder_loc}/common/")
    dtServo.setStorage(path=f'{folder_loc}/storage/')
    df = dtServo.get_bars(code, period, fromTime=start_time, endTime=end_time)
    if df:
        df = df.to_df().rename(columns={'hold': 'openInterest', 'diff': 'diff_oi'})
        #df['bartime'] = df['bartime']
        if 'd' in period:
            freq = 'd'
        else:
            freq = 'm'
        df = convert_wt_data(df, instID, freq=freq)
        df = df.reset_index(drop=True)
        if index_col:
            df = df.set_index(index_col)
    else:
        df = pd.DataFrame()
    return df


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
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    for exch in exchange_list:
        for prod in misc.product_code[exch]:
            if ('_Opt' in prod) or (prod in exclusion_list):
                continue
            print('product = %s\n' % prod)
            contlist = dbaccess.load_cont_by_prod(prod)
            multiple = misc.product_lotsize[prod]
            for cont in contlist:
                if process_min:
                    mdf = dbaccess.load_min_data_to_df(cnx, 'fut_min', cont, start_date, end_date, index_col=None, shift_datetime=False)
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
                                mdf = pd.concat([mdf, curr_df])
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
                                m5df = pd.concat([m5df, curr_df])
                        m5df['time'] = m5df['time'].astype('int64')
                        save_bars_to_dsb(m5df[mcol_list], contract=cont, folder_loc=f'{dst_folder}/{period}/{exch}',
                                         period='m5')

                if process_day:
                    ddf = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', cont, start_date, end_date, index_col=None)
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
                                ddf = pd.concat([ddf, curr_df])
                        ddf['time'] = ddf['time'].astype('int64')
                        save_bars_to_dsb(ddf[dcol_list], contract=cont, folder_loc=f'{dst_folder}/{period}/{exch}',
                                         period='d')


def save_ticks_to_wt_store(
        exchange_list=['DCE', 'CZCE', 'SHFE', 'INE', 'CFFEX', 'GFEX'],
        start_date=datetime.date(2000, 1, 1),
        end_date=misc.day_shift(datetime.date.today(), '-1b', misc.CHN_Holidays),
        dst_folder='../storage/his',
        exclusion_list=[]):
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    for exch in exchange_list:
        for prod in misc.product_code[exch]:
            if ('_Opt' in prod) or (prod in exclusion_list):
                continue
            print('product = %s\n' % prod)
            contlist, _ = misc.contract_range(prod, exch, range(1, 13), start_date, misc.day_shift(end_date, '1y'))
            multiple = misc.product_lotsize[prod]
            for cont in contlist:
                tdf = dbaccess.load_tick_to_df(cnx, 'fut_tick', cont, start_date, end_date, start_tick=0)
                ddf = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', cont, start_date, end_date,
                                                     index_col='date', field='instID', date_as_str=False)
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
        for exch in ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE', 'GFEX']:
            print(f'{period}-{exch}')
            src_path = '%s/%s/%s' % (src_folder, period, exch)
            dst_path = '%s/%s/%s' % (dst_folder, period, exch)
            file_list = [f for f in listdir(dst_path) if isfile(join(dst_path, f))]
            for file in file_list:
                cont = file.split('.')[0]
                dst_df = dtHelper.read_dsb_bars(f'{dst_path}/{file}')                
                dst_df = dst_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                dst_df['time'] = dst_df['time'] - 199000000000
                dst_df = dst_df[dst_df['vol'] > 0]
                src_df = dtHelper.read_dsb_bars(f'{src_path}/{file}')
                if src_df:
                    src_df = src_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
                    src_df['time'] = src_df['time'] - 199000000000
                    src_df = src_df[src_df['vol']>0]                
                    if cutoff is None:
                        cutoff = src_df['date'].iloc[-1]
                    src_df = src_df[src_df['date'] <= cutoff]
                    dst_df = dst_df[dst_df['date'] > cutoff]
                    dst_df = pd.concat([src_df, dst_df])
                dst_df['time'] = dst_df['time'].astype('int64')
                save_bars_to_dsb(dst_df, contract=cont, folder_loc=f'{target_folder}/{period}/{exch}',
                                 period=period_map[period])


def combine_data_wt_store(src_folder, dst_folder, target_folder, curr_date, time_range=[202504150859, 202504150905], exch_list=['INE']):
    dtHelper = WtDataHelper()
    period_map = {'min1': 'm1', 'min5': 'm5'}
    for period in ['min1', 'min5']:
        for exch in exch_list:            
            src_path = '%s/%s/%s' % (src_folder, period, exch)
            dst_path = '%s/%s/%s' % (dst_folder, period, exch)
            file_list = [f for f in listdir(dst_path) if isfile(join(dst_path, f))]
            for file in file_list:
                print(f'{period}-{exch}-{file}')
                cont = file.split('.')[0]                
                dst_df = dtHelper.read_dsb_bars(f'{dst_path}/{file}')                
                dst_df = dst_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})                
                src_df = dtHelper.read_dsb_bars(f'{src_path}/{file}')
                if src_df:
                    src_df = src_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})                    
                    src_df = pd.concat([
                        src_df[src_df['time'] < time_range[0]], 
                        dst_df[(dst_df['time'] >= time_range[0]) & (dst_df['time'] <= time_range[1])],
                        src_df[src_df['time'] > time_range[1]], 
                    ])
                    src_df['time'] = src_df['time'].astype('int64') - 199000000000
                    save_bars_to_dsb(src_df, contract=cont, folder_loc=f'{target_folder}/{period}/{exch}',
                                     period=period_map[period])
    # update ticks
    period = 'ticks'
    for exch in exch_list:        
        src_path = f'{src_folder}/{period}/{exch}/{curr_date}'
        dst_path = f'{dst_folder}/{period}/{exch}/{curr_date}'
        file_list = [f for f in listdir(dst_path) if isfile(join(dst_path, f))]
        for file in file_list:
            print(f'ticks-{exch}-{curr_date}-{file}')
            cont = file.split('.')[0]                
            dst_df = dtHelper.read_dsb_ticks(f'{dst_path}/{file}')                
            dst_df = dst_df.to_df()         
            src_df = dtHelper.read_dsb_ticks(f'{src_path}/{file}')
            if src_df:
                src_df = src_df.to_df()                 
                src_df = pd.concat([
                    src_df[src_df['time'] < time_range[0]*100000], 
                    dst_df[(dst_df['time'] >= time_range[0]*100000) & (dst_df['time'] <= time_range[1]*100000)],
                    src_df[src_df['time'] > time_range[1]*100000], 
                ])
            #src_df['time'] = src_df['time'].astype('int64') - 199000000000
                tick_settings = {
                    'exchange': exch,
                    'contract': cont,
                    'folder_loc': f'{target_folder}/{period}/{exch}/{curr_date}/',
                }
                save_ticks_to_dsb(src_df, tick_settings)

                
def zip_wt_dir(path, filename, cutoff=None, file_types=['.dsb', 'csv']):
    with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(tuple(file_types)):
                    time_mod = os.path.getmtime(f'{root}/{file}')
                    time_mod = datetime.datetime.fromtimestamp(time_mod)
                    if cutoff and (time_mod.date() < cutoff):
                        continue
                    ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


def update_wt_store(base_folder, update_folder, cutoff=None):
    if isinstance(cutoff, datetime.date):
        cutoff = int(datetime.datetime.strftime(cutoff, '%Y%m%d'))
    combine_bars_wt_store(base_folder, update_folder, base_folder, cutoff=cutoff)
    try:
        shutil.copytree(update_folder + '/ticks/', base_folder + '/ticks/', dirs_exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        shutil.copytree(update_folder + '/snapshot/', base_folder + '/snapshot/', dirs_exist_ok=True)
    except FileNotFoundError:
        pass


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

    cutoff_date = datetime.date(2022, 11, 26)
    zip_wt_dir('c:/dev/wtdev/storage/his/', 'C:/dev/data/test_zip.zip', cutoff=cutoff_date)
