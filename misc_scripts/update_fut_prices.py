import sys
import numpy as np
import datetime
import pandas as pd
import pickle
import gc
from pycmqlib3.utility import dataseries
from pycmqlib3.utility.misc import contract_expiry, inst2contmth, day_shift, \
    is_workday, CHN_Holidays

All_MARKETS = [
    'rb', 'hc', 'i', 'j', 'jm', 'FG', 'SM', 'SF', 'SA', 'ru', 'nr',
    'cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'si', 'ao', 'bc',
    'l', 'pp', 'v', 'TA', 'sc', 'lu', 'eb', 'eg',
    'pg', 'PF', 'MA', 'fu', 'bu',
    'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', 'lh', 'b', 'CY', 'cs',
    'AP', 'CJ', 'UR', 'PK', 'SR', 'sp', 'au', 'ag', 'lc', 'ec',
    'T', 'TF', 'IF', 'IH', 'IC',
]


def refresh_saved_fut_prices(
        run_date,
        nb_cont=2,
        data_file="C:/dev/data/cnc_fut_prices_latest.pkl"
):
    period_setup = {
        'n305': [303, 307],
        'n310': [307, 311],
        'n450': [450, 455],
        'a1505': [1503, 1507],
        'a1510': [1507, 1511],
        'a1635': [1635, 1640],
        'p1935': [1935, 1940],
        'p2055': [2055, 2058],
    }
    try:
        with open(data_file, 'rb') as handle:
            df_dict = pickle.load(handle)
    except:
        df_dict = {}
    if 'daily_data' in df_dict:
        daily_dict = df_dict['daily_data']
    else:
        daily_dict = {}

    if 'min_data' in df_dict:
        min_dict = df_dict['min_data']
    else:
        min_dict = {}
    start_date = datetime.date(2010, 1, 1)
    for asset in All_MARKETS:
        for nb in range(1, nb_cont+1):
            print(f"loading product={asset}, nb={nb}")
            cont = f"{asset}c{nb}"
            if cont in daily_dict:
                curr_ddf = daily_dict[cont]
                start_d = curr_ddf.index[-1]
            else:
                curr_ddf = pd.DataFrame()
                start_d = start_date

            ddf = dataseries.nearby(asset,
                                    nb,
                                    start_date=start_d, end_date=run_date,
                                    shift_mode=2, roll_name='hot', freq='d1')
            if len(ddf) == 0:
                print("no new data")
                continue
            ddf['expiry'] = pd.to_datetime(ddf['contract'].map(contract_expiry))
            ddf['contmth'] = ddf.apply(lambda x: inst2contmth(x['contract'], x['date']), axis=1)
            ddf = ddf.set_index('date')
            ddf.index = pd.to_datetime(ddf.index)
            if cont in min_dict:
                curr_mdf = min_dict[cont]
                start_d = curr_mdf.index[-1]
            else:
                curr_mdf = pd.DataFrame()
                start_d = start_date
            mdf = dataseries.nearby(asset,
                                    nb,
                                    start_date=start_d, end_date=run_date,
                                    shift_mode=2, roll_name='hot', freq='m1')
            if len(mdf) == 0:
                continue
            mdf['expiry'] = pd.to_datetime(mdf['contract'].map(contract_expiry))
            mdf['date'] = pd.to_datetime(mdf['date'])
            mdf['contmth'] = mdf.apply(lambda x: inst2contmth(x['contract'], x['date']), axis=1)
            mdf = mdf.set_index('datetime')
            mdf.index = pd.to_datetime(mdf.index)
            df_list = []
            for px_name in period_setup:
                twap = mdf[mdf['min_id'].isin(range(*period_setup[px_name]))][['date', 'close']].groupby('date').mean()
                twap.columns = [px_name]
                df_list.append(twap)
            twap_df = pd.concat(df_list, axis=1)
            twap_df.index = pd.to_datetime(twap_df)
            ddf = pd.concat([ddf, twap_df], axis=1)

            if len(curr_ddf) > 0:
                cutoff = curr_ddf.index[-1]
                new_shift = ddf.loc[cutoff, 'shift']
                if new_shift != 0:
                    for col in ['open', 'high', 'low', 'close', 'settle'] + twap_df.columns:
                        if col in curr_ddf.columns:
                            curr_ddf[col] = curr_ddf[col]/np.exp(new_shift)
                    curr_ddf['shift'] = curr_ddf['shift'] + new_shift
                ddf = ddf[ddf.index > cutoff]
            curr_ddf = pd.concat([curr_ddf, ddf])
            curr_ddf = curr_ddf[~curr_ddf.index.duplicated(keep='last')]
            daily_dict[cont] = curr_ddf
            if len(curr_mdf) > 0:
                cutoff = curr_mdf.index[-1]
                new_shift = mdf.loc[cutoff, 'shift']
                if new_shift != 0:
                    for col in ['open', 'high', 'low', 'close']:
                        if col in curr_mdf.columns:
                            curr_mdf[col] = curr_mdf[col]/np.exp(new_shift)
                    curr_mdf['shift'] = curr_mdf['shift'] + new_shift
                mdf = mdf[mdf.index > cutoff]
            curr_mdf = pd.concat([curr_mdf, mdf])
            curr_mdf = curr_mdf[~curr_mdf.index.duplicated(keep='last')]
            min_dict[cont] = curr_mdf
        with open(data_file, 'wb') as f:
            df_dict['daily_data'] = daily_dict
            df_dict['min_data'] = min_dict
            pickle.dump(df_dict, f)
    return df_dict


def update_saved_fut_prices(
        columns_to_drop=[],
        period_setup={
            'n305': [303, 307],
            'n310': [307, 311],
            'n450': [450, 455],
            'a1505': [1503, 1507],
            'a1510': [1507, 1511],
            'a1635': [1635, 1640],
            'p1935': [1935, 1940],
            'p2055': [2055, 2058],
        },
        data_file="C:/dev/data/cnc_fut_prices_latest.pkl"
):
    try:
        with open(data_file, 'rb') as handle:
            df_dict = pickle.load(handle)
    except:
        df_dict = {}
    if ('min_data' not in df_dict) or ('daily_data' not in df_dict):
        return

    for cont in df_dict['daily_data']:
        ddf = df_dict['daily_data'][cont]
        mdf = df_dict['min_data'][cont]
        df_list = []
        for px_name in period_setup:
            twap = mdf[mdf['min_id'].isin(range(*period_setup[px_name]))][['date', 'close']].groupby('date').mean()
            twap.columns=[px_name]
            df_list.append(twap)
        twap_df = pd.concat(df_list, axis=1)
        ddf = ddf.drop(columns=columns_to_drop)
        df_dict['daily_data'][cont] = pd.concat([ddf, twap_df], axis=1)
    with open(data_file, 'wb') as f:
        pickle.dump(df_dict, f)


def load_saved_fut(tday=datetime.date.today(),
                   freq='d',
                   data_file="C:/dev/data/cnc_fut_prices_latest.pkl"):
    tday = pd.to_datetime(tday)
    try:
        df = pd.read_parquet(f"C:/dev/data/fut_{freq}_%s.parquet" % tday.strftime("%Y%m%d"))
    except:
        try:
            with open(data_file, 'rb') as handle:
                df_dict = pickle.load(handle)
        except:
            print("load data error")
            return None
        df_list = []
        if freq == 'd':
            field = 'daily_data'
            df_dict['min_data'] = None
        else:
            field = 'min_data'
            df_dict['daily_data'] = None

        df = pd.DataFrame()
        for key in df_dict[field]:
            print("processing %s" % key)
            tdf = df_dict[field][key]
            if field == 'min_data':
                if key[-2:] == 'c2':
                    print("skip %s" % key)
                    df_dict[field][key] = None
                    continue
                tdf.drop(columns=["diff_oi", "contmth"], inplace=True)
                for col in ['open', 'high', 'low', 'close', 'shift']:
                    tdf[col] = tdf[col].astype('float32')
                for col in ['volume', 'openInterest', 'min_id']:
                    tdf[col] = tdf[col].astype('int32')
            tdf.columns = pd.MultiIndex.from_tuples([(key, col) for col in tdf.columns])
            df = pd.concat([df, tdf], axis=1, join='outer', copy=False)
            df_dict[field][key] = None
            del tdf
            gc.collect()
        try:
            df.to_parquet(f"C:/dev/data/fut_{freq}_%s.parquet" % tday.strftime("%Y%m%d"))
            print(f"fut_{freq} data saved")
        except:
            print(f"fut_{freq} save error")
    return df


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        now = datetime.datetime.now()
        tday = now.date()
        if (~is_workday(tday, 'CHN')) or (now.time() < datetime.time(17, 00, 0)):
            tday = day_shift(tday, '-1b', CHN_Holidays)
    # refresh_saved_fut_prices(run_date=tday)
    # _ = load_saved_fut(tday=tday, freq='d')
    _ = load_saved_fut(tday=tday, freq='m')

