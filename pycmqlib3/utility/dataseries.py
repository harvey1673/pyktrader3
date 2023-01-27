import numpy as np
import pandas as pd
import json
from typing import Union, List
import datetime
from pycmqlib3.utility import misc
from pycmqlib3.utility.dbaccess import prod_main_cont_exch
from pycmqlib3.utility.process_wt_data import load_fut_by_product, \
    load_bars_to_df, date_to_int, datetime_to_int, int_to_datetime


def multislice_many(df, label_map):
    idx_label_map = {idx: label_map[label] for idx, label in enumerate(df.columns.names) if label in label_map}
    num_levels = len(df.columns.names)
    idx_slice = tuple([idx_label_map.get(i, slice(None)) for i in range(num_levels)])
    return df.loc[:, idx_slice]


class DataSeriesError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def process_expiry(data_series, exp_list):
    if exp_list is None:
        return
    for expiries in exp_list:
        rows, cols = expiries.shape
        if rows != 1:
            raise DataSeriesError('Expecting one row of expiries, found {}'.format(rows))
        contracts = expiries.columns
        for c in contracts:
            idx = expiries[c][0]
            if c in data_series.columns:
                data_series.loc[data_series.index > idx, c] = np.nan


def prod_main_cont_filter(df, asset):
    contlist, exch = prod_main_cont_exch(asset)
    if asset == 'ni':
        flag = (df.expiry < datetime.date(2019, 6, 1)) & df.month.isin([1, 5, 9])
        flag = flag | ((df.expiry >= datetime.date(2019, 6, 1)) & df.month.isin(contlist))
    elif asset =='sn':
        flag = (df.expiry < datetime.date(2020, 6, 1)) & df.month.isin([1, 5, 9])
        flag = flag | ((df.expiry >= datetime.date(2020, 6, 1)) & df.month.isin(contlist))
    else:
        flag = df.month.isin(contlist)
    return flag


def load_processed_fut_by_product(prodcode, start_date=None, end_date=None, freq='d',
                                  roll_win=3, roll_cutoff='-20b', cont_ratio=[1.0, 1.0],
                                  contract_filter=None, min_thres=0):
    exch = misc.prod2exch(prodcode)
    code = exch + '.' + prodcode
    xdf = load_fut_by_product(code, start_date, end_date, freq=freq)
    inst_list = xdf['instID'].unique()
    expiry_map = dict([(inst, misc.contract_expiry(inst, hols=misc.CHN_Holidays)) for inst in inst_list])
    # expiry_inv_map = {str(v): k for k, v in expiry_map.items()}
    xdf['expiry'] = xdf['instID'].map(expiry_map)
    xdf['exp_str'] = xdf['expiry'].astype('str')
    xdf['month'] = xdf['instID'].apply(lambda x: misc.inst2contmth(x) % 100)
    if contract_filter:
        flag = contract_filter(xdf, prodcode)
        xdf = xdf[flag]
    if freq == 'd':
        index_cols = ['date']
    else:
        index_cols = ['date', 'min_id']
    xdf = xdf.sort_values(['instID'] + index_cols)
    if (roll_cutoff[0] == '-') and (roll_cutoff[-1] in ['b', 'd']):
        xdf['roll_date'] = xdf['expiry'].apply(lambda x: misc.day_shift(x, roll_cutoff, hols=misc.CHN_Holidays))
        xdf = xdf[xdf.date <= xdf['roll_date']]
    else:
        xdf['roll_date'] = xdf['expiry']
    xdf['roll_ind'] = (xdf['volume'] * cont_ratio[0] + xdf['openInterest'] * cont_ratio[1]).rolling(roll_win).mean()
    xdf['log_ret'] = np.log(xdf['close']).diff()
    xdf.loc[xdf['instID'] != xdf['instID'].shift(1), 'log_ret'] = 0
    xdf['price_chg'] = xdf['close'].diff()
    xdf.loc[xdf['instID'] != xdf['instID'].shift(1), 'price_chg'] = 0
    for w in range(1, roll_win):
        if w == 1:
            flag = (xdf['instID'].shift(w) != xdf['instID'])
        else:
            flag = flag | (xdf['instID'].shift(w) != xdf['instID'])
    if roll_win > 1:
        xdf.loc[flag, 'roll_ind'] = np.nan
    xdf['roll_ind'] = xdf['roll_ind'].fillna(method='bfill')
    xdf = xdf[xdf['roll_ind'] >= min_thres]
    return xdf


def rolling_fut_cont(xdf, nb_cont=2, cont_thres=10_000, roll_mode=1, curr_roll=pd.DataFrame()):
    roll_df = pd.pivot_table(xdf, index=['date'], columns='expiry', values='roll_ind', aggfunc='first')
    inst_df = pd.pivot_table(xdf, index=['date'], columns='expiry', values='exp_str', aggfunc='first')
    if len(curr_roll) > 0:
        curr_date = curr_roll.index[-1]
        curr_list = curr_roll.iloc[-1].values[:nb_cont]
        roll_df = roll_df[roll_df.index > curr_date]
        inst_df = inst_df[inst_df.index > curr_date]
    else:
        curr_list = []
    inst_list = list(set(np.append(xdf['instID'].unique(), curr_list)))
    expiry_map = dict([(inst, misc.contract_expiry(inst, hols=misc.CHN_Holidays).strftime("%Y-%m-%d"))
                       for inst in inst_list])
    expiry_inv_map = {str(v): k for k, v in expiry_map.items()}
    curr_list = [expiry_map[inst] for inst in curr_list]

    if roll_mode % 10 == 1:
        # roll roll_mode=1 find the top N threshold
        thres = roll_df.apply(lambda row: min(row.nlargest(nb_cont).values[-1], cont_thres), axis=1)
        inst_df = inst_df[roll_df.ge(thres, axis='rows')]
    inst_df = inst_df.apply(lambda x: pd.Series(x.dropna().values), axis=1)
    inst_df = inst_df.reset_index()
    data_list = []
    date_list = []
    for alist in inst_df.to_numpy():
        if len(curr_list) == 0:
            curr_date = alist[0]
            curr_list = alist[1:nb_cont + 1]
            curr_list = curr_list[~pd.isnull(curr_list)]
            curr_list.sort()
        else:
            alist = alist[~pd.isnull(alist)]
            exp_list = np.array([e for e in alist[1:] if (e in curr_list) or (e > curr_list[-1])])[:nb_cont]
            exp_list = exp_list[~pd.isnull(exp_list)]
            exp_list.sort()
            if ((roll_mode // 10 == 1) and ((len(exp_list) < nb_cont) or (np.array_equal(exp_list, curr_list)))) or \
                    ((roll_mode // 10 == 0) and ((len(exp_list) == 0) or (exp_list[0] == curr_list[0]))):
                continue
            else:
                curr_list = exp_list
                curr_date = alist[0]
        data_list.append(np.array([expiry_inv_map[e] for e in curr_list]))
        date_list.append(curr_date)

    try:
        data_list = [np.pad(data, (0, nb_cont - len(data)), mode='constant', constant_values=(np.nan, np.nan)) for data in data_list]
        new_roll = pd.DataFrame(data_list, columns=[str(i) for i in range(nb_cont)], index=date_list)
        new_roll.index.name = 'date'
    except ValueError:
        print(data_list, date_list)
        return {'roll_map': pd.DataFrame(), 'new': pd.DataFrame()}
    roll_map = curr_roll.append(new_roll)
    results = {'roll_map': roll_map, 'new': new_roll}
    return results


def nearby(code, n=1, start_date=None, end_date=None,
           freq='d', shift_mode=0, roll_name='hots',
           config_loc="C:/dev/wtdev/config/"):
    if '.' in code:
        exch, prodcode = code.split('.')
    else:
        prodcode = code
        exch = misc.prod2exch(prodcode)
    roll_file = f'{config_loc}{roll_name}{n}.json'
    try:
        with open(roll_file, 'r') as infile:
            roll_dict = json.load(infile)
        roll_list = roll_dict[exch][prodcode]
    except:
        print("cant find roll file %s" % roll_file)
        return pd.DataFrame()
    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365)
    hols = misc.get_hols_by_exch(exch)
    start_date = misc.day_shift(misc.day_shift(start_date, '-1b', hols), '1b', hols)
    end_date = misc.day_shift(misc.day_shift(end_date, '1b', hols), '-1b', hols)
    start_date = date_to_int(start_date)
    end_date = date_to_int(end_date)
    if freq in ['m', 'd']:
        period = f'{freq}1'
    else:
        period = freq
    df = pd.DataFrame()
    s_date = start_date
    n_period = len(roll_list)
    for idx, row in enumerate(roll_list):
        s_date = max(s_date, row['date'])
        if idx == n_period - 1:
            e_date = end_date
        else:
            nxt_date = roll_list[idx+1]['date']
            if nxt_date <= s_date:
                continue
            e_date = misc.day_shift(int_to_datetime(nxt_date), '-1b', misc.CHN_Holidays)
            e_date = min(end_date, date_to_int(e_date))
        if s_date <= e_date:
            product, contmth = misc.inst2product(row['to'], rtn_contmth=True)
            code = '.'.join([exch, product, str(contmth)])
            if 'd' in period:
                stime = s_date * 10000
                etime = e_date * 10000
            else:
                stime = misc.day_shift(int_to_datetime(s_date), '-1b', misc.CHN_Holidays)
                stime = datetime_to_int(stime, 2100)
                etime = e_date * 10000 + 1515
            new_df = load_bars_to_df(code, period=period,
                                     start_time=stime,
                                     end_time=etime)
            if len(new_df) > 0:
                new_df['shift'] = 0.0
            else:
                continue
        else:
            break
        if len(df) > 0 and shift_mode > 0:
            if row['newclose'] == 0.0 or row['oldclose'] == 0.0:
                print("warning: no close for contract %s or %s" % (row['from'], row['to']))
                continue
            if shift_mode == 1:
                shift = row['newclose'] - row['oldclose']
                df['shift'] = df['shift'] + shift
                for ticker in ['open', 'high', 'low', 'close', 'settle']:
                    if ticker in df.columns:
                        df[ticker] = df[ticker] + shift
            else:
                shift = row['newclose']/row['oldclose']
                df['shift'] = df['shift'] + np.log(shift)
                for ticker in ['open', 'high', 'low', 'close', 'settle']:
                    if ticker in df.columns:
                        df[ticker] = df[ticker] * shift
        df = df.append(new_df)
    return df.rename(columns={'instID': 'contract'})


def invert_dict(old_dict, return_flat = False):
    new_dict = {}
    for key, value in old_dict.items():
        for string in list(value):
            if return_flat == False:
                new_dict.setdefault(string, []).append(key)
            else:
                new_dict.update({string: key})
    return new_dict


def make_seasonal_df(ser, limit = 1, fill = False, weekly_dense = False):
    df =ser.to_frame('data')
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    elif isinstance(df.index, pd.Index):
        df.index = pd.to_datetime(df.index)
    else:
        pass

    df['year'] = df.index.year

    if weekly_dense and isinstance(ser.index, pd.PeriodIndex) and ser.index.freqstr.staretswith('W'):
        start = pd.datetime.today() - pd.offsets.YearBegin()
        end = pd.datetime.today() + pd.offsets.YearEnd()
        df['date_s'] = df.index.week
        pr_df = pd.period_range(start, end, freq = ser.index.freq).to_frame()
        pr_df['week'] = pr_df.index.week
        pr_df.index = pr_df.index.end_time.to_period('D')
        df['date_s'] = df['date_s'].map(invert_dict(pr_df['week'].to_dict(), return_flat = True))
    else:
        df['date_s'] = df.index.map(lambda t: t.replace(year = 2020))
    df = pd.pivot_table(df, values = 'data', index = 'date_s', columns = 'year', aggfunc=np.sum)

    if fill:
        df = df.fillna(method = 'ffill', limit = limit)

    if type(ser.index) == pd.PeriodIndex and ser.index.freqstr[0] == 'W':
        df = df.ffill(limit = 4)
    
    return df


def get_level_index(df: pd.DataFrame, level=Union[str, int]) -> int:
    """
    get the level index of `df` given `level`
    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level
    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    """
    fetch data from `data` with `selector` and `level`
    Parameters
    ----------
    selector : Union[pd.Timestamp, slice, str, list]
        selector
    level : Union[int, str]
        the level to use the selector
    Returns
    -------
    Data of the given index.
    """
    # level = None -> use selector directly
    if level == None:
        return df.loc(axis=0)[selector]
    # Try to get the right index
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[
                    pd.IndexSlice[idx_slc],
                ]
        else:
            return df
    else:
        return df.loc[
            pd.IndexSlice[idx_slc],
        ]


def convert_index_format(df: Union[pd.DataFrame, pd.Series], level: str = "datetime") -> Union[pd.DataFrame, pd.Series]:
    """
    Convert the format of df.MultiIndex according to the following rules:
        - If `level` is the first level of df.MultiIndex, do nothing
        - If `level` is the second level of df.MultiIndex, swap the level of index.
    NOTE:
        the number of levels of df.MultiIndex should be 2
    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        raw DataFrame/Series
    level : str, optional
        the level that will be converted to the first one, by default "datetime"
    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        converted DataFrame/Series
    """

    if get_level_index(df, level=level) == 1:
        df = df.swaplevel().sort_index()
    return df
