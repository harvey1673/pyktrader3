import numpy as np
import pandas as pd
from typing import Union, List
import datetime
from pycmqlib3.utility import misc
from pycmqlib3.utility.dbaccess import load_fut_by_product, prod_main_cont_exch


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


def nearby(prodcode, n = 1, start_date = None, end_date = None, 
           roll_rule = '-20b', freq = 'd', shift_mode = 0, 
           adj_field = 'close', calc_fields = ['open', 'close', 'high', 'low'], 
           contract_filter = prod_main_cont_filter, fill_cont = False,
          ):
    exch = misc.prod2exch(prodcode)
    xdf = load_fut_by_product(prodcode, exch, start_date ,end_date, freq = freq)
    xdf['expiry'] = xdf['instID'].apply(lambda x: misc.contract_expiry(x, hols = misc.CHN_Holidays))
    xdf['month'] = xdf['instID'].apply(lambda x: misc.inst2contmth(x)%100)
    if freq == 'd':
        index_cols = ['date']        
    elif freq == 'm':
        index_cols = ['date', 'min_id']
    xdf = xdf.sort_values(['instID'] + index_cols)
    if shift_mode == 2:
        xdf['price_chg'] = np.log(xdf[adj_field]).diff()
    else:
        xdf['price_chg'] = xdf[adj_field].diff()      
    xdf.loc[xdf['instID']!=xdf['instID'].shift(1), 'price_chg'] = 0
    if (roll_rule[0] == '-') and (roll_rule[-1] in ['b', 'd']):
        xdf['roll_date'] = xdf['expiry'].apply(lambda x: misc.day_shift(x, roll_rule, hols = misc.CHN_Holidays))
        xdf = xdf[xdf.date <= xdf['roll_date']]
    else:
        xdf['roll_date'] = xdf['expiry']
    if contract_filter:
        flag = contract_filter(xdf, prodcode)
        xdf = xdf[flag]
    df = pd.pivot_table(xdf, index = index_cols, columns = 'expiry', values = 'instID', aggfunc = 'first')
    df1 = df.apply(lambda x: pd.Series(x.dropna().values), axis=1)
    df1 = df1.reset_index()
    col_df = df1[['date', n-1]].rename(columns = {n-1: 'instID'})
    if len(col_df[col_df['instID'].isna()])> 0:
        if fill_cont:
            col_df = col_df.fillna(method = 'ffill')
        else:
            raise ValueError('There are nan values for product=%s, nearby=%s, roll=%s, dates = %s' % 
            (prodcode, str(n), roll_rule, col_df[col_df['instID'].isna()]['date']))
    out_df = pd.merge(col_df, xdf,left_on = ['date', 'instID'], right_on=['date', 'instID'], how = 'left')
    if shift_mode > 0:
        cum_adj = out_df.loc[::-1, 'price_chg'].cumsum().shift(1).fillna(0)[::-1]
        if shift_mode == 2:
            adj_price = out_df[adj_field].iloc[-1]/np.exp(cum_adj)
            out_df['shift'] = np.log(adj_price) - np.log(out_df[adj_field])
        else:
            adj_price = out_df[adj_field].iloc[-1] - cum_adj
            out_df['shift'] = adj_price - out_df[adj_field]    

        for cfield in calc_fields:
            if shift_mode == 2:
                out_df[cfield] = out_df[cfield] * np.exp(out_df['shift'])
            else:
                out_df[cfield] = out_df[cfield] + out_df['shift']
    else:
        out_df['shift'] = 0
    out_df = out_df.set_index('date').rename(columns = {'instID': 'contract'})
    return out_df

# def _move_rows_np2(roll_index, col_idx, ds, nan_value = np.nan):
#     print(ds)
#     nds = ds.values
#     len_ds = len(ds)
#     len_cols = len(ds.columns)
#     len_rollidx = len(roll_index)
#     rollidx = roll_index.values
#     col_index = col_idx.values
#     for j in range(len_rollidx - 1, 0, -1):
#         ri = rollidx[j-1]
#         ci = col_index[j]
#         if ri < 0:
#             ri = 0
#         np.copyto(nds[ri:len_ds, ci-1:len_cols -1], nds[ri:len_ds, ci:len_cols])
#         nds[ri:len_ds, len_cols-1:len_cols] = nan_value
#     ds[:] = nds
#     print(ds)


# def nearby(data_series, roll = None, fwd_expiries = None, clean_expiries = False, nan_value = np.nan, hols = 'CHN'):
#     ds = data_series.copy(deep = True)
#     if clean_expiries:
#         process_expiry(ds, fwd_expiries)
#     ds = ds.dropna(axis=0, how = 'all')
#     ds = ds.dropna(axis=1, how = 'all')
#     if ds.empty:
#         return ds

#     ds_index = ds.index
#     is_forward_success = False
#     if (roll is not None) and (fwd_expiries is not None):
#         lastdt = pd.to_datetime(ds.index[-1:][0])
#         slbl = ds.iloc[-1:].dropna(axis=1).columns
#         first_contract = slbl[0]
#         next_expiry = None
#         for expiry in fwd_expiries:
#             if expiry is not None:
#                 if first_contract in expiry.columns:
#                     if next_expiry is None:
#                         next_expiry = pd.to_datetime(expiry[first_contract][0])
#                     elif pd.to_datetime(expiry[first_contract][0]) < next_expiry:
#                         next_expiry = pd.to_datetime(expiry[first_contract][0])
#         if next_expiry is not None:
#             edts = pd.date_range(start = lastdt.to_pydatetime() + pd.DateOffset(days = 1), 
#                                 end = next_expiry.to_pydatetime() + pd.DateOffset(days = 20), 
#                                 freq = 'B')
#             hol = dates.get_holidays()
#             fdts = edts.difference(edts.intersection(hol))
#             newdts = fdts[fdts <= fdts[fdts > next_expiry][0]]
#             newvals = pd.DataFrame(index = newdts)
#             ds = ds.append(newvals)
#             ds.loc[lastdt:] = ds.loc[lastdt:].fillna(method = 'pad')
#             ds.reset_index(drop = True, inplace = True)
#             ds.columns = list(range(0, len(ds.columns)))
#             last_idx = ds[-1:].notnull.idxmax(axis = 1)
#             ds.iloc[-1:, last_idx] = np.nan
#             is_forward_success = True
#         else:
#             if roll is None:
#                 roll = 0
#             ds.reset_index(drop = True, inplace = True)
#             ds.columns = list(range(0, len(ds.columns)))
#     else:
#         if roll is None:
#             roll = 0
#         ds.reset_index(drop = True, inplace = True)
#         ds.columns = list(range(0, len(ds.columns)))            
    
#     roll = int(roll)    
#     roll_index = ds.iloc[::-1].notnull().idxmax()
#     max_roll_index = roll_index.idxmax()
#     if (roll_index[max_roll_index+1:] < roll_index[max_roll_index]).any():
#         print('warning: later contract is truncated...')
#     roll_index[max_roll_index:] = roll_index[max_roll_index]
#     col_index = ds.columns
#     full_series = roll_index[roll_index == len(ds) - 1]
#     full_series = full_series[1:]
#     roll_index = roll_index.drop(full_series.index)
#     col_index = col_index.difference(col_index.intersection(full_series.index))
#     roll_index = roll_index - roll + 1
#     print(roll_index, col_index)
#     _move_rows_np2(roll_index, col_index, ds, nan_value)

#     if is_forward_success:
#         ds = ds.drop(ds[-len(newdts):].index)

#     ds.columns = [col for col in range(len(ds.columns))]
#     ds.index = ds_index
#     return ds


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
