import numpy as np
import pandas as pd
import json
import datetime
from pycmqlib3.utility import misc
from pycmqlib3.utility.dbaccess import prod_main_cont_exch
from pycmqlib3.utility.process_wt_data import load_fut_by_product, load_bars_to_df
from matplotlib import font_manager
font = font_manager.FontProperties(fname='C:/windows/fonts/simsun.ttc')


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
    xdf['roll_ind'] = xdf['roll_ind'].bfill()
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
    roll_map = pd.concat([curr_roll, new_roll])
    results = {'roll_map': roll_map, 'new': new_roll}
    return results


def nearby(code, n=1, start_date=None, end_date=None,
           freq='d', shift_mode=0, roll_name='hot',
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
    else:
        end_date = pd.to_datetime(str(end_date)).date()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365)
    else:
        start_date = pd.to_datetime(str(start_date)).date()
    hols = misc.get_hols_by_exch(exch)
    if freq in ['m', 'd']:
        period = f'{freq}1'
    else:
        period = freq
    start_date = misc.day_shift(misc.day_shift(start_date, '-1b', hols), '1b', hols)
    end_date = misc.day_shift(misc.day_shift(end_date, '1b', hols), '-1b', hols)

    s_date = start_date
    n_period = len(roll_list)
    df = pd.DataFrame()
    for idx, row in enumerate(roll_list):
        s_date = max(s_date, pd.to_datetime(str(row['date'])).date())
        if idx == n_period - 1:
            e_date = end_date
        else:
            nxt_date = pd.to_datetime(str(roll_list[idx+1]['date'])).date()
            if nxt_date <= s_date:
                continue
            e_date = min(end_date, misc.day_shift(nxt_date, '-1b', hols))
        if s_date <= e_date:
            product, contmth = misc.inst2product(row['to'], rtn_contmth=True)
            code = '.'.join([exch, product, str(contmth)])
            new_df = load_bars_to_df(code, period=period, start_time=s_date, end_time=e_date)
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
        df = pd.concat([df, new_df])
    return df.rename(columns={'instID': 'contract'})
