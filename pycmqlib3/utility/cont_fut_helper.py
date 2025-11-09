import pandas as pd
import numpy as np


def build_roll_adj_px(df, expiry_dates, roll_day=0, mode='ret', contract=1):
    curr_cont = f"F{contract}"
    next_cont = f"F{contract+1}"
    df = df.sort_index()
    index = df.index
    if mode == 'ret':
        df = np.log(df)
    expiry_dates = pd.to_datetime(expiry_dates)
    aligned_expiry, aligned_roll = [], []
    for ed in expiry_dates:
        if ed < index[0]:
            continue
        elif ed > index[-1]:
            ed_aligned = ed
        else:
            ed_aligned = index[index <= ed].max()
        rd_nominal = ed_aligned - pd.tseries.offsets.BDay(roll_day)
        if rd_nominal > index[-1]:
            continue
        elif rd_nominal < index[0]:
            rd_aligned = rd_nominal
        else:
            rd_aligned = index[index <= rd_nominal].max()
        aligned_expiry.append(ed_aligned)
        aligned_roll.append(rd_aligned)
    if not aligned_expiry:
        raise ValueError("No expiry dates fall within the data index range")

    roll_diffs = []
    for rd in aligned_roll:
        if rd in df.index:
            diff = df.loc[rd, curr_cont] - df.loc[rd, next_cont]
            roll_diffs.append((rd, diff))
        else:
            roll_diffs.append((rd, 0))

    roll_df = pd.DataFrame(roll_diffs, columns=['roll_date', 'diff']).set_index('roll_date')
    adj_series_daily = roll_df['diff'].reindex(index=index, fill_value=0).shift(1).fillna(0).cumsum()

    unadj = df[curr_cont].copy()

    if roll_day > 0:
        for ed, rd in zip(aligned_expiry, aligned_roll):
            mask = (index > rd) & (index <= ed)
            unadj.loc[mask] += df.loc[mask, next_cont] - df.loc[mask, curr_cont]

    ts = unadj + adj_series_daily
    if mode == 'ret':
        ts = np.exp(ts)
        unadj = np.exp(unadj)
        adj_series_daily = np.exp(adj_series_daily)
    ts.name = curr_cont
    # debug = {"cont": unadj, 
    #          "adj": adj_series_daily,
    #          "roll_df": roll_df}
    return ts

