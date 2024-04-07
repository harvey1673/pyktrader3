from pycmqlib3.utility.misc import day_shift, sign, product_lotsize, CHN_Holidays
from pycmqlib3.utility.dbaccess import load_factor_data
from pycmqlib3.analytics.tstool import *
import pandas as pd
import numpy as np


def generate_strat_position(cur_date, prod_list, factor_repo,
                            repo_type='asset',
                            roll_label='CAL_30b',
                            freq='s1',
                            pos_scaler=1000,
                            fact_db_table='fut_fact_data',
                            hist_fact_lookback=100,
                            vol_key='atr'):
    if roll_label == 'CAL_30b':
        freq = 's1'
    res = {}
    fact_data = {}
    factor_pos = {}
    target_pos = {}
    vol_weight = [1.0] * len(prod_list)
    start_date = day_shift(cur_date, '-%sb' % (str(hist_fact_lookback)), CHN_Holidays)
    end_date = day_shift(day_shift(cur_date, '1b', CHN_Holidays), '-1d')
    cdates = pd.date_range(start=start_date, end=end_date, freq='D')
    bdates = pd.bdate_range(start=start_date, end=end_date, freq='C')
    vol_df = load_factor_data(prod_list,
                              factor_list=[vol_key],
                              roll_label=roll_label,
                              start=start_date,
                              end=end_date,
                              freq=freq,
                              db_table=fact_db_table)
    vol_df = pd.pivot_table(vol_df[vol_df['fact_name'] == vol_key], values='fact_val',
                            index='date',
                            columns=['product_code'],
                            aggfunc='last')

    fact_list = list(set([factor_repo[fact]['name'] for fact in factor_repo.keys()]))
    if repo_type == 'port':
        df = load_factor_data([], factor_list=fact_list, roll_label=roll_label,
                              start=start_date, end=end_date, freq=freq, db_table=fact_db_table)
    else:
        df = load_factor_data(prod_list, factor_list=fact_list, roll_label=roll_label,
                              start=start_date, end=end_date, freq=freq, db_table=fact_db_table)
    for idx, prod in enumerate(prod_list):
        vol_weight[idx] = vol_weight[idx]*pos_scaler/(vol_df[prod].iloc[-1]*product_lotsize[prod])
    if repo_type == 'port':
        xdf = pd.pivot_table(df, values='fact_val', index='date',
                             columns=['fact_name'], aggfunc='last')
        for fact in factor_repo:
            fact_data[fact] = pd.concat([xdf[fact]] * len(prod_list), axis=1)
            fact_data[fact].columns = prod_list
            if 'exec_assets' in factor_repo[fact]:
                for prod in factor_repo[fact]['exec_assets']:
                    if prod in prod_list:
                        fact_data[fact][prod] = np.nan
    else:
        for fact in factor_repo:
            xdf = pd.pivot_table(df[df['fact_name'] == factor_repo[fact]['name']],
                                 values='fact_val',
                                 index='date',
                                 columns=['product_code'],
                                 aggfunc='last')
            for prod in prod_list:
                if prod not in xdf.columns:
                    xdf[prod] = np.nan
                elif ('exec_assets' in factor_repo[fact]) and (prod in factor_repo[fact]['exec_assets']):
                    xdf[prod] = np.nan
            fact_data[fact] = xdf[prod_list].ffill()

    pos_sum = pd.DataFrame(index=prod_list)
    for fact in factor_repo:
        rebal = factor_repo[fact]['rebal']
        if type(rebal) == str:
            rebal_freq = int(rebal[3:])
            if rebal[:3] == 'sma':
                rebal_func = 'rolling'
            else:
                rebal_func = 'ewm'
        else:
            rebal_func = 'rolling'
            rebal_freq = rebal
        weight = factor_repo[fact]['weight']
        factor_pos[fact] = fact_data[fact].copy().ffill()
        if factor_repo[fact]['type'] != 'pos':
            if 'xs' in factor_repo[fact]['type']:
                xs_split = factor_repo[fact]['type'].split('-')
                if len(xs_split) <= 1:
                    xs_signal = 'rank_cutoff'
                else:
                    xs_signal = xs_split[1]
                if xs_signal == 'rank_cutoff':
                    cutoff = factor_repo[fact]['threshold']
                    lower_rank = int(len(prod_list) * cutoff) + 1
                    upper_rank = len(prod_list) - int(len(prod_list) * cutoff)
                    rank_df = factor_pos[fact].rank(axis=1)
                    factor_pos[fact] = rank_df.gt(upper_rank, axis=0) * 1.0 - rank_df.lt(lower_rank, axis=0) * 1.0
                elif xs_signal == 'demedian':
                    median_ts = factor_pos[fact].quantile(0.5, axis=1)
                    factor_pos[fact] = factor_pos[fact].sub(median_ts, axis=0)
                elif xs_signal == 'demean':
                    mean_ts = factor_pos[fact].mean(axis=1)
                    factor_pos[fact] = factor_pos[fact].sub(mean_ts, axis=0)
                elif xs_signal == 'rank':
                    rank_df = factor_pos[fact].rank(axis=1)
                    median_ts = rank_df.quantile(0.5, axis=1)
                    factor_pos[fact] = rank_df.sub(median_ts, axis=0)/len(prod_list) * 2.0
                elif xs_signal == 'xdemean':
                    factor_pos[fact] = xs_demean(factor_pos[fact])
                elif xs_signal == 'xscore':
                    factor_pos[fact] = xs_score(factor_pos[fact])
                elif len(xs_signal) > 0:
                    print('unsupported xs signal types')
            factor_pos[fact] = getattr(factor_pos[fact], rebal_func)(rebal_freq).mean().fillna(0.0)
        factor_pos[fact] = factor_pos[fact].ffill()
        pos_sum[fact] = pd.Series(factor_pos[fact].iloc[-1] * weight, name=fact)
    pos_sum['sum'] = pos_sum.sum(axis=1)
    pos_sum = pos_sum.round(2)
    for idx, prodcode in enumerate(prod_list):
        target_pos[prodcode] = pos_sum.loc[prodcode, 'sum'] * vol_weight[idx]
    res['target_pos'] = target_pos
    res['pos_sum'] = pos_sum.T
    res['vol_weight'] = vol_weight
    return res
