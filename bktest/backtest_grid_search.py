import copy
import numpy as np
import math
import matplotlib as mpl
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import datetime
import itertools
import pandas as pd

from pycmqlib3.utility import dbaccess, dataseries, misc
import pycmqlib3.analytics.data_handler as dh

from pycmqlib3.analytics.tstool import *
from pycmqlib3.analytics.btmetrics import *
from pycmqlib3.analytics.backtest_utils import *

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'SM', "SF", 'nr', 'SA', 'UR'] # 'ZC',
base_metal_mkts = ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss']
precious_metal_mkts = ['au', 'ag']
ind_metal_mkts = ferrous_products_mkts + ferrous_mixed_mkts + base_metal_mkts
petro_chem_mkts = ['l', 'pp', 'v', 'TA', 'MA', 'bu', 'sc', 'fu', 'eg', 'eb', 'lu', 'pg', 'PF']
ind_all_mkts = ind_metal_mkts + petro_chem_mkts
ags_oil_mkts = ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'b'] #, 'b']
ags_soft_mkts = ['CF', 'SR', 'jd', 'AP', 'sp', 'CJ', 'lh', 'PK', 'CY'] # 'CY',]

ags_all_mkts = ags_oil_mkts + ags_soft_mkts

eq_fut_mkts = ['IF', 'IH', 'IC']
bond_fut_mkts = ['T', 'TF', 'TS']

fin_all_mkts = eq_fut_mkts + bond_fut_mkts
commod_all_mkts = ind_all_mkts + ags_all_mkts + precious_metal_mkts
all_markets = commod_all_mkts + fin_all_mkts

daily_start_dict = { 'c': datetime.date(2011,1,1), 'm': datetime.date(2011,1,1),
    'y': datetime.date(2011,1,1), 'l': datetime.date(2011,1,1), 'rb':datetime.date(2011,1,1),
    'p': datetime.date(2011,1,1), 'cu':datetime.date(2011,1,1), 'al':datetime.date(2011,1,1),
    'zn':datetime.date(2011,1,1), 'au':datetime.date(2015,12,1), 'v': datetime.date(2011,1,1),
    'a': datetime.date(2011,1,1), 'ru':datetime.date(2011,1,1), 'ag':datetime.date(2012,6,1),
    'i': datetime.date(2014,1,1), 'j': datetime.date(2012,6,1), 'jm':datetime.date(2013,7,1),
    'CF':datetime.date(2012,5,1),  'TA':datetime.date(2012,4,15),
    'PM':datetime.date(2013,10,1), 'RM':datetime.date(2013,1,1),  'SR':datetime.date(2013,1,1),
    'FG':datetime.date(2013,1,1),  'OI':datetime.date(2013,5,1),  'RI':datetime.date(2013,1,1),
    'WH':datetime.date(2014,5,1),  'pp':datetime.date(2014,5,1),
    'IF':datetime.date(2010,5,1),  'MA':datetime.date(2012,1,1),  'TF':datetime.date(2019,6,1),
    'IH':datetime.date(2015,5,1),  'IC':datetime.date(2015,5,1),  'cs':datetime.date(2015,2,1),
    'jd':datetime.date(2014,5,1),  'ni':datetime.date(2015,9,1),  'sn':datetime.date(2017,5,1),
    'ZC':datetime.date(2013,11,1), 'hc':datetime.date(2016, 4, 1), 'SM': datetime.date(2017,1,1),
    'SF': datetime.date(2017,9,1), 'CY': datetime.date(2017, 9, 1), 'AP': datetime.date(2018, 1, 1),
    'TS': datetime.date(2018, 9, 1), 'fu': datetime.date(2018, 9, 1), 'sc': datetime.date(2018, 10, 1),
    'b': datetime.date(2018, 1, 1), 'pb': datetime.date(2016, 7, 1), 'bu': datetime.date(2015,9,15),
    'T':datetime.date(2019,4,1), 'ss': datetime.date(2020, 5, 1), 'sp': datetime.date(2019, 5, 1),
    'CJ': datetime.date(2019, 8, 9), 'UR': datetime.date(2019, 8, 9), 'SA': datetime.date(2020, 1, 1),
    'eb': datetime.date(2020, 2, 1), 'eg': datetime.date(2019, 4, 2), 'rr': datetime.date(2019, 9, 1),
    'pg': datetime.date(2020, 9, 5), 'lu': datetime.date(2020, 10, 1), 'nr': datetime.date(2020,1,1),
    'lh': datetime.date(2021,5,1), 'PF': datetime.date(2021,1,1), 'PK': datetime.date(2021,4,1), }


def transform_output(pnl_stats, metrics=['sharpe', 'std', 'sortino']):
    df_list = []
    for key in metrics:
        adf = pnl_stats[key].reset_index()
        adf['index'] = adf['index'].apply(lambda x: x.split('_')[1] if '_' in x else 'all')
        adf = adf.rename(columns={'index': 'tenor', 'total': key}).set_index('tenor')
        df_list.append(adf)
    perf_df = pd.concat(df_list, axis=1, join='outer')
    return perf_df


def load_hist_data(start_date, end_date,
                   roll_name='hot',
                   sim_markets=all_markets,
                   freq='d',
                   roll_file_loc="C:/dev/wtdev/config/",
                   shift_mode=1):
    field_list = ['contract', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'diff_oi', 'expiry', 'mth',
                  'shift']
    nb_cont = 2
    data_df = pd.DataFrame()
    error_list = []
    for prodcode in sim_markets:
        for nb in range(nb_cont):
            try:
                xdf = dataseries.nearby(prodcode,
                                        nb + 1,
                                        start_date=start_date,
                                        end_date=end_date,
                                        shift_mode=shift_mode,
                                        freq=freq,
                                        roll_name=roll_name,
                                        config_loc=roll_file_loc)
                xdf['expiry'] = xdf['contract'].map(misc.contract_expiry)
                xdf['contmth'] = xdf['contract'].map(misc.inst2contmth)
                xdf['mth'] = xdf['contmth'].apply(lambda x: x // 100 * 12 + x % 100)
                xdf['product'] = prodcode
                xdf['code'] = f'c{nb + 1}'
                data_df = data_df.append(xdf)
            except:
                error_list.append((prodcode, nb))

    df = pd.pivot_table(data_df.reset_index(),
                        index='date',
                        columns=['product', 'code'],
                        values=field_list,
                        aggfunc='last')
    df = df.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
    df.columns.rename(['product', 'code', 'field', ], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df, error_list


def run_grid_btest(df, start_date, end_date, sim_type, signal_name,
                   index_list=range(10, 250, 10),
                   column_list=[1, 3, 5, 10, 15, 20],
                   product_list=ind_all_mkts,
                   pnl_tenors=True,
                   exp_mean=False,
                   shift_mode=1,
                   save_loc="C:/dev/data/data_cache",
                   perf_metric='sharpe'):
    xdf = df.copy()
    rev_char = '!'
    exec_mode = 'open'
    total_risk = 4600.0
    asset_scaling = True
    cost_ratio = 0.0
    std_win = 20
    crv_param = 2.0

    pos_map = (None, {}, '')
    params = [0.0, 0.0]

    metrics_dict = {}
    stats_dict = {}

    scenarios = list(itertools.product(index_list, column_list))

    for scen in scenarios:
        scen_x = scen[0]
        scen_y = scen[1]
        quantile = 0.2
        if signal_name in ['ryield']:
            win = 1
            ma_win = 1
            rebal = scen_x
            pos_map = (dh.response_curve, {'param': crv_param, "response": scen_y}, scen_y)
            if sim_type == 'xscarry':
                quantile = scen_x * 0.1
        elif signal_name in ['basmom', 'mom', 'clbrk', 'hlbrk', ]:
            win = scen_x
            ma_win = 1
            rebal = scen_y
        elif signal_name in ['macddff']:
            win = scen_x
            rebal = scen_y
            params = params
        elif signal_name in ['skew!ema']:
            win = scen_x
            ma_win = scen_y
            rebal = 1
            params = [0.0, 0.0]
        elif signal_name in ['mixmom']:
            win = scen_x
            ma_win = 1
            rebal = scen_y
        elif ('ts' in sim_type) and (signal_name in ['ryieldxma', 'ryieldema', 'ryieldnma', 'ryieldnmb',
                                                     'ryieldzlv', 'ryieldelv', 'ryieldqtl', 'ryieldsma',
                                                     'lrskewsma', 'lrkurtsma', 'trdstrsma', 'upstdsma',
                                                     'volmfratiosma']):
            win = 1
            ma_win = scen_x
            rebal = scen_y
            # pos_map = (dh.response_curve, {'param': crv_param, "response": scen_y}, scen_y)
        elif ('xs' in sim_type) and (signal_name in ['ryieldxma', 'ryieldema', 'ryieldnma', 'ryieldnmb',
                                                     'ryieldzlv', 'ryieldelv', 'ryieldqtl', 'ryieldsma',
                                                     'lrskewsma', 'lrkurtsma', 'trdstrsma', 'upstdsma',
                                                     'volmfratiosma']):
            win = 1
            ma_win = scen_x
            rebal = scen_y
        elif signal_name in ['basmomxma', 'basmomsma', 'basmomnma', 'basmomnmb', 'basmomzlv', 'basmomelv', 'basmomqtl',
                             'momsma', 'momxma', 'momnma', 'momnmb', 'momzlv', 'macdnma']:
            win = scen_x
            ma_win = scen_y
            rebal = 1
            if signal_name in ['macdnma']:
                params = [scen_y, 80.0 / win]
        else:
            print("unsupported run_mode")
            continue

        run_name = '-'.join([sim_type, signal_name, str(scen_x), str(scen_y)])

        run_args = {}
        run_args['shift_mode'] = shift_mode
        run_args['exp_mean'] = exp_mean
        run_args['exec_mode'] = exec_mode
        run_args['rev_char'] = rev_char
        run_args['cost_ratio'] = cost_ratio
        run_args['total_risk'] = total_risk
        run_args['asset_scaling'] = asset_scaling
        run_args['pnl_tenors'] = pnl_tenors
        run_args['std_win'] = std_win
        run_args['xs_signal'] = ''
        run_args['xs_params'] = {'cutoff': quantile}

        run_args['start_date'] = start_date
        run_args['end_date'] = end_date
        run_args['product_list'] = product_list

        run_args['signal_name'] = signal_name
        run_args['win'] = win
        run_args['ma_win'] = ma_win
        run_args['rebal_freq'] = rebal
        run_args['params'] = params
        run_args['pos_map'] = pos_map
        run_args['xs_params'] = {'cutoff': 0.2}
        if 'xs' in sim_type:
            sim_split = sim_type.split('-')
            if len(sim_split) > 1:
                run_args['xs_signal'] = sim_split[1]
            else:
                run_args['xs_signal'] = 'rank_cutoff'
        else:
            run_args['xs_signal'] = ''
        if len(scen) > 8:
            run_args['xs_params'] = {'cutoff': scen[8]}

        bt_metrics = run_backtest(xdf, run_args)
        metrics_dict[run_name] = bt_metrics
        pnl_stats = bt_metrics.calculate_pnl_stats(shift=0, tenors=pnl_tenors)
        stats_dict[run_name] = pnl_stats

    data_df = pd.DataFrame()
    for (scen_x, scen_y) in scenarios:
        key = '-'.join([sim_type, signal_name, str(scen_x), str(scen_y)])
        pnl_stats = stats_dict[key]
        stat_df = pnl_stats[perf_metric].to_frame(perf_metric)
        stat_df.index.name = 'tenor'
        stat_df = stat_df.reset_index()
        stat_df['X'] = scen_x
        stat_df['Y'] = scen_y
        data_df = data_df.append(stat_df)

    save_file = f"{save_loc}/{sim_type}_{signal_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
    writer = pd.ExcelWriter(save_file, engine='xlsxwriter')
    metric_keys = data_df['tenor'].unique()
    for key in metric_keys:
        adf = pd.pivot_table(data_df[data_df['tenor'] == key], values=['sharpe'], index=['Y'], columns=['X'])
        print(f'{key}\n', adf,'\n\n')
        adf.to_excel(writer, sheet_name=key, startcol=1, startrow=2)
    writer.close()

    return metrics_dict, stats_dict

