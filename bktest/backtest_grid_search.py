import datetime
import itertools
import pandas as pd

from pycmqlib3.utility import dataseries, misc
from pycmqlib3.analytics.tstool import *
from pycmqlib3.analytics.btmetrics import *
from pycmqlib3.analytics.backtest_utils import *
from misc_scripts.daily_update_job import scenarios_test, scenarios_elite

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'SM', "SF", 'nr', 'SA', 'UR'] # 'ZC',
base_metal_mkts = ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao', 'si', 'bc']
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


product_grouping_complete = {
    'ind': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SM', "SF", 'SA', 'UR', 'cu', 'al', 'zn', 'ni', 'sn', 'ss', 'au'],
    'petro': ['l', 'pp', 'v', 'TA', 'MA', 'bu', 'sc', 'fu', 'eg', 'eb', 'lu', 'pg', 'PF', 'ru', 'nr', 'CF', 'SR'],
    'ags': ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'b', 'jd', 'AP', 'sp', 'CJ', 'lh', 'PK', 'pb'],
}

product_grouping_partial = {
    'ind': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SM', 'UR', 'cu', 'al', 'zn', 'ni', 'sn', 'ss'],
    'petro': ['l', 'pp', 'v', 'TA', 'MA', 'sc', 'eg', 'ru', 'CF', 'SR', 'fu'],
    'ags': ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'jd', 'AP', 'CJ', 'pb', 'b'],
}

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
    'IF':datetime.date(2010,5,1),  'MA':datetime.date(2015,1,1),  'TF':datetime.date(2019,6,1),
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
    field_list = ['contract', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'expiry', 'mth', 'shift']
    nb_cont = 2
    data_df = pd.DataFrame()
    error_list = []
    for prodcode in sim_markets:
        for nb in range(nb_cont):
            try:
                if roll_name == 'CAL_30b':
                    roll = '-30b'
                    if prodcode in eq_fut_mkts:
                        roll = '0b'
                    elif prodcode in ['cu', 'al', 'zn', 'pb', 'sn', 'ss', 'lu']:
                        roll = '-25b'
                    elif prodcode in ['ni', 'jd', 'lh', 'eg',]:
                        roll = '-35b'
                    elif prodcode in ['v', 'MA', 'rb', 'hc']:
                        roll = '-28b'
                    elif prodcode in ['sc', 'eb'] + bond_fut_mkts:
                        roll = '-20b'
                    elif prodcode in precious_metal_mkts:
                        roll = '-15b'
                    sdate = max(start_date, daily_start_dict.get(prodcode, start_date))
                    adf = misc.nearby(prodcode, nb+1,
                                      start_date=sdate,
                                      end_date=end_date,
                                      shift_mode=shift_mode,
                                      freq=freq,
                                      roll_rule=roll).reset_index()
                else:
                    adf = dataseries.nearby(prodcode,
                                            nb+1,
                                            start_date=start_date,
                                            end_date=end_date,
                                            shift_mode=shift_mode,
                                            freq=freq,
                                            roll_name=roll_name,
                                            config_loc=roll_file_loc)
                adf['expiry'] = adf['contract'].map(misc.contract_expiry)
                adf['contmth'] = adf['contract'].map(misc.inst2contmth)
                adf['mth'] = adf['contmth'].apply(lambda x: x // 100 * 12 + x % 100)
                adf['product'] = prodcode
                adf['code'] = f'c{nb + 1}'
                data_df = pd.concat([data_df, adf])
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
                   product_name='all',
                   pnl_tenors=True,
                   exp_mean=False,
                   shift_mode=1,
                   save_loc="C:/dev/data/data_cache",
                   perf_metric='sharpe'):
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
        elif signal_name in ['basmomxma', 'basmomsma', 'basmomema',
                             'basmomnma', 'basmomnmb', 'basmomzlv',
                             'basmomelv', 'basmomqtl',
                             'momsma', 'momxma', 'momnma', 'momnmb',
                             'momzlv', 'momelv', 'momqtl', 'macdnma']:
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

        bt_metrics = run_backtest(df, run_args)
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

    save_file = f"{save_loc}/{sim_type}_{signal_name}_{product_name}_{start_date.strftime('%Y%m%d')}.xlsx"
    writer = pd.ExcelWriter(save_file, engine='xlsxwriter')
    metric_keys = data_df['tenor'].unique()
    for key in metric_keys:
        adf = pd.pivot_table(data_df[data_df['tenor'] == key], values=['sharpe'], index=['Y'], columns=['X'])
        adf.to_excel(writer, sheet_name=key, startcol=1, startrow=2)
    writer.close()

    return metrics_dict, stats_dict


def run_xs_product(df,
                   group_keys=['ind', 'petro', 'ags', 'all'],
                   sim_group=[
                       ('xscarry-demedian', 'ryieldnma'),
                       ('xscarry-demedian', 'ryieldsma'),
                       ('xscarry-demedian', 'basmomnma'),
                       # ('xscarry-rank', 'ryieldnma'),
                       # ('xscarry-rank_cutoff', 'ryieldnma'),
                       # ('xscarry-rank', 'basmomnma'),
                       # ('xscarry-rank_cutoff', 'basmomnma'),
                       # ('xscarry-rank', 'ryieldsma'),
                       # ('xscarry-rank_cutoff', 'ryieldsma'),
                   ]):
    run_markets = product_grouping_partial['ind'] + product_grouping_partial['petro'] + product_grouping_partial['ags']

    range_1 = range(10, 260, 10)
    range_2 = [10, 20, 40, 61, 122, 183, 244]
    range_3 = [1, 3, 5, 10]
    range_4 = [4, 8, 12, 16, 24, 32, 64]
    range_5 = [10, 20, 40, 80, 160, 320]
    range_by_signal = {
        'ryieldnma': [range_2, range_3],
        'ryieldnmb': [range_2, range_3],
        'basmomnma': [range_1, range_2],
        'ryieldsma': [range_2, range_3],
        'momnma': [range_1, range_2],
        'macdnma': [range_4, range_2],
        'hlbrk': [range_2, range_5],
        'momelv': [range_1, range_2],
        'momqtl': [range_1, range_2],
    }
    start_d = datetime.date(2012, 1, 1)
    end_d = datetime.date(2020, 1, 1)

    bt_metric_dict = {}
    pnl_stats_dict = {}

    for group_key in group_keys:
        if group_key not in product_grouping_partial:
            product_list = run_markets
        else:
            product_list = product_grouping_partial[group_key]
        bt_metric_dict[group_key] = {}
        pnl_stats_dict[group_key] = {}
        for sim_type, signal_name in sim_group:
            index_list = range_by_signal[signal_name][0]
            column_list = range_by_signal[signal_name][1]
            print(f"processing product = {group_key} for {sim_type} - {signal_name}")
            metric_dict, stat_dict = run_grid_btest(df, start_d, end_d,
                                                    sim_type, signal_name,
                                                    index_list=index_list,
                                                    column_list=column_list,
                                                    product_list=product_list,
                                                    product_name=group_key,
                                                    pnl_tenors=True,
                                                    exp_mean=False)
            bt_metric_dict[group_key][(sim_type, signal_name)] = metric_dict
            pnl_stats_dict[group_key][(sim_type, signal_name)] = stat_dict

    return bt_metric_dict, pnl_stats_dict


def run_scenarios(df,
                  start_date,
                  end_date,
                  product_list,
                  scenarios):
    file_folder = "C:\\dev\\data\\data_cache\\"
    scenario_config = {
        'shift_mode': 1,
        'exec_mode': 'open',
        'rev_char': '!',
        'cost_ratio': 0,
        'total_risk': 4600,
        'asset_scaling': False,
        'pnl_tenors': ['6m', '1y', '2y', '3y', '4y', '5y', '6y', '7y', '8y', '9y', '10y', '11y'],
        'std_win': 20,
        'xs_signal': '',
        'xs_params': {'cutoff': 0.2},
    }
    product_offsets = misc.product_trade_offsets(product_list)
    run_pos_sum = True
    pos_sum = pd.DataFrame()
    scen_names = []
    scen_metrics = []
    scen_stats = []

    port_start = pd.to_datetime('2019-01-01')

    for scen in scenarios:
        sim_type = scen[0]
        signal_name = scen[1]
        weight = scen[2]
        win = scen[3]
        ma_win = scen[4]
        rebal = scen[5]
        pos_map = scen[6]
        params = scen[7]
        run_name = '-'.join([sim_type, signal_name, str(win), str(ma_win), str(rebal)])

        run_args = copy.deepcopy(scenario_config)
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

        bt_metrics = run_backtest(df, run_args)
        scen_names.append(run_name)
        scen_metrics.append(bt_metrics)
        pnl_stats = bt_metrics.calculate_pnl_stats(shift=0, tenors=run_args['pnl_tenors'])
        scen_stats.append(pnl_stats)
        pnl_stats['portfolio_cumpnl'][port_start:].plot(title=run_name)
        plt.show()
        perf_stats = transform_output(pnl_stats)
        print(perf_stats.round(2))

        if run_pos_sum:
            pos_sum = pos_sum.add(bt_metrics.holdings * weight, fill_value=0)

    df_pxchg = get_px_chg(df, exec_mode=scenario_config['exec_mode'], chg_type='px', contract='c1')
    df_pxchg = df_pxchg[product_list].reindex(index=pos_sum.index)

    bt_met = MetricsBase(holdings=pos_sum,
                         returns=df_pxchg,
                         offsets=product_offsets,
                         cost_ratio=scenario_config['cost_ratio'])
    port_stats = bt_met.calculate_pnl_stats(shift=0, tenors=scenario_config['pnl_tenors'])
    port_stats['portfolio_cumpnl'][port_start:].plot(title="Total portfolio ")
    plt.show()
    perf_stats = transform_output(port_stats)
    print(perf_stats.round(2))

    bt_metrics = bt_met

    close_prices = df.loc[:,
                   (df.columns.get_level_values(1) == 'c1') & (df.columns.get_level_values(2) == 'close')].droplevel(
        [1, 2], axis=1)
    close_prices = close_prices[product_list]
    open_prices = df.loc[:,
                  (df.columns.get_level_values(1) == 'c1') & (df.columns.get_level_values(2) == 'open')].droplevel(
        [1, 2], axis=1)
    open_prices = open_prices[product_list]
    asset_pnl = bt_met.calculate_daily_pnl(open_prices, close_prices)
    port_pnl = asset_pnl.sum(axis=1).cumsum().to_frame('total')
    print(port_pnl[-40:])
    port_pnl.to_csv(file_folder + "port_pnl.csv")


def run_scenarios(tday=datetime.date.today(), roll_names=['hot', 'CAL_30b']):
    product_list = ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'cu', 'al', 'zn', 'pb', 'sn', \
                    'l', 'pp', 'v', 'TA', 'sc', 'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', \
                    'AP', 'SM', 'eb', 'eg', 'UR', 'ss', 'lu', 'lh', 'ni', ]

    for roll_name in roll_names:
        df, error_list = load_hist_data(
            start_date=datetime.date(2011, 1, 1),
            end_date=tday,
            roll_name=roll_name,
            sim_markets=product_list,
            freq='d')

        if len(error_list) > 0:
            print(error_list)



