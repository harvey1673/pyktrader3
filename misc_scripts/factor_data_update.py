import sys
import json
import copy
import logging

import numpy as np
from sqlalchemy import create_engine
from pycmqlib3.utility.dbaccess import dbconfig, mysql_replace_into, connect, load_factor_data
from pycmqlib3.utility import dataseries
from pycmqlib3.utility.misc import inst2product, prod2exch, inst2contmth, day_shift, \
    sign, is_workday, CHN_Holidays, nearby
import pycmqlib3.analytics.data_handler as dh
from pycmqlib3.analytics.tstool import *
from pycmqlib3.utility import base
from pycmqlib3.strategy.strat_util import generate_strat_position
from pycmqlib3.strategy.signal_repo import signal_buffer_config

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'ZC', 'SM', "SF"]
base_metal_mkts = ['cu', 'al', 'zn', 'pb', 'ni', 'sn']
precious_metal_mkts = ['au', 'ag']
ind_metal_mkts = ferrous_products_mkts + ferrous_mixed_mkts + base_metal_mkts
petro_chem_mkts = ['l', 'pp', 'v', 'TA', 'MA', 'bu']  # , 'sc', 'fu', 'eg']
ind_all_mkts = ind_metal_mkts + petro_chem_mkts
ags_oil_mkts = ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs']  # , 'b']
ags_soft_mkts = ['CF', 'CY', 'SR', 'jd', 'AP', 'UR', 'CJ']  # , 'sp', 'CJ', 'UR']
ags_all_mkts = ags_oil_mkts + ags_soft_mkts
eq_fut_mkts = ['IF', 'IH', 'IC', "IM"]
bond_fut_mkts = ['T', 'TF', 'TS', 'TL']
fin_all_mkts = eq_fut_mkts + bond_fut_mkts
commod_all_mkts = ind_all_mkts + ags_all_mkts + precious_metal_mkts
all_markets = commod_all_mkts + fin_all_mkts

trade_cont_map = {}

sim_start_dict = {'c': datetime.date(2011, 1, 1), 'm': datetime.date(2011, 1, 1),
                  'y': datetime.date(2011, 1, 1), 'l': datetime.date(2011, 1, 1), 'rb': datetime.date(2011, 1, 1),
                  'p': datetime.date(2011, 1, 1), 'cu': datetime.date(2011, 1, 1), 'al': datetime.date(2011, 1, 1),
                  'zn': datetime.date(2011, 1, 1), 'au': datetime.date(2011, 1, 1), 'v': datetime.date(2011, 1, 1),
                  'a': datetime.date(2011, 1, 1), 'ru': datetime.date(2011, 1, 1), 'ag': datetime.date(2012, 6, 1),
                  'i': datetime.date(2014, 1, 1), 'j': datetime.date(2012, 6, 1), 'jm': datetime.date(2013, 7, 1),
                  'CF': datetime.date(2012, 5, 1), 'TA': datetime.date(2012, 4, 15),
                  'PM': datetime.date(2013, 10, 1), 'RM': datetime.date(2013, 1, 1), 'SR': datetime.date(2013, 1, 1),
                  'FG': datetime.date(2013, 1, 1), 'OI': datetime.date(2013, 5, 1), 'RI': datetime.date(2013, 1, 1),
                  'WH': datetime.date(2014, 5, 1), 'pp': datetime.date(2014, 5, 1),
                  'IF': datetime.date(2010, 5, 1), 'MA': datetime.date(2012, 1, 1), 'TF': datetime.date(2019, 6, 1),
                  'IH': datetime.date(2015, 5, 1), 'IC': datetime.date(2015, 5, 1), 'cs': datetime.date(2015, 2, 1),
                  'jd': datetime.date(2014, 5, 1), 'ni': datetime.date(2015, 9, 1), 'sn': datetime.date(2017, 5, 1),
                  'ZC': datetime.date(2013, 11, 1), 'hc': datetime.date(2016, 4, 1), 'SM': datetime.date(2017, 1, 1),
                  'SF': datetime.date(2017, 9, 1), 'CY': datetime.date(2017, 9, 1), 'AP': datetime.date(2018, 1, 1),
                  'TS': datetime.date(2018, 9, 1), 'fu': datetime.date(2018, 9, 1), 'sc': datetime.date(2018, 8, 1),
                  'b': datetime.date(2018, 1, 1), 'pb': datetime.date(2016, 7, 1), 'bu': datetime.date(2015, 9, 15),
                  'T': datetime.date(2019, 4, 1), 'ss': datetime.date(2020, 5, 1), 'sp': datetime.date(2019, 5, 1),
                  'CJ': datetime.date(2019, 8, 9), 'UR': datetime.date(2019, 8, 9), 'SA': datetime.date(2020, 1, 1),
                  'eb': datetime.date(2020, 2, 1), 'eg': datetime.date(2019, 5, 1), 'rr': datetime.date(2019, 9, 1),
                  'pg': datetime.date(2020, 9, 5), 'lu': datetime.date(2020, 10, 1), 'nr': datetime.date(2020,1,1),
                  'lh': datetime.date(2021,5,1), 'PF': datetime.date(2021,1,1), 'PK': datetime.date(2021,4,1),
                  }

field_list = ['open', 'high', 'low', 'close', 'volume', 'openInterest', 'contract', 'shift']

port_pos_config = {
    'PTSIM1_hot': {
        'pos_loc': 'C:/dev/pyktrader3/process/pt_test1',
        'strat_list': [
            ('PTSIM1_FACTPORT.json', 25000),
            ('PTSIM1_EXCHWNT.json', 14000),
            ('PTSIM1_SEAZN.json', 12000),
            ('PTSIM1_HRCRB.json', 14000),
            ('PTSIM1_LL.json', 14000),
            ('PTSIM1_LL2MR.json', 14000),
            ('PTSIM1_SPDTF.json', 14000),
            ('PTSIM1_MR1Y.json', 14000),
            ('PTSIM1_CNMAC1.json', 10000),
            ('PTSIM1_CNMAC2.json', 3000),
            ('PTSIM1_FUNFER.json', 25000),
            ('PTSIM1_FERSPD.json', 70000),
            ('PTSIM1_AUSPD.json', 30000),
            ('PTSIM1_FUNBASE.json', 25000),
            ('PTSIM1_FUNENE.json', 3000),
            ('PTSIM1_FUNMTL.json', 12000),
            ('PTSIM1_BND1.json', 50000),
        ], },
}

pos_chg_notification = ['PTSIM1_hot']


def update_factor_db(xdf, field, config, dbtable='fut_fact_data', flavor='mysql', start_date=None, end_date=None):
    logging.info('updating factor_name=%s' % field)
    df = xdf.copy()
    for key in config:
        df[key] = config[key]
    df['fact_name'] = field
    df['fact_val'] = df[field]
    df = df.dropna().reset_index()
    if start_date:
        df = df[pd.to_datetime(df['date']) >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['date']) <= pd.to_datetime(end_date)]
    df['date'] = pd.to_datetime(df['date'])
    df = df[['product_code', 'roll_label', 'exch', 'fact_name', 'freq', 'date', 'serial_no', 'serial_key', 'fact_val']]
    #insert_df_to_sql(df, dbtable, is_replace=True)
    if flavor == 'mysql':
        conn = create_engine(
            'mysql+mysqlconnector://{user}:{passwd}@{host}/{dbase}'.format(user=dbconfig['user'],
                                                                           passwd=dbconfig['password'],
                                                                           host=dbconfig['host'],
                                                                           dbase=dbconfig['database']),
            echo=False)
        func = mysql_replace_into
    else:
        conn = connect(**dbconfig)
        func = None
    df.to_sql(dbtable, con=conn, if_exists='append', index=False, method=func)
    if flavor == 'mysql':
        conn.dispose()


def update_factor_data(product_list, scenarios, start_date, end_date,
                       roll_rule='CAL_30b',
                       freq='d',
                       flavor='mysql',
                       shift_mode=1):
    col_list = ['open', 'high', 'low', 'close', 'volume', 'openInterest', 'contract', 'shift']
    update_start = day_shift(end_date, '-20b', CHN_Holidays)
    logging.info('updating factor data for %s ...' % roll_rule)
    fact_config = {}
    fact_config['roll_label'] = roll_rule
    if roll_rule == 'CAL_30b':
        fact_config['freq'] = 's1'
    else:
        fact_config['freq'] = freq
    fact_config['serial_key'] = '0'
    fact_config['serial_no'] = 0

    factor_repo = {}
    data_cache = {}
    vol_win = 20
    for idx, asset in enumerate(product_list):
        sdate = max(sim_start_dict.get(asset, start_date), start_date)
        logging.info("loading mkt = %s, end_date=%s, shift_mode=%s" % (asset, end_date, shift_mode))
        if roll_rule == 'CAL_30b':
            roll = '-30b'
            if asset in eq_fut_mkts:
                roll = '-1b'
            elif asset in ['cu', 'al', 'zn', 'pb', 'sn', 'ss', 'lu']:
                roll = '-25b'
            elif asset in ['ni', 'jd', 'lh', 'eg', ]:
                roll = '-35b'
            elif asset in ['sc', 'eb'] + bond_fut_mkts:
                roll = '-20b'
            elif asset in precious_metal_mkts:
                roll = '-15b'
            df1 = nearby(asset, 1, start_date=sdate, end_date=end_date, shift_mode=shift_mode, freq='d', roll_rule=roll)
            df2 = nearby(asset, 2, start_date=sdate, end_date=end_date, shift_mode=shift_mode, freq='d', roll_rule=roll)
        else:
            df1 = dataseries.nearby(asset, 1, start_date=sdate, end_date=end_date, shift_mode=shift_mode, freq='d',
                                    roll_name=roll_rule, config_loc="C:/dev/wtdev/config/").set_index('date')
            df2 = dataseries.nearby(asset, 2, start_date=sdate, end_date=end_date, shift_mode=shift_mode, freq='d',
                                    roll_name=roll_rule, config_loc="C:/dev/wtdev/config/").set_index('date')
        df1 = df1[col_list]
        df1['contmth'] = df1.apply(lambda x: inst2contmth(x['contract'], x.name), axis=1)
        df1['mth'] = df1['contmth'].apply(lambda x: x // 100 * 12 + x % 100)
        df1['atr'] = dh.ATR(df1, vol_win).fillna(method='bfill')

        df2 = df2[col_list]
        df2['contmth'] = df2.apply(lambda x: inst2contmth(x['contract'], x.name), axis=1)
        df2['mth'] = df2['contmth'].apply(lambda x: x // 100 * 12 + x % 100)

        df2.columns = [col + '_2' for col in df2.columns]
        xdf = pd.concat([df1, df2], axis=1, sort=False).sort_index()
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        if shift_mode == 1:
            xdf['ryield'] = (np.log(xdf['close'] - xdf['shift']) - np.log(xdf['close_2'] - xdf['shift_2'])) / (
                        xdf['mth_2'] - xdf['mth']) * 12.0
            xdf['logret'] = np.log(xdf['close'] - xdf['shift']) - np.log(xdf['close'].shift(1) - xdf['shift'])
            xdf['pct_chg'] = (xdf['close'] - xdf['shift'])/(xdf['close'].shift(1) - xdf['shift']) - 1
            xdf['logret_2'] = np.log(xdf['close_2'] - xdf['shift_2']) - np.log(xdf['close_2'].shift(1) - xdf['shift_2'])
        elif shift_mode == 2:
            xdf['ryield'] = (np.log(xdf['close']) - np.log(xdf['close_2']) - xdf['shift'] + xdf['shift_2']) / (
                        xdf['mth_2'] - xdf['mth']) * 12.0
            xdf['logret'] = np.log(xdf['close']) - np.log(xdf['close'].shift(1))
            xdf['pct_chg'] = xdf['close'].pct_change()
            xdf['logret_2'] = np.log(xdf['close_2']) - np.log(xdf['close_2'].shift(1))
        else:
            xdf['ryield'] = (np.log(xdf['close']) - np.log(xdf['close_2'])) / (xdf['mth_2'] - xdf['mth']) * 12.0
            xdf['logret'] = np.log(xdf['close']) - np.log(xdf['close'].shift(1))
            xdf['pct_chg'] = xdf['close'].pct_change()
            xdf['logret_2'] = np.log(xdf['close_2']) - np.log(xdf['close_2'].shift(1))
        xdf['px_chg'] = xdf['close'].diff()
        xdf['baslr'] = xdf['logret'] - xdf['logret_2']
        xdf['pct_vol'] = xdf['close'] * xdf['pct_chg'].rolling(vol_win).std()

        xdf.index.name = 'date'
        for field in ['logret', 'baslr', 'ryield', 'atr', 'pct_vol']:
            update_factor_db(xdf, field, fact_config, start_date=update_start, end_date=end_date, flavor=flavor)

        data_cache[asset] = xdf.copy(deep=True)
        updated_factors = ['logret', 'baslr', 'ryield', 'atr', 'pct_vol']

        for scen in scenarios:
            sim_name = scen[0]            
            if 'ts' in sim_name:
                sim_type = 'ts'
            elif 'xs' in sim_name:
                type_split = sim_name.split('-')
                if len(type_split) > 1:
                    sim_type = 'xs-' + type_split[1]
                else:
                    sim_type = 'xs'
            run_mode = data_field = scen[1]
            weight = scen[2]
            win = scen[3]
            ma_win = scen[4]
            rebal = scen[5]
            pos_func, pos_args, pos_name = scen[6]
            params = scen[7]
            extra_fields = []
            fact_name = None
            quantile = 0
            if len(scen) == 9:
                quantile = scen[8]            
            if 'basmom' in run_mode:
                xdf['basmom'] = xdf['baslr'].rolling(win).sum()                
            elif 'mom' in run_mode:
                xdf['mom'] = xdf['logret'].rolling(win).sum()
            xdf['upratio'] = xdf['logret'].rolling(win).agg(lambda x: (x>0).sum()/win) - 0.5
            if run_mode[:3] == 'rsi':
                rsi_output = dh.RSI_F(xdf, win)
                xdf['rsi'] = rsi_output['RSI' + str(win)]
            elif 'clbrk' in run_mode:
                chmax = xdf['close'].rolling(win).max()
                chmin = xdf['close'].rolling(win).min()
                chavg = (chmax + chmin)/2.0
                xdf['clbrk'] = (xdf['close'] - chavg)/(chmax - chmin)
            elif 'hlbrk' in run_mode:
                chmax = xdf['high'].rolling(win).max()
                chmin = xdf['low'].rolling(win).min()
                chavg = (chmax + chmin)/2.0
                xdf['hlbrk'] = (xdf['close'] - chavg)/(chmax - chmin)
            elif run_mode[:4] == 'macd':
                xdf['ema1'] = dh.EMA(xdf, win, field='close')
                xdf['ema2'] = dh.EMA(xdf, int(win * params[0]), field='close')
                xdf['mstd'] = dh.STDEV(xdf, int(win * params[1]), field='close')
                xdf['macd'] = (xdf['ema1'] - xdf['ema2']) / xdf['mstd']
                extra_fields += [str(int(win * params[0])), str(int(win * params[1]))]
            elif run_mode == 'mixmom':                
                xdf['mixmom'] = ((xdf['mom'] * xdf['upratio'])).apply(lambda x: x if x>0 else 0) * 1.0 \
                                                              * xdf['mom'].apply(lambda x: sign(x))
            if 'sma' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field].rolling(ma_win).mean()
                extra_fields += ['sma', str(ma_win)]
            elif 'ema' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = dh.EMA(xdf, ma_win, field = ref_field)
                extra_fields += ['ema', str(ma_win)]
            elif 'xma' == run_mode[-3:]:   
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field] - xdf[ref_field].rolling(ma_win).mean()
                extra_fields += ['xma', str(ma_win)]
            elif 'xea' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field] - dh.EMA(xdf, ma_win, field=ref_field)
                extra_fields += ['xea', str(ma_win)]
            elif 'nma' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field] / dh.STDEV(xdf, ma_win, field=ref_field)
                extra_fields += ['nma', str(ma_win)]
            elif 'nmb' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field] / ((xdf[ref_field]**2).rolling(ma_win).mean()**0.5)
                extra_fields += ['nmb', str(ma_win)]
            elif 'elv' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = (xdf[ref_field] - xdf[ref_field].ewm(span=ma_win, min_periods=ma_win // 2,
                                                                       ignore_na=True).mean()) / \
                                  (xdf[ref_field].ewm(span=ma_win, min_periods=ma_win // 2, ignore_na=True).std())
                extra_fields += ['elv', str(ma_win)]
            elif 'zlv' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = (xdf[ref_field] - xdf[ref_field].rolling(ma_win).mean()) \
                                        / dh.STDEV(xdf, ma_win, field = ref_field)
                extra_fields += ['zlv', str(ma_win)]
            elif 'qtl' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = 2.0 * (rolling_percentile(xdf[ref_field], win=ma_win) - 0.5)
                extra_fields += ['qtl', str(ma_win)]
            else:
                ref_field = run_mode
            if pos_func:
                xdf[data_field] = xdf[data_field].apply(lambda x: pos_func(x, **pos_args))
                if len(pos_name) > 0:
                    extra_fields.append(pos_name) 
            fact_name = '_'.join([ref_field, str(win)] + extra_fields)
            xdf[fact_name] = xdf[data_field]
            strat_fact_name = f'{fact_name}.{sim_type}'
            
            if strat_fact_name not in factor_repo:
                factor_repo[strat_fact_name] = {}
                factor_repo[strat_fact_name]['name'] = fact_name
                factor_repo[strat_fact_name]['type'] = sim_type
                if 'ts' in sim_name:
                    factor_repo[strat_fact_name]['threshold'] = 0.0               
                elif 'xs' in sim_name:
                    factor_repo[strat_fact_name]['threshold'] = quantile
                else:
                    print("unsupported run mode")
                factor_repo[strat_fact_name]['rebal'] = rebal
                factor_repo[strat_fact_name]['param'] = params
                factor_repo[strat_fact_name]['weight'] = weight 

            if fact_name not in updated_factors:
                update_factor_db(xdf, fact_name, fact_config, start_date=update_start, end_date=end_date, flavor=flavor)
                updated_factors.append(fact_name)

    logging.info("updating factor for hc-rb spread...")
    if ('rb' in product_list) and ('hc' in product_list):
        rb_df = data_cache['rb']
        hc_df = data_cache['hc']
        hc_df['rb_px_chg'] = rb_df['px_chg']
        hc_df['hc_rb_diff'] = hc_df['px_chg'] - hc_df['rb_px_chg']
        fact_config['product_code'] = 'hc_rb_diff'
        fact_config['exch'] = 'SHFE'
        for win in [20, 30, 40, 60]:
            fact_name = f'hc_rb_diff_{win}'
            hc_df[fact_name] = hc_df['hc_rb_diff'].ewm(span=win).mean()/hc_df['hc_rb_diff'].ewm(span=win).std()
            # hc_df[fact_name] = hc_df[fact_name].apply(lambda x: max(min(x, hc_df[fact_name].quantile(0.975)),
            #                                                         hc_df[fact_name].quantile(0.025)))
            update_factor_db(hc_df, fact_name, fact_config, start_date=update_start, end_date=end_date, flavor=flavor)

    #beta neutral
    logging.info("updating factor for beta neutral ratio ...")
    beta_win = 245
    asset_pairs = [('rb', 'i'), ('hc', 'i'), ('j', 'i')]
    fact_config['exch'] = 'xasset'
    for trade_asset, index_asset in asset_pairs:
        if (trade_asset not in product_list) or (index_asset not in product_list):
            continue
        key = '_'.join([trade_asset, index_asset, 'beta'])
        fact_config['product_code'] = key
        asset_df = pd.DataFrame(index=data_cache[index_asset].index)
        asset_df[trade_asset] = data_cache[trade_asset]['pct_chg'].fillna(0)
        asset_df[index_asset] = data_cache[index_asset]['pct_chg'].fillna(0)
        asset_df = asset_df.dropna(subset=[trade_asset]).ffill()
        for asset in asset_df:
            asset_df[f'{asset}_pct'] = asset_df[asset].rolling(5).mean()
            asset_df[f'{asset}_vol'] = asset_df[asset].rolling(vol_win).std()
        asset_df['beta'] = asset_df[f'{index_asset}_pct'].rolling(beta_win).cov(
            asset_df[f'{trade_asset}_pct']) / asset_df[f'{index_asset}_pct'].rolling(beta_win).var()
        asset_df['pct_chg'] = asset_df[trade_asset] - asset_df['beta'] * asset_df[index_asset].fillna(0)
        asset_df['pct_vol'] = asset_df['pct_chg'].rolling(vol_win).std()
        asset_df['trade_leg'] = asset_df[f'{trade_asset}_vol']/asset_df['pct_vol']
        asset_df['index_leg'] = - asset_df[f'{index_asset}_vol'] / asset_df['pct_vol'] * asset_df['beta']
        for field in ['pct_vol', 'beta', 'trade_leg', 'index_leg']:
            update_factor_db(asset_df, field, fact_config, start_date=update_start, end_date=end_date, flavor=flavor)

    #leader-lagger
    fact_name = 'leadlag_d_mid'
    leadlag_port_d = {
        'ferrous': {'lead': ['hc', 'rb', ],
                    'lag': [],
                    'param_rng': [40, 80, 2],
                    },
        'constrs': {'lead': ['hc', 'rb', 'v'],
                    'lag': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'v', 'SM', 'SF'],
                    'param_rng': [40, 80, 2],
                    },
        'petchem': {'lead': ['v'],
                    'lag': ['TA', 'MA', 'pp', 'eg', 'eb', 'PF'],
                    'param_rng': [40, 80, 2],
                    },
        'base': {'lead': ['al'],
                'lag': ['al', 'ni', 'sn', 'ss'],  # 'zn', 'cu'
                'param_rng': [40, 80, 2],
                },
        'oil': {'lead': ['sc'],
                'lag': ['sc', 'pg', 'bu', ],
                'param_rng': [20, 30, 2],
                },
        'bean': {'lead': ['b'],
                'lag': ['p', 'y', 'OI', ],
                'param_rng': [60, 80, 2],
                },
    }    
    logging.info(f"updating factor for {fact_name}...")
    leadlag_products = ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SM', 'SF', 'UR', 'cu', 'al', 'zn', 'sn', 'ss', 'ni',
                        'l', 'pp', 'v', 'TA', 'sc', 'eb', 'eg', 'y', 'p', 'OI']
    for sector in leadlag_port_d:
        for asset in leadlag_port_d[sector]['lead']:
            if asset not in data_cache:
                xdf = dataseries.nearby(asset, 1, start_date=sdate, end_date=end_date, shift_mode=shift_mode,
                                        freq='d', roll_name='hot',
                                        config_loc="C:/dev/wtdev/config/").set_index('date')
                data_cache[asset] = xdf
    for asset in leadlag_products:
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        for sector in leadlag_port_d:
            if asset in leadlag_port_d[sector]['lag']:
                signal_list = []
                for lead_prod in leadlag_port_d[sector]['lead']:
                    feature_ts = data_cache[lead_prod]['close']
                    signal_ts = calc_conv_signal(feature_ts.dropna(), 'qtl',
                                                 leadlag_port_d[sector]['param_rng'], signal_cap=None)
                    signal_list.append(signal_ts)
                signal_ts = pd.concat(signal_list, axis=1).mean(axis=1)
                asset_df = data_cache[asset].copy()
                asset_df[fact_name] = signal_ts
                update_factor_db(asset_df, fact_name, fact_config, start_date=update_start, end_date=end_date,
                                 flavor=flavor)

    idx = pd.bdate_range(start=start_date, end=end_date, freq='C', holidays=CHN_Holidays)
    fact_name = 'leadlag2_mr_d'
    logging.info(f"updating factor for {fact_name}...")
    win = 10
    signal_cap = [-3, 3]
    vol_win = 60
    param_rng = [1, 2, 1]
    signal_func = 'ma'
    mr_dict = {
        'hc': 'rb',
        'i': 'rb',
        'j': 'rb',
        'pp': 'l',
        'SA': 'FG',
        'sc': 'lu',
        # 'lu': 'fu',
        'm': 'y',
    }
    leadlag_dict = {
        'lu': 'sc',
        'RM': 'm',
        'a': 'm',
        'cs': 'c',
        'eb': 'sc',
        # 'fu': 'sc',
        'bu': 'sc',
        'bc': 'cu',
        'MA': 'pp',
        'ni': 'ss',
        'al': 'ao',
    }
    ll_assets = [
        'rb', 'hc', 'i', 'j', 'al', 'FG', 'SA',
        'l', 'pp', 'lu', 'sc', 'm', 'RM', 'y',
        'c', 'cs', 'MA', 'a', 'eb', 'bu',
        'cu', 'ss', 'ni', 'bc', 'ao', #'fu',
    ]
    for asset in ll_assets:
        if asset not in data_cache:
            xdf = dataseries.nearby(asset, 1, start_date=sdate, end_date=end_date, shift_mode=shift_mode,
                                    freq='d', roll_name='hot',
                                    config_loc="C:/dev/wtdev/config/").set_index('date')
            data_cache[asset] = xdf
    signal_df = pd.DataFrame(index=idx, columns=ll_assets)
    for asset in mr_dict:
        tmp_df = pd.concat([
            data_cache[asset]['close'].to_frame(asset),
            data_cache[mr_dict[asset]]['close'].to_frame(mr_dict[asset])], axis=1).dropna(how='all').ffill()
        tmp_df = tmp_df.pct_change()
        tmp_df = tmp_df.rolling(win).sum()
        feature_ts = tmp_df[asset] - tmp_df[mr_dict[asset]]
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng, signal_cap=signal_cap,
                                     vol_win=vol_win)
        signal_ts = signal_ts.ewm(3).mean()
        signal_df[asset] = signal_ts
    signal_df2 = pd.DataFrame(index=idx, columns=ll_assets)
    for asset in leadlag_dict:
        tmp_df = pd.concat([
            data_cache[asset]['close'].to_frame(asset),
            data_cache[leadlag_dict[asset]]['close'].to_frame(leadlag_dict[asset])], axis=1).dropna(how='all')
        tmp_df = tmp_df.pct_change()
        tmp_df = tmp_df.rolling(win).sum()
        feature_ts = tmp_df[leadlag_dict[asset]] - tmp_df[asset]
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng, signal_cap=signal_cap,
                                     vol_win=vol_win)
        signal_ts = signal_ts.ewm(3).mean()
        signal_df2[asset] = signal_ts
    signal_df = signal_df.fillna(0) + signal_df2.fillna(0)
    for asset in signal_df.columns:
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        asset_df = data_cache[asset].copy()
        asset_df[fact_name] = signal_df[asset]
        update_factor_db(asset_df, fact_name, fact_config, start_date=update_start, end_date=end_date,
                         flavor=flavor)

    fact_name = 'pair_mr_1y'
    logging.info(f"updating factor for {fact_name}...")
    pair_mr_products = ['cu', 'al', 'zn', 'ss', 'ni', 'rb', 'hc', 'SM', 'SF', 'FG', 'v',
                        'y', 'OI', 'm', 'RM', 'l', 'MA', 'pp', 'TA', 'eg']
    pair_list = [
        ('cu', 'zn'), ('cu', 'al'), ('al', 'zn'), ('ni', 'ss'),
        ('rb', 'hc'), ('SM', 'SF'), ('FG', 'v'),
        ('y', 'OI'), ('m', 'RM'),  ('l', 'MA'), ('pp', 'MA'), ('TA', 'eg'),
    ]
    param_rng = [200, 250, 5]
    bullish = False
    vol_win = 120
    signal_cap = None
    signal_func = 'zscore_adj'
    signal_df = pd.DataFrame(0, index=idx, columns=pair_mr_products)
    for asset in pair_mr_products:
        if asset not in data_cache:
            xdf = dataseries.nearby(asset, 1, start_date=sdate, end_date=end_date, shift_mode=shift_mode,
                                    freq='d', roll_name='hot',
                                    config_loc="C:/dev/wtdev/config/").set_index('date')
            data_cache[asset] = xdf

    for (asset_a, asset_b) in pair_list:
        pair_assets = [asset_a, asset_b]
        sig_df = pd.DataFrame(index=idx, columns=pair_assets)
        feature_ts = np.log(data_cache[asset_a]['close']) - np.log(data_cache[asset_b]['close'])
        sig_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng, signal_cap=signal_cap,
                                  vol_win=vol_win)
        sig_ts = sig_ts.apply(lambda x: np.sign(x) * min(abs(x), 1.25) ** 4).ewm(1).mean()
        if not bullish:
            sig_ts = -sig_ts
        sig_df[asset_a] = sig_ts
        sig_df[asset_b] = -sig_ts
        signal_df = signal_df + sig_df.reindex_like(signal_df).fillna(0)
    for asset in signal_df.columns:
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        asset_df = data_cache[asset].copy()
        asset_df[fact_name] = signal_df[asset]
        update_factor_db(asset_df, fact_name, fact_config, start_date=update_start, end_date=end_date,
                         flavor=flavor)
    return factor_repo


def create_strat_json(product_list, freq, roll_rule, factor_repo,
                      filename="C:\\dev\\data\\MM_FACT_PORT.json",
                      name='default'):
    strat_data = {}
    strat_data["class"] = "pycmqlib3.strategy.strat_factor_port.FactorPortTrader"
    strat_config = {}
    strat_config['name'] = name
    if freq == 'd':
        strat_config['freq'] = 's1'
    else:
        strat_config['freq'] = freq
    strat_config['roll_label'] = 'CAL_' + roll_rule
    strat_config['factor_repo'] = factor_repo
    strat_config['vol_win'] = 20
    strat_config['fact_db_table'] = 'fut_fact_data'
    strat_config['exec_bar_list'] = [1510]
    strat_config['pos_scaler'] = 1000

    assets = []
    for asset in product_list:
        asset_data = {}
        asset_data['underliers'] = [trade_cont_map[asset][0]]
        asset_data['volumes'] = [1]
        asset_data['alloc_w'] = 1.0
        asset_data['prev_underliers'] = ''
        assets.append(asset_data)
    strat_config['assets'] = assets

    filtered_factors = {}
    for fact_name in factor_repo:
        if factor_repo[fact_name]['type'] in ['xs', 'ts']:
            filtered_factors[fact_name] = copy.copy(factor_repo[fact_name])
    strat_config['factor_repo'] = filtered_factors
    strat_data['config'] = strat_config
    with open(filename, 'w') as f:
        json.dump(strat_data, f, indent=4)


def update_port_position(run_date=datetime.date.today()):
    results = {
        'pos_update': {},
        'details': {}
    }
    pos_date = day_shift(run_date, '1b', CHN_Holidays)
    pre_date = day_shift(pos_date, '-1b', CHN_Holidays)
    pos_date_str = pos_date.strftime('%Y%m%d')
    pre_date_str = pre_date.strftime('%Y%m%d')
    for port_name in port_pos_config.keys():
        target_pos = {}
        pos_by_strat = {}
        pos_loc = port_pos_config[port_name]['pos_loc']
        curr_signal_file = f'{pos_loc}/curr_signal_{pre_date_str}.json'
        curr_signal = {}
        next_signal = {}
        try:
            with open(curr_signal_file, 'r') as fp:
                curr_signal = json.load(fp)
        except:
            pass

        port_file = port_name
        for strat_file, pos_scaler in port_pos_config[port_name]['strat_list']:
            config_file = f'{pos_loc}/settings/{strat_file}'
            with open(config_file, 'r') as fp:
                strat_conf = json.load(fp)
            strat_args = strat_conf['config']
            assets = strat_args['assets']
            vol_key = strat_conf.get("vol_key", "pct_vol")
            roll = strat_conf.get("roll_label", "hot")
            repo_type = strat_args.get('repo_type', 'asset')            
            freq = strat_conf.get("freq", "d1")
            hist_fact_lookback = strat_conf.get("hist_fact_lookback", 20)
            factor_repo = strat_args['factor_repo']
            product_list = []
            for asset_dict in assets:
                under = asset_dict["underliers"][0]
                product = inst2product(under)
                product_list.append(product)

            logging.info(f"updating position for {strat_file}...")
            res = generate_strat_position(run_date, product_list, factor_repo,
                                          repo_type=repo_type,
                                          roll_label=roll,
                                          pos_scaler=pos_scaler,
                                          freq=freq,
                                          hist_fact_lookback=hist_fact_lookback,
                                          vol_key=vol_key,
                                          curr_signal=curr_signal,
                                          signal_config=signal_buffer_config)
            strat_target = res['target_pos']
            next_signal = {**(res['curr_signal'].to_dict()), **next_signal}
            results['details'][f'{port_name}:{strat_file}'] = res['pos_sum'].T
            pos_by_strat[strat_file] = strat_target
            for prod in strat_target:
                if prod not in target_pos:
                    target_pos[prod] = 0
                if ~np.isnan(strat_target[prod]):
                    target_pos[prod] += strat_target[prod]

        for prodcode in target_pos:
            if np.isnan(target_pos[prodcode]):
                target_pos[prodcode] = 0
                continue
            if prodcode in ['UR', 'SM']:
                target_pos[prodcode] = int((target_pos[prodcode] / 4 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 4
            elif prodcode in ['RM']:
                target_pos[prodcode] = int((target_pos[prodcode] / 10 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 10
            else:
                target_pos[prodcode] = int(target_pos[prodcode] + (0.5 if target_pos[prodcode] > 0 else -0.5))

        posfile = '%s/%s_%s.json' % (pos_loc, port_file, pos_date_str)
        with open(posfile, 'w') as ofile:
            json.dump(target_pos, ofile, indent=4)

        next_signal_file = f'{pos_loc}/curr_signal_{pos_date_str}.json'
        with open(next_signal_file, 'w') as ofile:
            json.dump(next_signal, ofile, indent=4)

        stratfile = '%s/pos_by_strat_%s_%s.json' % (pos_loc, port_file, pos_date_str)
        with open(stratfile, 'w') as ofile:
            json.dump(pos_by_strat, ofile, indent=4)

        if port_file in pos_chg_notification:
            with open('%s/%s_%s.json' % (pos_loc, port_file, pre_date_str), 'r') as fp:
                curr_pos = json.load(fp)
            pos_df = pd.DataFrame({'cur': curr_pos, 'tgt': target_pos})
            pos_df['diff'] = pos_df['tgt'] - pos_df['cur']
            results['pos_update'][port_file] = pos_df
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        now = datetime.datetime.now()
        tday = now.date()
        if (not is_workday(tday, 'CHN')) or (now.time() < datetime.time(14, 59, 0)):
            tday = day_shift(tday, '-1b', CHN_Holidays)
    folder = "C:/dev/data/"
    name = "pf_position_update"            
    base.config_logging(folder + name + ".log", level=logging.INFO,
                        format='%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                        to_console=True,
                        console_level=logging.INFO)      
    logging.info("running portfolio position for %s" % str(tday))        
    res = update_port_position(run_date=tday)
