import pandas as pd
import numpy as np
import json
import datetime
import copy
from sqlalchemy import create_engine
from pycmqlib3.utility.dbaccess import dbconfig, mysql_replace_into, connect
from pycmqlib3.utility.misc import nearby, cleanup_mindata, prod2exch, inst2contmth, \
    CHN_Holidays, contract_expiry, day_shift, sign
import pycmqlib3.analytics.data_handler as dh

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'ZC', 'SM', "SF"]
base_metal_mkts = ['cu', 'al', 'zn', 'pb', 'ni', 'sn']
precious_metal_mkts = ['au', 'ag']
ind_metal_mkts = ferrous_products_mkts + ferrous_mixed_mkts + base_metal_mkts
petro_chem_mkts = ['l', 'pp', 'v', 'TA', 'MA', 'bu']  # , 'sc', 'fu', 'eg']
ind_all_mkts = ind_metal_mkts + petro_chem_mkts
ags_oil_mkts = ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs']  # , 'b']
ags_soft_mkts = ['CF', 'CY', 'SR', 'jd', 'AP']  # , 'sp', 'CJ', 'UR']
ags_all_mkts = ags_oil_mkts + ags_soft_mkts
eq_fut_mkts = ['IF', 'IH', 'IC']
bond_fut_mkts = ['T', 'TF']
fin_all_mkts = eq_fut_mkts + bond_fut_mkts
commod_all_mkts = ind_all_mkts + ags_all_mkts + precious_metal_mkts
all_markets = commod_all_mkts + fin_all_mkts

trade_cont_map = {'rb': ['rb2110', 'rb2201'], 'hc': ['hc2110', 'hc2201'], 'i': ['i2109', 'i2201'],
                  'j': ['j2109', 'j2201'], 'jm': ['jm2109', 'jm2201'], 'ru': ['ru2109', 'ru2201'],
                  'FG': ['FG109', 'FG201'], 'ZC': ['ZC109', 'ZC201'], 'cu': ['cu2107', 'cu2108'],
                  'al': ['al2107', 'al2108'], 'zn': ['zn2107', 'zn2108'], 'ni': ['ni2107', 'ni2108'],
                  'sn': ['sn2107', 'sn2108'], 'pb': ['pb2107', 'pb2108'], 'l': ['l2109', 'l2201'], 
                  'pp': ['pp2109', 'pp2201'], 'v': ['v2109', 'v2201'], 'TA': ['TA109', 'TA201'],
                  'sc': ['sc2106', 'sc2107'], 'm': ['m2109', 'm2201'], 'RM': ['RM109', 'RM201'],
                  'y': ['y2109', 'y2201'], 'p': ['p2109', 'p2201'], 'OI': ['OI109', 'OI201'],
                  'a': ['a2109', 'a2201'], 'c': ['c2109', 'c2201'], 'cs': ['cs2109', 'cs2201'],
                  'CF': ['CF109', 'CF201'], 'jd': ['jd2109', 'jd2201'], 'AP': ['AP110', 'AP201'],
                  'ss': ['ss2107', 'ss2108'], 'SM': ['SM109', 'SM201'], 'SF': ['SF109', 'SF201'],
                  'SR': ['SR109', 'SR201'], 'CY': ['CY109', 'CY201'], 'eg': ['eg2109', 'eg2201'],
                  'ag': ['ag2112', 'ag2206'], 'au': ['au2112', 'au2206'], 
                  }

sim_start_dict = {'c': datetime.date(2008, 10, 1), 'm': datetime.date(2010, 10, 1),
                  'y': datetime.date(2010, 1, 1), 'l': datetime.date(2008, 1, 1), 'rb': datetime.date(2010, 1, 1),
                  'p': datetime.date(2010, 1, 1), 'cu': datetime.date(2010, 1, 1), 'al': datetime.date(2010, 1, 1),
                  'zn': datetime.date(2010, 1, 1), 'au': datetime.date(2010, 1, 1), 'v': datetime.date(2010, 1, 1),
                  'a': datetime.date(2010, 1, 1), 'ru': datetime.date(2010, 1, 1), 'ag': datetime.date(2012, 7, 6),
                  'i': datetime.date(2013, 12, 13), 'j': datetime.date(2012, 6, 1), 'jm': datetime.date(2013, 5, 24),
                  'CF': datetime.date(2008, 1, 1), 'TA': datetime.date(2007, 2, 15),
                  'PM': datetime.date(2012, 10, 1), 'RM': datetime.date(2013, 3, 16), 'SR': datetime.date(2006, 1, 6),
                  'FG': datetime.date(2013, 2, 1), 'OI': datetime.date(2013, 6, 1), 'RI': datetime.date(2013, 6, 1),
                  'WH': datetime.date(2014, 2, 1), 'pp': datetime.date(2014, 4, 28),
                  'IF': datetime.date(2010, 5, 1), 'MA': datetime.date(2011, 12, 23), 'TF': datetime.date(2014, 4, 1),
                  'IH': datetime.date(2015, 5, 1), 'IC': datetime.date(2015, 5, 1), 'cs': datetime.date(2015, 2, 17),
                  'jd': datetime.date(2014, 1, 6), 'ni': datetime.date(2015, 5, 26), 'sn': datetime.date(2017, 4, 1),
                  'ZC': datetime.date(2014, 5, 1), 'hc': datetime.date(2016, 4, 1), 'SM': datetime.date(2017, 1, 1),
                  'SF': datetime.date(2017, 5, 17), 'CY': datetime.date(2017, 7, 17), 'AP': datetime.date(2018, 2, 26),
                  'TS': datetime.date(2018, 8, 17), 'fu': datetime.date(2018, 7, 16), 'sc': datetime.date(2018, 5, 25),
                  'b': datetime.date(2017, 12, 26), 'pb': datetime.date(2016, 7, 1), 'bu': datetime.date(2015, 11, 1),
                  'T': datetime.date(2015, 6, 1), 'ss': datetime.date(2020, 4, 1), 'sp': datetime.date(2019, 1, 24),
                  'CJ': datetime.date(2019, 8, 1), 'UR': datetime.date(2019, 8, 9), 'SA': datetime.date(2019, 12, 9),
                  'eb': datetime.date(2019, 12, 1), 'eg': datetime.date(2019, 2, 13), 'rr': datetime.date(2019, 8, 19),
                  'pg': datetime.date(2020, 10, 1), 'lu': datetime.date(2020, 8, 1), }

field_list = ['open', 'high', 'low', 'close', 'volume', 'openInterest', 'contract', 'shift']

def update_factor_db(xdf, field, config, dbtable='fut_fact_data', flavor='mysql', start_date=None, end_date=None):
    df = xdf.copy()
    for key in config:
        df[key] = config[key]
    df['fact_name'] = field
    df['fact_val'] = df[field]
    df = df.dropna().reset_index()
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    df = df[['product_code', 'roll_label', 'exch', 'fact_name', 'freq', 'date', 'serial_no', 'serial_key', 'fact_val']]
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

def update_factor_data(product_list, scenarios, start_date, end_date, roll_rule = '30b', flavor = 'mysql', dbtbl_prefix = ''):
    update_start = start_date  # day_shift(end_date, '-3b')
    need_shift = 1
    freq = 'd'
    args = {'roll_rule': '-' + roll_rule, 'freq': freq, 'need_shift': need_shift, 'dbtbl_prefix': dbtbl_prefix}
    base_args = {'roll_rule': '-' + roll_rule, 'freq': freq, 'need_shift': need_shift, 'dbtbl_prefix': dbtbl_prefix}
    eq_args = {'roll_rule': '-1b', 'freq': freq, 'need_shift': need_shift, 'dbtbl_prefix': dbtbl_prefix}
    bond_args = {'roll_rule': '-' + roll_rule, 'freq': freq, 'need_shift': need_shift, 'dbtbl_prefix': dbtbl_prefix}
    precious_args = {'roll_rule': '-25b', 'freq': freq, 'need_shift': need_shift, 'dbtbl_prefix': dbtbl_prefix}

    fact_config = {}
    fact_config['roll_label'] = 'CAL_%s' % ('30b')
    if freq == 'd':
        fact_config['freq'] = 's1'
    else:
        fact_config['freq'] = freq
    fact_config['serial_key'] = '0'
    fact_config['serial_no'] = 0

    factor_repo = {}
    for idx, asset in enumerate(product_list):
        use_args = copy.copy(args)
        if asset in eq_fut_mkts:
            use_args = eq_args
        elif asset in ['cu', 'al', 'zn', 'pb', 'sn']:
            use_args = base_args
        elif asset in bond_fut_mkts:
            use_args = bond_args
        elif asset in precious_metal_mkts:
            use_args = precious_args
        use_args['start_date'] = max(sim_start_dict.get(asset, start_date), start_date)
        use_args['end_date'] = end_date

        use_args['n'] = 1
        print("loading mkt = %s, nb = %s, args = %s" % (asset, str(use_args['n']), use_args))
        df = nearby(asset, **use_args)
        if freq == 'm':
            df = cleanup_mindata(df, asset)
        # df['expiry'] = df['contract'].apply(lambda x: contract_expiry(x, CHN_Holidays))
        df['contmth'] = df['contract'].apply(lambda x: inst2contmth(x))
        df['mth'] = df['contmth'].apply(lambda x: x // 100 * 12 + x % 100)

        use_args['n'] = 2
        print("loading mkt = %s, nb = %s, args = %s" % (asset, str(use_args['n']), use_args))
        xdf = nearby(asset, **use_args)
        if freq == 'm':
            xdf = cleanup_mindata(xdf, asset)
        # xdf['expiry'] = xdf['contract'].apply(lambda x: contract_expiry(x, CHN_Holidays))
        xdf['contmth'] = xdf['contract'].apply(lambda x: inst2contmth(x))
        xdf['mth'] = xdf['contmth'].apply(lambda x: x // 100 * 12 + x % 100)

        xdf.columns = [col + '_2' for col in xdf.columns]
        xdf = pd.concat([df, xdf], axis=1, sort=False).sort_index()
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        if need_shift == 1:
            xdf['ryield'] = (np.log(xdf['close'] - xdf['shift']) - np.log(xdf['close_2'] - xdf['shift_2'])) / (
                        xdf['mth_2'] - xdf['mth']) * 12.0
            xdf['logret'] = np.log(xdf['close'] - xdf['shift']) - np.log(xdf['close'].shift(1) - xdf['shift'])
            xdf['logret_2'] = np.log(xdf['close_2'] - xdf['shift_2']) - np.log(xdf['close_2'].shift(1) - xdf['shift_2'])
        elif need_shift == 2:
            xdf['ryield'] = (np.log(xdf['close']) - np.log(xdf['close_2']) - xdf['shift'] + xdf['shift_2']) / (
                        xdf['mth_2'] - xdf['mth']) * 12.0
            xdf['logret'] = np.log(xdf['close']) - np.log(xdf['close'].shift(1))
            xdf['logret_2'] = np.log(xdf['close_2']) - np.log(xdf['close_2'].shift(1))
        else:
            xdf['ryield'] = (np.log(xdf['close']) - np.log(xdf['close_2'])) / (xdf['mth_2'] - xdf['mth']) * 12.0
            xdf['logret'] = np.log(xdf['close']) - np.log(xdf['close'].shift(1))
            xdf['logret_2'] = np.log(xdf['close_2']) - np.log(xdf['close_2'].shift(1))
        xdf['baslr'] = xdf['logret'] - xdf['logret_2']
        xdf.index.name = 'date'
        for field in ['logret', 'baslr', 'ryield']:
            update_factor_db(xdf, field, fact_config, start_date=update_start, end_date=end_date, flavor = flavor)
        for scen in scenarios:
            sim_name = scen[0]
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
                xdf[data_field] = xdf[ref_field] - dh.EMA(xdf, ma_win, field = ref_field)
                extra_fields += ['xea', str(ma_win)]
            elif 'nma' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field] / dh.STDEV(xdf, ma_win, field = ref_field)
                extra_fields += ['nma', str(ma_win)]
            elif 'nmb' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = xdf[ref_field] / dh.BSTDEV(xdf, ma_win, field = ref_field)
                extra_fields += ['nmb', str(ma_win)]
            elif 'zlv' == run_mode[-3:]:
                ref_field = run_mode[:-3]
                xdf[data_field] = (xdf[ref_field] - xdf[ref_field].rolling(ma_win).mean()) \
                                        / dh.STDEV(xdf, ma_win, field = ref_field)
                extra_fields += ['zlv', str(ma_win)]
            else:
                ref_field = run_mode
            if pos_func:
                xdf[data_field] = xdf[data_field].apply(lambda x: pos_func(x, **pos_args))
                if len(pos_name) > 0:
                    extra_fields.append(pos_name) 
            fact_name = '_'.join([ref_field, str(win)] + extra_fields)
            xdf[fact_name] = xdf[data_field]
            if fact_name not in factor_repo:
                factor_repo[fact_name] = {}
                factor_repo[fact_name]['name'] = fact_name
                if 'ts' in sim_name:
                    factor_repo[fact_name]['type'] = 'ts'
                    factor_repo[fact_name]['threshold'] = 0.0               
                elif 'xs' in sim_name:
                    factor_repo[fact_name]['type'] = 'xs'
                    factor_repo[fact_name]['threshold'] = quantile
                else:
                    print("unsupported run mode")
                factor_repo[fact_name]['rebal'] = rebal
                factor_repo[fact_name]['param'] = params
                factor_repo[fact_name]['weight'] = weight 
            update_factor_db(xdf, fact_name, fact_config, start_date=update_start, end_date=end_date, flavor = flavor)
    return factor_repo

def create_strat_json(product_list, freq, roll_rule, factor_repo, filename= "C:\\dev\\data\\MM_FACT_PORT.json", name = 'default'):
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
        json.dump(strat_data, f)

