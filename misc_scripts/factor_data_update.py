import pandas as pd
import numpy as np
import json
import datetime
import copy
from sqlalchemy import create_engine
from pycmqlib3.utility.dbaccess import dbconfig, mysql_replace_into, connect
from pycmqlib3.utility.misc import nearby, cleanup_mindata, prod2exch, inst2contmth, \
    CHN_Holidays, contract_expiry, day_shift
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

trade_cont_map = {'rb': ['rb2105', 'rb2110'], 'hc': ['hc2105', 'hc2110'], 'i': ['i2105', 'i2109'],
                  'j': ['j2105', 'j2109'], 'jm': ['jm2102', 'jm2105'], 'ru': ['ru2105', 'ru2109'],
                  'FG': ['FG105', 'FG109'], 'ZC': ['ZC105', 'ZC109'], 'cu': ['cu2101', 'cu2102'],
                  'al': ['al2101', 'al2102'], 'zn': ['zn2101', 'zn2102'], 'ni': ['ni2101', 'ni2102'],
                  'sn': ['sn2102', 'sn2103'], }

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
        xdf['basmom'] = xdf['logret'] - xdf['logret_2']
        xdf.index.name = 'date'
        for field in ['logret', 'basmom', 'ryield']:
            update_factor_db(xdf, field, fact_config, start_date=update_start, end_date=end_date, flavor = flavor)
        for scen in scenarios:
            sim_name = scen[0]
            run_mode = scen[1]
            win = scen[2]
            ma_win = scen[3]
            rebal = scen[4]
            params = scen[5]
            fact_name = None
            if 'mom' in scen[1]:
                xdf['mom'] = xdf['logret'].rolling(win).sum()
            if 'rsi' in scen[1]:
                rsi_output = dh.RSI_F(xdf, win)
                xdf['rsi'] = rsi_output['RSI' + str(win)]
            if scen[1] == 'madist':
                xdf['ema1'] = dh.EMA(xdf, win, field='close')
                xdf['ema2'] = dh.EMA(xdf, int(win * params[0]), field='close')
                xdf['std'] = dh.STDEV(xdf, ma_win, field='close')
                xdf['ma_dev'] = (xdf['ema1'] - xdf['ema2']) / xdf['std']
                xdf['sig'] = xdf['ma_dev'] / dh.STDEV(xdf, int(ma_win * params[1]), field='ma_dev')
                fact_name = '_'.join(
                    [scen[1], str(win), str(int(win * params[0])), 'reverting', str(ma_win), str(int(ma_win * params[1]))])
                xdf[fact_name] = xdf['sig'].apply(lambda x: dh.response_curve(x, "reverting", param=2.0))
            elif scen[1] == 'rsi':
                fact_name = '_'.join(['rsi', str(win)])
                xdf[fact_name] = xdf['rsi']
            elif scen[1] == 'mom':
                fact_name = '_'.join(['mom', str(win)])
                xdf[fact_name] = xdf['mom']
            elif scen[1] == 'ryield':
                fact_name = '_'.join(['ryield', 'ma', str(ma_win)])
                xdf[fact_name] = xdf['ryield'].rolling(ma_win).mean()
            elif scen[1] == 'basmom':
                fact_name = '_'.join([scen[1], str(win), 'ma', str(ma_win)])
                xdf[fact_name] = xdf['basmom'].rolling(win).sum()
            elif 'ts' in scen[0]:
                if scen[1] == 'momma':
                    fact_name = '_'.join(['mom', str(win), 'xma', str(ma_win)])
                    xdf[fact_name] = xdf['mom'] - xdf['mom'].rolling(ma_win).mean()
                elif scen[1] == 'mixedmom':
                    xdf['tmpos'] = xdf['logret'].rolling(win).agg(lambda x: (x > 0).sum() / win)
                    xdf['tmneg'] = xdf['logret'].rolling(win).agg(lambda x: (x < 0).sum() / win)
                    xdf['pos_long'] = np.nan
                    flag = (xdf['mom'] > 0) & (xdf['tmpos'] > 0.5)
                    xdf.loc[flag, 'pos_long'] = 1.0
                    flag = (xdf['mom'] <= 0) | (xdf['tmpos'] <= 0.5)
                    xdf.loc[flag, 'pos_long'] = 0.0
                    xdf['pos_short'] = np.nan
                    flag = (xdf['mom'] < 0) & (xdf['tmneg'] > 0.5)
                    xdf.loc[flag, 'pos_short'] = -1.0
                    flag = (xdf['mom'] >= 0) | (xdf['tmneg'] <= 0.5)
                    xdf.loc[flag, 'pos_short'] = 0.0
                    fact_name = '_'.join([scen[1], str(win)])
                    xdf[fact_name] = xdf['pos_long'].fillna(method='ffill') + xdf['pos_short'].fillna(
                        method='ffill').fillna(0.0)
                elif scen[1] == 'rsima':
                    fact_name = '_'.join(['rsi', str(win), 'xma', str(ma_win)])
                    xdf[fact_name] = xdf['rsi'] - xdf['rsi'].rolling(ma_win).mean()
                elif scen[1] == 'ryieldma':
                    fact_name = '_'.join([scen[1], 'xma', str(ma_win)])
                    xdf[fact_name] = xdf['ryield'] - xdf['ryield'].rolling(ma_win).mean()
                elif scen[1] == 'basmomma':
                    fact_name = '_'.join(['basmom', str(win), 'xma', str(ma_win)])
                    basmom = xdf['basmom'].rolling(win).sum()
                    xdf[fact_name] = basmom - basmom.rolling(ma_win).mean()
            elif 'xs' in scen[0]:
                if scen[1] == 'momma':
                    fact_name = '_'.join(['mom', str(win), 'ma', str(ma_win)])
                    xdf[fact_name] = xdf['mom'].rolling(ma_win).mean()
                elif scen[1] == 'rsima':
                    fact_name = '_'.join(['rsi', str(win), 'ma', str(ma_win)])
                    xdf[fact_name] = xdf['rsi'].rolling(ma_win).mean()
                elif scen[1] == 'ryieldma':
                    fact_name = '_'.join(['ryield', 'ma', str(ma_win)])
                    xdf[fact_name] = xdf['ryield'].rolling(ma_win).mean()
                elif scen[1] == 'basmomma':
                    fact_name = '_'.join(['basmom', str(win), 'ma', str(ma_win)])
                    xdf[fact_name] = xdf['basmom'].rolling(win).sum().rolling(ma_win).mean()
            if fact_name not in factor_repo:
                factor_repo[fact_name] = {}
                factor_repo[fact_name]['name'] = fact_name
                if run_mode == 'mixedmom':
                    factor_repo[fact_name]['type'] = 'pos'
                elif 'ts' in sim_name:
                    factor_repo[fact_name]['type'] = 'ts'
                elif 'xs' in sim_name:
                    factor_repo[fact_name]['type'] = 'xs'
                else:
                    print("unsupported run mode")
                factor_repo[fact_name]['rebal'] = rebal
                factor_repo[fact_name]['param'] = params
                factor_repo[fact_name]['weight'] = 1.0
                if factor_repo[fact_name]['type'] == 'ts':
                    if scen[1] in ['madist']:
                        factor_repo[fact_name]['threshold'] = []
                    else:
                        factor_repo[fact_name]['threshold'] = [params, params]
                else:
                    factor_repo[fact_name]['threshold'] = []
            update_factor_db(xdf, fact_name, fact_config, start_date=update_start, end_date=end_date, flavor = flavor)
    return factor_repo

def create_strat_json(product_list, freq, roll_rule, factor_repo):
    strat_data = {}
    strat_data["class"] = "strat_factor_port.FactorPortTrader"
    strat_config = {}
    strat_config['name'] = 'MM_FACT_PORT'
    if freq == 'd':
        strat_config['freq'] = 's1'
    else:
        strat_config['freq'] = freq
    strat_config['roll_label'] = 'CAL_' + roll_rule
    strat_config['factor_repo'] = factor_repo
    strat_config['vol_win'] = 20
    strat_config['fact_db_table'] = 'fut_fact_data'
    strat_config['exec_bar_list'] = [1510]

    assets = []
    for asset in product_list:
        asset_data = {}
        asset_data['underliers'] = [trade_cont_map[asset][0]]
        asset_data['volumes'] = [1]
        asset_data['alloc_w'] = 1.0
        assets.append(asset_data)
    strat_config['assets'] = assets

    filtered_factors = {}
    for fact_name in factor_repo:
        if factor_repo[fact_name]['type'] in ['pos', 'ts']:
            filtered_factors[fact_name] = copy.copy(factor_repo[fact_name])
    strat_config['factor_repo'] = filtered_factors
    strat_data['config'] = strat_config
    filename = "C:\\dev\\data\\MM_FACT_PORT.json"
    with open(filename, 'w') as f:
        json.dump(strat_data, f)

def run_update(tday = datetime.date.today(), hist_tenor = '-2y'):
    end_date = tday
    start_date = day_shift(end_date, hist_tenor)

    scenarios_mixed = [('tscarry', 'ryield', 1, 1, 5, [0.0, 0.0]), \
                 ('tscarry', 'basmom', 60, 1, 10, [0.0, 0.0]), \
                 ('tscarry', 'basmom', 100, 1, 10, [0.0, 0.0]), \
                 ('tscarry', 'basmom', 240, 1, 10, [0.0, 0.0]), \
                 ('xscarry', 'ryieldma', 1, 1, 5, [0.0, 0.0]), \
                 ('xscarry', 'ryieldma', 1, 50, 5, [0.0, 0.0]), \
                 ('xscarry', 'basmom', 110, 1, 5, [0.0, 0.0]), \
                 ('xscarry', 'basmom', 140, 1, 5, [0.0, 0.0]), \
                 ('xscarry', 'basmomma', 90, 20, 5, [0.0, 0.0]), \
                 ('xscarry', 'basmomma', 230, 20, 5, [0.0, 0.0]), \
                 ('tsmom', 'momma', 20, 50, 5, [0.0, 0.0]), \
                 ('tsmom', 'momma', 30, 120, 5, [0.0, 0.0]), \
                 ('tsmom', 'momma', 40, 30, 5, [0.0, 0.0]), \
                 ('tsmom', 'mixedmom', 10, 1, 10, [0.0, 0.0]), \
                 ('tsmom', 'mixedmom', 20, 1, 10, [0.0, 0.0]), \
                 ('tsmom', 'rsima', 20, 30, 5, [0.0, 0.0]), \
                 ('tsmom', 'rsima', 40, 30, 5, [0.0, 0.0]), \
                 ('tsmom', 'rsima', 60, 30, 5, [0.0, 0.0]), \
                 ('tsmom', 'madist', 8, 80, 5, [1.5, 2.0]), \
                 ('tsmom', 'madist', 16, 80, 5, [1.5, 2.0]), \
                 ('tsmom', 'madist', 24, 80, 5, [1.5, 2.0]), \
                 ('xsmom', 'mom', 130, 1, 5, [0.0]), \
                 ('xsmom', 'mom', 230, 1, 5, [0.0]), \
                 ('xsmom', 'rsima', 60, 80, 5, [0.0]), \
                 ('xsmom', 'rsima', 10, 80, 5, [0.0]), \
                 ('xsmom', 'rsima', 40, 20, 5, [0.0]), \
                 ('xsmom', 'madist', 16, 100, 5, [1.5, 2.0]), \
                 ('xsmom', 'madist', 40, 100, 5, [1.5, 2.0]), \
                 ('xsmom', 'madist', 56, 140, 5, [1.5, 2.0])]

    mixed_metal_mkts = ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'ZC', 'cu', 'al', 'zn', 'ni']
    update_factor_data(mixed_metal_mkts, scenarios_mixed, start_date, end_date, roll_rule='30b')

    # commod_mkt = ags_all_mkts + ind_all_mkts
    # scenarios_all = [('tscarry', 'ryield', 1, 1, 1, [0.0, 0.0]), \
    #              ('tscarry', 'basmom', 70, 1, 1, [0.0, 0.0]), \
    #              ('tscarry', 'basmom', 110, 1, 1, [0.0, 0.0]), \
    #              ('tscarry', 'basmom', 230, 1, 1, [0.0, 0.0]), \
    #              ('xscarry', 'ryieldma', 1, 1, 1, [0.0, 0.0], 0.2), \
    #              ('xscarry', 'ryieldma', 1, 30, 1, [0.0, 0.0], 0.2), \
    #              ('xscarry', 'ryieldma', 1, 120, 1, [0.0, 0.0], 0.2), \
    #              # ('xscarry', 'basmom', 10, 1, 10, [0.0, 0.0], 0.2), \
    #              # ('xscarry', 'basmom', 30, 1, 10, [0.0, 0.0], 0.2), \
    #              ('xscarry', 'basmom', 100, 1, 1, [0.0, 0.0], 0.2), \
    #              ('xscarry', 'basmom', 240, 1, 1, [0.0, 0.0], 0.2), \
    #              # ('xscarry', 'basmomma', 100, 10, 5, [0.0, 0.0], 0.2), \
    #              # ('xscarry', 'basmomma',240, 10, 5, [0.0, 0.0], 0.2), \
    #              ('tsmom', 'momma', 40, 30, 5, [0.0]), \
    #              ('tsmom', 'momma', 40, 80, 5, [0.0]), \
    #              ('tsmom', 'mixedmom', 10, 1, 10, [0.0]), \
    #              ('tsmom', 'mixedmom', 20, 1, 10, [0.0]), \
    #              ('tsmom', 'mixedmom', 220, 1, 10, [0.0]), \
    #              ('tsmom', 'rsima', 30, 40, 5, [0.0]), \
    #              ('tsmom', 'rsima', 30, 110, 5, [0.0]), \
    #              ('tsmom', 'madist', 8, 80, 5, [1.5, 2.0]), \
    #              ('tsmom', 'madist', 16, 80, 5, [1.5, 2.0]), \
    #              ('tsmom', 'madist', 24, 80, 5, [1.5, 2.0]), \
    #              ('xsmom', 'mom', 20, 1, 5, [0.0]), \
    #              ('xsmom', 'mom', 210, 1, 5, [0.0]), \
    #              # ('xsmom', 'mom', 160, 1, 5, [0.0]),
    #              ('xsmom', 'momma', 140, 120, 5, [0.0]),
    #              ('xsmom', 'momma', 240, 120, 5, [0.0]), \
    #              ('xsmom', 'rsima', 70, 60, 5, [0.0]), \
    #              ('xsmom', 'rsima', 100, 80, 5, [0.0]), \
    #              ('xsmom', 'rsima', 90, 10, 5, [0.0]), \
    #              ('xsmom', 'madist', 8, 100, 5, [1.5, 2.0]), \
    #              ('xsmom', 'madist', 16, 100, 5, [1.5, 2.0]), \
    #              ('xsmom', 'madist', 32, 100, 5, [1.5, 2.0]), \
    #              # ('madist', 64, 100, 5, [1.5, 2.0]),
    #              ]
    # update_factor_data(commod_mkt, scenarios_all, start_date, end_date, roll_rule='30b')