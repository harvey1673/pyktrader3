import json
import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict
from pycmqlib3.utility.dbaccess import connect, bktest_dbconfig
from pycmqlib3.utility.misc import day_shift, product_lotsize
from pycmqlib3.analytics.ts_tool import merge_df

sim_config_folder = "C:/dev/pycmqlib3/pycmqlib3/btest_setup/"

output_columns = ['asset', 'sim_name', 'scen_id', 'std_unit', \
                'w_sharp', 'sharp_ratio_3m', 'sharp_ratio_6m', 'sharp_ratio_1y', 'sharp_ratio_2y', 'sharp_ratio_3y',\
                'tot_pnl_3m', 'tot_pnl_6m','tot_pnl_1y', 'tot_pnl_2y','tot_pnl_3y', \
                'tot_cost_3m', 'tot_cost_6m', 'tot_cost_1y', 'tot_cost_2y', 'tot_cost_3y', 'trade_count',\
                'max_drawdown_3m', 'max_drawdown_6m', 'max_drawdown_1y', 'max_drawdown_2y', 'max_drawdown_3y', \
                'par_name0', 'par_value0', 'par_name1', 'par_value1','par_name2', 'par_value2',\
                'par_name3', 'par_value3', 'par_name4', 'par_value4', ]

asset_list =  ["rb", "hc", "i", "j", "jm", "ZC", "ni", "ru", \
              "m", "RM", "FG", "y", "p", "OI", "a", "cs", "c", \
              "jd", "SR", "CF", "pp", "l", "v", "TA", "MA", "ag", \
              "au", "cu", "al", "zn", "SM", "SF", \
              "IF", "IH", "IC", "TF", "T", "sn", "AP", \
               "rb$1$-35b$fut_hc$1$-35b$fut", "SM$1$-35b$fut_SF$1$-35b$fut", \
               "m$1$-35b$fut_RM$1$-35b$fut", "a$1$-35b$fut_m$1$-35b$fut", \
               "y$1$-35b$fut_p$1$-35b$fut", "y$1$-35b$fut_OI$1$-35b$fut", \
               "p$1$-35b$fut_OI$1$-35b$fut", "l$1$-35b$fut_pp$1$-35b$fut", \
               "l$1$-35b$fut_v$1$-35b$fut", "pp$1$-35b$fut_v$1$-35b$fut", \
               "cs$1$-35b$fut_c$1$-35b$fut", "rb$1$-35b$fut_rb$2$-35b$fut", \
               "hc$1$-35b$fut_hc$2$-35b$fut", "i$1$-35b$fut_i$2$-35b$fut", \
               "j$1$-35b$fut_j$2$-35b$fut", "jm$1$-35b$fut_jm$2$-35b$fut", \
               "FG$1$-35b$fut_FG$2$-35b$fut", "ru$1$-35b$fut_ru$2$-35b$fut", \
               "m$1$-35b$fut_m$2$-35b$fut", "RM$1$-35b$fut_RM$2$-35b$fut", \
               "y$1$-35b$fut_y$2$-35b$fut", "p$1$-35b$fut_p$2$-35b$fut", \
               "OI$1$-35b$fut_OI$2$-35b$fut", "CF$1$-35b$fut_CF$2$-35b$fut", \
               "SR$1$-35b$fut_SR$2$-35b$fut", "jd$1$-35b$fut_jd$2$-35b$fut", \
               "l$1$-35b$fut_l$2$-35b$fut", "pp$1$-35b$fut_pp$2$-35b$fut", \
               "v$1$-35b$fut_v$2$-35b$fut", "TA$1$-35b$fut_TA$2$-35b$fut", \
               "MA$1$-35b$fut_MA$2$-35b$fut"]

def load_btest_res(sim_names, dbtable = 'bktest_output'):
    cnx = connect(**bktest_dbconfig)
    stmt = "select * from {dbtable} where sim_name in ('{qlist}')".format( \
        dbtable=dbtable, qlist="','".join(sim_names))
    df = pd.read_sql(stmt, cnx)
    cnx.close()
    return df

def load_btest_from_file(csvfile):
    sim_df = pd.read_csv(csvfile)
    df = load_btest_pnl(sim_df[['sim_name', 'asset', 'scen_id']].values)
    return df

def load_btest_pnl(sim_keys, dbtable  = 'bktest_output'):
    cnx = connect(**bktest_dbconfig)
    df_list = []
    for sim_key in sim_keys:
        stmt = "select * from {dbtable} where sim_name = '{name}' and asset = '{asset}' and scen_id = {scen}".format( \
            dbtable=dbtable, name = sim_key[1], asset = sim_key[0], scen = int(sim_key[2]))
        tdf = pd.read_sql(stmt, cnx)
        pnl_file = tdf['pnl_file'][0]
        xdf = pd.read_csv(pnl_file)
        xdf['date'] = xdf['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
        xdf = xdf[['date', 'daily_pnl']].set_index('date')
        xdf.rename(columns = {'daily_pnl': '-'.join([ str(k) for k in sim_key])}, inplace = True)
        df_list.append(xdf)
    df = ts_tool.merge_df(df_list)
    return df

def calc_mthly_by_prod(csvfile, field = 'asset', products = None, xlfile = None, start_date = None, freq = '3M'):
    port_data = pd.read_csv(csvfile)
    if xlfile:
        writer = pd.ExcelWriter(xlfile)
    df_dict = {}
    if products == None:
        products = port_data[field].unique()
    for prod in products:
        prod_port = port_data[port_data[field]==prod]
        prod_df = load_btest_pnl(prod_port[['asset', 'sim_name', 'scen_id']].values)
        if start_date:
            prod_df = prod_df[prod_df.index >= start_date]
        if xlfile:
            prod_df.fillna(0.0).cumsum().to_excel(writer, prod + '_D', startcol= 0, startrow=1)
        prod_df.index = pd.to_datetime(prod_df.index)
        if xlfile:
            prod_xdf = prod_df.fillna(0.0).resample(freq).sum()
            prod_xdf.to_excel(writer, prod + '_' + freq, startcol=0, startrow=1)
        df_dict[prod] = prod_df
    if xlfile:
        writer.save()
    return df_dict

def calc_cov_by_asset(df, asset = None, start_date = None, end_date = None, tenor = None, bias = False):
    if asset == None:
        columns = df.columns
    else:
        columns = [ col for col in df.columns if col.startswith(asset + '-')]
    xdf = df[columns].fillna(0.0)
    if end_date:
        xdf = xdf[xdf.index <= end_date]
    end_date = xdf.index[-1]
    if tenor:
        start_date = day_shift(end_date, tenor)
    if start_date:
        xdf = xdf[xdf.index >= start_date]
    corr = np.corrcoef(xdf.values.T, bias = bias)
    avg = np.mean(xdf.values, axis = 0)
    #weights = np.linalg.inv(cov).dot(avg)
    res = {#'weight': weights, \
           'avg': avg, 'corr': corr, 'columns': columns}
    return res

def greedy_param_optimizer(prod_port, req_dict = {'corr_rng': [-0.6, 0.6], 'max_selected': 10, \
                                                'pnl_cost_ratio': {'2y': 1.5, }, 'w_sharp': 0.75, \
                                                'trade_count': 60, 'sort_columns': ['w_sharp'], \
                                                'sharp_ratio': {}}, filter_asset = True):
    flag = pd.Series(True, index = prod_port.index)
    for field in ['w_sharp', 'trade_count']:
        if field in req_dict:
            flag = flag & (prod_port[field] >= req_dict[field])
    if 'pnl_cost_ratio' in req_dict:
        for tenor in req_dict['pnl_cost_ratio']:
            flag = flag & (prod_port['tot_pnl_' + tenor] >= req_dict['pnl_cost_ratio'][tenor] * prod_port['tot_cost_' + tenor])
    if 'sharp_ratio' in req_dict:
        for tenor in req_dict['sharp_ratio']:
            flag = flag & (prod_port["sharp_ratio_" + tenor] >= req_dict['sharp_ratio'][tenor])
    selected_port = prod_port[flag]
    if filter_asset:
        assets = selected_port['asset'].unique()
    else:
        assets = [None]
    res_df = pd.DataFrame()
    for asset in assets:
        if asset == None:
            asset_port = selected_port
        else:
            asset_port = selected_port[selected_port['asset'] == asset]
        asset_port = asset_port.sort_values(by= req_dict['sort_columns'], ascending=False).copy()
        num_strat = len(asset_port)
        if num_strat == 0:
            return pd.DataFrame()
        prod_df = load_btest_pnl(asset_port[['asset', 'sim_name', 'scen_id']].values)
        if 'start_date' in req_dict:
            prod_df = prod_df[prod_df.index >= req_dict['start_date']]
        if 'end_date' in req_dict:
            prod_df = prod_df[prod_df.index <= req_dict['end_date']]
        prod_df.index = pd.to_datetime(prod_df.index)
        res = calc_cov_by_asset(prod_df, asset = asset, \
                                start_date = req_dict.get('start_date', None), \
                                end_date = req_dict.get('end_date', None))
        corr = res['corr']
        selected_idx = []
        for i in range(num_strat):
            cond = True
            for idx in selected_idx:
                if (corr[idx][i] >= req_dict['corr_rng'][1]) or (corr[idx][i] <= req_dict['corr_rng'][0]):
                    cond = False
                    break
            if cond:
                selected_idx.append(i)
            if len(selected_idx) > req_dict.get('max_selected', 20):
                break
        res_df = res_df.append(asset_port.iloc[selected_idx,:])
    return res_df

def calc_weighted_port(df, weights, columns):
    combo_pnl = pd.Series(0.0, index = df.index)
    for w, col in zip(weights, columns):
        combo_pnl = combo_pnl + w * df[col]
    return combo_pnl

def load_sim_config(sim_names, config_folder = sim_config_folder):
    sim_dict = {}
    for sname in sim_names:
        config_file = config_folder + sname + '.json'
        with open(config_file, 'r') as fp:
            sim_dict[sname] = json.load(fp)
    return sim_dict

def filter_cond(df, key = 'sharp_ratio', cond = {'6m': 0.0, '1y': 0.5, '2y': 0.5}):
    selector = True
    for tenor in cond:
        selector = (df[key + '_' + tenor] >= cond[tenor]) & selector
    return df[selector]

def calc_w_col(df, key = 'sharp_ratio', weight = {'6m': 0.2, '1y': 0.4, '2y': 0.4}):
    ts = 0
    for tenor in weight:
        ts = ts + df[key + '_' + tenor] * weight[tenor]
    return ts

def extract_element(df, col_name, n):
    return df[col_name].apply(lambda x: json.loads(x)[n])

def simprod2inst_map(sim_product, inst_dict):
    if '_' in sim_product:
        products = sim_product.split('_')
        inst_list = []
        for prod in products:
            prod_info = prod.split('$')
            inst_list.append(inst_dict[prod_info[0]][int(prod_info[1])-1])
    else:
        inst_list = [inst_dict[sim_product][0]]
    return inst_list

def create_strat_json(df, inst_dict, asset_keys, common_keys, shared_dict, capital = 4000.0, strat_class = "strat_ma_system.MASystemTrader"):
    xdf = df.dropna(subset = ['name'])
    xdf['instID'] = xdf['asset'].apply(lambda x: json.dumps(simprod2inst_map(x, inst_dict)))
    output = OrderedDict()
    sim_names = xdf['sim_name'].unique()
    sim_dict = load_sim_config(sim_names, config_folder = sim_config_folder)
    for idx, row in xdf.iterrows():
        if row['name'] not in output:
            output[row['name']]  = {'class': strat_class,
                                    'config': OrderedDict([('name', row['name']), ('num_tick', 1), ('daily_close_buffer', 5), \
                                                           ('pos_scaler', 1.0), ('trade_valid_time', 600), ]),}
            for key in common_keys:
                if key in xdf:
                    if isinstance(row[key], str) and ('[' in row[key] and ']' in row[key]):
                        output[row['name']]['config'][key] = json.loads(row[key])
                    else:
                        output[row['name']]['config'][key] = row[key]
                elif key in sim_dict[row['sim_name']]:
                    output[row['name']]['config'][key] = sim_dict[row['sim_name']][key]
                elif key in sim_dict[row['sim_name']]['config']:
                    output[row['name']]['config'][key] = sim_dict[row['sim_name']]['config'][key]
            for key in shared_dict:
                output[row['name']]['config'][key] = shared_dict[key]
            output[row['name']]['config']['assets'] = []
        conf_dict = OrderedDict()
        conf_dict["underliers"] = [str(inst) for inst in json.loads(row['instID'])]
        for key in asset_keys:
            if key == 'alloc_w':
                #conf_dict[key] = round(capital / row['std_unit'], 1)
                conf_dict[key] = round(capital/row['std_unit'] * int(row['w_sharp']/0.2)*0.2, 1)
            elif key == 'volumes':
                if 'trade_weight_dict' in sim_dict[row['sim_name']]:
                    conf_dict['volumes'] = sim_dict[row['sim_name']]['trade_weight_dict'].get(row['asset'], [1, -1])
                else:
                    conf_dict['volumes'] = [1]
            elif key in xdf:
                if isinstance(row[key], str) and ('[' in row[key] and ']' in row[key]):
                    conf_dict[key] = json.loads(row[key])
                else:
                    conf_dict[key] = row[key]
            elif key in sim_dict[row['sim_name']]:
                conf_dict[key] = sim_dict[row['sim_name']][key]
            elif key in sim_dict[row['sim_name']]['config']:
                conf_dict[key] = sim_dict[row['sim_name']]['config'][key]
        if len(conf_dict["underliers"]) > 1:
            conf_dict["exec_class"] = "ExecAlgoFixTimer"
            conf_dict["exec_args"] = {"max_vol": 50, "time_period": 600, "price_type": "2", "tick_num": 1, "order_offset": True}
        output[row['name']]['config']['assets'].append(conf_dict)
    return output

def create_xs_strat_json(df, inst_dict, asset_keys, common_keys, batch_keys, shared_dict, capital = 4000.0, strat_class = "strat_xsmom.XSMOMRetTrader"):
    xdf = df.dropna(subset = ['name'])
    output = OrderedDict()
    sim_names = xdf['sim_name'].unique()
    sim_dict = load_sim_config(sim_names, config_folder = sim_config_folder)
    for idx, row in xdf.iterrows():
        strat_key = row['name']
        if strat_key not in output:
            output[strat_key]  = {'class': strat_class,
                                    'config': OrderedDict([('name', strat_key), ('num_tick', 1), ('daily_close_buffer', 5), \
                                                           ('pos_scaler', capital), ('trade_valid_time', 600), \
                                                           ("batch_setup", {}), ("assets", [])]),}
        products = [simprod2inst_map(key.split('$')[0], inst_dict)[0] for key in row['asset'].split('_')]
        batch_name = ''.join([prod[0].upper() for prod in products])
        print(batch_name)
        output[strat_key]['config']['batch_setup'][batch_name] = [row[k] for k in batch_keys]
        for key in common_keys:
            if key in xdf:
                if isinstance(row[key], str) and ('[' in row[key] and ']' in row[key]):
                    output[row['name']]['config'][key] = json.loads(row[key])
                else:
                    output[row['name']]['config'][key] = row[key]
            elif key in sim_dict[row['sim_name']]:
                output[row['name']]['config'][key] = sim_dict[row['sim_name']][key]
            elif key in sim_dict[row['sim_name']]['config']:
                output[row['name']]['config'][key] = sim_dict[row['sim_name']]['config'][key]
        for key in shared_dict:
            output[row['name']]['config'][key] = shared_dict[key]
        for prod in products:
            conf_dict = OrderedDict()
            conf_dict["underliers"] = [prod]
            conf_dict['batch'] = batch_name
            for key in asset_keys:
                if key == 'alloc_w':
                    conf_dict[key] = 1.0
                elif key in xdf:
                    if isinstance(row[key], str) and ('[' in row[key] and ']' in row[key]):
                        conf_dict[key] = json.loads(row[key])
                    else:
                        conf_dict[key] = row[key]
                elif key in sim_dict[row['sim_name']]:
                    conf_dict[key] = sim_dict[row['sim_name']][key]
                elif key in sim_dict[row['sim_name']]['config']:
                    conf_dict[key] = sim_dict[row['sim_name']]['config'][key]
            if len(conf_dict["underliers"]) > 1:
                conf_dict["exec_class"] = "ExecAlgoFixTimer"
                conf_dict["exec_args"] = {"max_vol": 50, "time_period": 600, "price_type": "2", "tick_num": 1, "order_offset": True}
            print(conf_dict)
            output[row['name']]['config']['assets'].append(conf_dict)
    return output

def process_sim_results(sim_name, out, corr_scen = [0.7, 0.6, 0.5], ratio_scen = [(1.5, 0.6), (1.0, 1.0)], \
                        req_base ={'max_selected': 20, 'trade_count': 60, 'sort_columns': ['w_sharp'], \
                                   'sharp_ratio': {'1y': 0.0, '2y': 0.0}}):
    xlfile = sim_name + ".xlsx"
    if len(sim_name) > 0:
        writer = pd.ExcelWriter(xlfile)
        out.to_excel(writer, 'all', startcol=0, startrow=1)
    res = {}
    for idx, (cost_ratio, sr_cutoff) in enumerate(ratio_scen):
        for idy, corr in enumerate(corr_scen):
            print("run param optimization for corr = %s, pnl_cost_rato = %s, SR_cutoff = %s" % (corr, cost_ratio, sr_cutoff))
            req_dict = dict(req_base, **{'corr_rng': [-corr, corr], 'pnl_cost_ratio': {'2y': cost_ratio, }, 'w_sharp': sr_cutoff})
            filtered_set = greedy_param_optimizer(out, req_dict)
            res[(idx,idy)] = filtered_set
            if len(sim_name) > 0:
                filtered_set.to_excel(writer, 'cr=%s_sr=%s_corr=%s' % (cost_ratio, sr_cutoff, corr), startcol=0, startrow=1)
    if len(sim_name) > 0:
        writer.save()
    return res

def process_OpenbrkSim():
    sim_names = ['openbrk_200228', 'openbrk_dchan_200228', 'openbrk_pct25_200228', 'openbrk_pct45_200228', 'spd_openbrk_200228']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    ten = [0.5, 1.0, 2.0, 3.0, 4.0]
    tdiff = [0.5, 0.5, 1.0, 1.0, 1.0]
    w = [0.6, 0.5, 0.8, 0.5, 0.0]
    w_t = [ ten[i] * (w[i]/tdiff[i] - w[i+1]/tdiff[i+1]) for i in range(len(w)-1)]
    weight = {'6m': w_t[0]/sum(w), '1y': w_t[1]/sum(w), '2y': w_t[2]/sum(w), '3y': w_t[3]/sum(w)}
    df['w_sharp'] = calc_w_col(df, key = 'sharp_ratio', weight = weight)
    df['ma_chan'] = 0
    df['ma_width'] = 0.5
    df['channels'] = 0
    df['channel_type'] = 0
    df['min_rng'] = 0.4
    df['split_mode'] = df['par_value0']
    df['lookbacks'] = df['par_value1']
    df['ratios'] = df['par_value2']
    df['close_daily'] = False

    df['volumes'] = '[1]'
    flag = df['asset'].str.contains('_')
    df.ix[flag, 'volume'] = '[1, -1]'
    # flag = df['split_mode'] == 's1'
    # df.ix[flag, 'open_period'] = '[300, 2115]'
    # flag = df['split_mode'] == 's2'
    # df.ix[flag, 'open_period'] = '[300, 1500, 2115]'
    # flag = df['split_mode'] == 's3'
    # df.ix[flag, 'open_period'] = '[300, 1500, 1900, 2115]'
    # flag = df['split_mode'] == 's4'
    # df.ix[flag, 'open_period'] = '[300, 1500, 1630, 1900, 2115]'

    filter = (df.sim_name == 'openbrk_200228') | (df.sim_name == 'spd_openbrk_200228')
    df.ix[filter, 'ma_chan'] = 20
    df.ix[filter, 'trend_factor'] = df.ix[filter, 'par_value3']
    df.ix[filter, 'price_mode'] = df.ix[filter, 'par_value4']
    df.ix[filter, 'vol_ratio'] = '[1.0, 0.0]'
    df.ix[filter, 'channels'] = 0

    df.ix[~filter, 'channels'] = df.ix[~filter, 'par_value3']
    df.ix[~filter, 'vol_ratio'] = '[0.0, 1.0]'
    df.ix[~filter, 'price_mode'] = 'HL'
    df.ix[~filter, 'trend_factor'] = 0.0

    filter = (df.sim_name == 'openbrk_dchan_200228')
    df.ix[filter, 'channel_type'] = 0
    df.ix[filter, 'trend_factor'] = 0.0
    filter = (df.sim_name == 'openbrk_pct10_200228')
    df.ix[filter, 'channel_type'] = 1
    df.ix[filter, 'trend_factor'] = 0.0
    filter = (df.sim_name == 'openbrk_pct25_200228')
    df.ix[filter, 'channel_type'] = 2
    df.ix[filter, 'trend_factor'] = 0.0
    filter = (df.sim_name == 'openbrk_pct45_200228')
    df.ix[filter, 'channel_type'] = 3
    df.ix[filter, 'trend_factor'] = 0.0
    df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x.split('$')[0]])
    df['std_unit'] = df['std_pnl_1y'] * df['lot_size']
    assets = asset_list
    res = pd.DataFrame()
    for asset in assets:
        xdf = df[(df.asset==asset) & (df.w_sharp > 0.5)]
        xdf1 = xdf[((xdf.sim_name == 'openbrk_200228') | (xdf.sim_name == 'spd_openbrk_200228')) \
                   & (xdf.split_mode == 's1')].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 30:
            xdf1 = xdf1[:30]
        res = res.append(xdf1, ignore_index=True)
        xdf1 = xdf[((xdf.sim_name == 'openbrk_200228')| (xdf.sim_name == 'spd_openbrk_200228')) \
                   & (xdf.split_mode == 's2')].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 30:
            xdf1 = xdf1[:30]
        res = res.append(xdf1, ignore_index=True)
        xdf1 = xdf[((xdf.sim_name == 'openbrk_200228')| (xdf.sim_name == 'spd_openbrk_200228')) \
                   & ((xdf.split_mode == 's3') | (xdf.split_mode == 's4'))].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 30:
            xdf1 = xdf1[:30]
        res = res.append(xdf1, ignore_index=True)
        xdf1 = xdf[(xdf.sim_name == 'openbrk_dchan_200228') | (xdf.sim_name == 'openbrk_pct25_200228')\
            | (xdf.sim_name == 'openbrk_pct45_200228') | (xdf.sim_name == 'openbrk_pct10_200228')].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 50:
            xdf1 = xdf1[:50]
        res = res.append(xdf1, ignore_index=True)
    res['trend_factor'] = res['trend_factor'].fillna(0.0)
    res['price_mode'] =res['price_mode'].fillna('CL')
    out_cols = output_columns + ['close_daily', 'split_mode', 'close_daily', 'lookbacks', 'ratios', 'trend_factor', 'channel_type', 'channels', 'ma_chan', 'ma_width', 'price_mode', \
                                 'lot_size', 'min_rng', 'vol_ratio', 'volumes']
    out = res[out_cols]
    req_base = {'max_selected': 20, 'trade_count': 60, 'sort_columns': ['w_sharp'], \
                'sharp_ratio': {'1y': 0.0, '2y': 0.0}}
    corr_scen = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    ratio_scen = [(1.5, 0.6)]
    process_sim_results("openbrk_results_200228", out, corr_scen, ratio_scen, req_base)
    return out

def process_RsiatrSim():
    sim_names = ['rsi_atr_1m_200228', 'rsi_atr_3m_200228', 'rsi_atr_5m_200228', 'rsi_atr_15m_200228', \
                 'rsi_atr_30m_200228']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    ten = [0.5, 1.0, 2.0, 3.0, 4.0]
    tdiff = [0.5, 0.5, 1.0, 1.0, 1.0]
    w = [0.6, 0.5, 0.8, 0.5, 0.0]
    w_t = [ten[i] * (w[i] / tdiff[i] - w[i + 1] / tdiff[i + 1]) for i in range(len(w) - 1)]
    weight = {'6m': w_t[0] / sum(w), '1y': w_t[1] / sum(w), '2y': w_t[2] / sum(w), '3y': w_t[3] / sum(w)}
    df['w_sharp'] = calc_w_col(df, key = 'sharp_ratio', weight = weight)
    df['atr_win'] = 14
    df['freq'] = 1
    df.ix[(df.sim_name ==  'rsi_atr_3m_200228'), 'freq'] = 3
    df.ix[(df.sim_name ==  'rsi_atr_5m_200228'), 'freq'] = 5
    df.ix[(df.sim_name ==  'rsi_atr_15m_200228'), 'freq'] = 15
    df.ix[(df.sim_name ==  'rsi_atr_30m_200228'), 'freq'] = 30
    df['rsi_th'] = df['par_value0']
    df['rsi_win'] = df['par_value1']
    df['atrma_win'] = df['par_value2']
    df['stoploss'] = df['par_value3']
    df['close_tday'] = df['par_value4']
    df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x.split('$')[0]])
    df['std_unit'] = df['std_pnl_1y'] * df['lot_size']
    df['volumes'] = '[1]'
    assets = asset_list
    res = pd.DataFrame()
    for asset in assets:
        xdf = df[(df.asset==asset) & (df.w_sharp > 0.6)]
        xdf1 = xdf[(xdf.sim_name == 'rsi_atr_1m_200228') | (xdf.sim_name == 'rsi_atr_3m_200228') \
                   | (xdf.sim_name == 'rsi_atr_5m_200228')].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 30:
            xdf1 = xdf1[:30]
        res = res.append(xdf1, ignore_index=True)
        xdf2 = xdf[(xdf.sim_name == 'rsi_atr_15m_200228') | (xdf.sim_name == 'rsi_atr_30m_200228')].sort_values('w_sharp', ascending=False)
        if len(xdf2) > 30:
            xdf2 = xdf2[:30]
        res = res.append(xdf2, ignore_index=True)

    out_cols = output_columns + ['freq', 'rsi_th', 'rsi_win', 'stoploss', 'atr_win', 'atrma_win', 'close_tday', \
                                 'std_unit', 'lot_size', 'volumes']
    out = res[out_cols]
    req_base = {'max_selected': 20, 'trade_count': 60, 'sort_columns': ['w_sharp'], \
                'sharp_ratio': {'1y': 0.0, '2y': 0.0}}
    corr_scen = [0.75, 0.7, 0.65, 0.6]
    ratio_scen = [(1.5, 0.6), (1.0, 1.0)]
    process_sim_results("RsiAtr_results_200228", out, corr_scen, ratio_scen, req_base)
    return out

def process_ChanbreakSim():
    sim_names = ['chanbreak_200228', 'chanbreak_ts_200228', 'spd_chanbreak_200228']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    ten = [0.5, 1.0, 2.0, 3.0, 4.0]
    tdiff = [0.5, 0.5, 1.0, 1.0, 1.0]
    w = [0.6, 0.5, 0.8, 0.5, 0.0]
    w_t = [ten[i] * (w[i] / tdiff[i] - w[i + 1] / tdiff[i + 1]) for i in range(len(w) - 1)]
    weight = {'6m': w_t[0] / sum(w), '1y': w_t[1] / sum(w), '2y': w_t[2] / sum(w), '3y': w_t[3] / sum(w)}
    df['w_sharp'] = calc_w_col(df, key = 'sharp_ratio', weight = weight)
    df['split_mode'] = df['par_value0']
    df['stoploss_win'] = df['par_value1']
    df['stoploss'] = df['par_value2']
    df['channel'] = df['par_value3']
    df['entry_chan'] = extract_element(df, 'channel', 0)
    df['exit_chan'] = extract_element(df, 'channel', 1)
    df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x.split('$')[0]])
    df['std_unit'] = df['std_pnl_1y'] * df['lot_size']
    df['volumes'] = '[1]'
    #df['open_period'] = '[300, 2115]'
    #df.ix[(df.split_mode == 's2'), 'open_period'] = '[300, 1500, 2115]'
    #df.ix[(df.split_mode == 's3'), 'open_period'] = '[300, 1500, 1900, 2115]'
    #df.ix[(df.split_mode == 's4'), 'open_period'] = '[300, 1500, 1630, 1900, 2115]'
    assets = asset_list
    res = pd.DataFrame()
    for asset in assets:
        xdf = df[(df.asset==asset) & (df.w_sharp > 0.6)]
        filter = (xdf['split_mode'] == 's1') | (xdf['split_mode'] == 's2')
        xdf1 = xdf[filter].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 10:
            xdf1 = xdf1[:10]
        res = res.append(xdf1, ignore_index=True)
        filter = (xdf['split_mode'] == 's3') | (xdf['split_mode'] == 's4')
        xdf1 = xdf[filter].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 10:
            xdf1 = xdf1[:10]
        res = res.append(xdf1, ignore_index=True)
    out_cols = output_columns + ['stoploss_win', 'stoploss', 'channel', 'split_mode', 'std_unit', 'lot_size', 'volumes']
    out = res[out_cols]
    req_base = {'max_selected': 20, 'trade_count': 30, 'sort_columns': ['w_sharp'], \
                'sharp_ratio': {'1y': 0.0, '2y': 0.0}}
    corr_scen = [0.75, 0.7, 0.65, 0.6]
    ratio_scen = [(1.5, 0.6), (1.0, 1.2)]
    process_sim_results("Chanbreak_results_200228", out, corr_scen, ratio_scen, req_base)
    return out

def process_Ma2crossSim():
    sim_names = ['ma2cross_200228', 'spd_ma2cross_200228']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    ten = [0.5, 1.0, 2.0, 3.0, 4.0]
    tdiff = [0.5, 0.5, 1.0, 1.0, 1.0]
    w = [0.6, 0.5, 0.8, 0.5, 0.0]
    w_t = [ten[i] * (w[i] / tdiff[i] - w[i + 1] / tdiff[i + 1]) for i in range(len(w) - 1)]
    weight = {'6m': w_t[0] / sum(w), '1y': w_t[1] / sum(w), '2y': w_t[2] / sum(w), '3y': w_t[3] / sum(w)}
    df['w_sharp'] = calc_w_col(df, key = 'sharp_ratio', weight = weight)
    df['freq'] = df['par_value0']
    df['win_s'] = extract_element(df, 'par_value1', 0)
    df['win_l'] = extract_element(df, 'par_value1', 1)
    df['ma_win'] = df['par_value1']
    #flag = df.simname == 'ma2cross_200228'
    #df.ix[flag, 'lot_size'] = df.ix[flag, 'asset'].apply(lambda x: product_lotsize[x])
    #df.ix[~flag, 'lot_size'] = df.ix[~flag, 'asset'].apply(lambda x: product_lotsize[x.split('$')[0]])
    df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x.split('$')[0]])
    df['std_unit'] = df['std_pnl_1y'] * df['lot_size']
    df['volumes'] = '[1]'
    assets = asset_list
    res = pd.DataFrame()
    for asset in assets:
        xdf = df[(df.asset==asset) & (df.w_sharp > 0.5)]
        xdf1 = xdf.sort_values('w_sharp', ascending=False)
        if len(xdf1) > 20:
            xdf1 = xdf1[:20]
        res = res.append(xdf1, ignore_index=True)
    out_cols = output_columns + ['freq', 'ma_win', 'win_l', 'win_s', 'std_unit', 'lot_size', 'volumes']
    out = res[out_cols]
    req_base = {'max_selected': 20, 'trade_count': 10, 'sort_columns': ['w_sharp'], \
                'sharp_ratio': {'1y': 0.0, '2y': 0.0}}
    corr_scen = [0.75, 0.7, 0.65, 0.6]
    ratio_scen = [(1.5, 0.6), (1.0, 1.2)]
    process_sim_results("ma2cross_results_200228", out, corr_scen, ratio_scen, req_base)
    return out

def process_MAChanSim():
    sim_names = ['MA2cross_daily_200228']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    ten = [0.5, 1.0, 2.0, 3.0, 4.0]
    tdiff = [0.5, 0.5, 1.0, 1.0, 1.0]
    w = [0.6, 0.5, 0.8, 0.5, 0.0]
    w_t = [ten[i] * (w[i] / tdiff[i] - w[i + 1] / tdiff[i + 1]) for i in range(len(w) - 1)]
    weight = {'6m': w_t[0] / sum(w), '1y': w_t[1] / sum(w), '2y': w_t[2] / sum(w), '3y': w_t[3] / sum(w)}
    df['w_sharp'] = calc_w_col(df, key = 'sharp_ratio', weight = weight)
    df['win_s'] = extract_element(df, 'par_value0', 0)
    df['win_l'] = extract_element(df, 'par_value0', 1)
    df['win_list'] = df['par_value0']
    df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x])
    df['std_unit'] = df['std_pnl_1y'] * df['lot_size']
    df['volumes'] = '[1]'
    assets = asset_list
    res = pd.DataFrame()
    for asset in assets:
        xdf = df[(df.asset==asset) & (df.w_sharp > 0.5)]
        xdf1 = xdf.sort_values('w_sharp', ascending=False)
        if len(xdf1) > 20:
            xdf1 = xdf1[:20]
        res = res.append(xdf1, ignore_index=True)
    out_cols = output_columns + ['win_l', 'win_s', 'std_unit', 'lot_size', 'volumes']
    out = res[out_cols]
    req_base = {'max_selected': 20, 'trade_count': 10, 'sort_columns': ['w_sharp'], \
                'sharp_ratio': {'1y': 0.0, '2y': 0.0}}
    corr_scen = [0.75, 0.7, 0.65, 0.6]
    ratio_scen = [(1.5, 0.6), (1.0, 1.2)]
    process_sim_results("MA2cross_results_200228", out, corr_scen, ratio_scen, req_base)
    return out

def process_MARibbonSim():
    sim_names = ['ma_ribbon']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    weight = {'6m': 0.5/2.5, '1y': 1.0/3.5, '2y': 1.0/3.5, '3y': 1.0/3.5}
    df['w_sharp'] = calc_w_col(df, key = 'sharp_ratio', weight = weight)
    df['freq'] = df['par_value0']
    df['param'] = df['par_value1']
    df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x])
    df['std_unit'] = df['std_pnl_1y'] * df['lot_size']
    df['volumes'] = '[1]'
    assets = asset_list
    res = pd.DataFrame()
    for asset in assets:
        xdf = df[(df.asset==asset) & (df.w_sharp > 0.5)]
        filter = (xdf['freq']=='1min') | (xdf['freq']=='3min') | (xdf['freq']=='5min')
        xdf1 = xdf[filter].sort_values('w_sharp', ascending=False)
        if len(xdf1) > 20:
            xdf1 = xdf1[:20]
        res = res.append(xdf1, ignore_index=True)
        xdf2 = xdf[~filter].sort_values('w_sharp', ascending=False)
        if len(xdf2) > 20:
            xdf2 = xdf2[:20]
        res = res.append(xdf2, ignore_index=True)

    out_cols = output_columns + ['param', 'std_unit', 'lot_size', 'volumes', 'freq']
    out = res[out_cols]
    req_base = {'max_selected': 20, 'trade_count': 10, 'sort_columns': ['w_sharp'], \
                'sharp_ratio': {'1y': 0.0, '2y': 0.0}}
    corr_scen = [0.75, 0.7, 0.65, 0.6]
    ratio_scen = [(1.5, 0.6), (1.0, 1.2)]
    process_sim_results("ma_ribbon_200228", out, corr_scen, ratio_scen, req_base)
    return out

def process_XSMOMSim():
    sim_names = ['xsmom_ret_200228']
    df = load_btest_res(sim_names)
    filter = df['sharp_ratio_3y'].isnull()
    df.ix[filter, 'sharp_ratio_3y'] = df.ix[filter, 'sharp_ratio_2y']
    ten = [0.5, 1.0, 2.0, 3.0, 4.0]
    tdiff = [0.5, 0.5, 1.0, 1.0, 1.0]
    w = [0.6, 0.5, 0.8, 0.5, 0.0]
    w_t = [ten[i] * (w[i] / tdiff[i] - w[i + 1] / tdiff[i + 1]) for i in range(len(w) - 1)]
    weight = {'6m': w_t[0] / sum(w), '1y': w_t[1] / sum(w), '2y': w_t[2] / sum(w), '3y': w_t[3] / sum(w)}
    df['w_sharp'] = calc_w_col(df, key='sharp_ratio', weight=weight)
    df['freq'] = df['par_value0']
    df['lookback'] = extract_element(df, 'par_value1', 0)
    df['ma_win'] = df['par_value2']
    df['rebal_freq'] = df['par_value3']
    df['quantile'] = 0.2
    #df['lot_size'] = df['asset'].apply(lambda x: product_lotsize[x.split('$')[0]])
    df['std_unit'] = df['std_pnl_1y']
    df['volumes'] = '[1]'

    all_group = [
                [["rb$1$-35b$fut", "i$1$-35b$fut", "j$1$-35b$fut", "jm$1$-35b$fut"], \
                  ["rb$1$-35b$fut", "hc$1$-35b$fut", "i$1$-35b$fut", "j$1$-35b$fut"], \
                  ["rb$1$-35b$fut", "i$1$-35b$fut", "j$1$-35b$fut"], \
                  ["rb$1$-35b$fut", "hc$1$-35b$fut", "i$1$-35b$fut", "j$1$-35b$fut", "jm$1$-35b$fut"]],\
                [["ZC$1$-35b$fut", "ru$1$-35b$fut", "SM$1$-35b$fut", "FG$1$-35b$fut", "SF$1$-35b$fut"], \
                 ["ZC$1$-35b$fut", "jm$1$-35b$fut", "ru$1$-35b$fut", "FG$1$-35b$fut"], \
                 ["ZC$1$-35b$fut", "jm$1$-35b$fut", "ru$1$-35b$fut", "FG$1$-35b$fut", "SM$1$-35b$fut"], \
                 ["ZC$1$-35b$fut", "jm$1$-35b$fut", "ru$1$-35b$fut", "FG$1$-35b$fut", "SM$1$-35b$fut", "SF$1$-35b$fut"]],  \
                [["cu$1$-35b$fut", "al$1$-35b$fut", "zn$1$-35b$fut", "ni$1$-35b$fut"],\
                 ["cu$1$-35b$fut", "al$1$-35b$fut", "zn$1$-35b$fut", "ni$1$-35b$fut"], \
                 ["cu$1$-35b$fut", "al$1$-35b$fut", "zn$1$-35b$fut", "pb$1$-35b$fut"], \
                 ["cu$1$-35b$fut", "al$1$-35b$fut", "zn$1$-35b$fut"], \
                 ["cu$1$-35b$fut", "al$1$-35b$fut", "zn$1$-35b$fut", "pb$1$-35b$fut", "ni$1$-35b$fut", "sn$1$-35b$fut"], \
                 ["cu$1$-35b$fut", "al$1$-35b$fut", "zn$1$-35b$fut", "pb$1$-35b$fut", "sn$1$-35b$fut"]],\
                [["m$1$-35b$fut", "RM$1$-35b$fut", "y$1$-35b$fut", "p$1$-35b$fut", "OI$1$-35b$fut"], \
                 ["y$1$-35b$fut", "p$1$-35b$fut", "OI$1$-35b$fut"], ["m$1$-35b$fut", "RM$1$-35b$fut", "a$1$-35b$fut"]], \
                [["a$1$-35b$fut", "c$1$-35b$fut", "cs$1$-35b$fut", "jd$1$-35b$fut"],\
                 ["CF$1$-35b$fut", "SR$1$-35b$fut", "a$1$-35b$fut", "jd$1$-35b$fut"]],
                [["l$1$-35b$fut", "pp$1$-35b$fut", "v$1$-35b$fut", "TA$1$-35b$fut", "MA$1$-35b$fut"],\
                 ["l$1$-35b$fut", "pp$1$-35b$fut", "v$1$-35b$fut"]]
        ]
    req_base = {'max_selected': 20, 'sort_columns': ['w_sharp']}
    corr_scen = [0.8, 0.7, 0.6]
    ratio_scen = [(1.4, 0.6), (1.0, 1.0)]
    out_cols = output_columns + ['freq', 'lookback', 'ma_win', 'rebal_freq', 'std_unit', 'volumes']
    out_res = {'all': pd.DataFrame()}
    for sim_group in all_group:
        grp_res = pd.DataFrame()
        for idx, grp in enumerate(sim_group):
            grp_name = '_'.join(grp)
            #if idx == 0:
            flag = (df['asset'] == grp_name) & (df['w_sharp']>0)
            #else:
            #    flag = flag | ((df['asset'] == grp_name) & (df['w_sharp']>0))
            xdf = df[flag].copy()
            res = pd.DataFrame()
            for freq_key in ['m', 's']:
                flag = xdf['freq'].str.contains(freq_key) & (xdf['tot_pnl_3y']>xdf['tot_cost_3y'])
                xdf1 = xdf[flag].copy()
                xdf1 = xdf1.sort_values('w_sharp', ascending=False)
                if len(xdf1) > 50:
                    xdf1 = xdf1[:50]
                res = res.append(xdf1, ignore_index=True)
            res = res.sort_values('w_sharp', ascending=False)
            xdf2 = greedy_param_optimizer(res, {'corr_rng': [-0.8, 0.8], 'max_selected': 80, 'sort_columns': ['w_sharp']}, filter_asset = False)
            grp_res = grp_res.append(xdf2, ignore_index=True)
        grp_res = grp_res.sort_values('w_sharp', ascending=False)
        if len(grp_res) > 50:
            grp_res = grp_res[:50]
        out = grp_res[out_cols]
        out_res['all'] = out_res['all'].append(out, ignore_index=True)
        for idx, (cost_ratio, sr_cutoff) in enumerate(ratio_scen):
            for idy, corr in enumerate(corr_scen):
                print("run param optimization for group = %s, corr = %s, pnl_cost_rato = %s, SR_cutoff = %s" % (grp_name, corr, cost_ratio, sr_cutoff))
                key = (idx, idy)
                if key not in out_res:
                    out_res[key] = pd.DataFrame()
                req_dict = dict(req_base, **{'corr_rng': [-corr, corr], 'pnl_cost_ratio': {'3y': cost_ratio, },
                                             'w_sharp': sr_cutoff})
                if len(out) > 0:
                    filtered_set = greedy_param_optimizer(out, req_dict, filter_asset = False)
                    if len(filtered_set) > 0:
                        out_res[key] = out_res[key].append(filtered_set, ignore_index=True)
    xlfile = sim_names[0] + ".xlsx"
    writer = pd.ExcelWriter(xlfile)
    out_res['all'].to_excel(writer, 'all', startcol=0, startrow=1)
    for idx, (cost_ratio, sr_cutoff) in enumerate(ratio_scen):
        for idy, corr in enumerate(corr_scen):
                out_res[(idx,idy)].to_excel(writer, 'cr=%s_sr=%s_corr=%s' % (cost_ratio, sr_cutoff, corr), startcol=0, startrow=1)
    writer.save()
    return out_res

prod_inst_map = {
    'rb': ['rb2005', 'rb2010'], 'hc': ['hc2005', 'hc2010'], 'i': ['i2005', 'i2009'],    \
    'j': ['j2005', 'j2009'],    'jm': ['jm2005', 'jm2009'], 'ZC': ['ZC005', 'ZC009'],   \
    'ni': ['ni2005', 'ni2006'], 'ru': ['ru2005', 'ru2009'], 'FG': ['FG005', 'FG009'],   \
    'SM': ['SM005', 'SM009'],   'SF': ['SF005', 'SF009'],   'AP': ['AP005', 'AP010'],   \
    'm': ['m2005', 'm2009'],    'RM': ['RM005', 'RM009'],   'y': ['y2005', 'y2009'],    \
    'p': ['p2005', 'p2009'],    'OI': ['OI005', 'OI009'],   'cs': ['cs2005', 'cs2009'], \
    'c': ['c2005', 'c2009'],    'jd': ['jd2005', 'jd2009'], 'a': ['a2005', 'a2009'],    \
    'pp': ['pp2005', 'pp2009'], 'l': ['l2005', 'l2009'],    'v': ['v2005', 'v2009'],    \
    'MA': ['MA005', 'MA009'],   'CF': ['CF005', 'CF009'],   'TA': ['TA005', 'TA009'],   \
    'SR': ['SR005', 'SR009'],   'CJ': ['CJ005', 'CJ009'],   'CY': ['CY005', 'CY009'],   \
    'al': ['al2005', 'al2006'], 'cu': ['cu2005', 'cu2006'], 'sn': ['sn2006', 'sn2007'], \
    'zn': ['zn2005', 'zn2006'], 'pb': ['pb2005', 'pb2006'],  'sp': ['sp2005', 'sp2009'],\
    'bu': ['bu2006', 'bu2012'], 'fu': ['fu2005', 'fu2009'], 'eg': ['eg2005', 'eg2009'], \
    'ag': ['ag2006', 'ag2012'], 'au': ['au2006', 'au2012'], 'b': ['b2005', 'b2009'],    \
    'T': ['T2006', 'T2009'],    'TF': ['TF2006', 'TF2009'], 'TS': ['TS2006', 'TS2009'], \
    }

def create_openbrk_strat(label = 'pt', capital = 6000.0):
    asset_keys = ['alloc_w', 'vol_ratio', 'split_mode', 'lookbacks', 'ratios', 'trend_factor', \
                  'channels', 'channel_type', 'ma_chan', 'volumes', 'price_mode', 'close_tday', 'min_rng', 'ma_width']
    common_keys = []
    df = pd.read_excel(open('C:\\dev\\hist_bktest\\select_strat_port_200228.xlsx', 'rb'), sheetname = 'openbrk' + '_' + label)
    df.columns = [str(col) for col in df.columns]
    df.rename(columns = {'freq': 'split_mode', 'close_daily': 'close_tday'}, inplace = True)
    shared_dict = {"data_func": [["np.max", "high", {}], ["np.min", "low", {}], \
                           ["np.percentile", "high", {"q": 90}], ["np.percentile", "low", {"q": 10}], \
                           ["np.percentile", "high", {"q": 75}], ["np.percentile", "low", {"q": 25}], \
                           ["np.percentile", "high", {"q": 55}], ["np.percentile", "low", {"q": 45}]], }
    output = create_strat_json(df, prod_inst_map, asset_keys, common_keys, shared_dict, capital = capital, strat_class = "strat_openbrk.OpenBrkChan")
    for key in output:
        with open("C:\\dev\\hist_bktest\\" + label.upper() + '_' + key + ".json", 'w') as outfile:
            json.dump(output[key], outfile)

def create_chanbrk_strat(label = 'pt', capital = 6000.0):
    asset_keys = ['alloc_w', 'split_mode', 'atr_win', 'stoploss', 'entry_chan', 'exit_chan', 'volumes']
    common_keys = []
    df = pd.read_excel(open('C:\\dev\\hist_bktest\\select_strat_port_200228.xlsx', 'rb'), sheetname = 'chanbrk' +  '_' + label)
    df.rename(columns = {'freq': 'split_mode'}, inplace = True)
    output = create_strat_json(df, prod_inst_map, asset_keys, common_keys, shared_dict = {}, capital = capital, strat_class = "strat_chan_break.ChanBreak")
    for key in output:
        with open("C:\\dev\\hist_bktest\\" + label.upper() + '_' + key + ".json", 'w') as outfile:
            json.dump(output[key], outfile)

def create_RSIATR_strat(label = 'pt', capital = 6000.0):
    asset_keys = ['alloc_w', 'rsi_th', 'rsi_win', 'atr_win', 'atrma_win', 'stoploss',\
                  'close_tday', 'volumes', 'freq']
    common_keys = []
    df = pd.read_excel(open('C:\\dev\\hist_bktest\\select_strat_port_200228.xlsx', 'rb'), sheetname = 'rsiatr' +  '_' + label)
    df['freq'] = 'm' + df['freq'].astype(str)
    shared_dict = {"data_func": [["RSI", "dh.RSI_F", "dh.rsi_f", {"field": "close"}], ["ATR", "dh.ATR", "dh.atr"],
                  ["ATRMA", "dh.MA", "dh.ma"]], }
    output = create_strat_json(df, prod_inst_map, asset_keys, common_keys, shared_dict, capital = capital, strat_class = "strat_rsiatr.RsiAtrStrat")
    for key in output:
        with open("C:\\dev\\hist_bktest\\" + label.upper() + '_' + key + ".json", 'w') as outfile:
            json.dump(output[key], outfile)

def create_MAChan_strat(label = 'pt', capital = 6000.0):
    asset_keys = ['alloc_w', 'freq', 'ma_win', 'channels', 'close_tday', 'volumes']
    common_keys = []
    df = pd.read_excel(open('C:\\dev\\hist_bktest\\select_strat_port_200228.xlsx', 'rb'), sheetname = 'ma2cross' +  '_' + label )
    flag = df['freq'].str.contains("m")
    df.ix[flag, 'freq'] = df.ix[flag, 'freq'].apply(lambda x: "m" + x[:-1])
    shared_dict = {"data_func": [["MA_C", "dh.MA", "dh.ma"], ["DONCH_HH", "dh.DONCH_H", "dh.donch_h", {"field": "high"}], ["DONCH_LL", "dh.DONCH_L", "dh.donch_l", {"field": "low"}]],
                   "channel_keys": ["DONCH_HH", "DONCH_LL"],}
    output = create_strat_json(df, prod_inst_map, asset_keys, common_keys, shared_dict, capital = capital, strat_class = "strat_ma_system.MASystemTrader")
    for key in output:
        with open("C:\\dev\\hist_bktest\\" + label.upper() + '_' + key + ".json", 'w') as outfile:
            json.dump(output[key], outfile)

def create_XSMOM_strat(label = 'pt', capital = 1600000.0):
    asset_keys = ['alloc_w', 'volumes']
    common_keys = []
    batch_keys = ["freq", "lookback", "rebal_freq", 'quantile', 'ma_win']
    df = pd.read_excel(open('C:\\dev\\hist_bktest\\select_strat_port_200228.xlsx', 'rb'), sheetname = 'xsmom' +  '_' + label )
    flag = df['freq'].str.contains("m")
    df.ix[flag, 'freq'] = df.ix[flag, 'freq'].apply(lambda x: "m" + x[:-1])
    shared_dict = {}
    output = create_xs_strat_json(df, prod_inst_map, asset_keys, common_keys, batch_keys, shared_dict, capital = capital, strat_class = "strat_xsmom.XSMOMRetTrader")
    for key in output:
        with open("C:\\dev\\hist_bktest\\" + label.upper() + '_' + key + ".json", 'w') as outfile:
            json.dump(output[key], outfile)