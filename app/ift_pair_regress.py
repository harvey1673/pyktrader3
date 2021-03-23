import sys
sys.path.append("C:/dev/pycmqlib3/")
sys.path.append("C:/dev/pycmqlib3/scripts/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import datetime
import ts_tool
import misc
import email_tool
plt.rcParams.update({'figure.max_open_warning': 0})

ift_product_pairs = [[['plt_billet_sea', 'tsi_scrap', 'rebar_hz*USDCNY'], 'W'],
                [['plt_billet_sea', 'tsi_scrap', 'rb$1$35b*USDCNY'], 'W'],
                [['plt_billet_sea', 'plt_scrap_cfr_tw', 'rebar_hz*USDCNY'], 'W'],
                [['plt_billet_sea', 'plt_scrap_cfr_tw', 'rb$1$35b*USDCNY'], 'W'],
                [['plt_rebar_sea', 'tsi_scrap', 'rb$1$35b*USDCNY'], 'D'], \
                [['bs_billet', 'tsi_scrap', 'rb$1$35b*USDCNY'], 'D'], \
                [['bs_billet', 'tsi_scrap'], 'D'],\
                [['plt_billet_sea', 'tsi_scrap'], 'W'],\
                [['plt_billet_sea', 'plt_scrap_cfr_tw'], 'W'],\
                [['plt_billet_sea', 'rebar_hz*USDCNY'], 'W'],
                [['plt_billet_sea', 'rb$1$35b*USDCNY'], 'W'],
                [['plt_billet_sea', 'billet_ts*USDCNY'], 'W'],
                [['plt_rebar_sea', 'rb$1$35b*USDCNY'], 'D'], \
                [['plt_rebar_sea', 'rebar_hz*USDCNY'], 'D'], \
                [['plt_rebar_sea', 'billet_ts*USDCNY'], 'D'],\
                [['tsi_scrap', 'rebar_hz*USDCNY'], 'D'], \
                [['tsi_scrap', 'rb$1$35b*USDCNY'], 'D'],\
                [['tsi_scrap', 'billet_ts*USDCNY'], 'D'],\
                [['plt_scrap_cfr_tw', 'tsi_scrap'], 'W'],\
                [['plt_hrc_sae_sea', 'hc$1$35b*USDCNY'], 'D'],\
                [['plt_hrc_sae_sea', 'hrc_4_75mm_sh*USDCNY'], 'D'],\
                [['plt_hrc_sae_sea', 'tsi_scrap', 'hc$1$35b*USDCNY'], 'D'],\
                [['plt_hrc_sae_cn', 'hc$1$35b*USDCNY'], 'D'],\
                [['tsi_hrc_seu*USDEUR', 'plt_hrc_sae_cn', 'tsi_scrap'], 'D'],\
                [['tsi_hrc_neu', 'tsi_hrc_seu'], 'D'],\
                [['billet_ts', 'rebar_hz'], 'D'],\
                [['billet_ts', 'rb$1$35b'], 'D'],\
                [['billet_sth_sh', 'rb$1$35b'], 'D'],\
                [['hrc_4_75mm_sh', 'rebar_hz'], 'D'],\
                [['billet_Q235_js', 'billet_ts'], 'D'],\
                ]

Underlying_Name_Map = {
    'rebar_hz': 'Hangzhou Rebar (CNY)',
    'rebar_hz_USDCNY': 'Hangzhou Rebar (USD)',
    'hrc_4_75mm_sh': 'Shanghai HRC (CNY)',
    'hrc_4_75mm_sh_USDCNY': 'Shanghai HRC (USD)',
    'rb_1_35b': 'SHFE Rebar (CNY)',
    'rb_1_35b_USDCNY': 'SHFE Rebar (USD)',
    'hc_1_35b': 'SHFE HRC (CNY)',
    'hc_1_35b_USDCNY': 'SHFE HRC (USD)',
    'billet_ts': 'Tangshan billet (CNY)',
    'billet_Q235_js': 'Jiangsu billet (CNY)',
    'billet_ts_USDCNY': 'Tangshan billet (USD)',
    'billet_sth_sh': 'Shanghai Billet (CNY)',
    'tsi_hrc_seu_USDEUR': 'S EU HRC (USD)',
    'tsi_hrc_seu': 'S EU HRC (EUR)',
    'tsi_hrc_neu_USDEUR': 'N EU HRC (USD)',
    'tsi_hrc_neu': 'N EU HRC (EUR)',
    'plt_billet_sea': 'SEA billet',
    'plt_rebar_sea': 'SEA rebar',
    'tsi_scrap': 'LME Scrap',
    'plt_scrap_cfr_tw': 'TW Scrap',
    'bs_billet': 'Blacksea billet',
    'plt_hrc_sae_sea': 'SEA HRC',
    'plt_hrc_sae_cn': 'HRC China FOB TJ'
}

def parse_id(inst_key):
    name = inst_key.replace('$', '_').replace('*', '_').replace('^', '')
    inst_name = inst_key
    fx_pair = None
    if '*' in inst_name:
        sp = inst_key.split('*')
        inst_name = sp[0]
        fx_pair = sp[1][:3] + '/' + sp[1][3:]
    if "^" in inst_name:
        vat_adj = True
        temp = inst_name.split("^")
        inst_name = temp[0]
    else:
        vat_adj = False
    if '$' in inst_name:
        sp = inst_name.split('$')
        inst_name = sp[0]
        if len(sp) > 2:
            #name = sp[0] + '_' + sp[1]
            args = ['$'] + sp[1:]
        else:
            #name = sp[0]
            args = ['$']
    else:
        args = []
    return (inst_name, name, fx_pair, vat_adj, args)
            
def load_data(inst_key, start, end):
    inst_name, name, fx_pair, vat_adj, args = parse_id(inst_key)
    kargs = {}
    kargs['name'] = name
    if len(args) == 1:
        kargs['spot_table'] = 'fut_daily'
        kargs['field'] = 'instID'
    elif len(args) > 1:
        kargs['field'] = 'instID'
        kargs['args'] = {'n': int(args[1]),  'roll_rule': '-' + args[2], 'freq': 'd', 'need_shift': 0}
    if fx_pair:
        kargs['fx_pair'] = fx_pair
    df = ts_tool.get_data(inst_name, start, end, **kargs)
    if vat_adj:
        df = ts_tool.apply_vat(df)
    return df

def run_regress(product_pairs, start_date, end_date, save_plot = False):
    spot_data = {}
    fig_names = []
    out_columns = ['Underlier', 'last', 'proxy', 'Z-val', 'std', 'IndVar1', 'K1', 'X1', 'IndVar2', 'K2', 'X2', 'const', 'ldate']
    out_dict = {}
    key = 0
    for pair, freq in product_pairs:
        res = {}
        for spot in pair:
            if spot not in spot_data:
                spot_data[spot] = load_data(spot, start_date, end_date)
        data = ts_tool.merge_df([spot_data[spot] for spot in pair])
        df = data.dropna()
        df.index = pd.to_datetime(df.index)
        res['ldate'] = df.index[-1].date()
        res['last'] = df[df.columns[0]][-1]
        if freq != 'D':
            df = df.resample(freq).last().dropna()
        formula = '{} ~ '.format(df.columns[0])
        res['Underlier'] = df.columns[0]
        for idx, element in enumerate(df.columns[1:]):
            if idx == 0:
                formula += element
            else:
                formula += ' + {}'.format(element)
            res['IndVar%s' % (str(idx+1))] = element
            res['X%s' % (str(idx+1))] = data[element].dropna()[-1]
        if len(df.columns[1:]) < 2:
            res['IndVar2'] = None
            res['K2'] = 0.0
            res['X2'] = 0.0
        result = smf.ols(formula, df).fit()
        params = result.params
        resid_df = result.resid
        res['const'] = params[0]
        proxy = res['const']
        for idx, col in enumerate(df.columns[1:]):
            res['K%s' % (str(idx+1))] = params[idx+1]
            proxy += res['K%s' % (str(idx+1))] * res['X%s' % (str(idx+1))]
        resid_stats = resid_df.describe()
        res['std'] = resid_stats['std']
        res['proxy'] = proxy
        res['Z-val'] = (res['last'] - proxy)/res['std']
        out_dict[key] = res
        key += 1
        if save_plot:
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, result.resid, label='residual')
            plt.title('residual plot')
            num = len(df.index)
            for std_line in [1, 2]:
                plt.plot(df.index, std_line * res['std'] * np.ones(num), 'r--')
                plt.plot(df.index, -std_line * res['std'] * np.ones(num), 'r--')
            plt.title('residual plot ({})'.format('-'.join([Underlying_Name_Map[col] for col in df.columns])))
            fig_name = "C:\\dev\\data\\%s.png" % ('-'.join(df.columns))
            plt.savefig(fig_name)
            fig_names.append([fig_name, '-'.join(df.columns)])

    out_df = pd.DataFrame.from_dict(out_dict, orient='index')
    out_df = out_df[out_columns].round(3)
    return (out_df, fig_names)

def send_result(product_pairs = None, recepient = '', tday = datetime.date.today(), lookback = '-3y'):
    end_date = tday
    start_date = misc.day_shift(tday, lookback)
    if product_pairs == None:
        product_pairs = ift_product_pairs
    out_df, attach_files = run_regress(product_pairs, start_date, end_date, save_plot=True)
    for col in ['Underlier', 'IndVar1', 'IndVar2']:
        out_df[col] = out_df[col].apply(lambda x: Underlying_Name_Map.get(x, x))
    out_df.rename(columns = {'Z-val': 'Degree of under/over-valuation', 'K1': 'HedgeRatio1', 'K2': 'HedgeRatio2', \
                             'X1': 'LastVar1', 'X2': 'LastVar2'}, inplace = True)
    if len(recepient) > 0:
        subject = "IFT relative value - %s" % str(tday)
        html_text = "<html><head>IFT relative value table</head><body><p>"
        html_text += "<br>Table<br>{0}<br><div style=""width: 1000px; height: 600px;"">".format(out_df.fillna('-').to_html())
        for afile in attach_files:
            html_text += "<br>{name}<br><img src=""cid:{id}"" width=""50%"" height=""50%""><br>".format(\
                name = '-'.join([ Underlying_Name_Map[name] for name in afile[1].split('-')]), id = afile[1] + '.png')
        html_text += "</div></p></body></html>"
        email_tool.send_email_by_outlook(recepient, subject, attach_files = attach_files, html_text = html_text)
    return out_df
