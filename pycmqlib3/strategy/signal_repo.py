import pandas as pd
from pycmqlib3.analytics.tstool import *


signal_store = {
    'io_removal_lvl': ('io_removal_41ports', 'qtl', [20, 40, 2], '', 'diff', True, 'price'),
    'io_removal_lyoy': ('io_removal_41ports', 'qtl', [8, 12], 'lunar_yoy_day', 'diff', True, 'W-Fri'),
    'io_removal_wow': ('io_removal_41ports', 'zscore', [48, 53], 'df1', 'diff', True, 'W-Fri'),
    'io_millinv_lvl': ('io_inv_mill(64)', 'qtl', [20, 40, 2], '', 'diff', True, 'price'),
    'io_millinv_lyoy': ('io_inv_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'diff', True, 'W-Fri'),
    'io_invdays_lvl': ('io_invdays_imp_mill(64)', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price'),
    'io_invdays_lyoy': ('io_invdays_imp_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'pct_change', True, 'W-Fri'),

    'io_port_inv_lvl_slow': ('io_inv_imp_31ports_w', 'zscore', [240, 255, 5], '', 'pct_change', False, 'price'),

    'steel_major5_inv_lvl_fast': ('steel_major5_inv', 'qtl', [20, 32, 4], '', 'diff', False, 'W-Fri'),
    'steel_social_inv_lvl_fast': ('steel_inv_social', 'zscore', [20, 32, 4], '', 'diff', False, 'W-Fri'),
    'rebar_inv_social_lyoy_fast': ('rebar_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_wk', 'diff', False, 'W-Fri'),
    'wirerod_inv_social_lyoy_fast': (
    'wirerod_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_w', 'diff', False, 'W-Fri'),
    'hrc_inv_social_lyoy_fast': ('hrc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_wk', 'diff', False, 'W-Fri'),
    'crc_inv_social_lyoy_fast': ('crc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_wk', 'diff', False, 'W-Fri'),

    'billet_inv_chg_slow': ('billet_inv_social_ts', 'zscore', [240, 252, 2], '', 'diff', False, 'price'),

    'pbf_prem_yoy': ('pbf_prem', 'zscore', [20, 42, 2], 'df250', 'diff', True),
    # 'pbf_prem_lyoy_mom': ('pbf_prem', 'qtl', [12, 20, 2], 'lunar_yoy_wk', 'diff', True),

    'cons_steel_lyoy_slow': (
    'cons_steel_transact_vol_china', 'zscore', [240, 255, 5], 'lunar_yoy_day', 'diff', True, 'price'),

    # 'margin_sea_lvl_mid': ('hrc_margin_sb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),
    'sea_export_arb_lvl_mid': ('hrc_exp_sea_arb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),

    'steel_margin_lvl_fast': ('margin_hrc_macf', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price'),
    'strip_hsec_lvl_mid': ('strip_hsec', 'qtl', [60, 80, 2], '', 'pct_change', True, 'price'),
    'macf_cfd_lvl_mid': ('macf_cfd', 'qtl', [40, 82, 2], '', 'pct_change', True, 'price'),

    'hc_rb_diff_lvl_fast': ('hc_rb_diff', 'zscore', [20, 40, 2], '', '', True, 'price'),

}


def hc_rb_diff(df, input_args):
    shift_mode = input_args['shift_mode']
    product_list = input_args['product_list']
    win = input_args['win']
    xdf = df.loc[:, df.columns.get_level_values(0).isin(['rb', 'hc'])].copy(deep=True)
    xdf = xdf['2014-07-01':]
    if shift_mode == 2:
        xdf[('rb', 'c1', 'px_chg')] = np.log(xdf[('rb', 'c1', 'close')]).diff()
        xdf[('hc', 'c1', 'px_chg')] = np.log(xdf[('hc', 'c1', 'close')]).diff()
    else:
        xdf[('rb', 'c1', 'px_chg')] = xdf[('rb', 'c1', 'close')].diff()
        xdf[('hc', 'c1', 'px_chg')] = xdf[('hc', 'c1', 'close')].diff()
    hc_rb_diff = xdf[('hc', 'c1', 'px_chg')] - xdf[('rb', 'c1', 'px_chg')]
    signal_ts = hc_rb_diff.ewm(span=win).mean() / hc_rb_diff.ewm(span=win).std()
    signal_df = pd.concat([signal_ts] * len(product_list), axis=1)
    signal_df.columns = product_list
    return signal_df


def leader_lagger(df, input_args):
    leadlag_port = {
        'ferrous': {'lead': ['hc', 'rb', ],
                    'lag': ['rb', 'hc', 'i', 'j', 'jm', 'SM', ],
                    'param_rng': [40, 60, 2],
                    },
        'constrs': {'lead': ['hc', 'rb', 'v'],
                    'lag': ['FG', 'SA', 'v', 'UR', ],
                    'param_rng': [40, 60, 2],
                    },
        'petchem': {'lead': ['v'],
                    'lag': ['TA', 'MA', 'pp', 'eg', 'eb', 'PF', ],
                    'param_rng': [40, 60, 2],
                    },
        'base': {'lead': ['al'],
                 'lag': ['al', 'ni', 'sn', 'ss', ],  # 'zn', 'cu'
                 'param_rng': [40, 60, 2],
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
    product_list = input_args['product_list']
    signal_cap = input_args.get('signal_cap', None)
    conv_func = input_args.get('conv_func', 'qtl')
    signal_df = pd.DataFrame(index=df.index, columns=product_list)
    for asset in product_list:
        for sector in leadlag_port:
            if asset in leadlag_port[sector]['lag']:
                signal_list = []
                for lead_prod in leadlag_port[sector]['lead']:
                    feature_ts = df[(lead_prod, 'c1', 'close')]
                    signal_ts = calc_conv_signal(feature_ts.dropna(), conv_func,
                                                 leadlag_port[sector]['param_rng'], signal_cap=signal_cap)
                    signal_list.append(signal_ts)
                signal_df[asset] = pd.concat(signal_list, axis=1).mean(axis=1)
                break
            else:
                signal_df[asset] = 0
    return signal_df
