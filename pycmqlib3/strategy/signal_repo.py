import pandas as pd
from pycmqlib3.analytics.tstool import *
from pycmqlib3.utility.misc import CHN_Holidays, day_shift


signal_store = {
    'io_removal_lvl': ('io_removal_41ports', 'qtl', [20, 40, 2], '', 'diff', True, 'price'),
    'io_removal_lyoy': ('io_removal_41ports', 'qtl', [6, 10], 'lunar_yoy_day', 'diff', True, 'W-Fri'),
    'io_removal_wow': ('io_removal_41ports', 'zscore', [48, 53], 'df1', 'diff', True, 'W-Fri'),
    'io_millinv_lvl': ('io_inv_mill(64)', 'qtl', [20, 40, 2], '', 'diff', True, 'price'),
    'io_millinv_lyoy': ('io_inv_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'diff', True, 'W-Fri'),
    'io_invdays_lvl': ('io_invdays_imp_mill(64)', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price'),
    'io_invdays_lyoy': ('io_invdays_imp_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'pct_change', True, 'W-Fri'),
    'io_port_inv_lvl_slow': ('io_inv_imp_31ports_w', 'zscore', [240, 255, 5], '', 'pct_change', False, 'price'),
    'io_pinv31_lvl_zsa': ('io_inv_31ports', 'zscore_adj', [24, 36, 2], '', 'pct_change', True, ''),
    'io_pinv45_lvl_hlr': ('io_inv_45ports', 'hlratio', [24, 36, 2], '', 'pct_change', True, ''),
    'steel_sinv_lyoy_zs': ('steel_inv_social', 'zscore', [24, 30, 2], 'lunar_yoy_day', 'diff', False, ''),
    'steel_sinv_lyoy_mds': ('steel_inv_social', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, ''),

    'rbhc_dmd_mds': ('rb_hc_dmd_diff', 'ma_dff_sgn', [5, 9, 1], '', 'diff', True, ''),
    'rbhc_dmd_lyoy_mds': ('rb_hc_dmd_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', True, ''),
    'rbhc_sinv_mds': ('rb_hc_sinv_diff', 'ma_dff_sgn', [5, 9, 1], '', 'diff', False, ''),
    'rbhc_sinv_lyoy_mds': ('rb_hc_sinv_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, ''),

    'rb_sinv_lyoy_fast': ('rebar_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    'wr_sinv_lyoy_fast': ('wirerod_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    'hc_soinv_lyoy_fast': ('hrc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    'cc_soinv_lyoy_fast': ('crc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    'billet_inv_chg_slow': ('billet_inv_social_ts', 'zscore', [240, 252, 2], '', 'diff', False, 'price'),
    'pbf_prem_yoy': ('pbf_prem', 'zscore', [20, 42, 2], 'df250', 'diff', True),
    # 'pbf_prem_lyoy_mom': ('pbf_prem', 'qtl', [12, 20, 2], 'lunar_yoy_wk', 'diff', True),
    'cons_steel_lyoy_slow': ('cons_steel_transact_vol_china', 'zscore', [240, 255, 5],
                             'lunar_yoy_day', 'diff', True, 'price'),
    # 'margin_sea_lvl_mid': ('hrc_margin_sb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),
    'sea_export_arb_lvl_mid': ('hrc_exp_sea_arb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),
    'steel_margin_lvl_fast': ('margin_hrc_macf', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price'),
    'strip_hsec_lvl_mid': ('strip_hsec', 'qtl', [60, 80, 2], '', 'pct_change', True, 'price'),
    'macf_cfd_lvl_mid': ('macf_cfd', 'qtl', [40, 82, 2], '', 'pct_change', True, 'price'),
    'hc_rb_diff_lvl_fast': ('hc_rb_diff', 'zscore', [20, 40, 2], '', '', True, 'price'),

    'fef_c1_c2_ratio_or_qtl': ('FEF_c1_c2_ratio', 'qtl', [30, 60, 2], '', '', False, ''),
    'fef_c1_c2_ratio_spd_qtl': ('FEF_c1_c2_ratio', 'qtl', [30, 60, 2], '', '', False, ''),

    'cu_prem_usd_zsa': ('cu_prem_yangshan_warrant', 'zscore_adj', [20, 30, 2], '', '', True, 'price'),
    'cu_prem_usd_md': ('cu_prem_yangshan_warrant', 'ma_dff', [20, 30, 2], '', '', True, 'price'),
    'cu_phybasis_zsa': ('cu_cjb_phybasis', 'zscore_adj', [40, 60, 2], 'sma10', 'pct_change', True, 'price'),  # great
    'cu_phybasis_hlr': ('cu_cjb_phybasis', 'hlratio', [40, 60, 2], 'sma10', 'pct_change', True, 'price'),  # great

    'lme_base_ts_mds': ('lme_base_ts', 'ma_dff_sgn', [10, 30, 2], '', '', True, 'price'),
    'lme_base_ts_mds_xdemean': ('lme_base_ts', 'ma_dff_sgn', [10, 30, 4], '', '', True, 'price'),
    'lme_base_ts_hlr': ('lme_base_ts', 'hlratio', [10, 20, 2], '', '', True, 'price'),
    'lme_base_ts_hlr_xdemean': ('lme_base_ts', 'hlratio', [10, 20, 2], '', '', True, 'price'),
    'base_phybas_carry_ma': ('base_phybas_carry', 'ma', [1, 2], '', '', True, 'price'),
    'base_phybas_carry_ma_xdemean': ('base_phybas_carry', 'ma', [1, 2], '', '', True, 'price'),
    'base_inv_mds': ('base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price'),
    'base_inv_mds_xdemean': ('base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price'),

    # too short
    'cu_scrap1_margin_gd': ('cu_scrap1_diff_gd', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price'),  # too short
    'cu_scrap1_margin_tj': ('cu_scrap1_diff_tj', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price'),  # too short
    'cu_rod_procfee_2.6': ('cu_rod_2.6_procfee_nanchu', 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price'),
    'cu_rod_procfee_8.0': ('cu_rod_8_procfee_nanchu', 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price'),

    'r007_qtl': ('r007_cn', 'qtl', [80, 120, 2], 'ema5', 'pct_change', True, 'price'),
    'r_dr_spd_zs': ('r_dr_7d_spd', 'zscore', [20, 40, 2], 'ema5', 'pct_change', True, 'price'),
    'shibor1m_qtl': ('shibor_1m', 'qtl', [40, 80, 2], 'ema3', 'pct_change', True, 'price'),

    'cnh_mds': ('usdcnh_spot', 'ma_dff_sgn', [10, 30, 2], 'ema3', 'pct_change', False, 'price'),
    'cnh_cny_zsa': ('cnh_cny_spd', 'zscore_adj', [10, 20, 2], 'ema10', 'pct_change', False, 'price'),
    'cnyrr25_zsa': ('usdcny_rr25', 'zscore_adj', [10, 20, 2], 'ema10', 'pct_change', False, 'price'),

    'vhsi_mds': ('vhsi', 'ma_dff_sgn', [10, 20, 2], '', 'pct_change', False, 'price'),
    'vhsi_qtl': ('vhsi', 'qtl', [10, 30, 2], '', 'pct_change', False, 'price'),
    'sse50iv_mds': ('sse50_etf_iv', 'ma_dff_sgn', [20, 30, 2], '', 'pct_change', False, 'price'),
    'sse50iv_qtl': ('sse50_etf_iv', 'qtl', [20, 40, 2], '', 'pct_change', False, 'price'),
    'eqmargin_zsa': ('eq_margin_outstanding_cn', 'zscore_adj', [10, 20, 2], '', 'pct_change', False, 'price'),
    'eqmargin_zs': ('eq_margin_outstanding_cn', 'zscore_adj', [10, 20, 2], '', 'pct_change', False, 'price'),

    '10ybe_mds': ('usggbe10', 'ma_dff_sgn', [10, 30, 2], '', 'pct_change', True, 'price'),
    '10ybe_zsa': ('usggbe10', 'zscore_adj', [20, 40, 2], '', 'pct_change', True, 'price'),
    '10y_2y_mds': ('usgg10yr_2yr_spd', 'ma_dff_sgn', [20, 30, 2], '', 'pct_change', True, 'price'),

    'dxy_qtl_s': ('dxy', 'qtl', [40, 60, 2], '', 'pct_change', False, 'price'),
    'dxy_qtl_l': ('dxy', 'qtl', [480, 520, 2], '', 'pct_change', False, 'price'),

    'vix_mds': ('vix', 'ma_dff_sgn', [20, 40, 2], '', 'pct_change', False, 'price'),
    'vix_zsa': ('vix', 'zscore_adj', [40, 60, 2], 'ema3', 'pct_change', False, 'price'),

}

feature_to_feature_key_mapping = {
    'lme_base_ts': {
        'cu': 'cu_lme_3m_15m_spd',
        'al': 'al_lme_3m_15m_spd',
        'zn': 'zn_lme_3m_15m_spd',
        'ni': 'ni_lme_0m_3m_spd',
        'sn': 'sn_lme_0m_3m_spd',
        'pb': 'pb_lme_0m_3m_spd',
    },
    'base_phybas_carry': {
        'cu': 'cu_smm_phybasis',
        'al': 'al_smm0_phybasis',
        'zn': 'zn_smm1_sh_phybasis',
        'pb': 'pb_smm1_sh_phybasis',
        'ni': 'ni_smm1_jc_phybasis',
        'sn': 'sn_smm1_sh_phybasis',
    },
    'base_inv': {
        'cu': 'cu_inv_social_all',
        'al': 'al_inv_social_all',
        'zn': 'zn_inv_social_3p',
        'ni': 'ni_inv_social_6p',
        'pb': 'pb_inv_social_5p',
        'sn': 'sn_inv_social_all',
        'si': 'si_inv_social_all',
        'ao': 'bauxite_inv_az_ports',
    }
}

param_rng_by_feature_key = {
    'base_phybas_carry': {
        'cu': [10, 20],
        'al': [10, 20]
    }
}

leadlag_port_d = {
    # 'ferrous': {'lead': ['hc', 'rb', ],
    #             'lag': [],
    #             'param_rng': [40, 80, 2],
    #             },
    'constrs': {'lead': ['hc', 'rb', 'v'],
                'lag': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SA', 'v', 'UR', 'SM', 'SF'],
                'param_rng': [40, 80, 2],
                },
    'petchem': {'lead': ['v'],
                'lag': ['TA', 'MA', 'pp', 'eg', 'eb', 'PF', ],
                'param_rng': [40, 80, 2],
                },
    'base': {'lead': ['al'],
             'lag': ['al', 'ni', 'sn', 'ss', ],  # 'zn', 'cu'
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
    leadlag_port = leadlag_port_d
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


def long_break(df, input_args):
    product_list = input_args['product_list']
    gaps = input_args.get('gaps', 7)
    days = input_args.get('days', 2)
    signal_df = pd.DataFrame(index=df.index, columns=product_list)
    signal_ts = df.index.map(lambda x:
                             1 if ((day_shift(x.date(), f'{days}b', CHN_Holidays) - x.date()).days >= gaps) or
                                  ((x.date() - day_shift(x.date(), f'-{days}b', CHN_Holidays)).days >= gaps)
                             else 0)
    signal_ts = pd.Series(signal_ts, index=df.index)
    for asset in product_list:
        signal_df[asset] = signal_ts
    return signal_df


def funda_signal_by_name(spot_df, signal_name, price_df=None,
                         signal_cap=None, asset=None,
                         signal_repo=signal_store, feature_key_map=feature_to_feature_key_mapping):
    feature, signal_func, param_rng, proc_func, chg_func, bullish, freq = signal_repo[signal_name]
    vol_win = 120
    post_func = ''
    if asset and feature in feature_key_map:
        new_feature = feature_key_map[feature].get(asset, feature)
        if feature in param_rng_by_feature_key:
            param_rng = param_rng_by_feature_key[feature].get(asset, param_rng)
        feature = new_feature
    feature_ts = spot_df[feature].dropna()
    cdates = pd.date_range(start=feature_ts.index[0], end=feature_ts.index[-1], freq='D')
    bdates = pd.bdate_range(start=feature_ts.index[0], end=feature_ts.index[-1], freq='C', holidays=CHN_Holidays)
    if freq == 'price':
        feature_ts = spot_df[feature].reindex(index=cdates).ffill().reindex(index=bdates)
    elif len(freq) > 0:
        feature_ts = spot_df[feature].reindex(index=cdates).ffill().reindex(
            index=pd.date_range(start=feature_ts.index[0], end=feature_ts.index[-1], freq=freq))

    if 'yoy' in proc_func:
        if 'lunar' in proc_func:
            label_func = lunar_label
            label_args = {}
        else:
            label_func = calendar_label
            label_args = {'anchor_date': {'month': 1, 'day': 1}}
        if '_wk' in proc_func:
            group_col = 'label_wk'
        else:
            group_col = 'label_day'
        feature_ts = yoy_generic(feature_ts, label_func=label_func, group_col=group_col, func=chg_func,
                                 label_args=label_args)
    elif 'df' in proc_func:
        n_diff = int(proc_func[2:])
        feature_ts = getattr(feature_ts, chg_func)(n_diff)
    elif 'flr' in proc_func:
        feature_ts = feature_ts.apply(lambda x: max(x-param_rng[0], 0) / param_rng[1])
    elif 'sma' in proc_func:
        n_days = int(proc_func[3:])
        feature_ts = feature_ts.rolling(n_days).mean()
    elif 'ema' in proc_func:
        n_days = int(proc_func[3:])
        feature_ts = feature_ts.ewm(n_days).mean()
    elif '_lr' in proc_func:
        feature_ts = np.log(1+feature_ts)

    if signal_func == 'seasonal_score_w':
        signal_ts = seasonal_score(feature_ts.to_frame(),
                                   backward=10,
                                   forward=10,
                                   rolling_years=3,
                                   min_obs=10).reindex(index=bdates).ffill()
    elif signal_func == 'seasonal_score_d':
        signal_ts = seasonal_score(feature_ts.to_frame(), backward=15, forward=15, rolling_years=3, min_obs=30)
    elif len(signal_func) > 0:
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng,
                                     signal_cap=signal_cap, vol_win=vol_win)
    else:
        signal_ts = feature_ts
    if not bullish:
        signal_ts = -signal_ts
    # signal_ts = signal_ts.reindex(index=pd.bdate_range(
    #     start=spot_df.index[0], end=spot_df.index[-1], freq='C', holidays=CHN_Holidays)).ffill().dropna()
    if 'ema' in post_func:
        n_win = int(post_func[3])
        signal_ts = signal_ts.ewm(n_win, ignore_na=True).mean()
    elif 'sma' in post_func:
        n_win = int(post_func[3:])
        signal_ts = signal_ts.rolling(n_win).mean()
    elif 'hmp' in post_func:
        hump_lvl = float(post_func[3:])
        signal_ts = signal_hump(signal_ts, hump_lvl)
    return signal_ts


def custom_funda_signal(df, input_args):
    product_list = input_args['product_list']
    signal_cap = input_args.get('signal_cap', None)
    funda_df = input_args['funda_data']
    signal_name = input_args['signal_name']
    signal_type = input_args.get('signal_type', 1)
    vol_win = input_args.get('vol_win', 20)

    # get signal by asset
    if signal_type == 0:
        signal_df = pd.DataFrame()
        for asset in product_list:
            signal_ts = funda_signal_by_name(funda_df, signal_name, price_df=df, signal_cap=signal_cap, asset=asset)
            signal_ts = signal_ts.reindex(
                index=pd.date_range(start=df.index[0],
                                    end=df.index[-1],
                                    freq='C'
                                    )).ffill().reindex(index=df.index)
            signal_df[asset] = signal_ts

    # pair trading strategy, fixed ratio
    elif signal_type == 3:
        signal_df = pd.DataFrame()
        signal_ts = funda_signal_by_name(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
        signal_ts = signal_ts.reindex(
            index=pd.date_range(start=df.index[0],
                                end=df.index[-1],
                                freq='C'
                                )).ffill().reindex(index=df.index)
        if set(product_list) == set(['rb', 'hc']):
            signal_df['rb'] = signal_ts
            signal_df['hc'] = -signal_ts

    # beta neutral last asseet is index asset
    elif signal_type == 4:
        signal_ts = funda_signal_by_name(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
        signal_df = pd.DataFrame(0, index=signal_ts.index, columns=product_list)
        index_asset = product_list[-1]
        beta_win = 122
        for trade_asset in product_list[:-1]:
            asset_df = df[[(index_asset, 'c1', 'close'),
                           (trade_asset, 'c1', 'close')]].copy(deep=True).droplevel([1, 2], axis=1)
            asset_df = asset_df.dropna(subset=[trade_asset]).ffill()
            for asset in asset_df:
                asset_df[f'{asset}_pct'] = asset_df[asset].pct_change().fillna(0)
                asset_df[f'{asset}_pct_ma'] = asset_df[f'{asset}_pct'].rolling(5).mean()
                asset_df[f'{asset}_vol'] = asset_df[f'{asset}_pct'].rolling(vol_win).std()
            asset_df['beta'] = asset_df[f'{index_asset}_pct_ma'].rolling(beta_win).cov(
                asset_df[f'{trade_asset}_pct_ma']) / asset_df[f'{index_asset}_pct_ma'].rolling(beta_win).var()
            asset_df['signal'] = signal_ts
            asset_df = asset_df.ffill()
            asset_df['pct'] = asset_df[f'{trade_asset}_pct'] - asset_df['beta'] * asset_df[f'{index_asset}_pct']
            asset_df['vol'] = asset_df['pct'].rolling(vol_win).std()
            asset_df = asset_df.reindex(index=signal_ts.index).ffill()
            signal_df[trade_asset] += signal_ts * asset_df[f'{trade_asset}_vol']/asset_df['vol']
            signal_df[index_asset] -= signal_ts * asset_df['beta'] * asset_df[f'{index_asset}_vol'] / asset_df['vol']

    # apply same signal to all assets
    else:
        signal_ts = funda_signal_by_name(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
        signal_ts = signal_ts.reindex(
            index=pd.date_range(start=df.index[0],
                                end=df.index[-1],
                                freq='C'
                                )).ffill().reindex(index=df.index)
        signal_df = pd.DataFrame(dict([(asset, signal_ts) for asset in product_list]))

    signal_df = signal_df.shift(1)
    return signal_df
