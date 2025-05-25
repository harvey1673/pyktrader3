import sys
sys.path.append("C:/dev/pyktrader3")
sys.path.append("C:/dev/wtpy")

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import logging
from pycmqlib3.utility import base
from pycmqlib3.strategy.signal_repo import get_funda_signal_from_store, BROAD_MKTS, IND_MKTS, AGS_MKTS, commod_phycarry_dict
from pycmqlib3.utility.spot_idx_map import index_map, process_spot_df
from pycmqlib3.utility.dbaccess import load_codes_from_edb, load_factor_data, load_int_stock_daily
from pycmqlib3.utility import dataseries
from pycmqlib3.utility.exch_ctd_func import io_ctd_basis
from pycmqlib3.utility.backtest import sim_start_dict
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, prod2exch, is_workday, \
    nearby, contract_expiry, inst2contmth
from pycmqlib3.analytics.tstool import *
from misc_scripts.factor_data_update import update_factor_db
from misc_scripts.seasonality_update import seasonal_cal_update


single_factors = {
    'hc_rb_diff_20': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'UR', 'v', 'ru', 'au', 'ag', 'cu', 'al', 'zn', 'sn', 'ss', 'ni', 'ao'],
    'steel_margin_lvl_fast': ['i', 'j', 'jm', 'SF', 'SM'],
    'steel_margin_lvl_slow': ['SF', 'SM'],
    'strip_hsec_lvl_mid': ['rb', 'hc', 'i', 'j', 'jm'],
    'io_removal_lvl': ['i'],
    'io_removal_lyoy': ['i'],
    'io_millinv_lyoy': ['hc', 'i'],
    'io_invdays_lvl': ['hc', 'i'],
    'io_invdays_lyoy': ['hc', 'i'],
    'io_inv_rmv_ratio_1y': ['i'],
    'ioarb_px_hlr': ['rb', 'hc', 'i'],
    'ioarb_px_hlrhys': ['rb', 'hc', 'i'],
    'steel_sinv_lyoy_zs': ['rb', 'hc', 'i', 'FG', 'v'],
    'steel_sinv_lyoy_mds': ['rb', 'hc', 'i', 'FG', 'v'],
    'rbsales_lyoy_mom_lt': ['rb'],
    'rb_sales_inv_ratio_lyoy': ['rb'],
    'fef_c1_c2_ratio_or_qtl': ['rb', 'hc', 'j'],
    'fef_fly_ratio_or_qtl': ['rb', 'hc', 'j'],
    'fef_basmom_or_qtl': ['rb', 'hc'],
    'fef_basmom5_or_qtl': ['rb', 'hc'],
    'mn_mine_mom': ['SM'],
    'prop2hand_px_zs': ['rb', 'hc'],

    "al_alumina_qtl": ['al'],
    "al_alumina_yoy_qtl": ['al'],
    "al_coal_qtl": ['al'],
    "ni_nis_mom_qtl": ['ni'],
    "ni_ore_qtl": ['ni'],
    "sn_conc_spot_hlr": ['sn'],
    'cu_prem_usd_zsa': ['cu'],
    'cu_prem_usd_md': ['cu'],
    'cu_phybasis_zsa': ['cu'],
    'cu_phybasis_hlr': ['cu'],
    "base_etf_mom_zsa": ["cu", "al", "zn", "pb", "ni", "sn"],
    "base_etf_mom_ewm": ["cu", "al", "zn", "pb", "ni", "sn"],
    "const_etf_mom_zsa": ["rb", "i", "j", "FG", "v"],
    "const_etf_mom_ewm": ["rb", "i", "j", "FG", "v"],
    "prop_etf_mom_dbth_zs": ["rb", "i", "FG", "v"],
    "prop_etf_mom_dbth_qtl": ["rb", "i", "FG", "v"],
    "prop_etf_mom_dbth_qtl2": ["rb", "i", "FG", "v"],
    "glass_etf_mom_dbth_zs": ["FG"],
    "us_oil_prod_etf_mom": ['sc', 'bu', 'TA'],

    'shibor1m_qtl': ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'ag', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    'r007_lt_zs': ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'ag', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    "MCU3_zs": ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    "cgb_1_2_spd_zs": ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    "cgb_2_5_spd_zs": ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    "fxbasket_zs": ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    'cnh_cny_zs': ['rb', 'hc', 'FG', 'SA', 'cu', 'al', 'zn', 'au', 'ag', 'l', 'pp', 'v', 'TA', 'eg', 'MA'],
    'cny_dev_zs': ['rb', 'hc', 'i', 'FG', 'SA', 'cu', 'al', 'au', 'ag', 'l', 'pp', 'v', 'TA', 'eb', 'eg', 'MA', 'sc', 'lu', 'bu'],
    'vix_zs_st': ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA', 'ag'], 
    'vix_mds_st': ['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA', 'ag'],    
    
    "bond_fxbasket_zs":  ['T', 'TF', 'TL'],
    'bond_shibor1m_qtl': ['T', 'TF', 'TL'],
    'bond_r007_lt_zs': ['T', 'TF', 'TL'],
    'bond_au_st_qtl': ['T'],

     "pmi_stl_o2inv_zs": ['rb', 'hc', 'j', 'jm', 'i'],
     "pmi_stl_o2inv_qtl_sgn": ['rb', 'hc', 'j', 'jm', 'i'],
     "pmi_order_rminv_ratio_zs": ['rb', 'hc', 'v', 'FG', 'cu', 'al', 'zn'],
     "m1_m2_spd_zs_sgn": ['rb', 'hc', 'v', 'FG', 'cu', 'zn', 'al'],
}

factors_by_asset = {
    "ryield_ema": BROAD_MKTS,
    "ryield_st_zsa": BROAD_MKTS,
    "ryield_lt_zsa": BROAD_MKTS,
    "basmom20_ema": BROAD_MKTS,
    "basmom60_ema": BROAD_MKTS,
    "basmom120_ema": BROAD_MKTS,
    "mom_hlr_st": BROAD_MKTS,
    "mom_hlr_lt": BROAD_MKTS,
    "mom_momma20": BROAD_MKTS,
    "mom_momma240": BROAD_MKTS,
    'bond_mr_st_qtl': ['T', 'TF'],    
    'bond_tf_lt_qtl': ['T', 'TF', 'TL'],
    'bond_carry_ma': ['T', 'TL'],
    'bond_tf_st_eds':  ['T', 'TF', 'TL'],    
    'lme_base_ts_mds': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'lme_base_ts_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'lme_futbasis_ma': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'base_inv_shfe_ma': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'base_inv_lme_ma': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'base_inv_exch_ma': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'base_phybas_carry_ma': ['cu', 'al', 'zn', 'ni', 'sn', 'pb'],
    'base_inv_mds': ['cu', 'al', 'zn', 'ni', 'sn', 'pb', 'ss', 'si', 'ao'],
    'base_tc_1y_zs': ['cu', 'pb', 'zn'],
    'base_tc_2y_zs': ['cu', 'pb', 'sn'],
    'base_cifprem_1y_zs': ['cu', 'al', 'zn', 'ni'],
    'base_phybasmom_1m_zs': ['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
    'base_phybasmom_1y_zs': ['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
    'metal_pbc_ema': ['cu', 'al', 'zn', 'ni', 'ss', 'sn', 'ao', #'si', 'SM', 'SF', 'pb', 
                      'rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'SA', "au", "ag"],
    'metal_mom_hlrhys': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si', 'lc', 'au', 'ag',
                         'rb', 'hc', 'i', 'j', 'jm', 'SM', 'SF', 'v', 'FG', 'SA'],
    # 'metal_pbc_ema_xdemean': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
    #                           'rb', 'hc', 'i', 'j', 'jm', 'SM', 'SF', 'v', 'FG', 'SA'],
    'metal_inv_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si', 'rb', 'hc', 'i', 'v', 'FG', 'SA'],
    'metal_inv_lyoy_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si', 'rb', 'hc', 'i', 'v', 'FG', 'SA'],
    "exch_wnt_hlr": ["UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
    "exch_wnt_yoy_hlr": ["UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
    "exch_wnt_kdj": ["UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
    'auag_etf_mrev': ["au", "ag"],
    'smsf_prodcost_mom': ["SM", "SF"],
}

factors_by_spread = {
    "auag_cme_wratio_zs": [('au', 1), ('ag', -1)],
    'auag_vix_zsa_mt': [('au', 1), ('ag', -1)],
    'auag_fxbasket_zs_yr': [('au', 2), ('cu', -1), ('rb', -1)],
    'auag_fxbasket_zs_yr': [('au', 2), ('cu', -1), ('rb', -1)],
    'smsf_workrate_ratio': [('SM', 1), ('SF', -1)],
    'smsf_dmd_ratio': [('SM', 1), ('SF', -1)],
    'smsf_margin_diff': [('SM', 1), ('SF', -1)],
}

factors_by_spread2 = {
    'rbhc_px_diff_mds': ['spd_rb_hc_c1'],
    'rbhc_px_diff_lyoy_mds': ['spd_rb_hc_c1'],
    'rbhc_phycarry_diff_zs': ['spd_rb_hc_c1'],
    'rbhc_basmom_diff_hlr': ['spd_rb_hc_c1'],
    'rbhc_steel_spd_mds': ['spd_rb_hc_c1'],
    'rbhc_steel_spd_lyoy_mds': ['spd_rb_hc_c1'],
    'rbhc_dmd_ratio_mds': ['spd_rb_hc_c1'],
    'rbhc_dmd_ratio_lyoy_mds': ['spd_rb_hc_c1'],
    'rbhc_sinv_chg_mds': ['spd_rb_hc_c1'],
    'rbhc_sinv_chg_lyoy_mds': ['spd_rb_hc_c1'],
    'rbhc_sinv_lratio_mds': ['spd_rb_hc_c1'],
    'rbhc_sinv_lratio_lyoy_mds': ['spd_rb_hc_c1'],
    'rbhc_rbsales_lyoy_zs': ['spd_rb_hc_c1'],
}

spread_config = {
    'spd_rb_hc_c1': [[('rb', 1), ('hc', -1)], 20, '-30b', 1],
}

factors_by_beta_neutral = {
    'io_pinv31_lvl_zsa': [('rb', 'i', 1), ('hc', 'i', 1)],
    'io_pinv45_lvl_hlr': [('rb', 'i', 1), ('hc', 'i', 1)],
    'ioarb_spd_qtl_1y': [('rb', 'i', 1), ('hc', 'i', 1)],
    'fef_c1_c2_ratio_spd_qtl': [('rb', 'i', 1), ('hc', 'i', 1)],
    'fef_basmom5_spd_qtl': [('rb', 'i', 1), ('hc', 'i', 1)],
    'auag_csi500_zs_st': [('au', 'ag', 1),],
    'smsf_coal_mom_st': [('SF', 'SM', 1),],
    'smsf_coal_mom_yr': [('SF', 'SM', 1),],
}


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

mr_commod_pairs = [
    ('cu', 'zn'), ('cu', 'al'), ('al', 'zn'), ('ni', 'ss'),
    ('rb', 'hc'), ('SM', 'SF'), ('FG', 'v'),
    ('y', 'OI'), ('m', 'RM'),
    ('l', 'MA'), ('pp', 'MA'), ('TA', 'MA'), ('TA', 'eg')
]

def create_holiday_window_series(index, holidays, pre_days, post_days):
    chn_bday = CustomBusinessDay(holidays=pd.to_datetime(CHN_Holidays))
    series = pd.Series(0, index=index)        
    for holiday in holidays:        
        start = holiday + pre_days * chn_bday
        end = holiday + post_days * chn_bday               
        series.loc[(series.index >= start) & (series.index <= end)] = 1
    return series


def cnc_hol_seasonality(price_df, spot_df, product_list):
    signal_ts1 = create_holiday_window_series(price_df.index, 
                                            [pd.Timestamp(datetime.date(yr, 5, 1)) for yr in range(price_df.index[0].year-1, price_df.index[-1].year+2)], -2, 1) 
    signal_ts2 = create_holiday_window_series(price_df.index, 
                                            [pd.Timestamp(datetime.date(yr, 10, 1)) for yr in range(price_df.index[0].year-1, price_df.index[-1].year+2)], -2, 1)
    signal_ts3 = create_holiday_window_series(price_df.index, 
                                            [pd.Timestamp(datetime.date(yr, 1, 1)) for yr in range(price_df.index[0].year-1, price_df.index[-1].year+2)], -6, 1)
    signal_ts4 = create_holiday_window_series(price_df.index, 
                                            [pd.Timestamp(lunardate.LunarDate(yr,1,1).toSolarDate()) for yr in range(price_df.index[0].year-1, price_df.index[-1].year+2)], -17, 5)
    
    signal_ts = signal_ts1*0.8 + signal_ts2 + signal_ts3*1.2 + signal_ts4
    signal_df = pd.DataFrame(dict([(asset, signal_ts) for asset in product_list]))
    return signal_df


def beta_spd_seasonality(price_df, spot_df, product_list):
    signal_df = pd.DataFrame(index=price_df.index, columns=product_list)
    if ('rb' in product_list) and ('i' in product_list):
        rb_pxchg = price_df[('rbc1', 'close')].pct_change()
        rb_vol = rb_pxchg.rolling(20).std()
        io_pxchg = price_df[('ic1', 'close')].pct_change()
        io_vol = io_pxchg.rolling(20).std()
        beta = beta_hedge_ratio(rb_pxchg, io_pxchg, beta_win=245, beta_rng=[0, 2], corr_step=5).dropna()
        spd_pxchg = (rb_pxchg - beta['port'] * io_pxchg)
        spd_vol = spd_pxchg.dropna().rolling(20).std()    
        feature_ts = spd_pxchg.dropna().cumsum()
        signal_ts = calc_conv_signal(feature_ts, signal_func='zscore', 
                                    param_rng=[80, 120, 2], signal_cap=[-2, 2], vol_win=120)
        signal_ts.loc[signal_ts.index.month.isin([1, 2, 11, 12])] = -signal_ts.loc[signal_ts.index.month.isin([1, 2, 11, 12])]
        signal_df['rb'] = signal_ts*rb_vol/spd_vol
        signal_df['i'] = - signal_ts*beta['port']*io_vol/spd_vol
    if ('au' in product_list) and ('ag' in product_list):
        au_pxchg = price_df[('auc1', 'close')].pct_change()
        au_vol = au_pxchg.rolling(20).std()
        ag_pxchg = price_df[('agc1', 'close')].pct_change()
        ag_vol = ag_pxchg.rolling(20).std()
        beta = beta_hedge_ratio(au_pxchg, ag_pxchg, beta_win=245, beta_rng=[0, 2], corr_step=5).dropna()
        spd_pxchg = (au_pxchg - beta['port'] * ag_pxchg)
        spd_vol = spd_pxchg.dropna().rolling(20).std()    
        signal_ts = pd.Series(0, index=price_df.index)        
        signal_ts.loc[signal_ts.index.month.isin([1, 2, 3, 12])] = 1
        signal_df['au'] = signal_ts*au_vol/spd_vol
        signal_df['ag'] = - signal_ts*beta['port']*ag_vol/spd_vol
    return signal_df


def beta_spd_trend(price_df, spot_df, product_list, pair_list,
                   signal_func='qtl', param_rng=[40, 80, 2], 
                   vol_win=20, signal_cap=[-2, 2], bullish=True,
                   beta_win=245, signal_buf=0):
    signal_df = pd.DataFrame(0, index=price_df.index, columns=product_list)
    for trade_asset, index_asset in pair_list:
        if (trade_asset not in product_list) or (index_asset not in product_list):
            continue
        trade_pxchg = price_df[(f'{trade_asset}c1', 'close')].pct_change()
        trade_vol = trade_pxchg.rolling(20).std()
        index_pxchg = price_df[(f'{index_asset}c1', 'close')].pct_change()
        index_vol = index_pxchg.rolling(20).std()
        beta = beta_hedge_ratio(trade_pxchg, index_pxchg, beta_win=beta_win, beta_rng=[0, 100], corr_step=5).dropna()
        spd_pxchg = (trade_pxchg - beta['port'] * index_pxchg)
        spd_vol = spd_pxchg.dropna().rolling(vol_win).std()    
        feature_ts = spd_pxchg.dropna().cumsum()
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, 
                                     param_rng=param_rng, signal_cap=signal_cap, vol_win=120)        
        signal_ts = signal_buffer(signal_ts, signal_buf)
        if not bullish:
            signal_ts = -signal_ts
        signal_df[trade_asset] += signal_ts*trade_vol/spd_vol
        signal_df[index_asset] -= signal_ts*beta['port']*index_vol/spd_vol
    return signal_df


def leader_lagger(price_df, spot_df, product_list, leadlag_port=leadlag_port_d, conv_func='qtl', signal_cap=None):     
    signal_df = pd.DataFrame(index=price_df.index, columns=product_list)
    for prod in product_list:
        for sector in leadlag_port:
            if prod in leadlag_port[sector]['lag']:
                signal_list = []
                for lead_prod in leadlag_port[sector]['lead']:
                    feature_ts = price_df[(lead_prod+'c1', 'close')]
                    signal_ts = calc_conv_signal(feature_ts.dropna(), conv_func,
                                                 leadlag_port[sector]['param_rng'], signal_cap=signal_cap)
                    signal_list.append(signal_ts)
                signal_df[prod] = pd.concat(signal_list, axis=1).mean(axis=1)
                break
            else:
                signal_df[prod] = 0
    return signal_df


def leadlag2_mr(price_df, spot_df, product_list, mr_dict, leadlag_dict, 
                win=10, signal_func='ma', signal_cap= [-3, 3], 
                vol_win=60, param_rng=[1, 2, 1]): 
    signal_df = pd.DataFrame(index=price_df.index, columns=product_list)
    for asset in mr_dict:
        tmp_df = pd.concat([
            price_df[(asset+'c1', 'close')].to_frame(asset),
            price_df[(mr_dict[asset]+'c1', 'close')].to_frame(mr_dict[asset])], axis=1).dropna(how='all').ffill()
        tmp_df = tmp_df.pct_change()
        tmp_df = tmp_df.rolling(win).sum()
        feature_ts = tmp_df[asset] - tmp_df[mr_dict[asset]]
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng, signal_cap=signal_cap,
                                     vol_win=vol_win)
        signal_ts = signal_ts.ewm(3).mean()
        signal_df[asset] = signal_ts
    signal_df2 = pd.DataFrame(index=price_df.index, columns=product_list)
    for asset in leadlag_dict:
        tmp_df = pd.concat([
            price_df[(asset+'c1', 'close')].to_frame(asset),
            price_df[(leadlag_dict[asset]+'c1', 'close')].to_frame(leadlag_dict[asset])], axis=1).dropna(how='all').ffill() 
        tmp_df = tmp_df.pct_change()
        tmp_df = tmp_df.rolling(win).sum()
        feature_ts = tmp_df[leadlag_dict[asset]] - tmp_df[asset]
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng, signal_cap=signal_cap,
                                     vol_win=vol_win)
        signal_ts = signal_ts.ewm(3).mean()
        signal_df2[asset] = signal_ts
    signal_df = signal_df.fillna(0) + signal_df2.fillna(0)
    return signal_df


def mr_pair(price_df, spot_df, product_list, mr_pair_list=mr_commod_pairs,
            signal_func='zscore_adj', param_rng=[200, 250, 2], 
            vol_win=120, signal_cap=None, bullish=False):    
    signal_df = pd.DataFrame(0, index=price_df.index, columns=product_list)
    for (asset_a, asset_b) in mr_pair_list:
        pair_assets = [asset_a, asset_b]
        sig_df = pd.DataFrame(index=price_df.index, columns=pair_assets)
        feature_ts = np.log(price_df[(asset_a+'c1', 'close')]) - np.log(price_df[(asset_b+'c1', 'close')])
        sig_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng, signal_cap=signal_cap,
                                  vol_win=vol_win)
        sig_ts = sig_ts.apply(lambda x: np.sign(x) * min(abs(x), 1.25) ** 4).ewm(1).mean()
        if not bullish:
            sig_ts = -sig_ts
        sig_df[asset_a] = sig_ts
        sig_df[asset_b] = -sig_ts
        signal_df = signal_df + sig_df.reindex_like(signal_df).fillna(0)
    return signal_df


def seasonal_custom_1(price_df, spot_df, product_list, now=datetime.datetime.now()):
    signal_df = pd.DataFrame(0, index=price_df.index, columns=product_list)
    if 'au' in product_list:
        signal_ts = calc_conv_signal(price_df[("auc1", "close")], 
                                     signal_func='hlratio', 
                                     param_rng=[120, 160, 2], 
                                     signal_cap=[-2, 2], 
                                     vol_win=120)
        last_signal = signal_ts[-1]
        signal_ts.loc[signal_ts.index.weekday.isin([2, 3])] = 1
        if (signal_df.index[-1].weekday() == 2) and (now.weekday() == 2):
            signal_ts[-1] = last_signal
        elif (signal_df.index[-1].weekday() == 4) and (now.weekday() == 4):
            signal_ts[-1] = 1                    
        signal_df['au'] = signal_ts*1.5

    for asset in ['l', 'pp', 'v', 'MA']:
        if asset in product_list:
            signal_df.loc[signal_df.index.day.isin(range(4, 10)), asset] += -0.25
            signal_df.loc[~signal_df.index.day.isin(range(4, 10)), asset] += 0.25
            signal_df.loc[signal_df.index.day.isin(range(5, 23)), asset] += -0.25
            signal_df.loc[~signal_df.index.day.isin(range(5, 23)), asset] += 0.25

    if 'SF' in product_list:
        signal_df.loc[signal_df.index.weekday.isin([2, 3]), 'SF'] = 1

    for asset in ['T', 'TF']:
        if asset in product_list:
            signal_df.loc[signal_df.index.month.isin([1, 2, 3, 4, 5, 6, 7, 12]), asset] += 0.5
            signal_df.loc[signal_df.index.day.isin(range(16, 32)), asset] += 0.5
            signal_df.loc[signal_df.index.weekday.isin([1]), asset] += 0.25

    if 'bu' in product_list:
        signal_df.loc[signal_df.index.month.isin([1, 12]), 'bu'] = 1
        signal_df.loc[signal_df.index.month.isin([8, 9, 10]), 'bu'] = -0.5        

    if 'ru' in product_list:        
        flag = signal_df.index.month.isin([3, 4, 5, 6]) | signal_df.index.day.isin(range(18, 26))
        signal_df.loc[flag, 'ru'] = -1

    flag1 = signal_df.index.month.isin([1, 2])
    flag2 = signal_df.index.month.isin([10, 11])
    for asset in ['lu', 'fu']:
        if asset in product_list:
            signal_df.loc[flag1, asset] = 0.5
            signal_df.loc[flag2, asset] = -0.5

    if 'pb' in product_list:        
        signal_df.loc[signal_df.index.day.isin(range(17, 31)), 'pb'] = 1
    ferrous_products = ['rb', 'hc', 'i']
    if set(ferrous_products) <= set(product_list):
        start_mth = 11
        start_day = 24
        end_mth = 1
        end_day = 4
        date_list = [d for d in signal_df.index 
                    if ((d.date()>=datetime.date(d.year, start_mth, start_day)) & 
                        (d.date()<=datetime.date(d.year+1, end_mth, end_day))) | 
                        ((d.date()>=datetime.date(d.year-1, start_mth, start_day)) & 
                         (d.date()<=datetime.date(d.year, end_mth, end_day)))]
        signal_df.loc[signal_ts.index.isin(date_list), 'rb'] = 0.4
        signal_df.loc[signal_ts.index.isin(date_list), 'hc'] = 0.4
        signal_df.loc[signal_ts.index.isin(date_list), 'i'] = 0.7
    return signal_df


factors_by_func = {
    "seasonal_custom_1": {
        'func': seasonal_custom_1,
        'args': {            
            'now': datetime.datetime.now(),
            'product_list': [
                'au', 'l', 'pp', 'v', 'MA', 'pb', 'rb', 'hc', 'i', 'SF', 'ru', 'lu', 'fu', 'bu',
            ],
        },        
    },
    "long_hol_2b": {
        'func': cnc_hol_seasonality,
        'args': {
            'product_list': [
                'rb', 'hc', 'i', 'j', 'jm', 'zn', 'ni', 'ss', 'sn',
                'l', 'pp', 'v', 'TA', 'sc', 'eb', 'eg', 'lu',
                'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF'
            ],
        },
    },    
    "beta_spd_seazn": {
        'func': beta_spd_seasonality,
        'args': {
            'product_list': [
                'rb', 'i', 'au', 'ag',
            ],
        },        
    },
    "seazn_cal_mth_sr": {
        'func': seasonal_cal_update,
        'args': {            
            'cal_key':'cal_mth',
            'cal_func': calendar_label,
            'label_field': 'label_mth',
            'shift': 0,
            'season_rng': [1, 12],
            'product_list': [
                'rb', 'hc', 'i', 'j', 'FG', 
                'l', 'pp', 'v', 'TA', 'MA', 'eg', 
                'm', 'RM', 'p', 'OI', 
                'a', 'c', 'CF', 'jd',
            ],
        },        
    },
    "seazn_lunar_wk2_sr": {
        'func': seasonal_cal_update,
        'args': {            
            'cal_key':'lunar_wk2',
            'cal_func': lunar_label,
            'label_field': 'label_wk',
            'shift': 2,
            'season_rng': [-26, 26],
            'product_list': [
                'rb', 'hc', 'i', 'j', 'FG', 
                'l', 'pp', 'v', 'TA', 'MA', 'eg',
                'm', 'RM', 'p', 'OI', 
                'a', 'c', 'CF', 'jd',
            ],
        },        
    },
    "leadlag_d_mid": {
        'func': leader_lagger,
        'args': {
            'leadlag_port': leadlag_port_d, 
            'conv_func': 'qtl', 
            'signal_cap': [-2, 2],
            'product_list': [
                'rb', 'hc', 'i', 'j', 'jm', 'FG', 'SM', 'SF', 'UR', 
                'cu', 'al', 'zn', 'sn', 'ss', 'ni',
                'l', 'pp', 'v', 'TA', 'sc', 'eb', 'eg', 'y', 'p', 'OI'
            ],
        },
    },
    'leadlag2_mr_d': {
        'func': leadlag2_mr,        
        'args': {
            'win': 10, 
            'signal_cap': [-3, 3], 
            'vol_win': 60,
            'param_rng': [1, 2, 1],
            'signal_func': 'ma',
            'mr_dict': {
                'hc': 'rb',
                'i': 'rb',
                'j': 'rb',
                'pp': 'l',
                'SA': 'FG',
                'sc': 'lu',                
                'm': 'y'
                },
            'leadlag_dict': {
                'lu': 'sc',
                'RM': 'm',
                'a': 'm',
                'cs': 'c',
                'eb': 'sc',
                # 'fu': 'sc',
                'bu': 'sc',
                #'bc': 'cu',
                'MA': 'pp',
                'ni': 'ss',
                'al': 'ao',
                },
            'product_list': [
                'rb', 'hc', 'i', 'j', 'al', 'FG', 'SA',
                'l', 'pp', 'lu', 'sc', 'm', 'RM', 'y',
                'c', 'cs', 'MA', 'a', 'eb', 'bu',
                'cu', 'ss', 'ni', 'ao', #'bc', 
                ],
        },
    },
    'beta_spd_trend_3m': {
        'func': beta_spd_trend,
        'args': {
            'product_list': [
                'hc', 'FG', 'SA', 
                'zn', 'l', 'pp', 'ao', 'al',
                'm', 'y', 'p', 'OI', 'a',                 
                ],
            'pair_list': [
                ('hc', 'zn'), ('FG', 'SA'), ('SA', 'FG'), ('ao', 'al'),  ('pp', 'l'), 
                ('m', 'y'), ('y', 'm'), ('m', 'a'), ('y', 'p'), ('p', 'y'), ('OI', 'p'), ('p', 'OI'),
            ],
            'param_rng': [40, 80, 2],
            'bullish': True,
            'vol_win': 20, 
            'beta_win': 244,
            'signal_cap': [-2, 2],
            'signal_func': 'qtl',
            'signal_buf': 0.1
        }
    }, 
    'beta_spd_trend_1y': {
        'func': beta_spd_trend,
        'args': {
            'product_list': [
                'rb', 'i', 'j', 
                ],
            'pair_list': [
                ('rb', 'i'), ('i', 'rb'), ('j', 'i'),
            ],
            'param_rng': [240, 260, 2],
            'bullish': True,
            'vol_win': 20, 
            'beta_win': 244,
            'signal_cap': [-2, 2],
            'signal_func': 'qtl',
            'signal_buf': 0.0
        }
    },        
    'pair_mr_1y': {
        'func': mr_pair,
        'args': {
            'product_list': [
                'cu', 'al', 'zn', 'ss', 'ni', 
                'rb', 'hc', 'SM', 'SF', 'FG', 'v',
                'l', 'MA', 'pp', 'TA', 'eg',
                'y', 'OI', 'm', 'RM',
                ],
            'mr_pair_list': [
                ('cu', 'zn'), ('cu', 'al'), ('al', 'zn'), ('ni', 'ss'),
                ('rb', 'hc'), ('SM', 'SF'), ('FG', 'v'),
                ('y', 'OI'), ('m', 'RM'),  ('l', 'MA'), ('pp', 'MA'), ('TA', 'eg'),
            ],
            'param_rng': [200, 250, 5],
            'bullish': False,
            'vol_win': 120, 
            'signal_cap': None,
            'signal_func': 'zscore_adj',
        }
    },
}


def get_fun_data(start_date, run_date):
    run_date = pd.to_datetime(run_date)
    e_date = day_shift(run_date.date(), '5b', CHN_Holidays)
    cdate_rng = pd.date_range(start=start_date, end=e_date, freq='D', name='date')
    data_df = load_codes_from_edb(index_map.keys(), source='ifind', column_name='index_code')
    data_df = data_df.rename(columns=index_map)
    spot_df = data_df.dropna(how='all').copy(deep=True)
    spot_df = spot_df.reindex(index=cdate_rng)
    spot_df = process_spot_df(spot_df, adjust_time=True)

    stock_df = load_int_stock_daily(["XOM.N", "BP.N", 'CVX.N', 'SU.N', 'EOG.N', 'APA.N', 
                                     'COP.N', 'VLO.N', 'PSX.N', 'MPC.N', 'PBF.N', 
                                     "SPY.P", "GDX.P", "USO.P", "GLD.P"])
    stock_pct_chg = stock_df.loc[:, stock_df.columns.get_level_values(1)=='close'].droplevel(level=[1], axis=1).pct_change()
    spot_dict = {}
    spot_dict["us_oil_prod_etf_perf"] = (1 + stock_pct_chg[['VLO.N', 'PSX.N', 'MPC.N', 'PBF.N']].mean(axis=1)).cumprod()
    for nb in [2, 3, 4]:
        fef_nb = nearby('FEF', n=nb,
                        start_date=max(start_date, datetime.date(2016, 7, 1)),
                        end_date=run_date.date(),
                        roll_rule='-2b', roll_col='settle',
                        freq='d', shift_mode=2)
        fef_nb.index = pd.to_datetime(fef_nb.index)
        fef_nb.loc[fef_nb['settle'] <= 0, 'settle'] = np.nan
        fef_nb.loc[fef_nb['close'] <= 0, 'close'] = np.nan
        spot_dict[f'FEFc{nb-1}'] = fef_nb['settle']
        spot_dict[f'FEFc{nb-1}_close'] = fef_nb['close']
        spot_dict[f'FEFc{nb-1}_shift'] = fef_nb['shift'] 
    spot_dict['FEF_c1_c2_ratio'] = (spot_dict['FEFc1']/np.exp(spot_dict['FEFc1_shift'])) / \
                                 (spot_dict['FEFc2']/np.exp(spot_dict['FEFc2_shift']))
    spot_dict['FEF_c123fly_ratio'] = spot_dict['FEFc1'] * spot_dict['FEFc3'] / \
                                   (spot_dict['FEFc2'] * spot_dict['FEFc2']) * \
                                   np.exp(2 * spot_dict['FEFc2_shift'] - spot_dict['FEFc1_shift'] - spot_dict['FEFc3_shift'])
    spot_dict['FEF_ryield'] = (np.log(spot_dict['FEFc1'] / np.exp(spot_dict['FEFc1_shift'])) -
                             np.log(spot_dict['FEFc2'] / np.exp(spot_dict['FEFc2_shift']))) * 12
    spot_dict['FEF_basmom'] = np.log(1 + spot_dict['FEFc1'].dropna().pct_change()) - \
                            np.log(1 + spot_dict['FEFc2'].dropna().pct_change())
    spot_dict['FEF_basmom10'] = spot_dict['FEF_basmom'].dropna().rolling(10).sum()
    spot_dict['FEF_basmom5'] = spot_dict['FEF_basmom'].dropna().rolling(5).sum()    
    spot_df = pd.concat([spot_df, pd.DataFrame(spot_dict)], axis=1)
    return spot_df


def save_signal_to_db(asset, factor_name, signal_ts, run_date, roll_label='hot', freq='d1', flavor='mysql'):
    fact_config = {'roll_label': roll_label, 'freq': freq,
                   'serial_key': '0', 'serial_no': 0,
                   'product_code': asset, 'exch': prod2exch(asset)}
    if len(signal_ts) == 0:
        print(f"{factor_name}: {asset} for run_date={run_date} is empty")
        return
    asset_df = pd.DataFrame(signal_ts.to_frame(factor_name), index=signal_ts.index)
    asset_df.index = asset_df.index.map(lambda x: x.date())
    asset_df.index.name = 'date'
    update_factor_db(asset_df, factor_name, fact_config,
                     start_date=signal_ts.index[0].date(),
                     end_date=run_date,
                     flavor=flavor)


def load_hist_fut_prices(markets, start_date, end_date,
                         shift_mode=2, roll_name='hot', nb_cont=1, freq='d1'):
    fields = ['contract', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'diff_oi', 'expiry', 'mth', 'shift']
    data_df = pd.DataFrame()
    for prodcode in markets:
        for nb in range(nb_cont):
            freq = freq[0]
            if roll_name == 'CAL_30b':
                roll = '-30b'
                if prodcode in ["IF", "IC", "IH", "IM"]:
                    roll = '0b'
                elif prodcode in ['cu', 'al', 'zn', 'pb', 'sn', 'ss', 'lu']:
                    roll = '-25b'
                elif prodcode in ['ni', 'jd', 'lh', 'eg',]:
                    roll = '-35b'
                elif prodcode in ['v', 'MA', 'rb', 'hc']:
                    roll = '-28b'
                elif prodcode in ['sc', 'eb', 'T', 'TF', 'TS', 'TL']:
                    roll = '-20b'
                elif prodcode in ['au', 'ag']:
                    roll = '-15b'
                sdate = max(start_date, sim_start_dict.get(prodcode, start_date))
                xdf = nearby(prodcode, nb+1,
                                    start_date=sdate,
                                    end_date=end_date,
                                    shift_mode=shift_mode,
                                    freq=freq,
                                    roll_rule=roll).reset_index()
            else:            
                xdf = dataseries.nearby(prodcode,
                                        nb + 1,
                                        start_date=start_date,
                                        end_date=end_date,
                                        shift_mode=shift_mode,
                                        freq=freq,
                                        roll_name=roll_name)            
            xdf['expiry'] = pd.to_datetime(xdf.apply(lambda x: contract_expiry(x['contract'], curr_dt=x['date']), axis=1))
            xdf['contmth'] = xdf.apply(lambda x: inst2contmth(x['contract'], x['date']), axis=1)
            xdf['mth'] = xdf['contmth'].apply(lambda x: x // 100 * 12 + x % 100)
            xdf['product'] = f"{prodcode}c{nb + 1}"
            data_df = pd.concat([data_df, xdf])
    if 'm' in freq:
        index_col = 'datetime'
    else:
        index_col = 'date'
    df = pd.pivot_table(data_df.reset_index(), index=index_col, columns='product', values=list(fields),
                        aggfunc='last')
    df = df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.rename(['product', 'field'], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def update_db_factor(run_date=datetime.date.today(), flavor='mysql'):
    roll_name='hot'
    freq='d1'
    funda_start = day_shift(run_date, '-8y')
    update_start = day_shift(run_date, '-120b', CHN_Holidays)
    markets = [
        'rb', 'hc', 'i', 'j', 'jm', 'FG', 'v', 'SM', 'SF', 'SA', 'ru', 'nr',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'si', 'ao', 'au', 'ag', 'bc', 'lc', 'PX',
        'l', 'pp', 'TA', 'MA', 'sc', 'eb', 'eg', 'UR', 'lu', 'bu', 'fu', 'PF', 
        'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', 'AP', 'lh', 'cs', 'CJ', 'PK', 'b',        
        'T', 'TF', 'TL',
    ]
    price_start = day_shift(run_date, '-30m')
    # load tmp saved file
    logging.info("loading eod price data ... ")
    try:
        price_df = pd.read_parquet(f"C:/dev/data/fut_eod_%s.parquet" % run_date.strftime("%Y%m%d"))
    except:
        price_df = load_hist_fut_prices(markets, start_date=price_start, end_date=run_date, 
                                        roll_name=roll_name, nb_cont=2, freq=freq)
        spd_list = [price_df]
        for spd_name in spread_config:
            asset_list = spread_config[spd_name][0]
            roll_rule=spread_config[spd_name][2]   
            nb_cont=spread_config[spd_name][3]
            leg1_df = nearby(asset_list[0][0], n=nb_cont,
                             start_date=price_start,
                             end_date=run_date,
                             roll_rule=roll_rule, freq='d', shift_mode=1)
            leg1_df.index = pd.to_datetime(leg1_df.index)
            leg2_df = nearby(asset_list[1][0], n=nb_cont,
                             start_date=price_start,
                             end_date=run_date,
                             roll_rule=roll_rule, freq='d', shift_mode=1)
            leg2_df.index = pd.to_datetime(leg2_df.index) 
            for field in ['close', 'shift', 'open']:
                spd_list.append((leg1_df[field]*asset_list[0][1] + leg2_df[field]*asset_list[1][1]).to_frame((spd_name, field)))                    
        price_df = pd.concat(spd_list, axis=1)
        price_df.to_parquet(f"C:/dev/data/fut_eod_%s.parquet" % run_date.strftime("%Y%m%d"))

    logging.info("loading fundamental data ... ")
    cutoff_date = day_shift(day_shift(run_date, '1b', CHN_Holidays), '-1d')
    spot_df = get_fun_data(funda_start, run_date)
    spot_df['io_ctd_spot'] = io_ctd_basis(spot_df, price_df[('ic1', 'expiry')])
    fact_config = {'roll_label': roll_name, 'freq': freq, 'serial_key': 0, 'serial_no': 0}
    vol_win = 20
    logging.info("updating price vol data ... ")
    for asset_cont in price_df.columns.get_level_values(0).unique():
        asset = asset_cont[:-2]
        local_df = pd.DataFrame(index=price_df.index)
        local_df['close'] = price_df[(asset+'c1', 'close')]
        local_df['pct_chg'] = price_df[(asset+'c1', 'close')].pct_change()
        local_df['pct_vol'] = local_df['close'] * local_df['pct_chg'].rolling(vol_win).std()
        local_df.index.name = 'date'
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        update_factor_db(local_df, 'pct_vol', fact_config,
                         start_date=pd.to_datetime(update_start),
                         end_date=pd.to_datetime(run_date), flavor=flavor)

    logging.info("update ryield/basmom ... ")
    data_dict = {}
    for asset in markets:
        spot_df[f'{asset}_px'] = price_df[(asset+'c1', 'close')]
        if (asset+'c2', 'close') in price_df.columns:
            data_dict[f'{asset}_ryield'] = \
                (np.log(price_df[(asset+'c1', 'close')]) - np.log(price_df[(asset+'c2', 'close')]) - \
                 price_df[(asset+'c1', 'shift')] + price_df[(asset+'c2', 'shift')])/\
                    (pd.to_datetime(price_df[(asset+'c2', 'expiry')]) - pd.to_datetime(price_df[(asset+'c1', 'expiry')])).dt.days * 365.0 + \
                        spot_df['r007_cn'].reindex(index=price_df.index).ffill().ewm(5).mean()/100
        
            data_dict[f'{asset}_basmom'] = np.log(price_df[(asset+'c1', 'close')].pct_change()+1) - \
                np.log(price_df[(asset+'c2', 'close')].pct_change()+1)
            data_dict[f'{asset}_basmom5'] = data_dict[f'{asset}_basmom'].dropna().rolling(5).sum() 
            data_dict[f'{asset}_basmom10'] = data_dict[f'{asset}_basmom'].dropna().rolling(10).sum() 
            data_dict[f'{asset}_basmom20'] = data_dict[f'{asset}_basmom'].dropna().rolling(20).sum() 
            data_dict[f'{asset}_basmom60'] = data_dict[f'{asset}_basmom'].dropna().rolling(60).sum() 
            data_dict[f'{asset}_basmom120'] = data_dict[f'{asset}_basmom'].dropna().rolling(120).sum() 
        if asset in commod_phycarry_dict:
            asset_feature = commod_phycarry_dict[asset]
            tmp_df = pd.concat([spot_df[[asset_feature, 'r007_cn']].dropna(how='all'), 
                                (price_df[(asset+'c1', 'close')] / np.exp(price_df[(asset+'c1', 'shift')])).dropna().to_frame(f'{asset}_c1'),
                                pd.to_datetime(price_df[(asset+'c1', 'expiry')]).dropna().to_frame(f'{asset}_expiry')], axis=1).ffill()
            tmp_df['date'] = pd.to_datetime(tmp_df.index)            
            tmp_df['r007_cn'] = tmp_df['r007_cn'].ewm(5).mean()/100
            data_dict[f'{asset}_phycarry'] = (np.log(tmp_df[asset_feature]) - np.log(tmp_df[f'{asset}_c1'])) / \
                                                (tmp_df[f'{asset}_expiry'] - tmp_df['date']).dt.days * 365 + tmp_df['r007_cn']
    data_dict['hc_rb_diff'] = np.log(price_df[('hcc1', 'close')]) - np.log(price_df[('rbc1', 'close')])
    data_dict['rb_hc_basmom_diff'] = data_dict['rb_basmom'] - data_dict['hc_basmom']
    data_dict['rb_hc_phycarry_diff'] = data_dict['rb_phycarry'] - data_dict['hc_phycarry']
    spot_df = pd.concat([spot_df, pd.DataFrame(data_dict)], axis=1)

    logging.info("update factor by asset ... ")
    asset_factors = [] # only update ts factors, not xs
    for factor_name in factors_by_asset.keys():        
        if factor_name[-8:] == '_xdemean':
            db_fact_name = factor_name[:-8]
        elif factor_name[-7:] == '_xscore':
            db_fact_name = factor_name[:-7]
        elif factor_name[-6:] == '_xrank':
            db_fact_name = factor_name[:-6]
        else:
            db_fact_name = factor_name
        if db_fact_name in asset_factors:
            continue
        for asset in factors_by_asset[factor_name]:
            logging.info(f"calculating factor={factor_name} for asset={asset}")
            signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                    price_df=price_df,
                                                    asset=asset,
                                                    curr_date=run_date)
            save_signal_to_db(asset, db_fact_name, signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)
        asset_factors.append(db_fact_name)

    for factor_name in single_factors:
        logging.info(f"calculating factor={factor_name}")
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                curr_date=run_date)
        for asset in single_factors[factor_name]:
            save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_func:
        logging.info(f"calculating factor={factor_name}")
        func = factors_by_func[factor_name]['func']
        func_args = factors_by_func[factor_name]['args']        
        signal_df = func(price_df, spot_df, **func_args)
        for asset in signal_df.columns:
            signal_ts = signal_df[asset].dropna()
            if len(signal_ts) > 0:
                save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_spread.keys():
        logging.info(f"calculating factor={factor_name}")
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                curr_date=run_date)
        for asset, weight in factors_by_spread[factor_name]:
            save_signal_to_db(asset, factor_name, weight*signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_spread2.keys():
        logging.info(f"calculating factor={factor_name}")
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                curr_date=run_date)        
        for spd_name in factors_by_spread2[factor_name]:
            asset_list = spread_config[spd_name][0]
            vol_win = spread_config[spd_name][1]
            roll_label = spread_config[spd_name][2]
            nb_cont = spread_config[spd_name][3]
            vol_ts = price_df[(spd_name, 'close')].diff().rolling(vol_win).std()
            for asset, weight in asset_list: 
                spd_vol = vol_ts
                save_signal_to_db(asset, 'spd_vol', spd_vol[update_start:], run_date=cutoff_date, roll_label=roll_label, freq='d1', flavor=flavor) 
                save_signal_to_db(asset, factor_name, weight*signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)


    logging.info("updating factor for beta neutral ratio ...")
    beta_win = 244
    asset_pairs = [('rb', 'i'), ('hc', 'i'), ('j', 'i'), ('au', 'ag'), ('au', 'cu'), ('au', 'rb'), ('SF', 'SM')]
    beta_dict = {}
    for trade_asset, index_asset in asset_pairs:
        key = '_'.join([trade_asset, index_asset, 'beta'])        
        fact_config = {'roll_label': roll_name, 'freq': freq,
                'serial_key': '0', 'serial_no': 0,
                'product_code': key, 'exch': 'xasset'}
        asset_df = price_df[[(trade_asset+'c1', 'close'), (index_asset+'c1', 'close')]].droplevel([1], axis=1)
        asset_df = asset_df.pct_change().dropna()
        asset_df.columns = [col[:-2] for col in asset_df.columns]
        for asset in asset_df:
            asset_df[f'{asset}_pct'] = asset_df[asset].rolling(5).mean()
            asset_df[f'{asset}_vol'] = asset_df[asset].rolling(vol_win).std()
        asset_df['beta'] = asset_df[f'{index_asset}_pct'].rolling(beta_win).cov(
            asset_df[f'{trade_asset}_pct']) / asset_df[f'{index_asset}_pct'].rolling(beta_win).var()
        asset_df['pct_chg'] = asset_df[trade_asset] - asset_df['beta'] * asset_df[index_asset].fillna(0)
        asset_df['pct_vol'] = asset_df['pct_chg'].rolling(vol_win).std()
        asset_df['trade_leg'] = asset_df[f'{trade_asset}_vol']/asset_df['pct_vol']
        asset_df['index_leg'] = - asset_df[f'{index_asset}_vol'] / asset_df['pct_vol'] * asset_df['beta']
        beta_dict[key] = asset_df[['trade_leg', 'index_leg']]
        for field in ['pct_vol', 'beta', 'trade_leg', 'index_leg']:
            update_factor_db(asset_df, field, fact_config, start_date=update_start, end_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_beta_neutral.keys():
        logging.info(f"calculating factor={factor_name}")
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                curr_date=run_date)
        signal_df = pd.DataFrame(index=signal_ts.index)
        signal_df['raw_sig'] = signal_ts
        asset_list = []
        for trade_asset, index_asset, weight in factors_by_beta_neutral[factor_name]:
            asset_list = list(set(asset_list + [trade_asset, index_asset]))
            if trade_asset not in signal_df.columns:
                signal_df[trade_asset] = 0
            if index_asset not in signal_df.columns:
                signal_df[index_asset] = 0
            key = '_'.join([trade_asset, index_asset, 'beta'])
            signal_df['trade_ratio'] = beta_dict[key]['trade_leg']
            signal_df['index_ratio'] = beta_dict[key]['index_leg']
            signal_df = signal_df.ffill()
            signal_df[trade_asset] += signal_df['trade_ratio'] * signal_df['raw_sig'] * weight
            signal_df[index_asset] += signal_df['index_ratio'] * signal_df['raw_sig'] * weight
        for asset in asset_list:
            save_signal_to_db(asset, factor_name, signal_df[asset][update_start:], run_date=cutoff_date, flavor=flavor)


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
    name = "fun_factor_update"
    base.config_logging(folder + name + ".log", level=logging.INFO,
                        format='%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                        to_console=True,
                        console_level=logging.INFO)    
    logging.info("running for %s" % str(tday))
    update_db_factor(run_date=tday)
