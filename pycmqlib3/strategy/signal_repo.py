import pandas as pd
from pycmqlib3.analytics.tstool import *
from pycmqlib3.analytics.btmetrics import simple_cost
from pycmqlib3.utility.misc import CHN_Holidays, day_shift
from pycmqlib3.utility.exch_ctd_func import *

BROAD_MKTS = [
    'rb', 'hc', 'i', 'j', 'jm',
    'SM', 'SF', 'SA', 'FG', 'v', 'SH',
    'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao',
    'au', 'ag', #'pt', #'bc',
    'si', 'lc', 'ps', #'ec',
    'ru', 'UR', 'sp', 'nr', 'br',
    'l', 'pp', 'TA', 'PX', 'eg', 'MA', 'eb', #'PF',
    'sc', 'lu', 'bu', 'fu', 'pg',
    'm', 'RM', 'y', 'p', 'OI', 'a', 'b', 'c', 'cs',
    'CJ', 'CF', 'jd', 'AP', 'lh', 'SR', 'PK',
]

IND_MKTS = [
    'rb', 'hc', 'i', 'j', 'jm', 'FG', 'v', 'SH', #'SM', 'SF', 'SA', 'UR',
    'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'au', 'ag', 'si', 'lc', 'PX', 'ps', #'bc',
    'ru', 'l', 'pp', 'TA', 'MA', 'sc', 'eb', 'eg', 'lu', #'bu', 'fu',
]

AGS_MKTS = [
    'm', 'RM', 'y', 'p', 'OI', 'a', 'b', 'c', 'cs', 'CF', 'jd', 'AP', 'CJ', 'lh', 'SR', 'PK',
]

signal_store = {
    "ryield_ema": [BROAD_MKTS, ["ryield", "ema", [1, 2, 1], "ema1", "", True, "", "", 60, [-2.5, 2.5]]],
    "ryield_ema_xdemean": [BROAD_MKTS, ["ryield", "ema", [1, 2, 1], "ema1", "", True, "", "", 60, [-2.5,2.5]]],
    "ryield_st_zsa": [BROAD_MKTS, ["ryield", "zscore_adj", [20, 30, 1], "", "", True, "", "ema1", 240, [-2,2]]],
    "ryield_st_zsa_xdemean": [BROAD_MKTS, ["ryield", "zscore_adj", [20, 30, 1], "", "", True, "", "ema1", 240, [-2,2]]],
    "ryield_lt_zsa": [BROAD_MKTS, ["ryield", "zscore_adj", [80, 120, 2], "", "", True, "", "ema1", 240, [-2,2]]],
    "ryield_lt_zsa_xdemean": [BROAD_MKTS, ["ryield", "zscore_adj", [80, 120, 2], "", "", True, "", "ema1", 240, [-2,2]]],
    # "basmom5_ema": [BROAD_MKTS, ["basmom5", "ema", [10, 20, 1], "", "", True, "price", "", 240, [-2,2]]],
    # "basmom5_ema_xdemean": [BROAD_MKTS, ["basmom5", "ema", [10, 20, 1], "", "", True, "price", "", 240, [-2,2]]],
    # "basmom10_ema": [BROAD_MKTS, ["basmom10", "ema", [10, 20, 1], "", "", True, "price", "", 240, [-2,2]]],
    # "basmom10_ema_xdemean": [BROAD_MKTS, ["basmom10", "ema", [10, 20, 1], "", "", True, "price", "", 240, [-2,2]]],
    # "basmom10_qtl": [BROAD_MKTS, ["basmom10", "qtl", [230, 250, 2], "ema5", "", True, "price", "", 240, [-2,2]]],
    # "basmom10_qtl_xdemean": [BROAD_MKTS, ["basmom10", "qtl", [230, 250, 2], "ema5", "", True, "price", "", 240, [-2,2]]],
    "basmom20_ema": [BROAD_MKTS, ["basmom20", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2,2]]],
    "basmom20_ema_xdemean": [BROAD_MKTS, ["basmom20", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2,2]]],
    "basmom40_ema": [BROAD_MKTS, ["basmom40", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2, 2]]],
    "basmom40_ema_xdemean": [BROAD_MKTS, ["basmom40", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2, 2]]],
    "basmom60_ema": [BROAD_MKTS, ["basmom60", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2,2]]],
    "basmom60_ema_xdemean": [BROAD_MKTS, ["basmom60", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2,2]]],
    "basmom120_ema": [BROAD_MKTS, ["basmom120", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2,2]]],
    "basmom120_ema_xdemean": [BROAD_MKTS, ["basmom120", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240, [-2,2]]],
    "mom_ewmac": [BROAD_MKTS, ["px", "ewmac", [3, 10, 1], "", "", True, "price", "ema1", 40, [-2, 2]]],
    "mom_ewmac_xdemean": [BROAD_MKTS, ["px", "ewmac", [3, 10, 1], "", "", True, "price", "ema1", 40, [-2, 2]]],
    "mom_momma240": [BROAD_MKTS, ["px", "ema", [1, 2, 1], "df240", "pct_change", True, "price", "", 60, [-2, 2]]],
    "mom_momma240_xdemean": [BROAD_MKTS, ["px", "ema", [1, 2, 1], "df240", "pct_change", True, "price", "", 60, [-2, 2]]],
    "mom_momma20": [BROAD_MKTS, ["px", "ema", [1, 2, 1], "df20", "pct_change", True, "price", "ema1", 20, [-2, 2]]],
    "mom_momma20_xdemean": [BROAD_MKTS, ["px", "ema", [1, 2, 1], "df20", "pct_change", True, "price", "ema1", 20, [-2, 2]]],
    "mom_hlr_st": [BROAD_MKTS, ["px", "hlratio", [20, 40, 2], "", "", True, "", "ema1", 120, [-2, 2]]],
    "mom_hlr_st_xdemean": [BROAD_MKTS, ["px", "hlratio", [20, 40, 2], "", "", True, "", "ema1", 120, [-2, 2]]],
    "mom_hlr_mt": [BROAD_MKTS, ["px", "hlratio", [40, 60, 2], "", "", True, "", "ema1", 120, [-2, 2]]],
    "mom_hlr_mt_xdemean": [BROAD_MKTS, ["px", "hlratio", [40, 60, 2], "", "", True, "", "ema1", 120, [-2, 2]]],
    "mom_hlr_lt": [BROAD_MKTS, ["px", "hlratio", [80, 120, 2], "", "", True, "", "ema1", 120, [-2, 2]]],
    "mom_hlr_lt_xdemean": [BROAD_MKTS, ["px", "hlratio", [80, 120, 2], "", "", True, "", "ema1", 120, [-2, 2]]],
    #"mom_hlr_yr": [BROAD_MKTS, ["px", "hlratio", [240, 250, 2], "", "", True, "", "", 120, [-2, 2]]],
    #"mom_hlr_yr_xdemean": [BROAD_MKTS, ["px", "hlratio", [240, 250, 2], "", "", True, "", "", 120, [-2, 2]]],
    #"mom_kdj_st": [BROAD_MKTS, ["px", "kdj", [20, 40, 2], "", "", True, "", "buf0.3", 120, [-2, 2]]],
    #"drng_ma": [BROAD_MKTS, ["drng", "ma", [20, 40, 2], "", "", True, "price", "ema3", 20, [-2, 2]]],
    "cclr_mom_sgnma": [BROAD_MKTS, ["logret", "sgn_ma", [20, 40, 2], "", "", True, "price", "ema3", 20, [-2, 2]]],
    "cclr_mom_sgnma_xdemean": [BROAD_MKTS, ["logret", "sgn_ma", [20, 40, 2], "", "", True, "price", "ema3", 20, [-2, 2]]],
    "colr_mom_sgnma": [BROAD_MKTS, ["colr", "sgn_ma", [20, 40, 2], "", "", True, "price", "ema3", 20, [-2, 2]]],
    "colr_mom_sgnma_xdemean": [BROAD_MKTS, ["colr", "sgn_ma", [20, 40, 2], "", "", True, "price", "ema3", 20, [-2, 2]]],

    "nskew_245": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "skew_normal_245", "", False, "", "ema1", 60, [-2, 2]]],
    "nskew_245_xdemean": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "skew_normal_245", "", False, "", "ema1", 60, [-2, 2]]],
    "pskew_245": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "skew_pearson_245", "", False, "", "ema1", 60, [-2, 2]]],
    "pskew_245_xdemean": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "skew_pearson_245", "", False, "", "ema1", 60, [-2, 2]]],
    "bskew_245": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "skew_bowley_245", "", False, "", "ema1", 60, [-2, 2]]],
    "bskew_245_xdemean": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "skew_bowley_245", "", False, "", "ema1", 60, [-2, 2]]],
    "bconvex_81": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "bconvex_81", "", True, "", "ema1", 60, [-2, 2]]],
    "bconvex_81_xdemean": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "bconvex_81", "", True, "", "ema1", 60, [-2, 2]]],
    "bconvex_243": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "bconvex_243", "", False, "", "ema1", 60, [-2, 2]]],
    "bconvex_243_xdemean": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "bconvex_243", "", False, "", "ema1", 60, [-2, 2]]],
    "bconvex_486": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "bconvex_486", "", True, "", "ema1", 60, [-2, 2]]],
    "bconvex_486_xdemean": [BROAD_MKTS,
                 ["px", "", [1, 2, 1], "bconvex_486", "", True, "", "ema1", 60, [-2, 2]]],

    # bond future
    'bond_mr_st_qtl': [['T', 'TF'],
                       ['px', 'qtl', [2, 5, 1], 'df1', 'pct_change', False, '', "ema10|buf0.1", 120, [-2,2]]],
    'bond_tf_lt_qtl': [['T', 'TF', 'TL'],
                       ['px', 'qtl', [230, 250, 2], '', '', True, '', "buf0.1", 120, [-2,2]]],
    'bond_tf_st_eds': [['T', 'TF', 'TL', "TS"],
                       ['px', 'ema_dff_sgn', [20, 40, 2], '', '', True, '', "ema5", 120, [-2,2]]],
    'bond_carry_ma': [['T', 'TL'],
                      ['ryield', 'ma', [1, 2, 1], '', '', True, '', "", 120, [-2,2]]],
    "bond_fxbasket_zs": [['T', 'TF', 'TL', "TS"],
                    ['fxbasket_cumret', 'zscore', [480, 520, 2], '', '', True, '', '', 120, [-2,2]]],
    'bond_shibor1m_qtl': [['T', 'TF', 'TL'],
                     ['shibor_1m', 'qtl', [40, 80, 2], 'ema3', '', False, 'price', 'buf0.3', 120, [-2,2]]],
    'bond_r007_lt_zs': [['T', 'TF', 'TL'],
                     ['r007_cn', 'zscore', [40, 80, 2], 'ema5', '', False, 'price', 'buf0.4', 120, [-2,2]]],
    'bond_au_st_qtl': [['T', ],
                     ['au_td_sge', 'qtl', [20, 60, 2], 'ema1', '', True, 'price', 'buf0.1', 120, [-2,2]]],

    # ferrous
    'nmf_yoy_qtl': [['fef', 'i'], ["nmf_prem", 'qtl', [10, 20, 1], "cal_yoy", "diff", True, "", "ema1", 120, [-2,2]]],
    'macf_yoy_qtl': [['fef', 'i'], ["macf_prem", 'qtl', [10, 20, 1], "cal_yoy", "diff", True, "", "ema1", 120, [-2,2]]],
    'macf_spd_qtl': [['rb_i', "hc_i"], ["macf_prem", 'qtl', [40, 80, 2], "", "", False, "", "ema1", 120, [-2,2]]],
    'macf_spd2_qtl': [['i_rb', "i_hc"], ["macf_prem", 'qtl', [10, 30, 2], "", "", True, "", "ema3", 120, [-2,2]]],

    'nmf_arb_hlr': [['rb', 'hc', 'i', 'j', 'jm'], # macf jmb
                    ['import_arb_nmf', 'hlratio', [40, 60, 2], 'ema5', '', False, 'price', 'ema1', 120, [-2,2]]],
    'macf_arb_hlr': [['rb', 'hc', 'i', 'j', 'jm'], # macf jmb
                    ['import_arb_jmb', 'hlratio', [40, 60, 2], 'ema5', '', False, 'price', 'ema1', 120, [-2,2]]],
    'nmf_arb_spd_hlr': [['rb_i', 'hc_i'], # macf jmb
                       ['import_arb_nmf', 'hlratio', [40, 60, 2], 'ema5', '', False, 'price', 'ema1', 120, [-2,2]]],
    'jmb_arb_spd_hlr': [['rb_i', 'hc_i'], # macf jmb
                        ['import_arb_jmb', 'hlratio', [40, 60, 2], 'ema5', '', False, 'price', 'ema1', 120, [-2,2]]],

    'ioarb_px_hlr': [['rb', 'hc', 'i'],
                     ['io_on_off_arb', 'hlratio', [40, 80, 2], '', '', False, 'price', 'ema5', 120, [-2,2]]],
    'ioarb_spd_qtl_1y': [['rb_i', 'hc_i'],
                         ['io_on_off_arb', 'qtl', [240, 260, 2], 'ema1', '', False, 'price', 'sma1', 120, [-2,2]]],

    'io_mix_arb_hys2y': [['i'],
                       ['pbf_iocj_ssf_spd', 'hlratio', [480, 500, 2], "",  "", False, "", "hys_0.5_0.5", 120, [-2,2]]],
    'io_mix_arb_spd_hys2y': [['i_rb', 'i_hc'],
                           ['pbf_iocj_ssf_spd', 'hlratio', [480, 500, 2], "",  "", False, "", "hys_0.5_0.5", 120, [-2,2]]],
    'io_mix_arb_hys4y': [['i'],
                       ['pbf_iocj_ssf_spd', 'hlratio', [960, 1000, 2], "",  "", False, "", "hys_0.5_0.5", 120, [-2,2]]],
    'io_mix_arb_spd_hys4y': [['i_rb', 'i_hc'],
                           ['pbf_iocj_ssf_spd', 'hlratio', [960, 1000, 2], "",  "", False, "", "hys_0.5_0.5", 120, [-2,2]]],
    # io removal + pinv/rmv
    'io_removal_lvl': [['i'],
                       ['io_removal_45ports', 'qtl', [5, 9, 1], '', '', True, '', '', 120, [-2,2]]],
    'io_removal_lyoy': [['i'],
                        ['io_removal_45ports', 'qtl', [5, 9, 1], 'lunar_yoy_day', 'diff', True, '', '', 120, [-2,2]]],
    'io_removal_chg_hlr': [['i'],
                        ['io_removal_45ports', 'hlratio', [50, 55, 1], 'df1|ema1', 'diff', True, '', '', 120, [-2,2]]],
    'io_removal_chg_spd_hlr': [['i_rb', 'i_hc'],
                        ['io_removal_45ports', 'hlratio', [50, 55, 1], 'df1|ema1', 'diff', True, '', '', 120, [-2,2]]],
    'io_removal_lunar_hlr': [['i'],
                        ['io_removal_45ports', 'seasonal_lunar_wk_hlr', [2, 5, 10], '', '', True, '', '', 120, [-2,2]]],
    'io_removal_chg_lunar_hlr': [['i'],
                        ['io_removal_45ports', 'seasonal_lunar_wk_hlr', [2, 5, 10], 'df1|ema1', 'pct_change', True, '', '', 120, [-2,2]]],
    'io_removal_spd_lunar_hlr': [['i_rb', 'i_hc'],
                        ['io_removal_45ports', 'seasonal_lunar_wk_hlr', [2, 5, 10], '', '', True, '', '', 120, [-2,2]]],
    'io_invr_hlr_1y': [['i'],
                            ['io_inv_removal_ratio_45p', 'hlratio', [50, 55, 1], '', '', False, '', "", 120, [-2,2]]],
    'io_invr_spd_hlr_1y': [['i_rb', 'i_hc'],
                            ['io_inv_removal_ratio_45p', 'hlratio', [50, 55, 1], '', '', False, '', "", 120, [-2,2]]],
    'io_invr_chg_hlr': [['i'],
                            ['io_inv_removal_ratio_45p', 'hlratio', [50, 55, 1], 'df1', 'pct_change', False, '', "", 120, [-2,2]]],
    'io_invr_chg_spd_hlr': [['i_rb', 'i_hc'],
                            ['io_inv_removal_ratio_45p', 'hlratio', [50, 55, 1], 'df1', 'pct_change', False, '', "", 120, [-2,2]]],

    'io_millinv_lvl': [['i'],
                       ['io_inv_mill(64)', 'qtl', [20, 40, 2], '', '', True, 'price', 'sma1', 120, [-2,2]]],
    'io_minvdays_diff_ema': [['i'],
                             ['io_inv_imp_mill(247)', 'ema', [1, 2, 1], 'df1', 'pct_change', True, '', '', 26, [-2,2]]],
    'io_millinv_lyoy': [['hc', 'i'],
                        ['io_inv_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'diff', True, 'W-Thu', 'sma4', 120, [-2,2]]],
    'io_invdays_lvl': [['hc', 'i'],
                       ['io_invdays_imp_mill(64)', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price', 'sma1', 120, [-2,2]]],
    'io_invdays_lyoy': [['hc', 'i'],
                        ['io_invdays_imp_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'pct_change', True, 'W-Thu', "ema3", 120, [-2,2]]],

    'io_pinv45_lvl_hlr': [['rb_i', 'hc_i'],
                          ['io_inv_45ports', 'hlratio', [8, 56, 4], '', '', True, '', "ema1", 120, [-2,2]]],
    'ckc_pinv_spd_hlr': [['jm_rb', 'jm_hc'],
                        ['ckc_inv_6ports', 'hlratio', [40, 60, 2], '', '', False, 'price', "ema1", 120, [-2,2]]],

    'landsales_yoy_ma': [['rb', 'hc'], ['top100cities_land_supplied_area_res_all',
                                        'ma', [4, 8], 'lunar_yoy_day', 'diff', True, "", "", 52]],
    'prop2hand_px_zs': [['rb', 'hc'], ['prop_2ndhand_px_idx', 'zscore', [48, 56, 1], '', '', True, "", "", 120, [-2,2]]],
    #stl sales
    'rbsales_lyoy_mom_st': [['rb'],
                            ['consteel_dsales_mysteel', 'hlratio', [40, 80, 2],
                             'ema1|lunar_yoy_day', 'diff', True, 'price', "buf0.1", 120, [-2,2]]],
    'rbsales_lyoy_mom_lt': [['rb'],
                            ['consteel_dsales_mysteel', 'hlratio', [240, 260, 2],
                             'ema1|lunar_yoy_day', 'diff', True, 'price', "buf0.1", 120, [-2,2]]],
    'rb_sales_inv_ratio_lyoy': [['rb'],
                                ['rebar_sales_inv_ratio', 'hlratio', [20, 60, 2],
                                 'ema3|lunar_yoy_day', 'diff', True, 'price', "ema1", 120, [-2,2]]],
    'rbsales_spd_lunar_qtl': [['rb_i'],
                            ['consteel_dsales_mysteel', 'seasonal_lunar_wk_qtl', [2, 5, 30],
                             'ema3', '', True, 'price', "", 120, [-2,2]]],
    "stl_dmd_lunar_zs": [['hc_i', 'hc'],
                        ['steel_demand', 'seasonal_lunar_wk_zs', [2, 5, 10], '', '', True, '', "", 120, [-2,2]]],
    "stl_dmd_cal_zs": [['hc_i', 'hc'],
                        ['steel_demand', 'seasonal_cal_wk_zs', [2, 5, 10], '', '', True, '', "", 120, [-2,2]]],
    'iosales_lyoy_ema': [['i'],
                         ['io_trdvol_davg_majorports', 'ema', [1, 2, 1],
                          'lunar_yoy_day', 'diff', True, '', "", 20, [-2,2]]],
    'fef_phycarry_ema': [['fef', 'i'],
                         ['FEF_phycarry', 'ema', [1, 2, 1], '', '', True, 'price', "ema1", 240, [-2,2]]],
    'rb_phycarry_ema': [['fef', 'rb', 'i', 'j', 'jm'],
                        ['rb_phycarry', 'ema', [10, 20, 1], '', '', True, '', "sma3", 120, [-2,2]]],
    'hc_phycarry_ema': [['fef', 'hc', 'i', 'j', 'jm'],
                        ['hc_phycarry', 'ema', [10, 20, 1], '', '', True, '', "sma3", 120, [-2,2]]],
    'ckc_phycarry_ema': [['fef', 'i'],
                         ['jm_phycarry', 'ema', [10, 20, 1], '', '', True, '', "sma3", 120, [-2,2]]],
    'fef_c1_c2_ratio_or_qtl': [['rb', 'hc', 'j'],
                               ['FEF_c1_c2_ratio', 'hlratio', [40, 60, 2], '', '', False, '', "buf0.2", 120, [-2,2]]],
    'fef_c1_c2_ratio_spd_qtl': [['rb_i', 'hc_i', 'j_i'],
                                ['FEF_c1_c2_ratio', 'qtl', [20, 40, 2], '', '', False, '', "buf0.15", 120, [-2,2]]],
    'fef_fly_ratio_or_qtl': [['rb', 'hc', 'j'],
                             ['FEF_c123fly_ratio', 'hlratio', [40, 60, 2], 'ema1', '', False, '', "buf0.2", 120, [-2,2]]],
    # 'fef_fly_ratio_spd_qtl': [['rb_i', 'hc_i'],
    #                           ['FEF_c123_fly_ratio', 'qtl', [40, 60, 2], '', '', False, '', "", 120, [-2,2]]],
    'fef_ryieldmom_or_zs': [['rb', 'hc', 'j'],
                            ['FEF_ryield', 'zscore', [10, 30, 2], 'ema3', '', False, '', "buf0.25", 120, [-2,2]]],
    'fef_ryieldmom_spd_zs': [['rb_i', 'hc_i', 'j_i'],
                             ['FEF_ryield', 'zscore', [10, 30, 2], 'ema3', '', False, '', "buf0.25", 120, [-2,2]]],
    'fef_basmom_or_qtl': [['rb', 'hc'],
                          ['FEF_basmom', 'qtl', [40, 60, 2], 'ema20', '', False, 'price', "ema1", 120, [-2,2]]],
    'fef_basmom_spd_qtl': [['rb_i', 'hc_i', 'j_i'],
                          ['FEF_basmom', 'qtl', [40, 60, 2], 'ema20', '', False, 'price', "ema1", 120, [-2,2]]],
    'fef_basmom5_or_qtl': [['rb', 'hc'],
                          ['FEF_basmom5', 'qtl', [40, 60, 2], 'ema5', '', False, 'price', "", 120, [-2,2]]],
    'fef_basmom5_spd_qtl': [['rb_i', 'hc_i', 'j_i'],
                           ['FEF_basmom5', 'qtl', [40, 60, 2], 'ema5', '', False, 'price', "", 120, [-2,2]]],
    'fef_basmom5_spd_ema': [['rb_i', 'hc_i', 'j_i'],
                           ['FEF_basmom5', 'ema', [1, 2, 1], 'ema5', '', False, 'price', "ema1", 120, [-2,2]]],

    'scrap_arr_eaf_spd_lyoy_mds': [
        ['hc_i'], ['scrap_arrival_mill_49eaf', 'ma_dff_sgn', [30, 50, 2], 'lunar_yoy_day', 'diff', True, 'price', 'ema3', 120, [-2, 2]]],
    'scrap_arr_bof_spd_lyoy_mds': [
        ['hc_i'], ['scrap_arrival_mill_70bof', 'ma_dff_sgn', [30, 50, 2], 'lunar_yoy_day', 'diff', True, 'price', 'ema3', 120, [-2, 2]]],
    'scrap_inv_use_ratio_spd_lyoy_mds': [
        ['hc_i'], ['scrap_inv_use_total', 'ma_dff_sgn', [4, 8, 1], 'lunar_yoy_day', 'diff', False, '', 'ema5', 120, [-2, 2]]],
    'rebar_bof_eaf_diff_ema': [
        ['i_rb', 'i_hc'], ['rebar_bof_eaf_diff', 'ema', [1, 2, 1], '', '', False, 'price', 'ema1', 60, [-2, 2]]],

    'steel_sinv_spd2_hlr_1m': [['i_rb', 'i_hc'],
                               ['steel_social_inv', 'hlratio', [4, 8, 1], '', '', False, '', "ema1", 120, [-2,2]]],
    'steel_sinv_spd2_hlr_3m': [['i_rb', 'i_hc'],
                               ['steel_social_inv', 'hlratio', [12, 16, 1], '', '', False, '', "ema1", 120, [-2,2]]],
    'steel_total_stu_lyoy_hlr_st': [['rb', 'hc', 'i'],
                                    ['steel_total_stockdays', 'hlratio', [12, 24, 1], 'lunar_yoy_day', 'diff', False, '', "ema1", 120, [-2,2]]],
    'steel_total_stu_hlr_st': [['rb', 'hc', 'i'],
                               ['steel_total_stockdays', 'hlratio', [12, 24, 1], '', '', False, '', "ema1", 120, [-2,2]]],
    'steel_total_stu_lyoy_hlr_yr': [['rb', 'hc', 'i'],
                                    ['steel_total_stockdays', 'hlratio', [48, 56, 1], 'lunar_yoy_day', 'diff', False, '', "ema1", 120, [-2,2]]],
    'steel_total_stu_hlr_yr': [['rb', 'hc', 'i'],
                               ['steel_total_stockdays', 'hlratio', [48, 56, 1], '', '', False, '', "ema1", 120, [-2,2]]],

    'steel_sinv_lyoy_zs': [['rb', 'hc', 'i', 'FG', 'v'],
                           ['steel_inv_social', 'zscore', [24, 30, 2], 'lunar_yoy_day', 'diff', False, '', "sma1", 120, [-2,2]]],
    'steel_sinv_lyoy_mds': [['rb', 'hc', 'i', 'FG', 'v'],
                           ['steel_inv_social', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, '', "sma1", 120, [-2,2]]],
    'billet_inv_hlr_lt': [['rb', 'hc', 'i'],
                          ['billet_inv_social_ts', 'hlratio', [100, 110, 1], '', '', False, '', '', 120, [-2,2]]],
    'billet_inv_lyoy_hlr_lt': [['rb', 'hc', 'i'],
                          ['billet_inv_social_ts', 'hlratio', [50, 55, 1], 'lunar_yoy_day', 'diff', False, '', '', 120, [-2,2]]],
    'hrc_arb_mom': [['hc', 'rb'],
                    ['hrc_exp_sea_arb', 'qtl', [40, 80, 2], '', '', True, '', "ema1", 120, [-2,2]]],
    'hrc_mb_saudi_qtl': [['hc'],
                         ['hrc_mb_cn_saudi', 'qtl', [30, 40, 2], '', '', True, '', "ema1", 120, [-2,2]]],
    'hrc_mb_uae_qtl': [['hc'],
                       ['hrc_mb_cn_uae', 'qtl', [20, 30, 2], '', '', True, '', "ema1", 120, [-2,2]]],
    'hrc_cn_eu_qtl': [['hc'],
                      ['hrc_cn_eu_cfr', 'qtl', [20, 30, 2], '', '', True, '', "ema1", 120, [-2,2]]],
    'hrc_arb_ma': [['hc', 'rb'],
                   ['hrc_exp_sea_arb', 'ma', [1, 4, 1], '', '', True, '', "ema1", 120, [-2,2]]],

    'steel_margin_lvl_fast': [['i', 'j'],
                              ['margin_hrc_macf', 'hlratio', [20, 40, 2], '', '', True, 'price', "buf0.2", 120, [-2,2]]],
    'steel_margin_lvl_slow': [['SM', 'SF'],
                              ['margin_hrc_macf', 'hlratio', [240, 260, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'steel_margin_io_rev': [['i'],
                            ['margin_hrc_macf', 'hlratio', [240, 260, 2], '', '', True, 'price', "hys_0.5_0.5|pos", 120, [-2,2]]],
    'mn_mine_mom': [['SM'],
                    ['mn_44_gabon_tj', 'zscore', [40, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'smsf_prodcost_mom': [['SM', 'SF'],
                          ['smsf_prodcost', 'zscore', [40, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'smsf_prodcost_mom_xdemean': [['SM', 'SF'],
                          ['smsf_prodcost', 'zscore', [40, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'smsf_workrate_ratio': [['SM-SF'],
                            ['smsf_workrate_ratio', 'hlratio', [20, 40, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'smsf_dmd_ratio': [['SM-SF'],
                       ['smsf_dmd_ratio', 'hlratio', [120, 160, 2], '', '', True, 'price', "ema5", 120, [-2,2]]],
    'smsf_margin_diff': [['SM-SF'],
                         ['smsf_margin_diff', 'zscore', [40, 60, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'coal_mom_spd_st': [['SF_SM', 'j_i', 'jm_i'],
                         ['coal_5500_sx_qhd', 'zscore', [20, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'coal_mom_spd_yr': [['SF_SM', 'j_i', 'jm_i'],
                         ['coal_5500_sx_qhd', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'coal_mom_st_hlr': [['SF', 'j', 'jm'],
                        ['coal_5500_sx_qhd', 'hlratio', [40, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'coal_mom_yr_hlr': [['SF', 'j', 'jm'],
                        ['coal_5500_sx_qhd', 'hlratio', [240, 260, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'ferr_pinv_hlr_mt': [['j', 'jm', 'i', 'rb', 'hc'],
                        ['fer_pinv', 'hlratio', [80, 120, 2], '', '', False, 'price', "ema1", 120, [-2,2]]],
    'ferr_pinv_hlr_mt_xdemean': [['j', 'jm', 'i', 'rb', 'hc'],
                        ['fer_pinv', 'hlratio', [80, 120, 2], '', '', False, 'price', "ema1", 120, [-2,2]]],
    'ferr_pinv_hlr_yr': [['j', 'jm', 'i', 'rb', 'hc'],
                        ['fer_pinv', 'hlratio', [240, 260, 2], '', '', False, 'price', "ema1", 120, [-2,2]]],
    'ferr_pinv_lyoy_hlr_mt': [['j', 'jm', 'i', 'rb', 'hc'],
                        ['fer_pinv', 'hlratio', [40, 60, 2], 'lunar_yoy_day', 'diff', False, 'price', "ema1", 120, [-2,2]]],

    'strip_hsec_lvl_mid': [['rb', 'hc', 'i', 'j', 'jm'],
                           ['strip_hsec', 'qtl', [60, 80, 2], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'macf_cfd_lvl_mid': [['i'],
                         ['macf_cfd', 'qtl', [40, 82, 2], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'hc_rb_diff_20': [['rb', 'hc', 'i', 'j', 'jm', 'FG', 'v', 'au', 'ag', 'cu', 'al', 'zn', 'sn', 'ss', 'ni'],
                      ['hc_rb_diff', 'zscore', [20, 40, 2], '', '', True, 'price', "buf0.15", 120, [-2,2]]],

    'rbhc_px_diff_mds': [['spd_hc_rb_c1', 'hc_rb'],
                     ['rb_hc_diff', 'ma_dff_sgn', [5, 10, 1], 'ema1', '', False, '', "", 120, [-2,2]]],
    'rbhc_px_diff_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                          ['rb_hc_diff', 'ma_dff_sgn', [5, 10, 1], 'ema1|lunar_yoy_day', 'diff', False, '', "", 120, [-2,2]]],
    'rbhc_phycarry_diff_zs': [['spd_hc_rb_c1', 'hc_rb'],
                     ['rb_hc_phycarry_diff', 'zscore', [10, 20, 1], 'ema1', '', False, '', "buf0.4", 120, [-2,2]]],
    'rbhc_basmom_diff_hlr': [['spd_hc_rb_c1', 'hc_rb'],
                          ['rb_hc_basmom_diff', 'hlratio', [40, 60, 1], 'ema1', '', False, '', "ema1|buf0.25", 120, [-2,2]]],
    'rbhc_steel_spd_mds': [['spd_hc_rb_c1', 'hc_rb'],
                     ['rb_hc_steel_spd', 'ma_dff_sgn', [10, 30, 1], '', '', False, '', "buf0.25", 120, [-2,2]]],
    'rbhc_steel_spd_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                          ['rb_hc_steel_spd', 'ma_dff_sgn', [10, 30, 1], 'lunar_yoy_day', 'diff', False, '', "buf0.25", 120, [-2,2]]],
    'rbhc_dmd_ratio_mds': [['spd_hc_rb_c1', 'hc_rb'],
                     ['rb_hc_dmd_ratio', 'ma_dff_sgn', [5, 9, 1], '', '', False, '', "", 120, [-2,2]]],
    'rbhc_dmd_ratio_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                          ['rb_hc_dmd_ratio', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, '', "", 120, [-2,2]]],
    'rbhc_sinv_chg_mds': [['spd_hc_rb_c1', 'hc_rb'],
                          ['rb_hc_sinv_chg_diff', 'ma_dff_sgn', [5, 9, 1], '', '', True, '', "", 120, [-2,2]]],
    'rbhc_sinv_chg_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                               ['rb_hc_sinv_chg_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', True, '', "", 120, [-2,2]]],
    'rbhc_sinv_lratio_mds': [['spd_hc_rb_c1', 'hc_rb'],
                          ['rb_hc_sinv_lratio', 'ma_dff_sgn', [2, 4, 1], '', '', True, '', "", 120, [-2,2]]],
    'rbhc_sinv_lratio_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                               ['rb_hc_sinv_lratio', 'ma_dff_sgn', [2, 4, 1], 'lunar_yoy_day', 'diff', True, '', "", 120, [-2,2]]],
    'rbhc_rbsales_lyoy_zs': [['spd_hc_rb_c1', 'hc_rb'],
                            ['consteel_dsales_mysteel', 'zscore', [20, 30, 2],
                             'lunar_yoy_day|ema1', 'diff', False, 'price', "ema2|buf0.3", 120, [-2,2]]],
    'rbhc_eaf_util_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                               ["eaf_util_all", 'ma_dff_sgn', [20, 52, 1], 'lunar_yoy_day', 'diff', True, '', "", 120, [-2,2]]],
    'rbhc_scrap_use_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                                ["scrap_use_mill_49eaf", 'ma_dff_sgn', [16, 28, 1], 'lunar_yoy_day', 'diff', True, '', "", 120, [-2,2]]],
    'rbhc_scrap_inv_use_lyoy_mds': [['spd_hc_rb_c1', 'hc_rb'],
                                    ["scrap_inv_use_49eaf", 'ma_dff_sgn', [16, 28, 1], 'lunar_yoy_day', 'diff', False, '', "", 120, [-2,2]]],

    # 'margin_sea_lvl_mid': ('hrc_margin_sb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),

    'cu_prem_usd_zsa': [['cu'],
                        ['cu_prem_bonded_warrant', 'zscore', [20, 30, 2], '', '', True, 'price', "ema1", 120, [-2,2]]],
    'cu_prem_usd_md': [['cu'],
                       ['cu_prem_bonded_warrant', 'ma_dff', [20, 30, 2], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'cu_phybasis_zsa': [['cu'],
                        ['cu_cjb_phybasis', 'zscore_adj', [40, 60, 2], 'sma10', '', True, 'price', "sma1", 120, [-2,2]]],  # great
    'cu_phybasis_hlr': [['cu'],
                        ['cu_cjb_phybasis', 'hlratio', [40, 60, 2], 'sma10', '', True, 'price', "sma1", 120, [-2,2]]],  # great
    'cu_scrap1_zs': [['cu'],
                     ['cu_1#_scrap_diff', 'zscore', [40, 60, 2], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'cu_mine_tc_zs': [['cu'],
                      ['cu_mine_tc_cif_cn', 'zscore', [20, 30, 2], '', '', False, 'price', "sma1", 120, [-2,2]]],
    'cu_blister_rc_s_zs': [['cu'],
                      ['cu_blister_rc_south', 'zscore', [230, 250, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'cu_rod_procfee_for_qtl': [['cu'],
                      ['cu_rod_lowoxygen_8mm_procfee_for_gd', 'qtl', [20, 30, 2], '', '', True, 'price', "sma1", 120, [-2,2]]],
    # 'cu_scrap1_margin_gd': [['cu'],
    #                         ['cu_scrap1_diff_gd', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price', "", 120, [-2,2]]],
    # 'cu_scrap1_margin_tj': [['cu'],
    #                         ['cu_scrap1_diff_tj', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price', "", 120, [-2,2]]],
    # 'cu_rod_procfee_2.6': [['cu'], ['cu_rod_2.6_procfee_nanchu',
    #                                 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price', "", 120, [-2,2]]],
    # 'cu_rod_procfee_8.0': [['cu'], ['cu_rod_8_procfee_nanchu',
    #                                 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price', "", 120, [-2,2]]],
    'ni_nis_mom_qtl': [['ni'], ['ni_nis_cjb_spot', 'qtl', [10, 20, 2], '', '', True, '', "sma1", 120, [-2,2]]],
    'ni_ore_qtl': [['ni'], ['ni_1.8conc_spot_php_lianyungang', 'qtl', [10, 20], '', '', True, '', "sma1", 120, [-2,2]]],
    'al_alumina_qtl': [['al'],
                       ["alumina_spot_qd", 'qtl', [10, 20], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'al_alumina_yoy_qtl': [['al'],
                           ["alumina_spot_qd", 'qtl', [20, 40], 'cal_yoy_day', 'pct_change', True, 'price', "sma1", 120, [-2,2]]],
    'al_coal_qtl': [['al'],
                    ['coal_5500_sx_qhd', 'qtl', [240, 250], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'al_coal_yoy_qtl': [['al'],
                        ['coal_5500_sx_qhd', 'qtl', [20, 40], 'cal_yoy_day', 'pct_change', True, 'price', "sma1", 120, [-2,2]]],
    'sn_conc_spot_hlr': [['sn'], ['sn_60conc_spot_guangxi', 'hlratio', [40, 80, 2], '', '', True, '', "sma1", 120, [-2,2]]],

    'al_scrap_sh_zs': [['al'],
                       ['al_scrap_ex_diff_sh', 'zscore', [20, 30, 2], '', '', True, 'price', "sma1", 120, [-2,2]]],
    'zn_hrc_mom': [['zn'], ['hrc_sh', 'qtl', [20, 30], '', '', True, 'price', "", 120, [-2,2]]],
    'pb_sec_margin_zs': [['pb'], ['pb_sec_margin', 'zscore', [230, 250, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'ni_npi_cn_zs': [['ni'], ['npi_10_spot_cn', 'zscore', [40, 80, 2], '', '', True, '', "", 120, [-2,2]]],
    'ni_nsa_zs': [['ni'], ['ni_sul_spot', 'zscore', [40, 80, 2], '', '', True, '', "", 120, [-2,2]]],
    'ni_briq_qtl': [['ni'], ['ni_briq_prem_spot', 'qtl', [40, 60, 2], '', '', True, '', "", 120, [-2,2]]],
    'ni_npi_prem_hlr': [['ni'], ['ni_ni_prem', 'hlratio', [20, 60, 2], 'ema3', '', False, '', "", 120, [-2,2]]],
    'ni_npi_imp_spd_hlr': [['ni'], ['npi_ferronickle_spd', 'hlratio', [20, 60, 2], '', '', False, 'price', "", 120, [-2,2]]],

    'lme_base_ts_mds': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_base_ts', 'ma_dff_sgn', [10, 30, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'lme_base_ts_mds_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_base_ts', 'ma_dff_sgn', [10, 30, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'lme_base_ts_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_base_ts', 'hlratio', [10, 20, 1], '', '', True, 'price', "buf0.1", 120, [-2,2]]],
    'lme_base_ts_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_base_ts', 'hlratio', [10, 20, 1], '', '', True, 'price', "buf0.1", 120, [-2,2]]],
    'lme_futbasis_ma': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_futbasis', 'ma', [1, 2, 1], 'df1|ema1', 'diff', True, '', "buf0.5", 120, [-2,2]]],
    'lme_futbasis_ma_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_futbasis', 'ma', [1, 2, 1], 'df1|ema1', 'diff', True, '', "buf0.5", 120, [-2,2]]],
    'base_phybas_carry_ma': [['cu', 'al', 'zn', 'ni', 'sn'],
                             ['base_phybas', 'ma', [1, 2], 'sma2', '', True, 'price', "", 120, [-2,2]]],
    'base_phybas_carry_ma_xdemean': [['cu', 'al', 'zn', 'ni', 'sn'],
                                     ['base_phybas', 'ma', [1, 2], 'sma2', '', True, 'price', "", 120, [-2,2]]],
    'base_phybasmom_1m_zs': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                             ['base_phybas', 'zscore', [20, 30], 'sma2', '', True, 'price', "", 120, [-2,2]]],
    'base_phybasmom_1m_zs_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                     ['base_phybas', 'zscore', [20, 30], 'sma2', '', True, 'price', "", 120, [-2,2]]],
    'base_phybasmom_1y_zs': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                             ['base_phybas', 'zscore', [230, 250, 2], 'sma2', '', True, 'price', "", 120, [-2,2]]],
    'base_phybasmom_1y_zs_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                     ['base_phybas', 'zscore', [230, 250, 2], 'sma2', '', True, 'price', "", 120, [-2,2]]],
    'base_cifprem_1m_zs': [['cu', 'al', 'zn', 'ni', 'pb'],
                           ['prem_bonded_warrant', 'zscore_adj', [20, 30, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'base_cifprem_1y_zs': [['cu', 'al', 'zn', 'ni', 'pb'],
                           ['prem_bonded_warrant', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'base_cifprem_1y_zs_xdemean': [['cu', 'al', 'zn', 'ni', 'pb'],
                                   ['prem_bonded_warrant', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'base_tc_1y_zs': [['cu', 'pb', 'zn'], ['base_tc', 'zscore', [230, 250, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'base_tc_2y_zs': [['cu', 'pb', 'sn'], ['base_tc', 'zscore', [480, 500, 2], '', '', False, 'price', "", 120, [-2,2]]],

    'base_inv_mds': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                     ['base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price', "sma1", 120, [-2,2]]],
    'base_inv_mds_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                             ['base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price', "sma1", 120, [-2,2]]],
    'base_sinv_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
                     ['base_inv', 'hlratio', [240, 260, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'base_sinv_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
                             ['base_inv', 'hlratio', [240, 260, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'base_sinv_lunar_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
                     ['base_inv', 'seasonal_lunar_wk_hlr', [2, 6, 20], '', '', False, 'price', "", 120, [-2,2]]],
    'base_sinv_lunar_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
                     ['base_inv', 'seasonal_lunar_wk_hlr', [2, 6, 20], '', '', False, 'price', "", 120, [-2,2]]],
    'base_sinvchg_lunar_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
                     ['base_inv', 'seasonal_lunar_wk_hlr', [2, 6, 20], 'df5', 'diff', False, 'price', "", 120, [-2,2]]],
    'base_sinvchg_lunar_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
                     ['base_inv', 'seasonal_lunar_wk_hlr', [2, 6, 20], 'df5', 'diff', False, 'price', "", 120, [-2,2]]],

    'base_inv_shfe_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                         ['inv_shfe_d', 'hlratio', [240, 260, 2], '', '', False, '', "", 240, [-2,2]]],
    'base_inv_shfe_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                                 ['inv_shfe_d', 'hlratio', [240, 260, 2], '', '', False, '', "", 240, [-2,2]]],
    'base_invchg_shfe_mad': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                         ['inv_shfe_d', 'mad', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240, [-2,2]]],
    'base_invchg_shfe_mad_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                                 ['inv_shfe_d', 'mad', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240, [-2,2]]],
    'base_invchg_shfe_lunar_qtl': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                         ['inv_shfe_d', 'seasonal_lunar_wk_qtl', [2, 5, 30], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240, [-2,2]]],
    'base_invchg_shfe_lunar_qtl_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ao', 'ss'],
                                 ['inv_shfe_d', 'seasonal_lunar_wk_qtl', [2, 5, 30], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240, [-2,2]]],
    'base_inv_exch_ma': [['cu', 'zn', 'pb', 'ni', 'sn', 'al', 'ao', 'ss'],
                         ['inv_exch_d', 'mad', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240, [-2,2]]],
    'base_inv_exch_ma_xdemean': [['cu', 'zn', 'pb', 'ni', 'sn', 'al', 'ao', 'ss'],
                                 ['inv_exch_d', 'mad', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240, [-2,2]]],

    'lme_inv_wnt_mad': [['cu', 'zn', 'ni', 'sn', 'al'],
                         ['inv_lme_wnt', 'mad', [1, 2, 1], 'df1|ema1', 'pct_change', False, 'price', "buf0.2", 240, [-2,2]]],
    'lme_inv_wnt_mad_xdemean': [['cu', 'zn', 'ni', 'sn', 'al'],
                         ['inv_lme_wnt', 'mad', [1, 2, 1], 'df1|ema1', 'pct_change', False, 'price', "buf0.2", 240, [-2,2]]],
    'lme_inv_wnt_cal_qtl': [['cu', 'zn', 'ni', 'sn', 'al'],
                         ['inv_lme_wnt', 'seasonal_cal_wk_qtl', [2, 5, 30], 'df1|ema1', 'pct_change', False, 'price', "buf0.1", 240, [-2,2]]],

    'lme_cancel_ratio_mad': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI', 'cu', 'al', 'zn'],
                         ['lme_cancel_ratio', 'mad', [1, 2, 1], 'df1|ema1', 'diff', True, 'price', "", 120, [-2,2]]],
    'lme_cancel_ratio_cal_qtl': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI', 'cu', 'al', 'zn'],
                         ['lme_cancel_ratio', 'seasonal_cal_wk_qtl', [2, 5, 30], 'df1|ema1', 'diff', True, 'price', "", 120, [-2,2]]],

    # 'lme_mr_qtl_xdemean': [['MCU', 'MAL', 'MZN', 'MPB'],
    #                       ['px', 'qtl', [20, 40, 2], '', '', False, '', "ema5", 240, [-2,2]]],
    # 'lme_mr_zs_xdemean': [['MCU', 'MAL', 'MZN', 'MPB'],
    #                       ['px', 'zscore', [20, 160, 5], '', '', False, '', "ema4", 240, [-2,2]]],
    # 'lmebase_arb_zsa_1m': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
    #                        ['shfe_arb', 'zscore_adj', [20, 40, 1], 'ema3', '', True, 'price', "", 120, [-2,2]]],
    # 'lmebase_arb_zsa_1m_xdemean': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
    #                                ['shfe_arb', 'zscore_adj', [20, 40, 1], 'ema3', '', True, 'price', "", 120, [-2,2]]],
    # 'lmebase_arb_ma': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
    #                    ['shfe_arb', 'ma', [1, 2], '', '', True, 'price', "", 240, [-2,2]]],
    # 'lmebase_arb_ma_xdemean': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
    #                            ['shfe_arb', 'ma', [1, 2], '', '', True, 'price', "", 240, [-2,2]]],
    # 'lmebase_long_2y': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
    #                     ['shfe_arb', 'hysteresis', [0.9, 500, 0.1], '', 'pct_score', True, 'price', "pos", 240, [-2,2]]],

    # metals misc
    "lc_sam_fob_mom": [['lc'], ["lc_bat_sam_fob", 'zscore', [60, 80, 2], '', '', True, 'price', "", 120, [-2,2]]],
    "lc_mine_mom": [['lc'], ["lc_li2Omine_6pct_cif", 'zscore', [60, 80, 2], '', '', True, 'price', "", 120, [-2,2]]],
    "lc_sam_asia_arb_qtl": [['lc'], ["lc_sam_asia_arb", 'qtl', [60, 100, 2], '', '', True, 'price', "", 120, [-2,2]]],
    "SH_margin_mom": [['SH'], ['SH_margin_sd', 'qtl', [40, 80, 2], '', '', False, 'price', "", 120, [-2,2]]],
    "SH_50_32_spd_mom": [['SH'], ['SH_50_32_spd', 'zscore', [20, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],

    # metal group
    'metal_pbc_ema': [['i', 'rb', 'hc', 'jm', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
                       'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'au', 'ag'],
                      ['phycarry', 'ema', [5, 10], '', '', True, 'price', "", 120, [-2,2]]],
    'metal_pbc_ema_xdemean': [['i', 'rb', 'hc', 'jm', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
                               'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'au', 'ag'],
                              ['phycarry', 'ema', [5, 10], '', '', True, 'price', "", 120, [-2,2]]],
    'metal_mom_hlrhys': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v',
                          'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'au', 'ag'],
                         ['metal_px', 'hysteresis', [0.7, 60, 0.1], '', 'hlratio', True, 'price', "ema1", 120, [-2,2]]],
    'metal_mom_hlrhys_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v',
                                  'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'si', 'lc', 'au', 'ag'],
                         ['metal_px', 'hysteresis', [0.7, 60, 0.1], '', 'hlratio', True, 'price', "ema1", 120, [-2,2]]],
    'mtlmom_st_hlr': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'hlratio', [20, 60, 2], '', '', True, 'price', "ema1", 120, [-2,2]]],
    'mtlmom_st_hlr_xdemean': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'hlratio', [20, 60, 2], '', '', True, 'price', "ema1", 120, [-2,2]]],
    'mtlmom_lt_hlr': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'hlratio', [240, 260, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'mtlmom_lt_hlr_xdemean': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'hlratio', [240, 260, 2], '', '', True, 'price', "", 120, [-2,2]]],
    'mtlmom_regt_mt': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'ema', [1, 2, 1], 'regt40', '', True, 'price', "", 60, [-2,2]]],
    'mtlmom_regt_mt_xdemean': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'ema', [1, 2, 1], 'regt40', '', True, 'price', "", 60, [-2,2]]],
    'mtlmom_regt_lt': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'ema', [1, 2, 1], 'regt240', '', True, 'price', "", 60, [-2,2]]],
    'mtlmom_regt_lt_xdemean': [[
        'i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc', 'ps', 'au', 'ag'],
        ['px', 'ema', [1, 2, 1], 'regt240', '', True, 'price', "", 60, [-2,2]]],

    'metal_inv_hlr': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'hlratio', [240, 250, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'metal_inv_hlr_xdemean': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'hlratio', [240, 250, 2], '', '', False, 'price', "", 120, [-2,2]]],
    'metal_inv_lyoy_ma': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'ma', [1, 2, 1], 'lunar_yoy_day', 'diff', False, 'price', "", 120, [-2,2]]],
    'metal_inv_lyoy_ma_xdemean': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'ma', [1, 2, 1], 'lunar_yoy_day', 'diff', False, 'price', "", 120, [-2,2]]],
    'metal_invchg_ma': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'ma', [1, 2, 1], 'df5', 'diff', False, 'price', "", 120, [-2,2]]],
    'metal_invchg_ma_xdemean': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'ma', [1, 2, 1], 'df5', 'diff', False, 'price', "", 120, [-2,2]]],
    'metal_inv_seasonal_lunar': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'seasonal_lunar_wk_hlr', [2, 10, 40], '', '', False, 'price', "", 120, [-2,2]]],
    'metal_inv_seasonal_lunar_xdemean': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'seasonal_lunar_wk_hlr', [2, 10, 40], '', '', False, 'price', "", 120, [-2,2]]],
    'metal_invchg_seasonal_lunar': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'seasonal_lunar_wk_qtl', [2, 10, 40], 'df5', 'diff', False, 'price', "", 120, [-2,2]]],
    'metal_invchg_seasonal_lunar_xdemean': [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ['metal_inv', 'seasonal_lunar_wk_qtl', [2, 10, 40], 'df5', 'diff', False, 'price', "", 120, [-2,2]]],

    "momhys": [
        ['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'SA', 'v', 'SH',
         'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao', 'si', 'lc'],
        ["px", "hysteresis", [0.7, 20, 0.5], "ema3", "zscore_roll", True, "", "", 120, [-2,2]]],

    "base_etf_mom_zsa": [['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
                         ["base_sw_csi500_ret", "zscore_adj", [20, 40, 1], "csum", "", True, "", "", 120, [-2,2]]],
    "base_etf_mom_ewm": [['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
                         ["base_sw_csi500_ret", "ewmac", [2, 5, 1], "csum", "", True, "", "", 0, [-3,3]]],
    "const_etf_mom_zsa": [['rb', 'i', 'j', 'v', 'FG'],
                          ["const_sw_csi500_ret", "zscore_adj", [40, 80, 2], "csum", "", True, "", "", 120, [-2,2]]],
    "const_etf_mom_ewm": [['rb', 'i', 'j', 'v', 'FG'],
                          ["const_sw_csi500_ret", "ewmac", [2, 5, 1], "csum", "", True, "", "", 0, [-3,3]]],

    "prop_etf_mom_dbth_zs": [['rb', 'i', 'v', 'FG'],
                             ["prop_sw_csi500_ret", "hysteresis", [1, 120, 0.5], "ema3", "zscore_roll", True, "", "", 120, [-2,2]]],
    "prop_etf_mom_dbth_qtl": [['rb', 'i', 'v', 'FG'],
                              ["prop_sw_csi500_ret", "dbl_th", [0.75, 120, 0], "ema3", "pct_score", True, "", "", 120, [-2,2]]],
    "prop_etf_mom_dbth_qtl2": [['rb', 'i', 'v', 'FG'],
                               ["prop_sw_csi500_ret", "dbl_th", [0.8, 240, 1], "ema3", "pct_score", True, "", "", 120, [-2,2]]],
    "glass_etf_mom_dbth_zs": [['FG'],
                             ["glass_sw_csi500_ret", "hysteresis", [1.2, 120, 0.6], "ema3", "zscore_roll", True, "", "", 120, [-2,2]]],
    # "rubber_etf_mom_dbth_zs": [['ru', 'nr'],
    #                          ["rubber_sw_csi500_ret", "hysteresis", [1.5, 60, 1], "ema3", "zscore_roll", True, "", "", 120, [-2,2]]],
    "us_oil_prod_etf_mom": [['sc', 'bu', 'TA',],
                            ["us_oil_prod_etf_perf", "qtl", [40, 60, 2], '', '', True, 'price', "", 120, [-2,2]]],

    "exch_wnt_hlr": [
        ['rb', 'hc', "UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [230, 250, 2], "", "", False, "", "", 240, [-2,2]]],
    "exch_wnt_hlr_xdemean": [
        ['rb', 'hc', "UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [240, 260, 2], "", "", False, "", "", 240, [-2,2]]],
    "exch_wnt_yoy_hlr": [
        ['rb', 'hc', "UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [230, 250, 2], 'cal_yoy_day', "diff", False, "", "ema3", 240, [-2,2]]],
    "exch_wnt_yoy_hlr_xdemean": [
        ['rb', 'hc', "UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [240, 260, 2], 'cal_yoy_day', "diff", False, "", "ema3", 240, [-2,2]]],
    "exch_wnt_kdj": [
        ['rb', 'hc', "UR", "ru", 'si', 'lc', 'ao', 'ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "kdj", [230, 250, 2], "", "", False, "", "ema1", 240, [-2,2]]],

    "cgb_1_2_spd_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                       ["cgb_1_2_spd", "zscore", [40, 80, 2], "", "", True, "", "ema3", 120, [-2,2]]],
    "cgb_2_5_spd_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                       ["cgb_2_5_spd", "zscore", [40, 80, 2], "", "", True, "", "ema3", 120, [-2,2]]],
    "cgb_1_5_spd_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                       ["cgb_1_5_spd", "zscore", [40, 80, 2], "", "", True, "", "ema3", 120, [-2,2]]],

    "fxbasket_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA',],
                    ['fxbasket_cumret', 'zscore', [20, 40, 2], '', '', False, '', 'buf0.2', 120, [-2,2]]],
    'dxy_zsa_s': [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                  ['dxy', 'zscore_adj', [20, 30, 2], '', '', False, '', 'buf0.5', 120, [-2,2]]],
    'shibor1m_qtl': [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'ag', 'l', 'pp', 'v', 'TA', 'eg', 'MA',],
                     ['shibor_1m', 'qtl', [40, 80, 2], 'ema3', '', True, 'price', '', 120, [-2,2]]],
    'r007_lt_zs': [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'ag',  'l', 'pp', 'v', 'TA', 'eg', 'MA',],
                     ['r007_cn', 'zscore', [80, 120, 2], 'ema5', '', True, 'price', '', 120, [-2,2]]],

    "MCU3_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA',],
                ['cu_lme_3m_close', 'zscore', [40, 80, 2], '', '', True, '', 'buf0.2', 120, [-2,2]]],
    # "MAL3_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA',],
    #             ['al_lme_3m_close', 'zscore', [40, 80, 2], '', '', True, '', 'buf0.2', 120, [-2,2]]],

    'cnh_cny_zs': [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA', 'au', 'ag'],
                   ['cnh_cny_spd2', 'zscore', [10, 20, 1], 'ema1', '', False, '', 'buf0.4', 120, [-3, 3]]],
    'cny_dev_zs': [['rb', 'hc', 'i', 'FG', 'SA', 'cu', 'al', 'au', 'ag',
                    'l', 'pp', 'v', 'TA', 'eb', 'eg', 'MA', 'sc', 'lu', 'bu'],
                   ['cny_mid_dev1', 'zscore', [60, 80, 2], 'ema10', '', True, '', '', 120, [-2,2]]],

    'vix_zs_st': [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA', 'ag'],
                  ['vix', 'zscore', [20, 60, 2], '', '', False, '', 'buf0.5', 120, [-2,2]]],
    'vix_mds_st': [['cu', 'al', 'zn', 'rb', 'hc', 'FG', 'SA', 'l', 'pp', 'v', 'TA', 'eg', 'MA', 'ag'],
                   ['vix', 'ma_dff_sgn', [40, 80, 2], '', '', False, '', '', 120, [-2,2]]],

    "usggbe10_zs": [['rb', 'hc', 'cu', 'al', 'zn', 'ag', 'pb'], # shift_holding=2
                    ['usggbe10', 'zscore', [40, 80, 2], '', '', True, '', 'buf0.4', 120, [-2,2]]],
    # au-ag
    #'ag_etf_st_mom': [['au'], ['ag_etf_sivr_holding', 'qtl', [5, 10, 1], 'df252', 'diff', True, '', 'buf0.2', 120, [-2,2]]],
    "auag_cme_wratio_zs": [['au-ag'], ["auag_cme_warrant_ratio", 'zscore', [20, 40, 2], '', '', False, '', 'ema1', 120, [-2,2]]],
    'auag_vix_zsa_mt': [['au-ag'], ['vix', 'zscore_adj', [40, 80, 2], '', '', True, '', 'buf0.3', 120, [-2,2]]],
    'auag_fxbasket_zs_yr': [['au-cu', 'au-rb'], ['fxbasket_cumret', 'zscore', [240, 520, 5], '', '', True, '', '', 120, [-2,2]]],

    'auag_csi500_zs_st': [['au_ag'], ['csi500_idx', 'zscore', [20, 40, 2], '', '', False, '', 'ema5', 120, [-2,2]]],
    'auag_etf_mrev': [['au', 'ag'], ['etf_holdings', 'zscore', [480, 520, 2], '', '', False, '', 'ema3', 120, [-2,2]]], # dump
    'auag_etf_mrev_xdemean': [['au', 'ag'], ['etf_holdings', 'zscore', [480, 520, 2], '', '', False, '', 'ema3', 120, [-2,2]]],

    # petchem
    # PTA PX eg PF
    'TA_margin_st_zs': [['TA', 'PX', 'PF'],
                        ['PTA_margin_cn_d', 'zscore', [20, 40, 2], '', '', False, '', '', 120, [-2,2]]],
    'TA_wnt_yoy_zs': [['TA'],
                      ["TA_inv_czce_warrant", 'zscore', [720, 750, 2], 'cal_yoy_day', 'diff', False, 'price', '', 120, [-2,2]]],
    'TA_arb_ma': [['TA'], # sr ~ 0.5
                  ['PTA_cfr_dom_ratio', 'ma', [1, 2], 'ema3', '', True, 'price', '', 60, [-2,2]]],
    'TA_ryieldmom_zs': [['TA'], ["TA_ryield", 'zscore', [240, 250, 2], 'df20', 'diff', True, 'price', '', 120, [-2,2]]],
    'PX_margin_mds': [['PX', 'TA'], ['PX_margin_cn_w', 'ma_dff_sgn', [50, 55, 1], '', '', True, '', '', 120, [-2,2]]],
    'MX_invchg_ma': [['TA', 'PX'], ['MX_inv_combo', 'ma', [1, 2, 1], 'df1', 'diff', False, '', '', 26, [-2,2]]],
    'TA_minv_hys': [['TA', 'eg', 'PF', 'PX'],
                    ['PTA_invdays_mill', 'hlratio', [50, 55, 1], 'df1', 'diff', True, '', 'hys_0.5_0.5', 120, [-2,2]]],
    'DTY_invdays_yoy_hlr': [['TA', 'PX', 'eg', 'PF'], ['DTY_invdays_mill', 'hlratio', [50, 55, 1], 'cal_yoy_day', 'diff', False, '', '', 120, [-2,2]]],
    'FDY_invdays_lyoy_hlr': [['TA', 'PX', 'eg', 'PF'], ['FDY_invdays_mill', 'hlratio', [50, 55, 1], 'lunar_yoy_day', 'diff', False, '', '', 120, [-2,2]]],
    'POY_invdays_hlr': [['TA', 'PX', 'eg', 'PF'], ['POY_invdays_mill', 'hlratio', [50, 55, 1], '', '', False, '', '', 120, [-2,2]]],
    'eg_invchg_ma': [['eg'],
                     ['eg_inv_port_east', 'ma', [1, 2, 1], 'df1', 'diff', False, '', '', 52, [-2,2]]],
    'eg_invchg_lunar_hlr': [['eg'],
                            ['eg_inv_port_east', 'seasonal_lunar_wk_hlr', [2, 5, 10], 'df1', 'diff', False, '', '', 52, [-2,2]]],
    'ru_invchg_cal_zs': [['ru'],
                        ['ru_inv_shfe_all', 'seasonal_cal_wk_zs', [2, 5, 10], 'df1', 'diff', False, '', '', 52, [-2,2]]],
    'nr_invchg_ma': [['nr', 'ru'],
                     ['nr_inv_shfe_all', 'ma', [1, 2, 1], 'df1', 'diff', False, '', '', 26, [-2,2]]],
    'nr_inv_rev_hlr': [['br'],
                     ['nr_inv_shfe_all', 'hlratio', [50, 55, 1], '', '', True, '', '', 26, [-2,2]]],

    'fu_inv_sgp_yoy_hlr': [['fu-sc'],
                           ['fu_inv_sing', 'hlratio', [50, 55, 1], 'cal_yoy_day', 'diff', False, '', '', 120, [-2,2]]],

    'fgsa_margin_mom_st': [['FG-SA'],
                         ['fg_margin_petcoke', 'qtl', [2, 4, 1], '', '', True, '', "", 120, [-2,2]]],
    'FG_margin_hlr_1y': [['SA'], ['fg_margin_avg', 'hlratio', [48, 56, 1], '', '', True, '', '', 120, [-2,2]]],
    'FG_util_mom_st': [['FG'], ["fg_util_adj", 'hlratio', [8, 12, 1], '', '', True, '', '', 120, [-2,2]]],
    'FG_util_mom_spd_st': [['FG_SA'], ["fg_util_adj", 'qtl', [4, 12, 1], '', '', True, '', '', 120, [-2,2]]],

    'eb_pinv_east_cal_hlr': [['eb'],
                             ['eb_inv_port_east_trader', 'seasonal_cal_wk_hlr', [2, 4, 10], '', '', False, '', '', 120, [-2,2]]],
    'eb_pinvchg_east_cal_hlr': [['eb'],
                             ['eb_inv_port_east_trader', 'seasonal_cal_wk_hlr', [2, 4, 10], 'df1', 'diff', False, '', '', 120, [-2,2]]],
    # 'cnyrr25_zsa': ('usdcny_rr25', 'zscore_adj', [10, 20, 2], 'ema10', 'pct_change', False, 'price'),
    #
    # 'vhsi_mds': ('vhsi', 'ma_dff_sgn', [10, 20, 2], '', 'pct_change', False, 'price'),
    # 'vhsi_qtl': ('vhsi', 'qtl', [10, 30, 2], '', 'pct_change', False, 'price'),
    # 'sse50iv_mds': ('sse50_etf_iv', 'ma_dff_sgn', [20, 30, 2], '', 'pct_change', False, 'price'),
    # 'sse50iv_qtl': ('sse50_etf_iv', 'qtl', [20, 40, 2], '', 'pct_change', False, 'price'),
    # 'eqmargin_zsa': ('eq_margin_outstanding_cn', 'zscore_adj', [10, 20, 2], '', 'pct_change', False, 'price'),
    # 'eqmargin_zs': ('eq_margin_outstanding_cn', 'zscore_adj', [10, 20, 2], '', 'pct_change', False, 'price'),
    #
    # '10ybe_mds': ('usggbe10', 'ma_dff_sgn', [10, 30, 2], '', 'pct_change', True, 'price'),
    # '10ybe_zsa': ('usggbe10', 'zscore_adj', [20, 40, 2], '', 'pct_change', True, 'price'),
    # '10y_2y_mds': ('usgg10yr_2yr_spd', 'ma_dff_sgn', [20, 30, 2], '', 'pct_change', True, 'price'),
    #
    # 'dxy_qtl_s': ('dxy', 'qtl', [40, 60, 2], '', 'pct_change', False, 'price'),
    # 'dxy_qtl_l': ('dxy', 'qtl', [480, 520, 2], '', 'pct_change', False, 'price'),
    #
    # 'vix_mds': ('vix', 'ma_dff_sgn', [20, 40, 2], '', 'pct_change', False, 'price'),

    "pmi_stl_all_yoy": [['rb', 'hc', 'j', 'jm', 'i'],
                        ['pmi_cn_steel_all', 'ma', [1, 2, 1], 'df12', 'diff', True, '', '', 24, [-2,2]]],
    "pmi_stl_prod_yoy": [['rb', 'hc', 'j', 'jm', 'i'],
                        ['pmi_cn_steel_prod', 'ma', [1, 2, 1], 'df12', 'diff', True, '', '', 24, [-2,2]]],
    "pmi_cons_exp_yoy": [['rb', 'hc', 'j', 'jm', 'i'],
                        ['pmi_cn_cons_bus_exp', 'ma', [1, 2, 1], 'df12', 'diff', True, '', '', 24, [-2,2]]],
    "pmi_lgsc_stl_tot_order_yoy": [['rb', 'hc'],
                        ['pmi_lgsc_steel_tot_order', 'ma', [1, 2, 1], 'df12', 'diff', True, '', '', 24, [-2,2]]],
    "pmi_lgsc_stl_fund_yoy": [['rb', 'hc', "i"],
                        ['pmi_lgsc_steel_purchase_exp', 'ma', [1, 2, 1], 'df12', 'diff', True, '', '', 24, [-2,2]]],
    "pmi_cn_purchase_yoy": [['rb', 'hc', 'j', 'jm', 'i', 'FG', 'v', 'al', 'cu'],
                        ['pmi_cn_manu_purchase', 'ma', [1, 2, 1], 'df12', 'diff', True, '', '', 24, [-2,2]]],
    "pmi_stl_o2inv_zs": [['rb', 'hc', 'j', 'jm', 'i'],
                        ['pmi_steel_order_inv_ratio', 'zscore', [60, 72, 1], '', '', True, '', '', 24, [-2,2]]],
    "pmi_stl_o2inv_qtl_sgn": [['rb', 'hc', 'j', 'jm', 'i'],
                        ['pmi_steel_order_inv_ratio', 'qtl', [48, 60, 1], '', '', True, '', 'sgn', 24, [-2,2]]],
    "pmi_order_rminv_ratio_zs": [['rb', 'hc', 'v', 'FG', 'cu', 'al', 'zn'],
                        ['pmi_order_rminv_ratio', 'zscore', [60, 72, 1], '', '', True, '', 'sgn', 24, [-2,2]]],
    # "ppi_cpi_spd_zs": [['rb', 'hc', 'j', 'jm', 'v', 'SM', 'SF', 'cu', 'al', 'zn', 'ni', 'sn', 'ss', 'pp', 'sc', 'lu'],
    #                     ['ppi_cpi_mom_spd', 'zscore', [48, 60, 1], 'sum12', '', True, '', 'arr2|sgn', 24, [-2,2]]],
    "m1_m2_spd_zs_sgn": [['rb', 'hc', 'v', 'FG', 'cu', 'zn', 'al'],
                         ["m1_m2_spd", 'zscore', [24, 36, 1], '', '', True, '', 'sgn', 24, [-2,2]]],
}

signal_buffer_config = {
    'bond_mr_st_qtl': 0.1,
    'bond_tf_lt_qtl': 0.1,
    'bond_au_st_qtl': 0.1,
    'bond_r007_lt_zs': 0.4,
    'bond_shibor1m_qtl': 0.3,
    'steel_margin_lvl_fast': 0.2,
    'fef_ryieldmom_or_zs': 0.25,
    'fef_ryieldmom_spd_zs': 0.25,
    'rbsales_lyoy_mom_st': 0.1,
    'rbsales_lyoy_mom_lt': 0.1,

    'hc_rb_diff_20': 0.15,
    "lme_base_ts_hlr": 0.05,
    "lme_futbasis_ma": 0.5,
    "base_inv_exch_ma": 0.15,

    "lme_inv_wnt_mad": 0.2,
    "base_inv_shfe_mad": 0.2,
    "fxbasket_zs": 0.2,
    'dxy_zsa_s': 0.5,
    "MCU3_zs": 0.2,
    'cnh_cny_zs': 0.4,
    'vix_zs_st': 0.5,
    'rbhc_phycarry_diff_zs': 0.4,
    'rbhc_basmom_diff_hlr': 0.25,
    'rbhc_steel_spd_mds': 0.25,
    'rbhc_steel_spd_lyoy_mds': 0.25,
    'rbhc_rbsales_lyoy_zs': 0.25,
    'auag_vix_zsa_mt': 0.3,
}

signal_execution_config = {
    'lme_base_ts_mds': {"win": "a1505", "lag": 1},
    'lme_base_ts_mds_xdemean': {"win": "a1505", "lag": 1},
    'lme_base_ts_hlr': {"win": "a1505", "lag": 1},
    'lme_base_ts_hlr_xdemean': {"win": "a1505", "lag": 1},
    'lme_futbasis_ma': {"win": "a1505", "lag": 1},
    'lme_futbasis_ma_xdemean': {"win": "a1505", "lag": 1},
    'rbsales_lyoy_spd_st': {"win": "a1505", "lag": 1},
    'rbsales_lyoy_mom_st': {"win": "a1505", "lag": 1},
    'rbsales_lyoy_mom_lt': {"win": "a1505", "lag": 1},
    'rb_sales_inv_ratio_lyoy': {"win": "a1505", "lag": 1},
    'ioarb_px_hlr': {"win": "a1505", "lag": 1},
    'ioarb_px_hlrhys': {"win": "a1505", "lag": 1},
    'ioarb_spd_qtl_1y': {"win": "a1505", "lag": 1},
    'MCU3_zs': {"win": "a1505", "lag": 1},
    'fxbasket_zs': {"win": "a1505", "lag": 1},
    "us_oil_prod_etf_mom":  {"win": "a1505", "lag": 1},
    'rbhc_px_diff_mds': {"win": "close", "lag": 1},
    'rbhc_px_diff_lyoy_mds': {"win": "close", "lag": 1},
    'rbhc_phycarry_diff_zs': {"win": "close", "lag": 1},
    'rbhc_basmom_diff_hlr': {"win": "close", "lag": 1},
    'rbhc_steel_spd_mds': {"win": "close", "lag": 1},
    'rbhc_steel_spd_lyoy_mds': {"win": "close", "lag": 1},
    'rbhc_dmd_ratio_mds': {"win": "close", "lag": 1},
    'rbhc_dmd_ratio_lyoy_mds': {"win": "close", "lag": 1},
    'rbhc_sinv_chg_mds': {"win": "close", "lag": 1},
    'rbhc_sinv_chg_lyoy_mds': {"win": "close", "lag": 1},
    'rbhc_sinv_lratio_mds': {"win": "close", "lag": 1},
    'rbhc_sinv_lratio_lyoy_mds': {"win": "close", "lag": 1},
    'rbhc_rbsales_lyoy_zs': {"win": "close", "lag": 1},
    "auag_cme_wratio_zs": {"win": "a1505", "lag": 1},
    'auag_fxbasket_zs_yr': {"win": "a1505", "lag": 1},
    'auag_etf_mrev_xdemean': {"win": "a1505", "lag": 1},
}

feature_to_feature_key_mapping = {
    'prem_bonded_warrant': {
        'cu': 'cu_prem_bonded_warrant',
        'al': 'al_prem_bonded_warrant',
        'zn': 'zn_prem_bonded_warrant',
        'ni': 'ni_prem_bonded_warrant',
    },
    'base_tc': {
        'cu': 'cu_mine_tc',
        'zn': 'zn_50conc_tc_henan',
        'pb': 'pb_50conc_tc_neimeng',
        'sn': 'sn_40conc_tc_yunnan',
    },
    'lme_base_ts': {
        'cu': 'cu_lme_3m_15m_spd',
        'al': 'al_lme_3m_15m_spd',
        'zn': 'zn_lme_3m_15m_spd',
        # 'cu': 'cu_lme_0m_3m_spd',
        # 'al': 'al_lme_0m_3m_spd',
        # 'zn': 'zn_lme_0m_3m_spd',
        'ni': 'ni_lme_0m_3m_spd',
        'sn': 'sn_lme_0m_3m_spd',
        'pb': 'pb_lme_0m_3m_spd',
    },
    'base_phybas': {
        'cu': 'cu_smm_phybasis',
        'al': 'al_smm0_phybasis',
        'zn': 'zn_smm0_sh_phybasis',
        'pb': 'pb_smm1_sh_phybasis',
        'ni': 'ni_smm1_jc_phybasis',
        'sn': 'sn_smm1_sh_phybasis',
    },
    'base_inv': {
        'cu': "cu_inv_combo",
        'al': 'al_inv_social_all',
        'zn': "zn_inv_social_all",
        'ni': "ni_inv27_all",
        'pb': 'pb_inv_social_all',
        'sn': 'sn_inv_social_all',
        'ao': 'bauxite_inv_az_ports',
        'ss': "ss_inv_social_300",
    },
    'fer_pinv': {
        "j": "coke_inv_3ports",
        "jm": "ckc_inv_6ports",
        "i": 'io_inv_45ports',
        "rb": 'rebar_inv_social',
        "hc": 'hrc_inv_social',
    },
    'smsf_prodcost': {
        'SM': 'sm_neimeng_cost',
        'SF': 'sf_neimeng_cost',
    },
    'phycarry': {},
    'metal_px': {},
    'px': {},
    'ryield': {},
    'basmom20': {},
    'basmom40': {},
    'basmom60': {},
    'basmom120': {},
    "logret": {},
    "colr": {},
    'inv_shfe_d': {},
    'lme_futbasis': {},
    'inv_lme_total': {},
    'inv_exch_d': {},
    "exch_warrant": {},
    "etf_holdings": {},
    'metal_inv': {
        'cu': 'cu_inv_shfe_d', #"cu_inv_social_dom", # #
        'al': 'al_inv_social_all',
        'zn': "zn_inv_social_all",
        'ni': 'ni_inv_shfe_d', # "ni_inv27_all", #, #
        'pb': 'pb_inv_shfe_d', #'pb_inv_social_all',
        'sn': 'sn_inv_shfe_d',
        'si': "si_inv_gfex_d",
        'lc': "lc_inv_gfex_d",
        'ps': "ps_inv_gfex_d",
        'ao': 'ao_inv_shfe_d', #'bauxite_inv_az_ports',
        'ss': "ss_inv_social_300",
        'rb': 'rebar_inv_social',
        'hc': 'hrc_inv_social',
        'j' : "coke_inv_ports_tj",
        'jm': 'ckc_inv_ports', #"ckc_inv_cokery",
        'v':  "v_inv_social",
        'i': 'io_inv_45ports',
        'SM': 'sm_stockdays', #'sm_inv_mill', # #
        'SF': 'sf_inv_mill',
        'FG': "fg_inv_mill",
        'SA': 'sa_inv_mill_all',
        'SH': 'sh_inv_mill_all',
    },
    'sinv': {
        'v':  "v_inv_social",
        'eb': 'eb_inv_port_east_trader',

    }
}

param_rng_by_feature_key = {}

proc_func_by_feature_key = {
    'base_phybas_carry': {
        'cu': 'sma20',
        'al': 'sma20'
    }
}


commod_phycarry_dict= {
    "cu": "cu_smm1_spot",
    "al": "al_smm0_spot",
    "zn": "zn_smm0_spot",
    "pb": "pb_994_shmet_east", #"pb_smm1_spot",
    "sn": "sn_smm1_spot",
    "ni": "ni_smm1_spot",
    "ss": "ss_304_gross_wuxi",
    "ao": "alumina_spot_qd",
    "si": "si_ctd_spot",
    "rb": "rebar_sh",
    "hc": "hrc_sh",
    "i": "io_ctd_spot",
    "j": "coke_sub_a_rz",
    "jm": "ckc_outstock_ganqimaodu",
    "FG": "fg_5mm_shahe",
    "SM": "sm_65s17_tj",
    "SF": "sf_72_ningxia",
    "v": "pvc_cac2_east",
    "SH": "SH_ctd_spot",
    "SA": "sa_heavy_shahe",
    "au": 'au_9999_sge_close',
    "ag": 'ag_9999_sge_close',
    "MA": "ma_spot_jiangsu",
    "TA": "pta_east_spot",
    "eg": "eg_east_spot",
    "PF": "pf_fujian_spot",
    "l": "l_7042_tj",
    "pp": "pp_100ppi_spot",
    "eb": "eb_east_spot",
    'bu': "bu_heavy_shandong",
    "pg": "pg_sd_spot_idx",
    'ru': "ru_scrwf_kunming",
    'fu': "fo_180cst_xiamen",
}


def get_funda_signal_from_store(spot_df, signal_name, price_df=None,
                                asset=None, curr_date=None, signal_start=None,
                                signal_repo=signal_store,
                                feature_key_map=feature_to_feature_key_mapping):
    feature, signal_func, param_rng, proc_func, chg_func, bullish, freq, post_func, vol_win, signal_cap = \
        signal_repo[signal_name][1]
    if asset:
        if feature in feature_key_map:
            asset_feature = feature_key_map[feature].get(asset, f'{asset}_{feature}')
        else:
            asset_feature = f'{asset}_{feature}'
        if feature in param_rng_by_feature_key:
            param_rng = param_rng_by_feature_key[feature].get(asset, param_rng)
        if feature in proc_func_by_feature_key:
            proc_func = param_rng_by_feature_key[feature].get(asset, proc_func)
        feature = asset_feature
    signal_ts = calc_funda_signal(spot_df, feature, signal_func, param_rng,
                                  proc_func=proc_func, chg_func=chg_func,
                                  bullish=bullish, freq=freq, signal_cap=signal_cap,
                                  post_func=post_func, vol_win=vol_win,
                                  curr_date=curr_date, signal_start=signal_start)
    return signal_ts


def custom_funda_signal(df, input_args):
    product_list = input_args['product_list']
    funda_df = input_args['funda_data']
    signal_name = input_args['signal_name']
    signal_type = input_args.get('signal_type', 1)
    vol_win = input_args.get('vol_win', 20)

    # get signal by asset
    if signal_type == 0:
        signal_df = pd.DataFrame(index=pd.date_range(start=df.index[0], end=df.index[-1], freq='D'))
        for product in product_list:
            asset = product[:-2]
            signal_df[product] = get_funda_signal_from_store(funda_df, signal_name,
                                                             price_df=df, asset=asset)
        signal_df = signal_df.reindex(index=pd.date_range(start=df.index[0], end=df.index[-1], freq='D'))\
            .ffill().reindex(index=df.index)
        if "xdemean" in signal_name:
            signal_df = xs_demean(signal_df)
        elif "xscore" in signal_name:
            signal_df = xs_score(signal_df)
        elif "xrank" in signal_name:
            signal_df = xs_rank(signal_df, 0.2)

    # pair trading strategy, fixed ratio
    elif signal_type == 3:
        signal_df = pd.DataFrame()
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df)
        signal_ts = signal_ts.reindex(
            index=pd.date_range(start=df.index[0],
                                end=df.index[-1],
                                freq='D'
                                )).ffill().reindex(index=df.index)
        if set(product_list) == set(['rbc1', 'hcc1']):
            signal_df['rbc1'] = signal_ts
            signal_df['hcc1'] = -signal_ts

    # beta neutral last asseet is index asset
    elif signal_type == 4:
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df)
        signal_df = pd.DataFrame(0, index=df.index, columns=product_list)
        index_product = product_list[-1]
        beta_win = 122
        for trade_product in product_list[:-1]:
            asset_df = df[[(index_product, 'close'),
                           (trade_product, 'close')]].copy(deep=True).droplevel([1], axis=1)
            asset_df = asset_df.dropna(subset=[trade_product]).ffill()
            for asset in asset_df:
                asset_df[f'{asset}_pct'] = asset_df[asset].pct_change().fillna(0)
                asset_df[f'{asset}_pct_ma'] = asset_df[f'{asset}_pct'].rolling(5).mean()
                asset_df[f'{asset}_vol'] = asset_df[f'{asset}_pct'].rolling(vol_win).std()
            asset_df['beta'] = asset_df[f'{index_product}_pct_ma'].rolling(beta_win).cov(
                asset_df[f'{trade_product}_pct_ma']) / asset_df[f'{index_product}_pct_ma'].rolling(beta_win).var()
            asset_df['signal'] = signal_ts
            asset_df = asset_df.ffill()
            asset_df['pct'] = asset_df[f'{trade_product}_pct'] - asset_df['beta'] * asset_df[f'{index_product}_pct']
            asset_df['vol'] = asset_df['pct'].rolling(vol_win).std()
            asset_df = asset_df.reindex(index=signal_ts.index).ffill()
            signal_df[trade_product] += signal_ts * asset_df[f'{trade_product}_vol']/asset_df['vol']
            signal_df[index_product] -= signal_ts * asset_df['beta'] * asset_df[f'{index_product}_vol'] / asset_df['vol']

    # apply same signal to all assets
    else:
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df)
        signal_ts = signal_ts.reindex(
            index=pd.date_range(start=df.index[0],
                                end=df.index[-1],
                                freq='D'
                                )).ffill().reindex(index=df.index)
        signal_df = pd.DataFrame(dict([(asset, signal_ts) for asset in product_list]))
    _, _, _, _, _, _, _, post_func, _, _ = signal_store[signal_name][1]
    last_func = post_func.split('|')[-1]
    if 'buf' in last_func:
        buffer_size = float(last_func[3:])
        signal_df = signal_buffer(signal_df, buffer_size)
    elif 'bfc' in last_func:
        buffer_size = float(last_func[3:])
        prod_list = signal_df.columns
        df_px = df.loc[:, df.columns.get_level_values(0).isin(prod_list)].droplevel([1], axis=1).ffill()
        vol_df = df_px.pct_change().rolling(20).std()
        signal_df = signal_cost_optim(signal_df, buffer_size, vol_df,
                                      cost_dict=simple_cost(prod_list, trd_cost=2e-4),
                                      turnover_dict={},
                                      power=3)
    return signal_df

