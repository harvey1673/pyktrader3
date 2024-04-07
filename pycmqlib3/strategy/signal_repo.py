import pandas as pd
from pycmqlib3.analytics.tstool import *
from pycmqlib3.utility.misc import CHN_Holidays, day_shift
from pycmqlib3.utility.exch_ctd_func import *


signal_store = {
    'io_removal_lvl': [['rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'cu'],
                       ['io_removal_41ports', 'qtl', [20, 40, 2], '', '', True, 'price', 'sma2', 120]],
    'io_removal_lyoy': [['rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'cu'],
                        ['io_removal_41ports', 'qtl', [6, 10], 'lunar_yoy_day', 'diff', True, 'W-Fri', 'sma1', 120]],
    'io_removal_wow': [['i'],
                       ['io_removal_41ports', 'zscore', [48, 53], 'df1', 'diff', True, 'W-Fri', "", 120]],
    'io_millinv_lvl': [['i'],
                       ['io_inv_mill(64)', 'qtl', [20, 40, 2], '', 'diff', True, 'price', '', 120]],
    'io_millinv_lyoy': [['rb', 'hc', 'i', 'j', 'jm', 'FG'],
                        ['io_inv_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'diff', True, 'W-Fri', '', 120]],
    'io_invdays_lvl': [['rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'cu', 'al'],
                       ['io_invdays_imp_mill(64)', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price', '', 120]],
    'io_invdays_lyoy': [['i'],
                        ['io_invdays_imp_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'pct_change', True, 'W-Fri', "", 120]],
    # 'io_port_inv_lvl_slow': [['i'],
    #                          ['io_inv_imp_31ports_w', 'zscore', [240, 255, 5], '', '', False, 'price', "", 120]],

    'ioarb_px_hlr': [['rb', 'hc', 'i'], ['io_on_off_arb', 'hlratio', [40, 80, 2], 'sma2', '', False, 'price', '', 120]],
    'ioarb_px_hlrhys': [['rb', 'hc', 'i'], ['io_on_off_arb', 'hlratio', [40, 80, 2], 'sma2', '', False, 'price', '', 120]],
    'ioarb_spd_qtl_1y': [['rb_i', 'hc_i'], ['io_on_off_arb', 'qtl', [240, 260, 2], 'sma2', '', False, 'price', '', 120]],
    'io_pinv31_lvl_zsa': [['rb_i', 'hc_i'],
                          ['io_inv_31ports', 'zscore_adj', [8, 56, 4], '', 'pct_change', True, '', "sma2", 120]],
    'io_pinv45_lvl_hlr': [['rb_i', 'hc_i'],
                          ['io_inv_45ports', 'hlratio', [8, 56, 4], '', 'pct_change', True, '', "sma2", 120]],
    'steel_sinv_lyoy_zs': [['rb', 'hc', 'i', 'FG', 'v'],
                           ['steel_inv_social', 'zscore', [24, 30, 2], 'lunar_yoy_day', 'diff', False, '', "", 120]],
    'steel_sinv_lyoy_mds': [['rb', 'hc', 'i', 'FG', 'v'],
                           ['steel_inv_social', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, '', "", 120]],
    'rbhc_dmd_mds': [['rb-hc'],
                     ['rb_hc_dmd_diff', 'ma_dff_sgn', [5, 9, 1], '', 'diff', True, '', "", 120]],
    'rbhc_dmd_lyoy_mds': [['rb-hc'],
                          ['rb_hc_dmd_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', True, '', "", 120]],
    'rbhc_sinv_mds': [['rb-hc'],
                      ['rb_hc_sinv_diff', 'ma_dff_sgn', [5, 9, 1], '', 'diff', False, '', "", 120]],
    'rbhc_sinv_lyoy_mds': [['rb-hc'],
                           ['rb_hc_sinv_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, '', "", 120]],

    # 'rb_sinv_lyoy_fast': ('rebar_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'wr_sinv_lyoy_fast': ('wirerod_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'hc_soinv_lyoy_fast': ('hrc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'cc_soinv_lyoy_fast': ('crc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'billet_inv_chg_slow': ('billet_inv_social_ts', 'zscore', [240, 252, 2], '', 'diff', False, 'price'),
    'pbf_yoy_qtl': [['fef', 'i'], ["pbf_prem", 'qtl', [20, 30, 2], "cal_yoy",  "diff", True, "", "", 120]],
    'pbf_yoy_eds': [['fef', 'i'], ["pbf_prem", 'ema_dff_sgn', [5, 15, 1], "cal_yoy",  "diff", True, "", "", 120]],
    'pbf_spd': [['rb_i', "hc_i"], ["pbf_prem", 'zscore_adj', [40, 80, 2], "",  "diff", False, "", "", 120]],
    'cons_steel_lyoy_slow': [['rb', 'i', 'hc'],
                             ['cons_steel_transact_vol_china', 'zscore', [240, 255, 5],
                              'lunar_yoy_day', 'diff', True, 'price', "", 120]],
    # 'margin_sea_lvl_mid': ('hrc_margin_sb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),
    'sea_export_arb_lvl_mid': [['hc', 'rb'],
                               ['hrc_exp_sea_arb', 'zscore', [40, 82, 2], '', '', True, 'price', "", 120]],
    'steel_margin_lvl_fast': [['rb', 'hc', 'i', 'j'],
                              ['margin_hrc_macf', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price', "", 120]],
    'strip_hsec_lvl_mid': [['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM', 'cu'],
                           ['strip_hsec', 'qtl', [60, 80, 2], '', '', True, 'price', "", 120]],
    'macf_cfd_lvl_mid': [['i'],
                         ['macf_cfd', 'qtl', [40, 82, 2], '', 'pct_change', True, 'price', "", 120]],
    'hc_rb_diff_lvl_fast': [['rb', 'hc', 'i', 'j', 'jm', 'cu', 'al'],
                            ['hc_rb_diff', 'zscore', [20, 40, 2], '', '', True, 'price', "", 120]],
    'fef_c1_c2_ratio_or_qtl': [['rb', 'hc', 'j'],
                               ['FEF_c1_c2_ratio', 'qtl', [30, 60, 2], '', '', False, '', "", 120]],
    'fef_c1_c2_ratio_spd_qtl': [['rb_i', 'hc_i', 'j_i'],
                                ['FEF_c1_c2_ratio', 'qtl', [30, 60, 2], '', '', False, '', "", 120]],
    'cu_prem_usd_zsa': [['cu'],
                        ['cu_prem_bonded_warrant', 'zscore_adj', [20, 30, 2], '', '', True, 'price', "", 120]],
    'cu_prem_usd_md': [['cu'],
                       ['cu_prem_bonded_warrant', 'ma_dff', [20, 30, 2], '', '', True, 'price', "", 120]],
    'cu_phybasis_zsa': [['cu'],
                        ['cu_cjb_phybasis', 'zscore_adj', [40, 60, 2], 'sma10', '', True, 'price', "", 120]],  # great
    'cu_phybasis_hlr': [['cu'],
                        ['cu_cjb_phybasis', 'hlratio', [40, 60, 2], 'sma10', '', True, 'price', "", 120]],  # great
    'lme_base_ts_mds': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_base_ts', 'ma_dff_sgn', [10, 30, 2], '', '', True, 'price', "", 120]],
    'lme_base_ts_mds_xdemean': [['cu'],
                        ['lme_base_ts', 'ma_dff_sgn', [10, 30, 4], '', '', True, 'price', "", 120]],
    'lme_base_ts_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_base_ts', 'hlratio', [10, 20, 2], '', '', True, 'price', "", 120]],
    'lme_base_ts_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_base_ts', 'hlratio', [10, 20, 2], '', '', True, 'price', "", 120]],
    'base_phybas_carry_ma': [['cu', 'al', 'zn', 'ni', 'sn'],
                             ['base_phybas', 'ma', [1, 2], 'sma2', '', True, 'price', "", 120]],
    'base_phybas_carry_ma_xdemean': [['cu', 'al', 'zn', 'ni', 'sn'],
                                     ['base_phybas', 'ma', [1, 2], 'sma2', '', True, 'price', "", 120]],
    'base_phybasmom_1m_zs': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                             ['base_phybas', 'zscore', [20, 30], 'sma2', '', True, 'price', "", 120]],
    'base_phybasmom_1m_zs_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                     ['base_phybas', 'zscore', [20, 30], 'sma2', '', True, 'price', "", 120]],
    'base_phybasmom_1y_zs': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                             ['base_phybas', 'zscore', [230, 250, 2], 'sma2', '', True, 'price', "", 120]],
    'base_phybasmom_1y_zs_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                     ['base_phybas', 'zscore', [230, 250, 2], 'sma2', '', True, 'price', "", 120]],
    'base_cifprem_1y_zs': [['cu', 'al', 'zn', 'ni'], ['prem_bonded_warrant', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120]],
    'base_cifprem_1y_zs_xdemean': [['cu', 'al', 'zn', 'ni'], ['prem_bonded_warrant', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120]],
    'base_tc_1y_zs': [['cu', 'pb', 'zn'], ['base_tc', 'zscore', [230, 250, 2], '', '', False, 'price', "", 120]],

    'metal_pbc_ema': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                      ['metal_pbc', 'ema', [10, 20], '', '', True, 'price', "", 120]],
    'metal_pbc_ema_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                              ['metal_pbc', 'ema', [10, 20], '', '', True, 'price', "", 120]],
    'base_inv_mds': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                     ['base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price', "", 120]],
    'base_inv_mds_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                             ['base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price', "", 120]],
    'metal_inv_hlr': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                      ['metal_inv', 'hlratio', [240, 250], '', '', False, 'price', "", 120]],
    'metal_inv_hlr_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                              ['metal_inv', 'hlratio', [240, 250, 2], '', '', False, 'price', "", 120]],
    'metal_inv_lyoy_hlr': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                           ['metal_inv', 'hlratio', [240, 250], 'lunar_yoy_day', 'pct_change', False, 'price', "", 120]],
    'metal_inv_lyoy_hlr_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                                   ['metal_inv', 'hlratio', [240, 250], 'lunar_yoy_day',
                                    'pct_change', False, 'price', "", 120]],

    "base_etf_mom_zsa": [['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
                         ["base_sw_csi500_ret", "zscore_adj", [20, 40, 1], "csum", "", True, "", "", 120]],
    "base_etf_mom_ewm": [['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
                         ["base_sw_csi500_ret", "ewmac", [2, 4, 1], "csum", "", True, "", "", 120]],
    "const_etf_mom_zsa": [['rb', 'i', 'v', 'FG'],
                          ["const_sw_csi500_ret", "zscore_adj", [40, 80, 2], "csum", "", True, "", "", 120]],
    "const_etf_mom_ewm": [['rb', 'i', 'v', 'FG'],
                          ["const_sw_csi500_ret", "ewmac", [2, 5, 1], "csum", "", True, "", "", 120]],

    "prop_etf_mom_dbth_zs": [['rb', 'i', 'v', 'FG'],
                             ["prop_sw_csi500_ret", "hysteresis", [1, 120, 0.5], "ema3", "zscore_roll", True, "", "", 120]],
    "prop_etf_mom_dbth_qtl": [['rb', 'i', 'v', 'FG'],
                              ["prop_sw_csi500_ret", "dbl_th", [0.75, 120, 0], "ema3", "pct_score", True, "", "", 120]],
    "prop_etf_mom_dbth_qtl2": [['rb', 'i', 'v', 'FG'],
                               ["prop_sw_csi500_ret", "dbl_th", [0.8, 240, 1], "ema3", "pct_score", True, "", "", 120]],

    # # too short
    # 'cu_scrap1_margin_gd': ('cu_scrap1_diff_gd', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price'),  # too short
    # 'cu_scrap1_margin_tj': ('cu_scrap1_diff_tj', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price'),  # too short
    # 'cu_rod_procfee_2.6': ('cu_rod_2.6_procfee_nanchu', 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price'),
    # 'cu_rod_procfee_8.0': ('cu_rod_8_procfee_nanchu', 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price'),
    #
    # 'r007_qtl': ('r007_cn', 'qtl', [80, 120, 2], 'ema5', 'pct_change', True, 'price'),
    # 'r_dr_spd_zs': ('r_dr_7d_spd', 'zscore', [20, 40, 2], 'ema5', 'pct_change', True, 'price'),
    # 'shibor1m_qtl': ('shibor_1m', 'qtl', [40, 80, 2], 'ema3', 'pct_change', True, 'price'),
    #
    # 'cnh_mds': ('usdcnh_spot', 'ma_dff_sgn', [10, 30, 2], 'ema3', 'pct_change', False, 'price'),
    # 'cnh_cny_zsa': ('cnh_cny_spd', 'zscore_adj', [10, 20, 2], 'ema10', 'pct_change', False, 'price'),
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
    # 'vix_zsa': ('vix', 'zscore_adj', [40, 60, 2], 'ema3', 'pct_change', False, 'price'),
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
        'cu': 'cu_inv_social_all',
        'al': 'al_inv_social_all',
        'zn': 'zn_inv_social_3p',
        'ni': 'ni_inv_social_6p',
        'pb': 'pb_inv_social_5p',
        'sn': 'sn_inv_social_all',
        'si': 'si_inv_social_all',
        'ao': 'bauxite_inv_az_ports',
    },
    'metal_pbc': {
        "cu": "cu_smm1_spot",
        "al": "al_smm0_spot",
        "zn": "zn_smm0_spot",
        "pb": "pb_smm1_spot",
        "sn": "sn_smm1_spot",
        "ni": "ni_smm1_spot",
        "ss": "ss_304_gross_wuxi",
        "ao": "alumina_spot_qd",
        "si": "si_553_spot_smm",
        "rb": "rebar_sh",
        "hc": "hrc_sh",
        "i": "io_ctd_spot",
        "j": "coke_sh_xb",
        "jm": "ckc_a10v24s08_lvliang",
        "FG": "fg_5mm_shahe",
        "SM": "sm_65s17_neimeng",
        "SF": "sf_72_ningxia",
        "v": "pvc_cac2_east",
        "SA": "sa_heavy_east",
    },
    'metal_inv': {
        'cu': 'cu_inv_social_all',
        'al': 'al_inv_social_all',
        'zn': 'zn_inv_social_3p',
        'ni': 'ni_inv_social_6p',
        'pb': 'pb_inv_social_5p',
        'sn': 'sn_inv_social_all',
        'si': 'si_inv_social_all',
        'ao': 'bauxite_inv_az_ports',
        'ss': "ss_inv_social_300",
        'rb': 'rebar_inv_social',
        'hc': 'hrc_inv_social',
        'j': "coke_inv_ports_tj",
        'jm': "ckc_inv_cokery",
        'v': "v_inv_social",
        'i': 'io_inv_45ports',
        'SM': 'sm_inv_mill',
        'SF': 'sf_inv_mill',
        'FG': "fg_inv_mill",
        'SA': 'sa_inv_mill_all',
    }
}

param_rng_by_feature_key = {}

proc_func_by_feature_key = {
    'base_phybas_carry': {
        'cu': 'sma20',
        'al': 'sma20'
    }
}

leadlag_port_d = {
    # 'ferrous': {'lead': ['hc', 'rb', ],
    #             'lag': [],
    #             'param_rng': [40, 80, 2],
    #             },
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
    ('rb', 'hc'), ('j', 'jm'),  ('i', 'j'), ('SM', 'SF'), ('FG', 'v'),
    ('y', 'OI'), ('m', 'RM'),
    ('l', 'MA'), ('pp', 'MA'), ('TA', 'MA'), ('TA', 'eg')
]


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


def mr_pair(df, input_args):
    mr_pair_list = mr_commod_pairs
    product_list = input_args['product_list']
    signal_cap = input_args.get('signal_cap', None)
    conv_func = input_args.get('conv_func', 'zscore_adj')
    param_rng = input_args.get('params', [240, 250, 2])
    vol_win = input_args.get('vol_win', 120)
    signal_df = pd.DataFrame(index=df.index, columns=product_list)
    for (asset_a, asset_b) in mr_pair_list:
        pair_assets = [asset_a, asset_b]
        sig_df = pd.DataFrame(index=df.index, columns=pair_assets)
        feature_ts = np.log(df[(asset_a, 'c1', 'close')]) - np.log(df[(asset_b, 'c1', 'close')])
        sig_ts = calc_conv_signal(feature_ts, signal_func=conv_func, param_rng=param_rng, signal_cap=signal_cap,
                                  vol_win=vol_win)
        sig_ts = sig_ts.apply(lambda x: np.sign(x) * min(abs(x), 1.25) ** 4)

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


def get_funda_signal_from_store(spot_df, signal_name, price_df=None,
                         signal_cap=None, asset=None,
                         signal_repo=signal_store, feature_key_map=feature_to_feature_key_mapping):
    feature, signal_func, param_rng, proc_func, chg_func, bullish, freq, post_func, vol_win = \
        signal_repo[signal_name][1]
    if asset and feature in feature_key_map:
        asset_feature = feature_key_map[feature].get(asset, feature)
        if feature == 'metal_pbc':
            if price_df is None:
                print("ERROR: no future price is passed for metal_pbc")
                return pd.Series()
            spot_df['date'] = pd.to_datetime(spot_df.index)
            spot_df[f'{asset}_c1'] = price_df[(asset, 'c1', 'close')] / np.exp(price_df[(asset, 'c1', 'shift')])
            spot_df[f'{asset}_expiry'] = pd.to_datetime(price_df[(asset, 'c1', 'expiry')])
            if asset == 'i':
                spot_df['io_ctd_spot'] = io_ctd_basis(spot_df, price_df[('i', 'c1', 'expiry')])
            spot_df[f'{asset}_phybasis'] = (np.log(spot_df[asset_feature]) - np.log(spot_df[f'{asset}_c1'])) / \
                                           (spot_df[f'{asset}_expiry'] - spot_df['date']).dt.days * 365
            asset_feature = f'{asset}_phybasis'
        if feature in param_rng_by_feature_key:
            param_rng = param_rng_by_feature_key[feature].get(asset, param_rng)
        if feature in proc_func_by_feature_key:
            proc_func = param_rng_by_feature_key[feature].get(asset, proc_func)
        feature = asset_feature
    signal_ts = calc_funda_signal(spot_df, feature, signal_func, param_rng,
                                  proc_func=proc_func, chg_func=chg_func,
                                  bullish=bullish, freq=freq, signal_cap=signal_cap,
                                  post_func=post_func, vol_win=vol_win)
    return signal_ts


def custom_funda_signal(df, input_args):
    product_list = input_args['product_list']
    signal_cap = input_args.get('signal_cap', [-2, 2])
    funda_df = input_args['funda_data']
    signal_name = input_args['signal_name']
    signal_type = input_args.get('signal_type', 1)
    vol_win = input_args.get('vol_win', 20)

    # get signal by asset
    if signal_type == 0:
        signal_df = pd.DataFrame(index=pd.date_range(start=df.index[0], end=df.index[-1], freq='C'))
        for asset in product_list:
            signal_df[asset] = get_funda_signal_from_store(funda_df, signal_name,
                                                           price_df=df, signal_cap=signal_cap, asset=asset)
        signal_df = signal_df.ffill().reindex(index=df.index)
        if "xdemean" in signal_name:
            signal_df = xs_demean(signal_df)
        elif "xscore" in signal_name:
            signal_df = xs_score(signal_df)
        elif "xrank" in signal_name:
            signal_df = xs_rank(signal_df, 0.2)

    # pair trading strategy, fixed ratio
    elif signal_type == 3:
        signal_df = pd.DataFrame()
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
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
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
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
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
        signal_ts = signal_ts.reindex(
            index=pd.date_range(start=df.index[0],
                                end=df.index[-1],
                                freq='C'
                                )).ffill().reindex(index=df.index)
        signal_df = pd.DataFrame(dict([(asset, signal_ts) for asset in product_list]))

    signal_df = signal_df
    return signal_df
