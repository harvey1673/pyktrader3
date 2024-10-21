import pandas as pd
from pycmqlib3.analytics.tstool import *
from pycmqlib3.analytics.btmetrics import simple_cost
from pycmqlib3.utility.misc import CHN_Holidays, day_shift
from pycmqlib3.utility.exch_ctd_func import *

broad_mkts = [
        'rb', 'hc', 'i', 'j', 'jm', 'FG', 'v', 'SM', 'SF',
        'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss',  # 'si', #'bc'
        'ru', 'l', 'pp', 'TA', 'sc', 'eb', 'eg', 'UR', 'lu',
        'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', 'AP', 'lh']

signal_store = {
    'pbf_yoy_qtl': [['fef', 'i'], ["pbf_prem", 'qtl', [20, 30, 2], "cal_yoy", "diff", True, "", "", 120]],
    'pbf_yoy_eds': [['fef', 'i'], ["pbf_prem", 'ema_dff_sgn', [5, 15, 1], "cal_yoy", "diff", True, "", "", 120]],
    'pbf_spd': [['rb_i', "hc_i"], ["pbf_prem", 'zscore_adj', [40, 80, 2], "", "", False, "", "", 120]],
    'pbf_mix_spd_hys': [['fef', 'i'],
                        ['pbf_iocj_ssf_spd', 'hysteresis', [0.5, 500, 0.5], "",  "hlratio", False, "", "", 120]],
    'landsales_yoy_ma': [['rb', 'hc'], ['top100cities_land_supplied_area_res_all',
                                        'ma', [4, 8], 'lunar_yoy_day', 'diff', True, "", "", 52]],
    'prop2hand_px_qtl': [['rb', 'hc'], ['prop_2ndhand_px_idx', 'qtl', [50, 55], '', '', True, "", "", 120]],

    'rbsales_lyoy_spd_st': [['rb-hc'],
                            ['consteel_dsales_mysteel', 'zscore', [20, 30, 2],
                             'lunar_yoy_day|ema3', 'diff', True, 'price', "ema1", 120]],
    'rbsales_lyoy_mom_st': [['rb'],
                            ['consteel_dsales_mysteel', 'zscore', [40, 80, 2],
                             'lunar_yoy_day|ema3', 'diff', True, 'price', "ema1", 120]],
    'rbsales_lyoy_mom_lt': [['rb'],
                            ['consteel_dsales_mysteel', 'zscore', [230, 250, 2],
                             'lunar_yoy_day|ema3', 'diff', True, 'price', "", 120]],
    'rb_sales_inv_ratio_lyoy': [['rb'],
                                ['rebar_sales_inv_ratio', 'hlratio', [20, 60, 2],
                                 'ema3|lunar_yoy_day', 'diff', True, 'price', "ema1", 120]],
    'iosales_lyoy_ema': [['i'],
                         ['io_trdvol_davg_majorports', 'ema', [1, 2, 1],
                          'lunar_yoy_day', 'diff', True, '', "", 20]],

    'fef_phycarry_ema': [['fef', 'i'],
                         ['FEF_phycarry', 'ema', [1, 2, 1], '', '', True, 'price', "ema1", 240]],
    'rb_phycarry_ema': [['fef', 'rb', 'i', 'j', 'jm'],
                        ['rb_phycarry', 'ema', [10, 20, 1], '', '', True, '', "sma3", 120]],
    'hc_phycarry_ema': [['fef', 'hc', 'i', 'j', 'jm'],
                        ['hc_phycarry', 'ema', [10, 20, 1], '', '', True, '', "sma3", 120]],
    'ckc_phycarry_ema': [['fef', 'i'],
                         ['jm_phycarry', 'ema', [10, 20, 1], '', '', True, '', "sma3", 120]],
    'fef_c1_c2_ratio_or_qtl': [['rb', 'hc', 'j'],
                               ['FEF_c1_c2_ratio', 'hlratio', [40, 60, 2], 'ema1', '', False, '', "", 120]], #hmp0.35
    'fef_c1_c2_ratio_spd_qtl': [['rb_i', 'hc_i', 'j_i'],
                                ['FEF_c1_c2_ratio', 'qtl', [30, 60, 2], '', '', False, '', "hmp0.2", 120]],
    'fef_fly_ratio_or_qtl': [['rb', 'hc', 'j'],
                             ['FEF_c123fly_ratio', 'hlratio', [40, 60, 2], 'ema1', '', False, '', "hmp0.2", 120]],
    # 'fef_fly_ratio_spd_qtl': [['rb_i', 'hc_i'],
    #                           ['FEF_c123_fly_ratio', 'qtl', [40, 60, 2], '', '', False, '', "", 120]],
    'fef_ryieldmom_or_zs': [['rb', 'hc', 'j', 'jm'],
                            ['FEF_ryield', 'zscore', [10, 80, 2], 'ema3', '', False, '', "", 120]],
    'fef_ryieldmom_spd_zs': [['rb_i', 'hc_i'],
                             ['FEF_ryield', 'zscore', [10, 80, 2], 'ema1', '', False, '', "hmp0.4", 120]],
    'fef_basmom_or_qtl': [['rb', 'hc'],
                          ['FEF_basmom', 'qtl', [60, 80, 2], 'ema20', '', False, 'price', "ema1", 120]],
    'fef_basmom5_or_qtl': [['rb', 'hc'],
                          ['FEF_basmom5', 'qtl', [60, 80, 2], 'ema5', '', False, 'price', "", 120]],
    'fef_basmom5_spd_qtl': [['rb_i', 'hc_i'],
                           ['FEF_basmom5', 'qtl', [40, 80, 2], 'ema3', '', False, 'price', "hmp0.2", 120]],
    'fef_basmom_or_ema': [['rb', 'hc'],
                          ['FEF_basmom5', 'ema', [3, 6], '', '', False, 'price', "ema1", 120]],
    'fef_basmom5_spd_ema': [['rb_i', 'hc_i'],
                           ['FEF_basmom5', 'ema', [3, 6], '', '', False, 'price', "ema1", 120]],

    'pbf_arb_hlr': [['rb', 'hc', 'i', 'j', 'jm'], # macf jmb
                    ['pbf_imp_profit', 'hlratio', [40, 80, 2], '', '', True, 'price', 'sma3', 120]],
    'pbf_arb_hlrhys': [['rb', 'hc', 'i', 'j', 'jm'], # nmf macf jmb
                       ['pbf_imp_profit', 'hysteresis', [0.7, 240, 0.1], '', 'hlratio', False, '', 'ema1', 120]],
    'ioarb_px_hlr': [['rb', 'hc', 'i'],
                     ['io_on_off_arb', 'hlratio', [40, 80, 2], '', '', False, 'price', 'ema5', 120]],
    'ioarb_px_hlrhys': [['rb', 'hc', 'i'],
                        ['io_on_off_arb', 'hysteresis', [0.7, 240, 0.1], '', 'hlratio', False, 'price', 'ema1', 120]],
    'ioarb_spd_qtl_1y': [['rb_i', 'hc_i'],
                         ['io_on_off_arb', 'qtl', [240, 260, 2], 'sma2', '', False, 'price', '', 120]],

    'io_removal_lvl': [['rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'cu'],
                       ['io_removal_41ports', 'qtl', [20, 40, 2], '', '', True, 'price', 'sma2', 120]],
    'io_removal_lyoy': [['rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'cu'],
                        ['io_removal_41ports', 'qtl', [6, 10], 'lunar_yoy_day', 'diff', True, 'W-Fri', '', 120]],
    'io_inv_rmv_ratio_1y': [['i'],
                            ['io_inv_removal_ratio_41p', 'hlratio', [48, 56], 'df1', 'pct_change', False, '', "", 120]],
    'io_millinv_lvl': [['i'],
                       ['io_inv_mill(64)', 'qtl', [20, 40, 2], '', '', True, 'price', '', 120]],
    'io_millinv_lyoy': [['hc', 'i'],
                        ['io_inv_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'diff', True, 'W-Thu', 'sma4', 120]],
    'io_invdays_lvl': [['rb', 'hc', 'i', 'j', 'jm', 'v', 'FG', 'cu', 'al'],
                       ['io_invdays_imp_mill(64)', 'qtl', [20, 40, 2], '', 'pct_change', True, 'price', '', 120]],
    'io_invdays_lyoy': [['i'],
                        ['io_invdays_imp_mill(64)', 'qtl', [2, 4], 'lunar_yoy_day', 'pct_change', True, 'W-Thu', "ema3", 120]],
    'io_pinv31_lvl_zsa': [['rb_i', 'hc_i'],
                          ['io_inv_31ports', 'zscore_adj', [8, 56, 4], '', '', True, '', "sma2", 120]],
    'io_pinv45_lvl_hlr': [['rb_i', 'hc_i'],
                          ['io_inv_45ports', 'hlratio', [8, 56, 4], '', '', True, '', "sma2", 120]],
    # 'io_port_inv_lvl_slow': [['i'],
    #                          ['io_inv_imp_31ports_w', 'zscore', [240, 255, 5], '', '', False, 'price', "", 120]],
    'long_inv_mqs_st': [['rb', 'hc', 'i'],
                        ['long_inv_social', 'ma_dff_sgn', [40, 60, 2], '', '', False, 'price', "sma2", 120]],
    'long_inv_mqs_lt': [['rb', 'hc', 'i'],
                        ['long_inv_social', 'ma_dff_sgn', [240, 250, 2], '', '', False, 'price', "sma2", 120]],
    'long_inv_lyoy_mqs_st': [['rb', 'hc', 'i'], ['long_inv_social', 'ma_dff_sgn', [40, 60, 2],
                                                 'lunar_yoy_day', 'diff', False, 'price', "sma2", 120]],
    'long_inv_lyoy_mqs_lt': [['rb', 'hc', 'i'], ['long_inv_social', 'ma_dff_sgn', [240, 250, 2],
                                                 'lunar_yoy_day', 'diff', False, 'price', "sma2", 120]],
    'flat_inv_mqs_st': [['rb', 'hc', 'i'],
                        ['flat_inv_social', 'ma_dff_sgn', [40, 60, 2], '', '', False, 'price', "sma2", 120]],
    'flat_inv_mqs_lt': [['rb', 'hc', 'i'],
                        ['flat_inv_social', 'ma_dff_sgn', [240, 250, 2], '', '', False, 'price', "sma2", 120]],
    'steel_sinv_lyoy_zs': [['rb', 'hc', 'i', 'FG', 'v'],
                           ['steel_inv_social', 'zscore', [24, 30, 2], 'lunar_yoy_day', 'diff', False, '', "", 120]],
    'steel_sinv_lyoy_mds': [['rb', 'hc', 'i', 'FG', 'v'],
                           ['steel_inv_social', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, '', "", 120]],
    'rbhc_dmd_mds': [['rb-hc'],
                     ['rb_hc_dmd_diff', 'ma_dff_sgn', [5, 9, 1], '', '', True, '', "", 120]],
    'rbhc_dmd_lyoy_mds': [['rb-hc'],
                          ['rb_hc_dmd_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', True, '', "", 120]],
    'rbhc_sinv_mds': [['rb-hc'],
                      ['rb_hc_sinv_diff', 'ma_dff_sgn', [5, 9, 1], '', '', False, '', "", 120]],
    'rbhc_sinv_lyoy_mds': [['rb-hc'],
                           ['rb_hc_sinv_diff', 'ma_dff_sgn', [5, 9, 1], 'lunar_yoy_day', 'diff', False, '', "", 120]],
    # 'rb_sinv_lyoy_fast': ('rebar_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'wr_sinv_lyoy_fast': ('wirerod_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'hc_soinv_lyoy_fast': ('hrc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'cc_soinv_lyoy_fast': ('crc_inv_social', 'zscore', [20, 42, 2], 'lunar_yoy_day', 'diff', False, 'W-Fri'),
    # 'billet_inv_chg_slow': ('billet_inv_social_ts', 'zscore', [240, 252, 2], '', 'diff', False, 'price'),

    # 'margin_sea_lvl_mid': ('hrc_margin_sb', 'zscore', [40, 82, 2], '', 'pct_change', True, 'price'),
    'hrc_arb_mom': [['hc', 'rb'],
                    ['hrc_exp_sea_arb', 'qtl', [40, 80, 2], '', '', True, '', "ema1", 120]],
    'hrc_mb_saudi_qtl': [['hc'],
                         ['hrc_mb_cn_saudi', 'qtl', [30, 40, 2], '', '', True, '', "ema1", 120]],
    'hrc_mb_uae_qtl': [['hc'],
                       ['hrc_mb_cn_uae', 'qtl', [20, 30, 2], '', '', True, '', "ema1", 120]],
    'hrc_cn_eu_qtl': [['hc'],
                      ['hrc_cn_eu_cfr', 'qtl', [20, 30, 2], '', '', True, '', "ema1", 120]],
    'hrc_arb_ma': [['hc', 'rb'],
                   ['hrc_exp_sea_arb', 'ma', [1, 4, 1], '', '', True, '', "ema1", 120]],
    'steel_margin_lvl_fast': [['rb', 'hc', 'i', 'j'],
                              ['margin_hrc_macf', 'qtl', [20, 40, 2], '', '', True, 'price', "", 120]],
    'strip_hsec_lvl_mid': [['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM', 'cu'],
                           ['strip_hsec', 'qtl', [60, 80, 2], '', '', True, 'price', "", 120]],
    'macf_cfd_lvl_mid': [['i'],
                         ['macf_cfd', 'qtl', [40, 82, 2], '', '', True, 'price', "", 120]],
    'hc_rb_diff_lvl_fast': [['rb', 'hc', 'i', 'j', 'jm', 'cu', 'al'],
                            ['hc_rb_diff', 'zscore', [20, 40, 2], '', '', True, 'price', "", 120]],

    'cu_prem_usd_zsa': [['cu'],
                        ['cu_prem_bonded_warrant', 'zscore', [20, 30, 2], '', '', True, 'price', "ema1", 120]],
    'cu_prem_usd_md': [['cu'],
                       ['cu_prem_bonded_warrant', 'ma_dff', [20, 30, 2], '', '', True, 'price', "", 120]],
    'cu_phybasis_zsa': [['cu'],
                        ['cu_cjb_phybasis', 'zscore_adj', [40, 60, 2], 'sma10', '', True, 'price', "", 120]],  # great
    'cu_phybasis_hlr': [['cu'],
                        ['cu_cjb_phybasis', 'hlratio', [40, 60, 2], 'sma10', '', True, 'price', "", 120]],  # great
    'cu_scrap1_zs': [['cu'],
                     ['cu_1#_scrap_diff', 'zscore', [40, 60, 2], '', '', True, 'price', "", 120]],
    'cu_mine_tc_zs': [['cu'],
                      ['cu_mine_tc_cif_cn', 'zscore', [20, 30, 2], '', '', False, 'price', "", 120]],
    'cu_rod_procfee_for_qtl': [['cu'],
                      ['cu_rod_lowoxygen_8mm_procfee_for_gd', 'qtl', [20, 30, 2], '', '', True, 'price', "", 120]],
    # 'cu_scrap1_margin_gd': [['cu'],
    #                         ['cu_scrap1_diff_gd', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price', "", 120]],
    # 'cu_scrap1_margin_tj': [['cu'],
    #                         ['cu_scrap1_diff_tj', 'qtl', [40, 60, 2], '', 'pct_change', True, 'price', "", 120]],
    # 'cu_rod_procfee_2.6': [['cu'], ['cu_rod_2.6_procfee_nanchu',
    #                                 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price', "", 120]],
    # 'cu_rod_procfee_8.0': [['cu'], ['cu_rod_8_procfee_nanchu',
    #                                 'zscore_adj', [20, 30, 2], '', 'pct_change', True, 'price', "", 120]],
    'ni_nis_mom_qtl': [['ni'], ['ni_nis_cjb_spot', 'qtl', [10, 20, 2], '', '', True, '', "", 120]],
    'ni_ore_qtl': [['ni'], ['ni_1.8conc_spot_php_lianyungang', 'qtl', [10, 20], '', '', True, '', "", 120]],
    'al_alumina_qtl': [['al'],
                       ["alumina_spot_qd", 'qtl', [10, 20], '', '', True, 'price', "", 120]],
    'al_alumina_yoy_qtl': [['al'],
                           ["alumina_spot_qd", 'qtl', [20, 40], 'cal_yoy_day', 'pct_change', True, 'price', "", 120]],
    'al_coal_qtl': [['al'],
                    ['coal_5500_sx_qhd', 'qtl', [240, 250], '', '', True, 'price', "", 120]],
    'al_coal_yoy_qtl': [['al'],
                        ['coal_5500_sx_qhd', 'qtl', [20, 40], 'cal_yoy_day', 'pct_change', True, 'price', "", 120]],
    'sn_conc_spot_hlr': [['sn'], ['sn_60conc_spot_guangxi', 'hlratio', [40, 80, 2], '', '', True, '', "", 120]],

    'al_scrap_sh_zs': [['al'],
                       ['al_scrap_ex_diff_sh', 'zscore', [20, 30, 2], '', '', True, 'price', "", 120]],
    'zn_hrc_mom': [['zn'], ['hrc_sh', 'qtl', [20, 30], '', '', True, 'price', "", 120]],
    'pb_sec_margin_zs': [['pb'], ['pb_sec_margin', 'zscore', [230, 250, 2], '', '', False, 'price', "", 120]],
    'ni_npi_cn_zs': [['ni'], ['npi_10_spot_cn', 'zscore', [40, 80, 2], '', '', True, '', "", 120]],
    'ni_nsa_zs': [['ni'], ['ni_sul_spot', 'zscore', [40, 80, 2], '', '', True, '', "", 120]],
    'ni_briq_qtl': [['ni'], ['ni_briq_prem_spot', 'qtl', [40, 60, 2], '', '', True, '', "", 120]],
    'ni_npi_prem_hlr': [['ni'], ['ni_ni_prem', 'hlratio', [20, 60, 2], 'ema3', '', False, '', "", 120]],
    'ni_npi_imp_spd_hlr': [['ni'], ['npi_ferronickle_spd', 'hlratio', [20, 60, 2], '', '', False, 'price', "", 120]],

    'lme_base_ts_mds': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_base_ts', 'ma_dff_sgn', [10, 30, 2], '', '', True, 'price', "", 120]],
    'lme_base_ts_mds_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_base_ts', 'ma_dff_sgn', [10, 30, 2], '', '', True, 'price', "", 120]],
    'lme_base_ts_hlr': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_base_ts', 'hlratio', [10, 20, 1], '', '', True, 'price', "buf0.1", 120]],
    'lme_base_ts_hlr_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_base_ts', 'hlratio', [10, 20, 1], '', '', True, 'price', "buf0.1", 120]],
    'lme_futbasis_ma': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                        ['lme_futbasis', 'ma', [1, 2, 1], 'df1|ema1', 'diff', True, '', "buf0.5", 120]],
    'lme_futbasis_ma_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                ['lme_futbasis', 'ma', [1, 2, 1], 'df1|ema1', 'diff', True, '', "buf0.5", 120]],
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
    'base_cifprem_1y_zs': [['cu', 'al', 'zn', 'ni'],
                           ['prem_bonded_warrant', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120]],
    'base_cifprem_1y_zs_xdemean': [['cu', 'al', 'zn', 'ni'],
                                   ['prem_bonded_warrant', 'zscore', [230, 250, 2], '', '', True, 'price', "", 120]],
    'base_tc_1y_zs': [['cu', 'pb', 'zn'], ['base_tc', 'zscore', [230, 250, 2], '', '', False, 'price', "", 120]],
    'base_tc_2y_zs': [['cu', 'pb', 'sn'], ['base_tc', 'zscore', [480, 500, 2], '', '', False, 'price', "", 120]],
    'base_inv_mds': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                     ['base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price', "", 120]],
    'base_inv_mds_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                             ['base_inv', 'ma_dff_sgn', [180, 240, 2], '', '', False, 'price', "", 120]],
    'base_inv_shfe_ma': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                         ['inv_shfe_d', 'ma', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240]],
    'base_inv_shfe_ma_xdemean': [['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
                                 ['inv_shfe_d', 'ma', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240]],
    'base_inv_lme_ma': [['cu', 'zn', 'pb', 'ni', 'sn'],
                        ['inv_lme_total', 'ma', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240]],
    'base_inv_lme_ma_xdemean': [['cu', 'zn', 'pb', 'ni', 'sn'],
                                ['inv_lme_total', 'ma', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.2", 240]],
    'base_inv_exch_ma': [['cu', 'zn', 'pb', 'ni', 'sn'],
                         ['inv_exch_d', 'ma', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.15", 240]],
    'base_inv_exch_ma_xdemean': [['cu', 'zn', 'pb', 'ni', 'sn'],
                                 ['inv_exch_d', 'ma', [1, 2, 1], 'df1|ema1', 'pct_change', False, '', "buf0.15", 240]],
    'lme_mr_qtl_xdemean': [['MCU', 'MAL', 'MZN', 'MPB'],
                          ['px', 'qtl', [20, 40, 2], '', '', False, '', "ema5", 240]],
    'lme_mr_zs_xdemean': [['MCU', 'MAL', 'MZN', 'MPB'],
                          ['px', 'zscore', [20, 160, 5], '', '', False, '', "ema4", 240]],
    'lme_inv_total_st': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                         ['inv_lme_total', 'qtl', [5, 10, 1], '', '', False, 'price', "", 120]],
    'lme_inv_total_st_xdemean': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                                 ['inv_lme_total', 'qtl', [5, 10, 1], '', '', False, 'price', "", 120]],
    'lme_inv_cancelled_st': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                         ['inv_lme_cancelled', 'qtl', [10, 20, 1], '', '', True, 'price', "", 120]],
    'lme_inv_cancelled_st_xdemean': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                                     ['inv_lme_cancelled', 'qtl', [10, 20, 1], '', '', True, 'price', "", 120]],
    'lmebase_arb_zsa_1m': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                           ['shfe_arb', 'zscore_adj', [20, 40, 1], 'ema3', '', True, 'price', "", 120]],
    'lmebase_arb_zsa_1m_xdemean': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                                   ['shfe_arb', 'zscore_adj', [20, 40, 1], 'ema3', '', True, 'price', "", 120]],
    'lmebase_arb_ma': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                       ['shfe_arb', 'ma', [1, 2], '', '', True, 'price', "", 240]],
    'lmebase_arb_ma_xdemean': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                               ['shfe_arb', 'ma', [1, 2], '', '', True, 'price', "", 240]],
    'lmebase_long_2y': [['MCU', 'MAL', 'MZN', 'MPB', 'MNI'],
                        ['shfe_arb', 'hysteresis', [0.9, 500, 0.1], '', 'pct_score', True, 'price', "pos", 240]],

    'metal_pbc_ema': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                      ['metal_pbc', 'ema', [10, 20], '', '', True, 'price', "", 120]],
    'metal_pbc_ema_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v',
                               'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                              ['metal_pbc', 'ema', [10, 20], '', '', True, 'price', "", 120]],
    'metal_mom_hlrhys': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                         ['metal_px', 'hysteresis', [0.7, 60, 0.1], '', 'hlratio', True, 'price', "ema1", 120]],
    'metal_mom_hlrhys_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                         ['metal_px', 'hysteresis', [0.7, 60, 0.1], '', 'hlratio', True, 'price', "ema1", 120]],
    'metal_inv_hlr': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                      ['metal_inv', 'hlratio', [240, 250], '', '', False, 'price', "", 120]],
    'metal_inv_hlr_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v',
                               'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                              ['metal_inv', 'hlratio', [240, 250, 2], '', '', False, 'price', "", 120]],
    'metal_inv_lyoy_hlr': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v',
                            'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                           ['metal_inv', 'hlratio', [240, 250], 'lunar_yoy_day', 'pct_change', False, 'price', "", 120]],
    'metal_inv_lyoy_hlr_xdemean': [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v',
                                    'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
                                   ['metal_inv', 'hlratio', [240, 250], 'lunar_yoy_day',
                                    'pct_change', False, 'price', "", 120]],
    "momhys": [['i', 'rb', 'hc', 'jm', 'j', 'SM', 'SF', 'FG', 'v', 'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss'],
               ["px", "hysteresis", [0.7, 20, 0.5], "ema3", "zscore_roll", True, "", "", 120]],
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

    "ryield_ema_ts": [broad_mkts, ["ryield", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "ryield_ema_xdemean": [broad_mkts, ["ryield", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "ryield_zsa_ts": [broad_mkts, ["ryield", "zscore_adj", [20, 30, 1], "", "", True, "price", "ema1", 240]],
    "ryield_zsa_xdemean": [broad_mkts, ["ryield", "zscore_adj", [20, 30, 1], "", "", True, "price", "ema1", 240]],
    "basmom5_ema_ts": [broad_mkts, ["basmom5", "ema", [10, 20, 1], "", "", True, "price", "", 240]],
    "basmom5_ema_xdemean": [broad_mkts, ["basmom5", "ema", [10, 20, 1], "", "", True, "price", "", 240]],
    "basmom10_ema_ts": [broad_mkts, ["basmom10", "ema", [10, 20, 1], "", "", True, "price", "", 240]],
    "basmom10_ema_xdemean": [broad_mkts, ["basmom10", "ema", [10, 20, 1], "", "", True, "price", "", 240]],
    "basmom10_qtl_ts": [broad_mkts, ["basmom10", "qtl", [230, 250, 2], "ema5", "", True, "price", "", 240]],
    "basmom10_qtl_xdemean": [broad_mkts, ["basmom10", "qtl", [230, 250, 2], "ema5", "", True, "price", "", 240]],
    "basmom20_ema_ts": [broad_mkts, ["basmom20", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "basmom20_ema_xdemean": [broad_mkts, ["basmom20", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "basmom60_ema_ts": [broad_mkts, ["basmom60", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "basmom60_ema_xdemean": [broad_mkts, ["basmom60", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "basmom120_ema_ts": [broad_mkts, ["basmom120", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "basmom120_ema_xdemean": [broad_mkts, ["basmom120", "ema", [1, 2, 1], "", "", True, "price", "ema1", 240]],
    "exch_wnt_hlr": [
        ['ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [230, 250, 2], "", "", False, "", "ema3", 240]],
    "exch_wnt_hlr_xdemean": [
        ['ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [240, 260, 2], "", "", False, "", "ema3", 240]],
    "exch_wnt_yoy_hlr": [
        ['ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [230, 250, 2], 'cal_yoy_day', "diff", False, "", "ema3", 240]],
    "exch_wnt_yoy_hlr_xdemean": [
        ['ss', 'SA', 'FG', 'l', 'pp', 'v', 'TA', 'MA', 'eg', 'bu', 'fu', 'a', 'c', 'CF'],
        ["exch_warrant", "hlratio", [240, 260, 2], 'cal_yoy_day', "diff", False, "", "ema3", 240]],

    "cgb_1_2_spd_zs": [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                       ["cgb_1_2_spd", "zscore", [40, 80, 2], "ema5", "", True, "", "", 120]],
    'dxy_zsa_s': [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                  ['dxy', 'zscore_adj', [20, 30, 2], '', '', False, '', 'hmp0.5', 120]],
    'shibor1m_qtl': [['cu', 'al', 'zn', 'rb', 'hc', 'i'],
                     ['shibor_1m', 'qtl', [40, 80, 2], 'ema3', '', True, 'price', '', 120]],
    "MCU3_zs": [['cu', 'zn', 'ni', 'al', 'sn', 'ao', 'rb', 'hc', 'i', 'v', 'j', 'jm'],
                ['cu_lme_3m_close', 'zscore', [40, 80, 2], '', '', True, '', 'buf0.2', 120]],
    "MAL3_zs": [['cu', 'zn', 'ni', 'al', 'sn', 'ao', 'rb', 'hc', 'i', 'v', 'j', 'jm'],
                ['al_lme_3m_close', 'zscore', [40, 80, 2], '', '', True, '', 'buf0.2', 120]],
    # 'r007_qtl': ('r007_cn', 'qtl', [80, 120, 2], 'ema5', 'pct_change', True, 'price'),
    # 'r_dr_spd_zs': ('r_dr_7d_spd', 'zscore', [20, 40, 2], 'ema5', 'pct_change', True, 'price'),
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

signal_buffer_config = {
    "lme_base_ts_hlr": 0.1,
    "base_inv_exch_ma": 0.15,
    "base_inv_lme_ma": 0.2,
    "base_inv_shfe_ma": 0.2,
    "lme_futbasis_ma": 0.75,
}

signal_execution_win = {
    'lme_base_ts_mds': "a1505",
    'lme_base_ts_mds_xdemean': "a1505",
    'lme_base_ts_hlr': "a1505",
    'lme_base_ts_hlr_xdemean': "a1505",
    'lme_futbasis_ma': "a1505",
    'lme_futbasis_ma_xdemean': "a1505",
    'rbsales_lyoy_spd_st': "a1505",
    'rbsales_lyoy_mom_st': "a1505",
    'rbsales_lyoy_mom_lt': "a1505",
    'rb_sales_inv_ratio_lyoy': "a1505",
    'ioarb_px_hlr': "a1505",
    'ioarb_px_hlrhys': "a1505",
    'ioarb_spd_qtl_1y': "a1505",
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
        'ss': "ss_inv_social_300",
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
    'metal_px': {},
    'inv_shfe_d': {},
    'lme_futbasis': {},
    'inv_lme_total': {},
    'inv_exch_d': {},
    "exch_warrant": {},
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


def hc_rb_diff(df, input_args):
    shift_mode = input_args['shift_mode']
    product_list = input_args['product_list']
    win = input_args['win']
    xdf = df.loc[:, df.columns.get_level_values(0).isin(['rbc1', 'hcc1'])].copy(deep=True)
    xdf = xdf['2014-07-01':]
    if shift_mode == 2:
        xdf[('rbc1', 'px_chg')] = np.log(xdf[('rbc1', 'close')]).diff()
        xdf[('hcc1', 'px_chg')] = np.log(xdf[('hcc1', 'close')]).diff()
    else:
        xdf[('rbc1', 'px_chg')] = xdf[('rbc1', 'close')].diff()
        xdf[('hcc1', 'px_chg')] = xdf[('hcc1', 'close')].diff()
    hc_rb_diff = xdf[('hcc1', 'px_chg')] - xdf[('rbc1', 'px_chg')]
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
    for prod in product_list:
        for sector in leadlag_port:
            if prod[:-2] in leadlag_port[sector]['lag']:
                signal_list = []
                for lead_prod in leadlag_port[sector]['lead']:
                    feature_ts = df[(lead_prod+'c1', 'close')]
                    signal_ts = calc_conv_signal(feature_ts.dropna(), conv_func,
                                                 leadlag_port[sector]['param_rng'], signal_cap=signal_cap)
                    signal_list.append(signal_ts)
                signal_df[prod] = pd.concat(signal_list, axis=1).mean(axis=1)
                break
            else:
                signal_df[prod] = 0
    return signal_df


def mr_pair(df, input_args):
    mr_pair_list = mr_commod_pairs
    product_list = input_args['product_list']
    signal_cap = input_args.get('signal_cap', None)
    conv_func = input_args.get('conv_func', 'zscore_adj')
    param_rng = input_args.get('params', [200, 250, 2])
    vol_win = input_args.get('vol_win', 120)
    signal_df = pd.DataFrame(0, index=df.index, columns=product_list)
    bullish = False
    for (asset_a, asset_b) in mr_pair_list:
        pair_assets = [asset_a, asset_b]
        sig_df = pd.DataFrame(index=df.index, columns=pair_assets)
        feature_ts = np.log(df[(asset_a+'c1', 'close')]) - np.log(df[(asset_b+'c1', 'close')])
        sig_ts = calc_conv_signal(feature_ts, signal_func=conv_func, param_rng=param_rng, signal_cap=signal_cap,
                                  vol_win=vol_win)
        sig_ts = sig_ts.apply(lambda x: np.sign(x) * min(abs(x), 1.25) ** 4).ewm(1).mean()
        if not bullish:
            sig_ts = -sig_ts
        sig_df[asset_a+'c1'] = sig_ts
        sig_df[asset_b+'c1'] = -sig_ts
        signal_df = signal_df + sig_df.reindex_like(signal_df).fillna(0)
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
    for prod in product_list:
        signal_df[prod] = signal_ts
    return signal_df


def get_funda_signal_from_store(spot_df, signal_name, price_df=None,
                                signal_cap=None, asset=None,
                                curr_date=None, signal_repo=signal_store,
                                feature_key_map=feature_to_feature_key_mapping):
    feature, signal_func, param_rng, proc_func, chg_func, bullish, freq, post_func, vol_win = \
        signal_repo[signal_name][1]
    if asset and feature in feature_key_map:
        asset_feature = feature_key_map[feature].get(asset, f'{asset}_{feature}')
        if feature == 'metal_pbc':
            if price_df is None:
                print("ERROR: no future price is passed for metal_pbc")
                return pd.Series()
            spot_df['date'] = pd.to_datetime(spot_df.index)
            spot_df[f'{asset}_c1'] = price_df[(asset+'c1', 'close')] / np.exp(price_df[(asset+'c1', 'shift')])
            spot_df[f'{asset}_expiry'] = pd.to_datetime(price_df[(asset+'c1', 'expiry')])
            if asset == 'i':
                spot_df['io_ctd_spot'] = io_ctd_basis(spot_df, price_df[('i'+'c1', 'expiry')])
            spot_df[f'{asset}_phybasis'] = (np.log(spot_df[asset_feature]) - np.log(spot_df[f'{asset}_c1'])) / \
                                           (spot_df[f'{asset}_expiry'] - spot_df['date']).dt.days * 365
            asset_feature = f'{asset}_phybasis'
        if feature == 'metal_px':
            if price_df is None:
                print("ERROR: no future price is passed for metal_pbc")
                return pd.Series()
            spot_df[f'{asset}_px'] = price_df[(asset+'c1', 'close')]
            asset_feature = f'{asset}_px'
        if feature in param_rng_by_feature_key:
            param_rng = param_rng_by_feature_key[feature].get(asset, param_rng)
        if feature in proc_func_by_feature_key:
            proc_func = param_rng_by_feature_key[feature].get(asset, proc_func)
        feature = asset_feature
    signal_ts = calc_funda_signal(spot_df, feature, signal_func, param_rng,
                                  proc_func=proc_func, chg_func=chg_func,
                                  bullish=bullish, freq=freq, signal_cap=signal_cap,
                                  post_func=post_func, vol_win=vol_win, curr_date=curr_date)
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
        signal_df = pd.DataFrame(index=pd.date_range(start=df.index[0], end=df.index[-1], freq='D'))
        for product in product_list:
            asset = product[:-2]
            signal_df[product] = get_funda_signal_from_store(funda_df, signal_name,
                                                             price_df=df, signal_cap=signal_cap, asset=asset)
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
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
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
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
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
        signal_ts = get_funda_signal_from_store(funda_df, signal_name, price_df=df, signal_cap=signal_cap)
        signal_ts = signal_ts.reindex(
            index=pd.date_range(start=df.index[0],
                                end=df.index[-1],
                                freq='D'
                                )).ffill().reindex(index=df.index)
        signal_df = pd.DataFrame(dict([(asset, signal_ts) for asset in product_list]))
    _, _, _, _, _, _, _, post_func, _ = signal_store[signal_name][1]
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
