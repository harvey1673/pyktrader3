# Research fundamental spread signals

These formulas are test-side candidates built from current `spot_idx_map` aliases. They are not wired into production signals. Import-parity proxies omit any freight, port, financing or quality term not stated in the formula note.

| Signal | Products | Type | Inputs | History | Chinese description |
|---|---|---|---|---|---|
| `MA_taicang_curve` | MA | calendar_spread | `MA_taicang_paper_lm_ms`, `MA_taicang_paper_nm_ms` | 2021-01-04 to 2026-09-30 (1410) | 太仓甲醇当月下旬-次月下旬纸货 |
| `eb_N1_N2_curve` | eb | calendar_spread | `eb_jiangsu_n1`, `eb_jiangsu_n2` | 2020-07-24 to 2026-09-30 (1504) | 江苏苯乙烯N+1-N+2 |
| `fu_380_m1_m2_curve` | fu | calendar_spread | `fo_380cst_m1_sgp`, `fo_380cst_m2_sgp` | 2020-02-17 to 2026-10-02 (1662) | 新加坡380CST近月-次月纸货 |
| `SA_heavy_light_north_grade` | SA | grade_differential | `SA_heavy_north`, `SA_light_north` | 2009-08-19 to 2026-09-30 (4153) | 华北重质纯碱-轻质纯碱 |
| `SH_50_32_dry_basis` | SH | grade_differential | `SH_50_spot_sdjl_shandong`, `SH_32_spot_sdjl_shandong` | 2015-06-24 to 2026-09-29 (2746) | 山东50%液碱与32%液碱折百价差 |
| `br_qilu_yangzi_brand_location` | br | grade_differential | `br9000_qilu_sd`, `br9000_yangzi_sh` | 2017-09-15 to 2026-09-30 (1672) | 山东齐鲁BR9000-上海扬子BR9000 |
| `fu_380_180_grade` | fu | grade_differential | `fo_380cst_sgp_fob`, `fo_180cst_sgp_fob` | 2016-06-01 to 2026-09-29 (2372) | 新加坡船用380CST-180CST |
| `io_pbf_blend_grade_spread` | i | grade_differential | `pbf_qd`, `iocj_qd`, `ssf_qd` | 2017-06-26 to 2026-09-30 (2307) | PB粉-40%IOCJ-60%超特粉 |
| `lc_battery_industrial_grade` | lc | grade_differential | `lc_bat_dom_cn_spot`, `lc_ind_dom_cn_spot` | 2023-04-20 to 2026-09-30 (837) | 国产电池级-工业级碳酸锂 |
| `ni_jinchuan_import_grade` | ni | grade_differential | `ni_smm1_jc_spot`, `ni_smm1_imp_spot` | 2011-06-28 to 2026-09-30 (3664) | 金川镍-进口镍 |
| `pb_primary_secondary_grade` | pb | grade_differential | `pb_smm1_spot`, `pb_sec9997_spot` | 2011-07-29 to 2026-09-30 (3562) | 原生1#铅-再生精铅 |
| `pg_propane_butane_south` | pg | grade_differential | `propane_cfr_south`, `butane_cfr_south` | 2019-09-09 to 2026-09-30 (1777) | 华南CFR丙烷-丁烷 |
| `si_421_553_east_grade` | si | grade_differential | `si_421_east`, `si_553_nonoxy_east` | 2018-12-28 to 2026-09-30 (1881) | 华东421工业硅-553不通氧工业硅 |
| `ss_gross_hongwang_grade` | ss | grade_differential | `ss_304_gross_wuxi`, `ss_304_2b_hongwang_wuxi` | 2019-07-01 to 2026-09-30 (1762) | 无锡304/2B毛边卷-无锡宏旺2*1240*C |
| `pvc_ethylene_cac2_route` | v | grade_differential | `pvc_ethylene_east`, `pvc_cac2_east` | 2009-01-04 to 2026-09-30 (4402) | 华东乙烯法PVC-华东电石法PVC |
| `MA_cfr_import_parity` | MA | import_export_arb | `MA_cfr_cn`, `usdcny_xe`, `MA_spot_jiangsu` | 2012-01-04 to 2026-09-29 (3553) | 甲醇CFR中国完税代理-江苏现货 |
| `PX_cfr_fob_freight_proxy` | PX | import_export_arb | `PX_cfr_tw_usd`, `PX_fob_kr_usd` | 2010-05-03 to 2026-09-29 (4147) | PX CFR台湾-FOB韩国 |
| `TA_cfr_import_parity` | TA | import_export_arb | `TA_cfr_cn`, `usdcny_xe`, `TA_east_spot` | 2022-06-13 to 2026-09-30 (1044) | PTA CFR中国完税代理-华东现货 |
| `eb_cfr_import_parity` | eb | import_export_arb | `eb_cfr_cn_mid`, `usdcny_xe`, `eb_east_spot` | 2012-01-04 to 2026-09-30 (3582) | 苯乙烯CFR中国完税代理-华东现货 |
| `eg_cfr_import_parity` | eg | import_export_arb | `eg_cfr_nea`, `usdcny_xe`, `eg_east_spot` | 2022-02-07 to 2026-09-30 (1153) | MEG东北亚CFR完税代理-华东现货 |
| `l_cfr_cn_import_parity` | l | import_export_arb | `l_lldpe_cfr_cn`, `usdcny_xe`, `l_7042_east` | 2009-08-14 to 2026-09-29 (4090) | LLDPE CFR中国完税代理-华东7042 |
| `sp_silver_import_parity` | sp | import_export_arb | `sp_silver_cfr`, `usdcny_xe`, `sp_silver_sd` | 2019-08-09 to 2026-09-30 (1774) | 银星CFR完税代理-山东现货 |
| `FG_shahe_north_location` | FG | location_spread | `FG_5mm_shahe`, `FG_5mm_north` | 2020-08-11 to 2026-09-30 (1532) | 沙河5mm大板玻璃-华北5mm玻璃 |
| `MA_jiangsu_neimeng_location` | MA | location_spread | `MA_spot_jiangsu`, `MA_spot_neimeng` | 2012-01-04 to 2026-09-30 (3646) | 江苏甲醇-内蒙古甲醇 |
| `bu_shandong_east_location` | bu | location_spread | `bu_heavy_shandong`, `bu_heavy_east` | 2013-10-22 to 2026-09-30 (3212) | 山东重交沥青-华东重交沥青 |
| `eb_shandong_east_location` | eb | location_spread | `eb_shandong_delivered`, `eb_east_selfpickup` | 2022-02-07 to 2026-09-30 (1157) | 山东苯乙烯送到-华东自提 |
| `eg_sh_east_location` | eg | location_spread | `eg_sh_spot_ms`, `eg_east_spot_ms` | 2022-02-15 to 2026-09-30 (1155) | MEG上海-华东 |
| `l_7042_north_east_location` | l | location_spread | `l_7042_north`, `l_7042_east` | 2009-08-14 to 2026-09-30 (4236) | LLDPE7042华北均价-华东均价 |
| `l_7042_tj_sh_location` | l | location_spread | `l_7042_tj`, `l_7042_sh` | 2009-01-04 to 2026-09-30 (4373) | 大庆7042天津-上海 |
| `lu_zhoushan_qingdao_location` | lu | location_spread | `lu_bonded_zhoushan`, `lu_bonded_qingdao` | 2021-01-28 to 2026-09-30 (1413) | 舟山-青岛保税低硫船用油 |
| `pg_propane_south_east_location` | pg | location_spread | `propane_cfr_south`, `propane_cfr_east` | 2019-09-09 to 2026-09-30 (1777) | 丙烷CFR华南-华东 |
| `pp_daqing_east_exw_channel` | pp | location_spread | `pp_t30s_daqing_east`, `pp_t30s_daqing_exw` | 2016-01-04 to 2026-09-30 (2645) | 大庆T30S中油华东-企业出厂 |
| `pp_shaoxing_daqing_east_location` | pp | location_spread | `pp_t30s_shaoxing_hz`, `pp_t30s_daqing_east` | 2016-01-04 to 2026-09-30 (2503) | 杭州绍兴三圆T30S-大庆中油华东 |
| `ru_jiangsu_kunming_location` | ru | location_spread | `ru_scrwf_jiangsu`, `ru_scrwf_kunming` | 2015-06-29 to 2026-09-30 (2781) | 江苏全乳胶-昆明全乳胶 |
| `sp_silver_sd_jzh_location` | sp | location_spread | `sp_silver_sd`, `sp_silver_jzh` | 2019-08-09 to 2026-09-30 (1774) | 银星针叶浆山东-江浙沪 |
| `pvc_east_north_location` | v | location_spread | `pvc_cac2_east`, `pvc_cac2_north` | 2009-01-04 to 2026-09-30 (4393) | 华东电石法PVC-华北电石法PVC |
| `pvc_south_east_location` | v | location_spread | `pvc_cac2_south`, `pvc_cac2_east` | 2009-01-04 to 2026-09-30 (4369) | 华南电石法PVC-华东电石法PVC |
| `TA_PX_processing_proxy` | TA,PX | processing_margin | `TA_east_spot`, `PX_exw_east_spot` | 2009-01-04 to 2026-09-30 (4383) | PTA-PX加工差代理 |
| `eb_benzene_processing_proxy` | eb | processing_margin | `eb_east_spot`, `bz_east_spot` | 2012-01-04 to 2026-09-30 (3662) | 华东苯乙烯-华东纯苯 |
| `crc_hrc_processing_proxy` | hc | processing_margin | `crc_sh`, `hrc_sh` | 2010-09-25 to 2026-09-30 (3995) | 上海冷轧卷板-上海热轧卷板 |
| `hrc_pbf_coke_margin_proxy` | hc | processing_margin | `hrc_sh`, `pbf_cfd`, `coke_xuzhou_xb` | 2011-09-16 to 2026-09-30 (3670) | 热卷-铁矿-焦炭原料成本代理 |
| `pp_propylene_processing_proxy` | pp | processing_margin | `pp_t30s_shaoxing_hz`, `pl_east_spot` | 2014-09-16 to 2026-09-30 (2826) | 华东PP T30S-华东丙烯 |
| `rebar_billet_processing_proxy` | rb | processing_margin | `rebar_sh`, `billet_ts` | 2010-03-01 to 2026-09-30 (4134) | 上海螺纹钢-唐山钢坯 |
| `SM_SF_tianjin_product_spread` | SM,SF | product_spread | `SM_65s17_tj`, `SF_72_tj` | 2010-09-25 to 2026-09-30 (3988) | 天津锰硅65/17-硅铁72 |
| `lu_hsfo_zhoushan_product` | lu,fu | product_spread | `lu_05_zhoushan`, `fo_bonded_highsulfur_zhoushan` | 2021-01-28 to 2026-09-30 (1376) | 舟山0.5%低硫船用油-保税高硫船用油 |
| `rb_hc_spot_spread` | rb,hc | product_spread | `rebar_sh`, `hrc_sh` | 2007-05-18 to 2026-09-30 (4836) | 上海螺纹钢-上海热轧卷板 |
| `ru_br_product_spread` | ru,br | product_spread | `ru_scrwf_jiangsu`, `br9000_yangzi_sh` | 2017-09-15 to 2026-09-30 (1657) | 江苏全乳胶-上海扬子BR9000 |
| `coke_rizhao_outstock_pingcang` | j | quote_basis_spread | `coke_sub_a_rz_outstock`, `coke_sub_a_rz` | 2019-06-12 to 2026-09-30 (1820) | 日照港准一级焦出库价-平仓价 |
| `jm_ganqimaodu_quote_spread` | jm | quote_basis_spread | `ckc_outstock_ganqimaodu`, `ckc_stock_ganqimaodu` | 2021-09-30 to 2026-09-30 (1097) | 甘其毛都主焦煤库提含税价-库存提货价 |

Use the level and change of each series as separate candidates. Apply publication-date lags before backtesting weekly vendor series, and compare results after transaction costs and by subperiod before adding a formula to `signal_repo`.
