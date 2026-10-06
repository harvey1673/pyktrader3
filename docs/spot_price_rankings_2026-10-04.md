# Spot/futures price-change correlation ranking

Weekly Friday log changes from 2016 onward; 70% chronological train and 30% test. The score rewards positive, stable recent correlation and penalizes train/test drift. It is a screening statistic, not evidence of causality or signal profitability.

| Product | Rank | Alias | Source | Weekly obs | Full rho | Test rho | Score | Status |
|---|---:|---|---|---:|---:|---:|---:|---|
| rb | 1 | `rebar_sh` | ifind | 534 | 0.735 | 0.837 | 0.747 | ranked |
| rb | 2 | `rebar_cn_fob` | ifind | 146 | 0.815 | 0.734 | 0.429 | ranked |
| rb | 3 | `rebar_sea_cfr` | ifind | 148 | 0.229 | 0.405 | 0.143 | ranked |
| hc | 1 | `hrc_sh` | ifind | 534 | 0.869 | 0.964 | 0.880 | ranked |
| hc | 2 | `hrc_tj` | ifind | 534 | 0.791 | 0.913 | 0.801 | ranked |
| hc | 3 | `hrc_cn_fob` | ifind | 146 | 0.906 | 0.931 | 0.512 | ranked |
| hc | 4 | `hrc_sea_cfr` | ifind | 148 | 0.317 | 0.441 | 0.187 | ranked |
| hc | 5 | `hrc_bs_fob` | ifind | 148 | 0.076 | 0.204 | 0.051 | ranked |
| i | 1 | `jmb_qd` | ifind | 355 | 0.907 | 0.930 | 0.910 | ranked |
| i | 2 | `macf_qd` | ifind | 461 | 0.899 | 0.953 | 0.907 | ranked |
| i | 3 | `plt62` | ifind | 534 | 0.902 | 0.921 | 0.905 | ranked |
| i | 4 | `pbf_qd` | ifind | 534 | 0.871 | 0.962 | 0.884 | ranked |
| i | 5 | `pbf_cfd` | ifind | 534 | 0.867 | 0.950 | 0.878 | ranked |
| j | 1 | `coke_sub_a_rz_outstock` | ifind | 365 | 0.542 | 0.641 | 0.544 | ranked |
| j | 2 | `coke_sub_a_tj` | ifind | 534 | 0.157 | 0.038 | 0.082 | ranked |
| j | 3 | `coke_sub_a_rz` | ifind | 436 | 0.146 | 0.041 | 0.080 | ranked |
| j | 4 | `coke_ts_xb` | ifind | 534 | 0.102 | 0.061 | 0.076 | ranked |
| j | 5 | `coke_changzhi_xb` | ifind | 534 | 0.079 | 0.056 | 0.064 | ranked |
| jm | 1 | `ckc_outstock_ganqimaodu` | ifind | 218 | 0.472 | 0.697 | 0.395 | ranked |
| jm | 2 | `ckc_stock_ganqimaodu` | ifind | 465 | 0.229 | 0.374 | 0.222 | ranked |
| jm | 3 | `ckc_a9v18s10_lvliang` | ifind | 534 | 0.219 | 0.440 | 0.222 | ranked |
| jm | 4 | `ckc_a10v24s08_lvliang` | ifind | 534 | 0.210 | 0.341 | 0.207 | ranked |
| jm | 5 | `ckc_midsulfur_jiexiu_kaijia` | mysteel | 278 | 0.174 | 0.168 | 0.170 | ranked |
| SM | 1 | `SM_65s17_tj` | ifind | 486 | 0.502 | 0.910 | 0.528 | ranked |
| SM | 2 | `SM_65s17_gansu` | ifind | 486 | 0.385 | 0.691 | 0.407 | ranked |
| SM | 3 | `SM_65s17_neimeng` | ifind | 486 | 0.378 | 0.691 | 0.406 | ranked |
| SM | 4 | `SM_65s17_shmet` | ifind | 486 | 0.182 | 0.329 | 0.192 | ranked |
| SM | 5 | `SM_dprod_cn` | ifind | 346 | -0.030 | 0.057 | -0.025 | ranked |
| SF | 1 | `SF_72_tj` | mysteel | 479 | 0.546 | 0.725 | 0.564 | ranked |
| SF | 2 | `SF_72_ningxia` | ifind | 358 | 0.508 | 0.616 | 0.520 | ranked |
| SF | 3 | `SF_72_gansu` | ifind | 358 | 0.466 | 0.593 | 0.478 | ranked |
| SF | 4 | `SF_72_neimeng` | ifind | 358 | 0.427 | 0.572 | 0.444 | ranked |
| SF | 5 | `SF_72_tj_if` | ifind | 401 | 0.389 | 0.650 | 0.411 | ranked |
| SA | 1 | `SA_heavy_shahe` | ifind | 338 | 0.431 | 0.787 | 0.460 | ranked |
| SA | 2 | `SA_heavy_sys` | ifind | 314 | 0.156 | 0.209 | 0.156 | ranked |
| SA | 3 | `SA_light_north` | ifind | 338 | 0.173 | 0.059 | 0.100 | ranked |
| SA | 4 | `SA_light_east` | ifind | 338 | 0.168 | 0.021 | 0.075 | ranked |
| SA | 5 | `SA_heavy_east` | ifind | 338 | 0.201 | 0.002 | 0.074 | ranked |
| FG | 1 | `FG_5mm_shahe` | ifind | 306 | 0.414 | 0.490 | 0.414 | ranked |
| FG | 2 | `FG_5mm_north` | ifind | 306 | 0.288 | 0.322 | 0.283 | ranked |
| FG | 3 | `FG_100ppi` | ifind | 532 | 0.190 | 0.191 | 0.175 | ranked |
| FG | 4 | `FG_weekly_melt` | ifind | 215 | 0.046 | 0.067 | 0.043 | ranked |
| FG | 5 | `FG_dprod` | ifind | 215 | -0.007 | 0.014 | -0.006 | ranked |
| v | 1 | `pvc_cac2_east` | ifind | 532 | 0.784 | 0.930 | 0.799 | ranked |
| v | 2 | `pvc_cac2_sh` | mysteel | 534 | 0.774 | 0.886 | 0.784 | ranked |
| v | 3 | `pvc_cac2_south` | ifind | 530 | 0.725 | 0.910 | 0.744 | ranked |
| v | 4 | `pvc_cac2_tj` | mysteel | 502 | 0.685 | 0.816 | 0.696 | ranked |
| v | 5 | `pvc_cac2_north` | ifind | 532 | 0.649 | 0.782 | 0.659 | ranked |
| SH | 1 | `SH_32_spot_sdjl_shandong` | ifind | 151 | 0.196 | 0.070 | 0.068 | ranked |
| SH | 2 | `SH_50_spot_sdjl_shandong` | ifind | 151 | 0.140 | 0.101 | 0.067 | ranked |
| cu | 1 | `cu_cjb_spot` | ifind | 534 | 0.913 | 0.902 | 0.906 | ranked |
| cu | 2 | `cu_smm1_spot` | ifind | 534 | 0.914 | 0.900 | 0.905 | ranked |
| cu | 3 | `cu_spot_sh` | ifind | 534 | 0.911 | 0.894 | 0.900 | ranked |
| cu | 4 | `cu_scrap_2_spot_jzh` | ifind | 523 | 0.889 | 0.896 | 0.890 | ranked |
| cu | 5 | `cu_scrap_1_spot_jzh` | ifind | 523 | 0.894 | 0.887 | 0.890 | ranked |
| al | 1 | `al_smm0_spot` | ifind | 534 | 0.873 | 0.907 | 0.878 | ranked |
| al | 2 | `al_cjb_spot` | ifind | 534 | 0.869 | 0.906 | 0.874 | ranked |
| al | 3 | `al_scrap_shredded_sh_high` | ifind | 478 | 0.857 | 0.869 | 0.859 | ranked |
| al | 4 | `al_scrap_shredded_sh_low` | ifind | 478 | 0.857 | 0.867 | 0.859 | ranked |
| al | 5 | `al_scrap_shreded_spot_foshan` | ifind | 479 | 0.739 | 0.802 | 0.747 | ranked |
| zn | 1 | `zn_smm0_spot` | ifind | 534 | 0.916 | 0.933 | 0.918 | ranked |
| zn | 2 | `zn_cjb_spot` | ifind | 534 | 0.911 | 0.934 | 0.913 | ranked |
| zn | 3 | `zn_scrap_sh_low` | ifind | 525 | 0.882 | 0.905 | 0.886 | ranked |
| zn | 4 | `zn_scrap_sh_high` | ifind | 525 | 0.882 | 0.905 | 0.885 | ranked |
| zn | 5 | `zn_lme_3m_close` | ifind | 534 | 0.743 | 0.705 | 0.719 | ranked |
| ni | 1 | `ni_cj1_spot` | ifind | 478 | 0.907 | 0.945 | 0.913 | ranked |
| ni | 2 | `ni_scrap_spot_foshan` | ifind | 479 | 0.904 | 0.940 | 0.909 | ranked |
| ni | 3 | `ni_smm1_imp_spot` | ifind | 534 | 0.912 | 0.907 | 0.909 | ranked |
| ni | 4 | `ni_cjb_spot` | ifind | 534 | 0.907 | 0.904 | 0.905 | ranked |
| ni | 5 | `ni_smm1_spot` | ifind | 534 | 0.906 | 0.903 | 0.904 | ranked |
| pb | 1 | `pb_cjb_spot` | ifind | 534 | 0.872 | 0.897 | 0.875 | ranked |
| pb | 2 | `pb_smm1_spot` | ifind | 534 | 0.870 | 0.907 | 0.875 | ranked |
| pb | 3 | `pb_994_shmet_east` | ifind | 534 | 0.865 | 0.905 | 0.870 | ranked |
| pb | 4 | `pb_sec9997_spot` | ifind | 534 | 0.818 | 0.882 | 0.825 | ranked |
| pb | 5 | `pb_sec985_spot` | ifind | 534 | 0.801 | 0.877 | 0.810 | ranked |
| sn | 1 | `sn_smm1_spot` | ifind | 502 | 0.880 | 0.877 | 0.879 | ranked |
| sn | 2 | `sn_cjb_spot` | ifind | 502 | 0.883 | 0.876 | 0.878 | ranked |
| sn | 3 | `sn_60conc_spot_guangxi` | ifind | 373 | 0.878 | 0.891 | 0.875 | ranked |
| sn | 4 | `sn_scrap_pure_bulk_shandong` | ifind | 479 | 0.775 | 0.875 | 0.776 | ranked |
| sn | 5 | `sn_scrap_slag_shandong` | ifind | 479 | 0.747 | 0.856 | 0.752 | ranked |
| ss | 1 | `ss_304_2b_hongwang_wuxi_if` | ifind | 276 | 0.719 | 0.838 | 0.736 | ranked |
| ss | 2 | `ss_304_2b_hongwang_wuxi` | mysteel | 350 | 0.693 | 0.814 | 0.713 | ranked |
| ss | 3 | `ss_304_gross_wuxi` | ifind | 350 | 0.639 | 0.691 | 0.648 | ranked |
| ss | 4 | `ss_304_scrap_wuxi` | ifind | 350 | 0.456 | 0.484 | 0.459 | ranked |
| ao | 1 | `alumina_spot_cnports` | ifind | 163 | 0.309 | 0.129 | 0.123 | ranked |
| ao | 2 | `alumina_cfr_cn` | ifind | 163 | 0.311 | 0.082 | 0.104 | ranked |
| ao | 3 | `alumina_aus_fob` | ifind | 163 | 0.314 | 0.076 | 0.103 | ranked |
| ao | 4 | `alumina_spot_qd` | ifind | 163 | 0.402 | -0.037 | 0.077 | ranked |
| ao | 5 | `alumina_fob_au` | ifind | 163 | 0.294 | 0.022 | 0.076 | ranked |
| au | 1 | `au_9999_sge_close` | ifind | 534 | 0.986 | 0.993 | 0.986 | ranked |
| au | 2 | `au_9999_sh` | ifind | 534 | 0.947 | 0.959 | 0.946 | ranked |
| ag | 1 | `ag_1_9999_sh` | ifind | 534 | 0.943 | 0.925 | 0.932 | ranked |
| ag | 2 | `ag_9999_sge_close` | ifind | 323 | 0.746 | 0.807 | 0.748 | ranked |
| si | 1 | `si_553_oxy_east` | ifind | 186 | 0.431 | 0.461 | 0.298 | ranked |
| si | 2 | `si_553_nonoxy_east` | ifind | 186 | 0.418 | 0.326 | 0.257 | ranked |
| si | 3 | `si_421_east` | ifind | 186 | 0.394 | 0.156 | 0.173 | ranked |
| si | 4 | `si_553_nonoxy_sichuan` | ifind | 186 | 0.360 | 0.147 | 0.160 | ranked |
| si | 5 | `si_421_sichuan` | ifind | 186 | 0.233 | 0.194 | 0.149 | ranked |
| lc | 1 | `lc_ind_dom_sichuan_spot` | ifind | 158 | 0.723 | 0.842 | 0.432 | ranked |
| lc | 2 | `lc_ind_dom_east_spot` | ifind | 158 | 0.723 | 0.842 | 0.432 | ranked |
| lc | 3 | `lc_bat_dom_sichuan_spot` | ifind | 158 | 0.722 | 0.849 | 0.431 | ranked |
| lc | 4 | `lc_bat_dom_east_spot` | ifind | 158 | 0.719 | 0.843 | 0.429 | ranked |
| lc | 5 | `lc_bat_dom_cn_spot` | ifind | 158 | 0.695 | 0.817 | 0.412 | ranked |
| ps |  | `` |  |  |  |  |  | no_mapped_spot_price |
| ru | 1 | `ru_scrwf_zhejiang` | ifind | 530 | 0.892 | 0.965 | 0.900 | ranked |
| ru | 2 | `ru_scrwf_sh` | ifind | 530 | 0.890 | 0.973 | 0.899 | ranked |
| ru | 3 | `ru_scrwf_jiangsu` | ifind | 530 | 0.890 | 0.963 | 0.898 | ranked |
| ru | 4 | `ru_scrwf_kunming` | ifind | 530 | 0.730 | 0.895 | 0.743 | ranked |
| UR | 1 | `UR_shandong_spot` | ifind | 342 | 0.572 | 0.674 | 0.586 | ranked |
| UR | 2 | `UR_north_spot` | ifind | 356 | 0.540 | 0.625 | 0.553 | ranked |
| UR | 3 | `UR_henan_spot` | ifind | 209 | 0.520 | 0.566 | 0.422 | ranked |
| UR | 4 | `UR_cn_fob_usd` | ifind | 209 | 0.175 | 0.212 | 0.147 | ranked |
| UR | 5 | `UR_gcc_cfr_usd` | ifind | 209 | 0.139 | 0.321 | 0.123 | ranked |
| sp | 1 | `sp_silver_sd` | ifind | 356 | 0.729 | 0.772 | 0.735 | ranked |
| sp | 2 | `sp_silver_jzh` | ifind | 356 | 0.714 | 0.768 | 0.720 | ranked |
| sp | 3 | `sp_pz_ru_sd` | ifind | 352 | 0.707 | 0.822 | 0.719 | ranked |
| sp | 4 | `sp_pz_ca_ma_sh` | ifind | 379 | 0.548 | 0.524 | 0.533 | ranked |
| sp | 5 | `sp_pz_ch_si_sd` | ifind | 379 | 0.573 | 0.490 | 0.521 | ranked |
| nr | 1 | `nr_str20_mix_qd_bonded` | ifind | 356 | 0.920 | 0.954 | 0.919 | ranked |
| nr | 2 | `nr_str20_usd_qd_bonded` | ifind | 217 | 0.676 | 0.719 | 0.566 | ranked |
| br | 1 | `br9000_qilu_sd` | mysteel | 157 | 0.858 | 0.948 | 0.513 | ranked |
| br | 2 | `br9000_sichuan_sd` | ifind | 157 | 0.848 | 0.927 | 0.504 | ranked |
| br | 3 | `br9000_daqing_sd` | ifind | 157 | 0.844 | 0.928 | 0.502 | ranked |
| br | 4 | `br9000_daqing_east` | ifind | 157 | 0.812 | 0.911 | 0.483 | ranked |
| br | 5 | `br9000_yangzi_sh_if` | ifind | 145 | 0.870 | 0.947 | 0.481 | ranked |
| l | 1 | `l_7042_north` | ifind | 530 | 0.673 | 0.774 | 0.683 | ranked |
| l | 2 | `l_7042_east` | ifind | 530 | 0.609 | 0.711 | 0.623 | ranked |
| l | 3 | `l_7042_jilin_hz` | ifind | 517 | 0.611 | 0.686 | 0.620 | ranked |
| l | 4 | `l_7042_sh` | ifind | 530 | 0.598 | 0.661 | 0.607 | ranked |
| l | 5 | `l_7042_tj` | ifind | 530 | 0.594 | 0.587 | 0.590 | ranked |
| pp | 1 | `pp_t30s_shaoxing_hz` | ifind | 502 | 0.656 | 0.652 | 0.654 | ranked |
| pp | 2 | `pp_t30s_daqing_exw` | ifind | 530 | 0.583 | 0.658 | 0.591 | ranked |
| pp | 3 | `pp_t30s_daqing_east` | ifind | 528 | 0.180 | 0.185 | 0.180 | ranked |
| TA | 1 | `TA_east_spot` | ifind | 530 | 0.898 | 0.944 | 0.901 | ranked |
| TA | 2 | `TA_cfr_cn_long` | ifind | 515 | 0.771 | 0.868 | 0.777 | ranked |
| TA | 3 | `TA_cfr_cn` | ifind | 212 | 0.821 | 0.922 | 0.673 | ranked |
| TA | 4 | `TA_cfr_sea` | ifind | 207 | 0.802 | 0.919 | 0.642 | ranked |
| PX | 1 | `PX_fob_kr_cny` | mysteel | 151 | 0.899 | 0.946 | 0.520 | ranked |
| PX | 2 | `PX_cfr_tw_cny` | mysteel | 151 | 0.899 | 0.946 | 0.519 | ranked |
| PX | 3 | `PX_korea_fob_usd` | ifind | 151 | 0.897 | 0.952 | 0.519 | ranked |
| PX | 4 | `PX_fob_kr_usd` | mysteel | 151 | 0.897 | 0.952 | 0.519 | ranked |
| PX | 5 | `PX_cfr_tw_usd` | mysteel | 151 | 0.897 | 0.952 | 0.519 | ranked |
| eg | 1 | `eg_east_spot_ms` | mysteel | 389 | 0.920 | 0.924 | 0.919 | ranked |
| eg | 2 | `eg_east_spot` | ifind | 389 | 0.920 | 0.924 | 0.919 | ranked |
| eg | 3 | `eg_cfr_cn` | ifind | 389 | 0.898 | 0.926 | 0.899 | ranked |
| eg | 4 | `eg_east_spot_mid` | ifind | 228 | 0.973 | 0.949 | 0.840 | ranked |
| eg | 5 | `eg_cfr_nea` | ifind | 228 | 0.923 | 0.920 | 0.808 | ranked |
| MA | 1 | `MA_taicang_paper_nm_ms` | mysteel | 283 | 0.939 | 0.914 | 0.923 | ranked |
| MA | 2 | `MA_taicang_paper_lm_ms` | mysteel | 283 | 0.892 | 0.819 | 0.846 | ranked |
| MA | 3 | `MA_import_taicang_spot_ms` | mysteel | 355 | 0.860 | 0.826 | 0.839 | ranked |
| MA | 4 | `MA_spot_jiangsu` | ifind | 532 | 0.837 | 0.850 | 0.837 | ranked |
| MA | 5 | `MA_zj_spot` | ifind | 530 | 0.744 | 0.811 | 0.750 | ranked |
| eb | 1 | `eb_jiangsu_n2` | ifind | 321 | 0.933 | 0.934 | 0.930 | ranked |
| eb | 2 | `eb_jiangsu_n1` | ifind | 309 | 0.933 | 0.921 | 0.925 | ranked |
| eb | 3 | `eb_cfr_cn` | ifind | 348 | 0.919 | 0.948 | 0.921 | ranked |
| eb | 4 | `eb_cfr_tw` | ifind | 346 | 0.916 | 0.945 | 0.915 | ranked |
| eb | 5 | `eb_east_spot` | ifind | 350 | 0.916 | 0.895 | 0.903 | ranked |
| sc | 1 | `dubai_spot` | ifind | 424 | 0.837 | 0.851 | 0.837 | ranked |
| sc | 2 | `oman_spot` | ifind | 424 | 0.834 | 0.853 | 0.835 | ranked |
| sc | 3 | `espo_spot` | ifind | 424 | 0.818 | 0.831 | 0.818 | ranked |
| sc | 4 | `brent_dtd_spot` | ifind | 424 | 0.754 | 0.753 | 0.754 | ranked |
| lu | 1 | `lu_0.5_sgp` | ifind | 313 | 0.835 | 0.908 | 0.839 | ranked |
| lu | 2 | `lu_05_zhoushan` | ifind | 313 | 0.788 | 0.843 | 0.792 | ranked |
| lu | 3 | `lu_05_shanghai` | ifind | 313 | 0.784 | 0.846 | 0.789 | ranked |
| lu | 4 | `lu_05_qingdao` | ifind | 313 | 0.756 | 0.771 | 0.757 | ranked |
| lu | 5 | `lu_05_huangpu_cfr` | ifind | 295 | 0.713 | 0.798 | 0.716 | ranked |
| bu | 1 | `bu_heavy_shandong` | ifind | 529 | 0.434 | 0.623 | 0.453 | ranked |
| bu | 2 | `bu_heavy_north` | ifind | 529 | 0.420 | 0.619 | 0.441 | ranked |
| bu | 3 | `bu_heavy_east` | ifind | 529 | 0.257 | 0.349 | 0.266 | ranked |
| fu | 1 | `fo_380cst_m2_sgp` | ifind | 331 | 0.857 | 0.865 | 0.854 | ranked |
| fu | 2 | `fo_380cst_m1_sgp` | ifind | 331 | 0.843 | 0.844 | 0.840 | ranked |
| fu | 3 | `fo_380cst_sgp` | ifind | 408 | 0.824 | 0.880 | 0.827 | ranked |
| fu | 4 | `fo_180cst_sgp` | ifind | 408 | 0.822 | 0.865 | 0.824 | ranked |
| fu | 5 | `fo_380cst_sgp_fob` | ifind | 365 | 0.816 | 0.848 | 0.816 | ranked |
| pg | 1 | `propane_cfr_south` | ifind | 323 | 0.619 | 0.632 | 0.619 | ranked |
| pg | 2 | `propane_cfr_tw` | ifind | 323 | 0.619 | 0.632 | 0.619 | ranked |
| pg | 3 | `propane_cfr_east` | ifind | 323 | 0.619 | 0.630 | 0.619 | ranked |
| pg | 4 | `butane_cfr_south` | ifind | 323 | 0.563 | 0.582 | 0.564 | ranked |
| pg | 5 | `butane_cfr_tw` | ifind | 323 | 0.563 | 0.582 | 0.564 | ranked |
| m |  | `` |  |  |  |  |  | no_mapped_spot_price |
| RM |  | `` |  |  |  |  |  | no_mapped_spot_price |
| y |  | `` |  |  |  |  |  | no_mapped_spot_price |
| p |  | `` |  |  |  |  |  | no_mapped_spot_price |
| OI |  | `` |  |  |  |  |  | no_mapped_spot_price |
| a |  | `` |  |  |  |  |  | no_mapped_spot_price |
| b |  | `` |  |  |  |  |  | no_mapped_spot_price |
| c |  | `` |  |  |  |  |  | no_mapped_spot_price |
| cs |  | `` |  |  |  |  |  | no_mapped_spot_price |
| CJ |  | `` |  |  |  |  |  | no_mapped_spot_price |
| CF |  | `` |  |  |  |  |  | no_mapped_spot_price |
| jd |  | `` |  |  |  |  |  | no_mapped_spot_price |
| AP |  | `` |  |  |  |  |  | no_mapped_spot_price |
| lh |  | `` |  |  |  |  |  | no_mapped_spot_price |
| SR |  | `` |  |  |  |  |  | no_mapped_spot_price |
| PK |  | `` |  |  |  |  |  | no_mapped_spot_price |

A missing spot row means the current mapped workbooks contain balance or inventory data for that product but no normalized outright spot-price alias. Short histories remain in the CSV and are not ranked until they have 52 weekly changes.
