"""
Auto-generated full index mapping from ifind Excel headers.

Conventions:
- Preserve curated aliases from pycmqlib3.utility.spot_idx_map.index_map.
- For missing codes, derive alias from Excel index_name using a snake-case normalizer.
- Resolve alias collisions by appending _<index_code>.
- Non-curated codes include inline comments listing source worksheet names.
"""

index_map_full = {
    'G002600770': 'usgg2yr',
    'G002600774': 'usgg10yr',
    'G002600783': 'usggt10yr',
    'G002600791': 'libor3m',
    'G002600885': 'dxy',
    'G002601505': '标准普尔500波动率指数vix',  # sheets: macro_d
    'G002837002': 'shcmp_idx',
    'G002856503': 'hk_hsi_idx',
    'G002856504': 'hk_cncorp_idx',
    'G002856506': 'dji_idx',
    'G002856507': 'sp500_idx',
    'G002856508': 'nasdaq_idx',
    'G002856511': 'jp_nk225_idx',
    'G003082139': 'cboe_认沽_认购比率_波动率指数vix',  # sheets: macro_d
    'G003082171': 'cboe_尾对冲指数vxth_收盘',  # sheets: macro_d
    'G003082203': 'vix',
    'G003082207': 'vvix',
    'G003082211': 'vxeem',
    'G003082215': 'vxd',
    'G003082227': 'vxn',
    'G003082236': 'cl_vol_idx',
    'G003082240': 'gc_vol_idx',
    'G003146245': 'usdeur_xe',
    'G003146246': 'usdgbp_xe',
    'G003146248': 'usdaud_xe',
    'G003146249': 'usdcad_xe',
    'G003146252': 'usdcny_xe',
    'G003146256': 'usdjpy_xe',
    'G003146263': 'usdzar_xe',
    'G003146267': 'usdbrl_xe',
    'G003146268': 'usdnok_xe',
    'G003146276': 'usdkrw_xe',
    'G004849308': 'usdclp_xe',
    'G005326174': 'citi_eco_surprise_idx_us',
    'G005432431': 'citi_eco_surprise_idx_cn',
    'G005432432': 'citi_eco_surprise_idx_eu',
    'G005432436': 'citi_eco_surprise_idx_global',
    'G005432438': 'citi_eco_surprise_idx_em',
    'G005432439': 'citi_eco_surprise_idx_asia',
    'G009067321': 'eco_policy_uncertainty_idx_us',
    'G013233151': 'usggbe5',
    'G013233152': 'usggbe10',
    'G013233153': 'inflation_exp_5y_us',
    'G019711418': 'usdcnh_xe',
    'L001618480': 'cn_govbond_yield_5y',
    'L001618805': 'cn_govbond_yield_1y',
    'L001618809': '银行间国债到期收益率_30年',  # sheets: macro_d
    'L001619213': 'cn_govbond_yield_2y',
    'L001619214': 'cn_govbond_yield_10y',
    'L001619493': 'dr007_cn',
    'L004162230': '银行间政策性金融债收益率曲线国开行_1年',  # sheets: macro_d
    'L004162231': '银行间政策性金融债收益率曲线国开行_2年',  # sheets: macro_d
    'L004162234': '银行间政策性金融债收益率曲线国开行_5年',  # sheets: macro_d
    'L004162238': '银行间政策性金融债收益率曲线国开行_10年',  # sheets: macro_d
    'L004162241': '银行间政策性金融债收益率曲线国开行_30年',  # sheets: macro_d
    'L004366599': 'usdcny_on',
    'L004366605': 'usdcny_1m',
    'L004366607': 'usd_cny外汇掉期曲线_3m',  # sheets: macro_d
    'L004366612': 'usd_cny外汇掉期曲线_1y',  # sheets: macro_d
    'L015211333': 'cnh_hibor_1m',
    'M001622302': '规模以上工业增加值_当月同比',  # sheets: macro_m
    'M001625222': 'm2_cn_yoy',
    'M001625224': 'm1_cn_yoy',
    'M002043802': 'pmi_cn_manu_all',
    'M002043804': 'pmi_cn_manu_new_order',
    'M002043805': 'pmi_cn_manu_exports',
    'M002043806': 'pmi_cn_manu_curr_order',
    'M002043808': 'pmi_cn_manu_purchase',
    'M002043809': 'pmi_cn_manu_imports',
    'M002043811': 'pmi_cn_manu_rm_inv',
    'M002808932': '出口总值美元计价_当月同比',  # sheets: macro_m
    'M002808933': '进口总值美元计价_当月同比',  # sheets: macro_m
    'M002816448': 'shibor_on',
    'M002816449': 'shibor_1w',
    'M002816451': 'shibor_1m',
    'M002816452': 'shibor_3m',
    'M002816455': 'shibor_1y',
    'M002816575': 'r001_cn',
    'M002816576': 'r007_cn',
    'M002826785': 'cpi_cn_mom',
    'M002842089': 'usdcny_mid',
    'M002842661': 'ppi_cn_mom',
    'M002845714': 'csi300_idx',
    'M002845716': '上证50指数',  # sheets: macro_d
    'M002845725': 'csi500_idx',
    'M002859231': '金融机构_人民币贷款_当月增加_住户_短期',  # sheets: macro_m
    'M002859232': '金融机构_人民币贷款_当月增加_住户_中长期',  # sheets: macro_m
    'M002859234': '金融机构_人民币贷款_当月增加_企事业单位_短期贷款',  # sheets: macro_m
    'M002859235': '金融机构_人民币贷款_当月增加_企事业单位_中长期贷款',  # sheets: macro_m
    'M002917567': '社会融资规模增量_人民币贷款_当月值',  # sheets: macro_m
    'M003146166': '金融机构_新增人民币贷款_中长期贷款_当月值',  # sheets: macro_m
    'M003146168': '全社会用电量_工业用电量_当月值',  # sheets: macro_m
    'M003559320': 'pmi_cn_steel_all',
    'M003559341': 'pmi_cn_steel_prod',
    'M003559342': 'pmi_cn_steel_rm_vol',
    'M003559343': 'pmi_cn_steel_rm_inv',
    'M003559344': 'pmi_cn_steel_new_order',
    'M003559346': 'pmi_cn_steel_inv',
    'M003559347': 'pmi_cn_steel_rm_px',
    'M003588160': 'zx_sector_idx_oil_petchem',
    'M003588161': 'zx_sector_idx_coal',
    'M003588162': 'zx_sector_idx_basemetal',
    'M003588164': 'zx_sector_idx_steel',
    'M003588167': 'zx_sector_idx_const',
    'M003588182': 'zx_sector_idx_prop',
    'M003721097': 'pmi_cn_manu_bus_exp',
    'M003802386': 'sw_sector2_idx_rubber',
    'M003802454': 'sw_sector2_idx_glass',
    'M003802458': 'sw_sector2_idx_infra',
    'M004088026': 'epmi_cn_all',
    'M004088027': 'epmi_cn_prod',
    'M004088028': 'epmi_cn_order',
    'M004103409': 'gdp_现价_当季值',  # sheets: macro_m
    'M004147023': 'usdcny_spot',
    'M004147024': 'usdcnh_spot',
    'M004302214': 'epmi_cn_exports',
    'M004302216': 'epmi_cn_inv',
    'M004302217': 'epmi_cn_purchase',
    'M004323982': '社会融资规模存量_人民币贷款_期末值',  # sheets: macro_m
    'M004323990': '社会融资规模存量_人民币贷款_期末同比',  # sheets: macro_m
    'M004369935': 'pmi_cn_cons_all',
    'M004370159': 'usdcny_spot2',
    'M004377555': 'usdcny_spot_volume',
    'M004734589': '社会融资规模存量_企业债券_期末值',  # sheets: macro_m
    'M004734590': '社会融资规模存量_企业债券_期末同比',  # sheets: macro_m
    'M004891020': '社会融资规模存量_期末值',  # sheets: macro_m
    'M004891021': '社会融资规模存量_期末同比',  # sheets: macro_m
    'M005933607': 'pmi_cn_cons_new_order',
    'M005933608': 'pmi_cn_cons_rm_px',
    'M005933609': 'pmi_cn_cons_px',
    'M005933610': 'pmi_cn_cons_hr',
    'M005933611': 'pmi_cn_cons_bus_exp',
    'M009042847': 'sw_sector_idx_steel',
    'M009042848': 'sw_sector_idx_basemetal',
    'M009042858': 'sw_sector_idx_prop',
    'M009042864': 'sw_sector_idx_const',
    'M009042872': 'sw_sector_idx_coal',
    'M009042873': 'sw_sector_idx_petchem',
    'M011202650': 'usdcnh_close',
    'M012370785': '社会融资规模存量_人民币贷款_初值',  # sheets: macro_m
    'M012963695': 'csi1000_idx',
    'M013284221': 'gdp_初步核算数_当季值',  # sheets: macro_m
    'S000001484': 'y_inv_dce_warrant',
    'S000001485': 'a_inv_dce_warrant',
    'S000001487': 'c_inv_dce_warrant',
    'S000001488': 'SR_inv_czce_warrant',
    'S000001489': '仓单数量_强筋小麦',  # sheets: warrant_d
    'S000001490': 'CF_inv_czce_warrant',
    'S000001491': 'OI_inv_czce_warrant',
    'S000001493': 'OI_inv_czce_unwarrant',
    'S000001495': '有效仓单预报_强筋小麦_小计',  # sheets: warrant_d
    'S000001496': 'SR_inv_czce_unwarrant',
    'S000006462': 'dubai_spot',
    'S000009278': 'crude_refined_inv_eia_ex_spr',
    'S000009279': 'crude_inv_eia_ex_spr_all',
    'S000009280': 'oil_inv_eia_spr',
    'S000020868': 'rebar_sh',
    'S000020892': 'hrc_sh',
    'S000020903': 'crc_sh',
    'S000020933': 'gi_0.5_sh',
    'S000025471': 'cu_cjb_spot',
    'S000025473': 'al_cjb_spot',
    'S000025475': 'pb_cjb_spot',
    'S000025476': 'zn_cjb_spot',
    'S000025477': '现货均价_1_锌_长江有色',  # sheets: base_d
    'S000025478': 'sn_cjb_spot',
    'S000025479': 'ni_cjb_spot',
    'S000025544': 'au_9999_sge_close',
    'S000025546': 'au_td_sge',
    'S000025548': '上海黄金交易所_收盘价_白银_agt_plus_d',  # sheets: base_d
    'S000025556': '伦敦现货黄金_美元',  # sheets: base_d
    'S000025559': '伦敦现货白银_美元',  # sheets: base_d
    'S000025728': 'cu_inv_lme_total',
    'S000025729': 'al_inv_lme_total',
    'S000025730': 'zn_inv_lme_total',
    'S000025731': 'pb_inv_lme_total',
    'S000025732': 'sn_inv_lme_total',
    'S000025733': 'ni_inv_lme_total',
    'S000042897': 'ccfi_综合指数',  # sheets: const_d
    'S000042906': 'ccfi_欧洲航线',  # sheets: const_d
    'S000047990': '商品房销售面积_累计值',  # sheets: macro_m
    'S000047991': '商品房销售面积_住宅_累计值',  # sheets: macro_m
    'S002808963': 'plt65',
    'S002808964': 'plt58',
    'S002808967': 'container_exp_scfi',
    'S002825712': 'pvc_cac2_north',
    'S002825715': 'pvc_cac2_east',
    'S002825718': 'pvc_cac2_south',
    'S002825721': 'pvc_cac2_central',
    'S002825727': 'pvc_ethylene_east',
    'S002825730': 'pvc_ethylene_south',
    'S002825733': 'sa_light_north',
    'S002825734': 'sa_heavy_north',
    'S002825739': 'sa_light_east',
    'S002825740': 'sa_heavy_east',
    'S002825872': 'naph_cfr_jp',
    'S002827223': 'billet_js',
    'S002827225': 'billet_ts',
    'S002827258': 'scrap_zjg',
    'S002827270': 'scrap_ts',
    'S002835927': '电石_华中_主流均价',  # sheets: ferrous_d
    'S002835955': 'px_korea_fob_usd',
    'S002835961': 'px_taiwan_cfr_usd',
    'S002835975': 'pta_east_spot',
    'S002836796': 'l_7042_north',
    'S002836797': 'l_7042_east',
    'S002836798': 'l_7042_south',
    'S002836848': '电石_华北_主流均价',  # sheets: ferrous_d
    'S002836856': 'cu_inv_lme_cancelled',
    'S002836862': 'al_inv_lme_cancelled',
    'S002836868': 'pb_inv_lme_cancelled',
    'S002836874': 'zn_inv_lme_cancelled',
    'S002836880': 'sn_inv_lme_cancelled',
    'S002836886': 'ni_inv_lme_cancelled',
    'S002837009': 'coal_5500_qhd',
    'S002837160': 'io_inv_47ports',
    'S002837394': 'io_removal_47ports',
    'S002841991': 'lh_inv_mth',
    'S002841992': 'lh_breeding_sow_inv_mth',
    'S002854771': 'ur_north_spot',
    'S002855118': 'cu_inv_cme_total',
    'S002855119': 'au_cme_warrant_all',
    'S002855120': 'ag_cme_warrant_all',
    'S002858879': '京唐港_库提价含税_澳大利亚_主焦煤',  # sheets: ferrous_d
    'S002859801': 'hrc_tj',
    'S002859811': '热轧_4_75热轧板卷_全国均价',  # sheets: ferrous_d
    'S002860878': '市场价_螺纹钢_hrb400e_20mm_全国均价',  # sheets: ferrous_d
    'S002863167': 'pta_cfr_cn',
    'S002863173': 'px_exw_east_spot',
    'S002863469': 'oman_spot',
    'S002863470': 'brent_dtd_spot',
    'S002865578': 'zn_smm0_spot',
    'S002865583': 'pb_smm1_spot',
    'S002865585': '现货含税均价_1_锌锭zn99_99_上海有色',  # sheets: base_d
    'S002865591': 'pb_sec9997_spot',
    'S002865591.1': '平均价_再生精铅_pb99_97',  # sheets: base_d
    'S002865592': 'al_smm0_spot',
    'S002865595': 'pb_sec985_spot',
    'S002865595.1': '平均价_再生铅_pb98_5',  # sheets: base_d
    'S002865596': '现货均价含税_1_银99_99pct',  # sheets: base_d
    'S002865601': '现货均价含税_金99_95pct',  # sheets: base_d
    'S002865602': '现货均价含税_金99_99pct',  # sheets: base_d
    'S002865685': 'lc_bat_dom_cn_spot',
    'S002865711': '现货均价_1_电解铜_广东南储华南',  # sheets: base_d
    'S002877257': 'ckc_a10v24s08_lvliang',
    'S002877258': 'ckc_a9v18s10_lvliang',
    'S002877299': 'ckc_a10v24s10_ts',
    'S002882871': 'coal_5500_sx_qhd',
    'S002882887': '铁矿石国际运价_理查德兹_萨尔达尼亚_中国海岬型',  # sheets: ferrous_d
    'S002882893': '铁矿石国际运价_纽卡斯尔_中国海岬型',  # sheets: ferrous_d
    'S002882896': '铁矿石国际运价_西澳大利亚丹皮尔_青岛海岬型',  # sheets: ferrous_d
    'S002882897': '铁矿石国际运价_图巴朗_青岛海岬型',  # sheets: ferrous_d
    'S002883659': '环渤海动力煤_综合平均价格5500k',  # sheets: const_d
    'S002883666': '环渤海动力煤价格_秦皇岛5500k',  # sheets: const_d
    'S002893910': 'eg_north_exw',
    'S002911091': 'pbf_cfd',
    'S002911136': 'pbf_qd',
    'S002916792': '商品房销售面积_累计同比',  # sheets: macro_m
    'S002917430': 'plate_8mm',
    'S002917486': 'gi_0.5',
    'S002917646': 'hsec_400x200',
    'S002917688': 'strip_3.0x685',
    'S002917771': 'pipe_1.5x3.25',
    'S002933708': '库存_铜_上海_合计',  # sheets: base_w
    'S002933710': '库存_铜_完税总计',  # sheets: base_w
    'S002933720': '库存_铝_总计',  # sheets: base_w
    'S002933746': '库存_锌_总计',  # sheets: base_w
    'S002933764': '库存_铅_总计',  # sheets: base_w
    'S002950206': '车板价_含税_进口铁矿石61_5pct_pb粉_日照港',  # sheets: ferrous_d
    'S002954691': 'scrap_sh',
    'S002955332': 'bz_taiwan_cfr_usd',
    'S002955437': 'eb_cfr_cn',
    'S002955504': 'ma_cfr_cn',
    'S002956186': 'eg_cfr_cn',
    'S002956195': 'eg_cfr_sea',
    'S002956389': '含税价_兰炭中料_固定碳_84pct_神木',  # sheets: ferrous_d
    'S002958615': 'crude_inv_eia_ex_spr_cushing',
    'S002959172': 'ss_304_scrap_wuxi',
    'S002959491': 'sm_65s17_neimeng',
    'S002959495': 'sm_65s17_guangxi',
    'S002959498': 'sm_65s17_tj',
    'S002959499': 'sm_65s17_gansu',
    'S002959574': '天津航运指数tsi',  # sheets: const_d
    'S002966141': '出厂价_电石_陕西神木县昌明',  # sheets: ferrous_d
    'S002981535': 'cu_smm1_spot',
    'S002981536': 'cu_smm1_prem_spot',
    'S002981537': '现货含税均价_湿法铜cu_ag_99_95pct_上海有色',  # sheets: base_d
    'S002981538': '现货含税均价_贵溪铜cu_ag_99_95pct_上海有色',  # sheets: base_d
    'S002981539': 'sn_smm1_spot',
    'S002981540': 'ni_smm1_spot',
    'S002981541': 'ni_smm1_jc_spot',
    'S002981542': 'ni_smm1_imp_spot',
    'S002981543': '现货含税均价_平水铜cu_ag_99_95pct_上海有色',  # sheets: base_d
    'S002983448': 'pci_jincheng',
    'S002983449': 'pci_yangquan',
    'S003008076': 'TA_inv_czce_warrant',
    'S003008289': '开工率_pta_全国',  # sheets: petchem_w
    'S003008291': '开工率_织机_江浙地区',  # sheets: petchem_w
    'S003010762': '现货价_燃料油高硫180cst_阿拉伯海湾_中间价',  # sheets: petchem_d
    'S003010765': '现货价_燃料油高硫380cst_阿拉伯海湾_中间价',  # sheets: petchem_d
    'S003011277': '市场价_mtbe_华东',  # sheets: petchem_d
    'S003011278': '市场价_mtbe_华南',  # sheets: petchem_d
    'S003011283': 'fo_180cst_east',
    'S003011289': 'fo_180cst_sh',
    'S003011302': 'fo_180cst_xiamen',
    'S003011318': 'propane_cfr_asia_n',
    'S003011327': 'propane_cfr_china_s',
    'S003011336': 'propane_cfr_tw',
    'S003011351': 'butane_cfr_asia_n',
    'S003011360': 'butane_cfr_china_s',
    'S003011369': 'butane_cfr_tw',
    'S003014166': 'si_553_nonoxy_kunming',
    'S003018859': 'cu_lme_3m_15m_spd',
    'S003018860': 'cu_lme_3m_27m_spd',
    'S003018862': 'al_lme_3m_15m_spd',
    'S003018863': 'al_lme_3m_27m_spd',
    'S003018865': 'ni_lme_3m_15m_spd',
    'S003018866': 'ni_lme_3m_27m_spd',
    'S003018868': 'sn_lme_3m_15m_spd',
    'S003018871': 'zn_lme_3m_15m_spd',
    'S003018872': 'zn_lme_3m_27m_spd',
    'S003018874': 'pb_lme_3m_15m_spd',
    'S003018875': 'pb_lme_3m_27m_spd',
    'S003019324': 'plt62',
    'S003031623': 'fo_380cst_sgp',
    'S003031624': 'fo_180cst_sgp',
    'S003048722': 'cu_smm_phybasis',
    'S003048723': 'cu_flat_phybasis',
    'S003048724': 'cu_prem_phybasis',
    'S003048725': '现货升贴水_湿法铜cu_ag_99_95pct_上海有色',  # sheets: base_d
    'S003048726': '现货升贴水_贵溪铜cu_ag_99_95pct_上海有色',  # sheets: base_d
    'S003048727': 'al_smm0_phybasis',
    'S003052593': 'bean_inv_ports_d',
    'S003057206': 'ag_td_sge',
    'S003085584': 'PTA_invdays_mill',
    'S003085588': 'POY_invdays_mill',
    'S003085589': 'FDY_invdays_mill',
    'S003085590': 'DTY_invdays_mill',
    'S003131240': '市场价不含税_氧化锰矿_广西_mn30fe10p双零块',  # sheets: ferrous_d
    'S003131312': '市场价不含税_富锰渣_mn28pct_广西',  # sheets: ferrous_d
    'S003131313': '市场价不含税_富锰渣_mn30pct_广西',  # sheets: ferrous_d
    'S003131357': '锰矿_库存_合计',  # sheets: base_w
    'S003138068': 'coke_inv_ports_tj',
    'S003138069': 'coke_inv_ports_lyg',
    'S003138070': 'coke_inv_ports_rz',
    'S003148264': 'y_inv_ports',
    'S003154875': 'FG_inv_czce_warrant',
    'S003155008': 'p_inv_dce_warrant',
    'S003157699': 'l_7042_tj',
    'S003157759': 'l_7042_sh',
    'S003164358': 'cu_inv_shfe_d',
    'S003164359': 'al_inv_shfe_d',
    'S003164360': 'zn_inv_shfe_d',
    'S003164361': 'pb_inv_shfe_d',
    'S003164362': 'au_inv_shfe_warrant',
    'S003164363': 'ag_inv_shfe_warrant',
    'S003164365': '仓单数量_铜_完税总计',  # sheets: warrant_d
    'S003164385': '仓单数量_铝_完税总计',  # sheets: warrant_d
    'S003254707': 'p_inv_ports',
    'S003276468': '天津港_平仓价格含税_准一级冶金焦a_12_5_s_0_7_csr_60_mt8_山西',  # sheets: ferrous_d
    'S003277847': 'b_inv_dce_warrant',
    'S003277851': 'm_inv_dce_warrant',
    'S003277863': '仓单数量_早籼稻',  # sheets: warrant_d
    'S003278148': 'RM_inv_czce_warrant',
    'S003278182': 'CF_inv_czce_unwarrant',
    'S003278183': '有效仓单预报_早籼稻_小计',  # sheets: warrant_d
    'S003278185': 'RM_inv_czce_unwarrant',
    'S003281517': '全国_房地产施工面积_合计_累计值',  # sheets: macro_m
    'S003281522': '全国_房地产新开工面积_合计_累计值',  # sheets: macro_m
    'S003281527': '全国_房地产竣工面积_合计_累计值',  # sheets: macro_m
    'S003560513': '开工率_涤纶长丝_江浙地区',  # sheets: petchem_w
    'S003560514': '开工率_涤纶短纤_全国',  # sheets: petchem_w
    'S003563054': '现货均价_1_电解铜_广东南储华东',  # sheets: base_d
    'S003583313': 'au_etf_spdr_holding',
    'S003583337': 'm_inv_mill_sm',
    'S003583339': 'm_inv_mill_nonexec',
    'S003587817': 'margin_outstanding_total_cn',
    'S003715212': 'ag_etf_slv_holding',
    'S003787910': 'l_inv_dce_warrant',
    'S003787913': 'pp_inv_dce_warrant',
    'S003787915': 'v_inv_dce_warrant',
    'S003797045': 'ni_1.8conc_spot_php_lianyungang',
    'S003809462': '镍矿_港口库存_总计',  # sheets: base_w
    'S003817887': 'io_invdays_imp_mill(64)',
    'S003839317': 'ZC_inv_6gen',
    'S003839331': 'ZC_invdays_6gen',
    'S003852895': 'au_cme_warrant_reg',
    'S003852896': 'au_cme_warrant_unreg',
    'S003852912': 'ag_cme_warrant_reg',
    'S003852913': 'ag_cme_warrant_unreg',
    'S003852932': '库存_comex_铜_合计_注册',  # sheets: base_d
    'S003852933': '库存_comex_铜_合计_未注册',  # sheets: base_d
    'S003853061': '上海船舶价格指数spi',  # sheets: const_d
    'S003986222': 'CF_inv_social_mth',
    'S003994516': 'ma_spot_jiangsu',
    'S003994543': 'ma_spot_neimeng',
    'S003994600': 'eg_east_spot',
    'S003994603': 'eg_south_spot',
    'S004018814': 'io_loading_14ports_ausbzl',
    'S004018816': '铁矿石_澳洲发货量',  # sheets: ferrous_w
    'S004018822': '铁矿石_澳洲发货量_至中国',  # sheets: ferrous_w
    'S004018831': '铁矿石_巴西发货量',  # sheets: ferrous_w
    'S004018842': '铁矿石_中国到港量_北方六港',  # sheets: ferrous_w
    'S004029055': 'rebar_enduse_sales_sh',
    'S004031017': 'al_sh_phybasis',
    'S004038574': 'pmi_lgsc_steel_all',
    'S004038575': 'pmi_lgsc_steel_purchase_exp',
    'S004038576': 'pmi_lgsc_steel_tot_order',
    'S004039553': 'billet_inv_social_ts',
    'S004039587': 'ckc_inv_6ports',
    'S004045178': 'ag_inv_sge',
    'S004045185': 'ag_9999_sge_close',
    'S004077380': 'pg_cn_spot',
    'S004077382': 'pl_shandong_spot',
    'S004077398': 'ma_zj_spot',
    'S004077476': 'pp_linyi_spot',
    'S004077496': '现货基准价_烧碱32pct离子膜碱_山东',  # sheets: petchem_d
    'S004077505': 'cu_spot_sh',
    'S004077608': '到厂含税价_中间价_预焙阳极国标_山东',  # sheets: base_d
    'S004077728': 'alumina_spot_qd',
    'S004077746': '中间价_a0_1氧化铝al2o3_98_6pct_连云港',  # sheets: base_d
    'S004077839': '含税现货矿山价_铝土矿_三门峡al_55_60pct_si_12_13pct',  # sheets: base_d
    'S004077840': '含税现货矿山价_铝土矿_阳泉al_si_4_5',  # sheets: base_d
    'S004077841': '含税现货矿山价_铝土矿_贵阳al_60_65pct_si_9_11pct',  # sheets: base_d
    'S004077856': '到厂价_铅精矿_济源50pct',  # sheets: base_d
    'S004077857': '到厂价_铅精矿_郴州50pct',  # sheets: base_d
    'S004077858': '到厂价_铅精矿_个旧50pct',  # sheets: base_d
    'S004077859': '车板价_铅精矿_凉山50pct',  # sheets: base_d
    'S004077860': '车板价_铅精矿_昆明50pct',  # sheets: base_d
    'S004077861': '车板价_铅精矿_宝鸡50pct',  # sheets: base_d
    'S004085268': 'ckc_stock_ganqimaodu',
    'S004110574': 'idx_30大中城市_商品房成交面积',  # sheets: macro_d
    'S004127496': 'UR_inv_social',
    'S004155300': 'SH_32_spot_sdjl_shandong',
    'S004155302': 'SH_50_spot_sdjl_shandong',
    'S004156562': 'pl_east_spot',
    'S004156580': 'bz_east_spot',
    'S004156649': 'eb_north_spot',
    'S004156652': 'eb_east_spot',
    'S004157062': '主流价_煤沥青中温_河北地区',  # sheets: petchem_d
    'S004157068': '主流价_煤沥青改质_山东地区',  # sheets: petchem_d
    'S004157509': '出厂价_pvc_大连商品交易所_sg_5',  # sheets: ferrous_d
    'S004161475': 'pp_wenzhou_spot',
    'S004161916': '主流价_丁苯橡胶1502吉林石化_江苏',  # sheets: petchem_d
    'S004161931': '主流价_丁苯橡胶1502齐鲁石化_山东',  # sheets: petchem_d
    'S004161952': '主流价_丁苯橡胶1502齐鲁石化_浙江',  # sheets: petchem_d
    'S004161979': '主流价_合成胶乳羧基丁苯胶乳_山东市场',  # sheets: petchem_d
    'S004161982': '主流价_合成胶乳羧基丁腈胶乳_山东市场',  # sheets: petchem_d
    'S004161988': '主流价_合成胶乳羧基丁苯胶乳_华东市场',  # sheets: petchem_d
    'S004161991': '主流价_合成胶乳羧基丁腈胶乳_华东市场',  # sheets: petchem_d
    'S004163694': '市场价_沥青sbs改性沥青_华北',  # sheets: petchem_d
    'S004163695': '市场价_沥青sbs改性沥青_华东',  # sheets: petchem_d
    'S004163697': '市场价_沥青sbs改性沥青_山东',  # sheets: petchem_d
    'S004163701': 'bu_heavy_north',
    'S004163702': 'bu_heavy_east',
    'S004163704': 'bu_heavy_shandong',
    'S004163707': '市场价_沥青建筑沥青_华北',  # sheets: petchem_d
    'S004163708': '市场价_沥青建筑沥青_山东',  # sheets: petchem_d
    'S004210693': 'macf_cfd',
    'S004226161': 'io_inv_imp_mill(64)',
    'S004226162': '进口矿_烧结粉矿_总日耗',  # sheets: ferrous_w
    'S004226163': 'io_inv_dom_mill(64)',
    'S004226164': '国产矿_烧结粉矿_总日耗',  # sheets: ferrous_w
    'S004227265': '平均价_氧化铝_全国',  # sheets: base_d
    'S004227268': '平均价_氧化铝_河南',  # sheets: base_d
    'S004227274': '平均价_氧化铝_山西',  # sheets: base_d
    'S004227277': '平均价_氧化铝_广西',  # sheets: base_d
    'S004227280': '平均价_氧化铝_贵州',  # sheets: base_d
    'S004242343': 'espo_spot',
    'S004242346': 'ru_100ppi_spot',
    'S004242347': '现货价_石油沥青',  # sheets: petchem_d
    'S004242348': 'ma_100ppi_spot',
    'S004242351': 'pp_100ppi_spot',
    'S004242352': 'pta_100ppi_spot',
    'S004242725': 'fg_100ppi',
    'S004242733': '平均价_1_铜_长江有色',  # sheets: base_d
    'S004242750': '平均价_a00铝_上海',  # sheets: base_d
    'S004242751': '平均价_a00铝_长江有色',  # sheets: base_d
    'S004242813': '平均价_1_镍_长江有色',  # sheets: base_d
    'S004243249': 'al_scrap_shredded_sh_low',
    'S004243250': 'al_scrap_shredded_sh_high',
    'S004243369': 'zn_scrap_sh_low',
    'S004243370': 'zn_scrap_sh_high',
    'S004248145': 'cicfi_综合指数',  # sheets: const_d
    'S004248146': 'cicfi_欧洲航线',  # sheets: const_d
    'S004248148': 'cicfi_美西航线',  # sheets: const_d
    'S004248164': 'fu_inv_sing',
    'S004302740': 'MA_inv_czce_warrant',
    'S004302762': 'bu_inv_shfe_warrant',
    'S004302780': 'bu_inv_shfe_mill',
    'S004302793': 'bu_inv_shfe_social_addon',
    'S004302811': 'bu_inv_shfe_social',
    'S004302829': 'bu_inv_shfe_mill_w',
    'S004302841': 'bu_invcap_shfe_social',
    'S004302859': 'bu_invcap_shfe_mill',
    'S004303031': 'cu_lme_0m_3m_spd',
    'S004303032': 'sn_lme_0m_3m_spd',
    'S004303033': 'pb_lme_0m_3m_spd',
    'S004303034': 'zn_lme_0m_3m_spd',
    'S004303035': 'al_lme_0m_3m_spd',
    'S004303036': 'ni_lme_0m_3m_spd',
    'S004317118': 'macf_qd',
    'S004317120': 'iocj_qd',
    'S004317121': 'brbf_qd',
    'S004317128': 'fbf_qd',
    'S004317129': 'ssf_qd',
    'S004320033': 'au_etf_ishares_holding',
    'S004321822': 'ru_scrwf_jiangsu',
    'S004321831': 'ru_scrwf_zhejiang',
    'S004321834': 'ru_scrwf_kunming',
    'S004322735': 'ni_inv_shfe_d',
    'S004322736': 'sn_inv_shfe_d',
    'S004339370': 'sp_pz_ca_moon_sh',
    'S004349724': 'sp_pz_ch_si_sd',
    'S004349728': 'sp_pz_ca_lion_sd',
    'S004349732': 'sp_pk_br_yw_sd',
    'S004369291': 'coke_sub_a_tj',
    'S004369292': '天津港_平仓价格_准一级冶金焦a_13pct_s_0_70pct_mt_7pct_csr_60_山西',  # sheets: ferrous_d
    'S004369601': '水泥价格指数_全国',  # sheets: const_d
    'S004369602': '水泥价格指数_华东',  # sheets: const_d
    'S004370169': 'RS_inv_mill',
    'S004378418': 'rebar_inv_mill',
    'S004378419': 'wirerod_inv_mill',
    'S004378420': 'hrc_inv_mill',
    'S004378421': 'crc_inv_mill',
    'S004378422': 'plate_inv_mill',
    'S004378612': 'coal_5500_jingtang',
    'S004381180': 'coal_6000_api2_ara',
    'S004381181': 'coal_6000_api4_sa',
    'S004381182': 'coal_6000_newc_fob',
    'S004382952': '库存_原油变化_实际值',  # sheets: petchem_w
    'S004383001': 'eg_inv_port_east',
    'S004392973': 'polyolefin_inv',  # sheets: petchem_w
    'S004407062': 'pr_east_spot',
    'S004407077': 'pf_fujian_spot',
    'S004407080': 'pf_east_spot',
    'S004410360': 'ru_inv_shfe_warrant',
    'S004410392': 'ru_inv_shfe_all',
    'S004425257': 'zn_inv_social_all',
    'S004425257.1': '库存_锌锭_合计',  # sheets: base_w
    'S004425298': 'coke_sub_a_rz',
    'S004425304': '港口均价_平仓价格含税_一级冶金焦a12_5_s0_65_csr65_mt7',  # sheets: const_d
    'S004425326': 'alumina_inv_ports',
    'S004425326.1': '库存_氧化铝_总计',  # sheets: base_w
    'S004494138': 'bu_inv_social',
    'S004494149': 'bu_inv_mill_shandong',
    'S004494153': 'bu_inv_mill',
    'S004543083': 'prop_2ndhand_px_idx',
    'S004630824': 'cu_mine_tc',
    'S004630825': '铜精矿_现货_精炼费rc',  # sheets: base_d
    'S004647718': 'compound_fertilizer_inv_social',
    'S004724779': 'sp_100ppi_spot',
    'S004785205': 'ss_304_gross_wuxi',
    'S004785206': '现货价_304_2b卷_切边_无锡',  # sheets: base_d
    'S004785207': '现货价_304_no_1卷_无锡',  # sheets: base_d
    'S004785215': 'ss_304_wuxi_phybasis',
    'S004788710': 'UR_inv_mill',
    'S004789784': 'sf_72_ningxia',
    'S004789786': 'sf_72_gansu',
    'S004789790': 'sf_72_neimeng',
    'S004802760': 'rebar_prod_all',
    'S004802761': 'wirerod_prod_all',
    'S004807566': 'sp_pz_ru_sd',
    'S004869807': 'eb_inv_port_east',
    'S004869808': 'eb_inv_port_east_trader',
    'S004869809': 'eb_inv_mill',
    'S004869812': '开工率_聚氯乙烯pvc',  # sheets: petchem_w
    'S004869813': '开工率_聚氯乙烯pvc_电石法',  # sheets: petchem_w
    'S004869814': '开工率_聚氯乙烯pvc_乙烯法',  # sheets: petchem_w
    'S005028348': 'pg_100ppi_spot',
    'S005068027': 'sf_75_shmet',
    'S005068030': 'sf_72_shmet',
    'S005068033': 'au_9999_sh',
    'S005068036': 'ag_1_9999_sh',
    'S005068066': 'ag_td_phbasis',
    'S005068100': '平均价_a00铝锭99_7pct_华东',  # sheets: base_d2
    'S005068103': 'al_a00_phybasis_shmet',
    'S005068106': 'al_prem_bonded_cif',
    'S005068109': 'al_prem_bonded_warrant',
    'S005068112': 'sm_65s17_shmet',
    'S005068154': '平均价_硫酸镍21_8pct_全国',  # sheets: base_d2
    'S005068163': '平均价_1_电解镍99_9pct_上海',  # sheets: base_d2
    'S005068166': '平均价_俄镍_上海',  # sheets: base_d2
    'S005068169': 'ni_smm1_ru_phybasis',
    'S005068172': '平均价_金川镍_上海',  # sheets: base_d2
    'S005068175': 'ni_smm1_jc_phybasis',
    'S005068178': '平均价_金川镍出厂价99_96pct_上海',  # sheets: base_d2
    'S005068181': 'ni_prem_bonded_cif',
    'S005068184': 'ni_prem_bonded_warrant',
    'S005068187': 'ni_1.5conc_spot_rz',
    'S005068190': '平均价_镍铁7_10pct_全国',  # sheets: base_d2
    'S005068193': 'ni_smm1_phybasis',
    'S005068196': '平均价_镍铁8_10pct_内蒙古',  # sheets: base_d2
    'S005068199': '平均价_镍豆99_93pct_99_94pct_上海',  # sheets: base_d2
    'S005068208': 'pb_994_shmet_east',
    'S005068211': 'pb_smm1_sh_phybasis',
    'S005068217': 'pb_prem_bonded_cif',
    'S005068220': 'pb_prem_bonded_warrant',
    'S005068301': '平均价_无氧铜杆8mm_全国',  # sheets: base_d2
    'S005068304': '平均价_无氧铜丝3mm_全国',  # sheets: base_d2
    'S005068328': 'cu_prem_bonded_warrant_sx',
    'S005068331': 'cu_prem_bonded_warrant_er',
    'S005068334': 'cu_prem_bonded_cif_sx',
    'S005068337': 'cu_prem_bonded_cif_er',
    'S005068340': '平均价_升水铜升贴水_上海',  # sheets: base_d2
    'S005068343': '平均价_平水铜升贴水_上海',  # sheets: base_d2
    'S005068346': '平均价_差铜升贴水_上海',  # sheets: base_d2
    'S005068349': '平均价_1_电解铜_99_95pct_上海',  # sheets: base_d2
    'S005068352': '平均价_1_电解铜升贴水_99_95pct_上海',  # sheets: base_d2
    'S005068394': '平均价_1_锡99_9pct_华东',  # sheets: base_d2
    'S005068409': 'sn_smm1_sh_phybasis',
    'S005068412': '平均价_0_锌锭99_995pct_上海',  # sheets: base_d2
    'S005068418': '平均价_氧化锌99_7pct_全国',  # sheets: base_d2
    'S005068421': 'zn_smm0_sh_phybasis',
    'S005068424': 'zn_smm1_sh_phybasis',
    'S005068427': 'zn_prem_smm_import',
    'S005068430': '平均价_进口锌_上海',  # sheets: base_d2
    'S005068436': 'zn_prem_bonded_cif',
    'S005068439': 'zn_prem_bonded_warrant',
    'S005100607': 'lc_li2Omine_6pct_cif',
    'S005102262': 'sn_60conc_spot_guangxi',
    'S005107854': 'hrc_prod_all',
    'S005118141': 'cu_inv_bonded_sh',
    'S005118151': 'cu_inv_social_dom',
    'S005118151.1': '库存_铜_合计',  # sheets: base_w
    'S005126411': '现货价_燃料油船用180cst_fob_新加坡_中间价',  # sheets: petchem_d
    'S005126414': '现货价_燃料油船用380cst_fob_新加坡_中间价',  # sheets: petchem_d
    'S005126417': 'lu_0.5_sgp',
    'S005349977': 'ur_100ppi_spot',
    'S005349978': 'eb_100ppi_spot',
    'S005349979': 'sa_heavy_sys',
    'S005363047': '上海有色_库存_电解铝_总计',  # sheets: base_w
    'S005402481': 'px_100ppi_spot',
    'S005402526': 'pf_100ppi_spot',
    'S005429350': '车板价_含税_进口铁矿石金布巴粉60_5pct_bhp_青岛港',  # sheets: const_d
    'S005429351': 'jmb_qd',
    'S005439547': 'v_inv_social',
    'S005439550': 'v_inv_social_east',
    'S005439569': 'PTA_inv_social_mth',
    'S005439581': '开工率_eps',  # sheets: petchem_w
    'S005439582': '开工率_ps',  # sheets: petchem_w
    'S005439583': '开工率_abs',  # sheets: petchem_w
    'S005439586': 'sa_inv_mill_all',
    'S005439588': '企业库存_纯碱轻质_全国',  # sheets: const_d
    'S005439590': '企业库存_纯碱重质_全国',  # sheets: const_d
    'S005439592': 'sa_workrate_cn',
    'S005439594': 'sa_workrate_anjian',
    'S005439596': 'sa_workrate_lianchan',
    'S005450012': 'nr_inv_shfe_warrant',
    'S005450018': '仓单数量_燃料油_总计',  # sheets: warrant_d
    'S005450245': 'eg_inv_dce_warrant',
    'S005450249': 'eb_inv_dce_warrant',
    'S005451340': 'UR_inv_czce_warrant',
    'S005451360': 'SA_inv_czce_warrant',
    'S005451410': 'UR_inv_czce_unwarrant',
    'S005451430': 'SA_inv_czce_unwarrant',
    'S005451467': 'MA_inv_czce_unwarrant',
    'S005451492': 'TA_inv_czce_unwarrant',
    'S005451604': 'nr_inv_shfe_all',
    'S005451610': 'fu_inv_shfe',
    'S005470469': 'fg_5mm_north',
    'S005470470': 'fg_5mm_shahe',
    'S005470484': '国内市场价_玻璃5_0mm_华东',  # sheets: const_d
    'S005476287': 'sp_inv_shfe_warrant',
    'S005476308': 'rb_inv_shfe_warrant',
    'S005476309': 'hc_inv_shfe_warrant',
    'S005476310': 'i_inv_dce_warrant',
    'S005476311': 'SF_inv_czce_warrant',
    'S005476312': 'SM_inv_czce_warrant',
    'S005476313': 'SF_inv_czce_unwarrant',
    'S005476314': 'SM_inv_czce_unwarrant',
    'S005476601': 'sc_inv_ine_warrant',
    'S005476602': 'fu_inv_shfe_warrant',
    'S005476603': 'j_inv_dce_warrant',
    'S005476604': 'jm_inv_dce_warrant',
    'S005476605': '仓单数量_动力煤',  # sheets: warrant_d
    'S005532614': 'cs_inv_dce_warrant',
    'S005532615': 'jd_inv_dce_warrant',
    'S005532616': '仓单数量_晚籼稻',  # sheets: warrant_d
    'S005532617': 'CY_inv_czce_warrant',
    'S005532618': '仓单数量_粳稻',  # sheets: warrant_d
    'S005532619': 'CJ_inv_czce_warrant',
    'S005532620': 'AP_inv_czce_warrant',
    'S005532621': 'AP_inv_czce_unwarrant',
    'S005532622': '有效仓单预报_晚籼稻_小计',  # sheets: warrant_d
    'S005532623': '有效仓单预报_粳稻_小计',  # sheets: warrant_d
    'S005532625': 'CJ_inv_czce_unwarrant',
    'S005580356': '库存_镍_合计',  # sheets: base_w
    'S005580375': '库存_锡_合计',  # sheets: base_w
    'S005580617': '钢厂库存_涂镀钢厂_镀锌板卷_全国',  # sheets: ferrous_w
    'S005580618': '钢厂库存_涂镀钢厂_彩涂板卷_全国',  # sheets: ferrous_w
    'S005580633': 'wirerod_inv_social',
    'S005580634': 'rebar_inv_social',
    'S005580635': 'hrc_inv_social',
    'S005580636': 'plate_inv_social',
    'S005580637': '社会库存_镀锌板卷',  # sheets: ferrous_w
    'S005580638': '社会库存_彩涂板卷',  # sheets: ferrous_w
    'S005580639': 'crc_inv_social',
    'S005580640': 'hrc_inv_all',
    'S005580641': 'rebar_inv_all',
    'S005580642': 'wirerod_inv_all',
    'S005580643': 'plate_inv_all',
    'S005580644': '总库存_镀锌板卷',  # sheets: ferrous_w
    'S005580645': '总库存_彩涂板卷',  # sheets: ferrous_w
    'S005580646': 'crc_inv_all',
    'S005580652': 'crc_prod_all',
    'S005580993': 'ckc_au_cfr_cn',
    'S005616301': 'weaviing_dnstream_invdays_mill',
    'S005653695': 'bf_workrate_247',
    'S005653704': 'coke_inv_230cokery',
    'S005653705': 'ckc_inv_230cokery',
    'S005656437': 'consteel_dsales_banksteel',
    'S005658949': 'PF_inv_czce_warrant',
    'S005696248': 'fg_inv_mill',
    'S005696261': 'fg_util_adj',
    'S005696262': '浮法玻璃_生产线条数剔除僵尸产线_合计_当周值',  # sheets: const_d
    'S005696263': '浮法玻璃_生产线条数剔除僵尸产线_在产_当周值',  # sheets: const_d
    'S005696264': 'fg_util_w',
    'S005808359': 'cu_lme_3m_close',
    'S005808360': 'al_lme_3m_close',
    'S005808361': 'pb_lme_3m_close',
    'S005808362': 'zn_lme_3m_close',
    'S005808363': 'sn_lme_3m_close',
    'S005808364': 'ni_lme_3m_close',
    'S005814718': '库存_螺纹钢35城市',  # sheets: ferrous_w
    'S005949692': '国内市场价_玻璃5_0mm_全国均价',  # sheets: const_d
    'S005951203': 'pb_60conc_tc_ports',
    'S005953318': 'csteel_prod_cisa',
    'S005953322': 'pigiron_prod_cisa',
    'S005953326': 'steelproducts_prod_cisa',
    'S005953372': 'ur_shandong_spot',
    'S005955220': 'ma_spot_sd',
    'S005955224': 'ma_east_spot',
    'S005956443': 'si_421_sichuan',
    'S005961124': 'io_inv_31ports',
    'S005961126': 'io_inv_41ports',
    'S005961128': 'io_inv_45ports',
    'S005961196': 'io_inv_31ports_trade',
    'S005961326': 'io_removal_45ports',
    'S005971281': 'cu_mine_inv_ports',
    'S006018632': 'MA_inv_ports_total',
    'S006095407': 'l_tj_spot',
    'S006154226': 'eaf_prodcost_east',
    'S006154238': 'eaf_util_87mills',
    'S006154239': '开工率_独立电弧炉钢厂_全国',  # sheets: ferrous_w
    'S006154248': 'eaf_util_all',
    'S006154249': '开工率_电炉钢厂_全国',  # sheets: ferrous_w
    'S006157941': 'cu_prem_bonded_warrant',
    'S006157944': 'cu_prem_bonded_cif',
    'S006157947': 'cu_prem_bonded_cny',
    'S006157950': '平均价_tc指数_铜精矿_当周值_上海有色',  # sheets: base_w
    'S006158372': 'pb_50conc_tc_hunan',
    'S006158375': 'pb_50conc_tc_yunnan',
    'S006158378': 'pb_50conc_tc_guangxi',
    'S006158381': 'pb_50conc_tc_neimeng',
    'S006158384': 'pb_50conc_tc_henan',
    'S006158933': '平均价_锰矿加蓬mn44pct_钦州港',  # sheets: ferrous_d
    'S006158942': 'mn_44_gabon_tj',
    'S006159069': 'si_553_nonoxy_east',
    'S006159072': 'si_553_oxy_east',
    'S006159081': 'si_421_east',
    'S006159146': 'si_553_oxy_kunming',
    'S006159154': 'si_421_kunming',
    'S006159164': 'si_553_nonoxy_sichuan',
    'S006159859': '平均价_加工费_zamak3锌合金_广东_当周值_上海有色',  # sheets: base_w
    'S006159865': '平均价_加工费_zamak3锌合金_浙江_当周值_上海有色',  # sheets: base_w
    'S006159871': '平均价_加工费_zamak3锌合金_江苏_当周值_上海有色',  # sheets: base_w
    'S006159877': '平均价_加工费_zamak3锌合金_福建_当周值_上海有色',  # sheets: base_w
    'S006161093': 'ss_inv_social_200',
    'S006161094': 'ss_inv_social_300',
    'S006161095': 'ss_inv_social_400',
    'S006161096': 'ss_inv_social_all',
    'S006167225': 'bauxite_inv_az_ports',
    'S006167236': 'alumina_inv_az_ports',
    'S006374450': 'refined_products_inv_indep',
    'S006404817': 'sa_margin_anjian',
    'S006404818': 'sa_margin_lianchan',
    'S006404825': '仓单数量_粳米',  # sheets: warrant_d
    'S006404843': 'lu_inv_ine_warrant',
    'S006404844': 'pg_inv_dce_warrant',
    'S006409299': 'bc_inv_ine_warrant',
    'S006466874': 'AP_inv_frozen',
    'S006563225': 'al_6063rod_inv_social',
    'S006574700': 'io_inv_sf_45ports',
    'S006574701': '在港船舶数_铁矿石_总计',  # sheets: ferrous_w
    'S006574742': '平仓含税价_准一级冶金焦_a13_s0_7_csr60_mt7_沫7_青岛港_山西产',  # sheets: const_d
    'S006700187': 'lh_inv_dce_warrant',
    'S006955749': 'au_etf_gbs_holding',
    'S006955753': 'au_etf_sgbs_holding',
    'S006955757': 'au_etf_phau_holding',
    'S006955761': 'au_etf_gold_holding',
    'S006955770': 'au_etf_cef_holding',
    'S006955772': 'ag_etf_sivr_holding',
    'S006955778': 'ag_etf_phag_holding',
    'S006955782': 'ag_etf_etpmag_holding',
    'S006955786': 'ag_etf_pslv_holding',
    'S006955790': 'ag_etf_cef_holding',
    'S007247253': 'bu_inv_shfe_all',
    'S007247476': 'bu_invcap_shfe_all',
    'S007746654': 'nmf_qd',
    'S007756870': '主要航线即期运价_综合运价',  # sheets: const_d
    'S007756871': '主要航线即期运价_上海_鹿特丹',  # sheets: const_d
    'S007756874': '主要航线即期运价_上海_洛杉矶',  # sheets: const_d
    'S008061203': 'coke_ts_xb',
    'S008061213': 'coke_changzhi_xb',
    'S008061232': 'coke_sh_xb',
    'S008061234': 'coke_xuzhou_xb',
    'S008082266': 'sf_workrate_cn',
    'S008082267': 'sf_dprod_cn',
    'S008082268': 'sf_dmd_cn',
    'S008082269': 'sf_prod_cn',
    'S008082270': 'sm_workrate_cn',
    'S008082271': 'sm_dprod_cn',
    'S008082272': 'sm_dmd_cn',
    'S008082273': 'sm_prod_cn',
    'S008211716': 'tci_天津_欧洲基本港',  # sheets: const_d
    'S008211719': 'tci_天津_美国西岸基本港',  # sheets: const_d
    'S008527032': 'alumina_spot_guangxi',
    'S008527035': 'alumina_spot_guizhou',
    'S008527041': 'alumina_spot_shanxi',
    'S008527044': 'alumina_spot_henan',
    'S008527822': 'al_wm0_phybasis_low',
    'S008527823': 'al_wm0_phybasis_high',
    'S008527843': 'zn_wm0_phybasis',
    'S008527848': 'zn_wm1_phybasis',
    'S008545965': 'cu_scrap_1_sh',
    'S008545990': 'cu_scrap_2_sh',
    'S008546019': '平均价_废铜_国标低氧杆8mm含税_上海',  # sheets: base_d
    'S008546020': '平均价_废铜_国标无氧杆8mm含税_上海',  # sheets: base_d
    'S008618297': 'scfis_欧洲航线基本港',  # sheets: const_d
    'S008618298': 'scfis_美西航线基本港',  # sheets: const_d
    'S008618299': 'consteel_dsales_mysteel',
    'S008618440': 'sm_inv_mill',
    'S008618447': 'sf_inv_mill',
    'S008618451': 'sm_stockdays',
    'S008618763': '库存_苯乙烯工厂库存_全国',  # sheets: petchem_w
    'S008618767': 'eb_inv_port_south',
    'S008618768': 'eb_inv_port_south_trader',
    'S008618769': 'bz_inv_port_east',
    'S008679890': 'pbf_prem',
    'S008679891': 'nmf_prem',
    'S008679892': 'macf_prem',
    'S008679893': 'jmb_prem',
    'S008679896': 'iocj_prem',
    'S008679899': 'pbf_sb',
    'S008679900': 'nmf_sb',
    'S008679901': 'macf_sb',
    'S008679902': 'jmb_sb',
    'S008679905': 'ssf_sb',
    'S008679908': 'iocj_sb',
    'S008679910': 'brbf_sb',
    'S008871802': 'cu_rod_8_procfee_nanchu',
    'S008871805': 'cu_rod_2.6_procfee_nanchu',
    'S008871808': '平均价_8mm无氧铜杆_南储广东',  # sheets: base_d
    'S008871811': '平均价_2_6mm无氧铜杆_南储广东',  # sheets: base_d
    'S008871816': 'al_nanchu_phybasis',
    'S008871823': 'zn_nanchu_phybasis',
    'S009010885': 'al_inv_social_all',
    'S009010885.1': '电解铝_库存_合计',  # sheets: base_w
    'S009045420': 'steel_inv_social',
    'S009065244': 'PF_inv_mill_treasury',
    'S009065245': 'PF_inv_mill_physical',
    'S009065246': 'PTA_inv_mill',
    # 'S009065254': 'cement_idx_cj',
    # 'S009065264': 'cement_idx_cn',
    'S009067527': 'nmf_cfd',
    'S009097506': 'long_inv_social',
    'S009122299': 'bf_workrate_num',
    'S009122311': 'bf_workrate_cap',
    'S009128522': 'PTA_prodcost_cn_d',
    'S009128523': 'PTA_margin_cn_d',
    'S009134956': 'sa_wprod_cn',
    'S009137283': 'cu_cj_phybasis',
    'S009137286': 'cu_cjb_phybasis',
    'S009137289': 'al_cj_phybasis',
    'S009137292': 'al_cjb_phybasis',
    'S009137295': 'ni_nis_cjb_spot',
    'S009137298': '平均价_硫酸镍_长江有色',  # sheets: base_d
    'S009138370': 'fo_180cst_m1_sgp',
    'S009138371': 'fo_180cst_m2_sgp',
    'S009138374': 'fo_380cst_m1_sgp',
    'S009138375': 'fo_380cst_m2_sgp',
    'S009138389': 'lu_0.5_prem_sgp',
    'S009138595': '库存量_原油_api_同比',  # sheets: petchem_w
    'S009160074': 'cu_scrap_1_spot_jzh',
    'S009160107': 'cu_scrap_2_spot_jzh',
    'S009200256': 'ni_prem_import',
    'S009200262': '平均价_镍豆升贴水',  # sheets: base_d
    'S009200268': 'ni_nis_spot_gi',
    'S009222920': '仓单数量_黄金_总计',  # sheets: warrant_d
    'S009223764': 'ss_inv_shfe_d',
    'S009224314': 'pmi_lgsc_steel_sales',
    'S009224315': 'pmi_lgsc_steel_inv',
    'S009273405': 'ni_nis_spot_battery',
    'S009341250': 'coke_inv_247mill',
    'S009341251': 'ckc_inv_247mill',
    'S009341305': 'coke_inv_cokery',
    'S009341306': 'ckc_inv_cokery',
    'S009620177': 'sn_60conc_tc_jiangxi',
    'S009620198': 'sn_60conc_tc_guangxi',
    'S009620213': 'sn_40conc_tc_yunnan',
    'S009621341': 'al_rod_6063_procfee_jiangxi',
    'S009621410': 'al_rod_6063_procfee_sichuan',
    'S009621515': '平均价_加工费6063铝棒_江苏有色',  # sheets: base_d
    'S009621539': 'al_rod_6063_procfee_gansu',
    'S009621955': 'cu_sh_phybasis',
    'S009626046': 'al_scrap_shreded_spot_foshan',
    'S009626378': 'pb_scrap_ebike_sh',
    'S009626602': 'sn_scrap_pure_bulk_shandong',
    'S009626605': 'sn_scrap_bulk_shandong',
    'S009626611': 'sn_scrap_slag_shandong',
    'S009626791': 'ni_scrap_spot_foshan',
    'S009630097': '国内市场价_烧碱50pct离子膜碱_山东_主流价',  # sheets: petchem_d
    'S009637664': 'PK_inv_czce_warrant',
    'S009761758': 'espo_prem_sd',
    'S009761761': 'espo_spot_sd',
    'S009761779': 'oman_prem_sd',
    'S009761782': 'oman_spot_sd',
    'S009767207': 'crude_arrival_prem',
    'S009767208': 'crude_imp_spot_cn',
    'S009780583': 'pb_scrap_autostarter_sh',
    'S009785426': 'ckc_outstock_ganqimaodu',
    'S010308683': 'ckc_inv_110washery',
    'S010338984': 'coke_inv_ports',
    'S010361921': 'ni_cj1_spot',
    'S010596299': 'alumina_spot_cnports',
    'S010596302': 'alumina_aus_fob',
    'S010861418': 'sa_heavy_shahe',
    'S010998475': 'io_inv_sb_ausbrl_7ports',
    'S011214521': 'cu_inv_bonded_gd',
    'S011258003': '国产铝土矿_海漂量_总计',  # sheets: base_w
    'S011258007': '国产铝土矿_发运量_总计',  # sheets: base_w
    'S011258021': 'bauxite_inv_ports_inv',
    'S011311758': 'sa_sales_prod_ratio',
    'S011318571': 'syn_ammonia_margin_coal',
    'S011318572': 'syn_ammonia_margin_gas',
    'S011319336': 'ru_inv_bonded_traders_qd',
    'S011319345': 'br_inv_social',
    'S011319484': 'sh_inv_mill_all',
    'S011319499': 'BOPP_invdays_raw',
    'S011319502': 'CPP_invdays_finished',
    'S011319504': 'pp_nonwoven_inv_finished',
    'S011319505': 'PET_chip_invdays_mill',
    'S011319524': 'PF_viscosefiber_inv_mill',
    'S011319525': 'PF_viscosefiber_invdays_mill',
    'S011319572': 'wovenplastics_invdays_raw_mill_sm',
    'S011319574': 'wovenplastics_inv_finished_mill_lg',
    'S011319581': 'pe_pipe_invdays',
    'S011333399': 'ni_inv27_plate_dom',
    'S011333401': 'ni_inv27_plate_bonded',
    'S011333403': 'ni_inv27_all',
    'S011334192': 'pb_inv_social_all',
    'S011334489': 'zn_inv_smelter_finished',
    'S011334532': '港口库存_锌精矿_合计_当周值',  # sheets: base_w
    'S011334851': 'pg_east_spot_idx',
    'S011334852': 'pg_south_spot_idx',
    'S011334855': 'pg_sd_spot_idx',
    'S011799884': 'ckc_au_fob',
    'S011936208': 'CF_inv_weaving_mth',
    'S011936209': 'CF_inv_weaving_lg_mth',
    'S012116529': 'ckc_inv_ports',
    'S012116534': 'coke_inv_4ports',
    'S012185571': 'pta_east_spot2',
    'S012518267': 'lc_util_mth',
    'S012683154': 'cement_inv_ratio',
    'S012683281': 'cement_clinker_inv_ratio',
    # 'S012691163': 'concrete_idx_cn',
    # 'S012691166': '混凝土价格指数_华东指数',  # sheets: const_d
    'S012937595': '金属硅_社会库存_总计',  # sheets: base_w
    'S013050004': 'cement_dispatch_rate',
    'S013050080': 'cement_mill_run_rate',
    'S013735402': '锡锭_社会库存_合计_当周值',  # sheets: base_w
    'S015202398': 'cu_scrap1_diff_gd',
    'S015202399': 'cu_scrap1_diff_tj',
    'S015202400': 'cu_scrap1_fv_diff_gd',
    'S015202401': 'cu_scrap_import_margin',
    'S015202402': 'cu_import_margin_sh',
    'S015756661': 'cement_clinker_util',
    'S016635856': 'sp_pz_ca_ma_sh',
    'S016681640': 'bz_korea_fob_usd',
    'S016681643': 'bz_cn_cfr_usd',
    'S016702541': 'zn_50conc_tc_neimeng',
    'S016702544': 'zn_50conc_tc_yunnan',
    'S016702547': 'zn_50conc_tc_hunan',
    'S016702550': 'zn_50conc_tc_guangxi',
    'S016702553': 'zn_50conc_tc_henan',
    'S016702556': 'zn_50conc_tc_sichuan',
    'S016702559': 'zn_50conc_tc_shanaxi',
    'S016702562': 'zn_48conc_tc_ports',
    'S016720335': 'margin_mktcap_ratio_cn',
    'S016720336': 'margin_mktvol_ratio_cn',
    'S016734044': 'y_inv_mill',
    'S016841666': 'ur_henan_spot',
    'S016842079': 'ur_gcc_cfr_usd',
    'S016842082': 'ur_cn_fob_usd',
    'S017068418': 'fg_margin_natgas',
    'S017209293': 'c_inv_mill',
    'S017209307': 'c_inv_ports_4north',
    'S017209311': '玉米库存_广东港合计',  # sheets: ag_w
    'S017385971': 'm_invdays_downstream',
    'S017406429': 'CJ_inv_sample',
    'S017438009': 'fg_dprod',
    'S017498544': 'lc_bat_dom_jiangxi',
    'S017498565': 'lc_bat_sam_fob',
    'S017498571': 'lc_bat_asia_cif',
    'S017498574': 'lc_bat_eu_cif',
    'S017639430': 'io_inv_imp_mill(247)',
    'S017643268': 'pe_inv_mill',
    'S017647060': 'cs_inv_mill',
    'S017658895': 'sm_neimeng_cost',
    'S017658896': 'sm_ningxia_cost',
    'S017658905': 'sm_margin_north',
    'S017658906': 'sm_margin_south',
    'S017659509': 'sf_ningxia_cost',
    'S017659510': 'sf_neimeng_cost',
    'S017659519': 'sf_ningxia_margin',
    'S017659520': 'sf_neimeng_margin',
    'S017935492': 'ZC_inv_social',
    'S017942027': 'c_invdays_downstream',
    'S017992344': 'bean_inv_mill',
    'S017992372': 'bean_inv_ports_full',
    'S018042405': 'v_inv_mill_mth',
    'S018052590': 'm_inv_mill_lg',
    'S018380017': 'jd_invdays_prod',
    'S018380018': 'jd_invdays_transit',
    'S018479716': 'OI_inv_mill_east',
    'S018479717': 'OI_inv_mill_guangxi',
    'S018479719': 'OI_inv_mill_coastal',
    'S018489426': 'RM_inv_mill_mth',
    'S018493782': 'p_inv_mill',
    'S018521625': 'fu_inv_cnship',
    'S018553310': 'gasoline_inv_social',
    'S018553311': 'diesel_inv_social',
    'S018574405': '螺纹钢20mmhrb400_成本含税_江苏',  # sheets: ferrous_w
    'S018634610': 'PK_inv_mill',
    'S018634615': 'PK_oil_inv_mill',
    'S018696379': 'sn_inv_social_all',
    'S018791012': '社会库存_电解镍_镍豆_六地总计_当周值_上海有色',  # sheets: base_w
    'S018857355': '油厂库存量_菜籽粕_总计',  # sheets: ag_w
    'S018862300': 'RM_inv_mill_east',
    'S019254411': 'fg_margin_coal',
    'S019254412': 'fg_margin_petcoke',
    'S019254955': '社会库存_天然橡胶_中国_期末值',  # sheets: petchem_w
    'S019255498': 'solarglass_util',
    'S019255501': 'solarglass_dprod',
    'S019255501.1': '在产日熔量_光伏玻璃_当周值',  # sheets: ferrous_w
    'S019255552': '企业库存_光伏玻璃_当周值',  # sheets: const_d
    'S019506839': 'MX_inv_east',
    'S019506841': 'MX_inv_south',
    'S019732993': 'pe_inv_social',
    'S019732994': 'pe_inv_traders',
    'S019733356': 'pg_inv_port_south',
    'S019733357': 'pg_inv_port_east',
    'S019735959': 'si_inv_gfex_d',
    'S019779606': 'cu_blister_rc_south',
    'S019779609': 'cu_blister_rc_north',
    'S019779612': 'cu_anode_rc',
    'S019779625': 'lc_ind_dom_cn_spot',
    'S019822995': 'hrc_bs_fob',
    'S019823001': 'hrc_mideast_cfr',
    'S019823002': 'hrc_sea_cfr',
    'S019823003': 'hrc_cn_fob',
    'S019823046': 'rebar_sea_cfr',
    'S019823047': 'rebar_cn_fob',
    'S019823072': 'billet_bs_fob',
    'S019823080': 'billet_cn_fob',
    'S019823083': '出口价黑海_波罗的海fob_板坯_独联体',  # sheets: ferrous_w
    'S019833712': '产能利用率_浮法玻璃_当周值',  # sheets: ferrous_w
    'S019833798': '企业库存样本_浮法玻璃_当周值',  # sheets: const_d
    'S019848684': 'ao_inv_shfe_d',
    'S019848689': 'ao_inv_shfe_mill_d',
    'S019985011': 'PTA_inv_social_wk',
    'S020003107': 'SH_util_w',
    'S020003109': '企业库存_烧碱_期末值',  # sheets: petchem_w
    'S020003119': 'SH_margin_sd',
    'S020004695': 'PX_margin_cn_w',
    'S020089266': '仓单数量_丁二烯橡胶_仓库_总计',  # sheets: warrant_d
    'S020098434': 'lc_inv_gfex_d',
    'S020190572': 'lc_ind_dom_east_spot',
    'S020190575': 'lc_bat_dom_east_spot',
    'S020190578': 'lc_ind_dom_sichuan_spot',
    'S020190581': 'lc_bat_dom_sichuan_spot',
    'S020207789': 'ni_mhp_34_ports',
    'S020209589': 'PX_naph_spd_w',
    'S020209590': 'PX_MX_spd_w',
    'S020210081': 'PX_util_kr',
    'S020459829': '价差_烧碱_50_32_山东地区',  # sheets: petchem_d
    'S020559734': 'br_inv_traders',
    'S020602643': 'ru_half_steel_tire_invdays_sd',
    'S020602996': 'ru_all_steel_tire_invdays_sd',
    'S021039360': 'viu_fe',
    'S021182443': 'scrap_arr_300mill',
    'S021182444': 'scrap_use_300mill',
    'S021182446': 'scrap_invdays_300mill',
    'S021277623': 'scrap_use_mill_eaf',
    'S021277629': 'scrap_ratio_mill_all',
    'S021277634': 'scrap_inv_mill_eaf',
    'S021281525': 'rebar_eaf_prodcost_base_east',
    'S021281530': 'rebar_eaf_prodcost_base_cn',
    'S021281533': 'rebar_eaf_prodcost_pk_east',
    'S021281538': 'rebar_eaf_prodcost_pk_cn',
    'S021281541': 'rebar_eaf_prodcost_opk_east',
    'S021281546': 'rebar_eaf_prodcost_opk_cn',
    'S021281565': 'rebar_eaf_margin_base_east',
    'S021281570': 'rebar_eaf_margin_base_cn',
    'S021374817': 'scrap_use_mill_all',
    'S021374832': 'scrap_inv_mill_all',
    'S021794952': 'viu_al',
    'S021992679': '车板含税价_锰矿_mn_45_块矿_加蓬_南方港',  # sheets: ferrous_d
    'S021992693': 'mn_45_gabon_northports',
    'S022010507': 'pr_north_spot',
    'S022014758': 'PL_inv_all',
    'S022014760': 'bz_inv_ports',
    'S022117012': 'SH_inv_czce_warrant',
    'S022117035': 'PX_inv_czce_warrant',
    'S022319791': 'SH_inv_czce_unwarrant',
    'S022319792': 'PX_inv_czce_unwarrant',
    'S023487508': '仓单数量_氧化铝_总计',  # sheets: warrant_d
    'S023510418': 'alumina_cfr_cn',
    'S023510421': 'alumina_fob_au',
    'S023828847': 'cu_prem_cif_tw',
    'S023828850': 'cu_prem_cif_sea',
    'S024334444': 'fg_margin_avg',
    'S024761635': 'PET_invdays_mill',
    'S026354485': 'pg_inv_mill_res',
    'S026354493': 'pg_inv_mill_all',
    'S026354501': 'pg_inv_port_all',
    'S026354504': 'pg_inv_port_north',
    'S026354506': 'pg_invratio_ports',
    'S033367931': 'ps_inv_gfex_d',
    'S035614282': 'PR_inv_czce_warrant',
    'T025173022': 'si_421_prem_gd',
}
