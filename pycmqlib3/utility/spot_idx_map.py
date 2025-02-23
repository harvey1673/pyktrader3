import pandas as pd
import numpy as np
from pycmqlib3.analytics.tstool import vat_adj

index_map = {
    # macro
    'G002600770': 'usgg2yr',
    'G002600774': 'usgg10yr',
    'G002600783': 'usggt10yr',
    'G013233151': 'usggbe5',
    'G013233152': 'usggbe10',
    'G013233153': 'inflation_exp_5y_us',
    #'G005172253': 'usgg10yr_2yr_spd',

    'M002842089': 'usdcny_mid',
    'M004147024': 'usdcnh_spot',
    'M011202650': 'usdcnh_close',
    'M004147023': 'usdcny_spot', # 4:00pm
    'M004370159': 'usdcny_spot2', # 4:30pm
    'M004377555': 'usdcny_spot_volume',

    'G002600791': 'libor3m',
    'G002600885': 'dxy',
    #'G002601505': 'vix',
    'G003082203': 'vix',
    'G003082207': 'vvix',
    'G003082211': 'vxeem',
    'G003082215': 'vxd', # Vol for DJ
    'G003082227': 'vxn', # Vol for Nasdaq
    'G003146263': 'usdzar_xe',
    'G003146267': 'usdbrl_xe',
    'G003146268': 'usdnok_xe',
    'G003146276': 'usdkrw_xe',
    'G003146252': 'usdcny_xe',
    'G003146248': 'usdaud_xe',
    'G003146249': 'usdcad_xe',
    'G003146245': 'usdeur_xe',
    'G003146246': 'usdgbp_xe',
    'G003146256': 'usdjpy_xe',
    'G004849308': 'usdclp_xe',
    'G019711418': 'usdcnh_xe',

    'L004387127': 'usdcny_rr25_3m',
    'M020092912': 'fin_cond_idx_cn',
    'L003297115': 'corpbond_yield_aa_1y',
    'L002856618': 'schbill_yield_aa_1y',
    'L019834710': 'stbill_yield_aa_1y',
    'L004476676': 'banknotes_yield_aa_1y',

    'L001619493': 'dr007_cn',
    'L004317616': 'fr007_cn',
    'L004317619': 'fdr007_cn',
    'L004366599': 'usdcny_on',
    'L004366605': 'usdcny_1m',
    'L015211333': 'cnh_hibor_1m',
    'M002816451': 'shibor_1m',
    'M002816452': 'shibor_3m',
    'M002816455': 'shibor_1y',
    'M002816576': 'r007_cn',
    'L004349332': 'cn_govbond_yield_3m_sch',
    'L004349335': 'cn_govbond_yield_1y_sch',
    'L001618805': 'cn_govbond_yield_1y',
    'L001619213': 'cn_govbond_yield_2y',
    'L001618480': 'cn_govbond_yield_5y',
    'L001619214': 'cn_govbond_yield_10y',
    'G009067321': "eco_policy_uncertainty_idx_us",
    'G005432431': 'citi_eco_surprise_idx_cn',
    'G005326174': 'citi_eco_surprise_idx_us',
    'G005432432': 'citi_eco_surprise_idx_eu',
    'G005432436': 'citi_eco_surprise_idx_global',
    'G005432438': 'citi_eco_surprise_idx_em',
    'G005432439': 'citi_eco_surprise_idx_asia',
    'S003587817': "margin_outstanding_total_cn",
    'S016720335': "margin_mktcap_ratio_cn",
    'S016720336': "margin_mktvol_ratio_cn",

    # 'M016266040': 'icpi_all',
    # 'M016266041': 'icpi_food',
    # 'M016266042': 'icpi_clothes',
    # 'M016266043': 'icpi_housing',
    # 'M016266044': 'icpi_service',

    "M004369935": 'pmi_cn_cons_all',
    "M005933607": 'pmi_cn_cons_new_order',
    "M005933608": 'pmi_cn_cons_rm_px',
    "M005933609": 'pmi_cn_cons_px',
    "M005933610": 'pmi_cn_cons_hr',
    "M005933611": 'pmi_cn_cons_bus_exp',
    #"M011799928": 'pmi_cn_cons_exports',
    "M003559320": 'pmi_cn_steel_all',
    "M003559341": 'pmi_cn_steel_prod',
    "M003559342": 'pmi_cn_steel_rm_vol',
    "M003559343": 'pmi_cn_steel_rm_inv',
    "M003559344": 'pmi_cn_steel_new_order',
    "M003559345": 'pmi_cn_steel_exports',
    "M003559346": 'pmi_cn_steel_inv',
    "M003559347": 'pmi_cn_steel_rm_px',
    "S004038574": 'pmi_lgsc_steel_all',
    "S009224315": 'pmi_lgsc_steel_inv',
    "S009224314": 'pmi_lgsc_steel_sales',
    "S004038576": 'pmi_lgsc_steel_tot_order',
    "S009224316": 'pmi_lgsc_steel_mkt_exp',
    "S004038575": 'pmi_lgsc_steel_purchase_exp',
    "M002043802": 'pmi_cn_manu_all',
    "M002043804": 'pmi_cn_manu_new_order',
    "M002043811": 'pmi_cn_manu_rm_inv',
    "M002043805": 'pmi_cn_manu_exports',
    "M002043809": 'pmi_cn_manu_imports',
    "M002043808": 'pmi_cn_manu_purchase',
    "M002043806": 'pmi_cn_manu_curr_order',
    "M003721097": 'pmi_cn_manu_bus_exp',    
    #"M002811186": 'pmi_caixin_manu_all',
    "M004088026": "epmi_cn_all",
    "M004088027": "epmi_cn_prod",
    "M004088028": "epmi_cn_order",
    "M004302214": "epmi_cn_exports",
    "M004302216": "epmi_cn_inv",
    "M004302217": "epmi_cn_purchase",
    "S009065264": "cement_idx_cn",
    "S009065254": "cement_idx_cj",
    "S012691163": "concrete_idx_cn",

    # ferrous
    'S003019324': 'plt62',
    'S002808964': 'plt58',
    'S002808963': 'plt65',
    'S000020892': 'hrc_sh',
    'S000020891': 'hrc_gz',
    'S002859801': 'hrc_tj',
    'S000020890': 'hrc_bj',

    'S000020868': 'rebar_sh',
    'S000020867': 'rebar_gz',
    'S000020869': 'rebar_sy',
    'S002917430': 'plate_8mm',
    'S000020903': 'crc_sh',
    'S000020902': 'crc_gz',
    #'S000020904': 'crc_shenyang',
    #'S000020901': 'crc_bj',
    'S000020933': 'gi_0.5_sh',
    #'S000020939': 'gi_1.0_sh',
    'S002917486': 'gi_0.5',

    'S002911091': 'pbf_cfd',
    'S004210693': 'macf_cfd',
    'S004317128': 'fbf_qd',
    'S004317138': 'royhill_rz',
    'S005429350': 'jmb_qd',
    'S004317129': 'ssf_qd',

    'S002827225': 'billet_ts',
    'S002827223': 'billet_js',

    'S002954691': 'scrap_sh',
    'S002827258': 'scrap_zjg',
    'S002827270': 'scrap_ts',
    'S002983449': 'pci_yangquan',
    'S002983448': 'pci_jincheng',
    'S004369291': 'coke_tj',
    'S004425298': 'coke_sub_a_rz',
    'S004369291': 'coke_sub_a_tj',
    'S008061234': 'coke_xuzhou_xb',
    'S008061203': 'coke_ts_xb',
    'S008061232': 'coke_sh_xb',
    'S008061213': 'coke_changzhi_xb',
    'S004238793': "coke_shandong",
    'S004238790': "coke_hebei",
    'S004238796': "coke_henan",
    'S002877257': "ckc_a10v24s08_lvliang",
    'S002877258': "ckc_a9v18s10_lvliang",
    'S002877299': "ckc_a10v24s10_ts",
    'S009785426': "ckc_outstock_ganqimaodu",
    'S004085268': "ckc_stock_ganqimaodu",

    'S002917675': 'strip_2.5x355',
    'S002917688': 'strip_3.0x685',
    'S002917771': 'pipe_1.5x3.25',
    'S002917787': 'pipe_4x3.75',
    'S002917733': 'seemless_pipe_108x4.5',

    #'S002917631': 'hsec_300x300',
    'S002917646': 'hsec_400x200',
    #'S002917616': 'hsec_200x100',
    'S002917514': 'channel_16',
    'S002917532': 'ibeam_25',
    'S002917550': 'angle_50x5',
    'S002917279': 'highwire_6.5',
    'S004378612': 'coal_5500_jingtang',
    'S002882871': 'coal_5500_sx_qhd',
    'S002837009': 'coal_5500_qhd', # not good
    'S004381182': 'coal_6000_newc_fob',
    'S004381181': 'coal_6000_api4_sa',
    'S004381180': 'coal_6000_api2_ara',

    'S004018814': 'io_loading_14ports_ausbzl',
    'S004226190': 'io_inv_imp_mill_0_200',
    'S004226191': 'io_inv_mill_200_300',
    'S004226192': 'io_inv_mill_300_400',
    'S004226193': 'io_inv_mill_400',
    'S005961124': 'io_inv_31ports',
    'S005961126': 'io_inv_41ports',
    'S005961128': 'io_inv_45ports',
    'S005961196': 'io_inv_31ports_trade',

    'S004226161': 'io_inv_imp_mill(64)',
    'S004226163': 'io_inv_dom_mill(64)',
    'S003817887': 'io_invdays_imp_mill(64)',

    'S005961310': 'io_removal_31ports',
    'S005961321': 'io_removal_41ports',
    'S005961326': 'io_removal_45ports',
    'S008618299': 'consteel_dsales_mysteel',
    'S005656437': 'consteel_dsales_banksteel',

    "S005470470": 'fg_5mm_shahe',
    'S005470469': 'fg_5mm_north',

    "S002863470": "brent_dtd_spot",
    "S002863469": "oman_spot",
    "S000006462": "dubai_spot",
    "S004242343": "espo_spot",
    "S003011318": "propane_cfr_asia_n",
    "S003011351": "butane_cfr_asia_n",
    "S003011327": "propane_cfr_china_s",
    "S003011360": "butane_cfr_china_s",
    "S003011336": "propane_cfr_tw",
    "S003011369": "butane_cfr_tw",
    "S003031624": "fo_180cst_sgp",
    "S003031623": "fo_380cst_sgp",
    "S003011283": "fo_180cst_east",
    "S003011289": "fo_180cst_sh",
    "S003011302": "fo_180cst_xiamen",
    "S005126417": "lu_0.5_sgp",
    "S005028348": "pg_100ppi_spot",
    "S005028354": "pg_qd_spot",
    "S005028355": "pg_jincheng_shandong",

    "S004163704": "bu_heavy_shandong",
    "S004163701": "bu_heavy_north",
    "S004163702": "bu_heavy_east",
    "S004242346": "ru_100ppi_spot",
    "S004321834": "ru_scrwf_kunming",
    "S004321831": "ru_scrwf_zhejiang",
    "S004321822": "ru_scrwf_jiangsu",
    "S004156652": "eb_east_spot",
    "S004156649": "eb_north_spot",
    "S004156658": "eb_south_spot",
    "S002955437": "eb_cfr_cn",
    "S005349978": "eb_100ppi_spot",
    "S003994516": "ma_spot_jiangsu",
    "S003994507": "ma_spot_hebei",
    "S003994531": "ma_cfr_cn",
    "S004077476": "pp_linyi_spot",
    "S004242351": "pp_100ppi_spot",
    "S004161475": "pp_wenzhou_spot",
    "S003157699": "l_7042_tj",
    "S003157759": "l_7042_sh",
    "S002836797": "l_7042_east",
    "S002836796": "l_7042_north",
    "S002836798": "l_7042_south",
    "S003994600": "eg_east_spot",
    "S003994603": "eg_south_spot",
    "S002893910": "eg_north_exw",
    "S002956186": "eg_cfr_cn",
    "S002956195": "eg_cfr_sea",
    "S002835975": "pta_east_spot",
    "S004407074": "pf_north_spot",
    "S004407080": "pf_east_spot",
    "S004407077": "pf_fujian_spot",

    "S004242725": "fg_100ppi",  # x80 = RMB/ton
    "S002825712": "pvc_cac2_north",
    "S002825715": "pvc_cac2_east",
    "S002825718": "pvc_cac2_south",
    "S002825721": "pvc_cac2_central",
    "S002825727": "pvc_ethylene_east",
    "S002825730": "pvc_ethylene_south",

    "S002825734": "sa_heavy_north",
    "S002825740": "sa_heavy_east",
    "S002825733": "sa_light_north",
    "S002825739": "sa_light_east",
    "S010861418": "sa_heavy_shahe",
    "S005349979": "sa_heavy_sys",
    "S002959491": "sm_65s17_neimeng",
    "S002959498": "sm_65s17_tj",
    "S002959495": "sm_65s17_guangxi",
    "S002959499": "sm_65s17_gansu",
    "S017658895": "sm_neimeng_cost",
    "S017658896": "sm_ningxia_cost",
    "S017658905": "sm_margin_north",
    "S017658906": "sm_margin_south",
    "S006158942": "mn_44_gabon_tj",
    #"S006158933": "mn_44_gabon_qingzhou",
    #"S021992679": "mn_45_gabon_southports",
    "S021992693": "mn_45_gabon_northports",

    "S004789784": "sf_72_ningxia",
    "S004789790": "sf_72_neimeng",
    "S004789786": "sf_72_gansu",
    "S005068112": "sm_65s17_shmet",
    "S005068030": "sf_72_shmet",
    "S005068027": "sf_75_shmet",
    "S017659510": "sf_neimeng_cost",
    "S017659509": "sf_ningxia_cost",
    "S017659520": "sf_neimeng_margin",
    "S017659519": "sf_ningxia_margin",
    "S008082266": "sf_operating_rate",


    'S008618440': 'sm_inv_mill',
    'S008618447': 'sf_inv_mill',
    "S005696248": "fg_inv_mill",
    "S005439547": "v_inv_social",
    "S005439586": 'sa_inv_mill_all',
    "S011319484": "sh_inv_mill_all",
    "S003138068": "coke_inv_ports_tj", # 20110121
    "S003138069": "coke_inv_ports_lyg", # 20110121
    "S003138070": "coke_inv_ports_rz", # 20110121
    "S010338984": "coke_inv_ports", # 20211217
    "S012116529": "ckc_inv_ports", # 20220107
    "S009341306": "ckc_inv_cokery", #20210115

    'S005814718': 'rebar_inv_social',
    'S005580635': 'hrc_inv_social',
    'S005580633': 'wirerod_inv_social',
    'S005580639': 'crc_inv_social',
    'S005580636': 'plate_inv_social',

    'S009045420': 'steel_inv_social',
    'S009097506': 'long_inv_social',

    'S004378418': 'rebar_inv_mill',
    'S004378419': 'wirerod_inv_mill',
    'S004378420': 'hrc_inv_mill',
    'S004378421': 'crc_inv_mill',
    'S004378422': 'plate_inv_mill',

    'S005580641': 'rebar_inv_all',
    'S005580642': 'wirerod_inv_all',
    'S005580640': 'hrc_inv_all',
    'S005580646': 'crc_inv_all',
    'S005580643': 'plate_inv_all',

    'S004802760': 'rebar_prod_all',
    'S004802761': 'wirerod_prod_all',
    'S005580652': 'crc_prod_all',
    'S005107854': 'hrc_prod_all',

    'S004039553': 'billet_inv_social_ts',

    # base
    'S005808359': 'cu_lme_3m_close',
    'S005808360': 'al_lme_3m_close',
    'S005808361': 'pb_lme_3m_close',
    'S005808362': 'zn_lme_3m_close',
    'S005808363': 'sn_lme_3m_close',
    'S005808364': 'ni_lme_3m_close',

    'S004303031': 'cu_lme_0m_3m_spd',
    'S004303035': 'al_lme_0m_3m_spd',
    'S004303034': 'zn_lme_0m_3m_spd',
    'S004303033': 'pb_lme_0m_3m_spd',
    'S004303032': 'sn_lme_0m_3m_spd',
    'S004303036': 'ni_lme_0m_3m_spd',
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
    #'S003018876': 'aa_lme_0m_3m_spd',
    #'S003018877': 'aa_lme_3m_15m_spd',
    #'S003018878': 'aa_lme_3m_27m_spd',
    "S002855118": "cu_inv_cme_total",
    "S004303280": "cu_inv_lme_total",
    "S002836856": "cu_inv_lme_cancelled",
    "S004303370": "al_inv_lme_total",
    "S004303408": "al_inv_lme_cancelled",
    "S004303465": "pb_inv_lme_total",
    "S004303491": "pb_inv_lme_cancelled",
    "S004303530": "zn_inv_lme_total",
    "S004303550": "zn_inv_lme_cancelled",
    "S004303580": "sn_inv_lme_total",
    "S004303592": "sn_inv_lme_cancelled",
    "S004303610": "ni_inv_lme_total",
    "S004303642": "ni_inv_lme_cancelled",
    "S003164358": "cu_inv_shfe_d",
    "S003164360": "zn_inv_shfe_d",
    "S003164359": "al_inv_shfe_d",
    "S003164361": "pb_inv_shfe_d",
    "S004322735": "ni_inv_shfe_d",
    "S004322736": "sn_inv_shfe_d",
    "S019848684": "ao_inv_shfe_d",
    "S019848689": "ao_inv_shfe_mill_d",
    "S009223764": "ss_inv_shfe_d",
    "S019735959": "si_inv_gfex_d",
    "S020098434": "lc_inv_gfex_d",
    "S022117012": "SH_inv_czce_warrant",
    "S022319791": "SH_inv_czce_unwarrant",
    "S003008076": "TA_inv_czce_warrant",
    "S005451492": "TA_inv_czce_unwarrant",
    "S004302740": "MA_inv_czce_warrant",
    "S005451467": "MA_inv_czce_unwarrant",
    "S005451340": "UR_inv_czce_warrant",
    "S005451410": "UR_inv_czce_unwarrant",
    "S005658949": "PF_inv_czce_warrant",
    "S003787910": "l_inv_dce_warrant",
    "S003787913": "pp_inv_dce_warrant",
    "S003787915": "v_inv_dce_warrant",
    "S005450245": "eg_inv_dce_warrant",
    "S005450249": "eb_inv_dce_warrant",
    "S004302762": "bu_inv_shfe_warrant",
    "S004302780": "bu_inv_shfe_mill",
    "S003154875": "FG_inv_czce_warrant",
    "S005451360": "SA_inv_czce_warrant",
    "S005451430": "SA_inv_czce_unwarrant",
    "S022117035": "PX_inv_czce_warrant",
    "S022319792": "PX_inv_czce_unwarrant",
    "S005476601": "sc_inv_ine_warrant",
    "S005476602": "fu_inv_shfe_warrant",
    "S006404843": "lu_inv_ine_warrant",
    "S006404844": "pg_inv_dce_warrant",
    "S005476603": "j_inv_dce_warrant",
    "S005476604": "jm_inv_dce_warrant",
    "S005476308": "rb_inv_shfe_warrant",
    "S005476309": "hc_inv_shfe_warrant",
    "S005476310": "i_inv_dce_warrant",
    "S005476311": "SF_inv_czce_warrant",
    "S005476313": "SF_inv_czce_unwarrant",
    "S005476312": "SM_inv_czce_warrant",
    "S005476314": "SM_inv_czce_unwarrant",
    "S003277851": "m_inv_dce_warrant",
    "S003278148": "RM_inv_czce_warrant",
    "S003278185": "RM_inv_czce_unwarrant",
    "S000001487": "c_inv_dce_warrant",
    "S000001485": "a_inv_dce_warrant",
    "S003277847": "b_inv_dce_warrant",
    "S005532615": "jd_inv_dce_warrant",
    "S000001484": "y_inv_dce_warrant",
    "S003155008": "p_inv_dce_warrant",
    "S000001491": "OI_inv_czce_warrant",
    "S000001493": "OI_inv_czce_unwarrant",
    "S000001490": "CF_inv_czce_warrant",
    "S003278182": "CF_inv_czce_unwarrant",
    "S000001488": "SR_inv_czce_warrant",
    "S000001496": "SR_inv_czce_unwarrant",
    "S005532620": "AP_inv_czce_warrant",
    "S005532621": "AP_inv_czce_unwarrant",
    "S005532619": "CJ_inv_czce_warrant",
    "S005532625": "CJ_inv_czce_unwarrant",
    "S009637664": "PK_inv_czce_warrant",
    "S003164362": "au_inv_shfe_warrant",
    "S003164363": "ag_inv_shfe_warrant",
    "S005476287": "sp_inv_shfe_warrant",
    "S006700187": "lh_inv_dce_warrant",
    "S005532617": "CY_inv_czce_warrant",
    "S005532614": "cs_inv_dce_warrant",
    "S006409299": "bc_inv_ine_warrant",
    "S004410360": "ru_inv_shfe_warrant",
    "S005450012": "nr_inv_shfe_warrant",

    'S004630824': 'cu_mine_tc',
    #'S011211693': 'cu_25conc_tc',
    'S005951203': 'pb_60conc_tc_ports',
    'S006158372': 'pb_50conc_tc_hunan',
    'S006158375': 'pb_50conc_tc_yunnan',
    'S006158378': 'pb_50conc_tc_guangxi',
    'S006158381': 'pb_50conc_tc_neimeng',
    'S006158384': 'pb_50conc_tc_henan',
    'S009620177': 'sn_60conc_tc_jiangxi',
    'S009620198': 'sn_60conc_tc_guangxi',
    'S009620213': 'sn_40conc_tc_yunnan',
    'S016702541': 'zn_50conc_tc_neimeng',
    'S016702544': 'zn_50conc_tc_yunnan',
    'S016702547': 'zn_50conc_tc_hunan',
    'S016702550': 'zn_50conc_tc_guangxi',
    'S016702553': 'zn_50conc_tc_henan',
    'S016702556': 'zn_50conc_tc_sichuan',
    'S016702559': 'zn_50conc_tc_shanaxi',
    'S016702562': 'zn_48conc_tc_ports',

    'S003797045': 'ni_1.8conc_spot_php_lianyungang',
    "S005068187": "ni_1.5conc_spot_rz",
    'S005102262': 'sn_60conc_spot_guangxi',
    'S009137295': 'ni_nis_cjb_spot',
    'S009200268': 'ni_nis_spot_gi',
    'S009273405': 'ni_nis_spot_battery',
    'S009620789': 'ni_npi_10-15_sh',
    'S020207789': 'ni_mhp_34_ports',

    "S006157941": "cu_prem_bonded_warrant",
    "S006157947": "cu_prem_bonded_cny",
    "S006157944": "cu_prem_bonded_cif",
    "S005068109": "al_prem_bonded_warrant",
    "S005068106": "al_prem_bonded_cif",
    "S005068427": 'zn_prem_smm_import',
    "S005068439": "zn_prem_bonded_warrant",
    "S005068436": "zn_prem_bonded_cif",
    "S005068184": 'ni_prem_bonded_warrant',
    "S005068181": 'ni_prem_bonded_cif',
    "S009200256": "ni_prem_import",
    "S005068220": 'pb_prem_bonded_warrant',
    "S005068217": 'pb_prem_bonded_cif',
    "S005068331": 'cu_prem_bonded_warrant_er',
    "S005068328": 'cu_prem_bonded_warrant_sx',
    "S005068337": 'cu_prem_bonded_cif_er',
    "S005068334": 'cu_prem_bonded_cif_sx',

    "S000025471": "cu_cjb_spot",
    "S000025473": "al_cjb_spot",
    "S000025475": "pb_cjb_spot",
    "S000025476": "zn_cjb_spot",
    "S000025478": "sn_cjb_spot",
    "S000025479": "ni_cjb_spot",

    "S002981535": "cu_smm1_spot",
    "S002981536": "cu_smm1_prem_spot",
    "S002981537": "cu_smm1_shifa_spot",
    "S002981538": "cu_smm1_guixi_spot",
    'S004077505': 'cu_spot_sh',
    'S004077504': 'cu_spot_bj',
    "S002865592": "al_smm0_spot",
    "S002865578": "zn_smm0_spot",
    "S002865583": "pb_smm1_spot",
    "S002981539": "sn_smm1_spot",
    "S002981540": "ni_smm1_spot",
    "S002981541": "ni_smm1_jc_spot",
    "S002981542": "ni_smm1_imp_spot",
    "S010361921": "ni_cj1_spot",
    "S004785205": "ss_304_gross_wuxi",
    "S004785215": "ss_304_wuxi_phybasis",

    'S003048722': 'cu_smm_phybasis',
    'S003048723': 'cu_flat_phybasis',
    'S003048724': 'cu_prem_phybasis',
    'S003048725': 'cu_shifa_phybasis',
    'S003048726': 'cu_guixi_phybasis',
    "S009137283": "cu_cj_phybasis",
    "S009137286": "cu_cjb_phybasis",
    "S009621955": "cu_sh_phybasis",

    "S003048727": "al_smm0_phybasis",
    "S005068103": "al_a00_phybasis_shmet",
    "S004031017": "al_sh_phybasis",
    "S009137289": "al_cj_phybasis",
    "S009137292": "al_cjb_phybasis",
    "S008871816": "al_nanchu_phybasis",
    "S008527822": "al_wm0_phybasis_low",
    "S008527823": "al_wm0_phybasis_high",

    "S005068421": 'zn_smm0_sh_phybasis',
    "S005068424": 'zn_smm1_sh_phybasis',
    "S008871823": "zn_nanchu_phybasis", # fut0
    "S008527843": 'zn_wm0_phybasis',
    "S008527848": 'zn_wm1_phybasis',

    "S005068193": "ni_smm1_phybasis",
    "S005068175": "ni_smm1_jc_phybasis",
    "S005068169": "ni_smm1_ru_phybasis",
    "S005068211": "pb_smm1_sh_phybasis",
    "S005068409": "sn_smm1_sh_phybasis",

    "S009160074": "cu_scrap_1_spot_jzh",
    "S009160107": "cu_scrap_2_spot_jzh",
    "S008545965": "cu_scrap_1_sh",
    "S008545990": "cu_scrap_2_sh",
    "S009626046": "al_scrap_shreded_spot_foshan",
    "S009626791": "ni_scrap_spot_foshan",
    "S015202398": "cu_scrap1_diff_gd", # short history
    "S015202399": "cu_scrap1_diff_tj",
    "S004243370": "zn_scrap_sh_high",
    "S004243369": "zn_scrap_sh_low",
    "S004243249": "al_scrap_shredded_sh_low",
    "S004243250": "al_scrap_shredded_sh_high",
    "S009780583": "pb_scrap_autostarter_sh",
    "S009626378": "pb_scrap_ebike_sh",
    "S009626602": "sn_scrap_pure_bulk_shandong",
    "S009626605": "sn_scrap_bulk_shandong",
    "S009626611": "sn_scrap_slag_shandong",
    "S002959172": "ss_304_scrap_wuxi",

    "S008871802": "cu_rod_8_procfee_nanchu", # short history
    "S008871805": "cu_rod_2.6_procfee_nanchu",
    "S009621341": "al_rod_6063_procfee_jiangxi", # no good
    "S009621410": "al_rod_6063_procfee_sichuan",
    "S009621539": "al_rod_6063_procfee_gansu",

    'S005971281': 'cu_mine_inv_ports',
    "S006161499": "cu_inv_social_all",
    "S005363047": "al_inv_social_all",
    "S006161627": "zn_inv_social_3p",
    "S006161628": "zn_inv_social_7p",
    "S006161636": "ni_inv_social_6p",
    'S013735402': "sn_inv_social_all",
    'S006161617': 'pb_inv_social_5p',
    'S006167225': 'bauxite_inv_az_ports',
    'S006167236': 'alumina_inv_az_ports',
    'S004425326': 'alumina_inv_ports',
    'S011258021': 'bauxite_inv_ports_inv',

    'S012937595': 'si_inv_social_all',
    'S006563225': 'al_6063rod_inv_social',
    "S006161096": "ss_inv_social_all",
    "S006161093": "ss_inv_social_200",
    "S006161094": "ss_inv_social_300",
    "S006161095": "ss_inv_social_400",

    "S000025546": "au_td_sge",
    "S003057206": "ag_td_sge",
    "S002808967": "container_exp_scfi",
    "S008527041": "alumina_spot_shanxi",
    "S008527032": "alumina_spot_guangxi",
    "S008527044": "alumina_spot_henan",
    "S008527035": "alumina_spot_guizhou",
    "S004077728": "alumina_spot_qd",
    "S010596299": "alumina_spot_cnports",
    "S010596302": "alumina_aus_fob",
    "S002865625": 'si_553_spot_smm',

    "M002845714": "csi300_idx",
    "M002845725": "csi500_idx",
    "M012963695": "csi1000_idx",
    "M009042848": "sw_sector_idx_basemetal",
    "M009042858": "sw_sector_idx_prop",
    "M009042864": "sw_sector_idx_const",
    "M009042847": 'sw_sector_idx_steel',
    "M009042873": "sw_sector_idx_petchem",
    "M009042872": "sw_sector_idx_coal",
    "M003802454": "sw_sector2_idx_glass",
    "M003802458": "sw_sector2_idx_infra",
    #"M003802411": "sw_sector2_idx_weaving",
    "M003802386": "sw_sector2_idx_rubber",
    #"M003802385": "sw_sector2_idx_plastics",

    "M003588167": "zx_sector_idx_const",
    "M003588182": "zx_sector_idx_prop",
    "M003588162": "zx_sector_idx_basemetal",
    "M003588164": "zx_sector_idx_steel",
    "M003588160": "zx_sector_idx_oil_petchem",
    "M003588161": "zx_sector_idx_coal",
}


def data_wkday_adj(data_df, col_list, shift_map={0: -4, 1: -5, 2: 1, 3: 0, 4: -1, 5: -2, 6: -3}):
    col_list = [col for col in col_list if col in data_df.columns]
    data_df = data_df.reindex(
        index=pd.date_range(
            start=data_df.index[0],
            end=data_df.index[-1] + pd.DateOffset(days=max(shift_map.values())),
            freq='D'))
    ddf = data_df[col_list].dropna(how='all').copy(deep=True)
    ddf['date'] = ddf.index
    ddf['date'] = ddf['date'].map(lambda d: d + pd.DateOffset(days=shift_map.get(d.weekday(), 0)))
    ddf = ddf.drop_duplicates(subset=['date'], keep='first')
    ddf = ddf.set_index('date')
    for col in col_list:
        data_df[col] = ddf[col]
    return data_df


def adj_publish_time(spot_df):
    col_list = [
        'io_inv_imp_mill(64)', 'io_inv_dom_mill(64)', 'io_invdays_imp_mill(64)',
    ]
    shift_map = {0: -4, 1: -5, 2: 1, 3: 0, 4: -1, 5: -2, 6: -3}
    spot_df = data_wkday_adj(spot_df, col_list, shift_map=shift_map)

    col_list = [
        'io_removal_41ports', 'io_inv_31ports', 'io_inv_45ports',
    ]
    shift_map = {0: -3, 1: -4, 2: -5, 3: 1, 4: 0, 5: -1, 6: -2}
    spot_df = data_wkday_adj(spot_df, col_list, shift_map=shift_map)

    col_list = [
        'rebar_inv_mill', 'wirerod_inv_mill', 'hrc_inv_mill', 'crc_inv_mill', 'plate_inv_mill',
        'rebar_inv_social', 'wirerod_inv_social', 'hrc_inv_social', 'crc_inv_social', 'plate_inv_social',
        'steel_inv_social', 'rebar_inv_all', 'rebar_prod_all', 'wirerod_prod_all', 'wirerod_inv_all',
        'hrc_prod_all', 'hrc_inv_all', 'crc_prod_all', 'crc_inv_all',
    ]
    shift_map = {0: -4, 1: -5, 2: -6, 3: 0, 4: -1, 5: -2, 6: -3}
    spot_df = data_wkday_adj(spot_df, col_list, shift_map=shift_map)
    return spot_df


def process_spot_df(spot_df, adjust_time=False):
    if adjust_time:
        spot_df = adj_publish_time(spot_df)
    spot_dict = {}
    for asset in ['cu', 'al', 'zn', 'pb', 'ni', 'sn']:
        spot_dict[f'{asset}_inv_exch_d'] = spot_df[
            [f'{asset}_inv_shfe_d', f'{asset}_inv_lme_total']].sum(axis=1, skipna=False)
        spot_dict[f'{asset}_lme_futbasis'] = np.log(1 + spot_df[f'{asset}_lme_0m_3m_spd'] /
                                                  spot_df[f'{asset}_lme_3m_close'])

    spot_dict['usgg10_be'] = spot_df['usgg10yr'] - spot_df['usggt10yr']
    spot_dict['usgg10_2_spd'] = spot_df['usgg10yr'] - spot_df['usgg2yr']
    spot_dict['cgb_3m_1y_spd'] = spot_df['cn_govbond_yield_3m_sch'] - spot_df['cn_govbond_yield_1y_sch']
    spot_dict['cgb_2_5_spd'] = spot_df['cn_govbond_yield_2y'] - spot_df['cn_govbond_yield_5y']
    spot_dict['cgb_1_2_spd'] = spot_df['cn_govbond_yield_1y'] - spot_df['cn_govbond_yield_2y']
    spot_dict['cgb_1_5_spd'] = spot_df['cn_govbond_yield_1y'] - spot_df['cn_govbond_yield_5y']
    spot_dict['cgb_2_10_spd'] = spot_df['cn_govbond_yield_2y'] - spot_df['cn_govbond_yield_10y']
    spot_dict['fxbasket_cumret'] = spot_df[['usdzar_xe', 'usdaud_xe', 'usdclp_xe', 'usdbrl_xe']].dropna().pct_change().mean(axis=1).cumsum()
    spot_dict['pmi_steel_order2inv'] = (spot_df['pmi_cn_steel_new_order']/spot_df['pmi_cn_steel_inv']).dropna()
    spot_dict['pmi_steel_order2rminv'] = (spot_df['pmi_cn_steel_new_order']/spot_df['pmi_cn_steel_rm_inv']).dropna()
    spot_dict['cnh_cny_spd1'] = spot_df['usdcnh_xe'] - spot_df['usdcny_xe']
    spot_dict['cnh_cny_spd2'] = spot_df['usdcnh_close'] - spot_df['usdcny_spot2']
    spot_dict['cny_mid_dev1'] = spot_df['usdcny_spot'] - spot_df['usdcny_mid']
    spot_dict['cny_mid_dev2'] = spot_df['usdcny_spot2'] - spot_df['usdcny_mid']
    spot_dict['r_dr_7d_spd'] = spot_df['r007_cn'] - spot_df['dr007_cn']
    spot_dict['shibor_3m_1y_spd'] = spot_df['shibor_3m'] - spot_df['shibor_1y']

    spot_dict['cs_corp_aa_1y'] = spot_df['corpbond_yield_aa_1y'] -spot_df['cn_govbond_yield_1y_sch']
    spot_dict['cs_aa_1y'] = spot_df['corpbond_yield_aa_1y'] -spot_df['cn_govbond_yield_1y_sch']
    spot_dict['cs_schbill_aa_1y'] = spot_df['schbill_yield_aa_1y'] -spot_df['cn_govbond_yield_1y_sch']
    spot_dict['cs_stbill_aa_1y'] = spot_df['stbill_yield_aa_1y'] -spot_df['cn_govbond_yield_1y_sch']
    spot_dict['cs_bknote_aa_1y'] = spot_df['banknotes_yield_aa_1y'] -spot_df['cn_govbond_yield_1y_sch']

    spot_dict['steel_inv_mill'] = spot_df['rebar_inv_mill'] + spot_df['wirerod_inv_mill'] + \
                                spot_df['hrc_inv_mill'] + spot_df['crc_inv_mill'] #+ spot_df['plate_inv_mill']
    spot_dict['steel_inv_all'] = spot_df['steel_inv_social'] + spot_dict['steel_inv_mill']
    spot_dict['steel_social_inv'] = spot_df['rebar_inv_social'] + spot_df['wirerod_inv_social'] + \
                                  spot_df['hrc_inv_social'] + spot_df['crc_inv_social'] #+ spot_df['plate_inv_social']
    spot_dict['long_social_inv'] = spot_df['rebar_inv_social'] + spot_df['wirerod_inv_social']
    spot_dict['flat_social_inv'] = spot_df['hrc_inv_social'] + spot_df['crc_inv_social'] #+ spot_df['plate_inv_social']
    spot_dict['rebar_app_dmd'] = spot_df['rebar_prod_all'] - spot_df['rebar_inv_all'].dropna().diff()
    spot_dict['wirerod_app_dmd'] = spot_df['wirerod_prod_all'] - spot_df['wirerod_inv_all'].dropna().diff()
    spot_dict['hrc_app_dmd'] = spot_df['hrc_prod_all'] - spot_df['hrc_inv_all'].dropna().diff()
    spot_dict['crc_app_dmd'] = spot_df['crc_prod_all'] - spot_df['crc_inv_all'].dropna().diff()
    spot_dict['rb_hc_dmd_diff'] = spot_dict['rebar_app_dmd'] - spot_dict['hrc_app_dmd']
    spot_dict['rb_hc_sinv_diff'] = spot_df['rebar_inv_social'].dropna().diff() - spot_df['hrc_inv_social'].dropna().diff()
    spot_dict['rebar_sales_inv_ratio'] = spot_df['consteel_dsales_mysteel']/spot_df['rebar_inv_social'].ffill()

    spot_dict['crc_hrc'] = spot_df['crc_sh'] - spot_df['hrc_sh']
    spot_dict['pipe_strip'] = spot_df['pipe_1.5x3.25'] - spot_df['strip_3.0x685']
    spot_dict['hrc_billet'] = spot_df['hrc_sh'] - spot_df['billet_ts']
    spot_dict['rebar_billet'] = spot_df['rebar_sh'] - spot_df['billet_ts']
    spot_dict['plate_billet'] = spot_df['plate_8mm'] - spot_df['billet_ts']
    spot_dict['crc_billet'] = spot_df['crc_sh'] - spot_df['billet_ts']
    spot_dict['gi_billet'] = spot_df['gi_0.5_sh'] - spot_df['billet_ts']
    spot_dict['strip_billet'] = spot_df['strip_3.0x685'] - spot_df['billet_ts']
    spot_dict['pipe_billet'] = spot_df['pipe_1.5x3.25'] - spot_df['billet_ts']
    spot_dict['hsec_billet'] = spot_df['hsec_400x200'] - spot_df['billet_ts']
    spot_dict['channel_billet'] = spot_df['channel_16'] - spot_df['billet_ts']
    spot_dict['ibeam_billet'] = spot_df['ibeam_25'] - spot_df['billet_ts']
    spot_dict['angle_billet'] = spot_df['angle_50x5'] - spot_df['billet_ts']
    spot_dict['highwire_billet'] = spot_df['highwire_6.5'] - spot_df['billet_ts']

    spot_dict["zn_scrap_sh_mid"] = (spot_df["zn_scrap_sh_low"] + spot_df["zn_scrap_sh_high"])/2
    spot_dict["al_scrap_shredded_sh_mid"] = (spot_df["al_scrap_shredded_sh_low"] + spot_df["al_scrap_shredded_sh_high"])/2
    spot_dict['rebar_total_stockdays'] = (spot_df['rebar_inv_social']).dropna()/spot_dict['rebar_app_dmd'].dropna().rolling(52).mean()*7
    spot_dict['hrc_total_stockdays'] = (spot_df['hrc_inv_social']).dropna()/spot_dict['hrc_app_dmd'].dropna().rolling(52).mean()*7
    spot_dict['io_inv_removal_ratio_41p'] = spot_df['io_inv_41ports'] / spot_df['io_removal_41ports']
    spot_dict['io_inv_rmv_pctchg_41p'] = spot_dict['io_inv_removal_ratio_41p'].dropna().pct_change()
    spot_dict['io_inv_mill(64)'] = spot_df['io_inv_imp_mill(64)'] + spot_df['io_inv_dom_mill(64)']
    spot_dict['io_on_off_arb'] = vat_adj(spot_df['pbf_cfd'] - 25) / spot_df['usdcnh_spot'] / 0.915 / 61.5 * 62 \
                               - spot_df['plt62']
    spot_dict['margin_hrc_pbf'] = spot_df['hrc_sh'] - 1.7 * spot_df['pbf_cfd'] - 0.45 * spot_df['coke_xuzhou_xb']
    spot_dict['margin_hrc_macf'] = spot_df['hrc_sh'] - 1.7 * spot_df['macf_cfd'] - 0.45 * spot_df['coke_xuzhou_xb']
    spot_dict['strip_hsec'] = spot_df['strip_3.0x685'] - spot_df['hsec_400x200']
    if ('coal_5500_sx_qhd' in spot_df.columns) and ('coal_5500_qhd' in spot_df.columns):
        spot_df.loc[:'2022-02-11', 'coal_5500_sx_qhd'] = spot_df.loc[:'2022-02-11', 'coal_5500_qhd']
    warrant_dict = {
        "cu": ["cu_inv_shfe_d"],
        "bc": ["bc_inv_ine_warrant"],
        "zn": ["zn_inv_shfe_d"],
        "al": ["al_inv_shfe_d"],
        "pb": ["pb_inv_shfe_d"],
        "ni": ["ni_inv_shfe_d"],
        "sn": ["sn_inv_shfe_d"],
        "ao": ["ao_inv_shfe_d", "ao_inv_shfe_mill_d"],
        "ss": ["ss_inv_shfe_d"],
        "si": ["si_inv_gfex_d"],
        "lc": ["lc_inv_gfex_d"],
        "SH": ["SH_inv_czce_warrant", "SH_inv_czce_unwarrant"],
        "TA": ["TA_inv_czce_warrant", "TA_inv_czce_unwarrant"],
        "PX": ["PX_inv_czce_warrant", "PX_inv_czce_unwarrant"],
        "MA": ["MA_inv_czce_warrant", "MA_inv_czce_unwarrant"],
        "UR": ["UR_inv_czce_warrant", "UR_inv_czce_unwarrant"],
        "PF": ["PF_inv_czce_warrant"],
        "l": ["l_inv_dce_warrant"],
        "pp": ["pp_inv_dce_warrant"],
        "v": ["v_inv_dce_warrant"],
        "eg": ["eg_inv_dce_warrant"],
        "eb": ["eb_inv_dce_warrant"],
        "pg": ["pg_inv_dce_warrant"],
        "bu": ["bu_inv_shfe_warrant", "bu_inv_shfe_mill"],
        "SA": ["SA_inv_czce_warrant", "SA_inv_czce_unwarrant"],
        "FG": ["FG_inv_czce_warrant"],
        "j": ["j_inv_dce_warrant"],
        "jm": ["jm_inv_dce_warrant"],
        "rb": ["rb_inv_shfe_warrant"],
        "hc": ["hc_inv_shfe_warrant"],
        "i": ["i_inv_dce_warrant"],
        "SF": ["SF_inv_czce_warrant", "SF_inv_czce_unwarrant"],
        "SM": ["SM_inv_czce_warrant", "SM_inv_czce_unwarrant"],
        "m": ["m_inv_dce_warrant"],
        "RM": ["RM_inv_czce_warrant", "RM_inv_czce_unwarrant"],
        "c": ["c_inv_dce_warrant"],
        "cs": ["cs_inv_dce_warrant"],
        "a": ["a_inv_dce_warrant"],
        "b": ["b_inv_dce_warrant"],
        "jd": ["jd_inv_dce_warrant"],
        "lh": ["lh_inv_dce_warrant"],
        "y": ["y_inv_dce_warrant"],
        "p": ["p_inv_dce_warrant"],
        "OI": ["OI_inv_czce_warrant", "OI_inv_czce_unwarrant"],
        "CF": ["CF_inv_czce_warrant", "CF_inv_czce_unwarrant"],
        "CY": ["CY_inv_czce_warrant"],
        "SR": ["SR_inv_czce_warrant", "SR_inv_czce_unwarrant"],
        "AP": ["AP_inv_czce_warrant", "AP_inv_czce_unwarrant"],
        "CJ": ["CJ_inv_czce_warrant", "CJ_inv_czce_unwarrant"],
        "PK": ["PK_inv_czce_warrant"],
        "sp": ["sp_inv_shfe_warrant"],
        "ru": ["ru_inv_shfe_warrant"],
        "nr": ["nr_inv_shfe_warrant"],
        "fu": ["fu_inv_shfe_warrant"],
        "lu": ["lu_inv_ine_warrant"],
        "sc": ["sc_inv_ine_warrant"],
    }
    for asset in warrant_dict:
        spot_dict[f"{asset}_exch_warrant"] = spot_df[warrant_dict[asset]].dropna().sum(axis=1).dropna()

    asset_pairs = [
        ("sw_sector_idx_prop", 'csi500_idx', 'prop_sw_csi500'),
        ("sw_sector_idx_const", 'csi500_idx', 'const_sw_csi500'),
        ("sw_sector_idx_steel", 'csi500_idx', 'steel_sw_csi500'),
        ("sw_sector_idx_basemetal", 'csi500_idx', 'base_sw_csi500'),
        ("zx_sector_idx_prop", 'csi500_idx', 'prop_zx_csi500'),
        ("zx_sector_idx_const", 'csi500_idx', 'const_zx_csi500'),
        ("zx_sector_idx_steel", 'csi500_idx', 'steel_zx_csi500'),
        ("zx_sector_idx_basemetal", 'csi500_idx', 'base_zx_csi500'),
        ("sw_sector_idx_petchem", 'csi500_idx', 'petchem_sw_csi500'),
        ("sw_sector_idx_coal", 'csi500_idx', 'coal_sw_csi500'),
        ("sw_sector2_idx_glass", 'csi500_idx', 'glass_sw_csi500'),
        ("sw_sector2_idx_infra", 'csi500_idx', 'infra_sw_csi500'),        
        ("sw_sector2_idx_rubber", 'csi500_idx', 'rubber_sw_csi500'),        
    ]
    beta_win = 245
    for trade_asset, index_asset, key in asset_pairs:
        asset_df = spot_df[[index_asset, trade_asset]].dropna().copy(deep=True)
        for asset in asset_df:
            asset_df[f"{asset}_pct"] = asset_df[asset].pct_change().rolling(5).mean()
        asset_df['beta'] = asset_df[f"{index_asset}_pct"].rolling(beta_win).cov(
            asset_df[f"{trade_asset}_pct"]) / asset_df[f"{index_asset}_pct"].rolling(beta_win).var()
        asset_df['ret'] = asset_df[trade_asset].pct_change() - asset_df['beta'] * asset_df[
            index_asset].pct_change().fillna(0)
        spot_dict[key + "_ret"] = asset_df['ret'].dropna()
        spot_dict[key + '_beta'] = asset_df['beta']
        spot_dict[key + '_val'] = asset_df['ret'].dropna().cumsum()
    spot_df = pd.concat([spot_df, pd.DataFrame(spot_dict)], axis=1)
    return spot_df
