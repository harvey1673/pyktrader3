index_map = {
    # macro
    'G002600770': 'usgg2yr',
    'G002600774': 'usgg10yr',
    'G002600783': 'usggt10yr',
    'G013233151': 'usggbe5',
    'G005172253': 'usgg10yr_2yr_spd',

    'G002600791': 'libor3m',
    'G002600885': 'dxy',
    #'G002601505': 'vix',
    'G003082203': 'vix',
    'G003082207': 'vvix',
    'G003082211': 'vxeem',
    'G003082215': 'vxd', # Vol for DJ
    'G003082227': 'vxn', # Vol for Nasdaq
    'G002945506': 'ted_spd',

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
    # ferrous
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
    'S000020904': 'crc_shenyang',
    'S000020901': 'crc_bj',
    'S000020933': 'gi_0.5_sh',
    'S000020939': 'gi_1.0_sh',
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
    'S008061234': 'coke_xuzhou_xb',
    'S008061203': 'coke_ts_xb',
    'S008061232': 'coke_sh_xb',
    'S008061213': 'coke_changzhi_xb',

    'S002917675': 'strip_2.5x355',
    'S002917688': 'strip_3.0x685',
    'S002917771': 'pipe_1.5x3.25',
    'S002917787': 'pipe_4x3.75',
    'S002917733': 'seemless_pipe_108x4.5',

    'S002917631': 'hsec_300x300',
    'S002917646': 'hsec_400x200',
    'S002917616': 'hsec_200x100',
    'S002917514': 'channel_16',
    'S002917532': 'ibeam_25',
    'S002917550': 'angle_50x5',
    'S002917279': 'highwire_6.5',
    'S002959491': 'femn65si17_neimeng',
    'S002959498': 'femn65si17_tj',
    'S002959495': 'femn65si17_guangxi',
    'S004378612': 'coal_5500_jingtang',
    'S002882871': 'coal_5500_sx_qhd',
    'S002837009': 'coal_5500_qhd',

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
    'S008618299': 'consteel_daily_sales',

    'S008618440': 'SM_inv_mill',
    'S008618447': 'SF_inv_mill',

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
    'S004077505': 'cu_spot_sh',
    'S004077504': 'cu_spot_bj',
    'S003048722': 'cu_dianjie_spot_basis',
    'S003048723': 'cu_pingshui_spot_basis',
    'S003048724': 'cu_shengshui_spot_basis',
    'S003048725': 'cu_shifa_spot_basis',
    'S003048726': 'cu_guixi_spot_basis',

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
    'S003018876': 'aa_lme_0m_3m_spd',
    'S003018877': 'aa_lme_3m_15m_spd',
    'S003018878': 'aa_lme_3m_27m_spd',

    'S004630824': 'cu_mine_tc',
    "S006157941": "cu_prem_yangshan_warrant",
    "S006157947": "cu_prem_yangshan_cny",
    "S006157944": "cu_prem_yangshan_bl",
    "S005068109": "al_prem_bonded_warrant",
    "S005068106": "al_prem_bonded_bl",
    "S005068436": "zn_prem_bonded_bl",
    "S005068439": "zn_prem_bonded_warrant",

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
    "S002865592": "al_smm0_spot",
    "S002865578": "zn_smm0_spot",
    "S002865583": "pb_smm1_spot",
    "S002981539": "sn_smm1_spot",
    "S002981540": "ni_smm1_spot",
    "S002981541": "ni_smm1_jc_spot",
    "S002981542": "ni_smm1_imp_spot",
    "S010361921": "ni_cj1_spot",

    "S009137283": "cu_cj_phybasis",
    "S009137286": "cu_cjb_phybasis",
    "S009137289": "al_cj_phybasis",
    "S009137292": "al_cjb_phybasis",
    "S008871816": "al_nanchu_phybasis",
    "S005068421": 'zn_smm_phybasis',
    "S005068427": 'zn_smm_imp_phybasis', #
    "S008871823": "zn_nanchu_phybasis", # fut0
    "S005068193": "ni_smm1_phybasis",
    "S005068175": "ni_smm1_jc_phybasis",
    "S005068169": "ni_smm1_ru_phybasis",

    "S009160074": "cuscrap_1_spot_jzh",
    "S009160107": "cuscrap_2_spot_jzh",

    "S015202398": "cu_scrap1_diff_gd", # short history
    "S015202399": "cu_scrap1_diff_tj",

    "S008871802": "cu_rod_8_procfee_nanchu", # short history
    "S008871805": "cu_rod_2.6_procfee_nanchu",
    "S009621341": "al_rod_6063_procfee_jiangxi", # no good
    "S009621410": "al_rod_6063_procfee_sichuan",
    "S009621539": "al_rod_6063_procfee_gansu",
}


def process_spot_df(spot_df):
    spot_df['usggbe10'] = spot_df['usgg10yr'] - spot_df['usggt10yr']
    spot_df['r_dr_7d_spd'] = spot_df['r007_cn'] - spot_df['dr007_cn']

    #spot_df['usgg10yr_2yr_spd'] = spot_df['usgg10yr'] - spot_df['usgg2yr']
    spot_df['steel_inv_mill'] = spot_df['rebar_inv_mill'] + spot_df['wirerod_inv_mill'] + \
                                spot_df['hrc_inv_mill'] + spot_df['crc_inv_mill'] #+ spot_df['plate_inv_mill']
    spot_df['steel_inv_all'] = spot_df['steel_inv_social'] + spot_df['steel_inv_mill']
    spot_df['steel_social_inv'] = spot_df['rebar_inv_social'] + spot_df['wirerod_inv_social'] + \
                                  spot_df['hrc_inv_social'] + spot_df['crc_inv_social'] #+ spot_df['plate_inv_social']
    spot_df['long_social_inv'] = spot_df['rebar_inv_social'] + spot_df['wirerod_inv_social']
    spot_df['flat_social_inv'] = spot_df['hrc_inv_social'] + spot_df['crc_inv_social'] #+ spot_df['plate_inv_social']
    spot_df['rebar_app_dmd'] = spot_df['rebar_prod_all'] - spot_df['rebar_inv_all'].dropna().diff()
    spot_df['wirerod_app_dmd'] = spot_df['wirerod_prod_all'] - spot_df['wirerod_inv_all'].dropna().diff()
    spot_df['hrc_app_dmd'] = spot_df['hrc_prod_all'] - spot_df['hrc_inv_all'].dropna().diff()
    spot_df['crc_app_dmd'] = spot_df['crc_prod_all'] - spot_df['crc_inv_all'].dropna().diff()
    spot_df['rb_hc_dmd_diff'] = spot_df['rebar_app_dmd'] - spot_df['hrc_app_dmd']
    spot_df['rb_hc_sinv_diff'] = spot_df['rebar_inv_social'].dropna().diff() - spot_df['hrc_inv_social'].dropna().diff()

    spot_df['crc_hrc'] = spot_df['crc_sh'] - spot_df['hrc_sh']
    spot_df['pipe_strip'] = spot_df['pipe_1.5x3.25'] - spot_df['strip_3.0x685']
    spot_df['hrc_billet'] = spot_df['hrc_sh'] - spot_df['billet_ts']
    spot_df['rebar_billet'] = spot_df['rebar_sh'] - spot_df['billet_ts']
    spot_df['plate_billet'] = spot_df['plate_8mm'] - spot_df['billet_ts']
    spot_df['crc_billet'] = spot_df['crc_sh'] - spot_df['billet_ts']
    spot_df['gi_billet'] = spot_df['gi_0.5_sh'] - spot_df['billet_ts']
    spot_df['strip_billet'] = spot_df['strip_3.0x685'] - spot_df['billet_ts']
    spot_df['pipe_billet'] = spot_df['pipe_1.5x3.25'] - spot_df['billet_ts']
    spot_df['hsec_billet'] = spot_df['hsec_400x200'] - spot_df['billet_ts']
    spot_df['channel_billet'] = spot_df['channel_16'] - spot_df['billet_ts']
    spot_df['ibeam_billet'] = spot_df['ibeam_25'] - spot_df['billet_ts']
    spot_df['angle_billet'] = spot_df['angle_50x5'] - spot_df['billet_ts']
    spot_df['highwire_billet'] = spot_df['highwire_6.5'] - spot_df['billet_ts']

    spot_df['io_inv_mill(64)'] = spot_df['io_inv_imp_mill(64)'] + spot_df['io_inv_dom_mill(64)']

    spot_df['margin_hrc_pbf'] = spot_df['hrc_sh'] - 1.7 * spot_df['pbf_cfd'] - 0.45 * spot_df['coke_xuzhou_xb']
    spot_df['margin_hrc_macf'] = spot_df['hrc_sh'] - 1.7 * spot_df['macf_cfd'] - 0.45 * spot_df['coke_xuzhou_xb']
    spot_df['strip_hsec'] = spot_df['strip_3.0x685'] - spot_df['hsec_400x200']
    return spot_df
