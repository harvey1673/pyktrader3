index_map = {
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
    'S009097506': 'long_inv_social',
    'S009045420': 'steel_inv_social',

    'S005580634': 'rebar_inv_social',
    'S005580633': 'wirerod_inv_social',
    'S005580635': 'hrc_inv_social',
    'S005580639': 'crc_inv_social',
    'S005580636': 'plate_inv_social',

    'S004378418': 'rebar_inv_mill',
    'S004378419': 'wirerod_inv_mill',
    'S004378420': 'hrc_inv_mill',
    'S004378421': 'crc_inv_mill',
    'S004378422': 'plate_inv_mill',
}


def process_spot_df(spot_df):
    spot_df['steel_inv_mill'] = spot_df['rebar_inv_mill'] + spot_df['wirerod_inv_mill'] + \
                                spot_df['hrc_inv_mill'] + spot_df['crc_inv_mill'] + spot_df['plate_inv_mill']
    spot_df['steel_inv_all'] = spot_df['steel_inv_social'] + spot_df['steel_inv_mill']
    spot_df['steel_social_inv'] = spot_df['rebar_inv_social'] + spot_df['wirerod_inv_social'] + \
                                  spot_df['hrc_inv_social'] + spot_df['crc_inv_social'] + spot_df['plate_inv_social']
    spot_df['long_social_inv'] = spot_df['rebar_inv_social'] + spot_df['wirerod_inv_social']
    spot_df['flat_social_inv'] = spot_df['hrc_inv_social'] + spot_df['crc_inv_social'] + spot_df['plate_inv_social']

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
