"""Feature mappings shared by signal definitions and raw-data loaders."""


METAL_INV_FEATURES = {
    "cu": 'cu_sinv_cn_d', # "cu_inv_shfe_d",
    "al": "al_inv_social_all",
    "zn": "zn_inv_social_all",
    "ni": 'ni_sinv_27_cn', # 'ni_inv_shfe_d',
    "pb": 'pb_ingot_sinv_cn_d', # 'pb_inv_shfe_d',
    "sn": 'sn_sinv_cn_w', # 'sn_inv_shfe_d',
    "si": "si_inv_gfex_d",
    "lc": "lc_inv_gfex_d",
    "ps": 'ps_inv_cn_w', # "ps_inv_gfex_d",
    "ao": 'ao_inv_total_cn', # "ao_inv_shfe_d",
    "ss": "ss_inv_social_300",
    "rb": "rebar_inv_social",
    "hc": "hrc_inv_social",
    "j": "coke_inv_ports_tj",
    "jm": 'jm_inv_523mines', # 'ckc_inv_ports',
    "v": "v_inv_social",
    "i": "io_inv_45ports",
    "SM": "sm_stockdays",
    "SF": "sf_inv_mill",
    "FG": "fg_inv_mill",
    "SA": "sa_inv_mill_all",
    "SH": "sh_inv_mill_all",
}
