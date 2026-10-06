"""Recommended spot-price universe for CTD research.

Each row describes a candidate price series, not an exchange adjustment.  The
``existing_aliases`` values were inspected in ``spot_idx_map.py`` and
``index_map_full.py``; empty lists identify acquisition/enrichment gaps.
"""

CTD_SPOT_REQUIREMENTS = [
    {
        "product": "j",
        "candidate": "Rizhao port quasi-grade-1 wet-quench coke",
        "existing_aliases": ["coke_sub_a_rz_outstock", "coke_sub_a_rz"],
        "required_metadata": ["A", "S", "M40", "M10", "CRI", "CSR", "Mt", "Mf", "size_25_40", "quote_basis"],
        "priority": "must_have",
    },
    {
        "product": "j",
        "candidate": "Jinzhong/Lvliang wet- and dry-quench coke delivered to port",
        "existing_aliases": ["coke_sub_a_rz_haoyu", "coke_sub_a_dry_lvliang_jinyan", "coke_shanxi", "coke_sub_a_tj"],
        "required_metadata": ["full assay", "quenching_process", "rail_truck_freight", "loading_fee"],
        "priority": "must_have",
    },
    {
        "product": "jm",
        "candidate": "Mongolian No. 5 raw/washed coking coal",
        "existing_aliases": ["ckc_mongol5_ts", "ckc_outstock_ganqimaodu", "ckc_stock_ganqimaodu"],
        "requested_indicator_ids": ["ID01892939", "ID00103479"],
        "preferred_indicator_id": "ID01892939",
        "required_metadata": ["confirm Mongolian No. 5 and washed status", "Tangshan preferred; Ganqimaodu plus freight fallback", "A", "S", "V", "G", "Y", "CSR", "Mt", "tax/cash basis", "freight"],
        "priority": "must_have",
    },
    {
        "product": "jm",
        "candidate": "Shanxi mid-sulfur main coking coal",
        "existing_aliases": ["ckc_midsulfur_jiexiu_kaijia", "ckc_a10v24s08_lvliang", "ckc_a9v18s10_lvliang", "ckc_a10v24s10_ts"],
        "required_metadata": ["confirm quality bucket and origin", "mine_or_city", "A", "S", "V", "G", "Y", "CSR", "Mt", "tax_basis"],
        "priority": "must_have",
    },
    {
        "product": "jm",
        "candidate": "Shanxi low-sulfur and premium Australian hard coking coal",
        "existing_aliases": ["京唐港_库提价含税_澳大利亚_主焦煤"],
        "required_metadata": ["brand_or_origin", "full assay", "port", "warehouse_basis"],
        "priority": "must_have",
    },
    {
        "product": "ss",
        "candidate": "Registered-brand 304/2B coil in Wuxi and Foshan",
        "existing_aliases": ["ss_304_gross_wuxi", "ss_304_2b_hongwang_wuxi", "ss_304_wuxi_phybasis"],
        "required_metadata": ["brand", "thickness_mm", "width_mm", "edge", "registered_status", "tax_basis", "1240mm quote is comparison-only until converted to a deliverable width"],
        "priority": "must_have",
    },
    {
        "product": "SM",
        "candidate": "6517 at Tianjin, Ulanqab, Shizuishan, Qinzhou, Rizhao and Yingkou",
        "existing_aliases": ["SM_65s17_tj", "SM_65s17_neimeng", "SM_65s17_gansu", "SM_65s17_shmet", "SM_hesteel_purchase_px", "SM_hesteel_purchase_qty"],
        "required_metadata": ["exact_city", "ex_factory_or_delivered", "freight_to_delivery_point"],
        "priority": "must_have",
    },
    {
        "product": "SF",
        "candidate": "72 ferrosilicon at Tianjin, Zhongwei, Inner Mongolia, Gansu and consumer areas",
        "existing_aliases": ["SF_72_tj", "SF_72_ningxia", "SF_72_neimeng", "SF_72_gansu", "SF_72_shmet"],
        "required_metadata": ["exact_city", "ex_factory_or_delivered", "freight_to_delivery_point"],
        "priority": "must_have",
    },
    {
        "product": "l/pp/v",
        "candidate": "Registered producer-brand spot by East/North/South China",
        "existing_aliases": ["l_7042_tj", "l_7042_east", "l_7042_north", "l_7042_qilu_zibo", "l_7042_jilin_hz", "pp_t30s_shaoxing_hz", "pp_t30s_daqing_exw", "pp_t30s_daqing_east", "pvc_cac2_north", "pvc_cac2_east", "pvc_ethylene_east"],
        "required_metadata": ["producer", "brand_code", "registered_status", "grade_test", "warehouse_or_factory_pickup"],
        "priority": "must_have",
    },
    {
        "product": "eg/eb",
        "candidate": "East-China tank/warehouse spot plus North/South and CFR imports",
        "existing_aliases": ["eg_east_spot", "eg_east_spot_mid", "eg_cfr_cn", "eg_cfr_nea", "eb_east_spot", "eb_north_spot", "eb_cfr_cn", "eb_cfr_cn_mid", "eb_jiangsu_n1", "eb_jiangsu_n2"],
        "required_metadata": ["tank_or_warehouse", "incoterm", "tax_basis", "port_charges"],
        "priority": "must_have",
    },
    {
        "product": "TA/PX",
        "candidate": "East-China registered PTA brand and domestic/import PX",
        "existing_aliases": ["TA_east_spot", "PX_exw_east_spot", "PX_cfr_tw_usd", "PX_fob_kr_usd", "PX_korea_fob_usd"],
        "required_metadata": ["brand", "registered_or_immune", "pickup_location", "currency", "tax", "freight"],
        "priority": "must_have",
    },
    {
        "product": "MA/UR",
        "candidate": "Consumer and producer-area spot by deliverable grade",
        "existing_aliases": ["MA_zj_spot", "MA_spot_jiangsu", "MA_spot_sd", "MA_spot_neimeng", "UR_henan_spot", "UR_north_spot", "UR_shandong_spot"],
        "required_metadata": ["grade", "location_group", "factory_pickup", "freight", "tax_basis"],
        "priority": "must_have",
    },
    {
        "product": "sc",
        "candidate": "Deliverable Oman/Dubai/ESPO and other named crude grades",
        "existing_aliases": ["oman_spot", "dubai_spot", "espo_spot"],
        "required_metadata": ["grade", "exchange_grade_premium", "USD_CNY", "tariff_VAT", "freight", "insurance", "port_tank_fee"],
        "priority": "must_have",
    },
    {
        "product": "fu/lu",
        "candidate": "Singapore bunker cargo plus East-China delivered fuel oil",
        "existing_aliases": ["fo_380cst_sgp", "fo_180cst_sgp", "fo_380cst_sgp_fob", "fo_380cst_zhoushan", "lu_0.5_sgp", "lu_05_zhoushan", "lu_bonded_zhoushan"],
        "required_metadata": ["grade", "sulfur", "USD_CNY", "tax", "freight", "storage", "bunker_or_cargo_basis"],
        "priority": "must_have",
    },
    {
        "product": "bu",
        "candidate": "Registered heavy asphalt brands in Shandong/North/East China",
        "existing_aliases": ["bu_heavy_shandong", "bu_heavy_north", "bu_heavy_east"],
        "required_metadata": ["brand", "registered_status", "grade", "delivery_location", "factory_pickup"],
        "priority": "must_have",
    },
    {
        "product": "ru/nr/br",
        "candidate": "Registered SCR WF, TSR20/TSR10 origins, and BR9000 brands",
        "existing_aliases": ["ru_scrwf_kunming", "ru_scrwf_zhejiang", "ru_scrwf_jiangsu"],
        "required_metadata": ["brand", "origin", "grade", "production_year", "registered_status", "warehouse_or_factory_pickup"],
        "priority": "must_have",
    },
]


def requirements_for(product: str):
    token = product.lower()
    return [row for row in CTD_SPOT_REQUIREMENTS if token in row["product"].lower().split("/")]
