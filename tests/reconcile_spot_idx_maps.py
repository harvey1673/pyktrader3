"""Reconcile spot index maps with the active iFind and MySteel workbooks.

This is a local research maintenance helper.  It preserves aliases already
curated in ``spot_idx_map.py``, adds explicit aliases for newly introduced
workbook fields, and writes exact worksheet/Chinese-name comments.
"""

import ast
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUT = Path("C:/Users/harve/Nutstore/1/Nutstore")
SPOT_MAP_PATH = ROOT / "pycmqlib3/utility/spot_idx_map.py"
FULL_MAP_PATH = ROOT / "tests/index_map_full.py"
MYSTEEL_MAP_PATH = ROOT / "tests/mysteel_ticker_map.py"

CZCE_ALIAS_PREFIXES = (
    "AP", "CF", "CJ", "FG", "MA", "OI", "PF", "PK", "PR", "PX",
    "RM", "SA", "SF", "SH", "SM", "SR", "TA", "UR",
)

IFIND_SHEETS = OrderedDict([
    ("ifind_daily.xlsx", ["petchem_d", "macro_d", "ferrous_d", "base_d"]),
    ("ifind_data.xlsx", [
        "ferrous_w", "base_w", "const_d", "petchem_w", "ag_w",
        "base_d2", "macro_m", "warrant_d",
    ]),
])

NEW_IFIND_ALIASES = {
    "S006145828": "ss_304_2b_hongwang_wuxi_if",
    "S010795263": "br9000_yangzi_sh_if",
    "S016192874": "SF_72_tj_if",
    "S002836134": "l_lldpe_cfr_sea",
    "S002836137": "l_lldpe_cfr_cn",
    "S002950528": "eg_shpc_exw",
    "S002955440": "eb_cfr_tw",
    "S002955446": "eb_cfr_sea",
    "S003157687": "l_7042_qilu_zibo",
    "S003157729": "l_7042_jilin_hz",
    "S004321828": "ru_scrwf_sh",
    "S004414102": "pp_t30s_daqing_exw",
    "S004414126": "pp_t30s_daqing_east",
    "S004414755": "pp_t30s_shaoxing_hz",
    "S004414841": "eb_cfr_cn_mid",
    "S004807554": "sp_silver_sd",
    "S004807572": "sp_silver_jzh",
    "S004808173": "sp_silver_cfr",
    "S004812255": "nr_str20_mix_qd_bonded",
    "S005402570": "cement_spot_cn",
    "S005656440": "eaf_util_weekly",
    "S006731716": "propane_cfr_south",
    "S006731717": "butane_cfr_south",
    "S006731718": "propane_cfr_east",
    "S006731719": "butane_cfr_east",
    "S006731720": "propane_cfr_tw",
    "S006731721": "butane_cfr_tw",
    "S006731733": "propane_discount_me_fob",
    "S006731735": "propane_prem_jp_cfr",
    "S006731739": "propane_prem_south_cfr",
    "S006731741": "propane_prem_east_cfr",
    "S009065254": "cement_px_idx_yangtze",
    "S009065264": "cement_px_idx_cn",
    "S011004429": "eb_shandong_delivered",
    "S011004432": "eb_jiangsu_n1",
    "S011004435": "eb_jiangsu_n2",
    "S012417638": "TA_cfr_sea",
    "S012691163": "concrete_px_idx_cn",
    "S012691166": "concrete_px_idx_east",
    "S016695728": "eb_east_selfpickup",
    "S016701640": "eg_east_spot_mid",
    "S016702455": "eg_cfr_nea",
    "S017304503": "nr_str20_usd_qd_bonded",
    "S017438010": "FG_weekly_melt",
    "S019294884": "SF_hesteel_purchase_px",
    "S019294885": "SM_hesteel_purchase_px",
    "S019294886": "SF_hesteel_purchase_qty",
    "S019294887": "SM_hesteel_purchase_qty",
    "S019988718": "ferroalloy_power_px_gansu",
    "S019988719": "ferroalloy_power_px_qinghai",
    "S019988720": "ferroalloy_power_px_ningxia",
    "S019988722": "ferroalloy_power_px_neimeng",
    "S019988726": "ferroalloy_power_px_yunnan",
    "S019988727": "ferroalloy_power_px_guangxi",
}

NEW_MYSTEEL_ALIASES = {
    "ID00112684": "PX_cfr_tw_cny",
    "ID00112688": "PX_cfr_tw_usd",
    "ID00112716": "PX_fob_kr_cny",
    "ID00112728": "PX_fob_kr_usd",
    "ID00112732": "PX_fob_rotterdam_usd",
    "ID00115336": "PX_sinopec_sh_settlement_east",
    "ID00186597": "io_spot_trade_volume_ports_w_ms",
    "ID00187013": "io_trade_volume_ports_d_ms",
    "ID00187443": "ss_300_inv_wuxi_30_ms",
    "ID00188061": "OI_inv_east_w_ms",
    "ID00188062": "soybean_inv_ports_cn_w_ms",
    "ID00188063": "soybean_inv_crushers_111_w_ms",
    "ID00188064": "m_inv_crushers_111_w_ms",
    "ID00188065": "y_inv_crushers_90_w_ms",
    "ID00188307": "al_sinv_cn_d_ms",
    "ID00188359": "pe_inv_social_cn_w_ms",
    "ID00375008": "pg_inv_mill_cn_w_ms",
    "ID00384633": "ss_300_inv_foshan_14_ms",
    "ID00394230": "eg_east_spot_ms",
    "ID00408152": "ni_ore_1.6_php_cif",
    "ID01001977": "MA_import_taicang_spot_ms",
    "ID01002072": "al_inv_mill_cn_d_ms",
    "ID01024124": "MA_spot_ordos_south_ms",
    "ID01027073": "coal_inv_55ports_w_ms",
    "ID01030576": "RM_inv_crushers_cn_w_ms",
    "ID01201815": "RM_pellet_inv_nantong_w_ms",
    "ID01207170": "c_inv_4north_ports_w_ms",
    "ID01214595": "pp_inv_mill_cn_w_ms",
    "ID01216483": "PK_inv_cn_w_ms",
    "ID01218647": "MA_inv_ports_cn_w_ms",
    "ID01230664": "PET_bottle_invdays_mill_cn_w_ms",
    "ID01232909": "FG_inv_mill_shahe_w_ms",
    "ID01301727": "sp_inv_changshu_port_w_ms",
    "ID01301728": "sp_inv_qingdao_port_w_ms",
    "ID01301853": "c_inv_guangdong_ports_w_ms",
    "ID01369403": "FG_inv_mill_hubei_w_ms",
    "ID01370598": "sp_inv_ports_cn_w_ms",
    "ID01388077": "eg_sh_spot_ms",
    "ID01508544": "RM_inv_south_w_ms",
    "ID01616600": "MA_taicang_paper_lm_ms",
    "ID01616603": "MA_taicang_paper_nm_ms",
    "ID01709994": "soybean_inv_crushers_full_w_ms",
    "ID01733105": "bu_inv_social_104_cn_w_ms",
    "ID01862250": "OI_inv_small_sample_cn_w_ms",
    "ID01881060": "softwood_log_inv_cn_w_ms",
    "ID01891248": "RM_inv_north_w_ms",
    "ID01897968": "RM_inv_ports_cn_w_ms",
    "ID01990129": "v_inv_social_large_cn_w_ms",
    "ID02343778": "OI_inv_large_sample_cn_w_ms",
    "RE00010184": "eb_inv_commercial_js_w_ms",
    "RE00010776": "TA_feedstock_invdays_polyester_w_ms",
    "RE00033240": "pe_inv_mill_cn_w_ms",
}


def canonical_alias(alias):
    """Use the exchange-code case for commodity-specific CZCE aliases."""
    for product in CZCE_ALIAS_PREFIXES:
        prefix = f"{product.lower()}_"
        if alias.startswith(prefix):
            return product + alias[len(product):]
    if alias.startswith(("pta_", "PTA_")):
        return "TA_" + alias[4:]
    return alias


def literal_dict(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        items = [(ast.literal_eval(key), ast.literal_eval(value))
                 for key, value in zip(node.value.keys, node.value.values)]
        values = dict(items)
        order = list(dict.fromkeys(key for key, _ in items))
        return tree, node, OrderedDict((key, values[key]) for key in order)
    raise KeyError(f"{name} not found in {path}")


def workbook_rows(path, sheets, id_row, name_row):
    rows = defaultdict(list)
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        for sheet_name in sheets:
            sheet = workbook[sheet_name]
            for column in range(2, sheet.max_column + 1):
                code = str(sheet.cell(id_row, column).value or "").strip()
                name = str(sheet.cell(name_row, column).value or "").strip()
                if code and name:
                    item = (sheet_name, name)
                    if item not in rows[code]:
                        rows[code].append(item)
    finally:
        workbook.close()
    return rows


def comment_for(rows):
    return "; ".join(f"{sheet}: {name}" for sheet, name in rows)


def render_dict(name, mapping, metadata, used_codes=None):
    used_codes = set(used_codes or ())
    lines = [f"{name} = {{"]
    for code, alias in mapping.items():
        comment = comment_for(metadata[code])
        if code in used_codes:
            comment = f"used in prod; {comment}"
        lines.append(f"    {code!r}: {alias!r},  # {comment}")
    lines.append("}")
    return "\n".join(lines)


def replace_assignment(path, name, rendered):
    text = path.read_text(encoding="utf-8")
    tree, node, _ = literal_dict(path, name)
    lines = text.splitlines()
    lines[node.lineno - 1:node.end_lineno] = rendered.splitlines()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_unique(mapping, label):
    reverse = defaultdict(list)
    for code, alias in mapping.items():
        reverse[alias].append(code)
    duplicates = {alias: codes for alias, codes in reverse.items() if len(codes) > 1}
    if duplicates:
        raise ValueError(f"duplicate {label} aliases: {duplicates}")


def main():
    ifind_metadata = defaultdict(list)
    for filename, sheets in IFIND_SHEETS.items():
        current = workbook_rows(INPUT / filename, sheets, id_row=6, name_row=3)
        for code, rows in current.items():
            for row in rows:
                if row not in ifind_metadata[code]:
                    ifind_metadata[code].append(row)
    mysteel_metadata = workbook_rows(
        INPUT / "mysteel_data.xlsx", ["metal", "petchem"], id_row=5, name_row=2
    )

    _, _, production_ifind = literal_dict(SPOT_MAP_PATH, "index_map")
    _, _, full_ifind = literal_dict(FULL_MAP_PATH, "index_map_full")
    aliases = OrderedDict()
    for code, alias in production_ifind.items():
        if code in ifind_metadata:
            aliases[code] = canonical_alias(NEW_IFIND_ALIASES.get(code, alias))
    for code in sorted(ifind_metadata):
        if code in aliases:
            continue
        if code in NEW_IFIND_ALIASES:
            aliases[code] = canonical_alias(NEW_IFIND_ALIASES[code])
        elif code in full_ifind:
            aliases[code] = canonical_alias(full_ifind[code])
        else:
            raise KeyError(f"missing iFind alias for {code}: {ifind_metadata[code]}")
    assert set(aliases) == set(ifind_metadata)
    assert_unique(aliases, "iFind")

    _, _, production_mysteel = literal_dict(SPOT_MAP_PATH, "mysteel_index_map")
    mysteel_aliases = OrderedDict()
    for code, alias in production_mysteel.items():
        if code in mysteel_metadata:
            mysteel_aliases[code] = canonical_alias(alias)
    for code in sorted(mysteel_metadata):
        if code not in mysteel_aliases:
            mysteel_aliases[code] = canonical_alias(NEW_MYSTEEL_ALIASES[code])
    assert set(mysteel_aliases) == set(mysteel_metadata)
    assert_unique(mysteel_aliases, "MySteel production")
    shared_aliases = set(aliases.values()) & set(mysteel_aliases.values())
    if shared_aliases:
        raise ValueError(f"cross-provider alias collisions: {sorted(shared_aliases)}")

    replace_assignment(
        SPOT_MAP_PATH, "index_map", render_dict("index_map", aliases, ifind_metadata)
    )
    replace_assignment(
        SPOT_MAP_PATH,
        "mysteel_index_map",
        render_dict("mysteel_index_map", mysteel_aliases, mysteel_metadata),
    )

    # The dependency monitor must read the just-reconciled maps.  Annotate only
    # the production map; the test catalogs retain workbook-only comments.
    from misc_scripts.data_dependency_monitor import (
        build_dependency_rows,
        collect_production_index_codes,
    )

    production_codes = collect_production_index_codes(build_dependency_rows())
    used_ifind = {
        code.removeprefix("ifind:")
        for code in production_codes
        if code.startswith("ifind:")
    }
    used_mysteel = {
        code.removeprefix("mysteel:")
        for code in production_codes
        if code.startswith("mysteel:")
    }
    replace_assignment(
        SPOT_MAP_PATH,
        "index_map",
        render_dict("index_map", aliases, ifind_metadata, used_ifind),
    )
    replace_assignment(
        SPOT_MAP_PATH,
        "mysteel_index_map",
        render_dict(
            "mysteel_index_map", mysteel_aliases, mysteel_metadata, used_mysteel
        ),
    )

    full_sorted = OrderedDict((code, aliases[code]) for code in sorted(aliases))
    full_header = '''"""Full iFind mapping reconciled to the active Excel workbook headers.

Aliases match ``pycmqlib3.utility.spot_idx_map.index_map``.  Each entry carries
the exact Chinese indicator name and active worksheet for auditability.
"""\n\n'''
    FULL_MAP_PATH.write_text(
        full_header + render_dict("index_map_full", full_sorted, ifind_metadata) + "\n",
        encoding="utf-8",
    )

    _, _, ticker_names = literal_dict(MYSTEEL_MAP_PATH, "ticker_name")
    if set(ticker_names) != set(mysteel_metadata):
        raise ValueError("mysteel_ticker_map.py does not match the active workbook")
    ticker_sorted = OrderedDict(
        (code, canonical_alias(ticker_names[code])) for code in sorted(ticker_names)
    )
    ticker_header = '''"""Normalized ticker names from the active MySteel workbook.

The key set exactly matches ``mysteel_data.xlsx`` sheets ``metal`` and
``petchem``.  Inline comments retain the exact Chinese header and worksheet.
"""\n\n'''
    MYSTEEL_MAP_PATH.write_text(
        ticker_header + render_dict("ticker_name", ticker_sorted, mysteel_metadata)
        + "\n\n# Descriptive alias for callers that prefer a source-qualified variable name.\n"
        + "mysteel_ticker_map = ticker_name\n",
        encoding="utf-8",
    )

    print({
        "ifind_codes": len(aliases),
        "mysteel_codes": len(mysteel_aliases),
        "ifind_removed_from_production": len(set(production_ifind) - set(aliases)),
        "ifind_added_to_production": len(set(aliases) - set(production_ifind)),
        "mysteel_added_to_production": len(set(mysteel_aliases) - set(production_mysteel)),
        "ifind_used_in_prod": len(used_ifind),
        "mysteel_used_in_prod": len(used_mysteel),
    })


if __name__ == "__main__":
    main()
