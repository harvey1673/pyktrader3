"""Research-only product, location, grade, import-arb and margin signals.

Every formula uses named columns already present in ``spot_idx_map``.  Missing
legs skip one signal without preventing the rest of the research frame from
being built.  Import-parity fields are screening proxies: they include the
stated duty/VAT multipliers but exclude freight, port, financing and quality
adjustments unless the vendor quote already includes them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalSpec:
    products: str
    category: str
    operation: str
    legs: tuple[str, ...]
    chinese_name: str
    note: str = ""
    coefficients: tuple[float, ...] = ()
    duty: float = 1.0
    vat: float = 1.13


SPECS: Mapping[str, SignalSpec] = {
    # Ferrous product, location and grade structure.
    "rb_hc_spot_spread": SignalSpec("rb,hc", "product_spread", "diff", ("rebar_sh", "hrc_sh"), "上海螺纹钢-上海热轧卷板"),
    "rebar_billet_processing_proxy": SignalSpec("rb", "processing_margin", "diff", ("rebar_sh", "billet_ts"), "上海螺纹钢-唐山钢坯"),
    "crc_hrc_processing_proxy": SignalSpec("hc", "processing_margin", "diff", ("crc_sh", "hrc_sh"), "上海冷轧卷板-上海热轧卷板"),
    "hrc_pbf_coke_margin_proxy": SignalSpec("hc", "processing_margin", "linear", ("hrc_sh", "pbf_cfd", "coke_xuzhou_xb"), "热卷-铁矿-焦炭原料成本代理", coefficients=(1.0, -1.7, -0.45)),
    "io_pbf_blend_grade_spread": SignalSpec("i", "grade_differential", "linear", ("pbf_qd", "iocj_qd", "ssf_qd"), "PB粉-40%IOCJ-60%超特粉", coefficients=(1.0, -0.4, -0.6)),
    "coke_rizhao_outstock_pingcang": SignalSpec("j", "quote_basis_spread", "diff", ("coke_sub_a_rz_outstock", "coke_sub_a_rz"), "日照港准一级焦出库价-平仓价"),
    "jm_ganqimaodu_quote_spread": SignalSpec("jm", "quote_basis_spread", "diff", ("ckc_outstock_ganqimaodu", "ckc_stock_ganqimaodu"), "甘其毛都主焦煤库提含税价-库存提货价"),
    "SM_SF_tianjin_product_spread": SignalSpec("SM,SF", "product_spread", "diff", ("SM_65s17_tj", "SF_72_tj"), "天津锰硅65/17-硅铁72"),
    "SA_heavy_light_north_grade": SignalSpec("SA", "grade_differential", "diff", ("SA_heavy_north", "SA_light_north"), "华北重质纯碱-轻质纯碱"),
    "FG_shahe_north_location": SignalSpec("FG", "location_spread", "diff", ("FG_5mm_shahe", "FG_5mm_north"), "沙河5mm大板玻璃-华北5mm玻璃"),

    # Chlor-alkali, metals and new-energy materials.
    "pvc_east_north_location": SignalSpec("v", "location_spread", "diff", ("pvc_cac2_east", "pvc_cac2_north"), "华东电石法PVC-华北电石法PVC"),
    "pvc_south_east_location": SignalSpec("v", "location_spread", "diff", ("pvc_cac2_south", "pvc_cac2_east"), "华南电石法PVC-华东电石法PVC"),
    "pvc_ethylene_cac2_route": SignalSpec("v", "grade_differential", "diff", ("pvc_ethylene_east", "pvc_cac2_east"), "华东乙烯法PVC-华东电石法PVC"),
    "SH_50_32_dry_basis": SignalSpec("SH", "grade_differential", "dry_basis_diff", ("SH_50_spot_sdjl_shandong", "SH_32_spot_sdjl_shandong"), "山东50%液碱与32%液碱折百价差"),
    "ni_jinchuan_import_grade": SignalSpec("ni", "grade_differential", "diff", ("ni_smm1_jc_spot", "ni_smm1_imp_spot"), "金川镍-进口镍"),
    "pb_primary_secondary_grade": SignalSpec("pb", "grade_differential", "diff", ("pb_smm1_spot", "pb_sec9997_spot"), "原生1#铅-再生精铅"),
    "ss_gross_hongwang_grade": SignalSpec("ss", "grade_differential", "diff", ("ss_304_gross_wuxi", "ss_304_2b_hongwang_wuxi"), "无锡304/2B毛边卷-无锡宏旺2*1240*C"),
    "si_421_553_east_grade": SignalSpec("si", "grade_differential", "diff", ("si_421_east", "si_553_nonoxy_east"), "华东421工业硅-553不通氧工业硅"),
    "lc_battery_industrial_grade": SignalSpec("lc", "grade_differential", "diff", ("lc_bat_dom_cn_spot", "lc_ind_dom_cn_spot"), "国产电池级-工业级碳酸锂"),

    # Rubber and pulp.
    "ru_jiangsu_kunming_location": SignalSpec("ru", "location_spread", "diff", ("ru_scrwf_jiangsu", "ru_scrwf_kunming"), "江苏全乳胶-昆明全乳胶"),
    "br_qilu_yangzi_brand_location": SignalSpec("br", "grade_differential", "diff", ("br9000_qilu_sd", "br9000_yangzi_sh"), "山东齐鲁BR9000-上海扬子BR9000"),
    "ru_br_product_spread": SignalSpec("ru,br", "product_spread", "diff", ("ru_scrwf_jiangsu", "br9000_yangzi_sh"), "江苏全乳胶-上海扬子BR9000"),
    "sp_silver_sd_jzh_location": SignalSpec("sp", "location_spread", "diff", ("sp_silver_sd", "sp_silver_jzh"), "银星针叶浆山东-江浙沪"),
    "sp_silver_import_parity": SignalSpec("sp", "import_export_arb", "usd_import_diff", ("sp_silver_cfr", "usdcny_xe", "sp_silver_sd"), "银星CFR完税代理-山东现货", note="0% duty, 13% VAT; excludes freight, port and financing"),

    # Polyolefins and aromatics.
    "l_7042_tj_sh_location": SignalSpec("l", "location_spread", "diff", ("l_7042_tj", "l_7042_sh"), "大庆7042天津-上海"),
    "l_7042_north_east_location": SignalSpec("l", "location_spread", "diff", ("l_7042_north", "l_7042_east"), "LLDPE7042华北均价-华东均价"),
    "l_cfr_cn_import_parity": SignalSpec("l", "import_export_arb", "usd_import_diff", ("l_lldpe_cfr_cn", "usdcny_xe", "l_7042_east"), "LLDPE CFR中国完税代理-华东7042", note="6.5% duty, 13% VAT; excludes freight, port and financing", duty=1.065),
    "pp_daqing_east_exw_channel": SignalSpec("pp", "location_spread", "diff", ("pp_t30s_daqing_east", "pp_t30s_daqing_exw"), "大庆T30S中油华东-企业出厂"),
    "pp_shaoxing_daqing_east_location": SignalSpec("pp", "location_spread", "diff", ("pp_t30s_shaoxing_hz", "pp_t30s_daqing_east"), "杭州绍兴三圆T30S-大庆中油华东"),
    "pp_propylene_processing_proxy": SignalSpec("pp", "processing_margin", "diff", ("pp_t30s_shaoxing_hz", "pl_east_spot"), "华东PP T30S-华东丙烯"),
    "TA_cfr_import_parity": SignalSpec("TA", "import_export_arb", "usd_import_diff", ("TA_cfr_cn", "usdcny_xe", "TA_east_spot"), "PTA CFR中国完税代理-华东现货", note="13% VAT; excludes freight, port and financing"),
    "TA_PX_processing_proxy": SignalSpec("TA,PX", "processing_margin", "linear", ("TA_east_spot", "PX_exw_east_spot"), "PTA-PX加工差代理", coefficients=(1.0, -0.655)),
    "PX_cfr_fob_freight_proxy": SignalSpec("PX", "import_export_arb", "diff", ("PX_cfr_tw_usd", "PX_fob_kr_usd"), "PX CFR台湾-FOB韩国"),
    "eg_cfr_import_parity": SignalSpec("eg", "import_export_arb", "usd_import_diff", ("eg_cfr_nea", "usdcny_xe", "eg_east_spot"), "MEG东北亚CFR完税代理-华东现货", note="13% VAT; excludes freight, port and financing"),
    "eg_sh_east_location": SignalSpec("eg", "location_spread", "diff", ("eg_sh_spot_ms", "eg_east_spot_ms"), "MEG上海-华东"),
    "MA_jiangsu_neimeng_location": SignalSpec("MA", "location_spread", "diff", ("MA_spot_jiangsu", "MA_spot_neimeng"), "江苏甲醇-内蒙古甲醇"),
    "MA_taicang_curve": SignalSpec("MA", "calendar_spread", "diff", ("MA_taicang_paper_lm_ms", "MA_taicang_paper_nm_ms"), "太仓甲醇当月下旬-次月下旬纸货"),
    "MA_cfr_import_parity": SignalSpec("MA", "import_export_arb", "usd_import_diff", ("MA_cfr_cn", "usdcny_xe", "MA_spot_jiangsu"), "甲醇CFR中国完税代理-江苏现货", note="13% VAT; excludes duty/freight/port/financing"),
    "eb_N1_N2_curve": SignalSpec("eb", "calendar_spread", "diff", ("eb_jiangsu_n1", "eb_jiangsu_n2"), "江苏苯乙烯N+1-N+2"),
    "eb_shandong_east_location": SignalSpec("eb", "location_spread", "diff", ("eb_shandong_delivered", "eb_east_selfpickup"), "山东苯乙烯送到-华东自提"),
    "eb_cfr_import_parity": SignalSpec("eb", "import_export_arb", "usd_import_diff", ("eb_cfr_cn_mid", "usdcny_xe", "eb_east_spot"), "苯乙烯CFR中国完税代理-华东现货", note="13% VAT; excludes freight, port and financing"),
    "eb_benzene_processing_proxy": SignalSpec("eb", "processing_margin", "diff", ("eb_east_spot", "bz_east_spot"), "华东苯乙烯-华东纯苯"),

    # Energy and LPG.
    "lu_hsfo_zhoushan_product": SignalSpec("lu,fu", "product_spread", "diff", ("lu_05_zhoushan", "fo_bonded_highsulfur_zhoushan"), "舟山0.5%低硫船用油-保税高硫船用油"),
    "lu_zhoushan_qingdao_location": SignalSpec("lu", "location_spread", "diff", ("lu_bonded_zhoushan", "lu_bonded_qingdao"), "舟山-青岛保税低硫船用油"),
    "fu_380_m1_m2_curve": SignalSpec("fu", "calendar_spread", "diff", ("fo_380cst_m1_sgp", "fo_380cst_m2_sgp"), "新加坡380CST近月-次月纸货"),
    "fu_380_180_grade": SignalSpec("fu", "grade_differential", "diff", ("fo_380cst_sgp_fob", "fo_180cst_sgp_fob"), "新加坡船用380CST-180CST"),
    "bu_shandong_east_location": SignalSpec("bu", "location_spread", "diff", ("bu_heavy_shandong", "bu_heavy_east"), "山东重交沥青-华东重交沥青"),
    "pg_propane_butane_south": SignalSpec("pg", "grade_differential", "diff", ("propane_cfr_south", "butane_cfr_south"), "华南CFR丙烷-丁烷"),
    "pg_propane_south_east_location": SignalSpec("pg", "location_spread", "diff", ("propane_cfr_south", "propane_cfr_east"), "丙烷CFR华南-华东"),
}


def _calculate(frame: pd.DataFrame, spec: SignalSpec) -> pd.Series:
    values = [pd.to_numeric(frame[column], errors="coerce") for column in spec.legs]
    if spec.operation == "diff":
        return values[0] - values[1]
    if spec.operation == "linear":
        if len(spec.coefficients) != len(values):
            raise ValueError(f"coefficient count does not match {spec.legs}")
        result = values[0] * spec.coefficients[0]
        for value, coefficient in zip(values[1:], spec.coefficients[1:]):
            result = result + value * coefficient
        return result
    if spec.operation == "dry_basis_diff":
        return values[0] / 0.50 - values[1] / 0.32
    if spec.operation == "usd_import_diff":
        return values[0] * values[1] * spec.duty * spec.vat - values[2]
    raise ValueError(f"unsupported operation: {spec.operation}")


def build_fundamental_spreads(
    spot_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return derived series and a one-row-per-formula availability report."""
    output = {}
    audit = []
    for name, spec in SPECS.items():
        missing = [column for column in spec.legs if column not in spot_df.columns]
        if missing:
            audit.append({"signal": name, "products": spec.products,
                          "category": spec.category, "status": "missing_inputs",
                          "inputs": "|".join(spec.legs), "missing": "|".join(missing),
                          "non_null": 0, "first_observation": "", "last_observation": "",
                          "chinese_name": spec.chinese_name, "note": spec.note})
            continue
        series = _calculate(spot_df, spec).replace([np.inf, -np.inf], np.nan)
        output[name] = series
        valid = series.dropna()
        audit.append({"signal": name, "products": spec.products,
                      "category": spec.category,
                      "status": "available" if not valid.empty else "all_null",
                      "inputs": "|".join(spec.legs), "missing": "",
                      "non_null": len(valid),
                      "first_observation": valid.index.min().date().isoformat() if len(valid) else "",
                      "last_observation": valid.index.max().date().isoformat() if len(valid) else "",
                      "chinese_name": spec.chinese_name, "note": spec.note})
    return pd.DataFrame(output, index=spot_df.index), pd.DataFrame(audit)


def write_catalog_markdown(audit: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Research fundamental spread signals",
        "",
        "These formulas are test-side candidates built from current `spot_idx_map` aliases. "
        "They are not wired into production signals. Import-parity proxies omit any freight, "
        "port, financing or quality term not stated in the formula note.",
        "",
        "| Signal | Products | Type | Inputs | History | Chinese description |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in audit.sort_values(["category", "products", "signal"]).iterrows():
        history = (
            f"{row['first_observation']} to {row['last_observation']} ({row['non_null']})"
            if row["status"] == "available" else row["status"]
        )
        lines.append(
            f"| `{row['signal']}` | {row['products']} | {row['category']} | "
            f"`{row['inputs'].replace('|', '`, `')}` | {history} | {row['chinese_name']} |"
        )
    lines += [
        "",
        "Use the level and change of each series as separate candidates. Apply publication-date "
        "lags before backtesting weekly vendor series, and compare results after transaction costs "
        "and by subperiod before adding a formula to `signal_repo`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=Path, default=Path("C:/dev/data/spot_df_20261001.parquet"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "docs/fundamental_spread_catalog_2026-10-04.csv")
    parser.add_argument("--markdown", type=Path, default=Path(__file__).resolve().parents[1] / "docs/fundamental_spread_signals_2026-10-04.md")
    args = parser.parse_args()
    spreads, audit = build_fundamental_spreads(pd.read_parquet(args.spot).sort_index())
    audit.to_csv(args.catalog, index=False, encoding="utf-8-sig")
    write_catalog_markdown(audit, args.markdown)
    if args.output:
        spreads.to_parquet(args.output)
    print({"signals": len(audit), "available": int(audit["status"].eq("available").sum()),
           "missing": int(audit["status"].eq("missing_inputs").sum()),
           "output": str(args.output) if args.output else "not_written",
           "catalog": str(args.catalog), "markdown": str(args.markdown)})


if __name__ == "__main__":
    main()
