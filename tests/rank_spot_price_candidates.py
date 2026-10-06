"""Rank mapped spot-price series against same-product futures price changes.

The ranking is descriptive research, not a trading rule.  It uses Friday-to-
Friday log changes from 2016 onward and a chronological 70/30 split so a high
full-sample correlation cannot hide a weak recent relationship.
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.reconcile_spot_idx_maps import (
    IFIND_SHEETS,
    INPUT,
    SPOT_MAP_PATH,
    literal_dict,
    workbook_rows,
)


PRODUCTS = (
    "rb", "hc", "i", "j", "jm", "SM", "SF", "SA", "FG", "v", "SH",
    "cu", "al", "zn", "ni", "pb", "sn", "ss", "ao", "au", "ag",
    "si", "lc", "ps", "ru", "UR", "sp", "nr", "br", "l", "pp",
    "TA", "PX", "eg", "MA", "eb", "sc", "lu", "bu", "fu", "pg",
    "m", "RM", "y", "p", "OI", "a", "b", "c", "cs", "CJ", "CF",
    "jd", "AP", "lh", "SR", "PK",
)

# Matching is deliberately explicit.  Broad substring matching would assign
# inventories, premia, or unrelated products to short codes such as i/c/a/p.
PRODUCT_ALIAS_PREFIXES = {
    "rb": ("rebar_",), "hc": ("hrc_",),
    "i": ("pbf_", "nmf_", "macf_", "jmb_", "ssf_", "iocj_", "plt62"),
    "j": ("coke_",), "jm": ("ckc_",), "SM": ("SM_",), "SF": ("SF_",),
    "SA": ("SA_",), "FG": ("FG_",), "v": ("pvc_",), "SH": ("SH_",),
    "cu": ("cu_",), "al": ("al_",), "zn": ("zn_",), "ni": ("ni_",),
    "pb": ("pb_",), "sn": ("sn_",), "ss": ("ss_304",),
    "ao": ("alumina_",), "au": ("au_9999",), "ag": ("ag_1_", "ag_9999"),
    "si": ("si_",), "lc": ("lc_",), "ps": ("ps_",),
    "ru": ("ru_scrwf_",), "UR": ("UR_",), "sp": ("sp_",),
    "nr": ("nr_str20_",), "br": ("br9000_",), "l": ("l_7042_",),
    "pp": ("pp_t30s_",), "TA": ("TA_",), "PX": ("PX_",),
    "eg": ("eg_",), "MA": ("MA_",), "eb": ("eb_",),
    "sc": ("brent_dtd_spot", "oman_spot", "dubai_spot", "espo_spot"),
    "lu": ("lu_",), "bu": ("bu_heavy_",), "fu": ("fo_",),
    "pg": ("pg_cn_spot", "propane_cfr_", "butane_cfr_"),
    # The active workbooks contain fundamental inventory/balance fields for
    # these products but no mapped outright spot-price aliases.
    "m": (), "RM": (), "y": (), "p": (), "OI": (), "a": (), "b": (),
    "c": (), "cs": (), "CJ": (), "CF": (), "jd": (), "AP": (),
    "lh": (), "SR": (), "PK": (),
}

EXCLUDED_PRICE_TOKENS = (
    "_inv", "inv_", "_warrant", "_prod", "prodcost", "_cost", "margin",
    "_util", "workrate", "senti", "_prem", "phybasis", "_basis", "_tc",
    "procfee", "profit", "volume", "sales", "discount", "stockdays",
)


@lru_cache(maxsize=1)
def _metadata_and_maps():
    ifind_metadata = {}
    for filename, sheets in IFIND_SHEETS.items():
        rows = workbook_rows(INPUT / filename, sheets, id_row=6, name_row=3)
        for code, values in rows.items():
            ifind_metadata.setdefault(code, [])
            for value in values:
                if value not in ifind_metadata[code]:
                    ifind_metadata[code].append(value)
    mysteel_metadata = workbook_rows(
        INPUT / "mysteel_data.xlsx", ["metal", "petchem"], id_row=5, name_row=2
    )
    _, _, ifind_map = literal_dict(SPOT_MAP_PATH, "index_map")
    _, _, mysteel_map = literal_dict(SPOT_MAP_PATH, "mysteel_index_map")
    return (("ifind", ifind_map, ifind_metadata),
            ("mysteel", mysteel_map, mysteel_metadata))


def candidate_records(product: str):
    prefixes = PRODUCT_ALIAS_PREFIXES[product]
    records = []
    for source, mapping, metadata in _metadata_and_maps():
        for code, alias in mapping.items():
            if not prefixes or not alias.startswith(prefixes):
                continue
            description = "；".join(name for _, name in metadata[code])
            if any(token in alias.lower() for token in EXCLUDED_PRICE_TOKENS):
                continue
            records.append({
                "product": product,
                "source": source,
                "code": code,
                "alias": alias,
                "worksheet": "|".join(sheet for sheet, _ in metadata[code]),
                "chinese_name": description,
            })
    return records


def weekly_change_pair(spot: pd.Series, future: pd.Series, start="2016-01-01"):
    frame = pd.concat([
        pd.to_numeric(spot, errors="coerce").rename("spot"),
        pd.to_numeric(future, errors="coerce").rename("future"),
    ], axis=1).sort_index().loc[start:]
    frame = frame.where(frame > 0)
    weekly = frame.resample("W-FRI").last()
    return np.log(weekly).diff().dropna()


def correlation_metrics(pair: pd.DataFrame, min_observations=52):
    nobs = len(pair)
    result = {
        "n_weekly_changes": nobs,
        "first_change": pair.index.min().date().isoformat() if nobs else "",
        "last_change": pair.index.max().date().isoformat() if nobs else "",
        "pearson_full": np.nan,
        "spearman_full": np.nan,
        "spearman_train": np.nan,
        "spearman_test": np.nan,
        "stability_gap": np.nan,
        "score": np.nan,
    }
    if nobs < min_observations:
        return result
    split = max(26, min(nobs - 26, int(nobs * 0.70)))
    train, test = pair.iloc[:split], pair.iloc[split:]
    full_s = pair["spot"].corr(pair["future"], method="spearman")
    train_s = train["spot"].corr(train["future"], method="spearman")
    test_s = test["spot"].corr(test["future"], method="spearman")
    gap = abs(train_s - test_s)
    coverage = min(1.0, nobs / 260.0)
    score = coverage * (0.30 * full_s + 0.25 * train_s + 0.45 * test_s - 0.20 * gap)
    result.update({
        "pearson_full": pair["spot"].corr(pair["future"]),
        "spearman_full": full_s,
        "spearman_train": train_s,
        "spearman_test": test_s,
        "stability_gap": gap,
        "score": score,
    })
    return result


def build_rankings(spot_df: pd.DataFrame, futures_df: pd.DataFrame):
    rows = []
    for product in PRODUCTS:
        contract = f"{product}c1"
        if (contract, "close") not in futures_df.columns:
            rows.append({"product": product, "status": "futures_not_available"})
            continue
        candidates = candidate_records(product)
        if not candidates:
            rows.append({"product": product, "status": "no_mapped_spot_price"})
            continue
        future = futures_df[(contract, "close")]
        for record in candidates:
            alias = record["alias"]
            if alias not in spot_df:
                rows.append({**record, "status": "missing_from_spot_cache"})
                continue
            metrics = correlation_metrics(weekly_change_pair(spot_df[alias], future))
            status = "ranked" if math.isfinite(metrics["score"]) else "short_history"
            rows.append({**record, "status": status, **metrics})
    result = pd.DataFrame(rows)
    result["rank"] = np.nan
    ranked = result["status"].eq("ranked")
    result.loc[ranked, "rank"] = (
        result.loc[ranked].groupby("product")["score"]
        .rank(method="first", ascending=False)
    )
    return result.sort_values(
        ["product", "rank", "score"], ascending=[True, True, False], na_position="last"
    )


def write_markdown(result: pd.DataFrame, path: Path):
    lines = [
        "# Spot/futures price-change correlation ranking",
        "",
        "Weekly Friday log changes from 2016 onward; 70% chronological train and 30% test. "
        "The score rewards positive, stable recent correlation and penalizes train/test drift. "
        "It is a screening statistic, not evidence of causality or signal profitability.",
        "",
        "| Product | Rank | Alias | Source | Weekly obs | Full rho | Test rho | Score | Status |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for product in PRODUCTS:
        subset = result[result["product"].eq(product)]
        ranked = subset[subset["status"].eq("ranked")].head(5)
        shown = ranked if not ranked.empty else subset.head(1)
        for _, row in shown.iterrows():
            def fmt(name, digits=3):
                value = row.get(name, np.nan)
                return f"{value:.{digits}f}" if pd.notna(value) else ""
            alias = row.get("alias", "")
            source = row.get("source", "")
            alias = "" if pd.isna(alias) else str(alias)
            source = "" if pd.isna(source) else str(source)
            lines.append(
                f"| {product} | {fmt('rank', 0)} | `{alias}` | "
                f"{source} | {fmt('n_weekly_changes', 0)} | "
                f"{fmt('spearman_full')} | {fmt('spearman_test')} | "
                f"{fmt('score')} | {row.get('status', '')} |"
            )
    lines += [
        "",
        "A missing spot row means the current mapped workbooks contain balance or inventory data "
        "for that product but no normalized outright spot-price alias. Short histories remain in "
        "the CSV and are not ranked until they have 52 weekly changes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=Path, default=Path("C:/dev/data/spot_df_20261001.parquet"))
    parser.add_argument("--futures", type=Path, default=Path("C:/dev/data/fut_d_20260918.parquet"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spot_price_rankings_2026-10-04.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/spot_price_rankings_2026-10-04.md")
    args = parser.parse_args()
    result = build_rankings(
        pd.read_parquet(args.spot).sort_index(),
        pd.read_parquet(args.futures).sort_index(),
    )
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_markdown(result, args.markdown)
    print({"rows": len(result), "ranked": int(result["status"].eq("ranked").sum()),
           "products": result["product"].nunique(), "output": str(args.output)})


if __name__ == "__main__":
    main()
