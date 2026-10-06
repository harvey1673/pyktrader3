"""Build an auditable signal-research catalog from the active source maps."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

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


DEFAULT_SPOT = Path("C:/dev/data/spot_df_20261001.parquet")
DEFAULT_OUTPUT = ROOT / "docs/fundamental_signal_coverage_2026-10-04.csv"

CATEGORY_RULES = (
    ("spread_premium", ("价差", "升贴水", "贴水", "溢价", "基差", "spread", "prem", "discount", "basis", "arb")),
    ("cost_margin", ("利润", "毛利", "成本", "加工费", "盈亏", "电价", "margin", "profit", "cost", "power_px", "procfee")),
    ("sentiment", ("情绪", "竞拍成交率", "挂牌量", "senti", "auction", "listed")),
    ("inventory", ("库存", "仓单", "存栏", "库容", "储备", "_inv", "inv_", "stockdays", "_stock", "stock_")),
    ("supply", ("产量", "产能", "开工率", "产能利用率", "发货量", "到港量", "通关量", "日均产量", "_prod", "prod_", "dprod", "util", "workrate", "throughput")),
    ("demand", ("需求", "成交量", "采购数量", "销量", "日耗", "表观消费", "demand", "dmd", "sales", "volume", "purchase_qty")),
    ("price", ("价格", "市场价", "现货价", "出厂价", "到岸价", "平仓价", "均价", "spot", "price", "_px", "cfr", "fob")),
)


def classify(alias: str, description: str) -> str:
    text = f"{alias.lower()} {description.lower()}"
    for category, terms in CATEGORY_RULES:
        if any(term.lower() in text for term in terms):
            return category
    return "other"


def joined_description(rows):
    return "；".join(f"{sheet}: {name}" for sheet, name in rows)


def source_records(source, mapping, metadata, frame, mysteel_aliases):
    records = []
    for code, alias in mapping.items():
        series = frame[alias].dropna()
        rows = metadata[code]
        description = "；".join(name for _, name in rows)
        effective = source == "mysteel" or alias not in mysteel_aliases
        records.append({
            "source": source,
            "effective_loader_source": "yes" if effective else "no_mysteel_precedence",
            "code": code,
            "alias": alias,
            "category": classify(alias, description),
            "worksheet_and_description": joined_description(rows),
            "first_observation": series.index.min().date().isoformat(),
            "last_observation": series.index.max().date().isoformat(),
            "observation_count": int(series.notna().sum()),
            "available_at_2016_start": "yes" if series.index.min() <= pd.Timestamp("2016-01-01") else "no",
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=Path, default=DEFAULT_SPOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_parquet(args.spot).sort_index()
    _, _, ifind_map = literal_dict(SPOT_MAP_PATH, "index_map")
    _, _, mysteel_map = literal_dict(SPOT_MAP_PATH, "mysteel_index_map")

    ifind_metadata = {}
    for filename, sheets in IFIND_SHEETS.items():
        current = workbook_rows(INPUT / filename, sheets, id_row=6, name_row=3)
        for code, rows in current.items():
            ifind_metadata.setdefault(code, [])
            for row in rows:
                if row not in ifind_metadata[code]:
                    ifind_metadata[code].append(row)
    mysteel_metadata = workbook_rows(
        INPUT / "mysteel_data.xlsx", ["metal", "petchem"], id_row=5, name_row=2
    )

    records = source_records(
        "ifind", ifind_map, ifind_metadata, frame, set(mysteel_map.values())
    )
    records += source_records(
        "mysteel", mysteel_map, mysteel_metadata, frame, set(mysteel_map.values())
    )
    records.sort(key=lambda row: (row["category"], row["source"], row["alias"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    categories = Counter(row["category"] for row in records)
    long_history = Counter(
        row["category"] for row in records
        if row["available_at_2016_start"] == "yes"
    )
    suppressed = sum(
        row["effective_loader_source"] == "no_mysteel_precedence" for row in records
    )
    print({
        "rows": len(records),
        "categories": dict(categories),
        "available_at_2016_start": dict(long_history),
        "ifind_aliases_suppressed_by_mysteel": suppressed,
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
