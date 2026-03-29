"""Build portfolio class-level notional and risk summaries.

This script reads the daily portfolio target file and the per-strategy
decomposition file produced by ``factor_data_update.py`` and aggregates the
positions by level-1 and level-2 product classes using
``misc.product_class_map``.

Per-product notional and risk metrics reuse the same 20-day logic as
``misc_scripts/contract_summary.py``:

- ``vol_20d``: 20-day annualized percentage volatility
- ``risk_per_lot``: 1-standard-deviation daily PnL move for 1 lot based on a
  20-day window of price differences
- ``notional_10k_per_lot``: contract value per lot in 10k CNY

Aggregation conventions:

- notional (1m RMB) = ``lots * notional_10k_per_lot / 100``
- risk (1k RMB) = ``lots * risk_per_lot / 1000``

The output is emailed as HTML with two tables:

- ``notional_1m``: rows are strategies plus ``TOTAL_PORT`` and columns are
    2-level product classes ``(level1, level2)``
- ``risk_1k``: rows are strategies plus ``TOTAL_PORT`` and columns are
    2-level product classes ``(level1, level2)``

Usage:
    python misc_scripts/portfolio_class_summary.py 20260330
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WTPY_ROOT = Path(r"c:\dev\wtpy")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if WTPY_ROOT.exists() and str(WTPY_ROOT) not in sys.path:
    sys.path.insert(0, str(WTPY_ROOT))

from pycmqlib3.utility import misc
from pycmqlib3.utility.email_tool import send_html_by_smtp
from pycmqlib3.utility.process_wt_data import load_hist_bars_to_df
from pycmqlib3.utility.sec_bits import EMAIL_NOTIFY, EMAIL_QQ, LOCAL_PC_NAME, NOTIFIERS


DEFAULT_PORT_DIR = Path(r"c:\dev\pyktrader3\process\paper_sim1")
DEFAULT_PORT_FILE = "PTSIM1_FACTPORT1_hot"
DEFAULT_HOTMAP = Path(r"C:\Users\harve\Nutstore\1\Nutstore\hotmap_prod.json")
DEFAULT_COMMODITIES = Path(r"c:\dev\wtdev\common\commodities.json")
LOOKBACK_DAYS = 50
STD_WIN = 20


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build class-level notional and risk summary tables."
    )
    parser.add_argument(
        "run_date",
        nargs="?",
        help="Run date in YYYYMMDD format. Defaults to latest file date.",
    )
    parser.add_argument(
        "--port-dir",
        default=str(DEFAULT_PORT_DIR),
        help="Directory containing daily portfolio JSON files.",
    )
    parser.add_argument(
        "--port-file",
        default=DEFAULT_PORT_FILE,
        help="Portfolio file stem used by factor_data_update.py.",
    )
    parser.add_argument(
        "--hotmap",
        default=str(DEFAULT_HOTMAP),
        help="Path to the production hotmap JSON used to map product to contract.",
    )
    parser.add_argument(
        "--commodities",
        default=str(DEFAULT_COMMODITIES),
        help="Path to commodities.json for volscale lookup.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Disable email sending and only print the two tables.",
    )
    return parser.parse_args()


def infer_latest_run_date(port_dir: Path, port_file: str) -> str:
    """Infer the latest available run date from portfolio files."""
    prefix = f"{port_file}_"
    candidates = sorted(
        path.stem.replace(prefix, "")
        for path in port_dir.glob(f"{port_file}_*.json")
        if path.name.startswith(prefix)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No portfolio files matching {port_file}_*.json under {port_dir}"
        )
    return candidates[-1]


def load_flat_json(path: Path) -> Dict[str, dict]:
    """Load a JSON file and return its parsed object."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_exchange_map(path: Path) -> Dict[str, dict]:
    """Flatten an exchange-keyed JSON mapping into a product-keyed mapping."""
    raw = load_flat_json(path)
    result: Dict[str, dict] = {}
    for exch_dict in raw.values():
        result.update(exch_dict)
    return result


def get_product_class(product: str) -> Tuple[str, str]:
    """Return level-1 and level-2 product classes."""
    return misc.product_class_map.get(product, ("unknown", "unknown"))


def get_volscale(product: str, volscale_map: Dict[str, dict]) -> float | None:
    """Return the contract multiplier for a product if available."""
    info = (
        volscale_map.get(product)
        or volscale_map.get(product.lower())
        or volscale_map.get(product.upper())
    )
    return info.get("volscale") if isinstance(info, dict) else None


def normalize_contract(contract: str, ref_date: datetime.date) -> str:
    """Normalize external hotmap contract codes to local storage format."""
    if not contract:
        return contract

    product = misc.inst2product(contract)
    exch = misc.prod2exch(product)
    if exch == "CZCE" and len(contract) == len(product) + 3:
        cont_mth = misc.inst2contmth(contract, ref_date)
        return f"{product}{cont_mth % 10000:04d}"
    return contract


def load_bars(contract: str, end_date: datetime.date, n_days: int) -> pd.DataFrame:
    """Load daily bars for one contract."""
    contract = normalize_contract(contract, end_date)
    start_date = end_date - datetime.timedelta(days=n_days * 2)
    exch = misc.prod2exch(misc.inst2product(contract))
    return load_hist_bars_to_df(
        f"{exch}.{contract}",
        start_date=start_date,
        end_date=end_date,
        freq="d",
    )


def calc_contract_metrics(
    product: str,
    contract: str,
    end_date: datetime.date,
    volscale_map: Dict[str, dict],
) -> Dict[str, float | str | None]:
    """Calculate 20-day volatility, per-lot risk, and per-lot notional."""
    contract = normalize_contract(contract, end_date) if contract else contract
    row: Dict[str, float | str | None] = {
        "product": product,
        "contract": contract,
    }
    level1, level2 = get_product_class(product)
    row["level1"] = level1
    row["level2"] = level2

    if not contract:
        row.update(
            {
                "close": None,
                "vol_20d": None,
                "risk_per_lot": None,
                "notional_10k_per_lot": None,
            }
        )
        return row

    bars = load_bars(contract, end_date, LOOKBACK_DAYS)
    volscale = get_volscale(product, volscale_map)
    if bars is None or bars.empty or volscale is None:
        row.update(
            {
                "close": None,
                "vol_20d": None,
                "risk_per_lot": None,
                "notional_10k_per_lot": None,
            }
        )
        return row

    close = bars["close"].dropna()
    pct = close.pct_change().dropna()
    diff = close.diff().dropna()

    row["close"] = round(float(close.iloc[-1]), 4) if not close.empty else None
    row["vol_20d"] = (
        round(float(pct.iloc[-STD_WIN:].std() * (252 ** 0.5) * 100), 2)
        if len(pct) >= STD_WIN
        else None
    )
    row["risk_per_lot"] = (
        round(float(diff.iloc[-STD_WIN:].std() * volscale), 2)
        if len(diff) >= STD_WIN
        else None
    )
    row["notional_10k_per_lot"] = (
        round(float(close.iloc[-1] * volscale / 10000), 2)
        if not close.empty
        else None
    )
    return row


def build_product_metric_table(
    products: Iterable[str],
    hotmap: Dict[str, str],
    run_date: datetime.date,
    volscale_map: Dict[str, dict],
) -> pd.DataFrame:
    """Build one metric row per product using the hot contract map."""
    rows: List[Dict[str, float | str | None]] = []
    for product in sorted(set(products)):
        rows.append(
            calc_contract_metrics(product, hotmap.get(product), run_date, volscale_map)
        )
    return pd.DataFrame(rows).set_index("product")


def normalize_strategy_name(name: str) -> str:
    """Convert file-style strategy names into display-friendly labels."""
    return Path(name).stem


def positions_to_detail_rows(
    strategy_positions: Dict[str, Dict[str, float]],
    product_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Explode strategy positions into one row per strategy/product."""
    rows: List[Dict[str, float | str | None]] = []
    for strategy_name, product_pos in strategy_positions.items():
        row_name = normalize_strategy_name(strategy_name)
        for product, lots in product_pos.items():
            if product not in product_metrics.index:
                continue
            metric_row = product_metrics.loc[product]
            lots_value = float(lots)
            gross_lots = abs(lots_value)
            risk_per_lot = metric_row.get("risk_per_lot")
            notional_per_lot = metric_row.get("notional_10k_per_lot")
            rows.append(
                {
                    "strategy": row_name,
                    "product": product,
                    "contract": metric_row.get("contract"),
                    "level1": metric_row.get("level1"),
                    "level2": metric_row.get("level2"),
                    "lots": lots_value,
                    "gross_lots": gross_lots,
                    "close": metric_row.get("close"),
                    "vol_20d": metric_row.get("vol_20d"),
                    "risk_per_lot": risk_per_lot,
                    "notional_10k_per_lot": notional_per_lot,
                    "risk_rmb": (
                        round(lots_value * float(risk_per_lot), 2)
                        if pd.notna(risk_per_lot)
                        else None
                    ),
                    "risk_1k": (
                        round(lots_value * float(risk_per_lot) / 1000.0, 4)
                        if pd.notna(risk_per_lot)
                        else None
                    ),
                    "notional_10k": (
                        round(lots_value * float(notional_per_lot), 2)
                        if pd.notna(notional_per_lot)
                        else None
                    ),
                    "notional_1m": (
                        round(lots_value * float(notional_per_lot) / 100.0, 6)
                        if pd.notna(notional_per_lot)
                        else None
                    ),
                }
            )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError("No position rows were generated from the input files.")
    return detail_df.sort_values(["strategy", "level1", "level2", "product"])


def build_two_level_table(
    detail_df: pd.DataFrame,
    value_col: str,
    strategy_order: List[str] | None = None,
) -> pd.DataFrame:
    """Build one metric table with (level1, level2) columns."""
    table = detail_df.pivot_table(
        index="strategy",
        columns=["level1", "level2"],
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    )
    table = table.sort_index(axis=1, level=[0, 1])
    table[("total", "total")] = table.sum(axis=1)
    if "TOTAL_PORT" not in table.index:
        table.loc["TOTAL_PORT"] = table.sum(axis=0)

    if strategy_order:
        preferred = [name for name in strategy_order if name in table.index]
        others = [name for name in table.index if name not in preferred and name != "TOTAL_PORT"]
        row_order = preferred + others
        if "TOTAL_PORT" in table.index:
            row_order.append("TOTAL_PORT")
        table = table.reindex(row_order)

    return table.round(2)


def resolve_paths(
    port_dir: Path,
    port_file: str,
    run_date_str: str,
) -> Tuple[Path, Path]:
    """Resolve input paths."""
    port_path = port_dir / f"{port_file}_{run_date_str}.json"
    strat_path = port_dir / f"pos_by_strat_{port_file}_{run_date_str}.json"
    return port_path, strat_path


def get_strategy_order(port_name: str) -> List[str]:
    """Get display row order based on factor_data_update.port_pos_config."""
    try:
        from misc_scripts.factor_data_update import port_pos_config  # lazy import
    except Exception:
        return []

    cfg = port_pos_config.get(port_name, {})
    strat_list = cfg.get("strat_list", [])
    order: List[str] = []
    for item in strat_list:
        strat_file = item[0]
        order.append(normalize_strategy_name(strat_file))
    return order


def send_tables_email(
    run_date: datetime.date,
    port_name: str,
    notional_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    email_notify: bool = EMAIL_NOTIFY,
) -> None:
    """Send notional and risk tables by email."""
    if not email_notify:
        return

    sub = f"{LOCAL_PC_NAME} portfolio class summary <{run_date.strftime('%Y.%m.%d')}>"
    html = (
        "<html><head><style>"
        "table { border-collapse: collapse; font-size: 12px; margin-bottom: 18px; }"
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }"
        "th { background-color: #f2f2f2; }"
        "h3 { margin-top: 14px; margin-bottom: 6px; }"
        "</style></head><body>"
        f"<p>Portfolio class summary for <b>{port_name}</b> on <b>{run_date}</b></p>"
        "<p>Units: notional in 1m RMB, risk in 1k RMB.</p>"
        "<h3>Notional Table (1m RMB)</h3>"
        + notional_df.to_html()
        + "<h3>Risk Table (1k RMB)</h3>"
        + risk_df.to_html()
        + "</body></html>"
    )
    send_html_by_smtp(EMAIL_QQ, NOTIFIERS, sub, html)


def main() -> None:
    """Run the class summary workflow."""
    args = parse_args()
    port_dir = Path(args.port_dir)
    run_date_str = args.run_date or infer_latest_run_date(port_dir, args.port_file)
    run_date = datetime.datetime.strptime(run_date_str, "%Y%m%d").date()

    port_path, strat_path = resolve_paths(port_dir, args.port_file, run_date_str)

    total_port = load_flat_json(port_path)
    pos_by_strat = load_flat_json(strat_path)
    hotmap = flatten_exchange_map(Path(args.hotmap))
    volscale_map = flatten_exchange_map(Path(args.commodities))

    strategy_positions: Dict[str, Dict[str, float]] = {
        **pos_by_strat,
        "TOTAL_PORT": total_port,
    }
    products = {
        product
        for product_pos in strategy_positions.values()
        for product in product_pos.keys()
    }

    product_metrics = build_product_metric_table(products, hotmap, run_date, volscale_map)
    product_metrics["notional_1m_per_lot"] = (
        product_metrics["notional_10k_per_lot"] / 100.0
    )
    product_metrics["risk_1k_per_lot"] = product_metrics["risk_per_lot"] / 1000.0

    strategy_order = get_strategy_order(args.port_file)
    detail_df = positions_to_detail_rows(strategy_positions, product_metrics)
    notional_df = build_two_level_table(detail_df, "notional_1m", strategy_order)
    risk_df = build_two_level_table(detail_df, "risk_1k", strategy_order)

    send_tables_email(
        run_date,
        args.port_file,
        notional_df,
        risk_df,
        email_notify=(not args.no_email),
    )

    print(f"Run date: {run_date_str}")
    print(f"Portfolio file: {port_path}")
    print(f"Strategy file:  {strat_path}")
    print(f"Email sent: {not args.no_email}")
    print("\nNotional table (1m RMB) with (level1, level2) columns:")
    print(notional_df.to_string())
    print("\nRisk table (1k RMB) with (level1, level2) columns:")
    print(risk_df.to_string())


if __name__ == "__main__":
    main()
