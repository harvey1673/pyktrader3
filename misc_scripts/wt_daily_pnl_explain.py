"""Build a day-level PnL explain report for a WonderTrader portfolio.

This utility combines three local data sources:

1. ``generated/outputs/<strategy>/positions.csv`` for code-level PnL snapshots.
2. ``generated/outputs/<strategy>/funds.csv`` for portfolio daily equity PnL.
3. ``process/paper_sim1/pos_by_strat_<port_file>_<date>.json`` for per-strategy
   product attribution, including manual trading rows when present.

The report is intended for operational attribution rather than accounting. Code
Pnl is measured as the two-day change in ``closeprofit + dynprofit`` per code,
then rolled to product level and allocated to strategies using the previous
day's strategy product targets when available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_PORT_DIR = Path(r"c:\dev\pyktrader3\process\paper_sim1")
DEFAULT_GROUP_DIR = Path(r"c:\dev\wtdev\deploy\cta_prod")
DEFAULT_PORT_FILE = "PTSIM1_FACTPORT1_hot"
DEFAULT_OUTPUT_STRATEGY = "PTSIM1_FACTPORT1"
DEFAULT_ANALYTICS_DIR = Path(r"c:\dev\data\analytics")


@dataclass(frozen=True)
class ExplainPaths:
    """Resolved file paths for one explain run."""

    port_dir: Path
    group_dir: Path
    port_file: str
    output_strategy: str
    run_date_str: str
    prev_date_str: str
    port_path: Path
    prev_port_path: Path
    strat_path: Path
    prev_strat_path: Path
    positions_csv: Path
    funds_csv: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build a daily PnL explain report by code, product, and strategy."
    )
    parser.add_argument(
        "run_date",
        nargs="?",
        help="Run date in YYYYMMDD format. Defaults to the latest date in positions.csv.",
    )
    parser.add_argument(
        "--port-dir",
        default=str(DEFAULT_PORT_DIR),
        help="Directory containing paper_sim1 daily portfolio files.",
    )
    parser.add_argument(
        "--group-dir",
        default=str(DEFAULT_GROUP_DIR),
        help="WonderTrader group directory that owns generated outputs.",
    )
    parser.add_argument(
        "--port-file",
        default=DEFAULT_PORT_FILE,
        help="Portfolio file stem, for example PTSIM1_FACTPORT1_hot.",
    )
    parser.add_argument(
        "--output-strategy",
        default=DEFAULT_OUTPUT_STRATEGY,
        help="Strategy id used under generated/outputs/<strategy>/.",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Optional output directory. Defaults to "
            "c:/dev/data/analytics/<port-file>_<run-date>."
        ),
    )
    return parser.parse_args()


def infer_available_dates(positions_csv: Path) -> List[str]:
    """Return sorted available dates from the positions snapshot file."""

    if not positions_csv.exists():
        raise FileNotFoundError(f"Positions file not found: {positions_csv}")

    dates = pd.read_csv(
        positions_csv,
        usecols=["date"],
        dtype={"date": "string"},
    )["date"]
    dates = dates.dropna().astype(str).str.strip()
    date_candidates = dates[dates.str.fullmatch(r"\d{8}")]
    valid_mask = pd.to_datetime(
        date_candidates,
        format="%Y%m%d",
        errors="coerce",
    ).notna()
    unique_dates = sorted(date_candidates[valid_mask].unique().tolist())
    if not unique_dates:
        raise ValueError(
            "No valid YYYYMMDD dates found in positions file: "
            f"{positions_csv}"
        )
    return unique_dates


def resolve_dates(positions_csv: Path, run_date: str | None) -> Tuple[str, str]:
    """Resolve the requested date and the previous available date."""

    available_dates = infer_available_dates(positions_csv)
    run_date_str = run_date or available_dates[-1]
    if run_date_str not in available_dates:
        raise ValueError(
            f"Date {run_date_str} not found in {positions_csv}. "
            f"Latest dates: {available_dates[-5:]}"
        )

    run_index = available_dates.index(run_date_str)
    if run_index == 0:
        raise ValueError(
            f"Date {run_date_str} has no previous snapshot in {positions_csv}."
        )
    return run_date_str, available_dates[run_index - 1]


def build_paths(
    port_dir: Path,
    group_dir: Path,
    port_file: str,
    output_strategy: str,
    run_date: str | None,
) -> ExplainPaths:
    """Resolve all inputs required by the report."""

    positions_csv = (
        group_dir
        / "generated"
        / "outputs"
        / output_strategy
        / "positions.csv"
    )
    funds_csv = (
        group_dir
        / "generated"
        / "outputs"
        / output_strategy
        / "funds.csv"
    )
    run_date_str, prev_date_str = resolve_dates(positions_csv, run_date)

    port_path = port_dir / f"{port_file}_{run_date_str}.json"
    prev_port_path = port_dir / f"{port_file}_{prev_date_str}.json"
    strat_path = port_dir / f"pos_by_strat_{port_file}_{run_date_str}.json"
    prev_strat_path = port_dir / f"pos_by_strat_{port_file}_{prev_date_str}.json"

    missing_paths = [
        path
        for path in [port_path, prev_port_path, strat_path, prev_strat_path, funds_csv]
        if not path.exists()
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")

    return ExplainPaths(
        port_dir=port_dir,
        group_dir=group_dir,
        port_file=port_file,
        output_strategy=output_strategy,
        run_date_str=run_date_str,
        prev_date_str=prev_date_str,
        port_path=port_path,
        prev_port_path=prev_port_path,
        strat_path=strat_path,
        prev_strat_path=prev_strat_path,
        positions_csv=positions_csv,
        funds_csv=funds_csv,
    )


def load_json(path: Path) -> Dict:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_strategy_name(name: str) -> str:
    """Return a compact display name for a strategy config filename."""

    display = name.replace(".json", "").replace(".csv", "")
    display = display.replace("PTSIM1_", "")
    display = display.replace("PT_", "")
    return display


def extract_product(code: str) -> str:
    """Extract the product token from a full contract code."""

    parts = str(code).split(".")
    if len(parts) >= 3:
        return parts[1]
    return str(code)


def load_positions_snapshot(
    positions_csv: Path,
    date_str: str,
) -> pd.DataFrame:
    """Load one day of code-level positions and PnL state."""

    positions = pd.read_csv(positions_csv)
    day_positions = positions[positions["date"] == int(date_str)].copy()
    day_positions["product"] = day_positions["code"].map(extract_product)
    day_positions["total_pnl_state"] = (
        day_positions["closeprofit"] + day_positions["dynprofit"]
    )
    return day_positions


def build_code_pnl_detail(paths: ExplainPaths) -> pd.DataFrame:
    """Build the per-code daily explain by differencing two snapshots."""

    prev_day = load_positions_snapshot(paths.positions_csv, paths.prev_date_str)
    curr_day = load_positions_snapshot(paths.positions_csv, paths.run_date_str)

    prev_day = prev_day.rename(
        columns={
            "volume": "prev_volume",
            "closeprofit": "prev_closeprofit",
            "dynprofit": "prev_dynprofit",
            "total_pnl_state": "prev_total_pnl_state",
        }
    )
    curr_day = curr_day.rename(
        columns={
            "volume": "curr_volume",
            "closeprofit": "curr_closeprofit",
            "dynprofit": "curr_dynprofit",
            "total_pnl_state": "curr_total_pnl_state",
        }
    )

    merge_columns = [
        "code",
        "product",
        "prev_volume",
        "prev_closeprofit",
        "prev_dynprofit",
        "prev_total_pnl_state",
    ]
    code_detail = curr_day[
        [
            "code",
            "product",
            "curr_volume",
            "curr_closeprofit",
            "curr_dynprofit",
            "curr_total_pnl_state",
        ]
    ].merge(prev_day[merge_columns], on=["code", "product"], how="outer")

    numeric_columns = [
        "curr_volume",
        "curr_closeprofit",
        "curr_dynprofit",
        "curr_total_pnl_state",
        "prev_volume",
        "prev_closeprofit",
        "prev_dynprofit",
        "prev_total_pnl_state",
    ]
    for column in numeric_columns:
        code_detail[column] = code_detail[column].fillna(0.0)

    code_detail["daily_pnl"] = (
        code_detail["curr_total_pnl_state"] - code_detail["prev_total_pnl_state"]
    )
    code_detail["position_change"] = (
        code_detail["curr_volume"] - code_detail["prev_volume"]
    )
    code_detail = code_detail.sort_values(
        by="daily_pnl", key=lambda series: series.abs(), ascending=False
    )
    return code_detail.reset_index(drop=True)


def build_product_pnl_detail(code_detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the code-level explain to product level."""

    product_detail = code_detail.groupby("product", as_index=False).agg(
        daily_pnl=("daily_pnl", "sum"),
        prev_volume=("prev_volume", "sum"),
        curr_volume=("curr_volume", "sum"),
        position_change=("position_change", "sum"),
    )
    product_detail = product_detail.sort_values(
        by="daily_pnl", key=lambda series: series.abs(), ascending=False
    )
    return product_detail.reset_index(drop=True)


def load_strategy_maps(path: Path) -> Dict[str, Dict[str, float]]:
    """Load strategy-to-product target mappings as floats."""

    raw_data = load_json(path)
    strategy_map: Dict[str, Dict[str, float]] = {}
    for strategy_name, product_map in raw_data.items():
        strategy_map[strategy_name] = {
            str(product): float(value)
            for product, value in product_map.items()
            if float(value) != 0.0
        }
    return strategy_map


def choose_basis_map(
    prev_strategy_map: Dict[str, float],
    curr_strategy_map: Dict[str, float],
) -> Tuple[str, Dict[str, float], float]:
    """Choose a stable attribution basis for one product.

    Prefer previous-day targets because the daily PnL mostly belongs to the
    position carried into the session. If the product is new, fall back to
    current-day targets. When the net target is zero, use gross exposure.
    """

    prev_net = sum(prev_strategy_map.values())
    prev_gross = sum(abs(value) for value in prev_strategy_map.values())
    curr_net = sum(curr_strategy_map.values())
    curr_gross = sum(abs(value) for value in curr_strategy_map.values())

    if abs(prev_net) > 0:
        return "prev_net", prev_strategy_map, prev_net
    if prev_gross > 0:
        gross_map = {
            strategy: abs(value)
            for strategy, value in prev_strategy_map.items()
            if abs(value) > 0
        }
        return "prev_gross", gross_map, prev_gross
    if abs(curr_net) > 0:
        return "curr_net", curr_strategy_map, curr_net
    if curr_gross > 0:
        gross_map = {
            strategy: abs(value)
            for strategy, value in curr_strategy_map.items()
            if abs(value) > 0
        }
        return "curr_gross", gross_map, curr_gross
    return "none", {}, 0.0


def build_strategy_attribution(
    paths: ExplainPaths,
    product_detail: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate product daily PnL to strategies using pos_by_strat snapshots."""

    prev_total = load_json(paths.prev_port_path)
    curr_total = load_json(paths.port_path)
    prev_by_strat = load_strategy_maps(paths.prev_strat_path)
    curr_by_strat = load_strategy_maps(paths.strat_path)

    products = set(product_detail["product"].tolist())
    products.update(prev_total.keys())
    products.update(curr_total.keys())

    rows: List[Dict[str, object]] = []
    for product in sorted(products):
        product_pnl = float(
            product_detail.loc[
                product_detail["product"] == product, "daily_pnl"
            ].sum()
        )
        prev_strategy_product = {
            strategy: values.get(product, 0.0)
            for strategy, values in prev_by_strat.items()
            if values.get(product, 0.0) != 0.0
        }
        curr_strategy_product = {
            strategy: values.get(product, 0.0)
            for strategy, values in curr_by_strat.items()
            if values.get(product, 0.0) != 0.0
        }
        basis_name, basis_map, basis_total = choose_basis_map(
            prev_strategy_product,
            curr_strategy_product,
        )

        if basis_name == "none" or basis_total == 0:
            if product_pnl != 0:
                rows.append(
                    {
                        "product": product,
                        "strategy_file": "UNALLOCATED",
                        "strategy": "UNALLOCATED",
                        "basis": basis_name,
                        "weight": 1.0,
                        "attributed_pnl": product_pnl,
                        "prev_strategy_lots": 0.0,
                        "curr_strategy_lots": 0.0,
                        "prev_total_lots": float(prev_total.get(product, 0.0)),
                        "curr_total_lots": float(curr_total.get(product, 0.0)),
                        "product_daily_pnl": product_pnl,
                    }
                )
            continue

        for strategy_name, basis_value in basis_map.items():
            weight = basis_value / basis_total
            rows.append(
                {
                    "product": product,
                    "strategy_file": strategy_name,
                    "strategy": normalize_strategy_name(strategy_name),
                    "basis": basis_name,
                    "weight": weight,
                    "attributed_pnl": product_pnl * weight,
                    "prev_strategy_lots": float(
                        prev_strategy_product.get(strategy_name, 0.0)
                    ),
                    "curr_strategy_lots": float(
                        curr_strategy_product.get(strategy_name, 0.0)
                    ),
                    "prev_total_lots": float(prev_total.get(product, 0.0)),
                    "curr_total_lots": float(curr_total.get(product, 0.0)),
                    "product_daily_pnl": product_pnl,
                }
            )

    strategy_detail = pd.DataFrame(rows)
    if strategy_detail.empty:
        strategy_summary = pd.DataFrame(
            columns=["strategy", "strategy_file", "attributed_pnl"]
        )
        return strategy_detail, strategy_summary

    strategy_detail = strategy_detail.sort_values(
        by="attributed_pnl",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)
    strategy_summary = strategy_detail.groupby(
        ["strategy", "strategy_file"], as_index=False
    ).agg(attributed_pnl=("attributed_pnl", "sum"))
    strategy_summary = strategy_summary.sort_values(
        by="attributed_pnl",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)
    return strategy_detail, strategy_summary


def load_daily_funds_detail(paths: ExplainPaths) -> Dict[str, float]:
    """Load the portfolio daily equity PnL and fee change from funds.csv."""

    funds = pd.read_csv(paths.funds_csv).rename(
        columns={"positionprofit": "dynprofit"}
    )
    required_dates = {int(paths.prev_date_str), int(paths.run_date_str)}
    funds = funds[funds["date"].isin(required_dates)].copy()
    if len(funds) != 2:
        raise ValueError(
            "Could not find both dates in funds.csv for daily explain: "
            f"{paths.prev_date_str}, {paths.run_date_str}"
        )

    funds = funds.sort_values("date").reset_index(drop=True)
    prev_row = funds.iloc[0]
    curr_row = funds.iloc[1]
    return {
        "daily_equity_pnl": float(curr_row["dynbalance"] - prev_row["dynbalance"]),
        "daily_fee": float(curr_row["fee"] - prev_row["fee"]),
        "daily_close_pnl": float(
            curr_row["closeprofit"] - prev_row["closeprofit"]
        ),
        "curr_dynbalance": float(curr_row["dynbalance"]),
        "curr_dynprofit": float(curr_row["dynprofit"]),
        "curr_closeprofit": float(curr_row["closeprofit"]),
    }


def default_out_dir(port_dir: Path, port_file: str, run_date_str: str) -> Path:
    """Return the default report output directory."""

    _ = port_dir
    return DEFAULT_ANALYTICS_DIR / f"{port_file}_{run_date_str}"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact markdown table without extra dependencies."""

    if frame.empty:
        return "(empty)"

    headers = [str(column) for column in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append(values)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def build_markdown_report(
    paths: ExplainPaths,
    funds_detail: Dict[str, float],
    code_detail: pd.DataFrame,
    product_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
) -> str:
    """Build a short markdown summary for the explain run."""

    code_explained = float(code_detail["daily_pnl"].sum())
    residual = funds_detail["daily_equity_pnl"] - code_explained
    product_preview = product_detail.head(15).copy()
    strategy_preview = strategy_summary.head(15).copy()

    lines = [
        f"# Daily PnL Explain: {paths.port_file} {paths.run_date_str}",
        "",
        f"- Previous date: {paths.prev_date_str}",
        f"- Output strategy: {paths.output_strategy}",
        f"- Portfolio daily equity PnL: {funds_detail['daily_equity_pnl']:.2f}",
        f"- Daily close PnL: {funds_detail['daily_close_pnl']:.2f}",
        f"- Daily fees: {funds_detail['daily_fee']:.2f}",
        f"- Sum of code explain PnL: {code_explained:.2f}",
        f"- Residual not explained by code rows: {residual:.2f}",
        "",
        "## Top Product PnL",
        "",
        frame_to_markdown(product_preview),
        "",
        "## Top Strategy Attribution",
        "",
        frame_to_markdown(strategy_preview),
        "",
        "## Notes",
        "",
        "- Strategy attribution uses previous-day pos_by_strat weights when available.",
        "- Manual trading is included if it appears in pos_by_strat.",
        "- Residual PnL captures fees, cash movements, and any mismatch between",
        "  paper target attribution and runtime code-level state.",
    ]
    return "\n".join(lines)


def save_outputs(
    out_dir: Path,
    markdown_report: str,
    code_detail: pd.DataFrame,
    product_detail: pd.DataFrame,
    strategy_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    summary: Dict[str, float | str],
) -> None:
    """Persist the report and all supporting tables."""

    out_dir.mkdir(parents=True, exist_ok=True)
    code_detail.to_csv(out_dir / "code_pnl_detail.csv", index=False)
    product_detail.to_csv(out_dir / "product_pnl_detail.csv", index=False)
    strategy_detail.to_csv(out_dir / "strategy_pnl_detail.csv", index=False)
    strategy_summary.to_csv(out_dir / "strategy_pnl_summary.csv", index=False)
    with (out_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown_report)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def print_console_summary(
    paths: ExplainPaths,
    funds_detail: Dict[str, float],
    product_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
) -> None:
    """Print a compact console summary."""

    print(f"PnL explain for {paths.port_file} on {paths.run_date_str}")
    print(f"Previous date: {paths.prev_date_str}")
    print(f"Daily equity PnL: {funds_detail['daily_equity_pnl']:.2f}")
    print(f"Daily fees: {funds_detail['daily_fee']:.2f}")

    print("\nTop product PnL")
    print(product_detail.head(10).to_string(index=False))

    print("\nTop strategy attribution")
    print(strategy_summary.head(10).to_string(index=False))


def main() -> None:
    """Run the daily PnL explain workflow."""

    args = parse_args()
    port_dir = Path(args.port_dir).resolve()
    group_dir = Path(args.group_dir).resolve()
    paths = build_paths(
        port_dir=port_dir,
        group_dir=group_dir,
        port_file=args.port_file,
        output_strategy=args.output_strategy,
        run_date=args.run_date,
    )

    code_detail = build_code_pnl_detail(paths)
    product_detail = build_product_pnl_detail(code_detail)
    strategy_detail, strategy_summary = build_strategy_attribution(
        paths,
        product_detail,
    )
    funds_detail = load_daily_funds_detail(paths)

    explained_pnl = float(code_detail["daily_pnl"].sum())
    residual = funds_detail["daily_equity_pnl"] - explained_pnl
    summary = {
        "run_date": paths.run_date_str,
        "prev_date": paths.prev_date_str,
        "portfolio_daily_equity_pnl": funds_detail["daily_equity_pnl"],
        "portfolio_daily_fee": funds_detail["daily_fee"],
        "explained_code_pnl": explained_pnl,
        "residual_pnl": residual,
        "manual_trading_pnl": float(
            strategy_summary.loc[
                strategy_summary["strategy_file"] == "PTSIM1_MANUEL_TRADING.csv",
                "attributed_pnl",
            ].sum()
        ),
    }

    markdown_report = build_markdown_report(
        paths,
        funds_detail,
        code_detail,
        product_detail,
        strategy_summary,
    )
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else default_out_dir(port_dir, args.port_file, paths.run_date_str)
    )
    save_outputs(
        out_dir,
        markdown_report,
        code_detail,
        product_detail,
        strategy_detail,
        strategy_summary,
        summary,
    )
    print_console_summary(paths, funds_detail, product_detail, strategy_summary)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
