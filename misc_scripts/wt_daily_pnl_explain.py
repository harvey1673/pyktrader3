"""Build a day-level PnL explain report for a WonderTrader portfolio.

This utility combines three local data sources:

1. ``generated/outputs/<strategy>/positions.csv`` for EOD unrealized PnL.
2. ``generated/outputs/<strategy>/closes.csv`` for realized PnL.
3. ``generated/outputs/<strategy>/trades.csv`` for execution fees.
4. ``generated/outputs/<strategy>/funds.csv`` for authoritative portfolio PnL.
5. ``process/paper_sim1/pos_by_strat_<port_file>_<date>.json`` for per-strategy
   product attribution, including manual trading rows when present.

The report is intended for operational attribution rather than accounting. Code
PnL is measured as realized close PnL plus the change in EOD unrealized PnL,
less execution fees. Night-session events are assigned to the next available
trading date. Product PnL is then allocated to strategies using signed prior-day
strategy product targets when available.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PORT_DIR = Path(r"c:\dev\pyktrader3\process\paper_sim1")
DEFAULT_GROUP_DIR = Path(r"c:\dev\wtdev\deploy\cta_prod")
DEFAULT_PORT_FILE = "PTSIM1_FACTPORT1_hot"
DEFAULT_OUTPUT_STRATEGY = "PTSIM1_FACTPORT1"
DEFAULT_TRADER_CHANNEL = "cyqh_ctp"
DEFAULT_ANALYTICS_DIR = Path(r"c:\dev\data\analytics")
NIGHT_SESSION_START_HHMM = 1800
RECONCILIATION_TOLERANCE = 0.01


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
    closes_csv: Path
    trades_csv: Path
    rtdata_json: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build a daily PnL explain report by code, product, and strategy."
    )
    parser.add_argument(
        "run_date",
        nargs="?",
        help=(
            "Run date in YYYYMMDD format. Default: current CHN workday before "
            "21:00, next workday after 21:00, or previous workday on holidays/weekends."
        ),
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
        "--trader-channel",
        default=DEFAULT_TRADER_CHANNEL,
        help="Trader channel used for generated/traders/<channel>/rtdata.json.",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Optional output directory. Defaults to "
            "c:/dev/data/analytics/<port-file>_<run-date>."
        ),
    )
    email_group = parser.add_mutually_exclusive_group()
    email_group.add_argument(
        "--email",
        action="store_true",
        help="Send email even when the shared EMAIL_NOTIFY setting is disabled.",
    )
    email_group.add_argument(
        "--no-email",
        action="store_true",
        help="Build the report without sending the configured notification email.",
    )
    return parser.parse_args()


def resolve_default_run_date(now: datetime | None = None) -> date:
    """Resolve the default China trading date using the 21:00 session cutoff."""

    from pycmqlib3.utility.misc import CHN_Holidays, day_shift, is_workday

    current = datetime.now() if now is None else now
    current_date = current.date()
    if not is_workday(current_date, "CHN"):
        return day_shift(current_date, "-1b", CHN_Holidays)
    if current.time() < time(21, 0):
        return current_date
    return day_shift(current_date, "1b", CHN_Holidays)


def infer_available_dates(positions_csv: Path) -> List[str]:
    """Return sorted available dates from the positions snapshot file."""

    if not positions_csv.exists():
        raise FileNotFoundError(f"Positions file not found: {positions_csv}")

    dates = pd.read_csv(
        positions_csv,
        usecols=["date"],
        dtype={"date": "string"},
        index_col=False,
    )["date"]
    dates = dates.dropna().astype(str).str.strip()
    dates = dates.str.replace(r"\.0$", "", regex=True)
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
    trader_channel: str = DEFAULT_TRADER_CHANNEL,
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
    closes_csv = (
        group_dir
        / "generated"
        / "outputs"
        / output_strategy
        / "closes.csv"
    )
    trades_csv = (
        group_dir
        / "generated"
        / "outputs"
        / output_strategy
        / "trades.csv"
    )
    rtdata_json = group_dir / "generated" / "traders" / trader_channel / "rtdata.json"
    run_date_str, prev_date_str = resolve_dates(positions_csv, run_date)

    port_path = port_dir / f"{port_file}_{run_date_str}.json"
    prev_port_path = port_dir / f"{port_file}_{prev_date_str}.json"
    strat_path = port_dir / f"pos_by_strat_{port_file}_{run_date_str}.json"
    prev_strat_path = port_dir / f"pos_by_strat_{port_file}_{prev_date_str}.json"

    missing_paths = [
        path
        for path in [
            port_path,
            prev_port_path,
            strat_path,
            prev_strat_path,
            funds_csv,
            closes_csv,
            trades_csv,
            rtdata_json,
        ]
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
        closes_csv=closes_csv,
        trades_csv=trades_csv,
        rtdata_json=rtdata_json,
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


def load_positions_snapshot(positions_csv: Path, date_str: str) -> pd.DataFrame:
    """Load one EOD position snapshot grouped by contract code."""

    positions = pd.read_csv(
        positions_csv,
        usecols=["date", "code", "volume", "dynprofit"],
        index_col=False,
    )
    day_positions = positions[positions["date"] == int(date_str)].copy()
    return day_positions.groupby("code", as_index=False).agg(
        volume=("volume", "sum"),
        dynprofit=("dynprofit", "sum"),
    )


def event_trading_date(
    timestamp: object,
    available_dates: Iterable[str],
) -> str | None:
    """Map a WTPY event timestamp to its EOD trading date.

    WTPY records night-session events with the calendar date on which the
    session starts. Those events belong to the next available EOD trading date.
    Events after the latest completed EOD date deliberately return ``None``.
    """

    mapper = build_event_trading_date_mapper(available_dates)
    return mapper(timestamp)


def build_event_trading_date_mapper(available_dates: Iterable[str]):
    """Build an efficient timestamp-to-trading-date mapper."""

    ordered_dates = tuple(sorted({str(value) for value in available_dates}))

    def map_timestamp(timestamp: object) -> str | None:
        raw_value = str(timestamp).strip()
        if raw_value.endswith(".0"):
            raw_value = raw_value[:-2]
        if len(raw_value) < 12 or not raw_value[:12].isdigit():
            return None

        calendar_date = raw_value[:8]
        hhmm = int(raw_value[8:12])
        if hhmm >= NIGHT_SESSION_START_HHMM:
            date_index = bisect_right(ordered_dates, calendar_date)
        else:
            date_index = bisect_left(ordered_dates, calendar_date)
        if date_index >= len(ordered_dates):
            return None
        return ordered_dates[date_index]

    return map_timestamp


def event_session(timestamp: object) -> str | None:
    """Classify a WTPY event timestamp as day or night session."""

    raw_value = str(timestamp).strip()
    if raw_value.endswith(".0"):
        raw_value = raw_value[:-2]
    if len(raw_value) < 12 or not raw_value[:12].isdigit():
        return None
    hhmm = int(raw_value[8:12])
    return "night" if hhmm >= NIGHT_SESSION_START_HHMM else "day"


def load_daily_close_events(
    closes_csv: Path,
    date_str: str,
    available_dates: Iterable[str],
) -> pd.DataFrame:
    """Return realized close PnL assigned to one trading date by code."""

    closes = pd.read_csv(
        closes_csv,
        usecols=["code", "closetime", "qty", "profit"],
        index_col=False,
    )
    date_mapper = build_event_trading_date_mapper(available_dates)
    closes["trading_date"] = closes["closetime"].map(date_mapper)
    closes = closes[closes["trading_date"] == date_str].copy()
    if closes.empty:
        return pd.DataFrame(columns=["code", "realized_pnl", "close_qty"])
    closes["close_qty"] = closes["qty"].abs()
    return closes.groupby("code", as_index=False).agg(
        realized_pnl=("profit", "sum"),
        close_qty=("close_qty", "sum"),
    )


def load_daily_trade_events(
    trades_csv: Path,
    date_str: str,
    available_dates: Iterable[str],
) -> pd.DataFrame:
    """Return execution fees and traded quantity assigned to one trading date."""

    output_columns = [
        "code",
        "fee",
        "trade_qty",
        "night_avg_executed_price",
        "night_executed_volume",
        "day_avg_executed_price",
        "day_executed_volume",
    ]
    trades = pd.read_csv(
        trades_csv,
        usecols=["code", "time", "price", "qty", "fee"],
        index_col=False,
    )
    date_mapper = build_event_trading_date_mapper(available_dates)
    trades["trading_date"] = trades["time"].map(date_mapper)
    trades = trades[trades["trading_date"] == date_str].copy()
    if trades.empty:
        return pd.DataFrame(columns=output_columns)
    trades["session"] = trades["time"].map(event_session)
    trades["trade_qty"] = trades["qty"].abs()
    trades["executed_notional"] = trades["price"] * trades["trade_qty"]
    totals = trades.groupby("code", as_index=False).agg(
        fee=("fee", "sum"),
        trade_qty=("trade_qty", "sum"),
    )
    for session_name in ("night", "day"):
        session_rows = trades[trades["session"] == session_name]
        session_totals = session_rows.groupby("code", as_index=False).agg(
            executed_volume=("trade_qty", "sum"),
            executed_notional=("executed_notional", "sum"),
        )
        volume_column = f"{session_name}_executed_volume"
        price_column = f"{session_name}_avg_executed_price"
        session_totals[price_column] = np.where(
            session_totals["executed_volume"] != 0,
            session_totals["executed_notional"] / session_totals["executed_volume"],
            np.nan,
        )
        session_totals = session_totals.rename(
            columns={"executed_volume": volume_column}
        )[["code", price_column, volume_column]]
        totals = totals.merge(session_totals, on="code", how="left")
    return totals.reindex(columns=output_columns)


def build_code_pnl_detail(paths: ExplainPaths) -> pd.DataFrame:
    """Build a roll-safe per-code daily PnL ledger."""

    prev_day = load_positions_snapshot(paths.positions_csv, paths.prev_date_str)
    curr_day = load_positions_snapshot(paths.positions_csv, paths.run_date_str)
    prev_day = prev_day.rename(
        columns={"volume": "prev_volume", "dynprofit": "prev_dynprofit"}
    )
    curr_day = curr_day.rename(
        columns={"volume": "curr_volume", "dynprofit": "curr_dynprofit"}
    )

    available_dates = infer_available_dates(paths.funds_csv)
    close_events = load_daily_close_events(
        paths.closes_csv,
        paths.run_date_str,
        available_dates,
    )
    trade_events = load_daily_trade_events(
        paths.trades_csv,
        paths.run_date_str,
        available_dates,
    )

    code_detail = prev_day.merge(curr_day, on="code", how="outer")
    code_detail = code_detail.merge(close_events, on="code", how="outer")
    code_detail = code_detail.merge(trade_events, on="code", how="outer")
    code_detail["product"] = code_detail["code"].map(extract_product)
    code_detail["contract"] = code_detail["code"]

    numeric_columns = [
        "prev_volume",
        "prev_dynprofit",
        "curr_volume",
        "curr_dynprofit",
        "realized_pnl",
        "close_qty",
        "fee",
        "trade_qty",
        "night_executed_volume",
        "day_executed_volume",
    ]
    for column in numeric_columns:
        code_detail[column] = code_detail[column].fillna(0.0)

    code_detail["unrealized_pnl_change"] = (
        code_detail["curr_dynprofit"] - code_detail["prev_dynprofit"]
    )
    code_detail["gross_pnl"] = (
        code_detail["realized_pnl"] + code_detail["unrealized_pnl_change"]
    )
    code_detail["daily_pnl"] = code_detail["gross_pnl"] - code_detail["fee"]
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
        realized_pnl=("realized_pnl", "sum"),
        unrealized_pnl_change=("unrealized_pnl_change", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        fee=("fee", "sum"),
        daily_pnl=("daily_pnl", "sum"),
        prev_volume=("prev_volume", "sum"),
        curr_volume=("curr_volume", "sum"),
        position_change=("position_change", "sum"),
        trade_qty=("trade_qty", "sum"),
        close_qty=("close_qty", "sum"),
    )
    product_detail = product_detail.sort_values(
        by="daily_pnl", key=lambda series: series.abs(), ascending=False
    )
    return product_detail.reset_index(drop=True)


def reconcile_product_pnl(
    product_detail: pd.DataFrame,
    daily_equity_pnl: float,
) -> Tuple[pd.DataFrame, float]:
    """Reconcile product net PnL to funds.csv with an explicit adjustment row."""

    product_detail = product_detail.copy()
    explained_pnl = float(product_detail["daily_pnl"].sum())
    residual = float(daily_equity_pnl - explained_pnl)
    if abs(residual) <= RECONCILIATION_TOLERANCE:
        return product_detail, residual

    adjustment = {column: 0.0 for column in product_detail.columns}
    adjustment["product"] = "UNEXPLAINED_ADJUSTMENT"
    adjustment["gross_pnl"] = residual
    adjustment["daily_pnl"] = residual
    product_detail = pd.concat(
        [product_detail, pd.DataFrame([adjustment])],
        ignore_index=True,
    )
    product_detail = product_detail.sort_values(
        by="daily_pnl", key=lambda series: series.abs(), ascending=False
    ).reset_index(drop=True)
    return product_detail, residual


def load_strategy_maps_checked(
    path: Path,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Load strategy targets and report non-numeric/non-finite values."""

    raw_data = load_json(path)
    strategy_map: Dict[str, Dict[str, float]] = {}
    issues: List[Dict[str, object]] = []
    for strategy_name, product_map in raw_data.items():
        clean_product_map: Dict[str, float] = {}
        for product, raw_value in product_map.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                issues.append(
                    {
                        "file": str(path),
                        "strategy_file": str(strategy_name),
                        "product": str(product),
                        "raw_value": str(raw_value),
                        "issue": "non_finite_strategy_position",
                    }
                )
                continue
            if value != 0.0:
                clean_product_map[str(product)] = value
        strategy_map[str(strategy_name)] = clean_product_map
    issue_columns = ["file", "strategy_file", "product", "raw_value", "issue"]
    return strategy_map, pd.DataFrame(issues, columns=issue_columns)


def load_strategy_maps(path: Path) -> Dict[str, Dict[str, float]]:
    """Load finite strategy-to-product target mappings as floats."""

    strategy_map, _ = load_strategy_maps_checked(path)
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

    if abs(prev_net) > 1e-12:
        return "prev_net", prev_strategy_map, prev_net
    if prev_gross > 1e-12:
        return "prev_offsetting", {}, 0.0
    if abs(curr_net) > 1e-12:
        return "curr_net", curr_strategy_map, curr_net
    if curr_gross > 1e-12:
        return "curr_offsetting", {}, 0.0
    return "none", {}, 0.0


def build_strategy_attribution(
    paths: ExplainPaths,
    product_detail: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Allocate product daily PnL to strategies using pos_by_strat snapshots."""

    prev_total = load_json(paths.prev_port_path)
    curr_total = load_json(paths.port_path)
    prev_by_strat, prev_issues = load_strategy_maps_checked(paths.prev_strat_path)
    curr_by_strat, curr_issues = load_strategy_maps_checked(paths.strat_path)
    input_issues = pd.concat([prev_issues, curr_issues], ignore_index=True)
    prev_invalid_products = (
        set(prev_issues["product"].tolist()) if not prev_issues.empty else set()
    )
    curr_invalid_products = (
        set(curr_issues["product"].tolist()) if not curr_issues.empty else set()
    )

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
        prev_strategy_sum = float(sum(prev_strategy_product.values()))
        curr_strategy_sum = float(sum(curr_strategy_product.values()))
        product_row = product_detail.loc[product_detail["product"] == product]
        prev_runtime_lots = float(product_row["prev_volume"].sum())
        curr_runtime_lots = float(product_row["curr_volume"].sum())

        prev_strategy_gross = float(
            sum(abs(value) for value in prev_strategy_product.values())
        )
        curr_strategy_gross = float(
            sum(abs(value) for value in curr_strategy_product.values())
        )
        if product in prev_invalid_products:
            basis_name, basis_map, allocation_denominator = (
                "prev_invalid",
                {},
                0.0,
            )
        elif prev_strategy_gross > 1e-12 and abs(prev_runtime_lots) > 1e-12:
            basis_name, basis_map, allocation_denominator = (
                "prev_runtime",
                prev_strategy_product,
                prev_runtime_lots,
            )
        elif not prev_strategy_product and product in curr_invalid_products:
            basis_name, basis_map, allocation_denominator = (
                "curr_invalid",
                {},
                0.0,
            )
        elif curr_strategy_gross > 1e-12 and abs(curr_runtime_lots) > 1e-12:
            basis_name, basis_map, allocation_denominator = (
                "curr_runtime",
                curr_strategy_product,
                curr_runtime_lots,
            )
        elif prev_strategy_gross > 1e-12 and abs(prev_strategy_sum) <= 1e-12:
            basis_name, basis_map, allocation_denominator = (
                "prev_offsetting",
                {},
                0.0,
            )
        elif curr_strategy_gross > 1e-12 and abs(curr_strategy_sum) <= 1e-12:
            basis_name, basis_map, allocation_denominator = (
                "curr_offsetting",
                {},
                0.0,
            )
        else:
            basis_name, basis_map, allocation_denominator = "none", {}, 0.0

        if not basis_map or allocation_denominator == 0:
            if product_pnl != 0:
                if "invalid" in basis_name:
                    unallocated_name = "UNALLOCATED_INVALID_INPUT"
                    allocation_status = "invalid_input"
                elif "offsetting" in basis_name:
                    unallocated_name = "UNALLOCATED_OFFSETTING"
                    allocation_status = "offsetting_positions"
                else:
                    unallocated_name = "UNALLOCATED"
                    allocation_status = "missing_position_basis"
                rows.append(
                    {
                        "product": product,
                        "strategy_file": unallocated_name,
                        "strategy": unallocated_name,
                        "basis": basis_name,
                        "allocation_status": allocation_status,
                        "allocation_denominator": allocation_denominator,
                        "weight": 1.0,
                        "attributed_pnl": product_pnl,
                        "prev_strategy_lots": 0.0,
                        "curr_strategy_lots": 0.0,
                        "prev_strategy_sum": prev_strategy_sum,
                        "curr_strategy_sum": curr_strategy_sum,
                        "prev_total_lots": float(prev_total.get(product, 0.0)),
                        "curr_total_lots": float(curr_total.get(product, 0.0)),
                        "prev_runtime_lots": prev_runtime_lots,
                        "curr_runtime_lots": curr_runtime_lots,
                        "product_daily_pnl": product_pnl,
                    }
                )
            continue

        allocated_product_pnl = 0.0
        for strategy_name, basis_value in basis_map.items():
            weight = basis_value / allocation_denominator
            strategy_pnl = product_pnl * weight
            allocated_product_pnl += strategy_pnl
            rows.append(
                {
                    "product": product,
                    "strategy_file": strategy_name,
                    "strategy": normalize_strategy_name(strategy_name),
                    "basis": basis_name,
                    "allocation_status": "allocated",
                    "allocation_denominator": allocation_denominator,
                    "weight": weight,
                    "attributed_pnl": strategy_pnl,
                    "prev_strategy_lots": float(
                        prev_strategy_product.get(strategy_name, 0.0)
                    ),
                    "curr_strategy_lots": float(
                        curr_strategy_product.get(strategy_name, 0.0)
                    ),
                    "prev_strategy_sum": prev_strategy_sum,
                    "curr_strategy_sum": curr_strategy_sum,
                    "prev_total_lots": float(prev_total.get(product, 0.0)),
                    "curr_total_lots": float(curr_total.get(product, 0.0)),
                    "prev_runtime_lots": prev_runtime_lots,
                    "curr_runtime_lots": curr_runtime_lots,
                    "product_daily_pnl": product_pnl,
                }
            )

        allocation_residual = product_pnl - allocated_product_pnl
        if abs(allocation_residual) > RECONCILIATION_TOLERANCE:
            rows.append(
                {
                    "product": product,
                    "strategy_file": "UNALLOCATED_POSITION_MISMATCH",
                    "strategy": "UNALLOCATED_POSITION_MISMATCH",
                    "basis": basis_name,
                    "allocation_status": "runtime_position_mismatch",
                    "allocation_denominator": allocation_denominator,
                    "weight": allocation_residual / product_pnl
                    if product_pnl != 0
                    else 0.0,
                    "attributed_pnl": allocation_residual,
                    "prev_strategy_lots": 0.0,
                    "curr_strategy_lots": 0.0,
                    "prev_strategy_sum": prev_strategy_sum,
                    "curr_strategy_sum": curr_strategy_sum,
                    "prev_total_lots": float(prev_total.get(product, 0.0)),
                    "curr_total_lots": float(curr_total.get(product, 0.0)),
                    "prev_runtime_lots": prev_runtime_lots,
                    "curr_runtime_lots": curr_runtime_lots,
                    "product_daily_pnl": product_pnl,
                }
            )

    strategy_detail = pd.DataFrame(rows)
    if strategy_detail.empty:
        strategy_summary = pd.DataFrame(
            columns=["strategy", "strategy_file", "attributed_pnl"]
        )
        return strategy_detail, strategy_summary, input_issues

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
    return strategy_detail, strategy_summary, input_issues


def load_daily_funds_detail(paths: ExplainPaths) -> Dict[str, float]:
    """Load the portfolio daily equity PnL and fee change from funds.csv."""

    funds = pd.read_csv(paths.funds_csv, index_col=False).rename(
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


def load_rtdata_current_state(
    rtdata_json: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Load current exchange-reported account funds and positions read-only."""

    raw_data = load_json(rtdata_json)
    account_rows: List[Dict[str, object]] = []
    for currency, values in raw_data.get("funds", {}).items():
        account_row: Dict[str, object] = {"currency": str(currency)}
        for field in [
            "prebalance",
            "balance",
            "closeprofit",
            "dynprofit",
            "margin",
            "fee",
            "available",
            "deposit",
            "withdraw",
        ]:
            account_row[field] = float(values.get(field, 0.0))
        balance = float(account_row["balance"])
        account_row["margin_to_balance"] = (
            float(account_row["margin"]) / balance if balance != 0 else np.nan
        )
        account_rows.append(account_row)

    position_rows: List[Dict[str, object]] = []
    for item in raw_data.get("positions", []):
        contract = str(item.get("code", ""))
        long_position = item.get("long", {})
        short_position = item.get("short", {})
        long_new = float(long_position.get("newvol", 0.0))
        long_previous = float(long_position.get("prevol", 0.0))
        short_new = float(short_position.get("newvol", 0.0))
        short_previous = float(short_position.get("prevol", 0.0))
        long_total = long_new + long_previous
        short_total = short_new + short_previous
        if long_total == 0 and short_total == 0:
            continue
        position_rows.append(
            {
                "contract": contract,
                "product": extract_product(contract),
                "long_position": long_total,
                "short_position": short_total,
                "net_position": long_total - short_total,
                "long_today": long_new,
                "long_previous": long_previous,
                "short_today": short_new,
                "short_previous": short_previous,
                "long_available": float(long_position.get("newavail", 0.0))
                + float(long_position.get("preavail", 0.0)),
                "short_available": float(short_position.get("newavail", 0.0))
                + float(short_position.get("preavail", 0.0)),
            }
        )

    account_detail = pd.DataFrame(account_rows)
    current_positions = pd.DataFrame(position_rows)
    if not current_positions.empty:
        current_positions = current_positions.sort_values(
            ["product", "contract"]
        ).reset_index(drop=True)
    metadata = {
        "rtdata_path": str(rtdata_json),
        "rtdata_last_modified": datetime.fromtimestamp(
            rtdata_json.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "rtdata_position_count": int(len(current_positions)),
    }
    return account_detail, current_positions, metadata


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
    input_issues: pd.DataFrame,
    ledger_residual: float,
    account_detail: pd.DataFrame,
    current_positions: pd.DataFrame,
    rtdata_metadata: Dict[str, object],
) -> str:
    """Build a short markdown summary for the explain run."""

    ledger_net_pnl = float(code_detail["daily_pnl"].sum())
    ledger_gross_pnl = float(code_detail["gross_pnl"].sum())
    ledger_fees = float(code_detail["fee"].sum())
    unallocated_pnl = float(
        strategy_summary.loc[
            strategy_summary["strategy_file"].str.startswith("UNALLOCATED"),
            "attributed_pnl",
        ].sum()
    )
    product_table = product_detail.reindex(
        columns=[
            "product",
            "daily_pnl",
            "fee",
            "prev_volume",
            "curr_volume",
            "position_change",
        ]
    )
    contract_table = code_detail.reindex(
        columns=[
            "contract",
            "product",
            "daily_pnl",
            "fee",
            "prev_volume",
            "curr_volume",
            "position_change",
            "night_avg_executed_price",
            "night_executed_volume",
            "day_avg_executed_price",
            "day_executed_volume",
        ]
    )

    lines = [
        f"# Daily PnL Explain: {paths.port_file} {paths.run_date_str}",
        "",
        f"- Previous date: {paths.prev_date_str}",
        f"- Output strategy: {paths.output_strategy}",
        f"- Portfolio daily equity PnL: {funds_detail['daily_equity_pnl']:.2f}",
        f"- Daily close PnL: {funds_detail['daily_close_pnl']:.2f}",
        f"- Daily fees from funds: {funds_detail['daily_fee']:.2f}",
        f"- Gross ledger PnL: {ledger_gross_pnl:.2f}",
        f"- Execution fees assigned to date: {ledger_fees:.2f}",
        f"- Net ledger PnL: {ledger_net_pnl:.2f}",
        f"- Ledger reconciliation adjustment: {ledger_residual:.2f}",
        f"- Unallocated strategy PnL: {unallocated_pnl:.2f}",
        f"- Input data-quality issues: {len(input_issues)}",
        f"- Current rtdata timestamp: {rtdata_metadata['rtdata_last_modified']}",
        "",
        "## Product PnL",
        "",
        frame_to_markdown(product_table),
        "",
        "## Contract PnL and Exchange Executions",
        "",
        frame_to_markdown(contract_table),
        "",
        "## Strategy Attribution",
        "",
        frame_to_markdown(strategy_summary),
        "",
        "## Current Exchange-Reported Account",
        "",
        frame_to_markdown(account_detail),
        "",
        "## Current Exchange-Reported Positions",
        "",
        frame_to_markdown(current_positions),
        "",
        "## Notes",
        "",
        "- Product PnL is realized closes plus the change in unrealized PnL, less fees.",
        "- Night-session events are assigned to the next available EOD trading date.",
        "- Session execution prices are volume-weighted averages from trades.csv.",
        "- Current account and positions are read from rtdata.json without modifying it.",
        "- Strategy attribution uses signed previous-day pos_by_strat weights when available.",
        "- Manual trading is included if it appears in pos_by_strat.",
        "- Offsetting or invalid strategy positions are left explicitly unallocated.",
        "- Ledger residuals are retained as UNEXPLAINED_ADJUSTMENT.",
    ]
    return "\n".join(lines)


def frame_to_email_html(frame: pd.DataFrame) -> str:
    """Render a compact HTML table with rounded numeric values."""

    if frame.empty:
        return "<p><i>No rows.</i></p>"
    display = frame.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(2)
    return display.to_html(index=False, border=0, na_rep="")


def send_pnl_email(
    paths: ExplainPaths,
    funds_detail: Dict[str, float],
    code_detail: pd.DataFrame,
    product_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    input_issues: pd.DataFrame,
    account_detail: pd.DataFrame,
    current_positions: pd.DataFrame,
    rtdata_metadata: Dict[str, object],
    *,
    email_notify: bool,
) -> bool:
    """Send the full PnL explain using the production SMTP configuration."""

    if not email_notify:
        return False

    from pycmqlib3.utility.email_tool import send_html_by_smtp
    from pycmqlib3.utility.sec_bits import EMAIL_QQ, LOCAL_PC_NAME, NOTIFIERS

    product_table = product_detail.reindex(
        columns=[
            "product",
            "daily_pnl",
            "fee",
            "prev_volume",
            "curr_volume",
            "position_change",
        ]
    )
    contract_table = code_detail.reindex(
        columns=[
            "contract",
            "product",
            "daily_pnl",
            "fee",
            "prev_volume",
            "curr_volume",
            "position_change",
            "night_avg_executed_price",
            "night_executed_volume",
            "day_avg_executed_price",
            "day_executed_volume",
        ]
    )
    subject = (
        f"{LOCAL_PC_NAME} daily PnL explain "
        f"<{datetime.strptime(paths.run_date_str, '%Y%m%d'):%Y.%m.%d}>"
    )
    html = (
        "<html><head><style>"
        "body { font-family: Arial, sans-serif; font-size: 12px; }"
        "table { border-collapse: collapse; font-size: 11px; margin-bottom: 18px; }"
        "th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: right; }"
        "th { background-color: #f2f2f2; position: sticky; top: 0; }"
        "h2, h3 { margin-bottom: 6px; }"
        "</style></head><body>"
        f"<h2>Daily PnL Explain: {html_lib.escape(paths.port_file)} "
        f"{html_lib.escape(paths.run_date_str)}</h2>"
        f"<p>Previous date: <b>{html_lib.escape(paths.prev_date_str)}</b><br>"
        f"Portfolio daily PnL: <b>{funds_detail['daily_equity_pnl']:.2f}</b><br>"
        f"Daily fees: <b>{funds_detail['daily_fee']:.2f}</b><br>"
        "Current exchange-state timestamp: "
        f"<b>{html_lib.escape(str(rtdata_metadata['rtdata_last_modified']))}</b></p>"
        "<h3>Product PnL</h3>"
        + frame_to_email_html(product_table)
        + "<h3>Contract PnL and Exchange Executions</h3>"
        + frame_to_email_html(contract_table)
        + "<h3>Estimated Strategy Attribution</h3>"
        + frame_to_email_html(strategy_summary)
        + "<h3>Current Exchange-Reported Account</h3>"
        + frame_to_email_html(account_detail)
        + "<h3>Current Exchange-Reported Positions</h3>"
        + frame_to_email_html(current_positions)
        + "<h3>Input Issues</h3>"
        + frame_to_email_html(input_issues)
        + "<p><i>Current account and positions are read from rtdata.json; "
        "the runtime file is not modified.</i></p>"
        "</body></html>"
    )
    send_html_by_smtp(EMAIL_QQ, NOTIFIERS, subject, html)
    return True


def save_outputs(
    out_dir: Path,
    markdown_report: str,
    code_detail: pd.DataFrame,
    product_detail: pd.DataFrame,
    strategy_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    input_issues: pd.DataFrame,
    account_detail: pd.DataFrame,
    current_positions: pd.DataFrame,
    summary: Dict[str, float | str],
) -> None:
    """Persist the report and all supporting tables."""

    out_dir.mkdir(parents=True, exist_ok=True)
    code_detail.to_csv(out_dir / "code_pnl_detail.csv", index=False)
    product_detail.to_csv(out_dir / "product_pnl_detail.csv", index=False)
    strategy_detail.to_csv(out_dir / "strategy_pnl_detail.csv", index=False)
    strategy_summary.to_csv(out_dir / "strategy_pnl_summary.csv", index=False)
    input_issues.to_csv(out_dir / "input_issues.csv", index=False)
    account_detail.to_csv(out_dir / "current_account.csv", index=False)
    current_positions.to_csv(out_dir / "current_positions.csv", index=False)
    with (out_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown_report)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def print_console_summary(
    paths: ExplainPaths,
    funds_detail: Dict[str, float],
    code_detail: pd.DataFrame,
    product_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
) -> None:
    """Print a compact console summary."""

    print(f"PnL explain for {paths.port_file} on {paths.run_date_str}")
    print(f"Previous date: {paths.prev_date_str}")
    print(f"Daily equity PnL: {funds_detail['daily_equity_pnl']:.2f}")
    print(f"Daily fees: {funds_detail['daily_fee']:.2f}")

    print("\nProduct PnL")
    print(
        product_detail.reindex(
            columns=[
                "product",
                "daily_pnl",
                "fee",
                "prev_volume",
                "curr_volume",
                "position_change",
            ]
        ).to_string(index=False)
    )

    print("\nContract PnL and exchange executions")
    print(
        code_detail.reindex(
            columns=[
                "contract",
                "product",
                "daily_pnl",
                "night_avg_executed_price",
                "night_executed_volume",
                "day_avg_executed_price",
                "day_executed_volume",
                "prev_volume",
                "curr_volume",
            ]
        ).to_string(index=False)
    )

    print("\nStrategy attribution")
    print(strategy_summary.to_string(index=False))


def main() -> None:
    """Run the daily PnL explain workflow."""

    args = parse_args()
    port_dir = Path(args.port_dir).resolve()
    group_dir = Path(args.group_dir).resolve()
    run_date_str = (
        args.run_date
        if args.run_date
        else resolve_default_run_date().strftime("%Y%m%d")
    )
    paths = build_paths(
        port_dir=port_dir,
        group_dir=group_dir,
        port_file=args.port_file,
        output_strategy=args.output_strategy,
        run_date=run_date_str,
        trader_channel=args.trader_channel,
    )

    code_detail = build_code_pnl_detail(paths)
    product_detail = build_product_pnl_detail(code_detail)
    funds_detail = load_daily_funds_detail(paths)
    product_detail, ledger_residual = reconcile_product_pnl(
        product_detail,
        funds_detail["daily_equity_pnl"],
    )
    strategy_detail, strategy_summary, input_issues = build_strategy_attribution(
        paths,
        product_detail,
    )
    account_detail, current_positions, rtdata_metadata = load_rtdata_current_state(
        paths.rtdata_json
    )

    ledger_net_pnl = float(code_detail["daily_pnl"].sum())
    ledger_gross_pnl = float(code_detail["gross_pnl"].sum())
    ledger_fees = float(code_detail["fee"].sum())
    unallocated_pnl = float(
        strategy_summary.loc[
            strategy_summary["strategy_file"].str.startswith("UNALLOCATED"),
            "attributed_pnl",
        ].sum()
    )
    summary = {
        "run_date": paths.run_date_str,
        "prev_date": paths.prev_date_str,
        "portfolio_daily_equity_pnl": funds_detail["daily_equity_pnl"],
        "portfolio_daily_fee": funds_detail["daily_fee"],
        "ledger_gross_pnl": ledger_gross_pnl,
        "ledger_fee": ledger_fees,
        "ledger_net_pnl": ledger_net_pnl,
        "explained_code_pnl": ledger_net_pnl,
        "residual_pnl": ledger_residual,
        "strategy_unallocated_pnl": unallocated_pnl,
        "input_issue_count": int(len(input_issues)),
        **rtdata_metadata,
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
        input_issues,
        ledger_residual,
        account_detail,
        current_positions,
        rtdata_metadata,
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
        input_issues,
        account_detail,
        current_positions,
        summary,
    )
    email_notify = bool(args.email)
    if not args.no_email and not args.email:
        from pycmqlib3.utility.sec_bits import EMAIL_NOTIFY

        email_notify = bool(EMAIL_NOTIFY)

    email_sent = send_pnl_email(
        paths,
        funds_detail,
        code_detail,
        product_detail,
        strategy_summary,
        input_issues,
        account_detail,
        current_positions,
        rtdata_metadata,
        email_notify=email_notify,
    )
    print_console_summary(
        paths,
        funds_detail,
        code_detail,
        product_detail,
        strategy_summary,
    )
    print(f"\nEmail sent: {email_sent}")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
