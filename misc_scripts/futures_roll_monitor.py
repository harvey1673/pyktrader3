r"""Monitor liquidity migration from production c1 futures contracts.

The production hotmap remains read-only.  For each product this script reports
the current c1 contract and the two later-expiring contracts with the highest
open interest on the observation date.  Price changes are calculated on each
individual contract, so the result contains no continuous-contract roll jump.

Example:
    D:\miniconda3\python.exe misc_scripts\futures_roll_monitor.py 20261012 --products rb,TA,lc --output-dir tests\output
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pycmqlib3.utility.sec_bits import LOCAL_NUTSTORE_FOLDER


WTPY_ROOT = Path(r"C:\dev\wtpy")
DEFAULT_HOTMAP = Path(LOCAL_NUTSTORE_FOLDER) / "hotmap_prod.json"
DEFAULT_DATA_ROOT = Path(r"C:\dev\wtdev\storage\his")
DEFAULT_POSITION_DIR = REPO_ROOT / "process" / "paper_sim1"
DEFAULT_PORT_FILE = "PTSIM1_FACTPORT1_hot"
PRICE_HORIZONS = (1, 2, 3, 5)


class RollMonitorError(ValueError):
    """Raised when a product cannot produce a valid roll-monitor snapshot."""


def parse_date(value: str | dt.date | pd.Timestamp) -> dt.date:
    """Parse a date accepted by pandas into a ``datetime.date``."""
    return pd.Timestamp(value).date()


def load_hotmap(path: Path) -> list[dict[str, str]]:
    """Load the exchange-keyed production hotmap into flat product rows."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    rows: list[dict[str, str]] = []
    for exchange, product_map in raw.items():
        for product, contract in product_map.items():
            rows.append(
                {
                    "exchange": str(exchange),
                    "product": str(product),
                    "hotmap_contract": str(contract),
                }
            )
    return rows


def load_position_history(
    position_dir: Path,
    port_file: str,
) -> tuple[dict[dt.date, dict[str, float]], dict[dt.date, Path]]:
    """Load dated aggregate portfolio positions without changing any file."""
    pattern = re.compile(rf"^{re.escape(port_file)}_(\d{{8}})\.json$")
    snapshots: dict[dt.date, dict[str, float]] = {}
    paths: dict[dt.date, Path] = {}
    for path in position_dir.glob(f"{port_file}_*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        file_date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        snapshots[file_date] = {
            str(product): float(lots) for product, lots in raw.items()
        }
        paths[file_date] = path
    if not snapshots:
        raise FileNotFoundError(
            f"No {port_file}_YYYYMMDD.json files found under {position_dir}."
        )
    return snapshots, paths


def resolve_position_pair(
    position_history: dict[dt.date, dict[str, float]],
    observation_date: dt.date,
    product: str,
) -> tuple[float, dt.date | None, float | None, dt.date | None]:
    """Return current and next available positions around a market-data date."""
    dates = sorted(position_history)
    current_dates = [date for date in dates if date <= observation_date]
    current_date = current_dates[-1] if current_dates else None
    new_dates = [date for date in dates if date > observation_date]
    new_date = new_dates[0] if new_dates else None
    current_lots = (
        float(position_history[current_date].get(product, 0.0))
        if current_date is not None
        else 0.0
    )
    new_lots = (
        float(position_history[new_date].get(product, 0.0))
        if new_date is not None
        else None
    )
    return current_lots, current_date, new_lots, new_date


def active_products_for_position_pair(
    position_history: dict[dt.date, dict[str, float]],
    observation_date: dt.date,
) -> set[str]:
    """Return products listed in the current or first newer position snapshot."""
    dates = sorted(position_history)
    current_dates = [date for date in dates if date <= observation_date]
    new_dates = [date for date in dates if date > observation_date]
    selected_dates = []
    if current_dates:
        selected_dates.append(current_dates[-1])
    if new_dates:
        selected_dates.append(new_dates[0])
    return {
        product
        for date in selected_dates
        for product in position_history[date]
    }


def _split_contract(contract: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d{3,4})", contract.strip())
    if not match:
        raise RollMonitorError(f"Cannot parse futures contract '{contract}'.")
    return match.group(1), match.group(2)


def normalize_hotmap_contract(
    exchange: str,
    contract: str,
    ref_date: dt.date,
) -> str:
    """Normalize three-digit CZCE YYM codes to local four-digit YYMM codes."""
    product, digits = _split_contract(contract)
    if exchange.upper() != "CZCE" or len(digits) == 4:
        return contract

    year_digit = int(digits[0])
    month = int(digits[1:])
    if not 1 <= month <= 12:
        raise RollMonitorError(f"Invalid contract month in '{contract}'.")

    decade = (ref_date.year // 10) * 10
    candidate_years = [decade - 10 + year_digit, decade + year_digit, decade + 10 + year_digit]
    contract_year = min(
        candidate_years,
        key=lambda year: abs((year - ref_date.year) * 12 + month - ref_date.month),
    )
    return f"{product}{contract_year % 100:02d}{month:02d}"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _contract_metrics(
    history: pd.DataFrame,
    contract: str,
    observation_date: dt.date,
    price_field: str,
) -> dict[str, float | None]:
    contract_history = history[
        (history["instID"] == contract) & (history["date"] <= observation_date)
    ].sort_values("date")
    contract_history = contract_history.drop_duplicates("date", keep="last")
    if contract_history.empty:
        raise RollMonitorError(
            f"No history for contract '{contract}' through {observation_date}."
        )

    current = contract_history.iloc[-1]
    metrics: dict[str, float | None] = {
        "close": float(current["close"]),
        "settle": float(current["settle"]) if pd.notna(current.get("settle")) else None,
        "volume": float(current["volume"]),
        "open_interest": float(current["openInterest"]),
        "oi_change_1d": (
            float(current["diff_oi"])
            if "diff_oi" in current.index and pd.notna(current["diff_oi"])
            else None
        ),
    }

    prices = pd.to_numeric(contract_history[price_field], errors="coerce").dropna()
    current_price = float(prices.iloc[-1])
    for horizon in PRICE_HORIZONS:
        change_key = f"price_change_{horizon}d"
        return_key = f"price_return_{horizon}d_pct"
        if len(prices) <= horizon:
            metrics[change_key] = None
            metrics[return_key] = None
            continue
        previous_price = float(prices.iloc[-horizon - 1])
        metrics[change_key] = current_price - previous_price
        metrics[return_key] = (
            (current_price / previous_price - 1.0) * 100.0
            if previous_price != 0.0
            else None
        )
    return metrics


def _spread_change_metrics(
    data: pd.DataFrame,
    c1_contract: str,
    candidate_contract: str,
    observation_date: dt.date,
    price_field: str,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float | None]:
    """Return changes in the c1-minus-candidate spread on matched trading dates."""
    pair = data[
        (data["date"] <= observation_date)
        & data["instID"].isin([c1_contract, candidate_contract])
    ].pivot_table(index="date", columns="instID", values=price_field, aggfunc="last")
    pair = pair.dropna(subset=[c1_contract, candidate_contract]).sort_index()
    spread = pair[c1_contract] - pair[candidate_contract]
    metrics: dict[str, float | None] = {}
    for horizon in horizons:
        key = f"c1_minus_candidate_spread_change_{horizon}d"
        metrics[key] = (
            None
            if len(spread) <= horizon
            else float(spread.iloc[-1] - spread.iloc[-horizon - 1])
        )
    return metrics


def classify_roll_status(
    candidate_oi_ratio: float | None,
    candidate_volume_ratio: float | None,
    position_roll_yield_ann_pct: float | None = None,
    position_side: str = "FLAT",
    yield_candidate_oi_ratio: float | None = None,
    yield_candidate_volume_ratio: float | None = None,
    ready_oi_ratio: float = 1.0,
    ready_volume_ratio: float = 0.5,
    watch_oi_ratio: float = 0.8,
    yield_watch_ann_pct: float = 5.0,
    yield_min_oi_ratio: float = 0.2,
    yield_min_volume_ratio: float = 0.1,
) -> tuple[str, str]:
    """Classify liquidity migration using transparent configurable thresholds."""
    if candidate_oi_ratio is None:
        return "NO_CANDIDATE", "No later contract with positive open interest."
    if (
        candidate_oi_ratio >= ready_oi_ratio
        and candidate_volume_ratio is not None
        and candidate_volume_ratio >= ready_volume_ratio
    ):
        return (
            "READY",
            "Candidate OI has overtaken c1 and candidate volume passes the liquidity threshold.",
        )
    yield_has_liquidity = (
        yield_candidate_oi_ratio is not None
        and yield_candidate_oi_ratio >= yield_min_oi_ratio
        and yield_candidate_volume_ratio is not None
        and yield_candidate_volume_ratio >= yield_min_volume_ratio
    )
    if (
        position_side in {"LONG", "SHORT"}
        and position_roll_yield_ann_pct is not None
        and position_roll_yield_ann_pct >= yield_watch_ann_pct
        and yield_has_liquidity
    ):
        return (
            f"YIELD_WATCH_{position_side}",
            f"Annualized roll yield is attractive for {position_side.lower()} positions "
            "and the candidate passes the minimum liquidity thresholds.",
        )
    if candidate_oi_ratio >= watch_oi_ratio:
        return "WATCH", "Candidate OI is approaching or exceeding c1 OI."
    if candidate_volume_ratio is not None and candidate_volume_ratio >= 1.0:
        return "WATCH", "Candidate volume has overtaken c1, while OI has not yet confirmed."
    return "HOLD", "Liquidity remains concentrated in the current c1 contract."


def build_product_snapshot(
    history: pd.DataFrame,
    exchange: str,
    product: str,
    hotmap_contract: str,
    requested_date: dt.date,
    candidate_count: int = 2,
    price_field: str = "close",
    ready_oi_ratio: float = 1.0,
    ready_volume_ratio: float = 0.5,
    watch_oi_ratio: float = 0.8,
    yield_watch_ann_pct: float = 5.0,
    yield_min_oi_ratio: float = 0.2,
    yield_min_volume_ratio: float = 0.1,
    position_history: dict[dt.date, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build contract detail and one-row recommendation for one product."""
    required = {"instID", "date", "expiry", "close", "volume", "openInterest", price_field}
    missing = required.difference(history.columns)
    if missing:
        raise RollMonitorError(f"Missing required columns: {sorted(missing)}")
    if history.empty:
        raise RollMonitorError(f"No futures data found for product '{product}'.")

    data = history.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    data["expiry"] = pd.to_datetime(data["expiry"]).dt.date
    data = data[data["date"] <= requested_date]
    if data.empty:
        raise RollMonitorError(
            f"No observations for product '{product}' through {requested_date}."
        )

    c1_contract = normalize_hotmap_contract(exchange, hotmap_contract, requested_date)
    c1_dates = data.loc[data["instID"] == c1_contract, "date"]
    if c1_dates.empty:
        raise RollMonitorError(
            f"Hotmap c1 '{hotmap_contract}' normalized to '{c1_contract}', "
            "but that contract is absent from the data."
        )
    observation_date = max(c1_dates)
    current_position_lots, current_position_date, new_position_lots, new_position_date = (
        resolve_position_pair(position_history or {}, observation_date, product)
    )
    decision_position_lots = (
        new_position_lots if new_position_lots is not None else current_position_lots
    )
    decision_position_date = (
        new_position_date if new_position_lots is not None else current_position_date
    )
    snapshot = data[data["date"] == observation_date].copy()
    c1_rows = snapshot[snapshot["instID"] == c1_contract]
    if c1_rows.empty:
        raise RollMonitorError(f"No c1 row for '{c1_contract}' on {observation_date}.")
    c1_row = c1_rows.sort_values("openInterest", ascending=False).iloc[0]
    c1_expiry = c1_row["expiry"]

    candidates = snapshot[
        (snapshot["expiry"] > c1_expiry)
        & (snapshot["instID"] != c1_contract)
        & (pd.to_numeric(snapshot["openInterest"], errors="coerce") > 0)
    ].copy()
    candidates = candidates.sort_values(
        ["openInterest", "volume", "expiry", "instID"],
        ascending=[False, False, True, True],
    ).drop_duplicates("instID")
    selected_candidates = candidates.head(candidate_count)

    selected = [("c1", c1_contract, c1_row)]
    for rank, (_, row) in enumerate(selected_candidates.iterrows(), start=1):
        selected.append((f"next_oi_{rank}", str(row["instID"]), row))

    detail_rows: list[dict[str, object]] = []
    c1_oi = float(c1_row["openInterest"])
    c1_volume = float(c1_row["volume"])
    c1_price = float(c1_row[price_field])
    for role, contract, row in selected:
        metrics = _contract_metrics(data, contract, observation_date, price_field)
        spread_change_metrics = (
            {
                "c1_minus_candidate_spread_change_1d": None,
                "c1_minus_candidate_spread_change_2d": None,
                "c1_minus_candidate_spread_change_3d": None,
            }
            if role == "c1"
            else _spread_change_metrics(
                data, c1_contract, contract, observation_date, price_field
            )
        )
        expiry_gap_days = (row["expiry"] - c1_expiry).days
        contract_price = float(row[price_field])
        if role == "c1" or expiry_gap_days <= 0 or c1_price <= 0 or contract_price <= 0:
            price_spread_vs_c1 = 0.0 if role == "c1" else None
            long_roll_yield_pct = 0.0 if role == "c1" else None
            short_roll_yield_pct = 0.0 if role == "c1" else None
            long_roll_yield_ann_pct = 0.0 if role == "c1" else None
            short_roll_yield_ann_pct = 0.0 if role == "c1" else None
        else:
            price_spread_vs_c1 = contract_price - c1_price
            long_roll_yield_pct = math.log(c1_price / contract_price) * 100.0
            short_roll_yield_pct = -long_roll_yield_pct
            long_roll_yield_ann_pct = long_roll_yield_pct * 365.0 / expiry_gap_days
            short_roll_yield_ann_pct = -long_roll_yield_ann_pct
        detail_rows.append(
            {
                "requested_date": requested_date,
                "data_date": observation_date,
                "exchange": exchange,
                "product": product,
                "role": role,
                "contract": contract,
                "expiry": row["expiry"],
                "days_to_expiry": (row["expiry"] - observation_date).days,
                "expiry_gap_vs_c1_days": expiry_gap_days,
                **metrics,
                "oi_vs_c1": _safe_ratio(metrics["open_interest"], c1_oi),
                "volume_vs_c1": _safe_ratio(metrics["volume"], c1_volume),
                "price_spread_vs_c1": price_spread_vs_c1,
                **spread_change_metrics,
                "long_roll_yield_pct": long_roll_yield_pct,
                "short_roll_yield_pct": short_roll_yield_pct,
                "long_roll_yield_ann_pct": long_roll_yield_ann_pct,
                "short_roll_yield_ann_pct": short_roll_yield_ann_pct,
            }
        )
    detail = pd.DataFrame(detail_rows)
    current_position_side = (
        "LONG"
        if current_position_lots > 0
        else "SHORT"
        if current_position_lots < 0
        else "FLAT"
    )
    new_position_side = (
        "LONG"
        if new_position_lots is not None and new_position_lots > 0
        else "SHORT"
        if new_position_lots is not None and new_position_lots < 0
        else "FLAT"
        if new_position_lots is not None
        else None
    )
    position_side = (
        "LONG"
        if decision_position_lots > 0
        else "SHORT"
        if decision_position_lots < 0
        else "FLAT"
    )
    detail["current_position_date"] = current_position_date
    detail["current_position_lots"] = float(current_position_lots)
    detail["current_position_side"] = current_position_side
    detail["new_position_date"] = new_position_date
    detail["new_position_lots"] = new_position_lots
    detail["new_position_side"] = new_position_side
    detail["position_change_lots"] = (
        new_position_lots - current_position_lots
        if new_position_lots is not None
        else math.nan
    )
    detail["decision_position_date"] = decision_position_date
    detail["position_lots"] = float(decision_position_lots)
    detail["position_side"] = position_side
    if position_side == "LONG":
        detail["position_roll_yield_pct"] = detail["long_roll_yield_pct"]
        detail["position_roll_yield_ann_pct"] = detail["long_roll_yield_ann_pct"]
    elif position_side == "SHORT":
        detail["position_roll_yield_pct"] = detail["short_roll_yield_pct"]
        detail["position_roll_yield_ann_pct"] = detail["short_roll_yield_ann_pct"]
    else:
        detail["position_roll_yield_pct"] = math.nan
        detail["position_roll_yield_ann_pct"] = math.nan
    detail.loc[detail["role"] == "c1", "position_roll_yield_pct"] = math.nan
    detail.loc[detail["role"] == "c1", "position_roll_yield_ann_pct"] = math.nan

    candidate = detail[detail["role"] == "next_oi_1"]
    if candidate.empty:
        candidate_contract = None
        candidate_oi_ratio = None
        candidate_volume_ratio = None
    else:
        candidate_row = candidate.iloc[0]
        candidate_contract = candidate_row["contract"]
        candidate_oi_ratio = candidate_row["oi_vs_c1"]
        candidate_volume_ratio = candidate_row["volume_vs_c1"]
    later_contracts = detail[detail["role"] != "c1"].copy()
    if later_contracts.empty or position_side == "FLAT":
        position_roll_contract = None
        position_roll_yield_pct = None
        position_roll_yield = None
        position_roll_oi_ratio = None
        position_roll_volume_ratio = None
        best_long_roll_contract = None
        best_short_roll_contract = None
    else:
        liquid_for_early_roll = later_contracts[
            (later_contracts["oi_vs_c1"] >= yield_min_oi_ratio)
            & (later_contracts["volume_vs_c1"] >= yield_min_volume_ratio)
        ]
        position_roll_pool = (
            liquid_for_early_roll if not liquid_for_early_roll.empty else later_contracts
        )
        position_roll_row = position_roll_pool.sort_values(
            ["position_roll_yield_ann_pct", "open_interest"], ascending=[False, False]
        ).iloc[0]
        position_roll_contract = position_roll_row["contract"]
        position_roll_yield_pct = position_roll_row["position_roll_yield_pct"]
        position_roll_yield = position_roll_row["position_roll_yield_ann_pct"]
        position_roll_oi_ratio = position_roll_row["oi_vs_c1"]
        position_roll_volume_ratio = position_roll_row["volume_vs_c1"]
        best_long_roll_contract = later_contracts.sort_values(
            ["long_roll_yield_ann_pct", "open_interest"], ascending=[False, False]
        ).iloc[0]["contract"]
        best_short_roll_contract = later_contracts.sort_values(
            ["short_roll_yield_ann_pct", "open_interest"], ascending=[False, False]
        ).iloc[0]["contract"]
    roll_status, roll_reason = classify_roll_status(
        candidate_oi_ratio,
        candidate_volume_ratio,
        position_roll_yield_ann_pct=position_roll_yield,
        position_side=position_side,
        yield_candidate_oi_ratio=position_roll_oi_ratio,
        yield_candidate_volume_ratio=position_roll_volume_ratio,
        ready_oi_ratio=ready_oi_ratio,
        ready_volume_ratio=ready_volume_ratio,
        watch_oi_ratio=watch_oi_ratio,
        yield_watch_ann_pct=yield_watch_ann_pct,
        yield_min_oi_ratio=yield_min_oi_ratio,
        yield_min_volume_ratio=yield_min_volume_ratio,
    )
    if roll_status == "READY":
        recommended_contract = candidate_contract
    elif roll_status.startswith("YIELD_WATCH"):
        recommended_contract = position_roll_contract
    else:
        recommended_contract = None

    summary: dict[str, object] = {
        "requested_date": requested_date,
        "data_date": observation_date,
        "exchange": exchange,
        "product": product,
        "c1_contract": c1_contract,
        "c1_expiry": c1_expiry,
        "c1_days_to_expiry": (c1_expiry - observation_date).days,
        "c1_open_interest": c1_oi,
        "c1_volume": c1_volume,
        "current_position_date": current_position_date,
        "current_position_lots": float(current_position_lots),
        "current_position_side": current_position_side,
        "new_position_date": new_position_date,
        "new_position_lots": new_position_lots,
        "new_position_side": new_position_side,
        "position_change_lots": (
            new_position_lots - current_position_lots
            if new_position_lots is not None
            else None
        ),
        "decision_position_date": decision_position_date,
        "position_lots": float(decision_position_lots),
        "position_side": position_side,
        "recommended_contract": recommended_contract,
        "liquidity_candidate": candidate_contract,
        "candidate_oi_vs_c1": candidate_oi_ratio,
        "candidate_volume_vs_c1": candidate_volume_ratio,
        "position_roll_contract": position_roll_contract,
        "position_roll_yield_pct": position_roll_yield_pct,
        "position_roll_yield_ann_pct": position_roll_yield,
        "position_roll_oi_vs_c1": position_roll_oi_ratio,
        "position_roll_volume_vs_c1": position_roll_volume_ratio,
        "best_long_roll_contract": best_long_roll_contract,
        "best_short_roll_contract": best_short_roll_contract,
        "roll_status": roll_status,
        "roll_reason": roll_reason,
    }
    for rank in range(1, candidate_count + 1):
        row = detail[detail["role"] == f"next_oi_{rank}"]
        candidate_row = None if row.empty else row.iloc[0]
        summary[f"candidate_{rank}"] = (
            None if candidate_row is None else candidate_row["contract"]
        )
        summary[f"candidate_{rank}_oi"] = (
            None if candidate_row is None else candidate_row["open_interest"]
        )
        summary[f"candidate_{rank}_volume"] = (
            None if candidate_row is None else candidate_row["volume"]
        )
        summary[f"candidate_{rank}_oi_vs_c1"] = (
            None if candidate_row is None else candidate_row["oi_vs_c1"]
        )
        summary[f"candidate_{rank}_volume_vs_c1"] = (
            None if candidate_row is None else candidate_row["volume_vs_c1"]
        )
        for horizon in (1, 2, 3):
            source = f"c1_minus_candidate_spread_change_{horizon}d"
            summary[f"candidate_{rank}_{source}"] = (
                None if candidate_row is None else candidate_row[source]
            )
    return detail, summary


def load_product_history(
    exchange: str,
    product: str,
    start_date: dt.date,
    end_date: dt.date,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Load all listed contracts for one product from the existing WTPY DSB store."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if WTPY_ROOT.exists() and str(WTPY_ROOT) not in sys.path:
        sys.path.insert(0, str(WTPY_ROOT))
    from pycmqlib3.utility.process_wt_data import load_fut_by_product

    return load_fut_by_product(
        f"{exchange}.{product}",
        start_date=start_date,
        end_date=end_date,
        freq="d",
        folder_loc=str(data_root),
    )


def build_monitor(
    hotmap_rows: Iterable[dict[str, str]],
    requested_date: dt.date,
    products: set[str] | None = None,
    candidate_count: int = 2,
    lookback_calendar_days: int = 20,
    price_field: str = "close",
    ready_oi_ratio: float = 1.0,
    ready_volume_ratio: float = 0.5,
    watch_oi_ratio: float = 0.8,
    yield_watch_ann_pct: float = 5.0,
    yield_min_oi_ratio: float = 0.2,
    yield_min_volume_ratio: float = 0.1,
    data_root: Path = DEFAULT_DATA_ROOT,
    loader: Callable[..., pd.DataFrame] = load_product_history,
    position_history: dict[dt.date, dict[str, float]] | None = None,
    positions_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build detail, summary, and error tables for all selected hotmap products."""
    details: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    start_date = requested_date - dt.timedelta(days=lookback_calendar_days)
    position_history = position_history or {}
    active_products = (
        active_products_for_position_pair(position_history, requested_date)
        if positions_only
        else None
    )

    for hotmap_row in hotmap_rows:
        product = hotmap_row["product"]
        if products is not None and product not in products:
            continue
        if active_products is not None and product not in active_products:
            continue
        try:
            history = loader(
                hotmap_row["exchange"],
                product,
                start_date,
                requested_date,
                data_root=data_root,
            )
            detail, summary = build_product_snapshot(
                history,
                exchange=hotmap_row["exchange"],
                product=product,
                hotmap_contract=hotmap_row["hotmap_contract"],
                requested_date=requested_date,
                candidate_count=candidate_count,
                price_field=price_field,
                ready_oi_ratio=ready_oi_ratio,
                ready_volume_ratio=ready_volume_ratio,
                watch_oi_ratio=watch_oi_ratio,
                yield_watch_ann_pct=yield_watch_ann_pct,
                yield_min_oi_ratio=yield_min_oi_ratio,
                yield_min_volume_ratio=yield_min_volume_ratio,
                position_history=position_history,
            )
            details.append(detail)
            summaries.append(summary)
        except Exception as exc:
            errors.append(
                {
                    "exchange": hotmap_row["exchange"],
                    "product": product,
                    "hotmap_contract": hotmap_row["hotmap_contract"],
                    "error": str(exc),
                }
            )

    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    error_df = pd.DataFrame(errors)
    if not summary_df.empty:
        status_order = pd.CategoricalDtype(
            [
                "READY",
                "YIELD_WATCH_LONG",
                "YIELD_WATCH_SHORT",
                "WATCH",
                "HOLD",
                "NO_CANDIDATE",
            ],
            ordered=True,
        )
        summary_df["roll_status"] = summary_df["roll_status"].astype(status_order)
        summary_df = summary_df.sort_values(
            ["roll_status", "candidate_oi_vs_c1", "product"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)
        summary_df["roll_status"] = summary_df["roll_status"].astype(str)
    return detail_df, summary_df, error_df


def build_email_html(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    errors: pd.DataFrame,
    requested_date: dt.date,
) -> str:
    """Build an email body using the existing contract-summary table style."""
    summary_columns = [
        "product",
        "data_date",
        "c1_contract",
        "current_position_date",
        "current_position_lots",
        "current_position_side",
        "new_position_date",
        "new_position_lots",
        "new_position_side",
        "position_change_lots",
        "decision_position_date",
        "position_lots",
        "position_side",
        "c1_days_to_expiry",
        "c1_open_interest",
        "c1_volume",
        "candidate_1",
        "candidate_1_oi_vs_c1",
        "candidate_1_volume_vs_c1",
        "candidate_1_c1_minus_candidate_spread_change_1d",
        "candidate_1_c1_minus_candidate_spread_change_2d",
        "candidate_1_c1_minus_candidate_spread_change_3d",
        "candidate_2",
        "candidate_2_oi_vs_c1",
        "candidate_2_volume_vs_c1",
        "candidate_2_c1_minus_candidate_spread_change_1d",
        "candidate_2_c1_minus_candidate_spread_change_2d",
        "candidate_2_c1_minus_candidate_spread_change_3d",
        "position_roll_contract",
        "position_roll_yield_pct",
        "position_roll_yield_ann_pct",
        "recommended_contract",
        "best_long_roll_contract",
        "best_short_roll_contract",
        "roll_status",
    ]
    detail_columns = [
        "product",
        "role",
        "contract",
        "expiry",
        "open_interest",
        "volume",
        "price_return_1d_pct",
        "price_return_2d_pct",
        "price_return_3d_pct",
        "price_return_5d_pct",
        "price_spread_vs_c1",
        "c1_minus_candidate_spread_change_1d",
        "c1_minus_candidate_spread_change_2d",
        "c1_minus_candidate_spread_change_3d",
        "long_roll_yield_pct",
        "short_roll_yield_pct",
        "long_roll_yield_ann_pct",
        "short_roll_yield_ann_pct",
        "position_lots",
        "position_side",
        "current_position_lots",
        "new_position_lots",
        "position_change_lots",
        "position_roll_yield_pct",
        "position_roll_yield_ann_pct",
    ]
    actionable = summary[
        summary["roll_status"].isin(
            ["READY", "YIELD_WATCH_LONG", "YIELD_WATCH_SHORT", "WATCH"]
        )
    ]
    actionable_html = (
        "<p>No products are in READY or WATCH status.</p>"
        if actionable.empty
        else actionable[summary_columns].to_html(index=False, border=0, na_rep="")
    )
    detail_html = (
        "<p>No contract detail was produced.</p>"
        if detail.empty
        else detail[detail_columns].to_html(index=False, border=0, na_rep="")
    )
    error_html = ""
    if not errors.empty:
        error_html = "<h3>Data errors</h3>" + errors.to_html(
            index=False, border=0, na_rep="", escape=True
        )
    return (
        "<html><head><style>"
        "table { border-collapse: collapse; font-size: 12px; margin-bottom: 16px; }"
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }"
        "th { background-color: #f2f2f2; }"
        "h2, h3, p { font-family: Arial, sans-serif; }"
        "</style></head><body>"
        f"<h2>Futures roll monitor — {requested_date}</h2>"
        "<p>READY means the highest-OI later contract has overtaken c1 OI and "
        "passes the configured volume threshold. YIELD_WATCH identifies an "
        "attractive direction-specific roll yield with minimum liquidity. WATCH "
        "identifies an approaching liquidity migration. The production hotmap is "
        "read-only and has not been changed.</p>"
        "<h3>READY / YIELD_WATCH / WATCH</h3>"
        f"{actionable_html}"
        "<h3>Contract detail</h3>"
        f"{detail_html}"
        f"{error_html}"
        "</body></html>"
    )


def send_monitor_email(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    errors: pd.DataFrame,
    requested_date: dt.date,
) -> None:
    """Send the monitor through the repository's configured summary-email path."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from pycmqlib3.utility.email_tool import send_html_by_smtp
    from pycmqlib3.utility.sec_bits import EMAIL_QQ, LOCAL_PC_NAME, NOTIFIERS

    subject = f"{LOCAL_PC_NAME} futures roll monitor <{requested_date:%Y.%m.%d}>"
    html = build_email_html(summary, detail, errors, requested_date)
    send_html_by_smtp(EMAIL_QQ, NOTIFIERS, subject, html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_date",
        nargs="?",
        default=dt.date.today().strftime("%Y%m%d"),
        help="Observation date in YYYYMMDD format; defaults to today.",
    )
    parser.add_argument("--hotmap", type=Path, default=DEFAULT_HOTMAP)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--position-dir", type=Path, default=DEFAULT_POSITION_DIR)
    parser.add_argument("--port-file", default=DEFAULT_PORT_FILE)
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Do not load or filter by paper_sim1 positions; monitor all products as flat.",
    )
    parser.add_argument(
        "--products",
        help="Optional comma-separated product list, preserving case (for example rb,TA,lc).",
    )
    parser.add_argument("--price-field", choices=("close", "settle"), default="close")
    parser.add_argument("--ready-oi-ratio", type=float, default=1.0)
    parser.add_argument("--ready-volume-ratio", type=float, default=0.5)
    parser.add_argument("--watch-oi-ratio", type=float, default=0.8)
    parser.add_argument(
        "--yield-watch-ann-pct",
        type=float,
        default=5.0,
        help="Minimum position-adjusted annualized roll yield for YIELD_WATCH status.",
    )
    parser.add_argument("--yield-min-oi-ratio", type=float, default=0.2)
    parser.add_argument("--yield-min-volume-ratio", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for detail, summary, and error CSV files.",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send the HTML report through the configured contract-summary email path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_date = parse_date(args.run_date)
    products = set(args.products.split(",")) if args.products else None
    if args.no_positions:
        position_history: dict[dt.date, dict[str, float]] = {}
        position_paths: dict[dt.date, Path] = {}
    else:
        position_history, position_paths = load_position_history(
            args.position_dir, args.port_file
        )
    detail, summary, errors = build_monitor(
        load_hotmap(args.hotmap),
        requested_date=requested_date,
        products=products,
        price_field=args.price_field,
        ready_oi_ratio=args.ready_oi_ratio,
        ready_volume_ratio=args.ready_volume_ratio,
        watch_oi_ratio=args.watch_oi_ratio,
        yield_watch_ann_pct=args.yield_watch_ann_pct,
        yield_min_oi_ratio=args.yield_min_oi_ratio,
        yield_min_volume_ratio=args.yield_min_volume_ratio,
        data_root=args.data_root,
        position_history=position_history,
        positions_only=not args.no_positions,
    )

    display_columns = [
        "product",
        "data_date",
        "c1_contract",
        "current_position_date",
        "current_position_lots",
        "new_position_date",
        "new_position_lots",
        "position_change_lots",
        "decision_position_date",
        "position_lots",
        "position_side",
        "c1_days_to_expiry",
        "candidate_1",
        "candidate_1_oi_vs_c1",
        "candidate_1_volume_vs_c1",
        "candidate_1_c1_minus_candidate_spread_change_1d",
        "candidate_1_c1_minus_candidate_spread_change_2d",
        "candidate_1_c1_minus_candidate_spread_change_3d",
        "candidate_2",
        "candidate_2_oi_vs_c1",
        "candidate_2_volume_vs_c1",
        "candidate_2_c1_minus_candidate_spread_change_1d",
        "candidate_2_c1_minus_candidate_spread_change_2d",
        "candidate_2_c1_minus_candidate_spread_change_3d",
        "position_roll_contract",
        "position_roll_yield_pct",
        "position_roll_yield_ann_pct",
        "recommended_contract",
        "roll_status",
    ]
    if summary.empty:
        print("No product snapshots were produced.")
    else:
        print(summary[display_columns].to_string(index=False))
    if position_paths and not summary.empty:
        used_position_dates = sorted(
            {
                date
                for column in ("current_position_date", "new_position_date")
                for date in summary[column].dropna().tolist()
            }
        )
        for position_date in used_position_dates:
            print(f"\nPositions {position_date}: {position_paths[position_date]}")
    if not errors.empty:
        print("\nProducts with errors:")
        print(errors.to_string(index=False))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = requested_date.strftime("%Y%m%d")
        detail.to_csv(args.output_dir / f"futures_roll_detail_{suffix}.csv", index=False)
        summary.to_csv(args.output_dir / f"futures_roll_summary_{suffix}.csv", index=False)
        errors.to_csv(args.output_dir / f"futures_roll_errors_{suffix}.csv", index=False)
        print(f"\nCSV output written to {args.output_dir.resolve()}")
    if args.email:
        send_monitor_email(summary, detail, errors, requested_date)
        print("Email sent.")
    return 0 if not summary.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
