"""Verify hot roll oldclose/newclose values against daily DSB data.

This script checks records in hot config JSON files (for example hot1.json,
hot2.json). For each roll record, it looks up close prices on the previous
business day of record["date"] using daily bars under:
    C:/dev/wtdev/storage/his/day/<EXCHANGE>/<CONTRACT>.dsb

Usage:
    D:/miniconda3/python.exe misc_scripts/verify_roll_newclose_oldclose.py

    D:/miniconda3/python.exe misc_scripts/verify_roll_newclose_oldclose.py \
        --configs C:/dev/wtdev/config/hot1.json C:/dev/wtdev/config/hot2.json \
        --tolerance 0.05 \
        --output-csv C:/dev/pyktrader3/tests/roll_check_report.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Make local repo importable regardless of launch directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep compatibility with local wtpy checkout used in this workstation setup.
WTPY_ROOT = Path("C:/dev/wtpy")
if WTPY_ROOT.exists() and str(WTPY_ROOT) not in sys.path:
    sys.path.insert(0, str(WTPY_ROOT))

try:
    from pycmqlib3.utility import misc
    from pycmqlib3.utility.process_wt_data import load_hist_bars_to_df
except Exception as exc:  # pragma: no cover
    print(
        "Import error: cannot import pycmqlib3/wtpy dependencies.\n"
        f"Details: {exc}\n"
        "Tip: ensure your Python env can import both 'pycmqlib3' and 'wtpy'."
    )
    raise


@dataclass
class CheckRow:
    """Single verification row for one oldclose/newclose value."""

    config_file: str
    exchange: str
    product: str
    field: str
    roll_date: int
    check_date: int
    suggested_actual_roll_date: int
    roll_date_is_business: bool
    contract: str
    expected: float
    actual: Optional[float]
    diff_pct: Optional[float]
    status: str


def parse_date_int(date_int: int) -> dt.date:
    """Convert YYYYMMDD integer to date."""
    return dt.datetime.strptime(str(int(date_int)), "%Y%m%d").date()


def to_date_int(cur_date: dt.date) -> int:
    """Convert date to YYYYMMDD integer."""
    return int(cur_date.strftime("%Y%m%d"))


def previous_business_day(roll_date_int: int, exch: str) -> dt.date:
    """Get previous business day according to exchange holiday calendar."""
    roll_date = parse_date_int(roll_date_int)
    holidays = misc.get_hols_by_exch(exch)
    return misc.day_shift(roll_date, "-1b", holidays)


def next_business_day(roll_date_int: int, exch: str) -> dt.date:
    """Get next business day according to exchange holiday calendar."""
    roll_date = parse_date_int(roll_date_int)
    holidays = misc.get_hols_by_exch(exch)
    return misc.day_shift(roll_date, "1b", holidays)


def is_business_day(cur_date: dt.date, exch: str) -> bool:
    """Check whether a date is a business day for the exchange calendar."""
    holidays = misc.get_hols_by_exch(exch)
    return misc.day_shift(misc.day_shift(cur_date, "1b", holidays), "-1b", holidays) == cur_date


class DailyCloseLoader:
    """Load and cache close series per (exchange, contract) from day DSB files."""

    def __init__(self, his_root: str) -> None:
        self.his_root = his_root
        self._cache: Dict[Tuple[str, str], Dict[dt.date, float]] = {}

    def _load_contract(self, exch: str, contract: str) -> Dict[dt.date, float]:
        key = (exch, contract)
        if key in self._cache:
            return self._cache[key]

        code = f"{exch}.{contract}"
        try:
            df = load_hist_bars_to_df(
                code,
                start_date=None,
                end_date=None,
                freq="d",
                folder_loc=self.his_root,
            )
        except Exception:
            # Missing/invalid file or decode failure -> treat as empty contract series.
            self._cache[key] = {}
            return self._cache[key]

        if df is None or len(df) == 0:
            self._cache[key] = {}
            return self._cache[key]

        out: Dict[dt.date, float] = {}
        for idx, row in df.iterrows():
            idx_date = idx if isinstance(idx, dt.date) else idx.date()
            out[idx_date] = float(row["close"])

        self._cache[key] = out
        return out

    def get_close(
        self,
        exch: str,
        contract: str,
        cur_date: dt.date,
    ) -> Optional[float]:
        """Return close for (exchange, contract, date) if available."""
        if not contract:
            return None
        series = self._load_contract(exch, contract)
        return series.get(cur_date)


def build_check_row(
    config_file: str,
    exchange: str,
    product: str,
    field: str,
    roll_date: int,
    check_date: dt.date,
    suggested_actual_roll_date: dt.date,
    roll_date_is_business: bool,
    contract: str,
    expected: float,
    actual: Optional[float],
    tolerance: float,
) -> CheckRow:
    """Create one check row and classify status by percentage tolerance."""
    if actual is None:
        return CheckRow(
            config_file=config_file,
            exchange=exchange,
            product=product,
            field=field,
            roll_date=roll_date,
            check_date=to_date_int(check_date),
            suggested_actual_roll_date=to_date_int(suggested_actual_roll_date),
            roll_date_is_business=roll_date_is_business,
            contract=contract,
            expected=expected,
            actual=None,
            diff_pct=None,
            status="MISSING_DAILY_CLOSE",
        )

    if expected == 0.0:
        # Avoid division by zero for percentage diff.
        diff_pct = 0.0 if actual == 0.0 else math.inf
    else:
        diff_pct = (actual / expected - 1.0) * 100.0

    status = (
        "OK"
        if math.isfinite(diff_pct) and abs(diff_pct) <= tolerance
        else "MISMATCH"
    )
    return CheckRow(
        config_file=config_file,
        exchange=exchange,
        product=product,
        field=field,
        roll_date=roll_date,
        check_date=to_date_int(check_date),
        suggested_actual_roll_date=to_date_int(suggested_actual_roll_date),
        roll_date_is_business=roll_date_is_business,
        contract=contract,
        expected=expected,
        actual=actual,
        diff_pct=diff_pct,
        status=status,
    )


def iter_roll_events(config_data: dict) -> Iterable[Tuple[str, str, dict]]:
    """Yield (exchange, product, event) from nested hot config JSON."""
    for exch, exch_data in config_data.items():
        if not isinstance(exch_data, dict):
            continue
        for product, events in exch_data.items():
            if not isinstance(events, list):
                continue
            for event in events:
                if isinstance(event, dict):
                    yield exch, product, event


def verify_config(
    config_path: str,
    loader: DailyCloseLoader,
    tolerance: float,
    check_initial_oldclose: bool,
) -> List[CheckRow]:
    """Verify one hot config JSON against daily close data."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    rows: List[CheckRow] = []
    config_name = Path(config_path).name

    for exch, product, event in iter_roll_events(cfg):
        try:
            roll_date = int(event["date"])
        except Exception:
            continue

        roll_dt = parse_date_int(roll_date)
        roll_date_is_biz = is_business_day(roll_dt, exch)
        suggested_actual_roll_dt = (
            roll_dt if roll_date_is_biz else next_business_day(roll_date, exch)
        )

        to_contract = str(event.get("to", "") or "")
        from_contract = str(event.get("from", "") or "")
        oldclose_val = float(event.get("oldclose", 0.0) or 0.0)
        newclose = float(event.get("newclose", 0.0) or 0.0)
        # For first-roll records where oldclose is 0, validate newclose on
        # roll date itself; otherwise use previous business day.
        first_roll_case = (from_contract == "") and (oldclose_val == 0.0)
        if first_roll_case:
            newclose_check_date = roll_dt
        else:
            newclose_check_date = previous_business_day(roll_date, exch)

        actual_newclose = loader.get_close(exch, to_contract, newclose_check_date)
        rows.append(
            build_check_row(
                config_file=config_name,
                exchange=exch,
                product=product,
                field="newclose",
                roll_date=roll_date,
                check_date=newclose_check_date,
                suggested_actual_roll_date=suggested_actual_roll_dt,
                roll_date_is_business=roll_date_is_biz,
                contract=to_contract,
                expected=newclose,
                actual=actual_newclose,
                tolerance=tolerance,
            )
        )

        if from_contract or check_initial_oldclose:
            oldclose = oldclose_val
            if from_contract and oldclose == 0.0:
                continue
            oldclose_check_date = previous_business_day(roll_date, exch)
            actual_oldclose = loader.get_close(
                exch,
                from_contract,
                oldclose_check_date,
            )
            rows.append(
                build_check_row(
                    config_file=config_name,
                    exchange=exch,
                    product=product,
                    field="oldclose",
                    roll_date=roll_date,
                    check_date=oldclose_check_date,
                    suggested_actual_roll_date=suggested_actual_roll_dt,
                    roll_date_is_business=roll_date_is_biz,
                    contract=from_contract,
                    expected=oldclose,
                    actual=actual_oldclose,
                    tolerance=tolerance,
                )
            )

    return rows


def print_report(rows: List[CheckRow]) -> None:
    """Print concise summary and mismatches/missing rows."""
    total = len(rows)
    ok_count = sum(1 for r in rows if r.status == "OK")
    mismatch_count = sum(1 for r in rows if r.status == "MISMATCH")
    missing_count = sum(1 for r in rows if r.status == "MISSING_DAILY_CLOSE")
    non_biz_roll_count = sum(1 for r in rows if not r.roll_date_is_business)

    print("\n=== Roll Price Verification Summary ===")
    print(f"Total checks          : {total}")
    print(f"OK                    : {ok_count}")
    print(f"MISMATCH              : {mismatch_count}")
    print(f"MISSING_DAILY_CLOSE   : {missing_count}")
    print(f"NON_BIZ_ROLL_DATE     : {non_biz_roll_count}")

    bad_rows = [r for r in rows if r.status != "OK"]
    if not bad_rows:
        print("\nAll checks passed.")
        return

    print("\n=== Non-OK Details ===")
    header = (
        "status,config,exchange,product,field,roll_date,check_date,"
        "roll_date_is_business,suggested_actual_roll_date,contract,"
        "expected,actual,diff_pct"
    )
    print(header)
    for r in bad_rows:
        actual_str = "" if r.actual is None else f"{r.actual:.6f}"
        diff_str = "" if r.diff_pct is None else f"{r.diff_pct:.6f}"
        print(
            f"{r.status},{r.config_file},{r.exchange},{r.product},{r.field},"
            f"{r.roll_date},{r.check_date},{r.roll_date_is_business},"
            f"{r.suggested_actual_roll_date},{r.contract},{r.expected:.6f},"
            f"{actual_str},{diff_str}"
        )


def save_csv(rows: List[CheckRow], output_csv: str) -> None:
    """Save detailed result table as CSV."""
    lines = [
        "config_file,exchange,product,field,roll_date,check_date,"
        "roll_date_is_business,suggested_actual_roll_date,contract,"
        "expected,actual,diff_pct,status"
    ]
    for r in rows:
        actual_str = "" if r.actual is None else f"{r.actual:.10f}"
        diff_str = "" if r.diff_pct is None else f"{r.diff_pct:.10f}"
        lines.append(
            f"{r.config_file},{r.exchange},{r.product},{r.field},"
            f"{r.roll_date},{r.check_date},{r.roll_date_is_business},"
            f"{r.suggested_actual_roll_date},{r.contract},"
            f"{r.expected:.10f},{actual_str},{diff_str},{r.status}"
        )

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved detailed report: {out}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Verify hot roll newclose/oldclose using daily DSB closes.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "C:/dev/wtdev/config/hot1.json",
            "C:/dev/wtdev/config/hot2.json",
        ],
        help="Hot config JSON files to verify.",
    )
    parser.add_argument(
        "--his-root",
        default="C:/dev/wtdev/storage/his",
        help="WT historical storage root containing day/min1/min5 folders.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance in percentage points for diff_pct.",
    )
    parser.add_argument(
        "--check-initial-oldclose",
        action="store_true",
        help="Also check oldclose when from contract is empty.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path for detailed results.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    loader = DailyCloseLoader(his_root=args.his_root)
    all_rows: List[CheckRow] = []

    for cfg in args.configs:
        cfg_path = Path(cfg)
        if not cfg_path.exists():
            print(f"Config file not found: {cfg}")
            continue
        print(f"Checking config: {cfg}")
        rows = verify_config(
            config_path=str(cfg_path),
            loader=loader,
            tolerance=args.tolerance,
            check_initial_oldclose=args.check_initial_oldclose,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No records checked. Please confirm config paths and schema.")
        return 1

    print_report(all_rows)

    if args.output_csv:
        save_csv(all_rows, args.output_csv)

    has_fail = any(r.status != "OK" for r in all_rows)
    return 2 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
