"""Run the futures refresh and production portfolio backtest for one business day."""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SETTINGS = PROJECT_ROOT / "process" / "paper_sim1" / "settings"
DEFAULT_WEIGHTS = PROJECT_ROOT / "process" / "PTSIM1_signal_weights.xlsx"
DEFAULT_OUTPUT = Path("C:/dev/data/html/portfolio_backtest.html")
FUTURES_PICKLE = Path("C:/dev/data/cnc_fut_m5_latest.pkl")


def _date(value: str) -> dt.date:
    """Accept YYYY-MM-DD or YYYYMMDD command-line dates."""

    cleaned = value.strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(cleaned, pattern).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("date must be YYYY-MM-DD or YYYYMMDD")


def resolve_run_date(value: dt.date | None = None) -> dt.date:
    """Return the requested/current date, rolled back to a China business day."""

    from pycmqlib3.utility.misc import CHN_Holidays, day_shift, is_workday

    candidate = dt.date.today() if value is None else value
    if is_workday(candidate, "CHN"):
        return candidate
    return day_shift(candidate, "-1b", CHN_Holidays)


def _saved_marker(path: Path = FUTURES_PICKLE) -> dt.date | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle).get("job_marker")
    except Exception:
        return None
    if value is None:
        return None
    return value.date() if hasattr(value, "date") else value


def _run(command: Sequence[str]) -> None:
    print("Running:", subprocess.list2cmdline(list(command)), flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def run_eod_process(
    run_date: dt.date,
    *,
    start_date: dt.date,
    settings_dir: Path,
    weights_excel: Path,
    output_html: Path,
) -> Path:
    """Refresh exact-date futures data, validate it, and build the HTML bundle."""

    if not settings_dir.is_dir():
        raise FileNotFoundError(f"Settings directory does not exist: {settings_dir}")
    if not weights_excel.is_file():
        raise FileNotFoundError(f"Weights workbook does not exist: {weights_excel}")
    if start_date > run_date:
        raise ValueError("start_date must not be after run_date")

    dated_futures = Path(f"C:/dev/data/fut_d_{run_date:%Y%m%d}.parquet")
    marker = _saved_marker()
    if not dated_futures.is_file() and marker is not None and marker > run_date:
        raise RuntimeError(
            f"Cannot safely recreate {dated_futures.name}: the mutable futures "
            f"pickle is already adjusted through {marker}. Restore an as-of backup "
            "or use an existing exact-date parquet."
        )

    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "misc_scripts" / "update_fut_prices.py"),
            run_date.strftime("%Y%m%d"),
        ]
    )
    if not dated_futures.is_file():
        raise FileNotFoundError(
            f"Futures refresh completed without creating {dated_futures}"
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    iso_date = run_date.isoformat()
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "misc_scripts" / "portfolio_weight_backtest.py"),
            str(settings_dir),
            str(weights_excel),
            str(output_html),
            "--start-date",
            start_date.isoformat(),
            "--end-date",
            iso_date,
            "--as-of",
            iso_date,
        ]
    )
    return output_html.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_date",
        nargs="?",
        type=_date,
        help="YYYY-MM-DD or YYYYMMDD; default is today, rolled to the prior CHN business day",
    )
    parser.add_argument("--start-date", type=_date, default=dt.date(2016, 1, 1))
    parser.add_argument("--settings-dir", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--weights-excel", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_date = resolve_run_date(args.run_date)
    requested = args.run_date or dt.date.today()
    if run_date != requested:
        print(f"{requested} is not a CHN business day; using {run_date}", flush=True)
    else:
        print(f"Using run date {run_date}", flush=True)
    output = run_eod_process(
        run_date,
        start_date=args.start_date,
        settings_dir=args.settings_dir,
        weights_excel=args.weights_excel,
        output_html=args.output_html,
    )
    print(f"EOD portfolio report completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
