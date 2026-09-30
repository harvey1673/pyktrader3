"""
Download Binance spot + continuous futures (q1/q2) bars for symbols from an existing top list.

Output layout:
    C:/dev/crypto_storage/binance/
        day/
        min60/
        min5/
        min1/

File naming:
    BTCUSDT_spot.csv
    BTCUSDT_q1.csv
    BTCUSDT_q2.csv

Notes:
- q1 maps to Binance continuous contractType=CURRENT_QUARTER
- q2 maps to Binance continuous contractType=NEXT_QUARTER
- best effort: if q2 is unavailable for a symbol/timeframe, q1/spot still download
- expiry metadata is included:
  - spot: expiry is set to the bar date/time itself
  - q1/q2: expiry is inferred from Binance listed dated futures ladder at each bar timestamp
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ccxt
import requests


CRYPTO_DIR = "C:/dev/crypto_storage"
# Keep consistency with existing helpers in this folder.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from download_history import Candle, _parse_utc_date  # noqa: E402


BINANCE_FAPI_CONT_KLINES = "https://fapi.binance.com/fapi/v1/continuousKlines"

FREQ_TO_BINANCE = {
    "day": "1d",
    "min60": "1h",
    "min5": "5m",
    "min1": "1m",
}

TF_MS = {
    "1d": 24 * 60 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "1m": 60 * 1000,
}

CONT_TYPE_MAP = {
    "q1": "CURRENT_QUARTER",
    "q2": "NEXT_QUARTER",
}


@dataclass
class CsvRow:
    ts_ms: int
    dt_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    series: str
    timeframe: str
    expiry_utc: str
    expiry_date: str


def _fmt_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _dedup_sort(candles: Iterable[Candle]) -> List[Candle]:
    uniq: Dict[int, Candle] = {}
    for c in candles:
        uniq[c.ts_ms] = c
    return [uniq[k] for k in sorted(uniq.keys())]


def _write_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "datetime_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "series",
                "timeframe",
                "expiry_utc",
                "expiry_date",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.ts_ms,
                    r.dt_utc,
                    f"{r.open:.16g}",
                    f"{r.high:.16g}",
                    f"{r.low:.16g}",
                    f"{r.close:.16g}",
                    f"{r.volume:.16g}",
                    r.symbol,
                    r.series,
                    r.timeframe,
                    r.expiry_utc,
                    r.expiry_date,
                ]
            )


def _read_last_timestamp(path: Path) -> Optional[int]:
    """Read the latest timestamp from an existing CSV, or None if unavailable."""
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        if pos <= 0:
            return None

        line = b""
        while pos > 0:
            pos -= 1
            f.seek(pos)
            ch = f.read(1)
            if ch == b"\n":
                if line.strip():
                    break
            else:
                line = ch + line

        text = line.decode("utf-8", errors="ignore").strip()
        if not text or text.startswith("timestamp,"):
            return None

        first_col = text.split(",", 1)[0]
        try:
            return int(first_col)
        except Exception:
            return None


def _append_csv(path: Path, rows: Sequence[CsvRow]) -> None:
    if not rows:
        return

    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "timestamp",
                    "datetime_utc",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "series",
                    "timeframe",
                    "expiry_utc",
                    "expiry_date",
                ]
            )
        for r in rows:
            w.writerow(
                [
                    r.ts_ms,
                    r.dt_utc,
                    f"{r.open:.16g}",
                    f"{r.high:.16g}",
                    f"{r.low:.16g}",
                    f"{r.close:.16g}",
                    f"{r.volume:.16g}",
                    r.symbol,
                    r.series,
                    r.timeframe,
                    r.expiry_utc,
                    r.expiry_date,
                ]
            )


def _resolve_since_ms(
    out_file: Path,
    timeframe: str,
    explicit_start_ms: Optional[int],
    bootstrap_since_ms: int,
) -> Tuple[int, Optional[int]]:
    """Return (since_ms, last_ts) for incremental or explicit-start download."""
    last_ts = _read_last_timestamp(out_file)

    if explicit_start_ms is not None:
        return explicit_start_ms, last_ts

    if last_ts is not None:
        return last_ts + TF_MS[timeframe], last_ts

    return bootstrap_since_ms, None


def load_symbols_from_top_list(top_list_csv: Path, top_n: Optional[int]) -> List[str]:
    if not top_list_csv.exists():
        raise FileNotFoundError(f"Top list CSV not found: {top_list_csv}")

    out: List[str] = []
    seen = set()
    with top_list_csv.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            base = (row.get("base") or "").strip().upper()
            if not base:
                continue
            sym = f"{base}USDT"
            if sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
            if top_n is not None and len(out) >= top_n:
                break
    return out


def fetch_spot_bars(
    ex_spot: ccxt.Exchange,
    pair: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int,
    max_pages: int,
) -> List[Candle]:
    symbol = f"{pair[:-4]}/USDT"
    if symbol not in ex_spot.markets:
        return []

    step = TF_MS[timeframe]
    cursor = since_ms
    pages = 0
    rows: List[Candle] = []

    while cursor <= until_ms and pages < max_pages:
        chunk = ex_spot.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        pages += 1
        if not chunk:
            cursor += step * limit
            time.sleep(max(getattr(ex_spot, "rateLimit", 50) / 1000.0, 0.05))
            continue

        for r in chunk:
            if not r or len(r) < 6:
                continue
            ts = int(r[0])
            if ts < since_ms or ts > until_ms:
                continue
            rows.append(
                Candle(
                    ts_ms=ts,
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                )
            )

        nxt = int(chunk[-1][0]) + step
        if nxt <= cursor:
            nxt = cursor + step
        cursor = nxt
        time.sleep(max(getattr(ex_spot, "rateLimit", 50) / 1000.0, 0.05))

    return _dedup_sort(rows)


def fetch_continuous_bars(
    pair: str,
    contract_type: str,
    interval: str,
    since_ms: int,
    until_ms: int,
    limit: int,
    max_pages: int,
    session: requests.Session,
) -> List[Candle]:
    step = TF_MS[interval]
    cursor = since_ms
    pages = 0
    rows: List[Candle] = []

    while cursor <= until_ms and pages < max_pages:
        params = {
            "pair": pair,
            "contractType": contract_type,
            "interval": interval,
            "startTime": cursor,
            "endTime": until_ms,
            "limit": limit,
        }
        resp = session.get(BINANCE_FAPI_CONT_KLINES, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(1.5)
            continue
        resp.raise_for_status()

        chunk = resp.json()
        pages += 1
        if not chunk:
            break

        for r in chunk:
            if not r or len(r) < 6:
                continue
            ts = int(r[0])
            if ts < since_ms or ts > until_ms:
                continue
            rows.append(
                Candle(
                    ts_ms=ts,
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                )
            )

        nxt = int(chunk[-1][0]) + step
        if nxt <= cursor:
            nxt = cursor + step
        cursor = nxt
        time.sleep(0.06)

    return _dedup_sort(rows)


def build_futures_expiry_ladders(ex_usdm: ccxt.Exchange, pairs: Sequence[str]) -> Dict[str, List[int]]:
    """
    Build {pair: sorted expiry timestamps in ms} from Binance USDT-margined dated futures.
    pair example: BTCUSDT
    """
    ladders: Dict[str, List[int]] = {p: [] for p in pairs}
    seen: Dict[str, set] = {p: set() for p in pairs}

    for _, m in ex_usdm.markets.items():
        if not m.get("future"):
            continue
        if m.get("quote") != "USDT":
            continue
        base = (m.get("base") or "").upper()
        if not base:
            continue
        pair = f"{base}USDT"
        if pair not in ladders:
            continue
        exp = m.get("expiry")
        if exp is None:
            continue
        exp_ms = int(exp)
        if exp_ms in seen[pair]:
            continue
        seen[pair].add(exp_ms)
        ladders[pair].append(exp_ms)

    for p in ladders:
        ladders[p].sort()
    return ladders


def infer_nearby_expiry(ts_ms: int, ladder: Sequence[int], rank: int) -> Optional[int]:
    """
    rank=1 => nearest expiry >= ts (q1)
    rank=2 => second-nearest expiry >= ts (q2)
    """
    future_exps = [e for e in ladder if e >= ts_ms]
    idx = rank - 1
    if idx < len(future_exps):
        return future_exps[idx]
    return None


def convert_rows(
    pair: str,
    series: str,
    timeframe: str,
    candles: Sequence[Candle],
    expiry_ladder: Optional[Sequence[int]],
) -> List[CsvRow]:
    out: List[CsvRow] = []

    for c in candles:
        ts = c.ts_ms
        if series == "spot":
            exp_ms = ts
        elif series == "q1":
            exp_ms = infer_nearby_expiry(ts, expiry_ladder or [], rank=1)
        elif series == "q2":
            exp_ms = infer_nearby_expiry(ts, expiry_ladder or [], rank=2)
        else:
            exp_ms = None

        exp_utc = _fmt_utc(exp_ms) if exp_ms is not None else ""
        exp_date = _fmt_date(exp_ms) if exp_ms is not None else ""

        out.append(
            CsvRow(
                ts_ms=ts,
                dt_utc=_fmt_utc(ts),
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                symbol=pair,
                series=series,
                timeframe=timeframe,
                expiry_utc=exp_utc,
                expiry_date=exp_date,
            )
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Binance spot + q1/q2 bars from top list")
    ap.add_argument(
        "--top-list-csv",
        default="../symbols_top50.csv",
        help="CSV with at least a base column",
    )
    ap.add_argument("--top-n", type=int, default=None, help="optional cap from top list")
    ap.add_argument("--freqs", nargs="+", default=["day", "min60", "min5", "min1"], help="folder frequencies")
    ap.add_argument(
        "--start-date",
        default=None,
        help="start UTC date/time; default None means incremental from existing files",
    )
    ap.add_argument(
        "--bootstrap-start-date",
        default="2019-01-01",
        help="used only when start-date is None and target file does not exist",
    )
    ap.add_argument("--until", default=None, help="end UTC date/time (default now)")
    ap.add_argument("--output-root", default=f"{CRYPTO_DIR}/binance", help="output root folder")
    ap.add_argument("--limit", type=int, default=1500, help="page limit for API calls")
    ap.add_argument("--max-pages", type=int, default=200000, help="max paging loops")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    for f in args.freqs:
        if f not in FREQ_TO_BINANCE:
            raise ValueError(f"Unsupported frequency: {f}. Allowed: {sorted(FREQ_TO_BINANCE.keys())}")

    explicit_start_ms = _parse_utc_date(args.start_date) if args.start_date else None
    rebuild_from_start = explicit_start_ms is not None
    bootstrap_since_ms = _parse_utc_date(args.bootstrap_start_date)
    until_ms = _parse_utc_date(args.until) if args.until else int(time.time() * 1000)

    top_list_csv = (_THIS_DIR / args.top_list_csv).resolve()
    output_root = Path(args.output_root).resolve()

    symbols = load_symbols_from_top_list(top_list_csv, args.top_n)
    if not symbols:
        raise RuntimeError(f"No symbols found in top list: {top_list_csv}")

    print(f"Loaded {len(symbols)} symbols from {top_list_csv}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    if rebuild_from_start:
        print("Mode: rebuild from start-date (truncate prior data)", flush=True)
    else:
        print("Mode: incremental append from latest file timestamp", flush=True)

    ex_spot = ccxt.binance({"enableRateLimit": True})
    ex_spot.load_markets()

    ex_usdm = ccxt.binanceusdm({"enableRateLimit": True})
    ex_usdm.load_markets()

    ladders = build_futures_expiry_ladders(ex_usdm, symbols)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "wtpy-binance-q1q2-downloader/1.0", "Accept": "application/json"})

    total_jobs = len(symbols) * len(args.freqs) * 3  # spot, q1, q2
    done = 0

    for idx, pair in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] {pair}", flush=True)
        ladder = ladders.get(pair, [])

        for f in args.freqs:
            tf = FREQ_TO_BINANCE[f]
            folder = output_root / f

            # Spot
            done += 1
            out_spot = folder / f"{pair}_spot.csv"
            if args.overwrite and out_spot.exists() and not rebuild_from_start:
                out_spot.unlink()

            if rebuild_from_start:
                since_spot = explicit_start_ms
                t0 = time.time()
                try:
                    spot = fetch_spot_bars(ex_spot, pair, tf, since_spot, until_ms, args.limit, args.max_pages)
                    rows = convert_rows(pair, "spot", tf, spot, ladder)
                    _write_csv(out_spot, rows)
                    print(
                        f"  [{done}/{total_jobs}] {f}/spot: rebuilt {len(rows):,} rows in {out_spot.name} "
                        f"for [{_fmt_utc(since_spot)} -> {_fmt_utc(until_ms)}] in {time.time()-t0:.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [{done}/{total_jobs}] {f}/spot: ERROR {type(e).__name__}: {e}", flush=True)
            else:
                since_spot, last_spot_ts = _resolve_since_ms(out_spot, tf, explicit_start_ms, bootstrap_since_ms)
                if since_spot > until_ms:
                    print(f"  [{done}/{total_jobs}] {f}/spot: up-to-date", flush=True)
                else:
                    t0 = time.time()
                    try:
                        spot = fetch_spot_bars(ex_spot, pair, tf, since_spot, until_ms, args.limit, args.max_pages)
                        rows = convert_rows(pair, "spot", tf, spot, ladder)
                        if last_spot_ts is not None:
                            rows = [r for r in rows if r.ts_ms > last_spot_ts]

                        if rows:
                            _append_csv(out_spot, rows)
                            print(
                                f"  [{done}/{total_jobs}] {f}/spot: appended {len(rows):,} rows to {out_spot.name} "
                                f"in {time.time()-t0:.1f}s",
                                flush=True,
                            )
                        else:
                            print(f"  [{done}/{total_jobs}] {f}/spot: no new rows", flush=True)
                    except Exception as e:
                        print(f"  [{done}/{total_jobs}] {f}/spot: ERROR {type(e).__name__}: {e}", flush=True)

            # q1 / q2 via continuous futures
            for series in ("q1", "q2"):
                done += 1
                out_fut = folder / f"{pair}_{series}.csv"
                if args.overwrite and out_fut.exists() and not rebuild_from_start:
                    out_fut.unlink()

                if rebuild_from_start:
                    since_fut = explicit_start_ms
                    last_fut_ts = None
                else:
                    since_fut, last_fut_ts = _resolve_since_ms(out_fut, tf, explicit_start_ms, bootstrap_since_ms)

                if since_fut > until_ms:
                    print(f"  [{done}/{total_jobs}] {f}/{series}: up-to-date", flush=True)
                    continue

                ctype = CONT_TYPE_MAP[series]
                t0 = time.time()
                try:
                    fut = fetch_continuous_bars(
                        pair=pair,
                        contract_type=ctype,
                        interval=tf,
                        since_ms=since_fut,
                        until_ms=until_ms,
                        limit=args.limit,
                        max_pages=args.max_pages,
                        session=sess,
                    )
                    if not fut:
                        if rebuild_from_start:
                            _write_csv(out_fut, [])
                            print(
                                f"  [{done}/{total_jobs}] {f}/{series}: rebuilt empty file (no data in window)",
                                flush=True,
                            )
                        else:
                            print(f"  [{done}/{total_jobs}] {f}/{series}: no data (best effort skip)", flush=True)
                        continue

                    rows = convert_rows(pair, series, tf, fut, ladder)
                    if rebuild_from_start:
                        _write_csv(out_fut, rows)
                        print(
                            f"  [{done}/{total_jobs}] {f}/{series}: rebuilt {len(rows):,} rows in {out_fut.name} "
                            f"for [{_fmt_utc(since_fut)} -> {_fmt_utc(until_ms)}] in {time.time()-t0:.1f}s",
                            flush=True,
                        )
                    else:
                        if last_fut_ts is not None:
                            rows = [r for r in rows if r.ts_ms > last_fut_ts]

                        if not rows:
                            print(f"  [{done}/{total_jobs}] {f}/{series}: no new rows", flush=True)
                            continue

                        _append_csv(out_fut, rows)
                        print(
                            f"  [{done}/{total_jobs}] {f}/{series}: appended {len(rows):,} rows to {out_fut.name} "
                            f"in {time.time()-t0:.1f}s",
                            flush=True,
                        )
                except Exception as e:
                    print(f"  [{done}/{total_jobs}] {f}/{series}: ERROR {type(e).__name__}: {e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
