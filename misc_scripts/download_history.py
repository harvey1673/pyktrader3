"""
Download and probe historical crypto OHLCV data.

Focus: data download only (no DSB conversion).

Supports:
- CCXT mode (generic): OKX/Binance and other exchanges supported by ccxt.
- OKX REST native mode: uses /api/v5/market/history-candles with cursor paging
  to pull as much history as exposed by OKX.

Usage examples:

1) Probe availability (earliest/latest) for multiple timeframes:
   python download_history.py probe --exchange okx --symbol BTC-USDT-SWAP --timeframes 1d 1h 5m --mode auto
   python download_history.py probe --exchange binance --symbol BTC/USDT:USDT --timeframes 1d 1h 5m --mode auto

2) Download max history to CSV:
   python download_history.py download --exchange okx --symbol BTC-USDT-SWAP --timeframe 1h --mode auto --output ../Crypto_History/okx_btcusdtswap_1h.csv
   python download_history.py download --exchange binance --symbol BTC/USDT:USDT --timeframe 5m --mode auto --output ../Crypto_History/binance_btcusdt_5m.csv

3) Download a bounded date range:
   python download_history.py download --exchange binance --symbol BTC/USDT:USDT --timeframe 1h --since 2020-01-01 --until 2024-12-31 --output ./btc_1h_2020_2024.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

try:
    import ccxt
except Exception:  # pragma: no cover
    ccxt = None


@dataclass
class Candle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_utc_date(s: str) -> int:
    """Parse date string into UTC epoch ms. Accepts YYYY-MM-DD and ISO strings."""
    s = s.strip()
    if len(s) == 10:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    s2 = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _fmt_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _tf_ms(tf: str) -> int:
    unit = tf[-1].lower()
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    if unit == "w":
        return n * 7 * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def _dedup_sort(candles: Iterable[Candle]) -> List[Candle]:
    uniq: Dict[int, Candle] = {}
    for c in candles:
        uniq[c.ts_ms] = c
    return [uniq[k] for k in sorted(uniq.keys())]


def _write_csv(path: Path, candles: List[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "datetime_utc", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([
                c.ts_ms,
                datetime.fromtimestamp(c.ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                f"{c.open:.16g}",
                f"{c.high:.16g}",
                f"{c.low:.16g}",
                f"{c.close:.16g}",
                f"{c.volume:.16g}",
            ])


def _make_ccxt_exchange(exchange_id: str, sandbox: bool = False):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Install with: pip install ccxt")
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"ccxt exchange not found: {exchange_id}")
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if sandbox and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)
    ex.load_markets()
    return ex


def probe_ccxt(exchange_id: str, symbol: str, timeframes: List[str]) -> Dict[str, Dict[str, Optional[int]]]:
    ex = _make_ccxt_exchange(exchange_id)
    if symbol not in ex.markets:
        raise ValueError(f"Symbol not found on {exchange_id}: {symbol}")

    out: Dict[str, Dict[str, Optional[int]]] = {}
    now = int(time.time() * 1000)
    # Probe with multiple historical anchors to infer earliest available date.
    anchors = [
        _parse_utc_date("2013-01-01"),
        _parse_utc_date("2015-01-01"),
        _parse_utc_date("2017-01-01"),
        _parse_utc_date("2019-01-01"),
        _parse_utc_date("2021-01-01"),
        _parse_utc_date("2023-01-01"),
    ]

    for tf in timeframes:
        if tf not in ex.timeframes:
            out[tf] = {"earliest": None, "latest": None, "count": 0}
            continue
        earliest = None
        latest = None
        count = 0

        # First, recent slice for latest timestamp.
        recent = ex.fetch_ohlcv(symbol, timeframe=tf, limit=200)
        if recent:
            latest = recent[-1][0]

        # Then historical anchors for earliest estimate.
        for anc in anchors:
            if anc > now:
                continue
            bars = ex.fetch_ohlcv(symbol, timeframe=tf, since=anc, limit=1000)
            if bars:
                first = bars[0][0]
                earliest = first if earliest is None else min(earliest, first)
                count += len(bars)
            time.sleep(max(getattr(ex, "rateLimit", 50) / 1000.0, 0.05))

        out[tf] = {"earliest": earliest, "latest": latest, "count": count}

    return out


def download_ccxt(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: Optional[int],
    until_ms: Optional[int],
    limit: int = 1000,
    max_pages: int = 100000,
) -> List[Candle]:
    ex = _make_ccxt_exchange(exchange_id)
    if symbol not in ex.markets:
        raise ValueError(f"Symbol not found on {exchange_id}: {symbol}")
    if timeframe not in ex.timeframes:
        raise ValueError(f"Timeframe {timeframe} not supported on {exchange_id}")

    step = _tf_ms(timeframe)
    now_ms = int(time.time() * 1000)
    if since_ms is None:
        # Old but realistic default anchor; many exchanges reject since=0.
        since_ms = _parse_utc_date("2013-01-01")
    if until_ms is None:
        until_ms = now_ms

    all_rows: List[Candle] = []
    cursor = since_ms
    pages = 0

    while cursor <= until_ms and pages < max_pages:
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        pages += 1
        if not rows:
            # No rows from this cursor; advance by a large jump to escape gaps.
            cursor += step * limit
            if cursor > until_ms:
                break
            time.sleep(max(getattr(ex, "rateLimit", 50) / 1000.0, 0.05))
            continue

        candles = [
            Candle(
                ts_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
            )
            for r in rows
            if r and len(r) >= 6
        ]

        # keep only requested interval
        for c in candles:
            if c.ts_ms < since_ms:
                continue
            if c.ts_ms > until_ms:
                continue
            all_rows.append(c)

        last_ts = int(rows[-1][0])
        next_cursor = last_ts + step
        if next_cursor <= cursor:
            next_cursor = cursor + step
        cursor = next_cursor

        time.sleep(max(getattr(ex, "rateLimit", 50) / 1000.0, 0.05))

    return _dedup_sort(all_rows)


_OKX_BAR_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
    "1w": "1W",
}


def _okx_fetch_history_batch(inst_id: str, tf: str, after: Optional[int], limit: int) -> List[Candle]:
    bar = _OKX_BAR_MAP.get(tf)
    if bar is None:
        raise ValueError(f"Unsupported OKX timeframe for REST mode: {tf}")

    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(limit),
    }
    if after is not None:
        params["after"] = str(after)

    url = "https://www.okx.com/api/v5/market/history-candles"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error {data.get('code')}: {data.get('msg')}")

    rows = data.get("data", [])
    # rows are descending (newest -> oldest)
    out = []
    for r in rows:
        # [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]
        out.append(
            Candle(
                ts_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
            )
        )
    return out


def probe_okx_rest(inst_id: str, timeframes: List[str]) -> Dict[str, Dict[str, Optional[int]]]:
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for tf in timeframes:
        try:
            newest_batch = _okx_fetch_history_batch(inst_id, tf, after=None, limit=100)
            if not newest_batch:
                out[tf] = {"earliest": None, "latest": None, "count": 0}
                continue

            latest = max(c.ts_ms for c in newest_batch)
            oldest = min(c.ts_ms for c in newest_batch)

            # Walk backwards a limited number of pages to estimate depth quickly.
            cursor = oldest
            seen = list(newest_batch)
            for _ in range(30):
                older = _okx_fetch_history_batch(inst_id, tf, after=cursor, limit=100)
                if not older:
                    break
                old_min = min(c.ts_ms for c in older)
                if old_min >= cursor:
                    break
                seen.extend(older)
                cursor = old_min
                time.sleep(0.08)

            earliest = min(c.ts_ms for c in seen)
            out[tf] = {"earliest": earliest, "latest": latest, "count": len(seen)}
        except Exception:
            out[tf] = {"earliest": None, "latest": None, "count": 0}
    return out


def download_okx_rest(
    inst_id: str,
    timeframe: str,
    since_ms: Optional[int],
    until_ms: Optional[int],
    limit: int = 100,
    max_pages: int = 200000,
) -> List[Candle]:
    if timeframe not in _OKX_BAR_MAP:
        raise ValueError(f"Unsupported timeframe in OKX REST mode: {timeframe}")

    now_ms = int(time.time() * 1000)
    if since_ms is None:
        since_ms = _parse_utc_date("2013-01-01")
    if until_ms is None:
        until_ms = now_ms

    all_rows: List[Candle] = []
    pages = 0
    cursor_after = None

    while pages < max_pages:
        batch = _okx_fetch_history_batch(inst_id, timeframe, after=cursor_after, limit=limit)
        pages += 1
        if not batch:
            break

        oldest = min(c.ts_ms for c in batch)

        # Batch comes newest->oldest; keep rows within range.
        for c in batch:
            if since_ms <= c.ts_ms <= until_ms:
                all_rows.append(c)

        # If oldest already older than requested since, we can stop soon after this page.
        if oldest <= since_ms:
            break

        # Cursor must move to older side.
        next_after = oldest
        if cursor_after is not None and next_after >= cursor_after:
            break
        cursor_after = next_after
        time.sleep(0.08)

    return _dedup_sort(all_rows)


def _print_probe_result(exchange: str, symbol: str, res: Dict[str, Dict[str, Optional[int]]]) -> None:
    print(f"\nProbe result: exchange={exchange} symbol={symbol}")
    for tf, info in res.items():
        e = info.get("earliest")
        l = info.get("latest")
        c = info.get("count", 0)
        e_txt = _fmt_ms(e) if e is not None else "N/A"
        l_txt = _fmt_ms(l) if l is not None else "N/A"
        print(f"  {tf:>4}  earliest={e_txt}  latest={l_txt}  sampled_rows={c}")


def _auto_mode(exchange: str) -> str:
    # For OKX, native endpoint tends to expose deeper back-history reliably.
    if exchange.lower() == "okx":
        return "okx-rest"
    return "ccxt"


def main():
    ap = argparse.ArgumentParser(description="Crypto history probe/downloader")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="probe earliest/latest availability")
    p_probe.add_argument("--exchange", required=True, help="okx | binance | any ccxt exchange id")
    p_probe.add_argument("--symbol", required=True, help="CCXT symbol (e.g. BTC/USDT:USDT) or OKX instId when using okx-rest")
    p_probe.add_argument("--timeframes", nargs="+", default=["1d", "1h", "5m"], help="timeframes to probe")
    p_probe.add_argument("--mode", default="auto", choices=["auto", "ccxt", "okx-rest"], help="download backend")

    p_dl = sub.add_parser("download", help="download candles to CSV")
    p_dl.add_argument("--exchange", required=True, help="okx | binance | any ccxt exchange id")
    p_dl.add_argument("--symbol", required=True, help="CCXT symbol (e.g. BTC/USDT:USDT) or OKX instId in okx-rest mode")
    p_dl.add_argument("--timeframe", required=True, help="e.g. 1d, 1h, 5m")
    p_dl.add_argument("--since", default=None, help="UTC date/time, e.g. 2019-01-01 or 2019-01-01T00:00:00Z")
    p_dl.add_argument("--until", default=None, help="UTC date/time")
    p_dl.add_argument("--output", required=True, help="output CSV file path")
    p_dl.add_argument("--mode", default="auto", choices=["auto", "ccxt", "okx-rest"], help="download backend")
    p_dl.add_argument("--limit", type=int, default=1000, help="batch size; okx-rest max 100")
    p_dl.add_argument("--max-pages", type=int, default=200000, help="max page iterations")

    args = ap.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = _auto_mode(args.exchange)

    if args.cmd == "probe":
        if mode == "okx-rest":
            res = probe_okx_rest(args.symbol, args.timeframes)
        else:
            res = probe_ccxt(args.exchange, args.symbol, args.timeframes)
        _print_probe_result(args.exchange, args.symbol, res)
        return

    # download
    since_ms = _parse_utc_date(args.since) if args.since else None
    until_ms = _parse_utc_date(args.until) if args.until else None

    if mode == "okx-rest":
        limit = min(max(1, args.limit), 100)
        rows = download_okx_rest(
            inst_id=args.symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            max_pages=args.max_pages,
        )
    else:
        rows = download_ccxt(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=args.limit,
            max_pages=args.max_pages,
        )

    out = Path(args.output)
    _write_csv(out, rows)
    if rows:
        print(
            f"Downloaded {len(rows)} rows: {_fmt_ms(rows[0].ts_ms)} -> {_fmt_ms(rows[-1].ts_ms)}\n"
            f"Saved to {out}"
        )
    else:
        print(f"No rows downloaded. Saved empty file to {out}")


if __name__ == "__main__":
    main()
