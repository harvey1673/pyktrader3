"""Export bundled pysystemtrade CSV history; see docs/pysystemtrade_history.md.

Requires pandas and pyarrow. This is an additive price-change dataset, not an
OHLC or percentage-return-compatible replacement for the domestic futures cache.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow


def read_daily(path, cutoff, audit, hashes):
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError(f"Invalid timestamps: {path}")
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate source timestamps: {path}")
    frame = frame.sort_index()
    frame = frame.loc[frame.index < cutoff + pd.Timedelta(days=1)]
    dates = frame.index.normalize()
    audit.append(dict(folder=path.parent.name, instrument=path.stem,
                      rows=len(frame), first=str(frame.index.min()),
                      last=str(frame.index.max()),
                      extra_intraday_rows=int(dates.duplicated().sum()),
                      null_cells=int(frame.isna().sum().sum())))
    # Select an actual final row, not groupby.last() which mixes non-null cells
    # from different timestamps (and potentially different contracts).
    frame = frame.loc[~dates.duplicated(keep="last")].copy()
    frame.index = frame.index.normalize()
    frame.index.name = "date"
    return frame


def build(source, output, as_of):
    cutoff = pd.Timestamp(as_of).normalize()
    output.mkdir(parents=True, exist_ok=True)
    audit, hashes, markets = [], {}, {}
    config_path = source / "csvconfig/instrumentconfig.csv"
    metadata = pd.read_csv(config_path).set_index("Instrument")
    hashes[str(config_path)] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if metadata.index.duplicated().any():
        raise ValueError("Duplicate instrument configuration")
    for path in sorted((source / "multiple_prices_csv").glob("*.csv")):
        code = path.stem
        raw = read_daily(path, cutoff, audit, hashes)
        adj = read_daily(source / "adjusted_prices_csv" / path.name, cutoff, audit, hashes)
        d = raw.rename(columns={"PRICE": "raw_close", "PRICE_CONTRACT": "contract",
                               "CARRY": "carry", "CARRY_CONTRACT": "carry_contract",
                               "FORWARD": "forward", "FORWARD_CONTRACT": "forward_contract"})
        d = d.join(adj.rename(columns={"price": "close"}), how="outer")
        for col in ["contract", "carry_contract", "forward_contract"]:
            vals = pd.to_numeric(d[col], errors="raise")
            if ((vals.dropna() % 1) != 0).any():
                raise ValueError(f"Non-integral contract identifier in {code}")
            d[col] = vals.astype("Int64").astype("string")
        d["contmth"] = pd.to_numeric(d["contract"], errors="raise").floordiv(100).astype("Int64")
        d["adjustment"] = d["close"] - d["raw_close"]
        markets[code + "c1"] = d
    if not markets:
        raise ValueError("No futures files found")
    futures = pd.concat(markets, axis=1, sort=True).sort_index()
    futures.columns.names = [None, None]
    fx = {}
    for path in sorted((source / "fx_prices_csv").glob("*.csv")):
        fx[path.stem] = read_daily(path, cutoff, audit, hashes).iloc[:, 0]
    fx = pd.DataFrame(fx).sort_index()
    tag = cutoff.strftime("%Y%m%d")
    paths = {"futures": output / f"fut_d_pysystemtrade_{tag}.parquet",
             "fx": output / f"fx_pysystemtrade_{tag}.parquet"}
    for key, frame in [("futures", futures), ("fx", fx)]:
        # Run with the consuming environment (D:/miniconda3/python.exe).
        # A same-version round trip alone does not prove older-reader support.
        # Validate a temporary file before replacing an existing export.
        temporary = paths[key].with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, engine="pyarrow", write_statistics=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(temporary))
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        temporary.replace(paths[key])
    metadata.to_csv(output / "instrument_metadata.csv")
    profile = pd.DataFrame(audit)
    profile.to_csv(output / "source_profile.csv", index=False)
    missing = sorted(set(p[:-2] for p in markets) - set(metadata.index))
    summary = dict(as_of=str(cutoff.date()), source=str(source), markets=len(markets),
                   pandas_version=pd.__version__, pyarrow_version=pyarrow.__version__,
                   futures_shape=list(futures.shape), fx_pairs=len(fx.columns),
                   first_date=str(futures.index.min()), last_date=str(futures.index.max()),
                   fx_last_date=str(fx.index.max()), missing_metadata=missing,
                   adjustment="additive Panama; close is supplied adjusted price",
                   daily_policy="last actual row per source calendar date; no timezone conversion or filling",
                   source_sha256=hashes, files={k: str(v) for k, v in paths.items()})
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    print(profile.groupby("folder").agg(files=("instrument", "size"), first=("first", "min"),
                                        last=("last", "max"), intraday=("extra_intraday_rows", "sum")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("C:/dev/pysystemtrade/data/futures"))
    parser.add_argument("--output", type=Path, default=Path("output/pysystemtrade_history"))
    parser.add_argument("--as-of", default="2026-09-14")
    args = parser.parse_args()
    build(args.source, args.output, args.as_of)
