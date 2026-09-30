"""Research OI/volume conditioning of the production price-trend signals.

This module is deliberately kept under ``tests`` while the signal ideas are
experimental.  It combines the dated c1/c2 daily parquet with product-level
WTPY aggregates produced by
``pycmqlib3.utility.process_wt_data.aggregate_product_oi_volume``.

The production momentum definitions are read from ``signal_repo.signal_store``
and evaluated in two separate families:

* time-series signals (names without ``_xdemean``), and
* cross-sectional signals (the corresponding daily demeaned variants).

Typical usage::

    D:\\miniconda3\\python.exe tests\\oi_volume_trend_research.py \
        --price-file C:\\dev\\data\\fut_d_20260918.parquet \
        --start 2010-01-01 --validation-start 2019-01-01 \
        --oos-start 2024-01-01 --end 2026-09-18

The script writes bounded CSV evidence, charts, and a Markdown summary to a
``C:/dev/data/output/oi_volume_trend_research`` by default.  It never changes the WTPY store or production
signal configuration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WTPY_ROOT = Path("C:/dev/wtpy")
for _path in (REPO_ROOT, WTPY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pycmqlib3.analytics import tstool  # noqa: E402
from pycmqlib3.strategy.signal_repo import (  # noqa: E402
    BROAD_MKTS,
    get_funda_signal_from_store,
    signal_store,
)
from pycmqlib3.utility.process_wt_data import (  # noqa: E402
    aggregate_product_oi_volume,
)


PNL_BDAYS = tstool.PNL_BDAYS
BASELINE_NAMES = (
    "mom_ewmac",
    "mom_momma240",
    "mom_momma20",
    "mom_hlr_st",
    "mom_hlr_mt",
    "mom_hlr_lt",
)
REQUIRED_FRONT_FIELDS = ("close", "volume", "openInterest")
FEATURE_WINDOWS = (252, 488)
FORWARD_HORIZONS = (1, 5, 20)


@dataclass(frozen=True)
class ResearchConfig:
    price_file: Path
    start: pd.Timestamp
    end: pd.Timestamp
    output_dir: Path
    products: tuple[str, ...]
    aggregate_cache: Path
    validation_start: pd.Timestamp
    oos_start: pd.Timestamp
    contract_period: str = "12m"
    cost_bps: float = 2.0
    refresh_aggregate: bool = False


def _as_timestamp(value: str | dt.date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def available_products(columns: pd.MultiIndex) -> list[str]:
    """Return products that have both c1 and c2 daily fields."""
    level0 = set(columns.get_level_values(0))
    products = []
    for product in BROAD_MKTS:
        if f"{product}c1" in level0 and f"{product}c2" in level0:
            products.append(product)
    return products


def load_front_history(
    price_file: Path,
    products: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load c1 prices and c1+c2 OI/volume from the dated futures parquet."""
    frame = pd.read_parquet(price_file)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Expected a two-level futures parquet column index.")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    usable = available_products(frame.columns)
    if products is not None:
        requested = list(dict.fromkeys(products))
        usable = [product for product in requested if product in usable]
    if not usable:
        raise ValueError("No requested products have both c1 and c2 history.")

    close = pd.DataFrame(index=frame.index)
    c1_oi = pd.DataFrame(index=frame.index)
    c2_oi = pd.DataFrame(index=frame.index)
    c1_volume = pd.DataFrame(index=frame.index)
    c2_volume = pd.DataFrame(index=frame.index)
    execution_price = pd.DataFrame(index=frame.index)
    execution_source = pd.DataFrame(index=frame.index, dtype=object)
    for product in usable:
        missing = [
            (f"{product}{leg}", field)
            for leg in ("c1", "c2")
            for field in REQUIRED_FRONT_FIELDS
            if (f"{product}{leg}", field) not in frame.columns
        ]
        if missing:
            continue
        close[product] = pd.to_numeric(frame[(f"{product}c1", "close")], errors="coerce")
        c1_oi[product] = pd.to_numeric(
            frame[(f"{product}c1", "openInterest")], errors="coerce"
        )
        c2_oi[product] = pd.to_numeric(
            frame[(f"{product}c2", "openInterest")], errors="coerce"
        )
        c1_volume[product] = pd.to_numeric(
            frame[(f"{product}c1", "volume")], errors="coerce"
        )
        c2_volume[product] = pd.to_numeric(
            frame[(f"{product}c2", "volume")], errors="coerce"
        )
        selected = pd.Series(np.nan, index=frame.index, dtype=float)
        selected_source = pd.Series(pd.NA, index=frame.index, dtype="object")
        # Production notebook convention, extended with n310 as the second
        # eligible night-session window requested for this research.
        for field in ("n305", "n310", "a1505", "a1535", "d_twap", "close"):
            column = (f"{product}c1", field)
            if column not in frame.columns:
                continue
            candidate = pd.to_numeric(frame[column], errors="coerce").where(
                lambda values: values > 0
            )
            use = selected.isna() & candidate.notna()
            selected.loc[use] = candidate.loc[use]
            selected_source.loc[use] = field
        before_fill = selected.notna()
        selected = selected.ffill()
        selected_source.loc[~before_fill & selected.notna()] = "ffill"
        execution_price[product] = selected
        execution_source[product] = selected_source

    valid_products = [product for product in usable if product in close.columns]
    return {
        "close": close[valid_products],
        "c1_oi": c1_oi[valid_products],
        "c2_oi": c2_oi[valid_products],
        "c1_volume": c1_volume[valid_products],
        "c2_volume": c2_volume[valid_products],
        "front_oi": (c1_oi + c2_oi)[valid_products],
        "front_volume": (c1_volume + c2_volume)[valid_products],
        "execution_price": execution_price[valid_products],
        "execution_source": execution_source[valid_products],
    }


def build_aggregate_panel(
    products: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract_period: str = "12m",
    loader: Callable[..., pd.DataFrame] = aggregate_product_oi_volume,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aggregate WTPY OI/volume, retaining per-product failures."""
    series: dict[tuple[str, str], pd.Series] = {}
    failures: list[dict[str, str]] = []
    for number, product in enumerate(products, start=1):
        print(f"[{number:02d}/{len(products):02d}] aggregate {product}")
        try:
            product_frame = loader(
                product,
                start_date=start.date(),
                end_date=end.date(),
                freq="d",
                contract_period=contract_period,
            ).copy()
        except Exception as exc:  # retain partial research coverage
            failures.append({"product": product, "error": str(exc)})
            continue
        product_frame.index = pd.to_datetime(product_frame.index)
        for field in ("openInterest", "volume"):
            if field in product_frame:
                series[(product, field)] = pd.to_numeric(
                    product_frame[field], errors="coerce"
                )
    if not series:
        raise ValueError("No aggregate WTPY OI/volume series could be loaded.")
    panel = pd.concat(series, axis=1).sort_index()
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["product", "field"])
    return panel, pd.DataFrame(failures, columns=["product", "error"])


def load_or_build_aggregate_panel(config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.aggregate_cache.parent.mkdir(parents=True, exist_ok=True)
    failure_path = config.aggregate_cache.with_suffix(".failures.csv")
    if config.aggregate_cache.exists() and not config.refresh_aggregate:
        panel = pd.read_parquet(config.aggregate_cache)
        failures = (
            pd.read_csv(failure_path)
            if failure_path.exists()
            else pd.DataFrame(columns=["product", "error"])
        )
        return panel, failures

    # Extra history is required for 488-day transforms before the research start.
    load_start = config.start - pd.Timedelta(days=900)
    panel, failures = build_aggregate_panel(
        config.products,
        load_start,
        config.end,
        contract_period=config.contract_period,
    )
    panel.to_parquet(config.aggregate_cache)
    failures.to_csv(failure_path, index=False)
    return panel, failures


def _field(panel: pd.DataFrame, name: str, products: Sequence[str]) -> pd.DataFrame:
    available = [product for product in products if (product, name) in panel.columns]
    result = panel.loc[:, [(product, name) for product in available]].copy()
    result.columns = available
    return result


def _apply_by_column(frame: pd.DataFrame, function: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: function(frame[column].dropna()).reindex(frame.index) for column in frame},
        index=frame.index,
    )


def _mean_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    return sum(frame.fillna(0.0) for frame in frames) / sum(
        frame.notna().astype(float) for frame in frames
    ).replace(0.0, np.nan)


def build_feature_panels(
    front: Mapping[str, pd.DataFrame],
    aggregate_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create point-in-time OI, volume, volatility and relationship features."""
    close = front["close"].copy()
    products = [product for product in close if (product, "openInterest") in aggregate_panel]
    close = close[products]
    aggregate_oi = _field(aggregate_panel, "openInterest", products).reindex(close.index)
    aggregate_volume = _field(aggregate_panel, "volume", products).reindex(close.index)
    front_oi = front["front_oi"][products].reindex(close.index)
    front_volume = front["front_volume"][products].reindex(close.index)

    returns = close.pct_change(fill_method=None)
    vol20 = returns.rolling(20, min_periods=20).std() * math.sqrt(PNL_BDAYS)
    log_oi = np.log(aggregate_oi.where(aggregate_oi > 0))
    log_volume = np.log(aggregate_volume.where(aggregate_volume > 0))
    oi_change_1d = log_oi.diff()
    volume_change_1d = log_volume.diff()

    vol_rank_1y = _apply_by_column(vol20, lambda s: tstool.pct_score(s, 252))
    vol_rank_2y = _apply_by_column(vol20, lambda s: tstool.pct_score(s, 488))
    vol_zscore = _apply_by_column(vol20, lambda s: tstool.zscore_roll(s, 252)).clip(-2, 2) / 2
    vol_hlratio = _apply_by_column(vol20, lambda s: tstool.hlratio(s, 252)).clip(-1, 1)
    vol_regt = _apply_by_column(
        vol20,
        lambda s: tstool.rolling_trend(s, 60, return_mode="t-stat", log=False),
    ).clip(-5, 5) / 5
    vol_regime = _mean_frames(vol_rank_1y, vol_rank_2y).clip(-1, 1)

    oi_change_5d = log_oi.diff(5)
    oi_change_20d = log_oi.diff(20)
    oi_mom_5 = _apply_by_column(oi_change_5d, lambda s: tstool.zscore_roll(s, 252))
    oi_mom_20 = _apply_by_column(oi_change_20d, lambda s: tstool.pct_score(s, 252) * 2 - 1)
    oi_momentum = _mean_frames(oi_mom_5.clip(-2, 2) / 2, oi_mom_20).clip(-1, 1)
    oi_breakout = _apply_by_column(log_oi, lambda s: tstool.hlratio(s, 252)).clip(-1, 1)
    oi_level_qtl = _apply_by_column(log_oi, lambda s: tstool.pct_score(s, 252)).clip(-1, 1)
    oi_level_zscore = _apply_by_column(log_oi, lambda s: tstool.zscore_roll(s, 252)).clip(-2, 2) / 2
    oi_regt = _apply_by_column(
        log_oi,
        lambda s: tstool.rolling_trend(s, 60, return_mode="t-stat", log=False),
    ).clip(-5, 5) / 5

    volume_change_5d = log_volume.diff(5)
    volume_change_20d = log_volume.diff(20)
    volume_qtl = _apply_by_column(volume_change_5d, lambda s: tstool.pct_score(s, 252) * 2 - 1)
    volume_z = _apply_by_column(volume_change_20d, lambda s: tstool.zscore_roll(s, 252)).clip(-2, 2) / 2
    volume_activity = _mean_frames(volume_qtl, volume_z).clip(-1, 1)
    volume_breakout = _apply_by_column(log_volume, lambda s: tstool.hlratio(s, 252)).clip(-1, 1)
    volume_level_qtl = _apply_by_column(log_volume, lambda s: tstool.pct_score(s, 252)).clip(-1, 1)
    volume_regt = _apply_by_column(
        log_volume,
        lambda s: tstool.rolling_trend(s, 60, return_mode="t-stat", log=False),
    ).clip(-5, 5) / 5

    price_oi_corr = returns.rolling(60, min_periods=40).corr(oi_change_1d).clip(-1, 1)
    price_volume_corr = returns.rolling(60, min_periods=40).corr(volume_change_1d).clip(-1, 1)

    front_oi_share = (front_oi / aggregate_oi).replace([np.inf, -np.inf], np.nan)
    front_volume_share = (front_volume / aggregate_volume).replace([np.inf, -np.inf], np.nan)
    front_concentration = _mean_frames(
        _apply_by_column(front_oi_share, lambda s: tstool.pct_score(s, 252) * 2 - 1),
        _apply_by_column(front_volume_share, lambda s: tstool.pct_score(s, 252) * 2 - 1),
    ).clip(-1, 1)

    c1_oi_share = (
        front["c1_oi"][products] / front_oi
    ).replace([np.inf, -np.inf], np.nan)
    c1_volume_share = (
        front["c1_volume"][products] / front_volume
    ).replace([np.inf, -np.inf], np.nan)
    c1_roll_concentration = _mean_frames(
        _apply_by_column(c1_oi_share, lambda s: tstool.zscore_roll(s, 120)).clip(-2, 2) / 2,
        _apply_by_column(c1_volume_share, lambda s: tstool.zscore_roll(s, 120)).clip(-2, 2) / 2,
    ).clip(-1, 1)

    volume_oi_turnover = np.log(
        (aggregate_volume / aggregate_oi).where(
            (aggregate_volume > 0) & (aggregate_oi > 0)
        )
    )
    turnover_qtl = _apply_by_column(
        volume_oi_turnover, lambda s: tstool.pct_score(s, 252)
    ).clip(-1, 1)

    price_regt_20 = _apply_by_column(
        close,
        lambda s: tstool.rolling_trend(s, 20, return_mode="t-stat"),
    ).clip(-5, 5) / 5
    price_regt_60 = _apply_by_column(
        close,
        lambda s: tstool.rolling_trend(s, 60, return_mode="t-stat"),
    ).clip(-5, 5) / 5
    price_trend = _mean_frames(price_regt_20, price_regt_60).clip(-1, 1)
    price_oi_divergence = (price_trend - oi_regt).clip(-2, 2) / 2
    price_volume_divergence = (price_trend - volume_regt).clip(-2, 2) / 2

    price_hlr_20 = _apply_by_column(close, lambda s: tstool.hlratio(s, 20)).clip(-1, 1)

    return {
        "close": close,
        "returns": returns,
        "vol20": vol20,
        "execution_price": front.get("execution_price", close)[products].reindex(close.index),
        "aggregate_oi": aggregate_oi,
        "aggregate_volume": aggregate_volume,
        "vol_regime": vol_regime,
        "vol_zscore": vol_zscore,
        "vol_hlratio": vol_hlratio,
        "vol_regt": vol_regt,
        "oi_momentum": oi_momentum,
        "oi_breakout": oi_breakout,
        "oi_level_qtl": oi_level_qtl,
        "oi_level_zscore": oi_level_zscore,
        "oi_regt": oi_regt,
        "volume_activity": volume_activity,
        "volume_breakout": volume_breakout,
        "volume_level_qtl": volume_level_qtl,
        "volume_regt": volume_regt,
        "price_oi_corr": price_oi_corr,
        "price_volume_corr": price_volume_corr,
        "front_concentration": front_concentration,
        "c1_roll_concentration": c1_roll_concentration,
        "turnover_qtl": turnover_qtl,
        "price_trend": price_trend,
        "price_oi_divergence": price_oi_divergence,
        "price_volume_divergence": price_volume_divergence,
        "price_hlr_20": price_hlr_20,
    }


def build_price_baselines(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reproduce the selected production momentum definitions from signal_repo."""
    spot = pd.DataFrame(
        {f"{product}_px": close[product] for product in close.columns},
        index=close.index,
    )
    baselines: dict[str, pd.DataFrame] = {}
    for name in BASELINE_NAMES:
        if name not in signal_store:
            raise KeyError(f"Production signal is missing: {name}")
        raw = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for product in close:
            raw[product] = get_funda_signal_from_store(
                spot,
                name,
                asset=product,
            ).reindex(close.index)
        baselines[name] = raw
        baselines[f"{name}_xdemean"] = tstool.xs_demean(raw)
    return baselines


def build_candidate_signals(
    baselines: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Construct a compact, predeclared set of trend and reversal candidates."""
    candidates: dict[str, pd.DataFrame] = {}
    definitions: list[dict[str, str]] = []

    def add(name: str, parent: str, family: str, thesis: str, signal: pd.DataFrame) -> None:
        if family == "cross_sectional":
            signal = tstool.xs_demean(signal)
        candidates[name] = signal.clip(-2, 2)
        definitions.append(
            {"signal": name, "parent": parent, "family": family, "thesis": thesis}
        )

    for baseline_name, baseline in baselines.items():
        is_xs = baseline_name.endswith("_xdemean")
        family = "cross_sectional" if is_xs else "time_series"
        plain_parent = baseline_name.removesuffix("_xdemean")
        add(baseline_name, plain_parent, family, "Production price-only baseline.", baseline)

        direction = np.sign(baseline)
        oi_confirmation = (direction * features["oi_momentum"]).clip(-1, 1)
        relationship_confirmation = _mean_frames(
            features["price_oi_corr"], features["price_volume_corr"]
        ).clip(-1, 1)

        variants = {
            "oi_confirm": (
                baseline * (1.0 + 0.5 * oi_confirmation),
                "Scale price trend up when aggregate OI change agrees with its direction.",
            ),
            "volume_confirm": (
                baseline * (1.0 + 0.5 * features["volume_activity"]),
                "Scale price trend with unusual aggregate volume growth.",
            ),
            "joint_confirm": (
                baseline
                * (
                    1.0
                    + 0.25 * oi_confirmation
                    + 0.15 * features["volume_activity"]
                    + 0.10 * relationship_confirmation
                ),
                "Combine directional OI, volume activity, and price/position relationships.",
            ),
            "oi_breakout_confirm": (
                baseline * (1.0 + 0.5 * direction * features["oi_breakout"]),
                "Confirm trend direction with aggregate OI level breakouts.",
            ),
            "oi_regt_confirm": (
                baseline * (1.0 + 0.5 * direction * features["oi_regt"]),
                "Confirm price direction with the regression t-stat of aggregate OI.",
            ),
            "volume_breakout_confirm": (
                baseline * (1.0 + 0.5 * features["volume_breakout"]),
                "Prefer price trends occurring with aggregate volume breakouts.",
            ),
            "volume_regt_confirm": (
                baseline * (1.0 + 0.5 * features["volume_regt"]),
                "Scale trend with the regression t-stat of aggregate volume.",
            ),
            "turnover_confirm": (
                baseline * (1.0 + 0.5 * features["turnover_qtl"]),
                "Use aggregate volume/OI turnover as a participation confirmation.",
            ),
            "corr_confirm": (
                baseline * (1.0 + 0.5 * relationship_confirmation),
                "Scale trend by rolling price/OI and price/volume correlations.",
            ),
            "high_vol_regime": (
                baseline * (1.0 + 0.5 * features["vol_regime"]),
                "Prefer trend exposure when 20-day volatility ranks high versus 1-2 years.",
            ),
            "low_vol_regime": (
                baseline * (1.0 - 0.5 * features["vol_regime"]),
                "Prefer trend exposure when 20-day volatility ranks low versus 1-2 years.",
            ),
            "front_concentration": (
                baseline * (1.0 + 0.5 * features["front_concentration"]),
                "Scale trend with the c1+c2 share of aggregate product OI and volume.",
            ),
            "front_deconcentration": (
                baseline * (1.0 - 0.5 * features["front_concentration"]),
                "Prefer trends with participation beyond c1+c2 rather than front concentration.",
            ),
            "c1_roll_concentration": (
                baseline * (1.0 + 0.5 * features["c1_roll_concentration"]),
                "Scale trend with c1 versus c2 OI/volume concentration during roll migration.",
            ),
            "divergence_filter": (
                baseline
                * (
                    1.0
                    - 0.25 * (direction * features["price_oi_divergence"]).clip(lower=0)
                    - 0.25 * (direction * features["price_volume_divergence"]).clip(lower=0)
                ),
                "Reduce trend exposure when price direction outruns OI and volume trends.",
            ),
        }
        for suffix, (signal, thesis) in variants.items():
            candidate_name = f"{baseline_name}__{suffix}"
            add(candidate_name, plain_parent, family, thesis, signal)

    # Independent continuation and exhaustion/divergence hypotheses.
    divergence = (
        -features["price_hlr_20"] * features["oi_momentum"]
    ).clip(lower=0)
    exhaustion = (
        features["vol_regime"].clip(lower=0)
        * features["volume_activity"].clip(lower=0)
    )
    reversal = -features["price_hlr_20"] * exhaustion * (0.5 + 0.5 * divergence)
    independent = {
        "continuation_oi_build": (
            features["price_trend"]
            * (features["price_trend"] * features["oi_momentum"]).clip(lower=0),
            "Follow price trends only when aggregate OI builds in the same direction.",
        ),
        "continuation_volume": (
            features["price_trend"] * features["volume_activity"].clip(lower=0),
            "Follow price trends only when aggregate volume expands.",
        ),
        "continuation_joint": (
            features["price_trend"]
            * _mean_frames(
                (features["price_trend"] * features["oi_momentum"]).clip(lower=0),
                features["volume_activity"].clip(lower=0),
                features["turnover_qtl"].clip(lower=0),
            ),
            "Require aligned OI build, volume expansion, and participation turnover.",
        ),
        "reversal_exhaustion": (
            reversal,
            "Fade short-term price extremes when volatility and volume are unusually high, "
            "with extra weight for price/OI divergence.",
        ),
        "reversal_oi_divergence": (
            -features["price_trend"]
            * (features["price_trend"] * features["oi_momentum"]).clip(upper=0).abs(),
            "Fade price trends when aggregate OI moves against their direction.",
        ),
        "reversal_volume_climax": (
            -features["price_hlr_20"]
            * features["volume_breakout"].clip(lower=0)
            * features["vol_regime"].clip(lower=0),
            "Fade price extremes on joint volume and volatility breakouts.",
        ),
        "reversal_crowding": (
            -features["price_hlr_20"]
            * features["oi_breakout"].clip(lower=0)
            * features["front_concentration"].clip(lower=0),
            "Fade price extremes when aggregate OI is crowded and concentrated in c1+c2.",
        ),
        "reversal_roll_climax": (
            -features["price_hlr_20"]
            * features["c1_roll_concentration"].abs()
            * features["volume_activity"].clip(lower=0),
            "Fade extremes around unusually concentrated c1/c2 roll migration with heavy volume.",
        ),
    }
    for name, (signal, thesis) in independent.items():
        add(name, "independent", "time_series", thesis, signal)
        add(f"{name}_xdemean", "independent", "cross_sectional", thesis, signal)
    return candidates, pd.DataFrame(definitions)


def split_dates(
    index: pd.DatetimeIndex,
    validation_start: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Use fixed calendar splits so the out-of-sample period stays locked."""
    unique = pd.DatetimeIndex(index.unique()).sort_values()
    if len(unique) < 12:
        raise ValueError("At least 12 aligned dates are required for research splits.")
    if not unique[0] < validation_start < oos_start <= unique[-1]:
        raise ValueError(
            "Expected first_date < validation_start < oos_start <= last_date; "
            f"got {unique[0].date()}, {validation_start.date()}, "
            f"{oos_start.date()}, {unique[-1].date()}."
        )
    validation_pos = int(unique.searchsorted(validation_start, side="left"))
    oos_pos = int(unique.searchsorted(oos_start, side="left"))
    if validation_pos == 0 or oos_pos <= validation_pos:
        raise ValueError("Calendar split boundaries do not leave three non-empty samples.")
    return {
        "train": (unique[0], unique[validation_pos - 1]),
        "validation": (unique[validation_pos], unique[oos_pos - 1]),
        "oos": (unique[oos_pos], unique[-1]),
        "full": (unique[0], unique[-1]),
    }


def _max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.fillna(0).cumsum()
    return float((equity - equity.cummax()).min())


def _pnl_metrics(
    pnl: pd.Series,
    turnover: pd.Series,
    asset_pnl: pd.DataFrame,
) -> dict[str, float]:
    pnl = pnl.dropna()
    if len(pnl) < 20 or pnl.std() == 0:
        return {key: np.nan for key in (
            "sharpe", "annual_return", "annual_vol", "max_drawdown",
            "mean_turnover", "pnl_per_turnover", "positive_asset_share", "n_days",
        )}
    asset_sharpe = asset_pnl.mean() / asset_pnl.std() * math.sqrt(PNL_BDAYS)
    annual_return = float(pnl.mean() * PNL_BDAYS)
    annual_vol = float(pnl.std() * math.sqrt(PNL_BDAYS))
    mean_turnover = float(turnover.mean())
    return {
        "sharpe": annual_return / annual_vol if annual_vol else np.nan,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "max_drawdown": _max_drawdown(pnl),
        "mean_turnover": mean_turnover,
        "pnl_per_turnover": float(pnl.mean() / mean_turnover) if mean_turnover else np.nan,
        "positive_asset_share": float((asset_sharpe > 0).mean()),
        "n_days": int(len(pnl)),
    }


def evaluate_signal(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    vol20: pd.DataFrame,
    cost_bps: float,
    family: str,
    close_prices: pd.DataFrame | None = None,
    execution_prices: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Risk-scale and execute T's signal at T+1's point-in-time price.

    When close and execution prices are supplied, PNL mirrors the production
    notebook's ``MetricsBase.calculate_daily_pnl`` convention: shifted
    holdings earn close-to-close PNL and position changes receive the
    close/execution-price adjustment on the execution date.
    """
    aligned_signal, aligned_returns = signal.align(returns, join="inner", axis=0)
    aligned_signal, aligned_returns = aligned_signal.align(aligned_returns, join="inner", axis=1)
    risk = vol20.reindex_like(aligned_signal).replace(0, np.nan)
    if family == "cross_sectional":
        aligned_signal = tstool.xs_demean(aligned_signal)
    raw_weight = aligned_signal / risk
    weights = raw_weight.div(raw_weight.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    holdings = weights.shift(1)
    signed_trade = holdings - holdings.shift(1).fillna(0.0)
    if close_prices is not None and execution_prices is not None:
        close = close_prices.reindex_like(holdings)
        execution = execution_prices.reindex_like(holdings)
        execution_adjustment = signed_trade * (close / execution - 1.0)
        gross_asset_pnl = holdings * close.pct_change(fill_method=None)
        gross_asset_pnl = gross_asset_pnl + execution_adjustment
    else:
        execution_adjustment = pd.DataFrame(0.0, index=holdings.index, columns=holdings.columns)
        gross_asset_pnl = holdings * aligned_returns
    traded = signed_trade.abs()
    cost = traded * (cost_bps / 10_000.0)
    net_asset_pnl = gross_asset_pnl - cost
    return {
        "weights": weights,
        "holdings": holdings,
        "gross_asset_pnl": gross_asset_pnl,
        "execution_adjustment": execution_adjustment,
        "net_asset_pnl": net_asset_pnl,
        "net_pnl": net_asset_pnl.sum(axis=1, min_count=1),
        "turnover": traded.sum(axis=1, min_count=1),
    }


def evaluate_candidates(
    candidates: Mapping[str, pd.DataFrame],
    definitions: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
    validation_start: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = features["returns"].loc[start:end]
    vol20 = features["vol20"].loc[start:end]
    close_prices = features.get("close")
    execution_prices = features.get("execution_price")
    splits = split_dates(
        returns.dropna(how="all").index,
        validation_start=validation_start,
        oos_start=oos_start,
    )
    definition_map = definitions.set_index("signal").to_dict("index")
    rows: list[dict[str, object]] = []
    pnl_store: dict[str, pd.Series] = {}
    turnover_store: dict[str, pd.Series] = {}

    for name, signal in candidates.items():
        metadata = definition_map[name]
        result = evaluate_signal(
            signal.loc[start:end],
            returns,
            vol20,
            cost_bps,
            str(metadata["family"]),
            close_prices=None if close_prices is None else close_prices.loc[start:end],
            execution_prices=(
                None if execution_prices is None else execution_prices.loc[start:end]
            ),
        )
        pnl_store[name] = result["net_pnl"]
        turnover_store[name] = result["turnover"]
        for split, (split_start, split_end) in splits.items():
            mask = result["net_pnl"].index.to_series().between(split_start, split_end)
            split_index = result["net_pnl"].index[mask]
            metrics = _pnl_metrics(
                result["net_pnl"].loc[split_index],
                result["turnover"].loc[split_index],
                result["net_asset_pnl"].loc[split_index],
            )
            rows.append(
                {
                    "signal": name,
                    "parent": metadata["parent"],
                    "family": metadata["family"],
                    "split": split,
                    "start": split_start,
                    "end": split_end,
                    **metrics,
                }
            )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(pnl_store),
        pd.DataFrame(turnover_store),
    )


def feature_forward_correlations(
    features: Mapping[str, pd.DataFrame],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Report time-series and cross-sectional rank ICs to forward returns."""
    returns = features["returns"]
    rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        feature = features[feature_name]
        for horizon in FORWARD_HORIZONS:
            forward = (1 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
            time_series_ic = pd.Series(
                {
                    product: feature[product].corr(forward[product], method="spearman")
                    for product in feature.columns.intersection(forward.columns)
                }
            ).dropna()
            # Spearman correlation is Pearson correlation of ranks.  Ranking
            # first avoids thousands of scipy ConstantInputWarning messages
            # and is materially faster for the daily cross section.
            feature_rank = feature.rank(axis=1, method="average", pct=True)
            forward_rank = forward.rank(axis=1, method="average", pct=True)
            cross_sectional_ic = feature_rank.corrwith(
                forward_rank, axis=1, method="pearson"
            ).dropna()
            rows.append(
                {
                    "feature": feature_name,
                    "horizon": horizon,
                    "ts_ic_median": time_series_ic.median(),
                    "ts_ic_mean": time_series_ic.mean(),
                    "ts_positive_share": (time_series_ic > 0).mean(),
                    "ts_n_products": len(time_series_ic),
                    "xs_ic_mean": cross_sectional_ic.mean(),
                    "xs_ic_tstat": (
                        cross_sectional_ic.mean()
                        / cross_sectional_ic.std()
                        * math.sqrt(len(cross_sectional_ic))
                        if cross_sectional_ic.std() > 0
                        else np.nan
                    ),
                    "xs_n_days": len(cross_sectional_ic),
                }
            )
    return pd.DataFrame(rows)


def data_quality_summary(
    front: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame],
    failures: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product in front["close"].columns:
        close = front["close"][product].loc[start:end]
        oi = features["aggregate_oi"].get(product, pd.Series(dtype=float)).loc[start:end]
        volume = features["aggregate_volume"].get(product, pd.Series(dtype=float)).loc[start:end]
        common = close.index.intersection(oi.index).intersection(volume.index)
        rows.append(
            {
                "product": product,
                "price_start": close.first_valid_index(),
                "price_end": close.last_valid_index(),
                "price_coverage": close.notna().mean(),
                "oi_coverage": oi.reindex(close.index).notna().mean(),
                "volume_coverage": volume.reindex(close.index).notna().mean(),
                "aligned_rows": int(
                    (close.reindex(common).notna() & oi.reindex(common).notna() & volume.reindex(common).notna()).sum()
                ),
                "nonpositive_oi": int((oi <= 0).sum()),
                "nonpositive_volume": int((volume <= 0).sum()),
                "aggregate_error": (
                    failures.loc[failures["product"] == product, "error"].iloc[0]
                    if not failures.empty and (failures["product"] == product).any()
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def execution_quality_summary(
    front: Mapping[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Audit the per-date execution-price fallback used by the backtest."""
    if "execution_price" not in front or "execution_source" not in front:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for product in front["close"].columns:
        close = front["close"][product].loc[start:end]
        execution = front["execution_price"][product].loc[start:end]
        source = front["execution_source"][product].loc[start:end]
        eligible = close.notna()
        denominator = int(eligible.sum())
        counts = source[eligible].value_counts()
        rows.append(
            {
                "product": product,
                "close_rows": denominator,
                "execution_rows": int((execution.notna() & eligible).sum()),
                "execution_coverage": (
                    float((execution.notna() & eligible).sum() / denominator)
                    if denominator
                    else np.nan
                ),
                "n305_rows": int(counts.get("n305", 0)),
                "n310_rows": int(counts.get("n310", 0)),
                "a1505_rows": int(counts.get("a1505", 0)),
                "a1535_rows": int(counts.get("a1535", 0)),
                "d_twap_rows": int(counts.get("d_twap", 0)),
                "close_fallback_rows": int(counts.get("close", 0)),
                "ffill_rows": int(counts.get("ffill", 0)),
                "night_share": (
                    float((counts.get("n305", 0) + counts.get("n310", 0)) / denominator)
                    if denominator
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def incremental_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[~metrics["signal"].str.contains("__")].copy()
    baseline = baseline[baseline["parent"] != "independent"]
    baseline = baseline.set_index(["signal", "split"])
    rows = []
    for row in metrics[metrics["signal"].str.contains("__")].itertuples(index=False):
        baseline_name = row.parent + ("_xdemean" if row.family == "cross_sectional" else "")
        key = (baseline_name, row.split)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        rows.append(
            {
                "signal": row.signal,
                "baseline": baseline_name,
                "family": row.family,
                "split": row.split,
                "sharpe": row.sharpe,
                "baseline_sharpe": base["sharpe"],
                "delta_sharpe": row.sharpe - base["sharpe"],
                "delta_turnover": row.mean_turnover - base["mean_turnover"],
                "delta_positive_asset_share": (
                    row.positive_asset_share - base["positive_asset_share"]
                ),
            }
        )
    return pd.DataFrame(rows)


def cost_sensitivity_metrics(
    pnl: pd.DataFrame,
    turnover: pd.DataFrame,
    configured_cost_bps: float,
    validation_start: pd.Timestamp,
    oos_start: pd.Timestamp,
    costs_bps: Sequence[float] = (0.0, 2.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Reprice the same lagged holdings at alternative linear trading costs."""
    splits = split_dates(
        pnl.dropna(how="all").index,
        validation_start=validation_start,
        oos_start=oos_start,
    )
    rows: list[dict[str, object]] = []
    for cost_bps in costs_bps:
        repriced = pnl + turnover * ((configured_cost_bps - cost_bps) / 10_000.0)
        for split, (split_start, split_end) in splits.items():
            split_pnl = repriced.loc[split_start:split_end]
            annual_return = split_pnl.mean() * PNL_BDAYS
            annual_vol = split_pnl.std() * math.sqrt(PNL_BDAYS)
            sharpe = annual_return / annual_vol.replace(0, np.nan)
            for signal in repriced.columns:
                rows.append(
                    {
                        "signal": signal,
                        "cost_bps": cost_bps,
                        "split": split,
                        "sharpe": sharpe.get(signal, np.nan),
                        "annual_return": annual_return.get(signal, np.nan),
                        "annual_vol": annual_vol.get(signal, np.nan),
                    }
                )
    return pd.DataFrame(rows)


def yearly_oos_metrics(
    pnl: pd.DataFrame,
    turnover: pd.DataFrame,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    """Expose calendar-year stability inside the held-out period."""
    rows: list[dict[str, object]] = []
    for year, year_pnl in pnl.loc[oos_start:].groupby(pnl.loc[oos_start:].index.year):
        year_turnover = turnover.reindex(year_pnl.index)
        annual_return = year_pnl.mean() * PNL_BDAYS
        annual_vol = year_pnl.std() * math.sqrt(PNL_BDAYS)
        sharpe = annual_return / annual_vol.replace(0, np.nan)
        for signal in pnl.columns:
            rows.append(
                {
                    "signal": signal,
                    "year": int(year),
                    "sharpe": sharpe.get(signal, np.nan),
                    "annual_return": annual_return.get(signal, np.nan),
                    "annual_vol": annual_vol.get(signal, np.nan),
                    "mean_turnover": year_turnover[signal].mean(),
                    "n_days": int(year_pnl[signal].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    incremental: pd.DataFrame,
    pnl: pd.DataFrame,
    oos_start: pd.Timestamp,
    block_length: int = 20,
    n_bootstrap: int = 2_000,
    seed: int = 20260923,
) -> pd.DataFrame:
    """Block-bootstrap OOS incremental mean PnL for split-robust modifiers."""
    split_delta = incremental[incremental["split"].isin(["validation", "oos"])]
    robust = split_delta.pivot_table(
        index=["signal", "baseline", "family"],
        columns="split",
        values="delta_sharpe",
    ).reset_index()
    if not {"validation", "oos"}.issubset(robust.columns):
        return pd.DataFrame()
    robust = robust[(robust["validation"] > 0) & (robust["oos"] > 0)]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for row in robust.itertuples(index=False):
        diff = (pnl[row.signal] - pnl[row.baseline]).loc[oos_start:].dropna().to_numpy()
        if len(diff) < block_length:
            continue
        starts = np.arange(len(diff) - block_length + 1)
        blocks_needed = math.ceil(len(diff) / block_length)
        samples = np.empty(n_bootstrap)
        for sample_no in range(n_bootstrap):
            chosen = rng.choice(starts, size=blocks_needed, replace=True)
            sample = np.concatenate([diff[start : start + block_length] for start in chosen])
            samples[sample_no] = sample[: len(diff)].mean() * PNL_BDAYS
        observed = diff.mean() * PNL_BDAYS
        rows.append(
            {
                "signal": row.signal,
                "baseline": row.baseline,
                "family": row.family,
                "validation_delta_sharpe": row.validation,
                "oos_delta_sharpe": row.oos,
                "oos_incremental_annual_return": observed,
                "bootstrap_ci_2_5pct": np.quantile(samples, 0.025),
                "bootstrap_ci_97_5pct": np.quantile(samples, 0.975),
                "bootstrap_probability_positive": (samples > 0).mean(),
                "block_length": block_length,
                "n_bootstrap": n_bootstrap,
                "n_days": len(diff),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "bootstrap_probability_positive", ascending=False
    )


def mature_universe_products(
    quality: pd.DataFrame,
    latest_start: str,
    minimum_aligned_rows: int,
) -> list[str]:
    """Select a fixed universe using data availability, not strategy returns."""
    price_start = pd.to_datetime(quality["price_start"], errors="coerce")
    mask = (
        price_start.le(pd.Timestamp(latest_start))
        & quality["aligned_rows"].ge(minimum_aligned_rows)
        & quality["aggregate_error"].fillna("").eq("")
    )
    return quality.loc[mask, "product"].tolist()


def write_charts(metrics: pd.DataFrame, pnl: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    oos = metrics[metrics["split"] == "oos"].dropna(subset=["sharpe"])
    top = oos.sort_values("sharpe", ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#2f5d7c" if family == "time_series" else "#c47a2c" for family in top["family"]]
    ax.barh(top["signal"], top["sharpe"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Top 2024-2026 out-of-sample net Sharpe ratios")
    ax.set_xlabel("Annualized Sharpe after configured trading cost")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "oos_sharpe.png", dpi=150)
    plt.close(fig)

    selected = list(top.head(8)["signal"])
    if selected:
        oos_start = pd.to_datetime(oos["start"]).min()
        curves = pnl.loc[oos_start:, selected].fillna(0).cumsum()
        fig, ax = plt.subplots(figsize=(12, 6))
        curves.plot(ax=ax, linewidth=1.2)
        ax.set_title("Cumulative net PnL of leading out-of-sample candidates")
        ax.set_ylabel("Return on normalized gross exposure")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(output_dir / "candidate_cumulative_pnl.png", dpi=150)
        plt.close(fig)


def write_summary(
    config: ResearchConfig,
    quality: pd.DataFrame,
    execution_quality: pd.DataFrame,
    correlations: pd.DataFrame,
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    bootstrap: pd.DataFrame,
    universe_incremental: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    yearly: pd.DataFrame,
    output_dir: Path,
) -> None:
    valid_quality = quality[quality["aligned_rows"] > 0]
    execution_rows = execution_quality["execution_rows"].sum() if not execution_quality.empty else 0
    night_rows = (
        execution_quality[["n305_rows", "n310_rows"]].sum().sum()
        if not execution_quality.empty
        else 0
    )
    candidates = incremental[incremental["split"].isin(["validation", "oos"])]
    pivot = candidates.pivot_table(
        index=["signal", "baseline", "family"],
        columns="split",
        values="delta_sharpe",
    ).reset_index()
    if {"validation", "oos"}.issubset(pivot.columns):
        robust = pivot[(pivot["validation"] > 0) & (pivot["oos"] > 0)].sort_values(
            "oos", ascending=False
        )
    else:
        robust = pd.DataFrame()

    robust_names = robust["signal"].tolist() if not robust.empty else []
    fixed_robust = universe_incremental[
        universe_incremental["signal"].isin(robust_names)
        & universe_incremental["split"].isin(["validation", "oos"])
    ]
    if not fixed_robust.empty:
        fixed_robust = fixed_robust.pivot_table(
            index=["cohort", "n_products", "signal"],
            columns="split",
            values="delta_sharpe",
        ).reset_index()

    cost_delta = pd.DataFrame()
    if robust_names:
        robust_map = robust[["signal", "baseline"]]
        candidate_cost = cost_sensitivity[
            (cost_sensitivity["split"] == "oos")
            & cost_sensitivity["signal"].isin(robust_names)
        ].merge(robust_map, on="signal")
        baseline_cost = cost_sensitivity[cost_sensitivity["split"] == "oos"].rename(
            columns={"signal": "baseline", "sharpe": "baseline_sharpe"}
        )
        candidate_cost = candidate_cost.merge(
            baseline_cost[["baseline", "cost_bps", "baseline_sharpe"]],
            on=["baseline", "cost_bps"],
        )
        candidate_cost["delta_sharpe"] = (
            candidate_cost["sharpe"] - candidate_cost["baseline_sharpe"]
        )
        cost_delta = candidate_cost.pivot(
            index="signal", columns="cost_bps", values="delta_sharpe"
        ).reset_index()

    yearly_leader = pd.DataFrame()
    if not robust.empty:
        leader = robust.iloc[0]["signal"]
        baseline = robust.iloc[0]["baseline"]
        yearly_leader = yearly[yearly["signal"].isin([leader, baseline])].pivot(
            index="signal", columns="year", values="sharpe"
        ).reset_index()

    oos_leaders = (
        metrics[metrics["split"] == "oos"]
        .sort_values("sharpe", ascending=False)
        .head(12)
    )
    corr_leaders = correlations.reindex(
        correlations["xs_ic_tstat"].abs().sort_values(ascending=False).index
    ).head(12)

    def markdown_table(frame: pd.DataFrame, max_rows: int = 15) -> str:
        """Small dependency-free Markdown table for the research summary."""
        frame = frame.head(max_rows).copy()
        if frame.empty:
            return "No rows."
        formatted = frame.map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.4g}"
            if isinstance(value, (float, np.floating))
            else str(value)
        )
        columns = [str(column) for column in formatted.columns]
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = [
            "| " + " | ".join(str(row[column]) for column in formatted.columns) + " |"
            for _, row in formatted.iterrows()
        ]
        return "\n".join([header, divider, *body])

    lines = [
        "# OI/volume conditioning of price trend signals",
        "",
        f"Evidence cutoff: **{config.end.date()}**. Research period starts **{config.start.date()}**.",
        f"Trading-cost assumption: **{config.cost_bps:.1f} bps per unit turnover**.",
        "",
        "## Data quality",
        "",
        f"- {len(valid_quality)} products have aligned c1 price and aggregate WTPY OI/volume observations.",
        f"- Median aligned rows per included product: {valid_quality['aligned_rows'].median():.0f}.",
        f"- Products with aggregate loader errors: {(quality['aggregate_error'] != '').sum()}.",
        f"- Execution uses T+1 n305, then n310, then a1505 with notebook fallbacks; night-session prices supplied {night_rows / execution_rows:.1%} of populated product-days."
        if execution_rows
        else "- Execution-price coverage is unavailable.",
        "",
        "## Candidates improving their own production baseline in validation and OOS",
        "",
        markdown_table(robust) if not robust.empty else "No candidate passed both split checks.",
        "",
        "## 2024-2026 out-of-sample leaders",
        "",
        markdown_table(oos_leaders[["signal", "family", "sharpe", "mean_turnover", "positive_asset_share"]]),
        "",
        "## Paired block-bootstrap check of incremental OOS return",
        "",
        markdown_table(bootstrap) if not bootstrap.empty else "No split-robust modifiers to bootstrap.",
        "",
        "## Fixed-universe robustness",
        "",
        markdown_table(fixed_robust, max_rows=20)
        if not fixed_robust.empty
        else "No fixed-universe results.",
        "",
        "## OOS delta Sharpe under alternative costs",
        "",
        markdown_table(cost_delta) if not cost_delta.empty else "No cost-sensitivity results.",
        "",
        "## Calendar-year Sharpe of the leading modifier and its baseline",
        "",
        markdown_table(yearly_leader) if not yearly_leader.empty else "No yearly results.",
        "",
        "## Strongest feature/forward-return relationships",
        "",
        markdown_table(corr_leaders),
        "",
        "## Interpretation guardrails",
        "",
        "- All features use rolling or expanding history available at the signal date; positions are lagged one trading day.",
        "- Candidate selection is exploratory. Validation/OOS agreement and cross-product breadth are required before production review.",
        "- The bootstrap is paired by date in 20-day blocks and measures incremental annual return, not a selection-adjusted Sharpe test.",
        "- Fixed universes are selected only from history availability: pre-2012 with 3,000 aligned rows and pre-2014 with 2,000 aligned rows.",
        "- Aggregate OI is not additive across exchanges or economically comparable in level across products; transformations are within product.",
        "- Recent WTPY partitions and contract-roll behavior should be rechecked before live use.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_research(config: ResearchConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    front = load_front_history(config.price_file, config.products)
    actual_products = tuple(product for product in config.products if product in front["close"])
    if actual_products != config.products:
        config = ResearchConfig(**{**asdict(config), "products": actual_products})

    aggregate_panel, failures = load_or_build_aggregate_panel(config)
    features = build_feature_panels(front, aggregate_panel)
    common_products = list(features["returns"].columns)
    front = {name: frame[common_products] for name, frame in front.items()}
    baselines = build_price_baselines(front["close"])
    candidates, definitions = build_candidate_signals(baselines, features)

    correlations = feature_forward_correlations(
        features,
        (
            "vol_regime",
            "vol_zscore",
            "vol_hlratio",
            "vol_regt",
            "oi_momentum",
            "oi_breakout",
            "oi_level_qtl",
            "oi_level_zscore",
            "oi_regt",
            "volume_activity",
            "volume_breakout",
            "volume_level_qtl",
            "volume_regt",
            "price_oi_corr",
            "price_volume_corr",
            "front_concentration",
            "c1_roll_concentration",
            "turnover_qtl",
            "price_oi_divergence",
            "price_volume_divergence",
        ),
    )
    quality = data_quality_summary(
        front, features, failures, config.start, config.end
    )
    execution_quality = execution_quality_summary(front, config.start, config.end)
    metrics, pnl, turnover = evaluate_candidates(
        candidates,
        definitions,
        features,
        config.start,
        config.end,
        config.cost_bps,
        config.validation_start,
        config.oos_start,
    )
    increments = incremental_metrics(metrics)
    cost_sensitivity = cost_sensitivity_metrics(
        pnl,
        turnover,
        config.cost_bps,
        config.validation_start,
        config.oos_start,
    )
    yearly = yearly_oos_metrics(pnl, turnover, config.oos_start)
    bootstrap = paired_block_bootstrap(increments, pnl, config.oos_start)

    universe_metrics_parts: list[pd.DataFrame] = []
    universe_increment_parts: list[pd.DataFrame] = []
    cohorts = {
        "pre2012_min3000": mature_universe_products(quality, "2012-01-01", 3_000),
        "pre2014_min2000": mature_universe_products(quality, "2014-01-01", 2_000),
    }
    for cohort, cohort_products in cohorts.items():
        if len(cohort_products) < 2:
            continue
        cohort_candidates = {
            name: signal[cohort_products] for name, signal in candidates.items()
        }
        cohort_features = {
            "returns": features["returns"][cohort_products],
            "vol20": features["vol20"][cohort_products],
            "close": features["close"][cohort_products],
            "execution_price": features["execution_price"][cohort_products],
        }
        cohort_metrics, _, _ = evaluate_candidates(
            cohort_candidates,
            definitions,
            cohort_features,
            config.start,
            config.end,
            config.cost_bps,
            config.validation_start,
            config.oos_start,
        )
        cohort_metrics.insert(0, "cohort", cohort)
        cohort_metrics.insert(1, "n_products", len(cohort_products))
        cohort_increments = incremental_metrics(cohort_metrics)
        cohort_increments.insert(0, "cohort", cohort)
        cohort_increments.insert(1, "n_products", len(cohort_products))
        universe_metrics_parts.append(cohort_metrics)
        universe_increment_parts.append(cohort_increments)
    universe_metrics = pd.concat(universe_metrics_parts, ignore_index=True)
    universe_increments = pd.concat(universe_increment_parts, ignore_index=True)

    paths = {
        "quality": config.output_dir / "data_quality.csv",
        "execution_quality": config.output_dir / "execution_quality.csv",
        "definitions": config.output_dir / "candidate_definitions.csv",
        "correlations": config.output_dir / "feature_forward_correlations.csv",
        "metrics": config.output_dir / "strategy_metrics.csv",
        "incremental": config.output_dir / "incremental_metrics.csv",
        "pnl": config.output_dir / "daily_net_pnl.parquet",
        "turnover": config.output_dir / "daily_turnover.parquet",
        "cost_sensitivity": config.output_dir / "cost_sensitivity.csv",
        "yearly_oos": config.output_dir / "yearly_oos_metrics.csv",
        "bootstrap": config.output_dir / "paired_block_bootstrap.csv",
        "universe_metrics": config.output_dir / "fixed_universe_metrics.csv",
        "universe_incremental": config.output_dir / "fixed_universe_incremental.csv",
        "config": config.output_dir / "run_config.json",
        "summary": config.output_dir / "summary.md",
    }
    quality.to_csv(paths["quality"], index=False)
    execution_quality.to_csv(paths["execution_quality"], index=False)
    definitions.to_csv(paths["definitions"], index=False)
    correlations.to_csv(paths["correlations"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    increments.to_csv(paths["incremental"], index=False)
    pnl.to_parquet(paths["pnl"])
    turnover.to_parquet(paths["turnover"])
    cost_sensitivity.to_csv(paths["cost_sensitivity"], index=False)
    yearly.to_csv(paths["yearly_oos"], index=False)
    bootstrap.to_csv(paths["bootstrap"], index=False)
    universe_metrics.to_csv(paths["universe_metrics"], index=False)
    universe_increments.to_csv(paths["universe_incremental"], index=False)
    paths["config"].write_text(
        json.dumps(
            {
                **asdict(config),
                "price_file": str(config.price_file),
                "output_dir": str(config.output_dir),
                "aggregate_cache": str(config.aggregate_cache),
                "start": str(config.start.date()),
                "end": str(config.end.date()),
                "validation_start": str(config.validation_start.date()),
                "oos_start": str(config.oos_start.date()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_charts(metrics, pnl, config.output_dir)
    write_summary(
        config,
        quality,
        execution_quality,
        correlations,
        metrics,
        increments,
        bootstrap,
        universe_increments,
        cost_sensitivity,
        yearly,
        config.output_dir,
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> ResearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", type=Path, default=Path("C:/dev/data/fut_d_20260918.parquet"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-09-18")
    parser.add_argument("--validation-start", default="2019-01-01")
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("C:/dev/data/output") / "oi_volume_trend_research")
    parser.add_argument("--aggregate-cache", type=Path)
    parser.add_argument("--products", nargs="*", default=None)
    parser.add_argument("--contract-period", default="12m")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--refresh-aggregate", action="store_true")
    args = parser.parse_args(argv)

    products = tuple(args.products or BROAD_MKTS)
    output_dir = args.output_dir.resolve()
    cache = args.aggregate_cache or output_dir / f"aggregate_oi_volume_{args.end.replace('-', '')}.parquet"
    return ResearchConfig(
        price_file=args.price_file.resolve(),
        start=_as_timestamp(args.start),
        end=_as_timestamp(args.end),
        output_dir=output_dir,
        products=products,
        aggregate_cache=cache.resolve(),
        validation_start=_as_timestamp(args.validation_start),
        oos_start=_as_timestamp(args.oos_start),
        contract_period=args.contract_period,
        cost_bps=args.cost_bps,
        refresh_aggregate=args.refresh_aggregate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    paths = run_research(config)
    print("Research outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
