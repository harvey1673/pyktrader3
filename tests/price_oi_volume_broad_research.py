"""Broad price/volatility/aggregate-OI/volume signal and portfolio research.

The research universe covers time-series and cross-sectional demeaned variants.
Signal directions are selected on the 2010-2018 training sample; validation
screening and portfolio weights use data through 2023 only.  The 2024+ sample
is reported without using it for signs, screening, or weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WTPY_ROOT = Path("C:/dev/wtpy")
for _path in (REPO_ROOT, WTPY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pycmqlib3.analytics import tstool  # noqa: E402
from pycmqlib3.strategy.signal_repo import BROAD_MKTS  # noqa: E402
from pycmqlib3.utility.misc import product_lotsize  # noqa: E402
from tests.oi_volume_sharpe_search import (  # noqa: E402
    _series_metrics,
    optimize_nonnegative_sharpe,
)
from tests.oi_volume_trend_research import (  # noqa: E402
    PNL_BDAYS,
    ResearchConfig,
    build_feature_panels,
    data_quality_summary,
    execution_quality_summary,
    load_front_history,
    load_or_build_aggregate_panel,
    split_dates,
)


FAMILIES = ("time_series", "xs_demean")
VOL_FACTOR_K = 5.0
RISK_SCALING = 0.20
MAX_GROSS_EXPOSURE = 2.0


@dataclass(frozen=True)
class SignalRecipe:
    signal: pd.DataFrame
    category: str
    thesis: str
    parent: str
    position_scale: pd.DataFrame | None = None


def _apply_columns(frame: pd.DataFrame, function) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: function(frame[column].dropna()).reindex(frame.index)
            for column in frame
        },
        index=frame.index,
    )


def volatility_position_factor(
    vol_ratio: pd.DataFrame, k: float = 1.0
) -> pd.DataFrame:
    """Return min(exp(-k * (ratio - 1)), 1) for a non-negative k."""
    if k < 0:
        raise ValueError("k must be non-negative")
    excess = (vol_ratio - 1.0).clip(lower=0.0)
    return np.exp(-k * excess).where(vol_ratio.notna())


def efficiency_ratio(log_price: pd.DataFrame, window: int) -> pd.DataFrame:
    """Kaufman efficiency ratio: net movement divided by path length."""
    net_move = log_price.diff(window).abs()
    path_length = log_price.diff().abs().rolling(window, min_periods=window).sum()
    return (net_move / path_length.replace(0.0, np.nan)).clip(0.0, 1.0)


def build_broad_features(
    base: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    features = dict(base)
    returns = features["returns"]
    log_price = np.log(features["close"].where(features["close"] > 0))
    for window in (60, 120, 240, 480):
        long_vol = returns.rolling(window, min_periods=max(40, window // 2)).std()
        ratio = (
            features["vol20"] / (long_vol * math.sqrt(PNL_BDAYS)).replace(0.0, np.nan)
        )
        features[f"vol_ratio_{window}"] = ratio
        features[f"vol_factor_{window}"] = volatility_position_factor(
            ratio, k=VOL_FACTOR_K
        )
    for window in (20, 60, 120, 240):
        er = efficiency_ratio(log_price, window)
        features[f"efficiency_ratio_{window}"] = er
        features[f"efficiency_qtl_{window}"] = _apply_columns(
            er, lambda series: (tstool.pct_score(series, 252) + 1.0) / 2.0
        ).clip(0.0, 1.0)
    features["oi_high"] = ((features["oi_level_qtl"] + 1.0) / 2.0).clip(0.0, 1.0)
    features["volume_high"] = (
        (features["volume_level_qtl"] + 1.0) / 2.0
    ).clip(0.0, 1.0)
    features["vol_stress_240"] = (
        (features["vol_ratio_240"] - 1.0).clip(lower=0.0, upper=2.0) / 2.0
    )
    return features


def build_broad_library(
    close: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, SignalRecipe], pd.DataFrame]:
    """Build a bounded, predeclared library spanning trend and reversal ideas."""
    recipes: dict[str, SignalRecipe] = {}
    definitions: list[dict[str, str]] = []

    def add(
        name: str,
        signal: pd.DataFrame,
        category: str,
        thesis: str,
        parent: str | None = None,
        position_scale: pd.DataFrame | None = None,
    ) -> None:
        clean_signal = signal.reindex_like(close).replace([np.inf, -np.inf], np.nan)
        clean_signal = clean_signal.clip(-2.0, 2.0)
        clean_scale = None
        if position_scale is not None:
            clean_scale = (
                position_scale.reindex_like(close)
                .replace([np.inf, -np.inf], np.nan)
                .clip(0.0, 1.0)
            )
        recipes[name] = SignalRecipe(
            clean_signal, category, thesis, parent or name, clean_scale
        )
        definitions.append(
            {
                "recipe": name,
                "parent": parent or name,
                "category": category,
                "has_position_scale": str(position_scale is not None),
                "thesis": thesis,
            }
        )

    log_price = np.log(close.where(close > 0))
    daily_vol = features["returns"].rolling(60, min_periods=40).std()
    anchors: dict[str, tuple[pd.DataFrame, int]] = {}
    for horizon in (20, 60, 120, 240):
        momentum = (
            log_price.diff(horizon) / (daily_vol * math.sqrt(horizon))
        ).clip(-2.0, 2.0) / 2.0
        name = f"price_mom_{horizon}d"
        anchors[name] = (momentum, horizon)
        add(name, momentum, "price_trend", f"{horizon}-day volatility-scaled momentum.")

        hlr = _apply_columns(log_price, lambda series, h=horizon: tstool.hlratio(series, h))
        hlr = hlr.clip(-1.0, 1.0)
        name = f"price_hlr_{horizon}d"
        anchors[name] = (hlr, horizon)
        add(name, hlr, "price_trend", f"{horizon}-day price high/low location.")

        regt = _apply_columns(
            log_price,
            lambda series, h=horizon: tstool.rolling_trend(
                series, h, return_mode="t-stat", log=False
            ),
        ).clip(-5.0, 5.0) / 5.0
        name = f"price_regt_{horizon}d"
        anchors[name] = (regt, horizon)
        add(name, regt, "price_trend", f"{horizon}-day price regression t-statistic.")

    # True post-risk position scalers. These can reduce total gross exposure.
    for name, (signal, horizon) in anchors.items():
        add(
            f"{name}__posscale__vol_factor_240",
            signal,
            "vol_position_scale",
            "Reduce positions only when 20-day volatility exceeds 240-day volatility.",
            parent=name,
            position_scale=features["vol_factor_240"],
        )
        add(
            f"{name}__posscale__efficiency_{horizon}",
            signal,
            "quality_position_scale",
            f"Scale positions by the {horizon}-day price efficiency ratio.",
            parent=name,
            position_scale=features[f"efficiency_ratio_{horizon}"],
        )
        add(
            f"{name}__posscale__vol240_efficiency_{horizon}",
            signal,
            "joint_position_scale",
            "Jointly require controlled short volatility and an efficient price path.",
            parent=name,
            position_scale=(
                features["vol_factor_240"] * features[f"efficiency_ratio_{horizon}"]
            ),
        )

    # Check alternative long-volatility anchors for the canonical 240-day momentum.
    mom240 = anchors["price_mom_240d"][0]
    for window in (120, 480):
        add(
            f"price_mom_240d__posscale__vol_factor_{window}",
            mom240,
            "vol_position_scale",
            f"240-day momentum scaled by the 20-day/{window}-day volatility ratio.",
            parent="price_mom_240d",
            position_scale=features[f"vol_factor_{window}"],
        )

    # Participation and relationship modifiers on price momentum.
    for horizon in (20, 60, 120, 240):
        name = f"price_mom_{horizon}d"
        signal = anchors[name][0]
        direction = np.sign(signal)
        modifiers = {
            "oi_confirm": (
                signal * (1.0 + 0.5 * direction * features["oi_momentum"]),
                "Scale momentum up when aggregate OI change agrees with price direction.",
            ),
            "volume_confirm": (
                signal * (1.0 + 0.5 * features["volume_activity"]),
                "Scale momentum with unusual aggregate volume growth.",
            ),
            "joint_confirm": (
                signal
                * (
                    1.0
                    + 0.30 * direction * features["oi_momentum"]
                    + 0.20 * features["volume_activity"]
                ),
                "Combine directional OI build and volume participation.",
            ),
            "oi_breakout_confirm": (
                signal * (1.0 + 0.5 * direction * features["oi_breakout"]),
                "Confirm momentum using aggregate OI breakouts.",
            ),
            "volume_breakout_confirm": (
                signal * (1.0 + 0.5 * features["volume_breakout"]),
                "Prefer momentum with aggregate volume breakouts.",
            ),
        }
        for suffix, (modified, thesis) in modifiers.items():
            add(
                f"{name}__{suffix}",
                modified,
                "trend_participation",
                thesis,
                parent=name,
            )

    # Reversal hypotheses: fade price extremes under crowding/exhaustion regimes.
    for horizon in (20, 60, 120, 240):
        price_extreme = anchors[f"price_hlr_{horizon}d"][0]
        low_efficiency = 1.0 - features[f"efficiency_ratio_{horizon}"]
        reversal_specs = {
            "oi_high": features["oi_high"],
            "oi_high_low_efficiency": features["oi_high"] * low_efficiency,
            "oi_volume_high": features["oi_high"] * features["volume_high"],
            "oi_high_vol_stress": features["oi_high"] * features["vol_stress_240"],
            "volume_high_vol_stress": (
                features["volume_high"] * features["vol_stress_240"]
            ),
        }
        for suffix, regime in reversal_specs.items():
            add(
                f"reversal_hlr_{horizon}d__{suffix}",
                -price_extreme * regime,
                "reversal",
                f"Fade {horizon}-day price extremes when {suffix.replace('_', ' ')}.",
                parent=f"price_hlr_{horizon}d",
            )

    standalone = (
        "vol_regime",
        "vol_hlratio",
        "vol_regt",
        "oi_momentum",
        "oi_breakout",
        "oi_level_zscore",
        "oi_regt",
        "volume_activity",
        "volume_breakout",
        "volume_regt",
        "price_oi_corr",
        "price_volume_corr",
    )
    for feature_name in standalone:
        add(
            feature_name,
            features[feature_name],
            "standalone_state",
            f"Standalone point-in-time {feature_name} state.",
        )

    return recipes, pd.DataFrame(definitions)


def _family_transform(signal: pd.DataFrame, family: str) -> pd.DataFrame:
    if family == "time_series":
        return signal
    if family == "xs_demean":
        return tstool.xs_demean(signal)
    raise ValueError(f"Unknown signal family: {family}")


def evaluate_broad_signal(
    recipe: SignalRecipe,
    features: Mapping[str, pd.DataFrame],
    family: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build target weights and T+1 execution-aware gross PnL for one sleeve."""
    signal = recipe.signal.loc[start:end]
    returns = features["returns"].loc[start:end]
    signal, returns = signal.align(returns, join="inner", axis=0)
    signal, returns = signal.align(returns, join="inner", axis=1)
    transformed = _family_transform(signal, family)
    if recipe.position_scale is not None:
        scale = (
            recipe.position_scale.reindex_like(transformed)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        transformed = transformed * scale
        if family == "xs_demean":
            transformed = tstool.xs_demean(transformed).fillna(0.0)

    risk = features["vol20"].reindex_like(transformed).replace(0.0, np.nan)
    raw_weight = transformed / risk
    active_count = raw_weight.notna().sum(axis=1).replace(0, np.nan)
    weights = raw_weight.mul(RISK_SCALING).div(active_count, axis=0).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    weights = weights.div(
        (gross / MAX_GROSS_EXPOSURE).clip(lower=1.0), axis=0
    ).fillna(0.0)

    holdings = weights.shift(1)
    trade = holdings - holdings.shift(1).fillna(0.0)
    close = features["close"].reindex_like(holdings)
    execution = features["execution_price"].reindex_like(holdings)
    execution_adjustment = trade * (close / execution - 1.0)
    gross_asset_pnl = holdings * close.pct_change(fill_method=None)
    gross_asset_pnl = gross_asset_pnl + execution_adjustment
    multipliers = pd.Series(
        {column: float(product_lotsize.get(column, 1.0)) for column in holdings},
        dtype=float,
    )
    gross_asset_pnl_cny = (
        holdings * close.diff() + trade * (close - execution)
    ).mul(multipliers, axis=1)
    return {
        "weights": weights,
        "holdings": holdings,
        "gross_asset_pnl": gross_asset_pnl,
        "gross_pnl": gross_asset_pnl.sum(axis=1, min_count=1),
        "gross_pnl_cny": gross_asset_pnl_cny.sum(axis=1, min_count=1),
        "turnover": trade.abs().sum(axis=1, min_count=1),
        "gross_exposure": weights.abs().sum(axis=1),
    }


def evaluate_library(
    recipes: Mapping[str, SignalRecipe],
    definitions: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = split_dates(
        features["returns"].loc[config.start:config.end].dropna(how="all").index,
        config.validation_start,
        config.oos_start,
    )
    selection_end = splits["validation"][1]
    metric_periods = {
        "full": (config.start, config.end),
        "selection": (config.start, selection_end),
        **splits,
    }
    metadata = definitions.set_index("recipe").to_dict("index")
    rows: list[dict[str, object]] = []
    orientations: list[dict[str, object]] = []
    pnl_store: dict[str, pd.Series] = {}
    turnover_store: dict[str, pd.Series] = {}
    local_pnl_store: dict[str, pd.Series] = {}

    for recipe_name, recipe in recipes.items():
        for family in FAMILIES:
            result = evaluate_broad_signal(
                recipe, features, family, config.start, config.end
            )
            gross = result["gross_pnl"]
            turnover = result["turnover"]
            cost = turnover * config.cost_bps / 10_000.0
            train_start, train_end = splits["train"]
            positive_train = _series_metrics(
                (gross - cost).loc[train_start:train_end],
                turnover.loc[train_start:train_end],
            )["sharpe"]
            negative_train = _series_metrics(
                (-gross - cost).loc[train_start:train_end],
                turnover.loc[train_start:train_end],
            )["sharpe"]
            direction = (
                -1
                if pd.notna(negative_train)
                and (pd.isna(positive_train) or negative_train > positive_train)
                else 1
            )
            pnl = direction * gross - cost
            strategy = f"{recipe_name}__{family}"
            pnl_store[strategy] = pnl
            turnover_store[strategy] = turnover
            local_pnl_store[strategy] = direction * result["gross_pnl_cny"]
            orientations.append(
                {
                    "strategy": strategy,
                    "recipe": recipe_name,
                    "family": family,
                    "train_direction": direction,
                    "positive_train_sharpe": positive_train,
                    "negative_train_sharpe": negative_train,
                    "mean_target_gross": float(result["gross_exposure"].mean()),
                }
            )
            for split, (period_start, period_end) in metric_periods.items():
                values = _series_metrics(
                    pnl.loc[period_start:period_end],
                    turnover.loc[period_start:period_end],
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "recipe": recipe_name,
                        "family": family,
                        "category": metadata[recipe_name]["category"],
                        "parent": metadata[recipe_name]["parent"],
                        "train_direction": direction,
                        "split": split,
                        "start": period_start,
                        "end": period_end,
                        **values,
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(pnl_store),
        pd.DataFrame(turnover_store),
        pd.DataFrame(orientations),
        pd.DataFrame(local_pnl_store),
    )


def volatility_scaler_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    scaled = metrics[
        metrics["recipe"].str.endswith("__posscale__vol_factor_240")
    ].copy()
    scaled["baseline_recipe"] = scaled["recipe"].str.replace(
        "__posscale__vol_factor_240", "", regex=False
    )
    baseline = metrics.rename(
        columns={
            "recipe": "baseline_recipe",
            "strategy": "baseline_strategy",
            "sharpe": "baseline_sharpe",
            "annual_return": "baseline_annual_return",
            "annual_vol": "baseline_annual_vol",
            "mean_turnover": "baseline_mean_turnover",
            "train_direction": "baseline_train_direction",
        }
    )
    columns = [
        "baseline_recipe",
        "family",
        "split",
        "baseline_strategy",
        "baseline_sharpe",
        "baseline_annual_return",
        "baseline_annual_vol",
        "baseline_mean_turnover",
        "baseline_train_direction",
    ]
    comparison = scaled.merge(baseline[columns], on=["baseline_recipe", "family", "split"])
    comparison["delta_sharpe"] = comparison["sharpe"] - comparison["baseline_sharpe"]
    comparison["turnover_ratio"] = (
        comparison["mean_turnover"] / comparison["baseline_mean_turnover"]
    )
    comparison["same_train_direction"] = (
        comparison["train_direction"] == comparison["baseline_train_direction"]
    )
    return comparison[
        [
            "strategy",
            "baseline_strategy",
            "baseline_recipe",
            "family",
            "split",
            "sharpe",
            "baseline_sharpe",
            "delta_sharpe",
            "annual_return",
            "baseline_annual_return",
            "annual_vol",
            "baseline_annual_vol",
            "mean_turnover",
            "baseline_mean_turnover",
            "turnover_ratio",
            "train_direction",
            "baseline_train_direction",
            "same_train_direction",
        ]
    ]


def select_portfolio(
    metrics: pd.DataFrame,
    pnl: pd.DataFrame,
    turnover: pd.DataFrame,
    config: ResearchConfig,
    maximum_components: int = 30,
    correlation_limit: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sharpe = metrics.pivot_table(index="strategy", columns="split", values="sharpe")
    eligible = sharpe[
        (sharpe["train"] > 0.25)
        & (sharpe["validation"] > 0.0)
        & sharpe[["train", "validation"]].notna().all(axis=1)
    ].copy()
    eligible["robust_score"] = eligible[["train", "validation"]].min(axis=1)
    eligible = eligible.sort_values(["robust_score", "selection"], ascending=False)

    selection_end = config.oos_start - pd.Timedelta(days=1)
    selection_pnl = pnl.loc[:selection_end]
    selected: list[str] = []
    for strategy in eligible.index:
        if selected:
            correlations = selection_pnl[selected].corrwith(
                selection_pnl[strategy]
            ).abs()
            if correlations.max() >= correlation_limit:
                continue
        selected.append(strategy)
        if len(selected) >= maximum_components:
            break
    if not selected:
        raise RuntimeError("No strategies passed the pre-OOS portfolio screen.")

    equal_weights = np.repeat(1.0 / len(selected), len(selected))
    cap = max(0.10, 1.0 / len(selected))
    optimized_weights, _, success = optimize_nonnegative_sharpe(
        selection_pnl[selected], maximum_weight=cap
    )
    if not success or len(optimized_weights) != len(selected):
        optimized_weights = equal_weights.copy()
    recommended_weights = 0.5 * equal_weights + 0.5 * optimized_weights

    components = pd.DataFrame(
        {
            "strategy": selected,
            "equal_weight": equal_weights,
            "optimized_weight": optimized_weights,
            "recommended_weight": recommended_weights,
            "train_sharpe": sharpe.loc[selected, "train"].to_numpy(),
            "validation_sharpe": sharpe.loc[selected, "validation"].to_numpy(),
            "selection_sharpe": sharpe.loc[selected, "selection"].to_numpy(),
            "oos_sharpe": sharpe.loc[selected, "oos"].to_numpy(),
        }
    )
    component_metadata = metrics[metrics["split"] == "train"][
        [
            "strategy",
            "recipe",
            "family",
            "category",
            "parent",
            "train_direction",
        ]
    ].drop_duplicates("strategy")
    components = components.merge(component_metadata, on="strategy", how="left")

    portfolio_pnl = pd.DataFrame(index=pnl.index)
    portfolio_turnover = pd.DataFrame(index=pnl.index)
    for label, weights in (
        ("equal_weight", equal_weights),
        ("optimized", optimized_weights),
        ("recommended_shrunk_50", recommended_weights),
    ):
        portfolio_pnl[label] = pnl[selected].mul(weights, axis=1).sum(axis=1)
        portfolio_turnover[label] = turnover[selected].mul(weights, axis=1).sum(axis=1)

    periods = {
        "full": (config.start, config.end),
        "selection": (config.start, selection_end),
        **split_dates(pnl.index, config.validation_start, config.oos_start),
    }
    rows = []
    for portfolio in portfolio_pnl:
        for split, (period_start, period_end) in periods.items():
            rows.append(
                {
                    "portfolio": portfolio,
                    "split": split,
                    "start": period_start,
                    "end": period_end,
                    **_series_metrics(
                        portfolio_pnl[portfolio].loc[period_start:period_end],
                        portfolio_turnover[portfolio].loc[period_start:period_end],
                    ),
                }
            )
    portfolio_metrics = pd.DataFrame(rows)
    portfolio_daily = pd.concat(
        {
            "net_pnl": portfolio_pnl,
            "turnover": portfolio_turnover,
        },
        axis=1,
    )
    return components, portfolio_metrics, portfolio_daily


def _markdown_table(frame: pd.DataFrame, rows: int = 15) -> str:
    frame = frame.head(rows).copy()
    if frame.empty:
        return "No rows."
    formatted = frame.map(
        lambda value: ""
        if pd.isna(value)
        else f"{value:.4g}"
        if isinstance(value, (float, np.floating))
        else str(value)
    )
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, divider, *body])


def write_outputs(
    config: ResearchConfig,
    definitions: pd.DataFrame,
    quality: pd.DataFrame,
    execution_quality: pd.DataFrame,
    metrics: pd.DataFrame,
    pnl: pd.DataFrame,
    turnover: pd.DataFrame,
    orientations: pd.DataFrame,
    vol_comparison: pd.DataFrame,
    components: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    local_pnl: pd.DataFrame,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": config.output_dir / "summary.md",
        "definitions": config.output_dir / "recipe_definitions.csv",
        "quality": config.output_dir / "data_quality.csv",
        "execution_quality": config.output_dir / "execution_quality.csv",
        "metrics": config.output_dir / "strategy_metrics.csv",
        "orientations": config.output_dir / "train_orientations.csv",
        "pnl": config.output_dir / "daily_net_pnl.parquet",
        "local_pnl": config.output_dir / "daily_gross_pnl_cny.parquet",
        "turnover": config.output_dir / "daily_turnover.parquet",
        "vol_comparison": config.output_dir / "volatility_scaler_comparison.csv",
        "components": config.output_dir / "portfolio_components.csv",
        "components_risk": config.output_dir / "portfolio_components_with_unit_std.csv",
        "portfolio_metrics": config.output_dir / "portfolio_metrics.csv",
        "portfolio_daily": config.output_dir / "portfolio_daily.parquet",
        "chart": config.output_dir / "portfolio_cumulative_pnl.png",
        "config": config.output_dir / "run_config.json",
    }
    definitions.to_csv(paths["definitions"], index=False)
    quality.to_csv(paths["quality"], index=False)
    execution_quality.to_csv(paths["execution_quality"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    orientations.to_csv(paths["orientations"], index=False)
    pnl.to_parquet(paths["pnl"])
    local_pnl.to_parquet(paths["local_pnl"])
    turnover.to_parquet(paths["turnover"])
    vol_comparison.to_csv(paths["vol_comparison"], index=False)
    selection_risk = metrics[metrics["split"].eq("selection")][
        ["strategy", "annual_vol", "mean_turnover"]
    ].rename(
        columns={
            "annual_vol": "unit_std_annualized",
            "mean_turnover": "selection_mean_turnover",
        }
    )
    components_risk = components.merge(selection_risk, on="strategy", how="left")
    components_risk["unit_std_daily"] = (
        components_risk["unit_std_annualized"] / math.sqrt(PNL_BDAYS)
    )
    selection_end = config.oos_start - pd.Timedelta(days=1)
    local_daily_std = local_pnl.loc[:selection_end].std().rename(
        "unit_gross_pnl_daily_std_cny"
    )
    components_risk = components_risk.merge(
        local_daily_std.rename_axis("strategy").reset_index(),
        on="strategy",
        how="left",
    )
    components_risk["weight_per_unit_std_annualized"] = (
        components_risk["optimized_weight"]
        / components_risk["unit_std_annualized"].replace(0.0, np.nan)
    )
    components_risk["recommended_weight_per_unit_std_annualized"] = (
        components_risk["recommended_weight"]
        / components_risk["unit_std_annualized"].replace(0.0, np.nan)
    )
    components.to_csv(paths["components"], index=False)
    components_risk.to_csv(paths["components_risk"], index=False)
    portfolio_metrics.to_csv(paths["portfolio_metrics"], index=False)
    portfolio_daily.to_parquet(paths["portfolio_daily"])

    curves = portfolio_daily["net_pnl"].fillna(0.0).cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    curves.plot(ax=ax, linewidth=1.3)
    ax.axvline(config.validation_start, color="#777777", linestyle=":")
    ax.axvline(config.oos_start, color="#111111", linestyle=":")
    ax.set_title("Pre-OOS-selected price/OI/volume portfolios")
    ax.set_ylabel("Cumulative net PnL")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(paths["chart"], dpi=150)
    plt.close(fig)

    vol_pivot = vol_comparison.pivot_table(
        index=["baseline_recipe", "family"],
        columns="split",
        values="delta_sharpe",
    ).reset_index()
    robust_vol = vol_pivot[
        (vol_pivot.get("train", np.nan) > 0)
        & (vol_pivot.get("validation", np.nan) > 0)
    ].sort_values("validation", ascending=False)
    oos_leaders = metrics[metrics["split"] == "oos"].sort_values(
        "sharpe", ascending=False
    )
    reversal_oos = metrics[
        (metrics["split"] == "oos") & (metrics["category"] == "reversal")
    ]
    true_reversals = reversal_oos[reversal_oos["train_direction"] == 1].sort_values(
        "sharpe", ascending=False
    )
    inverted_reversals = reversal_oos[
        reversal_oos["train_direction"] == -1
    ].sort_values("sharpe", ascending=False)
    lines = [
        "# Broad price, volatility, aggregate OI, and volume research",
        "",
        f"Research period: **{config.start.date()} to {config.end.date()}**; "
        f"cost: **{config.cost_bps:.1f} bps per unit turnover**.",
        "",
        "Signals are oriented on the training sample only. Portfolio screening and "
        "weights stop before 2024; the 2024+ rows are not used in those choices.",
        "",
        "## Portfolio metrics",
        "",
        _markdown_table(portfolio_metrics),
        "",
        "## Volatility position scalers improving both train and validation Sharpe",
        "",
        _markdown_table(robust_vol, 20),
        "",
        "## Leading 2024+ standalone sleeves",
        "",
        _markdown_table(
            oos_leaders[
                [
                    "strategy",
                    "family",
                    "category",
                    "train_direction",
                    "sharpe",
                    "mean_turnover",
                ]
            ],
            20,
        ),
        "",
        "## True reversal sleeves (training-selected direction +1)",
        "",
        _markdown_table(
            true_reversals[
                ["strategy", "family", "train_direction", "sharpe", "mean_turnover"]
            ],
            15,
        ),
        "",
        "## Inverted reversal recipes (direction -1, economically continuation)",
        "",
        _markdown_table(
            inverted_reversals[
                ["strategy", "family", "train_direction", "sharpe", "mean_turnover"]
            ],
            15,
        ),
        "",
        "## Selected portfolio sleeves",
        "",
        _markdown_table(components_risk, 30),
        "",
        "## Guardrails",
        "",
        "- T signals are lagged once and receive the T+1 n305/n310/a1505 execution adjustment.",
        f"- Volatility position scalers use k={VOL_FACTOR_K:g} in min(exp(-k * (vr_240 - 1)), 1).",
        "- Cross-sectional modifiers are applied to the signal, the complete signal is demeaned, and holdings are then divided by vol20; this targets signed standalone-volatility risk neutrality rather than nominal neutrality.",
        "- All sleeves are capped only above 2x gross exposure.",
        "- The 2024+ period was not used mechanically for directions, screening, or weights, but the research hypotheses were developed after observing earlier results, so it is not a pristine external holdout.",
        "- Strategy-level costs are combined conservatively without cross-sleeve trade netting.",
        "- The recommended_shrunk_50 portfolio places 50% weight on the equal-weight solution and 50% on the constrained maximum-selection-Sharpe solution; this is the conservative recommendation rather than the raw optimizer weights.",
    ]
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")
    config_payload = {
        **asdict(config),
        "price_file": str(config.price_file),
        "output_dir": str(config.output_dir),
        "aggregate_cache": str(config.aggregate_cache),
        "aggregate_file": (
            None if config.aggregate_file is None else str(config.aggregate_file)
        ),
        "start": str(config.start.date()),
        "end": str(config.end.date()),
        "validation_start": str(config.validation_start.date()),
        "oos_start": str(config.oos_start.date()),
        "families": list(FAMILIES),
        "vol_factor_k": VOL_FACTOR_K,
        "xs_neutrality": "demean complete signal before dividing by vol20",
    }
    paths["config"].write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return paths


def run_research(config: ResearchConfig) -> dict[str, Path]:
    front = load_front_history(config.price_file, config.products)
    actual_products = tuple(product for product in config.products if product in front["close"])
    if actual_products != config.products:
        config = ResearchConfig(**{**asdict(config), "products": actual_products})
    aggregate, failures = load_or_build_aggregate_panel(config)
    base_features = build_feature_panels(front, aggregate)
    products = list(base_features["returns"].columns)
    front = {name: frame[products] for name, frame in front.items()}
    features = build_broad_features(base_features)
    recipes, definitions = build_broad_library(features["close"], features)
    quality = data_quality_summary(front, features, failures, config.start, config.end)
    execution_quality = execution_quality_summary(front, config.start, config.end)
    metrics, pnl, turnover, orientations, local_pnl = evaluate_library(
        recipes, definitions, features, config
    )
    vol_comparison = volatility_scaler_comparison(metrics)
    components, portfolio_metrics, portfolio_daily = select_portfolio(
        metrics, pnl, turnover, config
    )
    return write_outputs(
        config,
        definitions,
        quality,
        execution_quality,
        metrics,
        pnl,
        turnover,
        orientations,
        vol_comparison,
        components,
        portfolio_metrics,
        portfolio_daily,
        local_pnl,
    )


def parse_args(argv: Sequence[str] | None = None) -> ResearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-file",
        type=Path,
        default=Path("C:/dev/data/fut_d_20260930.parquet"),
    )
    parser.add_argument(
        "--aggregate-file",
        type=Path,
        default=Path("C:/dev/data/fut_oi_volume_20260930.parquet"),
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--validation-start", default="2019-01-01")
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/dev/data/output/price_oi_volume_broad_research"),
    )
    parser.add_argument("--products", nargs="*", default=None)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    return ResearchConfig(
        price_file=args.price_file.resolve(),
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=output_dir,
        products=tuple(args.products or BROAD_MKTS),
        aggregate_cache=output_dir / "unused_wtpy_cache.parquet",
        validation_start=pd.Timestamp(args.validation_start),
        oos_start=pd.Timestamp(args.oos_start),
        aggregate_file=args.aggregate_file.resolve(),
        cost_bps=args.cost_bps,
    )


if __name__ == "__main__":
    output_paths = run_research(parse_args())
    print("Broad research outputs:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
