"""Full-sample Sharpe search using price, aggregate OI, and volume signals.

This is intentionally an exploratory, test-side oracle search.  Recipe signs,
component selection, and static blend weights are all chosen on 2010-2026 net
PnL.  Split metrics are diagnostics, not honest out-of-sample estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
WTPY_ROOT = Path("C:/dev/wtpy")
for _path in (REPO_ROOT, WTPY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pycmqlib3.analytics import tstool
from pycmqlib3.strategy.signal_repo import BROAD_MKTS
from tests.oi_volume_trend_research import (
    PNL_BDAYS,
    ResearchConfig,
    build_feature_panels,
    data_quality_summary,
    evaluate_signal,
    execution_quality_summary,
    load_front_history,
    load_or_build_aggregate_panel,
    split_dates,
)

def _apply_columns(
    frame: pd.DataFrame,
    function,
) -> pd.DataFrame:
    return pd.DataFrame(
        {column: function(frame[column].dropna()).reindex(frame.index) for column in frame},
        index=frame.index,
    )


def _series_metrics(pnl: pd.Series, turnover: pd.Series) -> dict[str, float | int]:
    pnl = pnl.dropna()
    turnover = turnover.reindex(pnl.index).fillna(0.0)
    annual_return = float(pnl.mean() * PNL_BDAYS)
    annual_vol = float(pnl.std() * math.sqrt(PNL_BDAYS))
    cumulative = pnl.fillna(0.0).cumsum()
    drawdown = cumulative - cumulative.cummax()
    mean_turnover = float(turnover.mean())
    return {
        "sharpe": annual_return / annual_vol if annual_vol > 0 else np.nan,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
        "mean_turnover": mean_turnover,
        "pnl_per_turnover": float(pnl.mean() / mean_turnover) if mean_turnover else np.nan,
        "n_days": int(len(pnl)),
    }


def build_signal_library(
    close: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create standalone and nonlinear interaction recipes without production parents."""
    recipes: dict[str, pd.DataFrame] = {}
    definitions: list[dict[str, str]] = []

    def add(name: str, category: str, thesis: str, signal: pd.DataFrame) -> None:
        recipes[name] = signal.reindex_like(close).replace([np.inf, -np.inf], np.nan).clip(-2, 2)
        definitions.append({"recipe": name, "category": category, "thesis": thesis})

    log_price = np.log(close.where(close > 0))
    daily_vol = features["returns"].rolling(60, min_periods=40).std()
    price_anchors: dict[str, pd.DataFrame] = {}
    for horizon in (1, 3, 5, 10, 20, 40, 60, 120, 240):
        signal = (log_price.diff(horizon) / (daily_vol * math.sqrt(horizon))).clip(-2, 2) / 2
        name = f"price_mom_{horizon}d"
        add(name, "price", f"Volatility-scaled {horizon}-day log price change.", signal)
        if horizon in (20, 60, 120, 240):
            price_anchors[name] = signal

    for horizon in (10, 20, 40, 60, 120, 240):
        hlr = _apply_columns(log_price, lambda s, h=horizon: tstool.hlratio(s, h)).clip(-1, 1)
        regt = _apply_columns(
            log_price,
            lambda s, h=horizon: tstool.rolling_trend(
                s, h, return_mode="t-stat", log=False
            ),
        ).clip(-5, 5) / 5
        add(f"price_hlr_{horizon}d", "price", f"{horizon}-day price high/low ratio.", hlr)
        add(f"price_regt_{horizon}d", "price", f"{horizon}-day price regression t-stat.", regt)
        if horizon in (60, 120):
            price_anchors[f"price_hlr_{horizon}d"] = hlr
            price_anchors[f"price_regt_{horizon}d"] = regt

    log_oi = np.log(features["aggregate_oi"].where(features["aggregate_oi"] > 0))
    log_volume = np.log(
        features["aggregate_volume"].where(features["aggregate_volume"] > 0)
    )
    for label, values in (("oi", log_oi), ("volume", log_volume)):
        for horizon in (1, 5, 10, 20, 60, 120):
            change = values.diff(horizon)
            zscore = _apply_columns(change, lambda s: tstool.zscore_roll(s, 252)).clip(-2, 2) / 2
            add(
                f"{label}_change_z_{horizon}d",
                label,
                f"Rolling z-score of {horizon}-day aggregate {label} change.",
                zscore,
            )
        for horizon in (20, 60, 120, 252):
            hlr = _apply_columns(values, lambda s, h=horizon: tstool.hlratio(s, h)).clip(-1, 1)
            regt = _apply_columns(
                values,
                lambda s, h=horizon: tstool.rolling_trend(
                    s, h, return_mode="t-stat", log=False
                ),
            ).clip(-5, 5) / 5
            add(f"{label}_hlr_{horizon}d", label, f"{horizon}-day aggregate {label} breakout.", hlr)
            add(
                f"{label}_regt_{horizon}d",
                label,
                f"{horizon}-day aggregate {label} regression t-stat.",
                regt,
            )

    condition_names = (
        "vol_regime",
        "vol_zscore",
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
        "front_concentration",
        "c1_roll_concentration",
        "turnover_qtl",
        "price_oi_divergence",
        "price_volume_divergence",
    )
    conditions = {name: features[name].clip(-1, 1) for name in condition_names}
    for name, signal in conditions.items():
        add(name, "state", f"Standalone point-in-time {name} state.", signal)

    for price_name, price_signal in price_anchors.items():
        for condition_name, condition in conditions.items():
            add(
                f"{price_name}__scale__{condition_name}",
                "interaction",
                f"Scale {price_name} continuously by {condition_name}.",
                price_signal * (1.0 + condition),
            )
            add(
                f"{price_name}__cross__{condition_name}",
                "interaction",
                f"Signed interaction between {price_name} and {condition_name}.",
                price_signal * condition,
            )
            add(
                f"{price_name}__confirm__{condition_name}",
                "interaction",
                f"Use {condition_name} direction when absolute {price_name} is large.",
                price_signal.abs() * condition,
            )
            add(
                f"{price_name}__high_gate__{condition_name}",
                "gate",
                f"Trade {price_name} only when {condition_name} is positive.",
                price_signal * (condition > 0).astype(float),
            )
            add(
                f"{price_name}__low_gate__{condition_name}",
                "gate",
                f"Trade {price_name} only when {condition_name} is negative.",
                price_signal * (condition < 0).astype(float),
            )

    return recipes, pd.DataFrame(definitions)


def evaluate_library(
    recipes: Mapping[str, pd.DataFrame],
    features: Mapping[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
    validation_start: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose each recipe's sign on full-sample Sharpe and retain split diagnostics."""
    returns = features["returns"].loc[start:end]
    vol20 = features["vol20"].loc[start:end]
    splits = split_dates(returns.dropna(how="all").index, validation_start, oos_start)
    splits = {"full": (start, end), **splits}
    rows: list[dict[str, object]] = []
    pnl_store: dict[str, pd.Series] = {}
    turnover_store: dict[str, pd.Series] = {}

    for recipe_name, raw_signal in recipes.items():
        for family in ("time_series", "cross_sectional"):
            result = evaluate_signal(
                raw_signal.loc[start:end],
                returns,
                vol20,
                cost_bps,
                family,
                close_prices=features["close"].loc[start:end],
                execution_prices=features["execution_price"].loc[start:end],
            )
            gross = result["gross_asset_pnl"].sum(axis=1, min_count=1)
            turnover = result["turnover"]
            cost = turnover * cost_bps / 10_000.0
            positive = gross - cost
            negative = -gross - cost
            pos_sharpe = _series_metrics(positive, turnover)["sharpe"]
            neg_sharpe = _series_metrics(negative, turnover)["sharpe"]
            direction = -1 if pd.notna(neg_sharpe) and neg_sharpe > pos_sharpe else 1
            pnl = negative if direction == -1 else positive
            strategy = f"{recipe_name}__{'xs' if family == 'cross_sectional' else 'ts'}"
            pnl_store[strategy] = pnl
            turnover_store[strategy] = turnover
            for split, (split_start, split_end) in splits.items():
                rows.append(
                    {
                        "strategy": strategy,
                        "recipe": recipe_name,
                        "family": family,
                        "full_sample_direction": direction,
                        "split": split,
                        "start": split_start,
                        "end": split_end,
                        **_series_metrics(
                            pnl.loc[split_start:split_end],
                            turnover.loc[split_start:split_end],
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pnl_store), pd.DataFrame(turnover_store)


def diversified_selection(
    full_metrics: pd.DataFrame,
    pnl: pd.DataFrame,
    maximum: int,
    correlation_limit: float,
) -> list[str]:
    """Greedily retain high-Sharpe components that are not near duplicates."""
    ranked = full_metrics[
        (full_metrics["split"] == "full")
        & full_metrics["sharpe"].notna()
        & (full_metrics["sharpe"] > 0)
        & (full_metrics["n_days"] >= 1_500)
    ].sort_values("sharpe", ascending=False)
    selected: list[str] = []
    for strategy in ranked["strategy"]:
        if not selected:
            selected.append(strategy)
        else:
            correlations = pnl[selected].corrwith(pnl[strategy]).abs()
            if correlations.max() < correlation_limit:
                selected.append(strategy)
        if len(selected) >= maximum:
            break
    return selected


def optimize_nonnegative_sharpe(
    pnl: pd.DataFrame,
    maximum_weight: float,
) -> tuple[np.ndarray, float, bool]:
    """Maximize full-sample Sharpe over a long-only simplex of strategy books."""
    values = pnl.fillna(0.0).to_numpy(dtype=float)
    n_components = values.shape[1]
    if n_components == 0 or maximum_weight * n_components < 1.0 - 1e-12:
        return np.array([]), np.nan, False

    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    scale = math.sqrt(PNL_BDAYS)

    def objective(weights: np.ndarray) -> float:
        annualized_mean = float(mean @ weights) * scale
        variance = float(weights @ covariance @ weights)
        if variance <= 0:
            return 1e6
        return -(annualized_mean / math.sqrt(variance))

    def gradient(weights: np.ndarray) -> np.ndarray:
        portfolio_mean = float(mean @ weights)
        covariance_weight = covariance @ weights
        variance = float(weights @ covariance_weight)
        volatility = math.sqrt(max(variance, 1e-24))
        return -scale * (
            mean / volatility
            - portfolio_mean * covariance_weight / (volatility ** 3)
        )

    initial = np.repeat(1.0 / n_components, n_components)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, maximum_weight)] * n_components,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 600, "ftol": 1e-10, "disp": False},
    )
    return result.x, -float(result.fun), bool(result.success)


def search_blends(
    metrics: pd.DataFrame,
    pnl: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials: list[dict[str, object]] = []
    weights_store: dict[int, tuple[list[str], np.ndarray]] = {}
    for maximum in (15, 25, 40, 60, 100, 150, 250, 400):
        for correlation_limit in (0.80, 0.90, 0.97):
            selected = diversified_selection(metrics, pnl, maximum, correlation_limit)
            selected_pnl = pnl[selected]
            for maximum_weight in (0.05, 0.10, 0.20, 0.35, 0.60, 1.00):
                weights, sharpe, success = optimize_nonnegative_sharpe(
                    selected_pnl, maximum_weight
                )
                if not success:
                    continue
                trial_id = len(trials)
                trials.append(
                    {
                        "trial_id": trial_id,
                        "maximum_components": maximum,
                        "correlation_limit": correlation_limit,
                        "maximum_weight": maximum_weight,
                        "available_components": len(selected),
                        "active_components": int((weights > 1e-5).sum()),
                        "full_sample_sharpe": sharpe,
                    }
                )
                weights_store[trial_id] = (selected, weights)
    trials_frame = pd.DataFrame(trials).sort_values("full_sample_sharpe", ascending=False)
    if trials_frame.empty:
        raise RuntimeError("No feasible blend optimization trial completed.")
    best_id = int(trials_frame.iloc[0]["trial_id"])
    names, weights = weights_store[best_id]
    components = pd.DataFrame({"strategy": names, "weight": weights})
    components = components[components["weight"] > 1e-5].sort_values(
        "weight", ascending=False
    )
    return trials_frame, components


def write_outputs(
    config: ResearchConfig,
    definitions: pd.DataFrame,
    quality: pd.DataFrame,
    execution_quality: pd.DataFrame,
    metrics: pd.DataFrame,
    pnl: pd.DataFrame,
    turnover: pd.DataFrame,
    trials: pd.DataFrame,
    components: pd.DataFrame,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    weights = components.set_index("strategy")["weight"]
    blend_pnl = pnl[weights.index].fillna(0.0).mul(weights, axis=1).sum(axis=1)
    blend_turnover = turnover[weights.index].fillna(0.0).mul(weights, axis=1).sum(axis=1)
    splits = {
        "full": (config.start, config.end),
        **split_dates(blend_pnl.index, config.validation_start, config.oos_start),
    }
    blend_rows = [
        {
            "split": split,
            "start": start,
            "end": end,
            **_series_metrics(blend_pnl.loc[start:end], blend_turnover.loc[start:end]),
        }
        for split, (start, end) in splits.items()
    ]
    blend_metrics = pd.DataFrame(blend_rows)
    cost_rows: list[dict[str, float]] = []
    for cost_bps in (0.0, 2.0, 5.0, 10.0):
        repriced = blend_pnl + blend_turnover * (
            (config.cost_bps - cost_bps) / 10_000.0
        )
        cost_rows.append(
            {
                "cost_bps": cost_bps,
                **_series_metrics(repriced, blend_turnover),
            }
        )
    cost_sensitivity = pd.DataFrame(cost_rows)

    full_recipe_metadata = metrics[metrics["split"] == "full"][
        ["strategy", "recipe", "family", "full_sample_direction", "sharpe"]
    ]
    component_details = (
        components.merge(full_recipe_metadata, on="strategy", how="left")
        .merge(definitions[["recipe", "category", "thesis"]], on="recipe", how="left")
        .sort_values("weight", ascending=False)
    )
    category_weights = (
        component_details.groupby("category", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )
    family_weights = (
        component_details.groupby("family", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )

    paths = {
        "summary": config.output_dir / "summary.md",
        "definitions": config.output_dir / "recipe_definitions.csv",
        "quality": config.output_dir / "data_quality.csv",
        "execution_quality": config.output_dir / "execution_quality.csv",
        "metrics": config.output_dir / "oriented_recipe_metrics.csv",
        "trials": config.output_dir / "blend_optimization_trials.csv",
        "components": config.output_dir / "best_blend_components.csv",
        "blend_metrics": config.output_dir / "best_blend_metrics.csv",
        "cost_sensitivity": config.output_dir / "best_blend_cost_sensitivity.csv",
        "pnl": config.output_dir / "best_blend_daily_pnl.parquet",
        "library_pnl": config.output_dir / "oriented_recipe_daily_pnl.parquet",
        "library_turnover": config.output_dir / "oriented_recipe_daily_turnover.parquet",
        "chart": config.output_dir / "best_blend_cumulative_pnl.png",
        "config": config.output_dir / "run_config.json",
    }
    definitions.to_csv(paths["definitions"], index=False)
    quality.to_csv(paths["quality"], index=False)
    execution_quality.to_csv(paths["execution_quality"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    trials.to_csv(paths["trials"], index=False)
    component_details.to_csv(paths["components"], index=False)
    blend_metrics.to_csv(paths["blend_metrics"], index=False)
    cost_sensitivity.to_csv(paths["cost_sensitivity"], index=False)
    pd.DataFrame({"net_pnl": blend_pnl, "turnover": blend_turnover}).to_parquet(paths["pnl"])
    pnl.to_parquet(paths["library_pnl"])
    turnover.to_parquet(paths["library_turnover"])

    full_rank = metrics[metrics["split"] == "full"].sort_values("sharpe", ascending=False)
    top_names = full_rank.head(4)["strategy"].tolist()
    curves = pd.DataFrame({"optimized_blend": blend_pnl, **{name: pnl[name] for name in top_names}})
    fig, ax = plt.subplots(figsize=(12, 6))
    curves.fillna(0.0).cumsum().plot(ax=ax, linewidth=1.2)
    ax.set_title("Full-sample cumulative net PnL: optimized blend and top standalone recipes")
    ax.set_ylabel("Return on normalized gross exposure")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(paths["chart"], dpi=150)
    plt.close(fig)

    def table(frame: pd.DataFrame, rows: int = 15) -> str:
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

    lines = [
        "# Full-sample price/OI/volume Sharpe search",
        "",
        f"Search period: **{config.start.date()} to {config.end.date()}**; cost: **{config.cost_bps:.1f} bps per unit turnover**.",
        "",
        "> This is an oracle in-sample search. Recipe direction, component selection, and blend weights all use the full sample.",
        "",
        "## Optimized blend metrics",
        "",
        table(blend_metrics),
        "",
        "## Blend components",
        "",
        table(
            component_details[
                ["strategy", "weight", "category", "family", "full_sample_direction", "sharpe"]
            ],
            25,
        ),
        "",
        "## Blend composition",
        "",
        table(category_weights),
        "",
        table(family_weights),
        "",
        "## Full-sample trading-cost sensitivity",
        "",
        table(cost_sensitivity),
        "",
        "## Top standalone oriented recipes",
        "",
        table(full_rank[["strategy", "family", "full_sample_direction", "sharpe", "annual_return", "annual_vol", "mean_turnover"]]),
        "",
        "## Search diagnostics",
        "",
        f"- {len(definitions)} raw recipes were evaluated in time-series and cross-sectional form.",
        f"- {len(quality[quality['aligned_rows'] > 0])} products had aligned price and aggregate OI/volume data.",
        "- T's finalized signal is shifted once and executed at T+1 n305, then n310, then a1505 with the production-notebook fallback chain.",
        f"- {len(trials)} nonnegative blend configurations converged.",
        "- A recipe's inverse was selected whenever it produced the higher full-sample net Sharpe.",
        "- Component books are independently volatility-scaled, gross-normalized, lagged one day, and charged trading costs before blending.",
        "- Validation and OOS rows are descriptive only because the full sample controlled selection.",
    ]
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")
    paths["config"].write_text(
        json.dumps(
            {
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def run_search(config: ResearchConfig) -> dict[str, Path]:
    front = load_front_history(config.price_file, config.products)
    actual_products = tuple(product for product in config.products if product in front["close"])
    if actual_products != config.products:
        config = ResearchConfig(**{**asdict(config), "products": actual_products})
    aggregate, failures = load_or_build_aggregate_panel(config)
    features = build_feature_panels(front, aggregate)
    products = list(features["returns"].columns)
    front = {name: frame[products] for name, frame in front.items()}
    recipes, definitions = build_signal_library(front["close"], features)
    quality = data_quality_summary(front, features, failures, config.start, config.end)
    execution_quality = execution_quality_summary(front, config.start, config.end)
    metrics, pnl, turnover = evaluate_library(
        recipes,
        features,
        config.start,
        config.end,
        config.cost_bps,
        config.validation_start,
        config.oos_start,
    )
    trials, components = search_blends(metrics, pnl)
    return write_outputs(
        config,
        definitions,
        quality,
        execution_quality,
        metrics,
        pnl,
        turnover,
        trials,
        components,
    )


def parse_args(argv: Sequence[str] | None = None) -> ResearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", type=Path, default=Path("C:/dev/data/fut_d_20260930.parquet"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--validation-start", default="2019-01-01")
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/dev/data/output") / "oi_volume_sharpe_search",
    )
    parser.add_argument("--aggregate-cache", type=Path)
    parser.add_argument(
        "--aggregate-file",
        type=Path,
        default=Path("C:/dev/data/fut_oi_volume_20260930.parquet"),
    )
    parser.add_argument("--products", nargs="*", default=None)
    parser.add_argument("--contract-period", default="12m")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--refresh-aggregate", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    cache = args.aggregate_cache or (
        Path("C:/dev/data/output")
        / "oi_volume_trend_research"
        / f"aggregate_oi_volume_{args.end.replace('-', '')}.parquet"
    )
    return ResearchConfig(
        price_file=args.price_file.resolve(),
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=output_dir,
        products=tuple(args.products or BROAD_MKTS),
        aggregate_cache=cache.resolve(),
        validation_start=pd.Timestamp(args.validation_start),
        oos_start=pd.Timestamp(args.oos_start),
        aggregate_file=(
            None if args.aggregate_file is None else args.aggregate_file.resolve()
        ),
        contract_period=args.contract_period,
        cost_bps=args.cost_bps,
        refresh_aggregate=args.refresh_aggregate,
    )


if __name__ == "__main__":
    output_paths = run_search(parse_args())
    print("Sharpe-search outputs:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
