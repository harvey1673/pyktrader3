"""Sensitivity study for the volatility-ratio position-scaler steepness."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pycmqlib3.strategy.signal_repo import BROAD_MKTS
from tests.oi_volume_sharpe_search import _series_metrics
from tests.oi_volume_trend_research import (
    ResearchConfig,
    build_feature_panels,
    load_front_history,
    load_or_build_aggregate_panel,
    split_dates,
)
from tests.price_oi_volume_broad_research import (
    FAMILIES,
    SignalRecipe,
    build_broad_features,
    build_broad_library,
    evaluate_broad_signal,
    volatility_position_factor,
)


K_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0)


def _direction(net_positive: pd.Series, gross: pd.Series, cost: pd.Series) -> int:
    positive = _series_metrics(net_positive, cost * 0.0)["sharpe"]
    negative = _series_metrics(-gross - cost, cost * 0.0)["sharpe"]
    return -1 if pd.notna(negative) and negative > positive else 1


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    frame = frame.head(max_rows).copy()
    if frame.empty:
        return "No rows."
    formatted = frame.map(
        lambda value: ""
        if pd.isna(value)
        else f"{value:.4g}"
        if isinstance(value, float)
        else str(value)
    )
    columns = [str(column) for column in formatted.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in formatted.columns) + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def run(config: ResearchConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    front = load_front_history(config.price_file, config.products)
    aggregate, _ = load_or_build_aggregate_panel(config)
    features = build_broad_features(build_feature_panels(front, aggregate))
    recipes, definitions = build_broad_library(features["close"], features)
    anchor_names = definitions.loc[
        definitions["category"].eq("price_trend"), "recipe"
    ].tolist()
    splits = split_dates(
        features["returns"].loc[config.start : config.end].dropna(how="all").index,
        config.validation_start,
        config.oos_start,
    )
    periods = {"full": (config.start, config.end), **splits}
    cost_rate = config.cost_bps / 10_000.0
    factors = {
        k: volatility_position_factor(features["vol_ratio_240"], k=k)
        for k in K_VALUES
    }
    rows: list[dict[str, object]] = []
    pnl_by_k: dict[float, dict[str, pd.Series]] = {k: {} for k in K_VALUES}
    turnover_by_k: dict[float, dict[str, pd.Series]] = {k: {} for k in K_VALUES}

    for recipe_name in anchor_names:
        base_recipe = recipes[recipe_name]
        for family in FAMILIES:
            base_result = evaluate_broad_signal(
                base_recipe, features, family, config.start, config.end
            )
            train_start, train_end = splits["train"]
            train_gross = base_result["gross_pnl"].loc[train_start:train_end]
            train_cost = base_result["turnover"].loc[train_start:train_end] * cost_rate
            direction = _direction(train_gross - train_cost, train_gross, train_cost)

            for k in K_VALUES:
                scaled_recipe = SignalRecipe(
                    base_recipe.signal,
                    "vol_position_scale",
                    f"vol_ratio_240 position scale with k={k:g}",
                    recipe_name,
                    factors[k],
                )
                result = evaluate_broad_signal(
                    scaled_recipe, features, family, config.start, config.end
                )
                net = direction * result["gross_pnl"] - cost_rate * result["turnover"]
                key = f"{recipe_name}__{family}"
                pnl_by_k[k][key] = net
                turnover_by_k[k][key] = result["turnover"]
                for split, (start, end) in periods.items():
                    rows.append(
                        {
                            "recipe": recipe_name,
                            "family": family,
                            "train_direction": direction,
                            "k": k,
                            "split": split,
                            **_series_metrics(
                                net.loc[start:end], result["turnover"].loc[start:end]
                            ),
                        }
                    )

    metrics = pd.DataFrame(rows)
    portfolios: list[dict[str, object]] = []
    for k in K_VALUES:
        pnl = pd.DataFrame(pnl_by_k[k]).mean(axis=1)
        turnover = pd.DataFrame(turnover_by_k[k]).mean(axis=1)
        for split, (start, end) in periods.items():
            portfolios.append(
                {
                    "k": k,
                    "split": split,
                    **_series_metrics(pnl.loc[start:end], turnover.loc[start:end]),
                }
            )
    portfolio_metrics = pd.DataFrame(portfolios)

    comparison = metrics.merge(
        metrics[metrics["k"].eq(1.0)][
            ["recipe", "family", "split", "sharpe"]
        ].rename(columns={"sharpe": "sharpe_k1"}),
        on=["recipe", "family", "split"],
    )
    comparison["delta_sharpe_vs_k1"] = comparison["sharpe"] - comparison["sharpe_k1"]
    aggregate_comparison = (
        comparison.groupby(["k", "family", "split"], as_index=False)
        .agg(
            mean_delta_sharpe=("delta_sharpe_vs_k1", "mean"),
            median_delta_sharpe=("delta_sharpe_vs_k1", "median"),
            win_rate_vs_k1=("delta_sharpe_vs_k1", lambda values: (values > 0).mean()),
        )
    )

    validation = portfolio_metrics[portfolio_metrics["split"].eq("validation")]
    selected_k = float(validation.sort_values("sharpe", ascending=False).iloc[0]["k"])
    factor_examples = pd.DataFrame(
        {
            "vr_240": [0.8, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0],
            **{
                f"k_{k:g}": volatility_position_factor(
                    pd.DataFrame({"value": [0.8, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]}),
                    k=k,
                )["value"].to_numpy()
                for k in K_VALUES
            },
        }
    )

    paths = {
        "metrics": config.output_dir / "strategy_metrics.csv",
        "comparison": config.output_dir / "comparison_vs_k1.csv",
        "aggregate": config.output_dir / "aggregate_comparison.csv",
        "portfolio": config.output_dir / "equal_weight_portfolio_metrics.csv",
        "factor_examples": config.output_dir / "factor_examples.csv",
        "summary": config.output_dir / "summary.md",
        "config": config.output_dir / "run_config.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    aggregate_comparison.to_csv(paths["aggregate"], index=False)
    portfolio_metrics.to_csv(paths["portfolio"], index=False)
    factor_examples.to_csv(paths["factor_examples"], index=False)
    paths["config"].write_text(
        json.dumps(
            {
                **asdict(config),
                "price_file": str(config.price_file),
                "aggregate_file": str(config.aggregate_file),
                "aggregate_cache": str(config.aggregate_cache),
                "output_dir": str(config.output_dir),
                "start": str(config.start.date()),
                "end": str(config.end.date()),
                "validation_start": str(config.validation_start.date()),
                "oos_start": str(config.oos_start.date()),
                "k_values": K_VALUES,
                "families": FAMILIES,
                "direction_rule": "fixed from unscaled 2010-2018 price anchor",
                "selected_k_by_validation_sharpe": selected_k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = [
        "# Volatility-ratio position-scaler sensitivity",
        "",
        "The tested factor is `min(exp(-k * (vr_240 - 1)), 1)`, where "
        "`vr_240 = vol20 / vol240`.",
        "",
        "Directions are fixed from each unscaled price anchor's 2010-2018 result. "
        "The value of k selected mechanically from equal-weight validation Sharpe is "
        f"**{selected_k:g}**.",
        "",
        "## Equal-weight portfolio metrics",
        "",
        _markdown_table(portfolio_metrics),
        "",
        "## Aggregate recipe comparison versus k=1",
        "",
        _markdown_table(aggregate_comparison, 100),
        "",
        "## Factor shape",
        "",
        _markdown_table(factor_examples),
        "",
        "The 2024+ sample is reported but is not used to choose k.",
    ]
    paths["summary"].write_text("\n".join(summary), encoding="utf-8")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> ResearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", type=Path, default=Path("C:/dev/data/fut_d_20260930.parquet"))
    parser.add_argument("--aggregate-file", type=Path, default=Path("C:/dev/data/fut_oi_volume_20260930.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("C:/dev/data/output/vol_ratio_k_sensitivity"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--validation-start", default="2019-01-01")
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    return ResearchConfig(
        price_file=args.price_file.resolve(),
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=output_dir,
        products=tuple(BROAD_MKTS),
        aggregate_cache=output_dir / "unused_wtpy_cache.parquet",
        aggregate_file=args.aggregate_file.resolve(),
        validation_start=pd.Timestamp(args.validation_start),
        oos_start=pd.Timestamp(args.oos_start),
        cost_bps=args.cost_bps,
    )


if __name__ == "__main__":
    result = run(parse_args())
    for name, path in result.items():
        print(f"{name}: {path}")
