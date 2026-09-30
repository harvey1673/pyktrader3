"""Build the price/OI/volume signal research notebook under tests.

The notebook is generated with nbformat so its cell order and metadata remain
stable and reviewable.  Run this file again after changing the research helper
functions to refresh the notebook source.
"""

from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "bktest" / "bt_signal_price_oi_volume.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        r"""
# Price, aggregate OI, and volume signals

This notebook lives beside `bktest/bt_signal - px.ipynb`. The experimental
implementation remains under `tests`, and this notebook uses the research
helpers in `tests/oi_volume_trend_research.py` and
`tests/oi_volume_sharpe_search.py`.

It reconstructs the selected signals from the underlying price and WTPY
aggregate OI/volume data, builds the component sleeves and combined portfolio,
and reconciles the results to the saved full-sample search artifacts.
"""
    ),
    markdown(
        r"""
## tl;dr

The executed checks below are the source of truth. The prior search selected a
nonnegative blend of independently risk-scaled time-series and cross-sectional
sleeves. Recipe direction, selection, and blend weights were optimized on the
full 2010-2026 sample, so the 2024-2026 row is a descriptive holdout diagnostic,
not an untouched out-of-sample estimate.
"""
    ),
    markdown(
        r"""
## Context & Methods

### Key assumptions

- Signals use information through trading date **T**.
- Target holdings are shifted once and entered on **T+1**.
- Execution price priority is `n305`, then `n310`, then `a1505`, followed by
  the same fallback chain used by the research helper.
- Aggregate open interest and volume come from WTPY with
  `contract_period="12m"`.
- Each component sleeve is volatility-scaled and gross-normalized before the
  static blend weights are applied.
- The saved research result charges 2 bps to each sleeve's turnover before
  blending. This is conservative because opposite component trades are not
  netted for cost purposes.
"""
    ),
    code(
        r"""
%matplotlib inline

import json
import importlib.util
import math
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

repo_candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
repo_candidates.append(Path("C:/dev/pyktrader3"))
REPO_ROOT = next(
    (
        path
        for path in repo_candidates
        if (path / "tests" / "oi_volume_sharpe_search.py").exists()
    ),
    None,
)
if REPO_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the repository containing tests/oi_volume_sharpe_search.py"
    )
WTPY_ROOT = Path("C:/dev/wtpy")
for path in (REPO_ROOT, WTPY_ROOT):
    path_text = str(path)
    while path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)

from pycmqlib3.analytics import tstool


def load_research_module(module_name, filename):
    # Load a local research module without relying on the generic `tests` package.
    module_path = REPO_ROOT / "tests" / filename
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


# Some Python environments contain an unrelated package named `tests`. Register
# the local research namespace explicitly so the notebook is launch-directory safe.
local_tests_package = types.ModuleType("tests")
local_tests_package.__path__ = [str(REPO_ROOT / "tests")]
sys.modules["tests"] = local_tests_package

trend_research = load_research_module(
    "tests.oi_volume_trend_research", "oi_volume_trend_research.py"
)
sharpe_research = load_research_module(
    "tests.oi_volume_sharpe_search", "oi_volume_sharpe_search.py"
)

build_signal_library = sharpe_research.build_signal_library
_series_metrics = sharpe_research._series_metrics
ResearchConfig = trend_research.ResearchConfig
build_feature_panels = trend_research.build_feature_panels
data_quality_summary = trend_research.data_quality_summary
evaluate_signal = trend_research.evaluate_signal
execution_quality_summary = trend_research.execution_quality_summary
load_front_history = trend_research.load_front_history
load_or_build_aggregate_panel = trend_research.load_or_build_aggregate_panel
split_dates = trend_research.split_dates

pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = Path("C:/dev/data/output") / "oi_volume_sharpe_search"
with (OUTPUT_DIR / "run_config.json").open(encoding="utf-8") as handle:
    saved_config = json.load(handle)

config = ResearchConfig(
    price_file=Path(saved_config["price_file"]),
    start=pd.Timestamp(saved_config["start"]),
    end=pd.Timestamp(saved_config["end"]),
    output_dir=Path(saved_config["output_dir"]),
    products=tuple(saved_config["products"]),
    aggregate_cache=Path(saved_config["aggregate_cache"]),
    validation_start=pd.Timestamp(saved_config["validation_start"]),
    oos_start=pd.Timestamp(saved_config["oos_start"]),
    contract_period=saved_config["contract_period"],
    cost_bps=float(saved_config["cost_bps"]),
    refresh_aggregate=False,
)

display(pd.Series({
    "price_file": str(config.price_file),
    "aggregate_cache": str(config.aggregate_cache),
    "research_period": f"{config.start.date()} to {config.end.date()}",
    "validation_start": str(config.validation_start.date()),
    "oos_start": str(config.oos_start.date()),
    "cost_bps": config.cost_bps,
    "requested_products": len(config.products),
}, name="value").to_frame())
"""
    ),
    markdown("## Data\n\n### 1. Load c1/c2 history and WTPY aggregates"),
    code(
        r"""
front = load_front_history(config.price_file, config.products)
actual_products = tuple(product for product in config.products if product in front["close"])
if actual_products != config.products:
    config = ResearchConfig(**{**config.__dict__, "products": actual_products})

aggregate_panel, aggregate_failures = load_or_build_aggregate_panel(config)
features = build_feature_panels(front, aggregate_panel)
products = list(features["returns"].columns)
front = {name: frame[products] for name, frame in front.items()}

print(
    f"Loaded {len(products)} products, {len(features['close']):,} price dates, "
    f"and {len(aggregate_panel):,} aggregate-data dates."
)
print(f"Aligned research window: {config.start.date()} to {config.end.date()}")
"""
    ),
    markdown("### 2. Audit data and execution-price coverage"),
    code(
        r"""
quality = data_quality_summary(
    front, features, aggregate_failures, config.start, config.end
)
execution_quality = execution_quality_summary(front, config.start, config.end)

quality_view = quality.sort_values(
    ["aligned_rows", "product"], ascending=[False, True]
).reset_index(drop=True)
execution_columns = [
    "execution_rows", "n305_rows", "n310_rows", "a1505_rows", "a1535_rows",
    "d_twap_rows", "close_fallback_rows", "ffill_rows",
]
execution_totals = execution_quality[execution_columns].sum()

display(quality_view.head(15))
display(execution_totals.rename("rows").to_frame())

assert quality["aligned_rows"].gt(0).any(), "No aligned price/OI/volume observations."
assert execution_totals["execution_rows"] > 0, "No usable execution prices."
"""
    ),
    markdown("### 3. Inspect the feature panels"),
    code(
        r"""
feature_names = [
    "vol_regime", "vol_hlratio", "vol_regt",
    "oi_momentum", "oi_breakout", "oi_regt",
    "volume_activity", "volume_breakout", "volume_regt",
    "price_oi_corr", "price_volume_corr",
    "front_concentration", "c1_roll_concentration",
]

feature_coverage = pd.DataFrame({
    name: {
        "non_null_cells": int(features[name].loc[config.start:config.end].notna().sum().sum()),
        "products": int(features[name].loc[config.start:config.end].notna().any().sum()),
        "min": features[name].loc[config.start:config.end].min().min(),
        "max": features[name].loc[config.start:config.end].max().max(),
    }
    for name in feature_names
}).T
display(feature_coverage.round(4))
"""
    ),
    markdown(
        r"""
## Signal construction

`build_signal_library` creates the standalone price, OI, and volume signals and
the nonlinear `scale`, `cross`, `confirm`, `high_gate`, and `low_gate`
combinations. The table below selects the recipes and orientations retained by
the prior full-sample Sharpe search.
"""
    ),
    code(
        r"""
recipes, recipe_definitions = build_signal_library(front["close"], features)
components = pd.read_csv(OUTPUT_DIR / "best_blend_components.csv")
components = components.sort_values("weight", ascending=False).reset_index(drop=True)

missing_recipes = sorted(set(components["recipe"]) - set(recipes))
assert not missing_recipes, f"Selected recipes are missing: {missing_recipes}"
assert np.isclose(components["weight"].sum(), 1.0), "Blend weights must sum to one."

display(components[[
    "strategy", "weight", "family", "full_sample_direction", "sharpe", "thesis"
]].round({"weight": 5, "sharpe": 3}))
"""
    ),
    markdown("### 4. Independently verify representative formulas"),
    code(
        r"""
def apply_columns(frame, function):
    return pd.DataFrame(
        {column: function(frame[column].dropna()).reindex(frame.index) for column in frame},
        index=frame.index,
    )


log_price = np.log(features["close"].where(features["close"] > 0))
daily_vol_60 = features["returns"].rolling(60, min_periods=40).std()
manual_price_mom_240d = (
    log_price.diff(240) / (daily_vol_60 * math.sqrt(240))
).clip(-2, 2) / 2

manual_low_vol_gate = manual_price_mom_240d * (features["vol_regt"] < 0).astype(float)

log_oi = np.log(features["aggregate_oi"].where(features["aggregate_oi"] > 0))
manual_oi_change_z_10d = apply_columns(
    log_oi.diff(10), lambda series: tstool.zscore_roll(series, 252)
).clip(-2, 2) / 2

log_volume = np.log(features["aggregate_volume"].where(features["aggregate_volume"] > 0))
manual_volume_regt_120d = apply_columns(
    log_volume,
    lambda series: tstool.rolling_trend(
        series, 120, return_mode="t-stat", log=False
    ),
).clip(-5, 5) / 5

formula_checks = pd.Series({
    "price_mom_240d_max_abs_diff": (
        recipes["price_mom_240d"] - manual_price_mom_240d
    ).abs().max().max(),
    "low_gate_vol_regt_max_abs_diff": (
        recipes["price_mom_240d__low_gate__vol_regt"] - manual_low_vol_gate
    ).abs().max().max(),
    "oi_change_z_10d_max_abs_diff": (
        recipes["oi_change_z_10d"] - manual_oi_change_z_10d
    ).abs().max().max(),
    "volume_regt_120d_max_abs_diff": (
        recipes["volume_regt_120d"] - manual_volume_regt_120d
    ).abs().max().max(),
    "low_gate_xs_max_abs_daily_mean": tstool.xs_demean(
        recipes["price_mom_240d__low_gate__vol_regt"]
    ).mean(axis=1).abs().max(),
}, name="error")

display(formula_checks.to_frame())
assert formula_checks.max() < 1e-12, formula_checks
"""
    ),
    markdown(
        r"""
### 5. Convert each selected signal into an executable sleeve

For a cross-sectional recipe, daily demeaning occurs before risk scaling. Each
sleeve then divides by annualized 20-day volatility, normalizes gross target
exposure to one, shifts targets by one day, applies the execution-price
adjustment, and charges turnover cost. A direction of `-1` reverses gross P&L
but does not reverse or refund costs.
"""
    ),
    code(
        r"""
component_pnl = pd.DataFrame()
component_turnover = pd.DataFrame()
component_gross_pnl = pd.DataFrame()
component_targets = {}
component_holdings = {}
lag_errors = {}

for row in components.itertuples(index=False):
    result = evaluate_signal(
        recipes[row.recipe].loc[config.start:config.end],
        features["returns"].loc[config.start:config.end],
        features["vol20"].loc[config.start:config.end],
        cost_bps=0.0,
        family=row.family,
        close_prices=features["close"].loc[config.start:config.end],
        execution_prices=features["execution_price"].loc[config.start:config.end],
    )
    direction = float(row.full_sample_direction)
    gross_pnl = direction * result["gross_asset_pnl"].sum(axis=1, min_count=1)
    turnover = result["turnover"]
    net_pnl = gross_pnl - turnover * config.cost_bps / 10_000.0

    component_gross_pnl[row.strategy] = gross_pnl
    component_turnover[row.strategy] = turnover
    component_pnl[row.strategy] = net_pnl
    component_targets[row.strategy] = direction * result["weights"]
    component_holdings[row.strategy] = direction * result["holdings"]
    lag_errors[row.strategy] = (
        component_holdings[row.strategy]
        - component_targets[row.strategy].shift(1)
    ).abs().max().max()

lag_check = pd.Series(lag_errors, name="max_abs_lag_error")
display(lag_check.describe().to_frame())
assert lag_check.max() < 1e-12, lag_check.sort_values(ascending=False).head()
"""
    ),
    markdown("## Portfolio construction and validation"),
    code(
        r"""
blend_weights = components.set_index("strategy")["weight"]
ordered_strategies = blend_weights.index.tolist()

# This exactly mirrors the saved research blend: costs are charged inside each sleeve.
blend_pnl = component_pnl[ordered_strategies].mul(blend_weights, axis=1).sum(axis=1)
blend_gross_pnl = component_gross_pnl[ordered_strategies].mul(
    blend_weights, axis=1
).sum(axis=1)
blend_sleeve_turnover = component_turnover[ordered_strategies].mul(
    blend_weights, axis=1
).sum(axis=1)

# These are the actual aggregate portfolio targets/holdings after sleeves are combined.
portfolio_targets = sum(
    component_targets[name] * blend_weights[name] for name in ordered_strategies
)
portfolio_holdings = sum(
    component_holdings[name] * blend_weights[name] for name in ordered_strategies
)
portfolio_netted_turnover = (
    portfolio_holdings - portfolio_holdings.shift(1).fillna(0.0)
).abs().sum(axis=1, min_count=1)
portfolio_netted_cost_pnl = (
    blend_gross_pnl - portfolio_netted_turnover * config.cost_bps / 10_000.0
)

stored_library_pnl = pd.read_parquet(
    OUTPUT_DIR / "oriented_recipe_daily_pnl.parquet",
    columns=ordered_strategies,
)
stored_blend = pd.read_parquet(OUTPUT_DIR / "best_blend_daily_pnl.parquet")

component_reconciliation = (
    component_pnl[ordered_strategies] - stored_library_pnl[ordered_strategies]
).abs().max().sort_values(ascending=False)
blend_pnl_error = (blend_pnl - stored_blend["net_pnl"]).abs().max()
blend_turnover_error = (
    blend_sleeve_turnover - stored_blend["turnover"]
).abs().max()

reconciliation = pd.Series({
    "max_component_pnl_abs_diff": component_reconciliation.max(),
    "blend_pnl_max_abs_diff": blend_pnl_error,
    "blend_turnover_max_abs_diff": blend_turnover_error,
    "max_target_gross_exposure": portfolio_targets.abs().sum(axis=1).max(),
    "max_holding_gross_exposure": portfolio_holdings.abs().sum(axis=1).max(),
}, name="value")
display(reconciliation.to_frame())

assert component_reconciliation.max() < 1e-12, component_reconciliation.head()
assert blend_pnl_error < 1e-12
assert blend_turnover_error < 1e-12
assert portfolio_targets.abs().sum(axis=1).max() <= 1.0 + 1e-12
"""
    ),
    markdown("### 6. Recompute full, training, validation, and 2024-2026 metrics"),
    code(
        r"""
split_map = {
    "full": (config.start, config.end),
    **split_dates(blend_pnl.index, config.validation_start, config.oos_start),
}

metric_rows = []
for split_name, (split_start, split_end) in split_map.items():
    metric_rows.append({
        "split": split_name,
        "start": split_start,
        "end": split_end,
        **_series_metrics(
            blend_pnl.loc[split_start:split_end],
            blend_sleeve_turnover.loc[split_start:split_end],
        ),
    })
recomputed_metrics = pd.DataFrame(metric_rows)

stored_metrics = pd.read_csv(
    OUTPUT_DIR / "best_blend_metrics.csv",
    parse_dates=["start", "end"],
)
metric_columns = [
    "sharpe", "annual_return", "annual_vol", "max_drawdown",
    "mean_turnover", "pnl_per_turnover", "n_days",
]
metric_differences = (
    recomputed_metrics.set_index("split")[metric_columns]
    - stored_metrics.set_index("split")[metric_columns]
).abs()

display(recomputed_metrics.round(6))
display(metric_differences.max().rename("max_abs_difference").to_frame())
assert metric_differences.to_numpy().max() < 1e-12
"""
    ),
    markdown("## Results"),
    code(
        r"""
curves = pd.DataFrame({
    "saved-cost convention": blend_pnl,
    "cost after sleeve netting": portfolio_netted_cost_pnl,
}).fillna(0.0).cumsum()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(curves.index, curves["saved-cost convention"], color="#315b7d", label="saved-cost convention")
ax.plot(curves.index, curves["cost after sleeve netting"], color="#d08b34", linestyle="--", label="cost after sleeve netting")
ax.axvline(config.validation_start, color="#666666", linestyle=":", linewidth=1.2, label="validation start")
ax.axvline(config.oos_start, color="#111111", linestyle=":", linewidth=1.2, label="2024 diagnostic start")
ax.set_title("Price/OI/volume blend: cumulative net P&L")
ax.set_ylabel("Return on normalized gross exposure")
ax.set_xlabel("Trading date")
ax.legend(loc="upper left", ncol=2)
plt.show()
"""
    ),
    code(
        r"""
top_components = components.nlargest(15, "weight").sort_values("weight")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_components["strategy"], top_components["weight"], color="#315b7d")
ax.set_title("Largest component weights in the optimized blend")
ax.set_xlabel("Static blend weight")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
plt.show()
"""
    ),
    code(
        r"""
top_names = components.nlargest(15, "weight")["strategy"].tolist()
weekly_corr = component_pnl[top_names].resample("W-FRI").sum().corr()

fig, ax = plt.subplots(figsize=(11, 9))
image = ax.imshow(weekly_corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(top_names)), labels=top_names, rotation=55, ha="right")
ax.set_yticks(range(len(top_names)), labels=top_names)
ax.set_title("Weekly net-P&L correlation: 15 largest sleeves")
fig.colorbar(image, ax=ax, label="Correlation", fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()
"""
    ),
    markdown("### 7. Inspect the latest aggregate portfolio holdings"),
    code(
        r"""
latest_date = portfolio_holdings.dropna(how="all").index.max()
latest_holdings = portfolio_holdings.loc[latest_date].dropna().sort_values()
latest_table = pd.concat(
    [latest_holdings.head(10), latest_holdings.tail(10)]
).drop_duplicates().sort_values()

print(f"Latest portfolio holding date: {latest_date.date()}")
display(latest_table.rename("holding").to_frame().style.format("{:+.3%}"))

turnover_comparison = pd.DataFrame({
    "sleeve_level": blend_sleeve_turnover,
    "after_cross_sleeve_netting": portfolio_netted_turnover,
}).loc[config.start:config.end]
display(turnover_comparison.mean().rename("mean_daily_turnover").to_frame())
"""
    ),
    markdown(
        r"""
## Takeaways

- The formula checks independently rebuild representative price, volatility,
  aggregate-OI, and aggregate-volume recipes.
- The lag check verifies that every selected sleeve holds yesterday's target,
  so date-T information is not traded until T+1.
- The component and blend reconciliations must be at floating-point tolerance
  versus the saved research artifacts.
- `_xs` recipes are demeaned before risk scaling. A product gated to zero by a
  raw recipe can therefore receive a relative exposure after cross-sectional
  demeaning.
- The optimized Sharpe is an oracle full-sample result. The 2024-2026 metrics
  remain useful diagnostics but should not be interpreted as untouched OOS
  evidence.
- The saved cost convention is deliberately conservative: it charges component
  sleeve turnover separately. The netted-cost curve shows the implementation
  benefit available if trades are combined at the final portfolio level.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
