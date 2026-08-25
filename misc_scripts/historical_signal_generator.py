"""Generate strategy signal matrices from historical recipes, without factor DB reads.

The production factor updater combines three concerns: raw-data preparation,
signal routing, and MySQL persistence.  This module reuses its routing/config
objects and ``signal_repo`` recipes, but keeps generation entirely in memory so
scenario backtests do not depend on precomputed ``fut_fact_data`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratedSignalBundle:
    factor_values: Mapping[str, pd.DataFrame]
    volatility_overrides: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    traded_price_overrides: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    pnl_modes: Mapping[str, str] = field(default_factory=dict)
    execution_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    post_funcs: Mapping[str, str] = field(default_factory=dict)
    routes: Mapping[str, str] = field(default_factory=dict)


def prepare_historical_feature_data(
    price_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    assets: Sequence[str],
    *,
    commod_phycarry_dict: Mapping[str, str],
) -> pd.DataFrame:
    """Add the price-derived features used by ``signal_store`` recipes."""

    output = spot_df.copy().sort_index()
    output.index = pd.to_datetime(output.index)
    prices = price_df.sort_index()
    derived: dict[str, pd.Series] = {}
    for asset in assets:
        c1_close = (f"{asset}c1", "close")
        if c1_close not in prices.columns:
            continue
        close = pd.to_numeric(prices[c1_close], errors="coerce")
        derived[f"{asset}_px"] = close
        derived[f"{asset}_logret"] = np.log(close).diff()
        if (f"{asset}c1", "open") in prices.columns:
            derived[f"{asset}_colr"] = np.log(
                close / pd.to_numeric(prices[(f"{asset}c1", "open")], errors="coerce")
            )
        c2_close = (f"{asset}c2", "close")
        if c2_close in prices.columns:
            close2 = pd.to_numeric(prices[c2_close], errors="coerce")
            logret2 = np.log(close2).diff()
            basmom = derived[f"{asset}_logret"] - logret2
            derived[f"{asset}_logret2"] = logret2
            derived[f"{asset}_basmom"] = basmom
            for window in (5, 10, 20, 40, 60, 120, 180):
                derived[f"{asset}_basmom{window}"] = basmom.rolling(window).sum()

            c1_shift = (f"{asset}c1", "shift")
            c2_shift = (f"{asset}c2", "shift")
            c1_expiry = (f"{asset}c1", "expiry")
            c2_expiry = (f"{asset}c2", "expiry")
            required = (c1_shift, c2_shift, c1_expiry, c2_expiry)
            if all(column in prices.columns for column in required):
                days = (
                    pd.to_datetime(prices[c2_expiry])
                    - pd.to_datetime(prices[c1_expiry])
                ).dt.days.replace(0, np.nan)
                funding = output.get("r007_cn", pd.Series(index=output.index, dtype=float))
                funding = funding.reindex(prices.index).ffill().ewm(5).mean() / 100.0
                derived[f"{asset}_ryield"] = (
                    np.log(close)
                    - np.log(close2)
                    - pd.to_numeric(prices[c1_shift], errors="coerce")
                    + pd.to_numeric(prices[c2_shift], errors="coerce")
                ) / days * 365.0 + funding

        feature = commod_phycarry_dict.get(asset)
        expiry_col = (f"{asset}c1", "expiry")
        shift_col = (f"{asset}c1", "shift")
        if (
            feature
            and feature in output.columns
            and expiry_col in prices.columns
            and shift_col in prices.columns
        ):
            adder = {"SF": 350.0, "SM": 190.0}.get(asset, 0.0)
            frame = pd.concat(
                [
                    output[feature],
                    output.get("r007_cn", pd.Series(index=output.index, dtype=float)),
                    (close / np.exp(pd.to_numeric(prices[shift_col], errors="coerce"))).rename("c1"),
                    pd.to_datetime(prices[expiry_col]).rename("expiry"),
                ],
                axis=1,
            ).sort_index().ffill()
            funding = frame["r007_cn"].ewm(5).mean() / 100.0
            days = (frame["expiry"] - pd.Series(frame.index, index=frame.index)).dt.days
            derived[f"{asset}_phycarry"] = (
                (np.log(frame[feature] + adder) - np.log(frame["c1"]))
                / days.replace(0, np.nan)
                * 365.0
                + funding
            )

    if {"hc_logret", "rb_logret"}.issubset(derived):
        derived["hc_rb_diff"] = (
            np.log(prices[("hcc1", "close")]) - np.log(prices[("rbc1", "close")])
        )
    if {"rb_basmom", "hc_basmom"}.issubset(derived):
        derived["rb_hc_basmom_diff"] = derived["rb_basmom"] - derived["hc_basmom"]
    if {"rb_phycarry", "hc_phycarry"}.issubset(derived):
        derived["rb_hc_phycarry_diff"] = (
            derived["rb_phycarry"] - derived["hc_phycarry"]
        )
    if derived:
        output = pd.concat([output, pd.DataFrame(derived)], axis=1)
    return output.sort_index()


class HistoricalSignalGenerator:
    """Resolve configured signal names and construct their historical matrices."""

    ROUTE_ORDER = (
        "factors_by_func",
        "factors_by_spread",
        "factors_by_spread2",
        "factors_by_beta_neutral",
        "single_factors",
        "factors_by_asset",
    )

    def __init__(
        self,
        price_df: pd.DataFrame,
        spot_df: pd.DataFrame,
        assets: Sequence[str],
        *,
        signal_store: Mapping[str, Any],
        get_signal: Callable[..., pd.Series],
        factors_by_asset: Mapping[str, Sequence[str]],
        single_factors: Mapping[str, Sequence[str]],
        factors_by_spread: Mapping[str, Sequence[tuple[str, float]]],
        factors_by_spread2: Mapping[str, Sequence[str]],
        factors_by_beta_neutral: Mapping[str, Sequence[tuple[str, str, float]]],
        factors_by_func: Mapping[str, Mapping[str, Any]],
        spread_config: Mapping[str, Any],
        execution_name: Callable[[Any], str],
        vol_window: int = 20,
    ) -> None:
        self.price_df = price_df.sort_index().copy()
        self.price_df.index = pd.to_datetime(self.price_df.index)
        self.spot_df = spot_df.sort_index().copy()
        self.spot_df.index = pd.to_datetime(self.spot_df.index)
        self.assets = list(dict.fromkeys(assets))
        self.signal_store = signal_store
        self.get_signal = get_signal
        self.routes = {
            "factors_by_asset": factors_by_asset,
            "single_factors": single_factors,
            "factors_by_spread": factors_by_spread,
            "factors_by_spread2": factors_by_spread2,
            "factors_by_beta_neutral": factors_by_beta_neutral,
            "factors_by_func": factors_by_func,
        }
        self.spread_config = spread_config
        self.execution_name = execution_name
        self.vol_window = int(vol_window)

    def resolve_route(self, signal_name: str) -> str:
        matches = [route for route in self.ROUTE_ORDER if signal_name in self.routes[route]]
        if not matches:
            raise KeyError(
                f"Signal '{signal_name}' is not configured in any historical routing "
                f"registry: {', '.join(self.ROUTE_ORDER)}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Signal '{signal_name}' is ambiguously configured in: {', '.join(matches)}"
            )
        return matches[0]

    def _recipe_name(self, spec: Any) -> str:
        canonical = self.execution_name(spec)
        if canonical in self.signal_store:
            return canonical
        if spec.name in self.signal_store:
            return spec.name
        raise KeyError(
            f"Signal '{spec.name}' resolves to route '{self.resolve_route(spec.name)}' "
            f"but has no recipe in signal_store (checked '{canonical}' and '{spec.name}')"
        )

    def _signal_series(self, recipe_name: str, *, asset: str | None = None) -> pd.Series:
        return self.get_signal(
            self.spot_df,
            recipe_name,
            price_df=self.price_df,
            asset=asset,
            signal_repo=self.signal_store,
        )

    def _asset_close(self, asset: str, postfix: str = "c1") -> pd.Series:
        column = (f"{asset}{postfix}", "close")
        if column not in self.price_df.columns:
            raise KeyError(f"Missing historical price column {column}")
        return pd.to_numeric(self.price_df[column], errors="coerce")

    def _beta_neutral_frame(
        self,
        signal: pd.Series,
        pairs: Sequence[tuple[str, str, float]],
    ) -> pd.DataFrame:
        output = pd.DataFrame(0.0, index=signal.index, columns=self.assets)
        for trade_asset, index_asset, weight in pairs:
            if trade_asset not in output or index_asset not in output:
                continue
            returns = pd.concat(
                [
                    self._asset_close(trade_asset).pct_change().rename("trade"),
                    self._asset_close(index_asset).pct_change().rename("index"),
                ],
                axis=1,
            ).dropna(how="all").ffill()
            smoothed = returns.rolling(5).mean()
            beta = smoothed["index"].rolling(244).cov(smoothed["trade"]) / smoothed[
                "index"
            ].rolling(244).var()
            spread_returns = returns["trade"] - beta * returns["index"]
            spread_vol = spread_returns.rolling(self.vol_window).std()
            trade_vol = returns["trade"].rolling(self.vol_window).std()
            index_vol = returns["index"].rolling(self.vol_window).std()
            aligned = pd.concat(
                [signal.rename("signal"), beta, spread_vol, trade_vol, index_vol],
                axis=1,
            ).ffill()
            aligned.columns = ["signal", "beta", "spread_vol", "trade_vol", "index_vol"]
            output[trade_asset] += (
                weight * aligned["signal"] * aligned["trade_vol"] / aligned["spread_vol"]
            )
            output[index_asset] -= (
                weight
                * aligned["signal"]
                * aligned["beta"]
                * aligned["index_vol"]
                / aligned["spread_vol"]
            )
        return output

    def generate(self, specs: Sequence[Any]) -> GeneratedSignalBundle:
        factors: dict[str, pd.DataFrame] = {}
        volatility_overrides: dict[str, pd.DataFrame] = {}
        traded_price_overrides: dict[str, pd.DataFrame] = {}
        pnl_modes: dict[str, str] = {}
        execution_overrides: dict[str, Mapping[str, Any]] = {}
        post_funcs: dict[str, str] = {}
        route_results: dict[str, str] = {}

        unique_specs: dict[str, Any] = {}
        for spec in specs:
            unique_specs.setdefault(spec.factor_name, spec)

        for spec in unique_specs.values():
            factor_key = spec.factor_name
            route = self.resolve_route(spec.name)
            recipe_name = None
            if route != "factors_by_func":
                recipe_name = self._recipe_name(spec)
                post_funcs[self.execution_name(spec)] = str(
                    self.signal_store[recipe_name][1][7]
                )

            if route == "factors_by_func":
                config = self.routes[route][spec.name]
                frame = config["func"](
                    self.price_df, self.spot_df, **dict(config.get("args", {}))
                )
            elif route == "factors_by_asset":
                configured = self.routes[route][spec.name]
                tradable = [asset for asset in self.assets if asset in configured]
                frame = pd.DataFrame(
                    {
                        asset: self._signal_series(recipe_name, asset=asset)
                        for asset in tradable
                    }
                )
            elif route == "single_factors":
                signal = self._signal_series(recipe_name)
                configured = self.routes[route][spec.name]
                frame = pd.DataFrame(
                    {asset: signal for asset in self.assets if asset in configured}
                )
            elif route == "factors_by_spread":
                signal = self._signal_series(recipe_name)
                frame = pd.DataFrame(
                    {
                        asset: signal * float(weight)
                        for asset, weight in self.routes[route][spec.name]
                        if asset in self.assets
                    }
                )
            elif route == "factors_by_beta_neutral":
                signal = self._signal_series(recipe_name)
                frame = self._beta_neutral_frame(signal, self.routes[route][spec.name])
            elif route == "factors_by_spread2":
                signal = self._signal_series(recipe_name)
                spread_name = self.routes[route][spec.name][0]
                legs, vol_window, _roll_label, contract_number = self.spread_config[
                    spread_name
                ]
                frame = pd.DataFrame(
                    {
                        asset: signal * float(weight)
                        for asset, weight in legs
                        if asset in self.assets
                    }
                )
                postfix = f"d{int(contract_number)}"
                traded = pd.DataFrame(
                    {
                        asset: self._asset_close(asset, postfix=postfix)
                        for asset in frame.columns
                    }
                )
                spread_price = sum(
                    self._asset_close(asset, postfix=postfix) * float(weight)
                    for asset, weight in legs
                )
                spread_vol = spread_price.diff().rolling(int(vol_window)).std()
                volatility_overrides[factor_key] = pd.DataFrame(
                    {asset: spread_vol for asset in frame.columns}
                )
                traded_price_overrides[factor_key] = traded
                pnl_modes[factor_key] = "px"
                execution_overrides[self.execution_name(spec)] = {
                    "win": "close",
                    "lag": 1,
                }
            else:  # pragma: no cover - resolve_route owns the exhaustive list
                raise AssertionError(route)

            if frame.empty or len(frame.columns) == 0:
                raise ValueError(
                    f"Signal '{spec.name}' resolved through {route} but generated no "
                    "assets in the strategy universe"
                )
            frame = frame.sort_index()
            frame.index = pd.to_datetime(frame.index)
            if factor_key in factors:
                existing = factors[factor_key]
                if not existing.equals(frame):
                    raise ValueError(
                        f"Factor '{factor_key}' generated different raw matrices"
                    )
            else:
                factors[factor_key] = frame
            route_results[factor_key] = route

        return GeneratedSignalBundle(
            factor_values=factors,
            volatility_overrides=volatility_overrides,
            traded_price_overrides=traded_price_overrides,
            pnl_modes=pnl_modes,
            execution_overrides=execution_overrides,
            post_funcs=post_funcs,
            routes=route_results,
        )
