"""Focused current-versus-CTD physical-carry backtest for research review."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pycmqlib3.analytics.backtest_utils import generate_holding_from_signal
from pycmqlib3.analytics.btmetrics import MetricsBase
from pycmqlib3.analytics.tstool import calc_funda_signal
from pycmqlib3.utility.misc import CHN_Holidays
from tests.ctd_phycarry_adapter import (
    add_fx_converted_energy_spots,
    add_priority_ctd_phycarry,
)


ASSETS = ("j", "jm", "ss", "SM", "SF")
EXTENDED_ASSETS = ("l", "pp", "v", "eg", "eb", "TA", "MA", "ru", "bu", "pg")
CURRENT_SPOTS = {
    "j": ("coke_sh_xb", 0.0),
    "jm": ("ckc_outstock_ganqimaodu", 0.0),
    "ss": ("ss_304_gross_wuxi", 0.0),
    "SM": ("SM_65s17_tj", 190.0),
    "SF": ("SF_72_ningxia", 350.0),
}
EXTENDED_CURRENT_SPOTS = {
    "l": ("l_7042_tj", 0.0),
    "pp": ("pp_t30s_shaoxing_hz", 0.0),
    "v": ("pvc_cac2_east", 0.0),
    "eg": ("eg_east_spot", 0.0),
    "eb": ("eb_east_spot", 0.0),
    "TA": ("TA_east_spot", 0.0),
    "MA": ("MA_spot_jiangsu", 0.0),
    "ru": ("ru_scrwf_kunming", 0.0),
    "bu": ("bu_heavy_shandong", 0.0),
    "pg": ("propane_cfr_south_cny_vat", 0.0),
}
RECIPES = {
    "ema_5_10": ([5, 10], ""),
    "ema_1": ([1, 2, 1], "ema1"),
}
IFIND_CTD_CODES = {
    "S004626322": "coke_sub_a_rz_outstock",
}
MYSTEEL_CTD_CODES = {
    "ID00003727": "ss_304_2b_hongwang_wuxi",
    "ID00102802": "coke_sub_a_rz_haoyu",
    "ID00258827": "SF_72_tj",
    "ID01199235": "coke_sub_a_dry_lvliang_jinyan",
    "ID01892939": "ckc_mongol5_ts",
    "RE00035725": "ckc_midsulfur_jiexiu_kaijia",
}


def add_current_phycarry(
    price_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    spot_map=CURRENT_SPOTS,
) -> pd.DataFrame:
    """Reproduce the five current spot choices in the metal-summary notebook."""

    output = spot_df.copy().sort_index()
    output.index = pd.to_datetime(output.index)
    derived = {}
    for asset, (spot_col, adder) in spot_map.items():
        close = pd.to_numeric(price_df[(f"{asset}c1", "close")], errors="coerce")
        shift = pd.to_numeric(price_df[(f"{asset}c1", "shift")], errors="coerce")
        frame = pd.concat(
            [
                pd.to_numeric(output[spot_col], errors="coerce").rename("spot"),
                output["r007_cn"],
                (close / np.exp(shift)).rename("c1"),
                pd.to_datetime(price_df[(f"{asset}c1", "expiry")]).rename("expiry"),
            ],
            axis=1,
        ).sort_index().ffill()
        days = (frame["expiry"] - pd.Series(frame.index, index=frame.index)).dt.days
        derived[f"{asset}_phycarry"] = (
            (np.log(frame["spot"] + adder) - np.log(frame["c1"]))
            / days.replace(0, np.nan)
            * 365.0
            + frame["r007_cn"].ewm(5).mean() / 100.0
        )
    output = output.drop(columns=list(derived), errors="ignore")
    return pd.concat([output, pd.DataFrame(derived)], axis=1).sort_index()


def refresh_ctd_spots_from_db(spot_df: pd.DataFrame) -> pd.DataFrame:
    """Overlay the requested CTD inputs, preserving MySteel source priority."""

    from pycmqlib3.utility import dbaccess

    output = spot_df.copy()
    for source, mapping in (
        ("ifind", IFIND_CTD_CODES),
        ("mysteel", MYSTEEL_CTD_CODES),
    ):
        refreshed = dbaccess.load_codes_from_edb(
            list(mapping),
            source=source,
            column_name="index_code",
        ).rename(columns=mapping)
        output = output.drop(columns=list(mapping.values()), errors="ignore")
        output = pd.concat([output, refreshed], axis=1)
    return output.sort_index()


def execution_returns(price_df: pd.DataFrame, assets=ASSETS) -> pd.DataFrame:
    result = pd.DataFrame(index=price_df.index)
    for asset in assets:
        traded = pd.to_numeric(price_df[(f"{asset}c1", "n305")], errors="coerce").copy()
        fallback = pd.to_numeric(price_df[(f"{asset}c1", "a1505")], errors="coerce")
        traded = traded.fillna(fallback)
        result[asset] = traded.dropna().pct_change()
    return result


def signal_frame(feature_df, price_index, param_rng, post_func, assets=ASSETS):
    bdates = pd.bdate_range(
        start=price_index.min(),
        end=price_index.max(),
        freq="C",
        holidays=CHN_Holidays,
    )
    cdates = pd.date_range(price_index.min(), price_index.max(), freq="D")
    result = pd.DataFrame(index=price_index)
    for asset in assets:
        signal = calc_funda_signal(
            feature_df,
            f"{asset}_phycarry",
            "ema",
            param_rng,
            proc_func="",
            chg_func="",
            bullish=True,
            freq="price",
            signal_cap=[-2.0, 2.0],
            bdates=bdates,
            post_func=post_func,
            vol_win=120,
        )
        result[asset] = signal.reindex(cdates).ffill().reindex(price_index)
    return result


def backtest(feature_df, returns, param_rng, post_func, cutoff, assets=ASSETS):
    signals = signal_frame(feature_df, returns.index, param_rng, post_func, assets)
    volatility = returns.rolling(20).std()
    holdings = generate_holding_from_signal(
        signals,
        volatility,
        risk_scaling=1.0,
        asset_scaling=False,
    )
    holdings = holdings.loc[cutoff:, assets]
    selected_returns = returns.loc[cutoff:, assets]
    metrics = {}
    for label, cost in (("gross", 0.0), ("net_2bp", 2e-4)):
        report = MetricsBase(
            holdings=holdings.copy(),
            returns=selected_returns.copy(),
            shift_holdings=1,
            cost_dict={asset: cost for asset in assets},
        ).calculate_pnl_stats(
            shift=0,
            use_log_returns=False,
            tenors=["1y", "3y", "5y"],
            perf_metrics=["sharpe", "std"],
        )
        metrics[(label, "sharpe_all")] = report["sharpe"].loc["sharpe"]
        metrics[(label, "daily_std_all")] = report["std"].loc["std"]
        for tenor in ("1y", "3y", "5y"):
            metrics[(label, f"sharpe_{tenor}")] = report["sharpe"].loc[
                f"sharpe_{tenor}"
            ]
    return pd.Series(metrics)


def run(
    price_path: Path,
    spot_path: Path,
    cutoff: str,
    end_date: str,
    *,
    refresh_db: bool = False,
    assets=ASSETS,
    current_spots=CURRENT_SPOTS,
) -> pd.DataFrame:
    prices = pd.read_parquet(price_path).sort_index().loc[:end_date]
    spot = pd.read_parquet(spot_path).sort_index().loc[:end_date]
    if refresh_db:
        spot = refresh_ctd_spots_from_db(spot).loc[:end_date]
    spot = add_fx_converted_energy_spots(spot)
    current = add_current_phycarry(prices, spot, current_spots)
    ctd = add_priority_ctd_phycarry(prices, spot, products=assets)
    for asset in assets:
        column = f"{asset}_phycarry"
        common = current[column].notna() & ctd[column].notna()
        current.loc[~common, column] = np.nan
        ctd.loc[~common, column] = np.nan
    returns = execution_returns(prices, assets)

    rows = {}
    for recipe, (param_rng, post_func) in RECIPES.items():
        rows[(recipe, "current")] = backtest(
            current, returns, param_rng, post_func, cutoff, assets
        )
        rows[(recipe, "ctd")] = backtest(
            ctd, returns, param_rng, post_func, cutoff, assets
        )
    result = pd.DataFrame(rows).T
    result.index.names = ["recipe", "spot_method"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=Path, required=True)
    parser.add_argument("--spot", type=Path, required=True)
    parser.add_argument("--cutoff", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-09-18")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh-db", action="store_true")
    parser.add_argument("--group", choices=["priority", "extended"], default="priority")
    args = parser.parse_args()
    assets = ASSETS if args.group == "priority" else EXTENDED_ASSETS
    current_spots = CURRENT_SPOTS if args.group == "priority" else EXTENDED_CURRENT_SPOTS
    result = run(
        args.price,
        args.spot,
        args.cutoff,
        args.end_date,
        refresh_db=args.refresh_db,
        assets=assets,
        current_spots=current_spots,
    )
    print(result.to_string(float_format=lambda value: f"{value:.4f}"))
    if args.output:
        result.to_csv(args.output)


if __name__ == "__main__":
    main()
