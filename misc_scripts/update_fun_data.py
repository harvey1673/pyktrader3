"""Fundamental-data loading independent of signal definitions."""

import datetime

import numpy as np
import pandas as pd

from pycmqlib3.strategy.feature_config import METAL_INV_FEATURES
from pycmqlib3.utility.dbaccess import load_codes_from_edb, load_int_stock_daily
from pycmqlib3.utility.misc import CHN_Holidays, day_shift, nearby
from pycmqlib3.utility.spot_idx_map import index_map, mysteel_index_map, process_spot_df


PROD_FULL_HIST_TICKERS = list(METAL_INV_FEATURES.values())


def get_fun_data(start_date, run_date, full_hist_tickers=PROD_FULL_HIST_TICKERS):
    """Load and prepare iFind and MySteel fundamental-data features."""
    run_date = pd.to_datetime(run_date)
    e_date = day_shift(run_date.date(), "5b", CHN_Holidays)
    cdate_rng = pd.date_range(start=start_date, end=e_date, freq="D", name="date")

    # MySteel owns shared feature names, including dates where it has no value.
    mysteel_tickers = set(mysteel_index_map.values())
    ifind_index_map = {
        code: ticker for code, ticker in index_map.items()
        if ticker not in mysteel_tickers
    }
    source_frames = []
    for source, source_map in (("ifind", ifind_index_map), ("mysteel", mysteel_index_map)):
        rev_map = {value: key for key, value in source_map.items()}
        index_codes = [
            rev_map[ticker] for ticker in full_hist_tickers if ticker in rev_map
        ]
        if index_codes:
            spot_df = load_codes_from_edb(
                index_codes, source=source, column_name="index_code"
            )
            spot_df = spot_df.rename(columns=source_map)
            spot_df = spot_df.reindex(
                index=pd.date_range(
                    start=spot_df.index.min() if not spot_df.empty else start_date,
                    end=e_date, freq="D", name="date"
                )
            )
        else:
            spot_df = pd.DataFrame(index=cdate_rng)
        index_codes = [
            code for code, ticker in source_map.items()
            if ticker not in spot_df.columns
        ]
        if index_codes:
            data_df = load_codes_from_edb(
                index_codes,
                source=source,
                column_name="index_code",
                start_date=start_date,
            )
            data_df = data_df.rename(columns=source_map).dropna(how="all")
            data_df = data_df.reindex(index=cdate_rng)
            spot_df = pd.concat([spot_df, data_df], axis=1)
        source_frames.append(spot_df)
    spot_df = pd.concat(source_frames, axis=1)
    spot_df = process_spot_df(spot_df, adjust_time=True)

    stock_df = load_int_stock_daily(
        [
            "XOM.N",
            "BP.N",
            "CVX.N",
            "SU.N",
            "EOG.N",
            "APA.N",
            "COP.N",
            "VLO.N",
            "PSX.N",
            "MPC.N",
            "PBF.N",
            "SPY.P",
            "GDX.P",
            "USO.P",
            "GLD.P",
        ]
    )
    stock_pct_chg = (
        stock_df.loc[:, stock_df.columns.get_level_values(1) == "close"]
        .droplevel(level=[1], axis=1)
        .pct_change()
    )
    spot_dict = {
        "us_oil_prod_etf_perf": (
            1 + stock_pct_chg[["VLO.N", "PSX.N", "MPC.N", "PBF.N"]].mean(axis=1)
        ).cumprod()
    }
    for nb in [2, 3, 4]:
        fef_nb = nearby(
            "FEF",
            n=nb,
            start_date=max(start_date, datetime.date(2016, 7, 1)),
            end_date=run_date.date(),
            roll_rule="-2b",
            roll_col="settle",
            freq="d",
            shift_mode=2,
        )
        fef_nb.index = pd.to_datetime(fef_nb.index)
        fef_nb.loc[fef_nb["settle"] <= 0, "settle"] = np.nan
        fef_nb.loc[fef_nb["close"] <= 0, "close"] = np.nan
        fef_nb["fe_viu"] = spot_df["viu_fe"]
        spot_dict[f"FEFc{nb-1}"] = fef_nb["settle"]
        spot_dict[f"FEFc{nb-1}_close"] = fef_nb["close"]
        spot_dict[f"FEFc{nb-1}_shift"] = fef_nb["shift"]
        fef_nb["viu_fe"] = spot_df["viu_fe"].ffill()
        adj_flag = (fef_nb.index >= pd.Timestamp("2025-09-01")) & (
            fef_nb["contract"].apply(lambda contract: int(contract[-4:-2]) >= 26)
        )
        fef_nb["cont_adj"] = 0.0
        fef_nb.loc[adj_flag, "cont_adj"] = fef_nb["viu_fe"] * 1.4
        spot_dict[f"FEFc{nb-1}_pxadj"] = fef_nb["cont_adj"]

    spot_dict["FEF_c1_c2_ratio"] = (
        spot_dict["FEFc1"] / np.exp(spot_dict["FEFc1_shift"])
        + spot_dict["FEFc1_pxadj"]
    ) / (
        spot_dict["FEFc2"] / np.exp(spot_dict["FEFc2_shift"])
        + spot_dict["FEFc2_pxadj"]
    )
    spot_dict["FEF_c123fly_ratio"] = (
        (
            spot_dict["FEFc1"] / np.exp(spot_dict["FEFc1_shift"])
            + spot_dict["FEFc1_pxadj"]
        )
        * (
            spot_dict["FEFc3"] / np.exp(spot_dict["FEFc3_shift"])
            + spot_dict["FEFc3_pxadj"]
        )
        / (
            spot_dict["FEFc2"] / np.exp(spot_dict["FEFc2_shift"])
            + spot_dict["FEFc2_pxadj"]
        )
        ** 2
    )
    spot_dict["FEF_ryield"] = (
        np.log(
            spot_dict["FEFc1"] / np.exp(spot_dict["FEFc1_shift"])
            + spot_dict["FEFc1_pxadj"]
        )
        - np.log(
            spot_dict["FEFc2"] / np.exp(spot_dict["FEFc2_shift"])
            + spot_dict["FEFc2_pxadj"]
        )
    ) * 12
    spot_dict["FEF_basmom"] = np.log(
        1 + spot_dict["FEFc1"].dropna().pct_change()
    ) - np.log(1 + spot_dict["FEFc2"].dropna().pct_change())
    spot_dict["FEF_basmom10"] = spot_dict["FEF_basmom"].dropna().rolling(10).sum()
    spot_dict["FEF_basmom5"] = spot_dict["FEF_basmom"].dropna().rolling(5).sum()
    return pd.concat([spot_df, pd.DataFrame(spot_dict)], axis=1)
