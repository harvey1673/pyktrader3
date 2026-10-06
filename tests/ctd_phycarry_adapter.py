"""Test-side bridge from sparse CTD spots to historical physical carry.

The output column names match the existing notebook and historical signal
generator contract: ``<asset>_ctd_spot`` and ``<asset>_phycarry``.  CTD prices
are already normalized to the futures delivery basis, so this bridge does not
apply the legacy fixed SM/SF adders a second time.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from tests.ctd_adjustments import (
    MA_ctd_basis,
    PX_ctd_basis,
    SF_ctd_basis,
    SM_ctd_basis,
    TA_ctd_basis,
    UR_ctd_basis,
    bu_ctd_basis,
    br_ctd_basis,
    eb_ctd_basis,
    eg_ctd_basis,
    fu_ctd_basis,
    j_ctd_basis,
    jm_ctd_basis,
    l_ctd_basis,
    lu_ctd_basis,
    pg_ctd_basis,
    pp_ctd_basis,
    ru_ctd_basis,
    ss_ctd_basis,
    v_ctd_basis,
)


CTD_BUILDERS = {
    "j": j_ctd_basis,
    "jm": jm_ctd_basis,
    "ss": ss_ctd_basis,
    "SM": SM_ctd_basis,
    "SF": SF_ctd_basis,
    "l": l_ctd_basis,
    "pp": pp_ctd_basis,
    "v": v_ctd_basis,
    "eg": eg_ctd_basis,
    "eb": eb_ctd_basis,
    "TA": TA_ctd_basis,
    "PX": PX_ctd_basis,
    "MA": MA_ctd_basis,
    "UR": UR_ctd_basis,
    "ru": ru_ctd_basis,
    "bu": bu_ctd_basis,
    "pg": pg_ctd_basis,
    "br": br_ctd_basis,
    "fu": fu_ctd_basis,
    "lu": lu_ctd_basis,
}

CTD_PHYCARRY_MAP = {
    asset: f"{asset}_ctd_spot" for asset in CTD_BUILDERS
}


def add_fx_converted_energy_spots(spot_df: pd.DataFrame) -> pd.DataFrame:
    """Convert selected USD/tonne energy quotes to RMB/tonne proxies."""

    output = spot_df.copy().sort_index()
    offshore = output.get("usdcnh_spot", pd.Series(index=output.index, dtype=float))
    onshore = output.get("usdcny_spot", pd.Series(index=output.index, dtype=float))
    fx = pd.to_numeric(offshore, errors="coerce").combine_first(
        pd.to_numeric(onshore, errors="coerce")
    ).ffill()
    conversions = {
        "fo_380cst_zhoushan": "fo_380cst_zhoushan_cny",
        "fo_380cst_sgp_fob": "fo_380cst_sgp_fob_cny",
        "lu_bonded_zhoushan": "lu_bonded_zhoushan_cny",
        "lu_05_zhoushan": "lu_05_zhoushan_cny",
    }
    for source, target in conversions.items():
        if source in output.columns:
            output[target] = pd.to_numeric(output[source], errors="coerce") * fx
    if "propane_cfr_south" in output.columns:
        # PG is a domestic tax-inclusive contract.  This retains only the
        # major FX/VAT contribution; duty, port charges, freight and the
        # propane/butane composition basis remain explicit proxy gaps.
        output["propane_cfr_south_cny_vat"] = (
            pd.to_numeric(output["propane_cfr_south"], errors="coerce")
            * fx
            * 1.13
        )
    return output


def add_priority_ctd_spots(
    price_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    products: Iterable[str] = CTD_BUILDERS,
) -> pd.DataFrame:
    """Add sparse, rule-adjusted CTD spots on the futures observation dates."""

    prices = price_df.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    output = add_fx_converted_energy_spots(spot_df)
    output.index = pd.to_datetime(output.index)

    # Use only information known by each futures date.  The union preserves
    # spot observations that occur between futures trading dates before ffill.
    aligned_spot = (
        output.reindex(output.index.union(prices.index))
        .sort_index()
        .ffill()
        .reindex(prices.index)
    )
    derived: dict[str, pd.Series] = {}
    for product in products:
        if product not in CTD_BUILDERS:
            raise KeyError(f"unsupported priority CTD product: {product}")
        expiry_col = (f"{product}c1", "expiry")
        if expiry_col not in prices.columns:
            continue
        expiry = pd.to_datetime(prices[expiry_col], errors="coerce")
        ctd = CTD_BUILDERS[product](aligned_spot, expiry)
        derived[CTD_PHYCARRY_MAP[product]] = ctd.rename(
            CTD_PHYCARRY_MAP[product]
        )

    if not derived:
        return output
    output = output.drop(columns=list(derived), errors="ignore")
    return pd.concat([output, pd.DataFrame(derived)], axis=1).sort_index()


def add_priority_ctd_phycarry(
    price_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    products: Iterable[str] = CTD_BUILDERS,
    *,
    funding_col: str = "r007_cn",
) -> pd.DataFrame:
    """Add CTD spots and notebook-compatible annualized physical carry.

    This follows the current historical notebook formula, including shifted c1
    prices and the five-day EWM funding rate.  No fixed location adder is used
    because each CTD spot has already been converted to the delivery basis.
    Existing ``<asset>_phycarry`` columns are replaced only for requested CTD
    products; all other columns are preserved.
    """

    products = tuple(products)
    prices = price_df.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    output = add_priority_ctd_spots(prices, spot_df, products)

    derived: dict[str, pd.Series] = {}
    for product in products:
        spot_col = CTD_PHYCARRY_MAP[product]
        close_col = (f"{product}c1", "close")
        shift_col = (f"{product}c1", "shift")
        expiry_col = (f"{product}c1", "expiry")
        if spot_col not in output.columns or not all(
            column in prices.columns
            for column in (close_col, shift_col, expiry_col)
        ):
            continue

        c1 = (
            pd.to_numeric(prices[close_col], errors="coerce")
            / np.exp(pd.to_numeric(prices[shift_col], errors="coerce"))
        ).rename("c1")
        funding = output.get(
            funding_col,
            pd.Series(index=output.index, dtype=float),
        ).rename(funding_col)
        frame = pd.concat(
            [
                pd.to_numeric(output[spot_col], errors="coerce").rename("spot"),
                funding,
                c1,
                pd.to_datetime(prices[expiry_col], errors="coerce").rename("expiry"),
            ],
            axis=1,
        ).sort_index().ffill()
        days = (
            frame["expiry"] - pd.Series(frame.index, index=frame.index)
        ).dt.days.replace(0, np.nan)
        valid_spot = frame["spot"].where(frame["spot"] > 0)
        valid_c1 = frame["c1"].where(frame["c1"] > 0)
        funding_rate = frame[funding_col].ewm(5).mean() / 100.0
        derived[f"{product}_phycarry"] = (
            (np.log(valid_spot) - np.log(valid_c1)) / days * 365.0
            + funding_rate
        )

    if not derived:
        return output
    output = output.drop(columns=list(derived), errors="ignore")
    return pd.concat([output, pd.DataFrame(derived)], axis=1).sort_index()
