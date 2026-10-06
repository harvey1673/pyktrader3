"""Research implementation of CTD spot-price normalization.

This module intentionally lives under ``tests`` until the rule history and spot
metadata have been reviewed for production use.  It follows the sign convention
used by :mod:`pycmqlib3.utility.exch_ctd_func`::

    futures_equivalent = delivered_cash_price - exchange_adjustment

Positive exchange premiums therefore lower the futures-equivalent price, while
exchange discounts increase it.  Freight, tax, FX, storage and other cash-basis
conversion costs are added before exchange adjustments are subtracted.

The product functions accept candidate metadata instead of assuming that a
column name proves a brand, grade or delivery location is eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd


Number = int | float


class UnsupportedRulePeriod(ValueError):
    """Raised when a historical rule has not been transcribed safely."""


@dataclass(frozen=True)
class CTDCandidate:
    """One deliverable spot-price candidate.

    ``cash_cost`` converts the quoted spot price to the delivery basis.  It may
    include freight, loading, tax, FX, storage and finance.  ``*_adj_override``
    are useful for exchange schedules that are dynamic or not yet transcribed.
    """

    name: str
    price_col: str
    spec: Mapping[str, Any] = field(default_factory=dict)
    location: str | None = None
    brand: str | None = None
    cash_cost: Number | pd.Series = 0.0
    quality_adj_override: Number | None = None
    location_adj_override: Number | None = None
    brand_adj_override: Number | None = None
    eligible: bool = True


@dataclass(frozen=True)
class RuleResult:
    rule: str
    quality_adj: float = 0.0
    location_adj: float = 0.0
    brand_adj: float = 0.0
    weight_multiplier: float = 1.0
    eligible: bool = True
    note: str = ""

    @property
    def total_exchange_adj(self) -> float:
        return self.quality_adj + self.location_adj + self.brand_adj


def _month(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        return value if value >= 100000 else 200000 + value
    ts = pd.Timestamp(value)
    return ts.year * 100 + ts.month


def _step(value: float, base: float, increment: float) -> float:
    """Number of exchange increments, robust to binary floating point."""

    return round((value - base) / increment, 8)


def _normalise_token(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _override(candidate: CTDCandidate, result: RuleResult) -> RuleResult:
    return RuleResult(
        rule=result.rule,
        quality_adj=(result.quality_adj if candidate.quality_adj_override is None
                     else float(candidate.quality_adj_override)),
        location_adj=(result.location_adj if candidate.location_adj_override is None
                      else float(candidate.location_adj_override)),
        brand_adj=(result.brand_adj if candidate.brand_adj_override is None
                   else float(candidate.brand_adj_override)),
        weight_multiplier=result.weight_multiplier,
        eligible=result.eligible and candidate.eligible,
        note=result.note,
    )


# ---------------------------------------------------------------------------
# DCE coke and coking coal
# ---------------------------------------------------------------------------

_J_LOCATION_ADJ = {
    "port": 0.0,
    "rizhao": 0.0,
    "rizhao_port": 0.0,
    "qingdao": 0.0,
    "lianyungang": 0.0,
    "tianjin": 0.0,
    "tianjin_port": 0.0,
    "tangshan_port": 0.0,
    "shanxi": -170.0,
    "shanxi_factory": -170.0,
    "hebei_factory": -100.0,
}


def _j_quality_2021(spec: Mapping[str, Any], include_mf: bool) -> tuple[float, bool]:
    required = ("ash", "sulfur", "m40", "m10", "cri", "csr")
    if not all(key in spec for key in required):
        return 0.0, False

    ash = float(spec["ash"])
    sulfur = float(spec["sulfur"])
    m40 = float(spec["m40"])
    m10 = float(spec["m10"])
    cri = float(spec["cri"])
    csr = float(spec["csr"])
    size_25_40 = float(spec.get("size_25_40", 32.0))

    eligible = (
        ash <= 13.5 and sulfur <= 0.75 and m40 >= 78.0 and m10 <= 8.5
        and cri <= 32.0 and csr >= 58.0
    )
    if not eligible:
        return 0.0, False

    ash_for_premium = max(ash, 12.5)
    if ash_for_premium < 13.0:
        ash_adj = _step(13.0, ash_for_premium, 0.1) * 3.0
    elif ash_for_premium > 13.0:
        ash_adj = -_step(ash_for_premium, 13.0, 0.1) * 5.0
    else:
        ash_adj = 0.0

    sulfur_for_premium = max(sulfur, 0.65)
    if sulfur_for_premium < 0.70:
        sulfur_adj = _step(0.70, sulfur_for_premium, 0.01) * 3.0
    elif sulfur_for_premium > 0.70:
        sulfur_adj = -_step(sulfur_for_premium, 0.70, 0.01) * 5.0
    else:
        sulfur_adj = 0.0

    if csr >= 65.0 and cri <= 25.0:
        reaction_adj = 50.0
    elif (58.0 <= csr < 60.0) or (30.0 < cri <= 32.0):
        reaction_adj = -40.0
    else:
        reaction_adj = 0.0

    strength_adj = -30.0 if (78.0 <= m40 < 80.0 or 7.5 < m10 <= 8.5) else 0.0
    size_adj = -15.0 * max(size_25_40 - 32.0, 0.0)
    mf = spec.get("equilibrium_moisture")
    mf_adj = -110.0 if include_mf and (mf is None or float(mf) > 1.0) else 0.0
    return ash_adj + sulfur_adj + reaction_adj + strength_adj + size_adj + mf_adj, True


def j_adjustment(candidate: CTDCandidate, contract_month: int) -> RuleResult:
    """DCE J adjustment, with an explicit sparse proxy before J2201."""

    cm = _month(contract_month)
    location = _normalise_token(candidate.location)
    location_known = location in _J_LOCATION_ADJ
    moisture = float(candidate.spec.get("moisture", 0.0))
    if cm < 202201:
        # The selected series is already a quoted quasi-grade-1 benchmark. Keep
        # the material old-rule excess-moisture effect without inventing minor
        # assay premiums that are not available in the quote metadata.
        excess_moisture = max(moisture - 5.0, 0.0)
        result = RuleResult(
            rule="DCE J legacy representative-quote proxy",
            location_adj=_J_LOCATION_ADJ.get(location, 0.0),
            weight_multiplier=1.0 / (1.0 - excess_moisture / 100.0),
            eligible=location_known,
            note="Research proxy: legacy minor quality adjustments are not reconstructed.",
        )
        return _override(candidate, result)
    include_mf = cm >= 202604
    quality_adj, eligible = _j_quality_2021(candidate.spec, include_mf=include_mf)
    result = RuleResult(
        rule="F/DCE J003-2024" if include_mf else "F/DCE J001-2021",
        quality_adj=quality_adj,
        location_adj=_J_LOCATION_ADJ.get(location, 0.0),
        weight_multiplier=1.0 / (1.0 - moisture / 100.0) if moisture < 100.0 else np.nan,
        eligible=eligible and location_known,
        note="Full moisture is weight-deducted; location token must be reviewed.",
    )
    return _override(candidate, result)


_JM_PORTS = {"tangshan", "jingtang", "qingdao", "rizhao", "lianyungang", "tianjin"}
_JM_BRAND_ADJ = {
    "pingmei_main_coking_no_1": 225.0,
    "shanjiao_rizhao_no_1": 80.0,
    "xiangjiao_no_1": 100.0,
    "kaijia_no_1": 175.0,
}


def _jm_sulfur_legacy(sulfur: float) -> float:
    if sulfur < 0.50:
        return 10.0
    if sulfur < 0.70:
        return _step(0.70, sulfur, 0.01) * 0.5
    if sulfur <= 1.00:
        return -_step(sulfur, 0.70, 0.01) * 1.5
    if sulfur <= 1.30:
        return -45.0 - _step(sulfur, 1.00, 0.01) * 2.5
    return -120.0 - _step(sulfur, 1.30, 0.01) * 5.0


def _jm_quality(spec: Mapping[str, Any], cm: int) -> tuple[float, bool, str]:
    required = ("ash", "sulfur", "volatile", "g", "y", "csr")
    if not all(key in spec for key in required):
        return 0.0, False, "missing required assay"
    ash = float(spec["ash"])
    sulfur = float(spec["sulfur"])
    volatile = float(spec["volatile"])
    g = float(spec["g"])
    y = float(spec["y"])
    csr = float(spec["csr"])

    if cm < 201907:
        eligible = ash <= 10.5 and sulfur <= 1.6 and 16 <= volatile <= 28 and g > 65 and csr >= 50
        if ash < 9.0:
            ash_adj = 20.0
        elif ash < 10.0:
            ash_adj = _step(10.0, ash, 0.1) * 2.0
        elif ash > 10.0:
            ash_adj = -_step(ash, 10.0, 0.1) * 4.0
        else:
            ash_adj = 0.0
        return ash_adj + _jm_sulfur_legacy(sulfur), eligible, "DCE JM 2013 sparse proxy"

    if cm < 202304:
        eligible = ash <= 10.5 and sulfur <= 1.6 and 16 <= volatile <= 28 and g > 65 and y <= 25 and csr >= 55
        if ash < 9.0:
            ash_adj = 20.0
        elif ash < 10.0:
            ash_adj = _step(10.0, ash, 0.1) * 2.0
        elif ash > 10.0:
            ash_adj = -_step(ash, 10.0, 0.1) * 4.0
        else:
            ash_adj = 0.0
        sulfur_adj = _jm_sulfur_legacy(sulfur)
        csr_adj = -100.0 if 55 <= csr < 60 else 0.0
        return ash_adj + sulfur_adj + csr_adj, eligible, "F/DCE JM001-2018"

    eligible = ash <= 11.0 and sulfur <= 1.6 and 16 <= volatile <= 28 and g > 65 and y >= 10 and csr >= 60
    ash_adj = 30.0 if ash <= 10.0 else (-30.0 if ash > 10.5 else 0.0)
    if cm >= 202701:
        sulfur_adj = (_step(1.30, max(sulfur, 0.70), 0.01) * 1.5
                       if sulfur < 1.30 else -_step(sulfur, 1.30, 0.01) * 2.5)
        csr_adj = -50.0 if 60 <= csr < 65 else 0.0
        rule = "F/DCE JM004-2025"
    else:
        sulfur_adj = (_step(1.30, max(sulfur, 0.70), 0.01) * 2.5
                       if sulfur < 1.30 else -_step(sulfur, 1.30, 0.01) * 5.0)
        csr_adj = 80.0 if csr >= 65 else 0.0
        rule = "F/DCE JM003-2022"
    volatile_adj = -50.0 if volatile > 26 else 0.0
    return ash_adj + sulfur_adj + csr_adj + volatile_adj, eligible, rule


def jm_adjustment(candidate: CTDCandidate, contract_month: int) -> RuleResult:
    cm = _month(contract_month)
    quality_adj, eligible, rule = _jm_quality(candidate.spec, cm)
    location = _normalise_token(candidate.location)
    if cm < 202304:
        location_adj = 0.0
        location_known = bool(location)
    elif location in _JM_PORTS:
        location_adj = 140.0 if cm >= 202701 and location in {"tangshan", "tianjin"} else 170.0
        location_known = True
    elif location == "shanxi":
        location_adj = 0.0
        location_known = True
    elif cm >= 202406 and location in {"xingtai", "handan"}:
        location_adj = 70.0
        location_known = True
    else:
        location_adj = 0.0
        location_known = False

    brand = _normalise_token(candidate.brand)
    brand_adj = _JM_BRAND_ADJ.get(brand, 0.0) if cm >= 202601 else 0.0
    moisture = float(candidate.spec.get("moisture", 8.0))
    if moisture <= 8.0:
        weight_multiplier = 1.0
    elif cm >= 202304:
        weight_multiplier = 0.92 / (1.0 - moisture / 100.0)
    else:
        weight_multiplier = 1.0 / (1.0 - round(moisture - 8.0) / 100.0)
    result = RuleResult(
        rule=rule + (" + JM2601 brand overlay" if cm >= 202601 else ""),
        quality_adj=quality_adj,
        location_adj=location_adj,
        brand_adj=brand_adj,
        weight_multiplier=weight_multiplier,
        eligible=eligible and location_known,
        note="Moisture above 8% is weight-adjusted; brand premium is an overlay.",
    )
    return _override(candidate, result)


# ---------------------------------------------------------------------------
# SHFE stainless steel and CZCE ferroalloys
# ---------------------------------------------------------------------------

def ss_adjustment(candidate: CTDCandidate, contract_month: int, rule_date: Any | None = None) -> RuleResult:
    spec = candidate.spec
    registered = bool(spec.get("registered", False))
    grade = _normalise_token(str(spec.get("grade", "")))
    surface = _normalise_token(str(spec.get("surface", "2b")))
    thickness = float(spec.get("thickness_mm", np.nan))
    edge = _normalise_token(str(spec.get("edge", "")))
    width = float(spec.get("width_mm", np.nan))
    allowed_thickness = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0}
    eligible = registered and grade in {"304", "sus304", "06cr19ni10"} and surface == "2b"
    eligible = eligible and thickness in allowed_thickness and width in {1000.0, 1219.0, 1500.0}
    eligible = eligible and edge in {"trimmed", "mill"}

    effective_date = pd.Timestamp(rule_date) if rule_date is not None else pd.Timestamp(contract_month)
    eligible = eligible and effective_date >= pd.Timestamp("2019-09-25")
    if thickness <= 0.6:
        thickness_adj = 400.0
    elif thickness == 0.7:
        thickness_adj = 300.0 if effective_date >= pd.Timestamp("2026-07-20") else 400.0
    elif thickness <= 1.0:
        thickness_adj = 200.0
    else:
        thickness_adj = 0.0
    edge_adj = -170.0 if edge == "mill" else 0.0
    result = RuleResult(
        rule="SHFE SS schedule effective 2026-07-20" if effective_date >= pd.Timestamp("2026-07-20") else "SHFE SS launch schedule",
        quality_adj=thickness_adj + edge_adj,
        eligible=eligible,
        note="Registered-brand eligibility is mandatory; brand premium is zero.",
    )
    return _override(candidate, result)


_SM_PRE_2411 = {"tianjin": -150, "handan": -120, "anyang": -120, "jiangsu": 0, "hubei": -50}
_SM_2411 = {"tianjin": -190, "handan": -120, "anyang": -100, "jiangsu": 0, "hubei": -30,
             "shizuishan": -360, "ulanqab": -400, "qinzhou": -350, "rizhao": -90, "yingkou": -120}
_SF_PRE_2411 = {"tianjin": 0, "handan": 0, "anyang": 0, "jiangsu": 140, "hubei": 100}
_SF_2411 = {"tianjin": 0, "handan": 0, "anyang": 10, "jiangsu": 110, "hubei": 80,
             "zhongwei": -350, "rizhao": 50, "yingkou": 80}


def _ferroalloy_adjustment(product: str, candidate: CTDCandidate, contract_month: int) -> RuleResult:
    cm = _month(contract_month)
    location = _normalise_token(candidate.location)
    if product == "SM":
        schedule = ({"tianjin": 0.0} if cm < 201911 else
                    dict(_SM_2411 if cm >= 202411 else _SM_PRE_2411))
        if 202309 <= cm < 202411:
            schedule.update({"shizuishan": -360, "ulanqab": -400, "qinzhou": -350})
        if cm >= 202711:
            schedule.update({"qinzhou": -250, "ulanqab": -290, "shizuishan": -340})
        expected_grade = {"femn65si17", "6517"}
        rule = "CZCE SM location schedule"
    else:
        schedule = ({"tianjin": 0.0} if cm < 201907 else
                    dict(_SF_2411 if cm >= 202411 else _SF_PRE_2411))
        if 202309 <= cm < 202411:
            schedule["zhongwei"] = -350
        if cm >= 202707:
            schedule["zhongwei"] = -280
        expected_grade = {"pg_fesi72al2.5", "fesi72al2.5", "72"}
        rule = "CZCE SF location schedule"
    grade = _normalise_token(str(candidate.spec.get("grade", "")))
    result = RuleResult(
        rule=rule,
        location_adj=float(schedule.get(location, 0.0)),
        eligible=grade in expected_grade and location in schedule,
        note="No ordinary alternative-grade premium; freight remains a cash-basis cost.",
    )
    return _override(candidate, result)


def sm_adjustment(candidate: CTDCandidate, contract_month: int) -> RuleResult:
    return _ferroalloy_adjustment("SM", candidate, contract_month)


def sf_adjustment(candidate: CTDCandidate, contract_month: int) -> RuleResult:
    return _ferroalloy_adjustment("SF", candidate, contract_month)


# ---------------------------------------------------------------------------
# Petrochemicals and rubber.  Zero is explicit where eligibility, rather than
# a fixed exchange premium, is the principal rule.  Dynamic schedules must be
# supplied through overrides and cash-cost fields.
# ---------------------------------------------------------------------------

_REGISTERED_BRAND_PRODUCTS = {"l", "pp", "v", "TA", "bu", "ru", "nr", "br", "pg"}
_STANDARD_ONLY_PRODUCTS = {"eg", "eb", "PX", "MA", "sc", "fu", "lu"}


def simple_adjustment(product: str, candidate: CTDCandidate, contract_month: int) -> RuleResult:
    cm = _month(contract_month)
    spec = candidate.spec
    quality = _normalise_token(str(spec.get("quality", "standard")))
    registered = bool(spec.get("registered", product not in _REGISTERED_BRAND_PRODUCTS))
    quality_adj = 0.0
    eligible = candidate.eligible and registered
    note = "No fixed premium encoded; use cash_cost/overrides for delivery-basis conversion."

    if product == "l":
        quality_adj = -20.0 if quality in {"qualified", "substitute"} else 0.0
        eligible = eligible and quality in {"standard", "premium", "qualified", "substitute"}
        note = "Registered brand required from L2104; qualified substitute is -20 yuan/t."
    elif product in {"pp", "v", "TA", "bu", "br", "pg"}:
        note = "Registered brand/producer eligibility required; no universal fixed brand premium encoded."
    elif product in {"eg", "eb", "PX", "MA"}:
        note = "Standard-grade eligibility; location/factory-pickup schedule belongs in overrides."
    elif product in {"sc", "fu", "lu"}:
        note = "Grade plus tax/FX/import or bunker conversion must be supplied in overrides/cash_cost."
    elif product == "UR":
        quality_adj = -20.0 if quality in {"qualified", "small_qualified"} else 0.0
        northeast = _normalise_token(candidate.location) in {"heilongjiang", "jilin", "liaoning", "northeast"}
        if cm >= 202703 and quality == "large_granule" and northeast:
            quality_adj = 20.0
        eligible = quality in {"standard", "premium", "qualified", "small_qualified", "large_granule"}
        note = "UR2703 adds +20 large granule only in specified northeast delivery; location group remains dynamic."
    elif product == "ru":
        location_adj = {"yunnan": -480.0, "hainan": -210.0}.get(_normalise_token(candidate.location), 0.0)
        result = RuleResult(
            rule="SHFE RU registered SCR WF schedule",
            location_adj=location_adj,
            eligible=registered and _normalise_token(str(spec.get("grade", ""))) in {"scr_wf", "scrwf"},
            note="Other approved regions use zero unless an explicit override is supplied.",
        )
        return _override(candidate, result)
    elif product == "nr":
        if quality == "tsr10":
            quality_adj = -400.0
        eligible = registered and quality in {"tsr20", "tsr10"}
        note = "TSR10 -400 is encoded for the reviewed 2026 list; origin/brand eligibility is still mandatory."

    result = RuleResult(
        rule=f"{product} research rule",
        quality_adj=quality_adj,
        eligible=eligible,
        note=note,
    )
    return _override(candidate, result)


AdjustmentFunction = Callable[[CTDCandidate, int], RuleResult]

_ADJUSTERS: dict[str, AdjustmentFunction] = {
    "j": j_adjustment,
    "jm": jm_adjustment,
    "ss": ss_adjustment,
    "SM": sm_adjustment,
    "SF": sf_adjustment,
}


def _get_adjuster(product: str) -> AdjustmentFunction:
    if product in _ADJUSTERS:
        return _ADJUSTERS[product]
    supported_simple = {"l", "pp", "v", "eg", "eb", "TA", "PX", "MA", "sc", "fu", "lu", "bu", "UR", "ru", "nr", "br", "pg"}
    if product in supported_simple:
        return lambda candidate, cm: simple_adjustment(product, candidate, cm)
    raise KeyError(f"unsupported CTD product: {product}")


def adjusted_candidates(
    product: str,
    spot_df: pd.DataFrame,
    expiry: pd.Series,
    candidates: Iterable[CTDCandidate],
) -> pd.DataFrame:
    """Return an auditable normalized value for every candidate and date."""

    expiry = pd.to_datetime(expiry.dropna())
    index = expiry.index.intersection(spot_df.index)
    adjuster = _get_adjuster(product)
    output: dict[tuple[str, str], pd.Series] = {}

    for candidate in candidates:
        if candidate.price_col not in spot_df.columns:
            continue
        raw = spot_df[candidate.price_col].reindex(index).astype(float)
        cash_cost = (candidate.cash_cost.reindex(index).astype(float)
                     if isinstance(candidate.cash_cost, pd.Series)
                     else pd.Series(float(candidate.cash_cost), index=index))
        if product == "ss":
            rules = [ss_adjustment(candidate, _month(expiry.loc[date]), rule_date=date) for date in index]
        else:
            rules = [adjuster(candidate, _month(expiry.loc[date])) for date in index]
        quality = pd.Series([rule.quality_adj for rule in rules], index=index)
        location = pd.Series([rule.location_adj for rule in rules], index=index)
        brand = pd.Series([rule.brand_adj for rule in rules], index=index)
        multiplier = pd.Series([rule.weight_multiplier for rule in rules], index=index)
        eligible = pd.Series([rule.eligible for rule in rules], index=index)
        equivalent = (raw * multiplier + cash_cost - quality - location - brand).where(eligible)

        output[(candidate.name, "raw_price")] = raw
        output[(candidate.name, "cash_cost")] = cash_cost
        output[(candidate.name, "quality_adj")] = quality
        output[(candidate.name, "location_adj")] = location
        output[(candidate.name, "brand_adj")] = brand
        output[(candidate.name, "weight_multiplier")] = multiplier
        output[(candidate.name, "eligible")] = eligible
        output[(candidate.name, "equivalent")] = equivalent

    result = pd.DataFrame(output, index=index)
    if len(result.columns):
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=["candidate", "field"])
    return result


def ctd_basis(
    product: str,
    spot_df: pd.DataFrame,
    expiry: pd.Series,
    candidates: Iterable[CTDCandidate] | None = None,
    return_details: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    default_priority = None
    if candidates is None:
        default_priority = {
            "j": ["rizhao_quasi_1_outstock", "rizhao_quasi_1", "tianjin_early_history"],
            "SF": ["tianjin_72", "national_72_proxy"],
            "fu": ["zhoushan_380cst_cny_proxy", "singapore_380cst_fob_cny_proxy"],
            "lu": ["zhoushan_bonded_05_cny_proxy", "zhoushan_05_cny_proxy"],
        }.get(product)
    candidates = list(candidates if candidates is not None else priority_ctd_candidates(product))
    details = adjusted_candidates(product, spot_df, expiry, candidates)
    if details.empty:
        ctd = pd.Series(np.nan, index=pd.to_datetime(expiry.dropna()).index, name=f"{product}_ctd")
    else:
        equivalents = details.xs("equivalent", axis=1, level="field")
        if default_priority:
            ctd = pd.Series(np.nan, index=equivalents.index, dtype=float)
            for candidate_name in default_priority:
                candidate = equivalents.get(
                    candidate_name,
                    pd.Series(index=equivalents.index, dtype=float),
                )
                ctd = ctd.combine_first(candidate)
        else:
            ctd = equivalents.min(axis=1, skipna=True).where(equivalents.notna().any(axis=1))
        ctd.name = f"{product}_ctd"
    return (ctd, details) if return_details else ctd


def _wrapper(product: str):
    def wrapped(spot_df, expiry, candidates=None, return_details=False):
        return ctd_basis(product, spot_df, expiry, candidates, return_details=return_details)
    wrapped.__name__ = f"{product}_ctd_basis"
    return wrapped


def priority_ctd_candidates(product: str) -> list[CTDCandidate]:
    """Small default basket for research; no production mappings are changed."""

    defaults = {
        "j": [
            CTDCandidate(
                "rizhao_quasi_1_outstock", "coke_sub_a_rz_outstock", location="rizhao",
                spec={"ash": 13, "sulfur": .7, "m40": 80, "m10": 7.5,
                      "cri": 30, "csr": 60, "moisture": 7,
                      "equilibrium_moisture": None},
            ),
            CTDCandidate(
                "rizhao_quasi_1", "coke_sub_a_rz", location="rizhao",
                spec={"ash": 13, "sulfur": .7, "m40": 80, "m10": 7.5,
                      "cri": 30, "csr": 60, "moisture": 7,
                      "equilibrium_moisture": None},
            ),
            CTDCandidate(
                "tianjin_early_history", "coke_sub_a_tj", location="tianjin",
                spec={"ash": 12.5, "sulfur": .7, "m40": 80, "m10": 7.5,
                      "cri": 30, "csr": 62, "moisture": 7,
                      "equilibrium_moisture": None},
            ),
        ],
        "jm": [
            CTDCandidate(
                "xiaoyi_domestic", "ckc_a10v24s08_lvliang", location="shanxi",
                spec={"ash": 10, "sulfur": .8, "volatile": 24, "g": 75,
                      "y": 24, "csr": 63, "moisture": 8},
            ),
            CTDCandidate(
                "tangshan_mongol_5_proxy", "ckc_mongol5_ts", location="tangshan",
                spec={"ash": 10.5, "sulfur": .75, "volatile": 28, "g": 78,
                      "y": 14, "csr": 60, "moisture": 8,
                      "assumed_fields": ("y",)},
            ),
            CTDCandidate(
                "jiexiu_kaijia", "ckc_midsulfur_jiexiu_kaijia", location="shanxi",
                brand="kaijia_no_1",
                spec={"ash": 10.5, "sulfur": 1.3, "volatile": 25, "g": 80,
                      "y": 14, "csr": 65, "moisture": 8},
            ),
        ],
        "ss": [
            CTDCandidate(
                "wuxi_304_2b_mill", "ss_304_gross_wuxi", location="wuxi",
                spec={"registered": True, "grade": "304", "surface": "2B",
                      "thickness_mm": 2.0, "width_mm": 1219, "edge": "mill"},
            ),
            CTDCandidate(
                "hongwang_2x1240_market_proxy", "ss_304_2b_hongwang_wuxi", location="wuxi",
                brand="hongwang",
                spec={"registered": True, "grade": "304", "surface": "2B",
                      "thickness_mm": 2.0, "width_mm": 1240, "edge": "trimmed"},
            ),
        ],
        "SM": [
            CTDCandidate("tianjin_6517", "SM_65s17_tj", location="tianjin",
                         spec={"grade": "6517"}),
        ],
        "SF": [
            CTDCandidate("tianjin_72", "SF_72_tj", location="tianjin",
                         spec={"grade": "72"}),
            CTDCandidate("national_72_proxy", "SF_72_shmet", location="tianjin",
                         spec={"grade": "72"}),
        ],
        "l": [
            CTDCandidate("tianjin_7042_proxy", "l_7042_tj", location="tianjin",
                         spec={"registered": True, "quality": "standard"}),
        ],
        "pp": [
            CTDCandidate("t30s_market_proxy", "pp_t30s_shaoxing_hz", location="hangzhou",
                         spec={"registered": True, "quality": "standard"}),
        ],
        "v": [
            CTDCandidate("shanghai_carbide_sg5_proxy", "pvc_cac2_sh", location="shanghai",
                         spec={"registered": True, "quality": "standard"}),
            CTDCandidate("east_china_carbide_sg5_proxy", "pvc_cac2_east", location="east_china",
                         spec={"registered": True, "quality": "standard"}),
        ],
        "eg": [
            CTDCandidate("east_china_tank", "eg_east_spot", location="east_china",
                         spec={"quality": "standard"}),
        ],
        "eb": [
            CTDCandidate("east_china_spot", "eb_east_spot", location="east_china",
                         spec={"quality": "standard"}),
        ],
        "TA": [
            CTDCandidate("east_china_registered_proxy", "TA_east_spot", location="east_china",
                         spec={"registered": True, "quality": "standard"}),
        ],
        "PX": [
            CTDCandidate("east_china_ex_factory", "PX_exw_east_spot", location="east_china",
                         spec={"quality": "standard"}),
        ],
        "MA": [
            CTDCandidate("jiangsu_spot", "MA_spot_jiangsu", location="jiangsu",
                         spec={"quality": "standard"}),
        ],
        "UR": [
            CTDCandidate("shandong_small_granule", "UR_shandong_spot", location="shandong",
                         spec={"quality": "standard"}),
        ],
        "ru": [
            CTDCandidate("jiangsu_scr_wf", "ru_scrwf_jiangsu", location="jiangsu",
                         spec={"registered": True, "grade": "scr_wf"}),
        ],
        "bu": [
            CTDCandidate("shandong_heavy_asphalt_proxy", "bu_heavy_shandong", location="shandong",
                         spec={"registered": True, "quality": "standard"}),
        ],
        "pg": [
            CTDCandidate("south_china_import_propane_proxy", "propane_cfr_south_cny_vat", location="south_china",
                         spec={"registered": True, "quality": "standard"}),
        ],
        "br": [
            CTDCandidate("qilu_br9000_shandong", "br9000_qilu_sd", location="shandong",
                         brand="sinopec", spec={"registered": True, "quality": "standard", "grade": "br9000"}),
            CTDCandidate("daqing_br9000_shandong", "br9000_daqing_sd", location="shandong",
                         brand="kunlun", spec={"registered": True, "quality": "standard", "grade": "br9000"}),
        ],
        "fu": [
            CTDCandidate("zhoushan_380cst_cny_proxy", "fo_380cst_zhoushan_cny", location="zhoushan",
                         spec={"quality": "standard", "grade": "rmg380"}),
            CTDCandidate("singapore_380cst_fob_cny_proxy", "fo_380cst_sgp_fob_cny", location="singapore",
                         spec={"quality": "standard", "grade": "rmg380"}),
        ],
        "lu": [
            CTDCandidate("zhoushan_bonded_05_cny_proxy", "lu_bonded_zhoushan_cny", location="zhoushan",
                         spec={"quality": "standard", "sulfur_pct": 0.5}),
            CTDCandidate("zhoushan_05_cny_proxy", "lu_05_zhoushan_cny", location="zhoushan",
                         spec={"quality": "standard", "sulfur_pct": 0.5}),
        ],
    }
    if product not in defaults:
        raise ValueError(f"no default sparse basket for {product}; pass candidates explicitly")
    return defaults[product]


j_ctd_basis = _wrapper("j")
jm_ctd_basis = _wrapper("jm")
ss_ctd_basis = _wrapper("ss")
SM_ctd_basis = _wrapper("SM")
SF_ctd_basis = _wrapper("SF")
l_ctd_basis = _wrapper("l")
pp_ctd_basis = _wrapper("pp")
v_ctd_basis = _wrapper("v")
eg_ctd_basis = _wrapper("eg")
eb_ctd_basis = _wrapper("eb")
TA_ctd_basis = _wrapper("TA")
PX_ctd_basis = _wrapper("PX")
MA_ctd_basis = _wrapper("MA")
sc_ctd_basis = _wrapper("sc")
fu_ctd_basis = _wrapper("fu")
lu_ctd_basis = _wrapper("lu")
bu_ctd_basis = _wrapper("bu")
UR_ctd_basis = _wrapper("UR")
ru_ctd_basis = _wrapper("ru")
nr_ctd_basis = _wrapper("nr")
br_ctd_basis = _wrapper("br")
pg_ctd_basis = _wrapper("pg")

# Lower-case convenience aliases for callers that use product keys rather than
# exchange display codes.
sm_ctd_basis = SM_ctd_basis
sf_ctd_basis = SF_ctd_basis
ta_ctd_basis = TA_ctd_basis
px_ctd_basis = PX_ctd_basis
ma_ctd_basis = MA_ctd_basis
ur_ctd_basis = UR_ctd_basis
pvc_ctd_basis = v_ctd_basis


RULE_COVERAGE = {
    "automatic": ["j:J2201+", "jm", "ss", "SM", "SF", "l", "UR", "ru", "nr"],
    "eligibility_plus_overrides": ["pp", "v", "eg", "eb", "TA", "PX", "MA", "sc", "fu", "lu", "bu", "br", "pg"],
    "known_gap": ["j:pre-J2201", "dynamic factory-pickup guidance", "historical registered-brand lists"],
}


RULE_SOURCES = {
    "j": [
        "https://www.glqh.com/u/cms/www/202507/30110346mtf5.pdf",
        "https://www.cjfco.com.cn/ueditor/jsp/upload/file/20250328/1743143292092043647.pdf",
    ],
    "jm": [
        "https://www.glqh.com/u/cms/www/202507/30110306fm3q.pdf",
        "https://www.yhqh.com.cn/upload/cn/file/2025-12/col16/1766397928916.pdf",
    ],
    "ss": [
        "https://www.shfe.com.cn/regulation/exchangerules/productrules/202512/t20251231_829965.html",
        "https://www.shfe.com.cn/products/futures/metal/ferrousandpreciousmetal/ss_f/attach/201909/t20190918_795033.html",
    ],
    "SM": [
        "https://www.czce.com.cn/cn/content_file/flfg/zcjywgz/pzxz/2026/1/6766f6603fcb423da02ea7ff54c6f706.pdf",
        "https://www.ghlsqh.com.cn/company/news/show-22559.html",
    ],
    "SF": [
        "https://www.czce.com.cn/cn/content_file/flfg/zcjywgz/pzxz/2026/1/2c614450dc974dfcbd4e3c08473cffd9.pdf",
        "https://www.btqh.com/index.php?a=show&c=index&catid=26&id=18075&m=content",
    ],
    "petchem_and_rubber_survey": [
        "https://www.cs.com.cn/zzqh2020/202004/t20200421_6048374.html",
        "https://www.shfe.cn/regulation/ineregulation/businessmethods/delivery/202606/t20260626_832297.html",
        "https://www.shfe.com.cn/products/futures/energyandchemical/nr_f/attach/202412/P020260601763317743470.pdf",
    ],
}
