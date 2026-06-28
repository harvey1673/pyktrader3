"""Build signal-to-data dependency mapping for fundamental signals.

This script inspects:
- signal definitions in pycmqlib3.strategy.signal_repo
- routing in misc_scripts.fun_factor_update.factors_by_asset
- index alias mapping in pycmqlib3.utility.spot_idx_map.index_map

It produces a report that links each signal to required spot_df data fields and,
where possible, to upstream index codes in index_map.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc_scripts.fun_factor_update import (
    factors_by_asset,
    factors_by_beta_neutral,
    factors_by_spread,
    factors_by_spread2,
    single_factors,
    update_db_factor,
)
from pycmqlib3.strategy.signal_repo import (
    commod_phycarry_dict,
    feature_to_feature_key_mapping,
    signal_store,
)
from pycmqlib3.utility.dbaccess import load_codes_from_edb
from pycmqlib3.utility.email_tool import send_html_by_smtp
from pycmqlib3.utility.exch_ctd_func import io_brand_dict, io_ctd_basis, si_ctd_basis
from pycmqlib3.utility.sec_bits import EMAIL_NOTIFY, EMAIL_QQ, LOCAL_PC_NAME, NOTIFIERS
from pycmqlib3.utility.spot_idx_map import index_map, process_spot_df


IGNORED_PRICE_FEATURES: Set[str] = {
    "px",
    "metal_px",
    "ryield",
    "basmom",
    "basmom5",
    "basmom10",
    "basmom20",
    "basmom40",
    "basmom60",
    "basmom120",
    "logret",
    "colr",
}

TEMP_EXCLUDED_FEATURES: Set[str] = {
    "FEF_basmom5",
    "FEF_basmom",
    "FEF_c1_c2_ratio",
    "FEF_c123fly_ratio",
    "FEF_phycarry",
    "FEF_ryield",
}

# Markets used to build per-asset derived columns in update_db_factor.
MARKETS_IN_UPDATE_DB_FACTOR: List[str] = [
    "rb", "hc", "i", "j", "jm",
    "SM", "SF", "SA", "FG", "v", "SH",
    "cu", "al", "zn", "ni", "pb", "sn", "ss", "ao",
    "au", "ag",
    "si", "lc", "ps", "ec",
    "ru", "UR", "sp", "nr", "br",
    "l", "pp", "TA", "PX", "eg", "MA", "eb", "PF",
    "sc", "lu", "bu", "fu", "pg",
    "m", "RM", "y", "p", "OI", "a", "b", "c", "cs",
    "CJ", "CF", "jd", "AP", "lh", "SR", "PK",
    "T", "TF", "TL",
]

# Columns appended in get_fun_data via spot_dict (before update_db_factor data_dict).
GET_FUN_DATA_DERIVED_COLUMNS: Set[str] = {
    "us_oil_prod_etf_perf",
    "FEFc1", "FEFc1_close", "FEFc1_shift", "FEFc1_pxadj",
    "FEFc2", "FEFc2_close", "FEFc2_shift", "FEFc2_pxadj",
    "FEFc3", "FEFc3_close", "FEFc3_shift", "FEFc3_pxadj",
    "FEF_c1_c2_ratio", "FEF_c123fly_ratio", "FEF_ryield", "FEF_basmom",
    "FEF_basmom10", "FEF_basmom5",
    "io_ctd_spot",
}

# Derived in process_spot_df and consumed by multiple by-asset signals.
PROCESS_SPOT_DERIVED_COLUMNS: Set[str] = {
    "au_etf_holdings",
    "ag_etf_holdings",
}

# Direct formula dependencies for key process_spot_df derived columns.
PROCESS_SPOT_FORMULA_DEPS: Dict[str, List[str]] = {
    "cgb_3m_1y_spd": ["cn_govbond_yield_3m_sch", "cn_govbond_yield_1y_sch"],
    "cgb_2_5_spd": ["cn_govbond_yield_2y", "cn_govbond_yield_5y"],
    "cgb_1_2_spd": ["cn_govbond_yield_1y", "cn_govbond_yield_2y"],
    "cgb_1_5_spd": ["cn_govbond_yield_1y", "cn_govbond_yield_5y"],
    "cgb_2_10_spd": ["cn_govbond_yield_2y", "cn_govbond_yield_10y"],
    "shibor_3m_1y_spd": ["shibor_3m", "shibor_1y"],
    "usgg10_be": ["usgg10yr", "usggt10yr"],
    "usgg10_2_spd": ["usgg10yr", "usgg2yr"],
    "cnh_cny_spd1": ["usdcnh_xe", "usdcny_xe"],
    "cnh_cny_spd2": ["usdcnh_close", "usdcny_spot2"],
    "cny_mid_dev1": ["usdcny_spot", "usdcny_mid"],
    "cny_mid_dev2": ["usdcny_spot2", "usdcny_mid"],
    "r_dr_7d_spd": ["r007_cn", "dr007_cn"],
    "m1_m2_spd": ["m1_cn_yoy", "m2_cn_yoy"],
    "ppi_cpi_mom_spd": ["ppi_cn_mom", "cpi_cn_mom"],
    "auag_cme_warrant_ratio": ["au_cme_warrant_all", "ag_cme_warrant_all"],
    "au_etf_holdings": ["au_etf_spdr_holding"],
    "ag_etf_holdings": ["ag_etf_sivr_holding"],
}

EXCH_WARRANT_SOURCE_COLUMNS: Dict[str, List[str]] = {
    "cu": ["cu_inv_shfe_d"],
    "bc": ["bc_inv_ine_warrant"],
    "zn": ["zn_inv_shfe_d"],
    "al": ["al_inv_shfe_d"],
    "pb": ["pb_inv_shfe_d"],
    "ni": ["ni_inv_shfe_d"],
    "sn": ["sn_inv_shfe_d"],
    "ao": ["ao_inv_shfe_d", "ao_inv_shfe_mill_d"],
    "ss": ["ss_inv_shfe_d"],
    "si": ["si_inv_gfex_d"],
    "lc": ["lc_inv_gfex_d"],
    "SH": ["SH_inv_czce_warrant", "SH_inv_czce_unwarrant"],
    "TA": ["TA_inv_czce_warrant", "TA_inv_czce_unwarrant"],
    "PX": ["PX_inv_czce_warrant", "PX_inv_czce_unwarrant"],
    "MA": ["MA_inv_czce_warrant", "MA_inv_czce_unwarrant"],
    "UR": ["UR_inv_czce_warrant", "UR_inv_czce_unwarrant"],
    "PF": ["PF_inv_czce_warrant"],
    "l": ["l_inv_dce_warrant"],
    "pp": ["pp_inv_dce_warrant"],
    "v": ["v_inv_dce_warrant"],
    "eg": ["eg_inv_dce_warrant"],
    "eb": ["eb_inv_dce_warrant"],
    "pg": ["pg_inv_dce_warrant"],
    "bu": ["bu_inv_shfe_warrant", "bu_inv_shfe_mill"],
    "SA": ["SA_inv_czce_warrant", "SA_inv_czce_unwarrant"],
    "FG": ["FG_inv_czce_warrant"],
    "j": ["j_inv_dce_warrant"],
    "jm": ["jm_inv_dce_warrant"],
    "rb": ["rb_inv_shfe_warrant"],
    "hc": ["hc_inv_shfe_warrant"],
    "i": ["i_inv_dce_warrant"],
    "SF": ["SF_inv_czce_warrant", "SF_inv_czce_unwarrant"],
    "SM": ["SM_inv_czce_warrant", "SM_inv_czce_unwarrant"],
    "m": ["m_inv_dce_warrant"],
    "RM": ["RM_inv_czce_warrant", "RM_inv_czce_unwarrant"],
    "c": ["c_inv_dce_warrant"],
    "cs": ["cs_inv_dce_warrant"],
    "a": ["a_inv_dce_warrant"],
    "b": ["b_inv_dce_warrant"],
    "jd": ["jd_inv_dce_warrant"],
    "lh": ["lh_inv_dce_warrant"],
    "y": ["y_inv_dce_warrant"],
    "p": ["p_inv_dce_warrant"],
    "OI": ["OI_inv_czce_warrant", "OI_inv_czce_unwarrant"],
    "CF": ["CF_inv_czce_warrant", "CF_inv_czce_unwarrant"],
    "CY": ["CY_inv_czce_warrant"],
    "SR": ["SR_inv_czce_warrant", "SR_inv_czce_unwarrant"],
    "AP": ["AP_inv_czce_warrant", "AP_inv_czce_unwarrant"],
    "CJ": ["CJ_inv_czce_warrant", "CJ_inv_czce_unwarrant"],
    "PK": ["PK_inv_czce_warrant"],
    "sp": ["sp_inv_shfe_warrant"],
    "ru": ["ru_inv_shfe_warrant"],
    "nr": ["nr_inv_shfe_warrant"],
    "fu": ["fu_inv_shfe_warrant"],
    "lu": ["lu_inv_ine_warrant"],
    "sc": ["sc_inv_ine_warrant"],
}


def _resolve_static_value(expr: ast.AST, env: Dict[str, Any]) -> Any:
    """Resolve simple AST expressions to Python values for static analysis."""
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.List):
        return [_resolve_static_value(x, env) for x in expr.elts]
    if isinstance(expr, ast.Tuple):
        return tuple(_resolve_static_value(x, env) for x in expr.elts)
    if isinstance(expr, ast.Dict):
        keys = [_resolve_static_value(k, env) for k in expr.keys]
        vals = [_resolve_static_value(v, env) for v in expr.values]
        return dict(zip(keys, vals))
    if isinstance(expr, ast.Name):
        return env.get(expr.id)
    if isinstance(expr, ast.JoinedStr):
        parts: List[str] = []
        for v in expr.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                rv = _resolve_static_value(v.value, env)
                if rv is None:
                    return None
                parts.append(str(rv))
            else:
                return None
        return "".join(parts)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        lv = _resolve_static_value(expr.left, env)
        rv = _resolve_static_value(expr.right, env)
        if isinstance(lv, str) and isinstance(rv, str):
            return lv + rv
    if isinstance(expr, ast.Subscript):
        target = _resolve_static_value(expr.value, env)
        key = _resolve_static_value(expr.slice, env)
        if isinstance(target, (dict, list, tuple)) and key is not None:
            try:
                return target[key]
            except Exception:
                return None
    return None


def _extract_key_names(expr: ast.AST, env: Dict[str, Any]) -> List[str]:
    """Resolve spot_dict key expression to one or multiple concrete names."""
    v = _resolve_static_value(expr, env)
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if isinstance(x, (str, int, float))]
    return []


def _collect_value_deps(expr: ast.AST, env: Dict[str, Any], out: Set[str]) -> None:
    """Collect direct dependencies in a spot_dict assignment expression."""
    if isinstance(expr, ast.Subscript):
        # spot_df[...] references are base dependencies.
        if isinstance(expr.value, ast.Name) and expr.value.id == "spot_df":
            names = _extract_key_names(expr.slice, env)
            for n in names:
                out.add(n)
        # spot_dict[...] references are transitive dependencies.
        elif isinstance(expr.value, ast.Name) and expr.value.id == "spot_dict":
            names = _extract_key_names(expr.slice, env)
            for n in names:
                out.add(f"@derived:{n}")

    for child in ast.iter_child_nodes(expr):
        _collect_value_deps(child, env, out)


def _collect_spot_df_subscripts(expr: ast.AST, env: Dict[str, Any], out: Set[str]) -> None:
    """Collect spot_df[...] key usage from an AST node."""
    if isinstance(expr, ast.Subscript):
        if isinstance(expr.value, ast.Name) and expr.value.id == "spot_df":
            names = _extract_key_names(expr.slice, env)
            for n in names:
                out.add(n)

    for child in ast.iter_child_nodes(expr):
        _collect_spot_df_subscripts(child, env, out)


def _bind_function_default_args(fn_node: ast.FunctionDef, env: Dict[str, Any]) -> None:
    """Bind function default argument values into analysis env."""
    total_args = fn_node.args.args
    defaults = fn_node.args.defaults
    if not defaults:
        return

    start_idx = len(total_args) - len(defaults)
    for i, default_expr in enumerate(defaults):
        arg_name = total_args[start_idx + i].arg
        val = _resolve_static_value(default_expr, env)
        if val is not None:
            env[arg_name] = val


def build_function_spot_df_dependency_set(
    func: Any,
    env_seed: Dict[str, Any] | None = None,
) -> Set[str]:
    """Build a set of spot_df aliases used by a function body."""
    src = textwrap.dedent(inspect.getsource(func))
    mod = ast.parse(src)
    if not mod.body or not isinstance(mod.body[0], ast.FunctionDef):
        return set()
    fn_node = mod.body[0]

    deps: Set[str] = set()

    def walk(stmts: List[ast.stmt], env: Dict[str, Any]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                # Keep simple bindings (e.g., spot_name = io_brand_dict[brand_name]).
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    val = _resolve_static_value(stmt.value, env)
                    if val is not None:
                        env[stmt.targets[0].id] = val

                _collect_spot_df_subscripts(stmt.value, env, deps)

            elif isinstance(stmt, ast.For):
                iter_val = _resolve_static_value(stmt.iter, env)
                if isinstance(iter_val, list):
                    for item in iter_val:
                        loop_env = dict(env)
                        if isinstance(stmt.target, ast.Name):
                            loop_env[stmt.target.id] = item
                        elif isinstance(stmt.target, ast.Tuple) and isinstance(item, tuple):
                            for i, elt in enumerate(stmt.target.elts):
                                if isinstance(elt, ast.Name) and i < len(item):
                                    loop_env[elt.id] = item[i]
                        walk(stmt.body, loop_env)
                elif isinstance(iter_val, dict) and isinstance(stmt.target, ast.Name):
                    for k in iter_val.keys():
                        loop_env = dict(env)
                        loop_env[stmt.target.id] = k
                        walk(stmt.body, loop_env)
                else:
                    walk(stmt.body, dict(env))

                walk(stmt.orelse, dict(env))

            elif isinstance(stmt, ast.If):
                _collect_spot_df_subscripts(stmt.test, env, deps)
                walk(stmt.body, dict(env))
                walk(stmt.orelse, dict(env))

            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                _collect_spot_df_subscripts(stmt.value, env, deps)

            elif isinstance(stmt, ast.Expr):
                _collect_spot_df_subscripts(stmt.value, env, deps)

    env = dict(env_seed or {})
    _bind_function_default_args(fn_node, env)
    walk(fn_node.body, env)
    return deps


def build_ctd_basis_formula_dependency_map() -> Dict[str, Set[str]]:
    """Build dependency map for ctd-basis-derived spot aliases."""
    io_deps = build_function_spot_df_dependency_set(
        io_ctd_basis,
        env_seed={"io_brand_dict": io_brand_dict},
    )
    si_deps = build_function_spot_df_dependency_set(si_ctd_basis)
    return {
        "io_ctd_spot": io_deps,
        "si_ctd_spot": si_deps,
        # Backward/alternate alias seen in some local configs.
        "si_std_spot": si_deps,
    }


def build_update_db_factor_formula_dependency_map() -> Dict[str, Set[str]]:
    """Build spot_df assignment dependency map from update_db_factor.

    Captures any key assigned via spot_df[...] in update_db_factor and extracts
    underlying spot_df alias usage from the RHS expression. For helper calls
    like io_ctd_basis/si_ctd_basis, dependencies are expanded to the helper's
    internal spot_df ticker usage.
    """
    src = textwrap.dedent(inspect.getsource(update_db_factor))
    mod = ast.parse(src)
    if not mod.body or not isinstance(mod.body[0], ast.FunctionDef):
        return {}
    fn_node = mod.body[0]

    helper_dep_map = build_ctd_basis_formula_dependency_map()
    func_name_dep_map: Dict[str, Set[str]] = {
        "io_ctd_basis": set(helper_dep_map.get("io_ctd_spot", set())),
        "si_ctd_basis": set(helper_dep_map.get("si_ctd_spot", set())),
    }

    dep_map: Dict[str, Set[str]] = {}

    def collect_call_deps(expr: ast.AST, out: Set[str]) -> None:
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in func_name_dep_map:
                    out.update(func_name_dep_map[func_name])

    def walk(stmts: List[ast.stmt], env: Dict[str, Any]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                # Keep simple bindings (markets list, local config dicts, etc.).
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    val = _resolve_static_value(stmt.value, env)
                    if val is not None:
                        env[stmt.targets[0].id] = val

                for tgt in stmt.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "spot_df"
                    ):
                        key_names = _extract_key_names(tgt.slice, env)
                        if not key_names:
                            continue
                        deps: Set[str] = set()
                        _collect_spot_df_subscripts(stmt.value, env, deps)
                        collect_call_deps(stmt.value, deps)
                        for key in key_names:
                            dep_map.setdefault(key, set()).update(deps)

            elif isinstance(stmt, ast.For):
                iter_val = _resolve_static_value(stmt.iter, env)
                if isinstance(iter_val, list):
                    for item in iter_val:
                        loop_env = dict(env)
                        if isinstance(stmt.target, ast.Name):
                            loop_env[stmt.target.id] = item
                        elif isinstance(stmt.target, ast.Tuple) and isinstance(item, tuple):
                            for i, elt in enumerate(stmt.target.elts):
                                if isinstance(elt, ast.Name) and i < len(item):
                                    loop_env[elt.id] = item[i]
                        walk(stmt.body, loop_env)
                elif isinstance(iter_val, dict) and isinstance(stmt.target, ast.Name):
                    for k in iter_val.keys():
                        loop_env = dict(env)
                        loop_env[stmt.target.id] = k
                        walk(stmt.body, loop_env)
                else:
                    walk(stmt.body, dict(env))
                walk(stmt.orelse, dict(env))

            elif isinstance(stmt, ast.If):
                walk(stmt.body, dict(env))
                walk(stmt.orelse, dict(env))

            elif isinstance(stmt, ast.Try):
                walk(stmt.body, dict(env))
                for h in stmt.handlers:
                    walk(h.body, dict(env))
                walk(stmt.orelse, dict(env))
                walk(stmt.finalbody, dict(env))

    walk(fn_node.body, {})
    return dep_map


def build_process_spot_formula_dependency_map() -> Dict[str, Set[str]]:
    """Build derived_key -> base spot alias dependencies for process_spot_df."""
    src = textwrap.dedent(inspect.getsource(process_spot_df))
    fn_node = ast.parse(src).body[0]
    if not isinstance(fn_node, ast.FunctionDef):
        return {}

    # dynamic map from derived key -> direct deps (base or @derived:...)
    raw_dep_map: Dict[str, Set[str]] = {}

    def walk(stmts: List[ast.stmt], env: Dict[str, Any]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                # keep simple variable bindings (warrant_dict, asset_pairs, etc.)
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    val = _resolve_static_value(stmt.value, env)
                    if val is not None:
                        env[stmt.targets[0].id] = val

                for tgt in stmt.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "spot_dict"
                    ):
                        key_names = _extract_key_names(tgt.slice, env)
                        if not key_names:
                            continue
                        deps: Set[str] = set()
                        _collect_value_deps(stmt.value, env, deps)
                        for key in key_names:
                            raw_dep_map.setdefault(key, set()).update(deps)

            elif isinstance(stmt, ast.For):
                iter_val = _resolve_static_value(stmt.iter, env)
                if isinstance(iter_val, list):
                    for item in iter_val:
                        loop_env = dict(env)
                        if isinstance(stmt.target, ast.Name):
                            loop_env[stmt.target.id] = item
                        elif isinstance(stmt.target, ast.Tuple) and isinstance(item, tuple):
                            for i, elt in enumerate(stmt.target.elts):
                                if isinstance(elt, ast.Name) and i < len(item):
                                    loop_env[elt.id] = item[i]
                        walk(stmt.body, loop_env)
                elif isinstance(iter_val, dict) and isinstance(stmt.target, ast.Name):
                    for k in iter_val.keys():
                        loop_env = dict(env)
                        loop_env[stmt.target.id] = k
                        walk(stmt.body, loop_env)
                else:
                    # Unknown iteration source, still inspect body once with current env.
                    walk(stmt.body, dict(env))
            elif isinstance(stmt, ast.If):
                walk(stmt.body, dict(env))
                walk(stmt.orelse, dict(env))

    walk(fn_node.body, {})

    # Resolve transitive @derived refs to base aliases.
    resolved_cache: Dict[str, Set[str]] = {}

    def resolve_key(key: str, stack: Set[str]) -> Set[str]:
        if key in resolved_cache:
            return set(resolved_cache[key])
        if key in stack:
            return set()

        stack.add(key)
        deps = raw_dep_map.get(key, set())
        base: Set[str] = set()
        for dep in deps:
            if dep.startswith("@derived:"):
                base.update(resolve_key(dep.split(":", 1)[1], stack))
            else:
                base.add(dep)
        stack.remove(key)
        resolved_cache[key] = set(base)
        return base

    final_map: Dict[str, Set[str]] = {}
    for dk in raw_dep_map:
        final_map[dk] = resolve_key(dk, set())

    return final_map

EXCH_WARRANT_ASSETS: List[str] = [
    "cu", "bc", "zn", "al", "pb", "ni", "sn", "ao", "ss", "si", "lc",
    "SH", "TA", "PX", "MA", "UR", "PF", "l", "pp", "v", "eg", "eb",
    "pg", "bu", "SA", "FG", "j", "jm", "rb", "hc", "i", "SF", "SM",
    "m", "RM", "c", "cs", "a", "b", "jd", "lh", "y", "p", "OI", "CF",
    "CY", "SR", "AP", "CJ", "PK", "sp", "ru", "nr", "fu", "lu", "sc",
]

INV_EXCH_LME_BASE_ASSETS: List[str] = ["cu", "al", "zn", "pb", "ni", "sn"]


def derive_process_spot_df_columns(base_columns: Set[str]) -> Set[str]:
    """Infer columns produced by process_spot_df using synthetic input data.

    get_fun_data constructs spot_df from index_map aliases and then calls
    process_spot_df. To capture those dependencies comprehensively, this helper
    runs process_spot_df on a synthetic DataFrame and records resulting columns.
    """
    required_columns = set(base_columns)
    date_index = pd.date_range(start="2018-01-01", periods=520, freq="D")

    for _ in range(10):
        test_df = pd.DataFrame(1.0, index=date_index, columns=sorted(required_columns))
        try:
            out_df = process_spot_df(test_df, adjust_time=False)
            return set(out_df.columns)
        except KeyError as exc:
            # Some features pull additional aliases. Add missing columns and retry.
            msg = str(exc)
            found = re.findall(r"'([^']+)'", msg)
            new_cols = {
                col for col in found
                if col and col not in {"not in index"}
            }
            if not new_cols:
                break
            required_columns.update(new_cols)
        except Exception:
            # Keep script resilient: fallback to known columns if inference fails.
            break

    return set(required_columns)


def build_spot_df_column_universe() -> Set[str]:
    """Build an approximate set of spot_df columns used in factor update.

    The set includes:
    - aliases loaded via index_map and renamed in get_fun_data
    - extra derived columns from get_fun_data
    - per-asset additions in update_db_factor (px/drng/logret/colr/ryield/basmom*/phycarry)
    - hardcoded spread additions from update_db_factor
    """
    base_cols: Set[str] = set(index_map.values())
    cols: Set[str] = set(base_cols)
    update_spot_map = build_update_db_factor_formula_dependency_map()
    cols.update(derive_process_spot_df_columns(base_cols))
    cols.update(GET_FUN_DATA_DERIVED_COLUMNS)
    cols.update(PROCESS_SPOT_DERIVED_COLUMNS)
    cols.update(update_spot_map.keys())

    for asset in EXCH_WARRANT_ASSETS:
        cols.add(f"{asset}_exch_warrant")

    for asset in INV_EXCH_LME_BASE_ASSETS:
        cols.add(f"{asset}_inv_exch_d")
        cols.add(f"{asset}_lme_futbasis")

    for asset in MARKETS_IN_UPDATE_DB_FACTOR:
        cols.add(f"{asset}_px")
        cols.add(f"{asset}_drng")
        cols.add(f"{asset}_logret")
        cols.add(f"{asset}_colr")
        cols.add(f"{asset}_ryield")
        cols.add(f"{asset}_basmom")
        cols.add(f"{asset}_basmom5")
        cols.add(f"{asset}_basmom10")
        cols.add(f"{asset}_basmom20")
        cols.add(f"{asset}_basmom40")
        cols.add(f"{asset}_basmom60")
        cols.add(f"{asset}_basmom120")
        if asset in commod_phycarry_dict:
            cols.add(f"{asset}_phycarry")

    cols.update({"hc_rb_diff", "rb_hc_basmom_diff", "rb_hc_phycarry_diff"})
    return cols


def invert_index_map(index_mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """Create alias -> [index_code] mapping (list for duplicate aliases)."""
    alias_to_codes: Dict[str, List[str]] = {}
    for idx_code, alias in index_mapping.items():
        alias_to_codes.setdefault(alias, []).append(idx_code)
    return alias_to_codes


def extract_required_keys_for_signal(signal_name: str) -> Tuple[Set[str], Set[str], str, bool]:
    """Return required spot keys and metadata for one signal.

    Returns:
    - required_keys: direct spot_df keys needed by signal calculation logic
    - transitive_keys: additional upstream keys used to form required_keys
    - feature_name: item[1][0] from signal_store
    - is_factor_by_asset: whether signal is routed in factors_by_asset
    """
    item = signal_store[signal_name]
    assets = list(item[0]) if isinstance(item[0], Sequence) else []
    config = item[1]
    feature_name = str(config[0])
    is_factor_by_asset = is_by_asset_signal(signal_name)

    # Use production routing assets when available so dependency output mirrors
    # update_db_factor behavior, even if signal_store's asset list is stale.
    if is_factor_by_asset and signal_name in factors_by_asset:
        assets = list(factors_by_asset[signal_name])

    required_keys: Set[str] = set()
    transitive_keys: Set[str] = set()

    if is_factor_by_asset:
        if feature_name in IGNORED_PRICE_FEATURES:
            return required_keys, transitive_keys, feature_name, is_factor_by_asset

        feature_map = feature_to_feature_key_mapping.get(feature_name)
        for asset in assets:
            if feature_name == "phycarry" and asset in commod_phycarry_dict:
                # The signal uses {asset}_phycarry directly, but this input key
                # is built from asset-level physical spot aliases.
                transitive_keys.add(commod_phycarry_dict[asset])

            if isinstance(feature_map, dict) and asset in feature_map:
                required_keys.add(str(feature_map[asset]))
            else:
                required_keys.add(f"{asset}_{feature_name}")
    else:
        required_keys.add(feature_name)

    return required_keys, transitive_keys, feature_name, is_factor_by_asset


def is_by_asset_signal(signal_name: str) -> bool:
    """Determine whether a signal should be treated as by-asset.

    Besides explicit factors_by_asset routing, cross-sectional variants are also
    treated as by-asset signals:
    - *_xdemean
    - *_xscore
    - *_xrank or *_xrank{n}
    - *_xsecmean
    """
    if signal_name in factors_by_asset:
        return True

    if signal_name.endswith("_xdemean"):
        return True
    if signal_name.endswith("_xscore"):
        return True
    if signal_name.endswith("_xsecmean"):
        return True
    if re.search(r"_xrank\\d*$", signal_name):
        return True

    return False


def get_process_spot_formula_deps(required_key: str) -> List[str]:
    """Get base spot aliases used to compute a process_spot_df-derived key."""
    if required_key in PROCESS_SPOT_FORMULA_DEPS:
        return PROCESS_SPOT_FORMULA_DEPS[required_key]

    if required_key.endswith("_exch_warrant"):
        asset = required_key[: -len("_exch_warrant")]
        return EXCH_WARRANT_SOURCE_COLUMNS.get(asset, [])

    if required_key.endswith("_inv_exch_d"):
        asset = required_key[: -len("_inv_exch_d")]
        if asset in INV_EXCH_LME_BASE_ASSETS:
            return [f"{asset}_inv_shfe_d", f"{asset}_inv_lme_total"]

    if required_key.endswith("_lme_futbasis"):
        asset = required_key[: -len("_lme_futbasis")]
        if asset in INV_EXCH_LME_BASE_ASSETS:
            return [f"{asset}_lme_0m_3m_spd", f"{asset}_lme_3m_close"]

    return []


def get_production_signal_list() -> List[str]:
    """Return the default production signal list.

    Includes keys from single_factors, factors_by_asset,
    factors_by_spread, factors_by_spread2, and
    factors_by_beta_neutral.
    """
    names: List[str] = (
        list(single_factors.keys())
        + list(factors_by_asset.keys())
        + list(factors_by_spread.keys())
        + list(factors_by_spread2.keys())
        + list(factors_by_beta_neutral.keys())
    )
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    result: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def build_dependency_rows(
    signal_names: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build row-wise dependency table for selected signals.

    Args:
        signal_names: Signals to include. Defaults to the full
            production signal list (single_factors +
            factors_by_asset + factors_by_spread +
            factors_by_spread2 + factors_by_beta_neutral).
    """
    if signal_names is None:
        signal_names = get_production_signal_list()

    unknown = sorted(s for s in signal_names if s not in signal_store)
    if unknown:
        print(
            f"Warning: {len(unknown)} signal(s) not found in "
            f"signal_store and will be skipped: {unknown[:5]}"
            + ("..." if len(unknown) > 5 else "")
        )

    selected = [s for s in signal_names if s in signal_store]

    spot_df_cols = build_spot_df_column_universe()
    alias_to_codes = invert_index_map(index_map)
    process_formula_deps = build_process_spot_formula_dependency_map()
    ctd_formula_deps = build_ctd_basis_formula_dependency_map()
    update_formula_deps = build_update_db_factor_formula_dependency_map()

    rows: List[Dict[str, Any]] = []
    for signal_name in selected:
        required_keys, transitive_keys, feature_name, is_factor_by_asset = (
            extract_required_keys_for_signal(signal_name)
        )

        if feature_name == "metal_px" or (
            is_factor_by_asset and feature_name in IGNORED_PRICE_FEATURES
        ):
            rows.append(
                {
                    "signal": signal_name,
                    "feature": feature_name,
                    "route": "factors_by_asset" if is_factor_by_asset else "non_factors_by_asset",
                    "required_key": "",
                    "transitive_key": "",
                    "in_spot_df": "ignored_price_feature",
                    "index_codes": "",
                    "notes": "Feature ignored by design (price-derived).",
                }
            )
            continue

        if feature_name in TEMP_EXCLUDED_FEATURES:
            rows.append(
                {
                    "signal": signal_name,
                    "feature": feature_name,
                    "route": "factors_by_asset" if is_factor_by_asset else "non_factors_by_asset",
                    "required_key": "",
                    "transitive_key": "",
                    "in_spot_df": "temporarily_excluded",
                    "index_codes": "",
                    "transitive_index_codes": "",
                    "notes": "Feature excluded temporarily from dependency tracking.",
                }
            )
            continue

        if not required_keys:
            rows.append(
                {
                    "signal": signal_name,
                    "feature": feature_name,
                    "route": "factors_by_asset" if is_factor_by_asset else "non_factors_by_asset",
                    "required_key": "",
                    "transitive_key": "",
                    "in_spot_df": "no_required_key",
                    "index_codes": "",
                    "notes": "No required key extracted.",
                }
            )
            continue

        for req_key in sorted(required_keys):
            idx_codes = alias_to_codes.get(req_key, [])
            related_transitive: List[str] = []

            # process_spot_df formula tracing for derived fields
            related_transitive.extend(get_process_spot_formula_deps(req_key))
            related_transitive.extend(sorted(process_formula_deps.get(req_key, set())))
            related_transitive.extend(sorted(ctd_formula_deps.get(req_key, set())))
            related_transitive.extend(sorted(update_formula_deps.get(req_key, set())))

            if req_key.endswith("_phycarry"):
                asset = req_key.split("_", 1)[0]
                trans_key = commod_phycarry_dict.get(asset)
                if trans_key:
                    related_transitive.append(trans_key)
                    related_transitive.extend(
                        sorted(ctd_formula_deps.get(trans_key, set()))
                    )
            related_transitive = sorted(set(related_transitive))
            transitive_codes: List[str] = []
            for t_key in related_transitive:
                transitive_codes.extend(alias_to_codes.get(t_key, []))

            rows.append(
                {
                    "signal": signal_name,
                    "feature": feature_name,
                    "route": "factors_by_asset" if is_factor_by_asset else "non_factors_by_asset",
                    "required_key": req_key,
                    "transitive_key": "|".join(related_transitive),
                    "in_spot_df": "yes" if req_key in spot_df_cols else "no",
                    "index_codes": "|".join(sorted(idx_codes)),
                    "transitive_index_codes": "|".join(sorted(set(transitive_codes))),
                    "notes": "",
                }
            )

    return rows


def collect_production_index_codes(
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, str]:
    """Return {index_code: alias} for codes referenced in dependency rows.

    Only rows with resolved index_codes (in_spot_df == 'yes' or the
    code is directly found in index_map) are included.  Transitive
    codes are included as well so upstream data can also be monitored.
    """
    code_to_alias: Dict[str, str] = {v: k for k, v in index_map.items()}
    # Build alias -> [codes] once.
    alias_to_codes = invert_index_map(index_map)
    result: Dict[str, str] = {}
    for row in rows:
        for field in ("index_codes", "transitive_index_codes"):
            for code in row.get(field, "").split("|"):
                code = code.strip()
                if code and code in index_map:
                    result[code] = index_map[code]
    return result


def generate_data_freshness_report(
    index_codes: Dict[str, str],
    output_csv: Path,
    source: List[str] | None = None,
) -> None:
    """Query DB for tracked index codes and write a data freshness CSV.

    For each code, the report contains:
    - spot_id      : index code
    - alias        : human-readable alias from index_map
    - last_date    : most recent date with a value
    - last_value   : value on last_date
    - prev_date    : second most recent date
    - prev_value   : value on prev_date

    Args:
        index_codes: Mapping of {index_code: alias} to query.
        output_csv:  Destination CSV file path.
        source:      EDB source list (defaults to ['ifind']).
    """
    if source is None:
        source = ["ifind"]

    code_list = sorted(index_codes.keys())
    if not code_list:
        print("No index codes to report on.")
        return

    try:
        pivot = load_codes_from_edb(
            code_list,
            source=source,
            column_name="index_code",
        )
    except Exception as exc:
        print(f"Error loading EDB data: {exc}")
        return

    records: List[Dict[str, Any]] = []
    for code in code_list:
        alias = index_codes.get(code, "")
        if code not in pivot.columns:
            records.append(
                {
                    "spot_id": code,
                    "alias": alias,
                    "last_date": "",
                    "last_value": "",
                    "prev_date": "",
                    "prev_value": "",
                    "notes": "not_in_db",
                }
            )
            continue

        series = pivot[code].dropna().sort_index()
        if len(series) == 0:
            records.append(
                {
                    "spot_id": code,
                    "alias": alias,
                    "last_date": "",
                    "last_value": "",
                    "prev_date": "",
                    "prev_value": "",
                    "notes": "no_data",
                }
            )
            continue

        last_date = series.index[-1].strftime("%Y-%m-%d")
        last_value = series.iloc[-1]
        prev_date = ""
        prev_value = ""
        if len(series) >= 2:
            prev_date = series.index[-2].strftime("%Y-%m-%d")
            prev_value = series.iloc[-2]

        records.append(
            {
                "spot_id": code,
                "alias": alias,
                "last_date": last_date,
                "last_value": last_value,
                "prev_date": prev_date,
                "prev_value": prev_value,
                "notes": "",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "spot_id",
        "alias",
        "last_date",
        "last_value",
        "prev_date",
        "prev_value",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Data freshness report written to: {output_csv}")
    print(f"  Total codes tracked: {len(code_list)}")
    stale = [
        r for r in records
        if r["notes"] in ("not_in_db", "no_data")
    ]
    if stale:
        print(f"  Codes with no data: {len(stale)}")


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate quick metrics for dependency report."""
    rows_list = list(rows)
    unique_signals = sorted({row["signal"] for row in rows_list})
    active_rows = [
        row
        for row in rows_list
        if row["required_key"] and row["in_spot_df"] in {"yes", "no"}
    ]

    unresolved = [
        row
        for row in active_rows
        if row["in_spot_df"] == "no"
    ]
    unresolved_with_no_index = [
        row for row in unresolved if not row["index_codes"]
    ]

    return {
        "signal_count": len(unique_signals),
        "dependency_rows": len(active_rows),
        "unresolved_rows": len(unresolved),
        "unresolved_without_index_code": len(unresolved_with_no_index),
    }


def write_csv(rows: Sequence[Dict[str, Any]], output_csv: Path) -> None:
    """Write dependency details to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "signal",
        "feature",
        "route",
        "required_key",
        "transitive_key",
        "in_spot_df",
        "index_codes",
        "transitive_index_codes",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: Dict[str, Any], output_json: Path) -> None:
    """Write report summary JSON."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_email_html(
    dep_summary: Dict[str, Any],
    csv_path: Path,
    summary_json_path: Path,
    freshness_csv_path: Path | None,
) -> str:
    """Build HTML body for dependency and data freshness report notification."""
    html = "<html><head></head><body><p><br>"
    html += "<b>Data Health Report</b><br>"
    html += f"Dependency CSV: {csv_path}<br>"
    html += f"Dependency summary JSON: {summary_json_path}<br><br>"

    dep_df = pd.DataFrame([dep_summary])
    html += "Dependency summary:<br>"
    html += dep_df.to_html(index=False)

    if freshness_csv_path is None or (not freshness_csv_path.exists()):
        html += "<br><b>Data freshness report:</b> not generated in this run.<br>"
        html += "</p></body></html>"
        return html

    fdf = pd.read_csv(freshness_csv_path)
    for col in ["last_date", "prev_date"]:
        fdf[col] = pd.to_datetime(fdf[col], errors="coerce")

    today = pd.Timestamp(datetime.date.today())
    fdf["age_days"] = (today - fdf["last_date"]).dt.days

    stats = {
        "rows": int(len(fdf)),
        "notes_non_empty": int((fdf["notes"].fillna("") != "").sum()),
        "age_le_2": int((fdf["age_days"] <= 2).sum()),
        "age_3_7": int(((fdf["age_days"] >= 3) & (fdf["age_days"] <= 7)).sum()),
        "age_8_30": int(((fdf["age_days"] >= 8) & (fdf["age_days"] <= 30)).sum()),
        "age_gt_30": int((fdf["age_days"] > 30).sum()),
    }

    html += f"<br>Freshness CSV: {freshness_csv_path}<br>"
    html += "Freshness summary:<br>"
    html += pd.DataFrame([stats]).to_html(index=False)

    stale_df = (
        fdf.loc[fdf["age_days"] >= 3, ["spot_id", "alias", "last_date", "age_days"]]
        .sort_values("age_days", ascending=False)
        .head(30)
        .copy()
    )
    if len(stale_df) > 0:
        stale_df["last_date"] = stale_df["last_date"].dt.strftime("%Y-%m-%d")
        html += "<br>Top stale series (age >= 3d):<br>"
        html += stale_df.to_html(index=False)

    html += "</p></body></html>"
    return html


def send_report_email(
    dep_summary: Dict[str, Any],
    csv_path: Path,
    summary_json_path: Path,
    freshness_csv_path: Path | None,
) -> None:
    """Send data health report as HTML email."""
    sub = (
        f"{LOCAL_PC_NAME} data health report"
        f"<{datetime.date.today().strftime('%Y.%m.%d')}>"
    )
    html = _build_email_html(
        dep_summary=dep_summary,
        csv_path=csv_path,
        summary_json_path=summary_json_path,
        freshness_csv_path=freshness_csv_path,
    )
    send_html_by_smtp(EMAIL_QQ, NOTIFIERS, sub, html)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build signal data dependency map."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("C:/dev/data") / "signal_data_dependency_map.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("C:/dev/data") / "signal_data_dependency_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--signals",
        nargs="*",
        default=None,
        metavar="SIGNAL",
        help=(
            "Signals to include in the dependency map. "
            "Defaults to all production signals "
            "(single_factors + factors_by_asset + factors_by_spread "
            "+ factors_by_spread2 + factors_by_beta_neutral)."
        ),
    )
    parser.add_argument(
        "--data-report",
        action="store_true",
        default=False,
        help=(
            "Generate a data freshness report for all index codes "
            "used by the selected production signals."
        ),
    )
    parser.add_argument(
        "--data-report-csv",
        type=Path,
        default=Path("C:/dev/data") / "data_freshness_report.csv",
        help="Output path for the data freshness report CSV.",
    )
    parser.add_argument(
        "--edb-source",
        nargs="+",
        default=["ifind"],
        metavar="SOURCE",
        help="EDB source(s) to query (default: ifind).",
    )
    parser.add_argument(
        "--email-notify",
        action="store_true",
        default=EMAIL_NOTIFY,
        help=(
            "Send HTML email with dependency and freshness summary. "
            "Defaults to EMAIL_NOTIFY from sec_bits."
        ),
    )
    return parser.parse_args()


def get_codes_by_signals(signal_names):
    """Helper to get all index codes required by a list of signals."""
    rows = build_dependency_rows(signal_names)
    code_to_alias = collect_production_index_codes(rows)
    return code_to_alias


def main() -> int:
    """Generate dependency mapping artifacts."""
    args = parse_args()
    signal_names: List[str] | None = (
        args.signals if args.signals is not None else None
    )
    rows = build_dependency_rows(signal_names)
    summary = summarize_rows(rows)

    write_csv(rows, args.csv)
    write_json(summary, args.summary_json)

    print("Generated dependency map")
    print(f"  CSV: {args.csv}")
    print(f"  Summary: {args.summary_json}")
    print(f"  Signals: {summary['signal_count']}")
    print(f"  Dependency rows: {summary['dependency_rows']}")
    print(f"  Unresolved rows: {summary['unresolved_rows']}")
    print(
        "  Unresolved rows without index code: "
        f"{summary['unresolved_without_index_code']}"
    )

    freshness_csv_path: Path | None = None
    if args.data_report:
        index_codes = collect_production_index_codes(rows)
        generate_data_freshness_report(
            index_codes,
            args.data_report_csv,
            source=args.edb_source,
        )
        freshness_csv_path = args.data_report_csv

    if args.email_notify:
        send_report_email(
            dep_summary=summary,
            csv_path=args.csv,
            summary_json_path=args.summary_json,
            freshness_csv_path=freshness_csv_path,
        )
        print("HTML report email sent.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
