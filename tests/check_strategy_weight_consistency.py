"""Check consistency of strategy weights across notebook, JSON, and scaler config.

This script validates two things:
1) Group scaler consistency between notebook *_fact values and
   misc_scripts/factor_data_update.py port_pos_config strat_list scalers.
2) Per-signal weight consistency between notebook group lists and
   process/paper_sim1/settings/*.json factor_repo weights.

Exit code is 0 when all checks pass, otherwise 1.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_NOTEBOOK = Path("c:/dev/pyktrader3/bktest/bktest_prod_daily_run.ipynb")
DEFAULT_SETTINGS = Path("c:/dev/pyktrader3/process/paper_sim1/settings")
DEFAULT_FACTOR_UPDATE = Path("c:/dev/pyktrader3/misc_scripts/factor_data_update.py")
DEFAULT_PORT_NAME = "PTSIM1_FACTPORT1_hot"

GROUP_TO_JSON: Dict[str, List[str]] = {
    "prem_strats": ["PTSIM1_FACTPORT1.json"],
    "misc_strats": [
        "PTSIM1_EXCHWNT.json",
        "PTSIM1_LL.json",
        "PTSIM1_LL2MR.json",
        "PTSIM1_MR1Y.json",
        "PTSIM1_SPDTF.json",
        "PTSIM1_HRCRB.json",
    ],
    "metal_strats": ["PTSIM1_FUNMTL.json"],
    "eqmtl_strats": ["PTSIM1_EQMTL.json"],
    "ferrous_strats": ["PTSIM1_FUNFER.json"],
    "smsf_spd_strats": ["PTSIM1_SMSFSPD.json"],
    "rbhc_spd_strats": ["PTSIM1_RBHCSPD.json"],
    "base_strats": ["PTSIM1_FUNBASE.json"],
    "mixmtl_strats": ["PTSIM1_FUNMIXMTL.json"],
    "energy_strats": ["PTSIM1_FUNENE.json"],
    "macro_strats": ["PTSIM1_CNMAC1.json"],
    "macro2_strats": ["PTSIM1_CNMAC2.json"],
    "seazn_strats": ["PTSIM1_SEAZN.json"],
    "bond_strats": ["PTSIM1_BND1.json"],
    "auspd_strats": ["PTSIM1_AUSPD.json"],
}


@dataclass
class NotebookSignal:
    signal: str
    coeff: float
    factor_var: Optional[str]
    abs_weight: float


@dataclass
class JsonSignal:
    signal: str
    weight: float
    source_file: str


def _read_notebook_code(notebook_path: Path) -> str:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    chunks: List[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        txt = "\n".join(source)
        if "prem_strats = [" in txt and "strat_group =" in txt:
            chunks.append(txt)
    if not chunks:
        raise ValueError("Could not find strategy-group code cell in notebook")
    return "\n\n".join(chunks)


def _eval_num_expr(node: ast.AST, env: Dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return float(env[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_num_expr(node.operand, env)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _eval_num_expr(node.left, env)
        right = _eval_num_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError(f"Unsupported numeric expression: {ast.dump(node)}")


def _parse_coeff_and_var(
    node: ast.AST, env: Dict[str, float]
) -> Tuple[float, Optional[str], float]:
    if isinstance(node, ast.Name) and node.id in env:
        return 1.0, node.id, float(env[node.id])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.left, ast.Name) and node.left.id in env:
            coeff = _eval_num_expr(node.right, env)
            var = node.left.id
            return coeff, var, coeff * env[var]
        if isinstance(node.right, ast.Name) and node.right.id in env:
            coeff = _eval_num_expr(node.left, env)
            var = node.right.id
            return coeff, var, coeff * env[var]
    coeff = _eval_num_expr(node, env)
    return coeff, None, coeff


def _parse_notebook_groups(
    notebook_code: str,
) -> Tuple[Dict[str, float], Dict[str, List[NotebookSignal]]]:
    tree = ast.parse(notebook_code)
    scalar_env: Dict[str, float] = {}

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        if not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        try:
            val = _eval_num_expr(stmt.value, scalar_env)
        except Exception:
            continue
        scalar_env[name] = val

    group_map: Dict[str, List[NotebookSignal]] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        group_name = target.id
        if not group_name.endswith("_strats"):
            continue
        if not isinstance(stmt.value, ast.List):
            continue

        sigs: List[NotebookSignal] = []
        for elt in stmt.value.elts:
            if not isinstance(elt, ast.List) or len(elt.elts) != 2:
                continue
            sig_node = elt.elts[0]
            w_node = elt.elts[1]
            if not isinstance(sig_node, ast.Constant) or not isinstance(
                sig_node.value, str
            ):
                continue
            coeff, factor_var, abs_weight = _parse_coeff_and_var(w_node, scalar_env)
            sigs.append(
                NotebookSignal(
                    signal=sig_node.value,
                    coeff=float(coeff),
                    factor_var=factor_var,
                    abs_weight=float(abs_weight),
                )
            )
        group_map[group_name] = sigs

    return scalar_env, group_map


def _suffix_from_type(type_val: str) -> str:
    t = type_val.lower()
    if "xdemean" in t or ("xs" in t and "demean" in t):
        return "_xdemean"
    if "xscore" in t or ("xs" in t and "score" in t):
        return "_xscore"
    if "xrank" in t or ("xs" in t and "rank" in t):
        return "_xrank"
    return ""


def _canonical_signal(name: str, type_val: str) -> str:
    suffix = _suffix_from_type(type_val)
    if suffix and not name.endswith(suffix):
        return f"{name}{suffix}"
    return name


def _load_json_signals(path: Path) -> Dict[str, JsonSignal]:
    data = json.loads(path.read_text(encoding="utf-8"))
    repo = data.get("config", {}).get("factor_repo", {})
    out: Dict[str, JsonSignal] = {}
    for raw_key, info in repo.items():
        if not isinstance(info, dict):
            continue
        name = str(info.get("name") or str(raw_key).split(".")[0])
        type_val = str(info.get("type", ""))
        signal = _canonical_signal(name, type_val)
        weight = float(info.get("weight", 0.0))
        out[signal] = JsonSignal(signal=signal, weight=weight, source_file=path.name)
    return out


def _parse_port_pos_config(
    factor_update_path: Path, port_name: str
) -> Dict[str, float]:
    tree = ast.parse(factor_update_path.read_text(encoding="utf-8"))
    port_data = None
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and target.id == "port_pos_config":
            port_data = ast.literal_eval(stmt.value)
            break
    if port_data is None:
        config_path = factor_update_path.resolve().parents[1] / "process" / "port_pos_config.json"
        port_data = json.loads(config_path.read_text(encoding="utf-8"))
    if port_name not in port_data:
        raise ValueError(f"Port '{port_name}' not found in port_pos_config")
    strat_list = port_data[port_name]["strat_list"]
    return {str(name): float(scale) for name, scale in strat_list}


def _fmt_float(v: float) -> str:
    return f"{v:.8f}".rstrip("0").rstrip(".")


def run_check(
    notebook_path: Path,
    settings_dir: Path,
    factor_update_path: Path,
    port_name: str,
    tolerance: float,
) -> int:
    code = _read_notebook_code(notebook_path)
    scalar_env, group_signals = _parse_notebook_groups(code)
    scaler_by_file = _parse_port_pos_config(factor_update_path, port_name)

    failures: List[str] = []
    warnings: List[str] = []

    for group_name, json_files in GROUP_TO_JSON.items():
        if group_name not in group_signals:
            warnings.append(f"Group missing in notebook: {group_name}")
            continue

        nb_signals = group_signals[group_name]
        nb_by_name = {s.signal: s for s in nb_signals}

        used_factors = {s.factor_var for s in nb_signals if s.factor_var}
        used_factors_vals = {
            name: scalar_env[name] for name in sorted(used_factors) if name in scalar_env
        }

        json_by_name: Dict[str, JsonSignal] = {}
        for fname in json_files:
            full = settings_dir / fname
            if not full.exists():
                failures.append(
                    f"[{group_name}] JSON file not found: {full.as_posix()}"
                )
                continue
            sigs = _load_json_signals(full)
            overlap = set(json_by_name).intersection(sigs)
            if overlap:
                warnings.append(
                    f"[{group_name}] duplicate signals across JSON files: "
                    f"{sorted(overlap)}"
                )
            json_by_name.update(sigs)

        for fname in json_files:
            if fname not in scaler_by_file:
                failures.append(
                    f"[{group_name}] missing scaler in port_pos_config for {fname}"
                )
                continue
            if not used_factors_vals:
                warnings.append(
                    f"[{group_name}] no notebook factor var detected; "
                    f"cannot verify scaler for {fname}"
                )
                continue
            target_scale = scaler_by_file[fname]
            for fact_name, fact_val in used_factors_vals.items():
                if abs(target_scale - fact_val) > tolerance:
                    failures.append(
                        f"[{group_name}] scaler mismatch {fname}: "
                        f"port_pos_config={_fmt_float(target_scale)} vs "
                        f"{fact_name}={_fmt_float(fact_val)}"
                    )

        for sig, nb_sig in sorted(nb_by_name.items()):
            if sig not in json_by_name:
                failures.append(f"[{group_name}] signal missing in JSON: {sig}")
                continue
            js = json_by_name[sig]
            if abs(nb_sig.coeff - js.weight) > tolerance:
                failures.append(
                    f"[{group_name}] weight mismatch {sig}: "
                    f"notebook={_fmt_float(nb_sig.coeff)} vs "
                    f"json={_fmt_float(js.weight)} ({js.source_file})"
                )

        extra = sorted(set(json_by_name) - set(nb_by_name))
        if extra:
            warnings.append(f"[{group_name}] extra signals in JSON only: {extra}")

    print("=== Strategy Weight Consistency Check ===")
    print(f"Notebook: {notebook_path.as_posix()}")
    print(f"Settings: {settings_dir.as_posix()}")
    print(f"Scaler source: {factor_update_path.as_posix()}::{port_name}")

    if warnings:
        print("\nWarnings:")
        for msg in warnings:
            print(f"- {msg}")

    if failures:
        print("\nFailures:")
        for msg in failures:
            print(f"- {msg}")
        print(f"\nResult: FAIL ({len(failures)} issue(s))")
        return 1

    print("\nResult: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check strategy weight consistency across notebook/JSON/scaler"
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="Path to backtest notebook",
    )
    parser.add_argument(
        "--settings-dir",
        type=Path,
        default=DEFAULT_SETTINGS,
        help="Path to strategy JSON settings directory",
    )
    parser.add_argument(
        "--factor-update",
        type=Path,
        default=DEFAULT_FACTOR_UPDATE,
        help="Path to factor_data_update.py",
    )
    parser.add_argument(
        "--port-name",
        type=str,
        default=DEFAULT_PORT_NAME,
        help="port_pos_config key name",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for float comparisons",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_check(
        notebook_path=args.notebook,
        settings_dir=args.settings_dir,
        factor_update_path=args.factor_update,
        port_name=args.port_name,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    raise SystemExit(main())
