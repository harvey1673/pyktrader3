"""Extract formula-required data IDs from all base-metal workbooks.

This script scans workbooks in xldata/base, finds margin/profit formula targets,
traces formula dependencies recursively, and outputs:
1) detailed usage rows
2) deduplicated master requirements
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string, get_column_letter


ROOT = Path("c:/dev/pyktrader3")
BASE_DIR = ROOT / "xldata/base"
OUT_DETAIL = ROOT / "docs/formula_required_data_all_detail.csv"
OUT_MASTER = ROOT / "docs/formula_required_data_all_master.csv"
OUT_MD = ROOT / "docs/formula_required_data_all_summary.md"


TARGET_KEYWORDS = (
    "margin",
    "profit",
    "利润",
    "毛利",
    "smelting margin",
)


SHEET_CELL_REF_RE = re.compile(
    r"(?:'([^']+)'|([A-Za-z0-9_\-]+))!\$?([A-Z]{1,3})\$?(\d+)"
)
LOCAL_CELL_REF_RE = re.compile(r"(?<![A-Za-z0-9_])\$?([A-Z]{1,3})\$?(\d+)")
VLOOKUP_RE = re.compile(
    r"VLOOKUP\(\s*[^,]+,\s*('[^']+'|[A-Za-z0-9_\-]+)!"
    r"\$?([A-Z]{1,3})\$?:\$?([A-Z]{1,3})\$?,\s*(\d+)",
    re.IGNORECASE,
)
VALID_ID_RE = re.compile(
    r"^(?:[sSmMgGlLaAyY]\d+|ID\d+|TODO_[A-Z0-9_]+|COMEX_[A-Z0-9_]+|"
    r"LBMA_[A-Z0-9_]+|MCU[a-zA-Z0-9]+|CM[A-Z0-9]+)$"
)


@dataclass(frozen=True)
class FormulaTarget:
    workbook: str
    target_sheet: str
    target_cell: str
    target_header: str


@dataclass(frozen=True)
class SourceUsage:
    workbook: str
    target_sheet: str
    target_cell: str
    target_header: str
    driver_sheet: str
    driver_cell: str
    source_sheet: str
    source_col: str
    source_name: str
    source_id: str


def normalize_sheet(sheet_token: str) -> str:
    """Normalize quoted and unquoted sheet tokens."""
    return sheet_token.strip("'")


def parse_cell_refs(formula: str, current_sheet: str) -> List[Tuple[str, str]]:
    """Parse local and external cell references in a formula."""
    refs: List[Tuple[str, str]] = []
    for match in SHEET_CELL_REF_RE.finditer(formula):
        sheet_q, sheet_u, col, row = match.groups()
        refs.append((normalize_sheet(sheet_q or sheet_u or current_sheet), f"{col}{row}"))

    local_formula = SHEET_CELL_REF_RE.sub("", formula)
    for match in LOCAL_CELL_REF_RE.finditer(local_formula):
        col, row = match.groups()
        refs.append((current_sheet, f"{col}{row}"))

    return refs


def parse_vlookups(formula: str) -> List[Tuple[str, str]]:
    """Parse VLOOKUP source references as (source_sheet, source_col)."""
    results: List[Tuple[str, str]] = []
    for match in VLOOKUP_RE.finditer(formula):
        raw_sheet, start_col, _, idx_str = match.groups()
        sheet = normalize_sheet(raw_sheet)
        start_idx = column_index_from_string(start_col)
        idx = int(idx_str)
        source_col = get_column_letter(start_idx + idx - 1)
        results.append((sheet, source_col))
    return results


def is_margin_header(value: object) -> bool:
    """Check whether a header/value indicates a margin/profit target."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return any(keyword in text for keyword in TARGET_KEYWORDS)


def find_formula_targets(wb, workbook_name: str) -> List[FormulaTarget]:
    """Find target formula cells for margin/profit in each worksheet."""
    targets: List[FormulaTarget] = []
    for ws in wb.worksheets:
        max_col = ws.max_column
        for col in range(1, max_col + 1):
            header = ws.cell(row=1, column=col).value
            subtitle = ws.cell(row=2, column=col).value

            # Prefer row2 labels (common in these templates), fallback to row1.
            chosen_label = subtitle if is_margin_header(subtitle) else header
            if not is_margin_header(chosen_label):
                continue

            cell_ref = f"{get_column_letter(col)}3"
            formula = ws[cell_ref].value
            if isinstance(formula, str) and formula.startswith("="):
                targets.append(
                    FormulaTarget(
                        workbook=workbook_name,
                        target_sheet=ws.title,
                        target_cell=cell_ref,
                        target_header=str(chosen_label),
                    )
                )
    return targets


def trace_sources_for_target(wb, target: FormulaTarget) -> List[SourceUsage]:
    """Trace recursive dependencies from one target formula."""
    usages: List[SourceUsage] = []
    visited: Set[Tuple[str, str]] = set()

    def walk(sheet_name: str, cell_ref: str) -> None:
        key = (sheet_name, cell_ref)
        if key in visited:
            return
        visited.add(key)

        if sheet_name not in wb.sheetnames:
            return
        ws = wb[sheet_name]
        value = ws[cell_ref].value
        if not (isinstance(value, str) and value.startswith("=")):
            return

        for source_sheet, source_col in parse_vlookups(value):
            if source_sheet not in wb.sheetnames:
                continue
            source_ws = wb[source_sheet]
            source_name = source_ws[f"{source_col}1"].value
            source_id = source_ws[f"{source_col}2"].value
            source_id_text = str(source_id or "")
            if not VALID_ID_RE.match(source_id_text):
                continue

            usages.append(
                SourceUsage(
                    workbook=target.workbook,
                    target_sheet=target.target_sheet,
                    target_cell=target.target_cell,
                    target_header=target.target_header,
                    driver_sheet=sheet_name,
                    driver_cell=cell_ref,
                    source_sheet=source_sheet,
                    source_col=source_col,
                    source_name=str(source_name or ""),
                    source_id=source_id_text,
                )
            )

        for next_sheet, next_cell in parse_cell_refs(value, sheet_name):
            walk(next_sheet, next_cell)

    walk(target.target_sheet, target.target_cell)
    return usages


def workbook_paths() -> Iterable[Path]:
    """Yield workbook paths from base directory, excluding lock files."""
    for path in sorted(BASE_DIR.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        yield path


def write_outputs(usages: List[SourceUsage], targets: List[FormulaTarget]) -> None:
    """Write detail/master CSV and markdown summary."""
    usage_rows = sorted(
        usages,
        key=lambda x: (
            x.workbook,
            x.target_sheet,
            x.target_cell,
            x.source_sheet,
            x.source_id,
        ),
    )

    with OUT_DETAIL.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "workbook",
                "target_sheet",
                "target_cell",
                "target_header",
                "driver_sheet",
                "driver_cell",
                "source_sheet",
                "source_col",
                "source_name",
                "source_id",
            ],
        )
        writer.writeheader()
        for row in usage_rows:
            writer.writerow(row.__dict__)

    master: Dict[Tuple[str, str], Dict[str, str]] = {}
    refs: Dict[Tuple[str, str], Set[str]] = {}

    for row in usage_rows:
        key = (row.source_sheet, row.source_id)
        if key not in master:
            master[key] = {
                "source_sheet": row.source_sheet,
                "source_name": row.source_name,
                "source_id": row.source_id,
                "used_in_targets": "",
            }
            refs[key] = set()
        refs[key].add(f"{row.workbook}:{row.target_sheet}!{row.target_cell}")

    master_rows = []
    for key, record in master.items():
        record["used_in_targets"] = " | ".join(sorted(refs[key]))
        master_rows.append(record)

    master_rows.sort(key=lambda x: (x["source_sheet"], x["source_id"]))

    with OUT_MASTER.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_sheet",
                "source_name",
                "source_id",
                "used_in_targets",
            ],
        )
        writer.writeheader()
        writer.writerows(master_rows)

    by_workbook: Dict[str, int] = {}
    for t in targets:
        by_workbook[t.workbook] = by_workbook.get(t.workbook, 0) + 1

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Formula Required Data Summary (All Base Workbooks)\n\n")
        f.write("从所有 base 工作簿中，按 margin/profit 公式递归追踪得到。\n\n")
        f.write(f"- 扫描工作簿数: {len(by_workbook)}\n")
        f.write(f"- 识别目标公式数: {len(targets)}\n")
        f.write(f"- 明细依赖条数: {len(usage_rows)}\n")
        f.write(f"- 去重后数据ID条数: {len(master_rows)}\n\n")
        f.write("## 工作簿目标公式数量\n\n")
        for workbook, count in sorted(by_workbook.items(), key=lambda x: (x[0])):
            f.write(f"- {workbook}: {count}\n")
        f.write("\n## 输出文件\n\n")
        f.write(f"- {OUT_DETAIL.name}\n")
        f.write(f"- {OUT_MASTER.name}\n")


def main() -> None:
    """Run extraction across all base workbooks."""
    all_targets: List[FormulaTarget] = []
    all_usages: List[SourceUsage] = []

    for workbook_path in workbook_paths():
        try:
            wb = load_workbook(workbook_path, data_only=False)
        except Exception as exc:  # pragma: no cover - resilience for bad files
            print(f"skip {workbook_path.name}: {exc}")
            continue

        targets = find_formula_targets(wb, workbook_path.name)
        all_targets.extend(targets)
        for target in targets:
            all_usages.extend(trace_sources_for_target(wb, target))

    write_outputs(all_usages, all_targets)
    print(f"generated {OUT_DETAIL}")
    print(f"generated {OUT_MASTER}")
    print(f"generated {OUT_MD}")


if __name__ == "__main__":
    main()
