import os
import json
import re
from typing import Any, Dict, List


_3DIGIT_RE = re.compile(r'^([A-Z]+)(\d{3})$')

def convert_czce_3digit_by_date(contract: str, date_int: int) -> str:
    """
    Convert a CZCE 3-digit contract (like CF609, CF009) into a 4-digit YYMM
    using the event date (YYYYMMDD). The resulting YYMM will be the smallest
    possible that is >= the date's YYMM (avoids mapping into the past).

    Examples:
      date=20160401, CF609 -> CF1609
      date=20260401, CF609 -> CF2609
      date=20200401, CF009 -> CF2009
      date=20300401, CF009 -> CF3009
      date=20191101, CF001 -> CF2001  (not CF1001, since 1001 < 1911)
    """
    m = _3DIGIT_RE.match(contract or "")
    if not m or not date_int:
        return contract  # leave non-3digit or missing-date untouched

    symbol, ymm3 = m.groups()
    y_digit = int(ymm3[0])          # single-digit "year within decade"
    mm = int(ymm3[1:])              # month 01..12

    # Derive YYMM for the date
    year_full = date_int // 10000
    month = (date_int // 100) % 100
    date_y = year_full % 100
    date_yymm = date_y * 100 + month

    # Start at the date's decade (e.g., 2019 -> decade YY=10), then search forward
    base_decade = (date_y // 10) * 10  # 10, 20, 30, ...
    for k in range(0, 10):  # search up to 10 decades ahead (should never need that many)
        YY = base_decade + y_digit + 10 * k
        cand_yymm = YY * 100 + mm
        if cand_yymm >= date_yymm:
            return f"{symbol}{YY:02d}{mm:02d}"

    # Fallback (theoretical only)
    YY = base_decade + y_digit
    return f"{symbol}{YY:02d}{mm:02d}"


def _fix_event_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Fix a single event object with keys like 'date', 'from', 'to'."""
    date_val = obj.get("date", 0)
    for key in ("from", "to"):
        val = obj.get(key)
        if isinstance(val, str) and _3DIGIT_RE.match(val):
            obj[key] = convert_czce_3digit_by_date(val, date_val)
    return obj


def _process_czce_node(node: Any) -> Any:
    """
    Recursively process the CZCE subtree. We expect either:
      - dict of symbols -> list[events]
      - list[events]
      - nested dicts/lists (be tolerant)
    """
    if isinstance(node, list):
        out: List[Any] = []
        for item in node:
            if isinstance(item, dict):
                # Treat dict as an event; also recurse into nested structures if present
                fixed = _fix_event_obj(item.copy())
                for k, v in list(fixed.items()):
                    if isinstance(v, (dict, list)):
                        fixed[k] = _process_czce_node(v)
                out.append(fixed)
            else:
                out.append(item)
        return out

    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if isinstance(v, list):
                out[k] = _process_czce_node(v)
            elif isinstance(v, dict):
                out[k] = _process_czce_node(v)
            else:
                out[k] = v
        return out

    return node  # primitives unchanged


def process_file_only_czce(src_path: str, dst_path: str) -> None:
    """Load JSON, adjust only the 'CZCE' branch, and write out."""
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "CZCE" in data:
        data = data.copy()
        data["CZCE"] = _process_czce_node(data["CZCE"])

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    src_folder = r"C:/dev/wtdev/common"
    target_folder = r"c:/dev/wtdev/config"

    os.makedirs(target_folder, exist_ok=True)

    in1 = os.path.join(src_folder, "hots.json")
    in2 = os.path.join(src_folder, "seconds.json")
    out1 = os.path.join(target_folder, "hot1.json")
    out2 = os.path.join(target_folder, "hot2.json")

    process_file_only_czce(in1, out1)
    process_file_only_czce(in2, out2)

    print(f"Done. Wrote:\n  {out1}\n  {out2}")
