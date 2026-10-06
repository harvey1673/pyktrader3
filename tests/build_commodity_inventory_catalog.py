"""Build Chinese sourcing tables from the audited inventory and research plan."""

import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/commodity_inventory_research"
AS_OF = dt.date(2026, 10, 3)
CHINESE_FIELDS = {
    "source_file": "工作簿", "sheet": "工作表", "column": "列号",
    "country": "国家或地区", "indicator": "原始指标名称", "frequency": "原始频率",
    "unit": "原始单位", "indicator_id": "供应商指标ID", "date_range": "元数据时间区间",
    "source": "原始数据来源", "updated_at": "元数据更新时间", "description": "原始描述",
    "indicator_type": "指标类型", "first_observation": "首个有效数值日期",
    "last_observation": "最后有效数值日期", "last_value": "最后有效数值",
    "observation_count": "缓存有效数值个数",
}


def write_csv(name, rows, fields=None):
    with (DIRECTORY / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    with (DIRECTORY / "existing_inventory_indicators.csv").open(encoding="utf-8-sig") as handle:
        existing = list(csv.DictReader(handle))
    with (DIRECTORY / "product_inventory_plan.psv").open(encoding="utf-8") as handle:
        products = list(csv.DictReader(handle, delimiter="|"))
    assert len({row["代码"] for row in products}) == len(products)
    assert all(row["indicator_id"] and row["last_observation"] for row in existing)
    chinese = [{CHINESE_FIELDS[key]: value for key, value in row.items()} for row in existing]
    write_csv("existing_inventory_indicators_zh.csv", chinese)
    plan, checklist, references = [], [], []
    for product in products:
        terms = product["本品匹配词"].split(";")
        matches = [row for row in existing if not row["indicator_type"].startswith("非库存")
                   and any(term in row["indicator"] for term in terms)]
        physical = [row for row in matches if row["indicator_type"] in ("库存数量", "库存天数", "库容比")]
        warrants = [row for row in matches if row["indicator_type"] == "仓单数量"]
        examples = physical[:2] + warrants[:1]
        evidence = "；".join(f"{row['indicator']} [{row['indicator_id']}; {row['last_observation']}]" for row in examples)
        coverage = "检出相关库存字段，逐条口径仍需核对" if physical else (
            "仅检出相关仓单/辅助字段；本品产业库存待确认" if matches else "本品关键词未命中；待供应商确认")
        record = {key: value for key, value in product.items() if key != "本品匹配词"}
        record.update({"本品相关库存字段列数": len(physical), "相关仓单字段列数": len(warrants),
                       "字段核查状态": coverage, "已有字段示例": evidence})
        plan.append(record)
        for relationship, field, priority in (("本品", "本品重点库存", "P0先核对已有，再补缺口"),
                                               ("上游", "上游重点库存", "P1补充"),
                                               ("下游或替代", "下游或替代品库存", "P1补充")):
            for concept in product[field].split("；"):
                checklist.append({"板块": product["板块"], "品种代码": product["代码"], "品种": product["品种"],
                                  "产业链位置": relationship, "建议检索概念": concept,
                                  "优先级": priority, "建议查询渠道": product["建议查询渠道"],
                                  "可得性状态": "研究建议；精确字段、ID、历史和权限待终端确认",
                                  "本品已有证据索引": f"inventory_research_zh.md#{product['代码'].lower()}",
                                  "供应商完整指标名": "", "供应商ID": "", "实际发布频率": "",
                                  "单位": "", "历史起始日期": "", "样本与覆盖地区": "", "发布时间": "",
                                  "订阅或API权限": "", "口径提醒": product["优先补充及口径提醒"]})
        for row in matches:
            references.append({"品种代码": product["代码"], "品种": product["品种"],
                               "关联方式": "名称关键词相关性；不保证交割品质或全国覆盖",
                               **{CHINESE_FIELDS[key]: value for key, value in row.items()}})
    write_csv("product_inventory_plan.csv", plan)
    write_csv("inventory_sourcing_checklist.csv", checklist)
    write_csv("product_existing_indicator_links.csv", references)
    counts = Counter(row["indicator_type"] for row in existing)
    identities = {(row["source_file"], row["indicator_id"]) for row in existing}
    workbook_summary = json.loads((DIRECTORY / "workbook_audit.json").read_text(encoding="utf-8"))
    lines = ["# 中国商品期货关键库存目录（中文）", "", "核查日期：2026-10-03。",
             "", "本目录用于向 iFind / MySteel 确认并采购数据。已有字段来自实际读取的三个工作簿；建议检索概念是研究清单，不能视作已确认可订阅的精确指标。",
             "", "阅读顺序：先看[口径、优先补数和供应商要求](research_notes_zh.md)，再查下表；每个品种的精确已有字段在后面的证据索引中。",
             "", "## 工作簿核查", "", "|工作簿|全部指标列|库存相关候选列|", "|---|---:|---:|"]
    for name, summary in workbook_summary.items():
        lines.append(f"|{name}|{sum(x['indicators'] for x in summary['sheets'].values())}|{sum(x['inventory'] for x in summary['sheets'].values())}|")
    lines += ["", f"共 {len(existing)} 个候选列，按工作簿和指标ID去重为 {len(identities)} 个来源限定字段。列数包含同ID重复列，不等于独立经济概念数；iFind和MySteel同源字段也可能重复。", "",
              "|类型|列数|", "|---|---:|"]
    lines += [f"|{key}|{value}|" for key, value in counts.items()]
    lines += ["", "所有候选列均检出至少一个缓存数值；不代表完整历史连续或数值质量已检验。库存关键词提取会保留误命中，已将仓单价格、未执行合同等单列为非库存。",
              "", f"研究范围共 {len(products)} 个品种/例外条目。生成 {len(checklist)} 条产业概念检索项；上游下游存在共享概念，采购时应去重。",
              "", "## 逐品种采集清单", "",
              "下表‘已有’是本品匹配词检出的相关字段：库存包括数量、天数、库容比；仓单另计。匹配不保证品质、地区、总量或子项与推荐概念完全一致。上下游可在原始核查CSV检索。",
              ""]
    for sector in dict.fromkeys(row["板块"] for row in products):
        lines += [f"### {sector}", "", "|品种|本品重点库存|上游|下游或替代|渠道建议|已有相关列（库存/仓单）|优先补充与口径|", "|---|---|---|---|---|---|---|"]
        for row in plan:
            if row["板块"] == sector:
                values = [f"[{row['品种']} {row['代码']}](#{row['代码'].lower()})", row["本品重点库存"],
                          row["上游重点库存"], row["下游或替代品库存"], row["建议查询渠道"],
                          f"{row['本品相关库存字段列数']}/{row['相关仓单字段列数']}", row["优先补充及口径提醒"]]
                lines.append("|" + "|".join(escape(value) for value in values) + "|")
        lines.append("")
    lines += ["## 旧观测待核查", "", "以下是距核查日超过90日的候选列。这个日期筛选只提示核查，不能自动判定停更；存栏、季节统计、仓单零库存和原始频率尤其需要确认。", "",
              "|原始名称|ID|最后数值日期|元数据频率|来源与位置|", "|---|---|---|---|---|"]
    old = [row for row in existing if (AS_OF - dt.date.fromisoformat(row["last_observation"])).days > 90]
    for row in old:
        lines.append("|" + "|".join(escape(value) for value in (row["indicator"], row["indicator_id"], row["last_observation"], row["frequency"], f"{row['source_file']}/{row['sheet']}/列{row['column']}")) + "|")
    lines += ["", "## 已有字段证据索引", "", "同一字段可用于多个产业链，以下允许重复列示；不能将重复条目加总。LME注销仓单与总库存、COMEX注册与未注册库存须按层级解释。",
              "", "原始字段名称及ID保持不变；观察日期为缓存数据日期，字段的原始频率和原始来源以核查CSV为准。", ""]
    for product in products:
        lines += [f"<a id=\"{product['代码'].lower()}\"></a>", f"### {product['品种']} {product['代码']}", ""]
        rows = [row for row in references if row["品种代码"] == product["代码"]]
        if not rows:
            lines += ["本品关键词未命中三个工作簿中的库存候选字段；不等于供应商不存在该数据。上下游推荐见主表。", ""]
            continue
        lines += ["|原始指标名称|ID|类型/单位|最后数值日期|工作簿/工作表/列|", "|---|---|---|---|---|"]
        for row in rows:
            values = (row["原始指标名称"], row["供应商指标ID"], f"{row['指标类型']}/{row['原始单位']}", row["最后有效数值日期"], f"{row['工作簿']}/{row['工作表']}/{row['列号']}")
            lines.append("|" + "|".join(escape(value) for value in values) + "|")
        lines.append("")
    (DIRECTORY / "inventory_research_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"products": len(products), "sourcing_concepts": len(checklist),
                      "candidate_columns": len(existing), "source_qualified_fields": len(identities),
                      "type_counts": counts, "old_observations_to_review": len(old)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
