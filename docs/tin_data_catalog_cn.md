# 锡冶炼数据清单（按来源分层）

本清单用于锡冶炼三模型（Model1/Model2/Model3）数据选型与映射。

## 一、iFind/SMM（收入与核心成本）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | SMM 1#锡现货均价 | S002981539 | ★★★★★ |
| 国际价格 | LME锡3M收盘 | S005808363 | ★★★★ |
| 原料成本 | 锡精矿TC(60品位江西) | S009620177 | ★★★★ |
| 原料成本 | 锡精矿TC(60品位广西) | S009620198 | ★★★★ |

## 二、Wind（能耗与加工）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 电力成本 | 工业用电价（地区） | S5443317 |
| 能源成本 | LNG市场价（全国） | S5914475 |
| 冶炼成本 | Smelting Cost（待映射） | TODO_SMELTING_COST_SN |

## 三、三模型最小可用数据集合（建议）

### Model1（交易）
- 公式A: Margin = Tin Price - Tin Concentrate Cost
- 公式B: Margin = TC + Tin Price
- 评级: ★★★★★

### Model2（研究）
- 公式: Margin = Tin + Byproduct - Concentrate - Smelting Cost - Energy
- 评级: ★★★★★

### Model3（Sell-side）
- Excel-style full tin smelter model（20+变量）
- 评级: ★★★★
