# 氧化铝数据清单（按来源分层）

本清单用于氧化铝三模型（Model1/Model2/Model3）数据选型与映射。

## 一、iFind/SMM（收入与核心成本）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | 氧化铝现货价（待映射） | TODO_AO_SPOT | ★★★★★ |
| 原料成本 | 铝土矿价格（待映射） | TODO_BAUXITE_AO | ★★★★★ |
| 原料成本 | 烧碱价格 | S5438501 | ★★★★ |

## 二、Wind（能耗与加工）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 化学品成本 | 烧碱出厂价(中间价):烧碱(32%离子膜):河南 | S5438501 |
| 能源成本 | LNG市场价（全国） | S5914475 |
| 精炼成本 | Refining Cost（待映射） | TODO_REFINING_COST_AO |

## 三、三模型最小可用数据集合（建议）

### Model1（交易）
- 公式: Margin = Alumina Price - Bauxite Cost - Caustic Soda Cost
- 评级: ★★★★★

### Model2（研究）
- 公式: Margin = Alumina - Bauxite - Caustic Soda - Energy - Refining Cost
- 评级: ★★★★★

### Model3（Sell-side）
- Excel-style full refining model（20+变量）
- 评级: ★★★★
