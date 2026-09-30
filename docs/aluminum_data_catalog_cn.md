# 铝冶炼数据清单（按来源分层）

本清单用于铝冶炼三模型（Model1/Model2/Model3）数据选型与映射。

## 一、iFind/SMM（收入与核心成本）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | A00铝现货均价 | S002865592 | ★★★★★ |
| 国际价格 | LME铝3M收盘 | S005808360 | ★★★★ |
| 成本代理 | 氧化铝现货价（待映射） | TODO_AO_SPOT | ★★★★★ |
| 成本代理 | 工业电价（地区） | S5443317 | ★★★★★ |

## 二、Wind（能耗与辅料）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 电力成本 | 工业用电价（地区） | S5443317 |
| 碳素成本 | 预焙阳极价格（待映射） | TODO_ANODE_AL |
| 能源成本 | LNG市场价（全国） | S5914475 |

## 三、三模型最小可用数据集合（建议）

### Model1（交易）
- 公式: Margin = Aluminum Price - Alumina Cost - Power Cost
- 评级: ★★★★★

### Model2（研究）
- 公式: Margin = Aluminum + Byproduct - Alumina - Power - Carbon - Smelting Cost
- 评级: ★★★★★

### Model3（Sell-side）
- Excel-style full chain model（25+变量）
- 评级: ★★★★
