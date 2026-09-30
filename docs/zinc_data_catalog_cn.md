# 锌冶炼数据清单（按来源分层）

本清单用于锌冶炼三模型（Model1/Model2/Model3）数据选型与映射。

## 一、iFind/SMM（收入、原料、TC）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | SMM0#锌现货均价 | S002865578 | ★★★★★ |
| 原料成本 | 锌精矿TC(48品位港口) | S016702562 | ★★★★★ |
| 原料成本 | 锌精矿TC(50品位河南) | S016702553 | ★★★★ |
| 原料成本 | 锌精矿TC(50品位云南) | S016702544 | ★★★★ |
| 原料成本 | 锌精矿TC(50品位湖南) | S016702547 | ★★★★ |

## 二、Wind（能耗与化学品）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 电力成本 | 工业用电价（地区） | S5443317 |
| 燃料成本 | LNG市场价（全国） | S5914475 |
| 化学品成本 | 纯碱现货价:轻质纯碱:国内 | S5470435 |
| 化学品/副产参考 | 硫酸出厂价:98%硫酸:河南豫光金铅 | S5442694 |

## 三、Reuters（国际定价/套利）

| 用途 | 指标名称 | Reuters ID |
|---|---|---|
| 进口套利 | 锌近月 | CMZN0 |
| 进口套利 | 锌远月 | CMZN3 |
| 汇率 | 离岸人民币 | CNH= |

## 四、三模型最小可用数据集合（建议）

### Model1（交易）
- Zn价
- Zn精矿成本（或TC代理）
- 公式A: Margin = Zn Price - Concentrate Cost
- 公式B: Margin = TC + Zn Price
- 评级: ★★★★★

### Model2（研究）
- 在Model1基础上增加:
  - Sulfuric Acid收益
  - Byproduct收益
  - Smelting Cost
- 公式: Margin = Zn + Sulfuric Acid + Byproduct - Concentrate - Smelting Cost
- 评级: ★★★★★

### Model3（Sell-side）
- 对齐 new cost economics file - Copy.xlsx 全量参数（20+变量）
- 评级: ★★★★
