# 铜冶炼数据清单（按来源分层）

本清单用于铜冶炼三模型（Model1/Model2/Model3）数据选型与映射。

## 一、iFind/SMM（核心价格与成本）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | LME铜3M收盘 | S005808359 | ★★★★★ |
| 销售收入 | SMM 1#电解铜现货均价 | S002981535 | ★★★★★ |
| 原料成本 | 铜精矿TC | S004630824 | ★★★★★ |
| 汇率折算 | 美元兑人民币即期 | M004147023 | ★★★★★ |
| 现货基差 | 铜现货升贴水 | S003048722 | ★★★ |

## 二、Wind（能耗与辅料）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 电力成本 | 工业用电价（地区） | S5443317 |
| 燃料成本 | LNG市场价（全国） | S5914475 |
| 化学品成本 | 纯碱现货价:轻质纯碱:国内 | S5470435 |
| 资金参数 | 贷款利率LPR一年期 | M0096870 |

## 三、Reuters（国际端定价/期限结构）

| 用途 | 指标名称 | Reuters ID |
|---|---|---|
| 国际铜价 | LME铜近月 | MCUc1 |
| 国际铜价 | LME铜远月 | MCUc2/MCUc5 |
| 汇率 | 离岸人民币 | CNY= |
| 利率 | 美元LIBOR3M | USDLIBOR3M= |

## 四、三模型最小可用数据集合（建议）

### Model1（交易）
- Cu价（LME3M或SMM现货）
- 铜精矿成本（或TC代理）
- 公式A: Margin = Copper Price - Concentrate Cost
- 公式B: Margin = TC + Copper Price

### Model2（研究）
- 在Model1基础上增加:
  - Gold收益
  - Silver收益
  - Sulfuric Acid收益
  - Smelting Cost
- 公式: Margin = Copper + Gold + Silver + Sulfuric Acid - Concentrate - Smelting Cost

### Model3（Sell-side）
- 对齐 Copper smelter cost economics-Latest.xlsx 全量参数（30+变量）
