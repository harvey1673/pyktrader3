# 原生铅数据清单（按来源分层）

本清单用于原生铅三模型（Model1/Model2/Model3）数据选型与映射。

## 一、SMM（收入、原料、TC）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | SMM 1#铅锭-平均价 | s20017202 | ★★★★★ |
| 原料成本 | Pb50河南TC(周)-平均价 | s20017262 | ★★★★★ |
| 原料成本 | Pb50内蒙TC(周)-平均价 | s20017257 | ★★★★ |
| 原料成本 | Pb50广西TC(周)-平均价 | s20017267 | ★★★★ |
| 原料成本 | Pb50云南TC(周)-平均价 | s20017272 | ★★★★ |
| 原料成本 | Pb50湖南TC(周)-平均价 | s20017277 | ★★★★ |
| 原料成本 | Pb60进口TC(周)-平均价 | s20097246 | ★★★ |

## 二、Wind（能耗与辅料）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 电力成本 | 工业用电价（地区） | S5443317 |
| 燃料成本 | 天然气市场价:液化天然气(LNG):全国 | S5914475 |
| 化学品成本 | 烧碱出厂价(中间价):烧碱(32%离子膜):河南 | S5438501 |
| 化学品成本 | 纯碱现货价:轻质纯碱:国内 | S5470435 |

## 三、Reuters/国际副产收益参考

| 用途 | 指标名称 | Reuters ID |
|---|---|---|
| 副产品收益 | COMEX白银 | COMEX_SI |
| 副产品收益 | LBMA白银 | LBMA_AG |

## 四、三模型最小可用数据集合（建议）

### Model1（交易）
- Pb价
- Pb精矿成本（或TC代理）
- 公式A: Margin = Pb Price - Concentrate Cost
- 公式B: Margin = TC + Pb Price
- 评级: ★★★★★

### Model2（研究）
- 在Model1基础上增加:
  - Silver收益
  - Sulfuric Acid收益
  - Smelting Cost
- 公式: Margin = Pb + Silver + Sulfuric Acid - Concentrate - Smelting Cost
- 评级: ★★★★★

### Model3（Sell-side）
- 对齐 Primary smelting margin.xlsx 全量参数（20+变量）
- 评级: ★★★★
