# 镍链数据清单（按来源分层）

本清单用于镍链（Ni/NPI/MHP/不锈钢）三模型（Model1/Model2/Model3）数据选型与映射。

## 一、iFind/SMM（销售、库存、成本代理）

| 用途 | 指标名称（完整） | ID | 是否直接进入利润 |
|---|---|---|---|
| 销售收入 | 镍SMM1#现货均价 | S002981540 | ★★★★★ |
| 中间品库存 | MHP港口库存 | S020207789 | ★★★ |
| 成本代理 | 304废料价格（无锡） | S002959172 | ★★★ |
| 终端约束 | 304社会库存 | S006161096 | ★★ |
| 国际价格 | LME镍3M收盘 | S005808364 | ★★★★ |

## 二、Wind（能耗与化学品）

| 用途 | 完整名称 | Wind ID |
|---|---|---|
| 电力成本 | 工业用电价（地区） | S5443317 |
| 燃料成本 | LNG市场价（全国） | S5914475 |
| 化学品成本 | 烧碱出厂价(中间价):烧碱(32%离子膜):河南 | S5438501 |

## 三、Reuters/SMM（国际链条与中间品）

| 用途 | 指标名称 | ID |
|---|---|---|
| 中间品价格 | NPI价格（需SMM映射） | TODO_SMM_NPI |
| 中间品价格 | MHP价格（需SMM映射） | TODO_SMM_MHP |
| 国际定价 | 镍近月合约 | TODO_REUTERS_NI_NEAR |
| 运费参数 | CIF/FOB相关运费序列 | TODO_REUTERS_FRT |

## 四、三模型最小可用数据集合（建议）

### Model1（交易）
- Ni价（或SS代理价）
- NPI/矿成本代理
- 公式: Margin = Ni Price - Feed Cost (NPI/Ore)
- 评级: ★★★★★

### Model2（研究）
- 在Model1基础上增加:
  - MHP/NPI价格或库存模块
  - Energy/化学品/运费
  - 终端不锈钢库存约束
- 公式: Margin = Product + Byproduct - Feed - Energy - Chemical - Freight
- 评级: ★★★★★

### Model3（Sell-side）
- 对齐 02. Cost & Price Analysis - New.xlsx 全量参数（20+变量）
- 评级: ★★★★

## 五、镍链关键衍生因子（优先实现）

| 因子 | 意义 | 重要性 |
|---|---|---|
| Ore -> NPI Margin | RKEF利润 | ★★★★★ |
| NPI -> SS Margin | 304利润 | ★★★★★ |
| NPI -> Matte Margin | 304 vs Battery切换 | ★★★★★ |
| Matte -> Sulphate Margin | 硫酸镍利润 | ★★★★☆ |
| MHP -> Sulphate Margin | HPAL利润 | ★★★★☆ |
| Ni Futures - NPI | Class1-Class2价差 | ★★★★★ |
| SS Futures - NPI Cost | 不锈钢利润 | ★★★★★ |
| Ni Futures - SS Futures | Ni-SS价差 | ★★★★★ |
| LME - SHFE | 进口窗口 | ★★★★☆ |
| Nickel Sulphate - SHFE Ni | 电池溢价 | ★★★★☆ |

说明:

- 以上10个因子作为镍链第一优先级因子池，建议先实现日频计算。
- 其中五星因子优先进入Model1交易监控，四星因子优先进入Model2解释框架。
