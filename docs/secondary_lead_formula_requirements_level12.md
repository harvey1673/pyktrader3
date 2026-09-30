# Secondary Lead Margin Formula Requirements (Level1/Level2)

来源: 通过追踪 Excel 公式得到，非人工主观筛选。

- 工作簿: 1_NEW secondary lead smelting margin.xlsx
- 输出CSV: secondary_lead_formula_requirements_level12.csv
- Level1 条数: 10
- Level2 条数: 12

## 目标公式

- Level1:
  - economics-Henan: AU3, BS3
  - economics-Anhui: AW3
- Level2:
  - economics-Henan: BB3, BD3
  - economics-Anhui: BD3, BI3

## 统一口径公式

### 1) Production Cost

按你的要求统一成:

- Production Cost = sum(输入原料价格 × 权重) + 税费 + 人工 + 融资

在表内对应为:

- Henan Level1 (AU3):
  - AU3 = SUM(AM3:AP3) - (AF3 + AI3 + AH3 + AJ3) - AT3
  - 因此成本项为 C_henan_L1 = AF3 + AI3 + AH3 + AJ3 + AT3

- Anhui Level1 (AW3):
  - AW3 = SUM(AO3:AR3) - (AF3 + AI3 + AH3 + AL3) - AV3 - AJ3 - AK3
  - 因此成本项为 C_anhui_L1 = AF3 + AI3 + AH3 + AL3 + AV3 + AJ3 + AK3

- Henan Level2 (BB3):
  - BB3 = SUM(AO3:AP3,AX3) - (SUM(Z3:AE3) + AI3 + AH3 + AJ3) - BA3
  - 成本项为 C_henan_L2 = SUM(Z3:AE3) + AI3 + AH3 + AJ3 + BA3

- Anhui Level2 (BD3):
  - BD3 = SUM(AQ3:AR3,AZ3) - (SUM(Z3:AE3) + AI3 + AH3 + AL3) - BC3 - AJ3 - AK3
  - 成本项为 C_anhui_L2 = SUM(Z3:AE3) + AI3 + AH3 + AL3 + BC3 + AJ3 + AK3

### 2) Revenue

统一成:

- Revenue = sum(产出价格 × 权重)

在表内对应为:

- Henan 主收入 (AM3:AP3)
  - AM3 = G3*R2 + K3*R3
  - AN3 = V3*R4
  - AO3 = S3*R6
  - AP3 = ((T3+U3)/2)*R7
  - Revenue_henan = AM3 + AN3 + AO3 + AP3

- Anhui 主收入 (AO3:AR3)
  - AO3 = F3*AF3 + J3*AF4 + F3*AF2 = F3*(AF2+AF3) + J3*AF4
  - AP3 = 常数项(Assumptions!AH15)
  - AQ3 = T3*AF16*0.3
  - AR3 = ((U3+V3)/2)*AF16*0.7
  - Revenue_anhui = AO3 + AP3 + AQ3 + AR3

### 3) 权重参数(当前缓存值)

- Henan:
  - R2 = 0.7246903989856514
  - R3 = 0.2753096010143486
  - R4 = 0.20667094817463436
  - R6 = 0.04864894960010864
  - R7 = 0.11351421573358682

- Anhui:
  - AF2 = 0.25
  - AF3 = 0.5
  - AF4 = 0.25
  - AF16 = 0.25758333333333333

## 清单文件

- secondary_lead_formula_requirements_level12.csv: 公式依赖的去重后数据项(按Level+目标公式)
- secondary_lead_formula_cost_revenue_checklist.csv: 按成本/收入拆分的可执行清单
