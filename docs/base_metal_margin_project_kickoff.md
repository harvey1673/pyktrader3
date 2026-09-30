# Base Metal Margin/Spread Project Kickoff

## Goal
Build three model layers for base-metal smelting margin and product spread:

1. Model 1 (Trading): daily update, 2 variables, most robust for signal usage.
2. Model 2 (Research): around 7 variables, strong explanatory power.
3. Model 3 (Sell-side): Excel-equivalent full model with 20+ variables.

This document is the step-1 to step-4 working baseline before Python
production implementation.

## Scope Of The 5 Excel Workbooks (xldata/base)

1. 02. Cost & Price Analysis - New.xlsx
- Focus: nickel chain and stainless conversion (ore -> NPI/MHP/matte -> SS -> sulfate/cathode)
- Typical sheets: daily, ore, power, CN NPI, Indo NPI, Matte, SS, Sulphate, Cathode
- Modeling value: nickel cost transmission and class-I/class-II spread logic

2. 1_NEW secondary lead smelting margin.xlsx
- Focus: secondary lead economics (battery scrap route)
- Typical sheets: Assumptions, economics-Anhui, economics-Henan, wind, SMM
- Modeling value: secondary lead margin and policy/tax sensitivity

3. Copper smelter cost economics-Latest.xlsx
- Focus: copper smelter spot and tolling economics
- Typical sheets: Spot (China), Tolling, Smelter Cost, Economics, Scrap
- Modeling value: TC/RC-led smelter margin framework + by-product credits

4. Primary smelting margin.xlsx
- Focus: primary lead smelting and tolling variants (including silver/acid/by-product)
- Typical sheets: Margin, Ag-tolling margin, CIF low/high Ag margin, TCRC, MB
- Modeling value: primary lead route decomposition and Ag-linked earnings effects

5. new cost economics file - Copy.xlsx
- Focus: zinc cost economics and conversion chain
- Typical sheets: Calculation, SMM, REUTERS, WIND, zn cif arb
- Modeling value: zinc margin with conc/tc/power/acid/by-product relationship

## Three-Model Design (Step-1 Updated)

This three-model requirement applies to all target base metals in this project,
not only secondary lead. Current scope metals:

- Copper (Cu)
- Primary Lead (Pb-primary)
- Secondary Lead (Pb-secondary)
- Zinc (Zn)
- Nickel chain (Ni/NPI/MHP/SS)
- Aluminum (Al)
- Alumina (AO)
- Tin (Sn)

### Model 1 (Trading Model)
Use for daily signal and fast monitoring. This is the recommended production
signal model.

- Formula (secondary lead):
	- Margin = Pb - 1.62 * Battery
- Update frequency: daily
- Variable count: 2
- Expected explanatory power: about 85%

Why this model is preferred:

- Very stable and easy to maintain
- Data availability risk is low
- Suitable for daily trading and regime detection

### Model 2 (Research Model)
Use for deeper attribution and regime diagnostics.

- Formula (secondary lead):
	- Margin = Pb + PP + ABS - Battery - Energy - Sb - NaOH
- Update frequency: daily/weekly (depending on data lag)
- Variable count: around 7 variables
- Expected explanatory power: about 95%

Role in workflow:

- Explain margin change sources by component
- Support parameter review for Model 1
- Bridge between trading model and sell-side full workbook logic

### Model 3 (Sell-side Model)
Use for full reconciliation with analyst workbook.

- Formula style: Excel-equivalent with 20+ variables and detailed assumptions
- Expected explanatory power: about 98% (but practical maintenance score: ★★★★)
- Limitation: very high maintenance cost and uneven data update quality

Recommended use:

- Monthly/quarterly cross-check only
- Do not use as first-line daily trading model

### Cross-Metal Extension Template
Cross-metal implementation standard (mandatory for each metal):

1. Model 1: 2-3 key variables + fixed processing term
2. Model 2: 6-10 variables with key by-products and energy terms
3. Model 3: workbook-level detailed economics for reconciliation

## Per-Metal Model Definitions (Unified Standard)

### Copper (Cu)
- Model 1:
	- Margin_Cu_M1 = Copper Price - Concentrate Cost
	- or Margin_Cu_M1 = TC + Copper Price
- Model 2:
	- Margin_Cu_M2 = Copper + Gold + Silver + Sulfuric Acid - Concentrate - Smelting Cost
- Model 3:
	- Full workbook economics from Copper smelter cost economics-Latest.xlsx (30+ variables)

### Primary Lead (Pb-primary)
- Model 1:
	- Margin = Pb Price - Concentrate Cost
	- or Margin = TC + Pb Price
- Model 2:
	- Margin = Pb + Silver + Sulfuric Acid - Concentrate - Smelting Cost
- Model 3:
	- Full workbook economics from Primary smelting margin.xlsx (20+ variables)

### Secondary Lead (Pb-secondary)
- Model 1:
	- Margin = Pb - 1.62 * Battery
- Model 2:
	- Margin = Pb + PP + ABS - Battery - Energy - Sb - NaOH
- Model 3:
	- Full workbook economics from 1_NEW secondary lead smelting margin.xlsx (20+ variables)

### Zinc (Zn)
- Model 1:
	- Margin = Zn Price - Concentrate Cost
	- or Margin = TC + Zn Price
- Model 2:
	- Margin = Zn + Sulfuric Acid + Byproduct - Concentrate - Smelting Cost
- Model 3:
	- Full workbook economics from new cost economics file - Copy.xlsx (20+ variables)

### Nickel chain (Ni/NPI/MHP/SS)
- Model 1:
	- Margin = Ni Price - Feed Cost (NPI/Ore)
- Model 2:
	- Margin = Product + Byproduct - Feed - Energy - Chemical - Freight
- Model 3:
	- Full workbook economics from 02. Cost & Price Analysis - New.xlsx (20+ variables)

### Aluminum (Al)
- Model 1:
	- Margin = Aluminum Price - Alumina Cost - Power Cost
- Model 2:
	- Margin = Aluminum + Byproduct - Alumina - Power - Carbon - Smelting Cost
- Model 3:
	- Excel-style full chain model (25+ variables)

### Alumina (AO)
- Model 1:
	- Margin = Alumina Price - Bauxite Cost - Caustic Soda Cost
- Model 2:
	- Margin = Alumina - Bauxite - Caustic Soda - Energy - Refining Cost
- Model 3:
	- Excel-style full refining model (20+ variables)

### Tin (Sn)
- Model 1:
	- Margin = Tin Price - Tin Concentrate Cost
	- or Margin = TC + Tin Price
- Model 2:
	- Margin = Tin + Byproduct - Concentrate - Smelting Cost - Energy
- Model 3:
	- Excel-style full tin smelter model (20+ variables)

## Step-2 Data Dictionary Standard (Chinese)

For each required dataset, use these fields:

- 指标中文名
- 指标说明
- 数据源 (SMM/Mysteel/Reuters/Wind/Bloomberg/iFind)
- 数据ID
- 价格位置与规格 (地区, 含税/不含税, 品位, 单位)
- 适用模型 (如: 铜冶炼毛利-简化)
- 重要性 (高/中/低; 对毛利波动影响)
- 备注 (是否已在index_map_full映射, 是否需手工补录)

See the first-pass catalog in docs/base_metal_data_requirements_cn.csv.

## Step-3 Knowledge Base Output In docs/

Recommended doc set:

1. docs/base_metal_margin_project_kickoff.md (this file)
2. docs/base_metal_data_requirements_cn.csv (data dictionary)
3. docs/base_metal_excel_code_inventory.csv (excel extracted IDs/tickers)
4. docs/base_metal_ifind_mapping_status.md (mapping progress/gaps)
5. docs/secondary_lead_data_catalog_cn.md (再生铅来源分层清单)
6. docs/secondary_lead_data_catalog_cn.csv (再生铅结构化数据清单)
7. docs/copper_data_catalog_cn.md
8. docs/copper_data_catalog_cn.csv
9. docs/primary_lead_data_catalog_cn.md
10. docs/primary_lead_data_catalog_cn.csv
11. docs/zinc_data_catalog_cn.md
12. docs/zinc_data_catalog_cn.csv
13. docs/nickel_chain_data_catalog_cn.md
14. docs/nickel_chain_data_catalog_cn.csv
15. docs/base_metal_data_catalog_master.csv (跨品种总汇总)
16. docs/base_metal_data_catalog_master.md (总汇总标记规则)
17. docs/nickel_chain_factor_priority.csv (镍链关键因子优先级)
18. docs/aluminum_data_catalog_cn.md
19. docs/aluminum_data_catalog_cn.csv
20. docs/alumina_data_catalog_cn.md
21. docs/alumina_data_catalog_cn.csv
22. docs/tin_data_catalog_cn.md
23. docs/tin_data_catalog_cn.csv
24. docs/base_metal_execution_queue_model1_high_unmapped.csv
25. docs/base_metal_minimum_runnable_fields.csv
26. docs/base_metal_execution_queue_summary.md

## Step-4 Link To iFind index_map_full.py

Current map file:
- tests/index_map_full.py

Execution logic:

1. Extract candidate IDs/tickers from all 5 Excel files.
2. Match IDs against tests/index_map_full.py.
3. Mark rows as:
- 已映射: has alias in index_map_full
- 待映射: exists in excel but not mapped
- 非iFind: Reuters ticker or local calc field
4. Prioritize high-impact "待映射" items first.

## How To Update Inventory

Use:

- D:/miniconda3/python.exe tests/extract_base_metal_excel_map.py

Output:

- docs/base_metal_excel_code_inventory.csv

## Next Iteration Entry Criteria

Proceed to Python implementation only after:

1. Model 1 data coverage >= 95% (must be stable daily).
2. Model 2 high-importance data coverage >= 80%.
3. Model 3 high-impact subset coverage >= 60% (full coverage can be staged).
4. All key IDs have source + spec + owner confirmation.
