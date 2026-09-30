# Base Metal iFind Mapping Status

## Purpose
Track link status between:

1. Excel raw fields from xldata/base (SMM/WIND/REUTERS/assumption sheets)
2. iFind code map in tests/index_map_full.py
3. Model data requirements in docs/base_metal_data_requirements_cn.csv

## Files In Workflow

1. Input workbooks:
- xldata/base/02. Cost & Price Analysis - New.xlsx
- xldata/base/1_NEW secondary lead smelting margin.xlsx
- xldata/base/Copper smelter cost economics-Latest.xlsx
- xldata/base/Primary smelting margin.xlsx
- xldata/base/new cost economics file - Copy.xlsx

2. Mapping assets:
- tests/index_map_full.py

3. Generated inventory:
- docs/base_metal_excel_code_inventory.csv

4. Requirement catalog:
- docs/base_metal_data_requirements_cn.csv

## Mapping Rules

1. 已映射
- Excel code is iFind style and exists in index_map_full.py.

2. 待映射
- Excel code exists but missing in index_map_full.py.
- Or code exists but alias is unclear and needs manual naming policy.

3. 非iFind
- Reuters ticker or free-text field that should be handled by separate ETL.

## Priority Rules For Iteration

1. First priority: Model 1 (trading) high-impact fields
- 铜: LME3M, 铜TC, SMM铜现货, USDCNY
- 铅: SMM铅现货, 铅精矿TC, 再生铅现货
- 锌: SMM锌现货, 锌精矿TC, 硫酸
- 镍: 镍现货/NPI/MHP关键价格与库存

2. Second priority: Model 2 (research) high-impact fields
- 税负、副产品收益、区域参数、品位回收率

3. Third priority: Model 3 (sell-side) explanatory/auxiliary fields
- 利率、库存结构、期限结构等

## Recommended Update Routine

1. Run extractor:
- D:/miniconda3/python.exe tests/extract_base_metal_excel_map.py

2. Review docs/base_metal_excel_code_inventory.csv
- Filter mapped_alias empty rows.
- Mark whether they are iFind / Reuters / manual inputs.

3. Update docs/base_metal_data_requirements_cn.csv
- Fill data ID and price spec.
- Update importance and model usage.

4. Add missing high-priority iFind codes into tests/index_map_full.py
- Keep alias naming consistent with existing conventions.

## Current Notes

- Existing index_map_full.py already contains many base-metal series
  (cu/pb/zn/ni/sn/al, inventories, spreads, TC, basis).
- Workbook-specific codes in WIND/SMM/REUTERS sheets still need systematic
  normalization before final ETL and Python model implementation.

## Snapshot (2026-07-11)

- Inventory rows: 288
- By type:
  - reuters_ticker: 152
  - ifind_code: 81
  - smm_id: 55
- Current mapped_alias hit in tests/index_map_full.py: 0

Interpretation:

- The five Excel files mostly store vendor-native IDs (SMM/Wind/Reuters) that
  are not yet normalized to current index_map_full aliases.
- Next action is not direct alias lookup only. A translation layer is needed:
  Excel vendor ID -> canonical factor alias -> iFind/index map (if available).
