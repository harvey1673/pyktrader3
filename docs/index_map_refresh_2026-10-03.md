# Indicator map and CTD spot refresh — 2026-10-03

Workbook scope follows the active `file_setup` in `misc_scripts/auto_update_data_xl.py`.

- MySteel: `mysteel_data.xlsx`, sheets `metal` and `petchem`
- iFind spot counterparts: `ifind_daily.xlsx`, sheets `ferrous_d` and `petchem_d`

## Map reconciliation

- `mysteel_ticker_map.py` exactly matches the 111 current workbook IDs.
- The refreshed `metal` sheet adds eight IDs and removes five relative to the immediately preceding map.
- `index_map_full.py` received the iFind spot aliases needed by this CTD update. This was a targeted update, not a full iFind workbook reconciliation.

Removed from the current MySteel workbook:

| Code | Previous name |
|---|---|
| `ID00184490` | 焦煤_31家钢铁企业_市场情绪指数周 |
| `ID00184492` | 焦煤_38家煤矿企业_市场情绪指数周 |
| `ID00184494` | 焦煤_14家贸易商_市场情绪指数周 |
| `ID00184497` | 焦煤_150家独立焦化厂_市场情绪指数周 |
| `ID01017688` | pp_贸易商_库存_中国周 |

The four retired coking-coal sentiment IDs are retained only as inactive comments in `mysteel_index_map` so their production status is explicit.

## Updated spot aliases

### CZCE exchange-code capitalization

Commodity-specific aliases now use the exact uppercase CZCE product code. The source maps, production signal configuration, fundamental-data full-history configuration, and research code all use the canonical names. No legacy alias columns are added by `process_spot_df`.

| Product | Representative canonical changes | Production reference migration |
|---|---|---|
| TA | `pta_east_spot` → `TA_east_spot`; `pta_cfr_cn` → `TA_cfr_cn`; `PTA_margin_cn_d` → `TA_margin_cn_d`; `PTA_invdays_mill` → `TA_invdays_mill` | Carry, margin, import-arbitrage, and mill-inventory signal references now use `TA_*`. |
| PX | `px_exw_east_spot` → `PX_exw_east_spot`; all iFind/MySteel `px_*` aliases now begin `PX_` | Existing `PX_margin_cn_w` was already canonical; CTD and source-price references now use `PX_*`. |
| SM | all `sm_*` source aliases → `SM_*` | Carry, production-cost, and full-history inventory references now use `SM_*`. Composite derived features such as `smsf_dmd_ratio` keep their established names. |
| SF | all `sf_*` source aliases → `SF_*` | Carry, production-cost, and full-history inventory references now use `SF_*`. |
| FG | all `fg_*` source aliases → `FG_*` | Carry, inventory, margin, and utilization signal references now use `FG_*`. |
| SA | all `sa_*` source aliases → `SA_*` | Carry and full-history inventory references now use `SA_*`. |

The same capitalization rule is applied to other CZCE prefixes, including `MA_`, `PF_`, `PR_`, `UR_`, `RM_`, `OI_`, `CF_`, `CJ_`, `AP_`, `SR_`, `PK_`, and `SH_`.

| Alias | Source ID | Worksheet | Current use |
|---|---|---|---|
| `ssf_qd` | MySteel `ID00104205` | `metal` | Replaces iFind `S004317129` for iron-ore spreads. The overlap correlation is 0.999999 and the median difference is zero. |
| `coke_sub_a_rz_outstock` | iFind `S004626322` | `ferrous_d` | First-priority J research CTD quote; Rizhao port outstock, cash and tax included. |
| `coke_sub_a_rz_haoyu` | MySteel `ID00102802` | `metal` | J comparison quote. Ex-factory basis and incomplete strength metadata prevent automatic inclusion in the CTD minimum. |
| `coke_sub_a_dry_lvliang_jinyan` | MySteel `ID01199235` | `metal` | J dry-coke comparison quote. Freight to delivery location is still required. |
| `ckc_mongol5_ts` | MySteel `ID01892939` | `metal` | Included in the JM research minimum using `Y=14` as an explicit proxy assumption because Y is absent from the header. |
| `ckc_midsulfur_jiexiu_kaijia` | MySteel `RE00035725` | `metal` | Included in the JM research minimum, including the dated Kaijia brand overlay already encoded in the rule engine. |
| `SF_72_tj` | MySteel `ID00258827`; iFind `S016192874` | `metal`; `ferrous_d` | Direct Tianjin SF72 quote. MySteel is authoritative and the national quote remains a fallback. |
| `ss_304_2b_hongwang_wuxi` | MySteel `ID00003727`; iFind `S006145828` | `metal`; `ferrous_d` | Market comparison only. The stated 1240 mm width is not one of the exchange delivery widths, so it cannot win the CTD minimum. |
| `pvc_cac2_tj` | MySteel `ID00399689` | `metal` | PVC regional spot input; not added to the current J/JM/SS/SM/SF CTD defaults. |
| `pvc_cac2_sh` | MySteel `ID00399698` | `metal` | PVC regional spot input; not added to the current J/JM/SS/SM/SF CTD defaults. |

## Existing code changes reviewed

- `ssf_qd`: the MySteel replacement is valid and improves coverage from 2017-06-26 to 2012-06-07 while extending the latest date from 2026-07-23 to 2026-09-30.
- `TA_cfr_cn`: `S016571550` is the fresher current quote. `S005594160` is retained as `TA_cfr_cn_long`; it exactly matches the retired `S002863167` over their 2,646-date overlap and provides current-workbook history from 2016-06-01.

The CTD implementation remains under `tests`. Production mapping changes are limited to the explicitly requested spot aliases and comments.

## Physical-carry bridge and preliminary backtest

`tests/ctd_phycarry_adapter.py` adds `j/jm/ss/SM/SF_ctd_spot` and then writes the existing `<asset>_phycarry` interface used by `bt_signal - metal summary.ipynb`. The adapter applies no extra fixed adder after CTD normalization. This matters for SM and SF: passing normalized spots through the current legacy `+190/+350` path would double count the delivery-location adjustment.

The notebook can apply the research overlay after its current `spot_dict` block:

```python
from tests.ctd_phycarry_adapter import add_priority_ctd_phycarry

spot_df = add_priority_ctd_phycarry(
    df,
    spot_df,
    products=["j", "jm", "ss", "SM", "SF"],
)

feature_setup["ctd_phycarry_ema"] = [
    ["j", "jm", "ss", "SM", "SF"],
    ["phycarry", "ema", [5, 10], "", "", True, "price", "", 120, [-2, 2]],
]
```

This also works with the faster recipe `["phycarry", "ema", [1, 2, 1], "ema1", "", True, "", "", 60, [-2.5, 2.5]]`. The existing `metal_pbc_ema` recipe can consume the overwritten columns for assets already in its universe, but the separate five-asset recipe is cleaner for measuring the CTD change.

The focused runner is `tests/run_ctd_phycarry_backtest.py`. It refreshes the new CTD inputs directly from MySQL, uses the cached notebook futures and spot frames through 2026-09-18, starts performance at 2016-01-01, and masks current and CTD series to common per-asset availability. The execution return, 20-day volatility scaling, one-day signal lag plus notebook `shift_holdings=1`, and 2 bp trading cost follow the notebook.

Net-of-2-bp portfolio results for the five-asset basket:

| Recipe | Spot method | Full Sharpe | 1y | 3y | 5y | Daily std |
|---|---:|---:|---:|---:|---:|---:|
| EMA 5–9 | current | 0.957 | 0.456 | 0.670 | 0.647 | 2.232 |
| EMA 5–9 | CTD | 0.716 | 1.121 | 0.480 | 0.452 | 2.353 |
| EMA 1 | current | 0.761 | 0.351 | 0.615 | 0.620 | 2.403 |
| EMA 1 | CTD | 0.525 | 0.901 | 0.342 | 0.373 | 2.496 |

`[5, 10]` follows Python `range(5, 10)`, so the first recipe averages EMA windows 5 through 9.

For the EMA 5–9 recipe, full-sample / one-year net Sharpes by asset were:

| Asset | Current full | CTD full | Current 1y | CTD 1y |
|---|---:|---:|---:|---:|
| J | 0.737 | 0.732 | -0.320 | 1.399 |
| JM | 0.644 | 0.461 | 0.249 | 0.914 |
| SS | 0.332 | 0.292 | -0.048 | -0.004 |
| SM | 0.372 | 0.223 | -0.052 | -0.057 |
| SF | 0.743 | 0.413 | 1.892 | 1.385 |

The CTD construction is therefore research-ready but not ready to replace the current physical-carry inputs. It improves the recent J and JM behavior substantially, while the full-history basket remains weaker, especially for JM, SM and SF. The next review should compare each candidate and rule-period transition, then test a stable priority quote against the cross-candidate minimum before any production move.

## Extended industrial and petrochemical group

The same test-side interface now builds domestic-CNY delivery-basis proxies for:

| Product | Default research input | Earliest cached CTD observation |
|---|---|---:|
| L | `l_7042_tj` | 2009-01-05 |
| PP | `pp_t30s_shaoxing_hz` | 2014-03-03 |
| V | Minimum of `pvc_cac2_sh` and `pvc_cac2_east` | 2009-05-26 |
| EG | `eg_east_spot` | 2018-12-11 |
| EB | `eb_east_spot` | 2019-09-27 |
| TA | `TA_east_spot` | 2009-01-05 |
| PX | `PX_exw_east_spot` | 2023-09-15 |
| MA | `MA_spot_jiangsu` | 2012-01-04 |
| UR | `UR_shandong_spot` | 2019-11-20 |
| RU | `ru_scrwf_jiangsu` | 2015-06-29 |
| BU | `bu_heavy_shandong` | 2015-06-02 |
| PG | `propane_cfr_south_cny_vat` | 2020-03-31 |

These are deliberately named and treated as sparse research proxies. PP, V, TA, BU and PG still need dated registered-brand verification; EG, EB, PX and MA still need warehouse/factory-pickup checks. The code does not manufacture premiums where no stable universal adjustment exists.

The October 4 workbook reconciliation retired the old generic `pp_100ppi_spot` and `pg_sd_spot_idx` inputs. The PP test basket now uses the active Hangzhou T30S market quote. The PG proxy converts the active South-China propane CFR quote with daily USD/CNY and 13% VAT; propane/butane composition, duty, port costs and inland freight remain unmodelled.

The matched-history comparison could cover the ten products that already had a current production physical-carry input: L, PP, V, EG, EB, TA, MA, RU, BU and PG. PX and UR had CTD carry output but no current baseline in `commod_phycarry_dict`. The results below predate the October 4 PP/PG remap and are retained as a historical benchmark; they must be rerun before judging the current aliases.

| Recipe | Spot method | Full net Sharpe | 1y | 3y | 5y | Daily std |
|---|---:|---:|---:|---:|---:|---:|
| EMA 5–9 | current | 0.539 | 0.369 | 0.131 | 0.175 | 5.041 |
| EMA 5–9 | CTD proxy | **0.608** | **0.420** | **0.224** | **0.237** | 4.933 |
| EMA 1 | current | 0.534 | 0.461 | 0.232 | 0.174 | 5.302 |
| EMA 1 | CTD proxy | **0.608** | **0.500** | **0.317** | **0.226** | 5.161 |

`NR` and `SC` remain explicit default-basket gaps. BR, FU and LU now have test-side defaults: BR uses certified-producer BR9000 quotes, while FU and LU use USD-denominated marine-fuel proxies converted with daily USD/CNY.

Newly confirmed indicators:

| Alias | Source ID | Exact indicator | Unit | History |
|---|---|---|---|---|
| `br9000_qilu_sd` | MySteel `ID00407482` | 顺丁橡胶：BR9000：市场价：山东：齐鲁石化（日） | 元/吨 | 2010-05-04 onward in MySQL |
| `br9000_yangzi_sh` | MySteel `ID00178979` / iFind `S010795263` | 顺丁橡胶：BR9000：市场价：上海：扬子石化（日） | 元/吨 | 2017-09-15 onward |
| `br9000_daqing_sd` | iFind `S017304605` | 市场估价:顺丁橡胶BR9000:山东:大庆石化:均价 | 元/吨 | 2022-04-24 onward |
| `fo_380cst_sgp_fob` | iFind `S005126414` | 现货价:燃料油(船用380Cst,FOB):新加坡:中间价 | 美元/吨 | 2016-06-01 onward |
| `fo_380cst_zhoushan` | iFind `S006854048` | 市场价:船用油(380CST):舟山 | 美元/吨 | 2019-09-23 onward |
| `lu_05_zhoushan` | iFind `S006854047` | 市场价:船用油(0.5%低硫燃料油):舟山 | 美元/吨 | 2019-09-23 onward |
| `lu_bonded_zhoushan` | iFind `S009395713` | 主流价:燃料油(低硫燃料油,保税船用):舟山 | 美元/吨 | 2021-01-28 onward |

The test-side adapter converts the FU and LU USD/tonne quotes with `usdcnh_spot`, using `usdcny_spot` as a same-date fallback. FU prefers Zhoushan 380CST and falls back to Singapore FOB 380CST; LU prefers Zhoushan bonded 0.5% fuel and falls back to the Zhoushan 0.5% marine-fuel quote. Currency conversion is therefore covered, but bunker-versus-cargo basis, freight, storage, tax treatment and deliverable-quality adjustments remain proxy limitations.

## Full workbook reconciliation — 2026-10-04

The production maps now match the active workbook ID sets rather than only the CTD subset:

- iFind: 1,098 unique current IDs across the active `ifind_daily.xlsx` and `ifind_data.xlsx` worksheets.
- MySteel: 111 current IDs across `mysteel_data.xlsx` worksheets `metal` and `petchem`.
- `spot_idx_map.index_map` and `index_map_full` now have identical iFind code-to-alias mappings.
- Every production-map entry has an inline comment with the exact Chinese indicator name and worksheet.
- MySteel aliases ending in `_ms` intentionally keep overlapping provider series separate for later comparison; the three established shared aliases remain `br9000_yangzi_sh`, `SF_72_tj`, and `ss_304_2b_hongwang_wuxi`.

The newly exposed series include physical and warrant inventory, inventory days, production and utilization, purchase quantities, power and production costs, margins, market sentiment, import profit, international benchmarks, location prices, and product/location premiums.

The reconciled October 1 cache is `C:\dev\data\spot_df_20261001.parquet` with 9,786 rows and 1,428 columns. All 1,206 distinct aliases from the two production source maps are present and have at least one non-null observation; the cache has no duplicate columns. The previous cache is preserved as `C:\dev\data\spot_df_20261001.pre_full_map.parquet`.

Compared with repository `HEAD`, the following 77 iFind IDs were removed from the production map because they are not present in the active Excel worksheets:

| Removed ID | Previous alias |
|---|---|
| `G002600791` | `libor3m` |
| `G005326174` | `citi_eco_surprise_idx_us` |
| `G005432431` | `citi_eco_surprise_idx_cn` |
| `G005432432` | `citi_eco_surprise_idx_eu` |
| `G005432436` | `citi_eco_surprise_idx_global` |
| `G005432438` | `citi_eco_surprise_idx_em` |
| `G005432439` | `citi_eco_surprise_idx_asia` |
| `L015211333` | `cnh_hibor_1m` |
| `M002816452` | `shibor_3m` |
| `M002816455` | `shibor_1y` |
| `S002825721` | `pvc_cac2_central` |
| `S002825730` | `pvc_ethylene_south` |
| `S002835961` | `px_taiwan_cfr_usd` |
| `S002836798` | `l_7042_south` |
| `S002863167` | `TA_cfr_cn` |
| `S002893910` | `eg_north_exw` |
| `S002959495` | `sm_65s17_guangxi` |
| `S002983448` | `pci_jincheng` |
| `S003011283` | `fo_180cst_east` |
| `S003011289` | `fo_180cst_sh` |
| `S003011302` | `fo_180cst_xiamen` |
| `S003011318` | `propane_cfr_asia_n` |
| `S003011327` | `propane_cfr_china_s` |
| `S003011336` | `propane_cfr_tw` |
| `S003011351` | `butane_cfr_asia_n` |
| `S003011360` | `butane_cfr_china_s` |
| `S003011369` | `butane_cfr_tw` |
| `S003994603` | `eg_south_spot` |
| `S004077476` | `pp_linyi_spot` |
| `S004161475` | `pp_wenzhou_spot` |
| `S004242346` | `ru_100ppi_spot` |
| `S004242348` | `ma_100ppi_spot` |
| `S004242351` | `pp_100ppi_spot` |
| `S004242352` | `pta_100ppi_spot` |
| `S004317129` | `ssf_qd` |
| `S004724779` | `sp_100ppi_spot` |
| `S005028348` | `pg_100ppi_spot` |
| `S005349977` | `ur_100ppi_spot` |
| `S005349978` | `eb_100ppi_spot` |
| `S005402481` | `px_100ppi_spot` |
| `S005402526` | `pf_100ppi_spot` |
| `S005476287` | `sp_inv_shfe_warrant` |
| `S005532614` | `cs_inv_dce_warrant` |
| `S005532617` | `CY_inv_czce_warrant` |
| `S005580993` | `ckc_au_cfr_cn` |
| `S005953326` | `steelproducts_prod_cisa` |
| `S005955224` | `ma_east_spot` |
| `S005961124` | `io_inv_31ports` |
| `S005961126` | `io_inv_41ports` |
| `S005961196` | `io_inv_31ports_trade` |
| `S006095407` | `l_tj_spot` |
| `S006700187` | `lh_inv_dce_warrant` |
| `S008527032` | `alumina_spot_guangxi` |
| `S008527035` | `alumina_spot_guizhou` |
| `S008527041` | `alumina_spot_shanxi` |
| `S008527044` | `alumina_spot_henan` |
| `S008527822` | `al_wm0_phybasis_low` |
| `S008527823` | `al_wm0_phybasis_high` |
| `S008679890` | `pbf_prem` |
| `S008679899` | `pbf_sb` |
| `S009122311` | `bf_workrate_cap` |
| `S009138370` | `fo_180cst_m1_sgp` |
| `S009138371` | `fo_180cst_m2_sgp` |
| `S009138389` | `lu_0.5_prem_sgp` |
| `S009761758` | `espo_prem_sd` |
| `S009761761` | `espo_spot_sd` |
| `S009761779` | `oman_prem_sd` |
| `S009761782` | `oman_spot_sd` |
| `S009767207` | `crude_arrival_prem` |
| `S009767208` | `crude_imp_spot_cn` |
| `S010308683` | `ckc_inv_110washery` |
| `S011334851` | `pg_east_spot_idx` |
| `S011334852` | `pg_south_spot_idx` |
| `S011334855` | `pg_sd_spot_idx` |
| `S011799884` | `ckc_au_fob` |
| `S012185571` | `TA_east_spot2` |
| `S018042405` | `v_inv_mill_mth` |
