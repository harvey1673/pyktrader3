# Fundamental signal research after the workbook reconciliation

The source catalog is `fundamental_signal_coverage_2026-10-04.csv`. It contains 1,209 source-qualified rows: 1,098 iFind IDs and 111 MySteel IDs. Every mapped alias is present in `C:\dev\data\spot_df_20261001.parquet` with at least one non-null value. The rebuilt cache has 1,433 distinct columns, including the two currency-converted PG/FU carry proxies.

## Coverage

| Family | Mapped source rows | Available by 2016-01-01 |
|---|---:|---:|
| Inventory, warrants and inventory days | 386 | 158 |
| Supply, production and utilization | 56 | 14 |
| Demand, sales and transaction volume | 13 | 3 |
| Cost and margin | 84 | 9 |
| Product/location spread and premium | 84 | 38 |
| Sentiment and auction activity | 8 | 0 |
| Price | 310 | 171 |
| Other/macro | 268 | 207 |

The classification is a research index based on the exact Chinese header and alias. It does not replace field-level economic review. The maps now have no cross-provider alias collisions. The three overlapping iFind quotes use an `_if` suffix while the established MySteel aliases remain unchanged, so both providers can be tested without changing the previously selected production series.

## First signal basket to test

Use a small basket per market. Start with one broad stock measure, one flow or inventory-days measure, and one margin/spread measure when available. Do not sum total and component inventories or treat the same vendor concept as independent evidence.

| Market | Level or stock | Flow, supply or demand | Cost, margin or spread |
|---|---|---|---|
| I | `io_inv_45ports`, `io_invdays_mill(64)` | `io_spot_trade_volume_ports_w_ms`, `io_loading_14ports_ausbzl` | `pbf_import_profit`, `nmf_import_profit` |
| J | `coke_inv_4ports`, `coke_inv_230cokery` | `coke_dprod_230cokery`, `coke_dprod_247mill` | `coke_senti_124cokery`, `coke_senti_31mills` |
| JM | `ckc_stock_ganqimaodu`, `jm_inv_523mines` | `jm_dprod_523mines`, `jm_import_throughput_gantimaodu` | `jm_auction_rate_cn_d`, `jm_listed_cn_d` |
| SM | `SM_inv_mill`, `SM_stockdays` | `SM_dmd_cn`, `SM_prod_cn`, `SM_hesteel_purchase_qty` | `SM_margin_north`, `SM_neimeng_cost`, `ferroalloy_power_px_neimeng` |
| SF | `SF_inv_mill` | `SF_dmd_cn`, `SF_prod_cn`, `SF_hesteel_purchase_qty` | `SF_neimeng_margin`, `SF_neimeng_cost`, `ferroalloy_power_px_ningxia` |
| CU | `cu_inv_combo`, `cu_inv_exch_d` | — | `cu_mine_tc`, `cu_import_margin_sh`, `cu_lme_futbasis` |
| AL/AO | `al_inv_social_all`, `ao_inv_total_cn` | alumina/bauxite shipment series | alumina regional spread and aluminum processing fees |
| ZN/PB/NI/SN | domestic social inventory plus `*_inv_exch_d` | refined or ore availability where present | treatment charge, secondary-production margin and import premium |
| L | `pe_inv_social`, `pe_inv_mill_cn_w_ms` | `pe_pipe_invdays` | domestic 7042 location spread versus CFR LLDPE |
| PP | `pp_inv_mill_cn_w_ms`, `polyolefin_inv` | downstream nonwoven/BOPP inventory days | T30S producer/location spreads |
| V | `v_inv_social`, `v_inv_social_large_cn_w_ms` | PVC operating rates | carbide-route versus ethylene-route PVC spread |
| EG | `eg_inv_port_east` | — | East-China spot versus CFR Northeast Asia |
| EB | `eb_inv_commercial_js_w_ms`, `eb_inv_port_east` | — | Jiangsu N+1/N+2 and East-China versus CFR spreads |
| TA/PX | `TA_inv_social_wk`, `TA_invdays_mill` | PTA/PX utilization | `TA_margin_cn_d`, `PX_margin_cn_w`, `PX_naph_spd_w`, `PX_MX_spd_w` |
| MA | `MA_inv_ports_total`, `MA_inv_ports_cn_w_ms` | — | Taicang paper-month structure and port/interior spread |
| RU/NR/BR | exchange/bonded/social rubber inventory | tire inventory days as downstream demand proxy | natural-versus-synthetic rubber and regional import parity |
| BU | `bu_inv_social`, `bu_inv_mill`, `bu_invcap_shfe_all` | refinery-side production/utilization when added | regional heavy-asphalt and crude/feedstock spread |
| PG | `pg_inv_port_all`, `pg_inv_mill_all`, `pg_invratio_ports` | — | propane/butane CFR and location premiums |
| M/Y | crusher meal/oil inventory and port soybean inventory | crushing throughput when added | crush margin and meal-oil spread |
| RM/OI | crusher, port and regional inventories | — | rapeseed crush margin and regional spreads |
| C/CS | north-port, Guangdong-port and mill inventory | downstream inventory days | corn-starch processing margin |
| SP | `sp_inv_ports_cn_w_ms` plus Changshu/Qingdao components | — | Silver/other deliverable-brand regional spread |

## First-pass transformations

1. Inventory level: negative seasonal percentile or seasonal z-score of `log1p(inventory)`. Use a 3–5 year day/week-of-year history where available.
2. Inventory change: negative standardized 4-week change. Test separately from level before combining them.
3. Balance: standardized demand growth minus supply growth. Do not subtract levels with incompatible units.
4. Inventory days: use the published ratio directly; avoid rebuilding it from a different sample unless both numerator and denominator match.
5. Margin and product spread: use standardized level and change as separate variants. The economic direction must be set by product; high producer margin can encourage future supply but can also indicate strong current demand.
6. Sentiment and auction data: lag to the known publication date and cap outliers. These histories mostly begin after 2016 and should be treated as later-sample signals.

All weekly and monthly features need release-date alignment before backtesting. Publication timing is more important than filling every daily date: forward-fill only after the observation became available.

## Updated physical-carry benchmark

Using the reconciled October cache and futures data through 2026-09-18, the ten-market L/PP/V/EG/EB/TA/MA/RU/BU/PG basket produced:

| Recipe | Current spot net Sharpe | CTD proxy net Sharpe | Current 1y | CTD 1y |
|---|---:|---:|---:|---:|
| EMA 5–9 | 0.586 | 0.660 | 0.805 | 0.870 |
| EMA 1 | 0.590 | 0.669 | 0.861 | 0.926 |

Current and CTD inputs are identical for eight of the ten markets in this benchmark. The difference is concentrated in RU and V; the portfolio result validates the refreshed pipeline but does not establish broad CTD alpha.

The retired production physical-carry references were replaced after the live updater exposed `pp_100ppi_spot` as a hard failure. PP now uses `pp_t30s_shaoxing_hz`; PG uses `propane_cfr_south_cny_vat`; FU uses `fo_380cst_zhoushan_cny`. The PG proxy includes USD/CNY and 13% VAT but omits duty, port costs, financing and inland freight. The FU proxy converts the USD/t quote to CNY/t but still needs bonded/domestic tax and storage validation.
