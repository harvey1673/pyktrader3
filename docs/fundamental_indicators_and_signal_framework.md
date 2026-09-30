# Fundamental indicators and signal framework

## Scope

This note covers DCE coke (`J`) and coking coal (`JM`), CZCE manganese silicon (`SM`) and ferrosilicon (`SF`), CZCE soda ash (`SA`) and flat glass (`FG`), SHFE base metals (copper, aluminum/alumina, zinc, lead, nickel, tin, and stainless steel), and GFEX lithium carbonate (`LC`), industrial silicon (`SI`), and polysilicon (`PS`). It also develops dedicated relative-value frameworks for `SS-NI`, `AL-AO`, and `PS-SI`.

The current-data assessment uses these workbooks as of 2026-09-22:

- `C:\Users\harve\Nutstore\1\Nutstore\ifind_daily.xlsx`
- `C:\Users\harve\Nutstore\1\Nutstore\ifind_data.xlsx`
- `C:\Users\harve\Nutstore\1\Nutstore\mysteel_metal.xlsx`

The normalized inventory is in `docs/current_fundamental_indicator_inventory.csv` and contains 1,200 series. Futures prices, term structure, volume, open interest, and position data are assumed to come from the trading database rather than these three fundamental workbooks.

## Recommended sourcing order

The highest-value additions are not more spot-price series. The workbooks already contain good spot, inventory, warehouse-receipt, cost, and selected operating data. Prioritize the following:

1. Physical demand: hot-metal output, downstream operating rates, orders, and purchases.
2. Supply response: capacity utilization, maintenance, production, and new capacity by process and region.
3. Flows: imports, port arrivals, shipments, customs data, and regional transfers.
4. True economics: plant margins with the correct raw-material mix, energy, freight, tax, and by-product credits.
5. Inventory in days of use, not only absolute tonnes.
6. Publication timestamps and revision history for every weekly series.

Mysteel is generally the first choice for plant surveys, production, utilization, inventories, maintenance, port flows, and tender data. iFind is generally the first choice for official customs/NBS series, exchange data, macro variables, FX, spot prices, and long histories. Keep one preferred series per economic concept and retain alternatives only for robustness checks.

## Coverage and gaps by product

### Coke and coking coal (`J` / `JM`)

Current coverage is already useful:

- Regional coke spot and delivered prices.
- Mongolian, Australian, and domestic coking-coal prices.
- Coke stocks at ports, 230 independent cokers, and 247 steel mills.
- Coking-coal stocks at ports, cokers, and steel mills.
- Coke output for 230 cokers and 247 steel mills.
- Clean-coal mine output/inventory, washer inventory, auction completion, listings, and Ganqimaodu throughput.
- Mysteel sentiment surveys across miners, cokers, steel mills, and traders.
- Exchange warehouse receipts and 247-mill blast-furnace operating rate.

Add these series, in priority order:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | 247-mill daily hot-metal output and blast-furnace capacity utilization | `247家钢厂 日均铁水 产能利用率` / Mysteel | Best high-frequency physical demand proxy for coke. |
| P0 | Steel-mill profitability and profitable-mill share | `247家钢厂 盈利率` / Mysteel | Determines whether hot-metal demand is sustainable. |
| P0 | Full-sample coker capacity utilization and coke output | `独立焦企 全样本 产能利用率 焦炭产量` / Mysteel | Current output coverage should be paired with capacity response. |
| P0 | Independent-coker profit and blended-coal cost | `独立焦企 吨焦利润 配煤成本` / Mysteel | Required for a defensible `J-JM` processing margin. |
| P0 | Mine and washer operating rate, raw-coal and clean-coal output | `炼焦煤 矿山 开工率 原煤 精煤 洗煤厂 产量` / Mysteel | Separates mine supply from washing yield and inventory movements. |
| P0 | Coking-coal imports by origin and border/port arrivals | `炼焦煤 进口 蒙古 澳大利亚 俄罗斯 加拿大` / iFind customs; Mysteel arrivals | Captures the largest supply shocks and quality-mix changes. |
| P1 | Coke and coking-coal inventory days at mills/cokers | `可用天数 焦炭 炼焦煤 钢厂 焦企` / Mysteel | More comparable through time than tonnes when output changes. |
| P1 | Coke purchase-price increase/decrease rounds | `焦炭 提涨 提降 轮次` / Mysteel | Event variable for spot-futures convergence and margin transfer. |
| P1 | Coke-oven maintenance, new capacity and closures | `焦化 产能 新增 淘汰 检修` / Mysteel | Defines medium-horizon supply regime. |
| P1 | By-product prices and credits | coal tar, crude benzene, ammonium sulfate, coke-oven gas / iFind or Mysteel | Needed for full coker margin rather than a raw `J-1.33*JM` proxy. |
| P2 | Freight by route and delivery-location basis | Mongolia-border freight, Shanxi-Tangshan, port freight / Mysteel | Prevents false regional-arbitrage signals. |
| P2 | Delivered quality adjustment | CSR/CRI, sulfur, ash, G/Y values / Mysteel | JM futures are one deliverable specification, not the entire blend. |

### Manganese silicon and ferrosilicon (`SM` / `SF`)

Current coverage is strong on daily spot prices, production costs and margins, weekly production/utilization/demand, national stocks, Hebei Steel tenders, provincial electricity prices, manganese-ore prices/stocks, semi-coke, and warehouse receipts.

Add these series:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | Producer inventory by province and deliverable grade | `硅锰/硅铁 厂库 内蒙古 宁夏 广西` / Mysteel | National inventory can hide a deliverable-region squeeze. |
| P0 | Plant utilization, maintenance, and furnace restarts by province | `铁合金 开工率 产能利用率 检修 复产` / Mysteel | Power and ore shocks are regional. |
| P0 | Manganese-ore arrivals, shipments, port stocks by origin/grade | `锰矿 到港 发运 港口库存 南非 加蓬 澳块` / Mysteel | Key leading input for SM cost and availability. |
| P0 | Steel-mill tender price and quantity beyond Hebei Steel | `钢招 硅锰 硅铁 采购价 采购量` / Mysteel | Measures real downstream clearing price and volume. |
| P0 | Provincial industrial power tariff and curtailment/event flags | iFind tariff; Mysteel plant survey | Electricity is the dominant SF cost and a major SM cost. |
| P1 | Silica, semi-coke, coke, electrode paste, and freight | relevant regional spot series / Mysteel or iFind | Converts quoted margins into auditable plant economics. |
| P1 | Magnesium output, price, and plant utilization | `金属镁 产量 开工率 价格` / Mysteel | Important non-steel demand for SF. |
| P1 | Rebar/hot-metal output and mill alloy consumption | `钢厂 合金 消耗 硅锰 硅铁` / Mysteel | Links steel output to actual alloy demand. |
| P2 | Overseas manganese-ore mine disruptions and export data | South Africa, Gabon, Australia / iFind customs and shipping | Useful for medium-horizon SM trades. |
| P2 | Deliverable-production share and warehouse-location stock | exchange/Mysteel | Improves basis and squeeze-risk modeling. |

### Soda ash and flat glass (`SA` / `FG`)

Current coverage includes light/heavy soda-ash spot prices, soda-ash output/utilization/inventory/profit and production-sales ratio, FG regional spot prices, producer inventories, operating lines, daily melt, utilization, fuel-route margins, photovoltaic-glass utilization/melt/inventory, and warehouse receipts.

Add these series:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | Soda-ash maintenance, effective capacity, new capacity, and output by process | `纯碱 检修 产能 天然碱 氨碱 联碱 周产量` / Mysteel | Supply changes are lumpy and process-specific. |
| P0 | Heavy/light output split, heavy-ash orders, and delivery volume | `重碱 轻碱 产量 订单 待发量` / Mysteel | FG consumes heavy ash; total soda-ash inventory is not enough. |
| P0 | Float/PV glass ignition, cold-repair, and restart schedule | `浮法玻璃 光伏玻璃 点火 冷修 复产 产线` / Mysteel | Creates a forward soda-ash demand schedule. |
| P0 | Float-glass deep-processing orders and original-sheet sales | `玻璃 深加工订单 原片 产销率` / Mysteel | Better demand signal than property data alone. |
| P0 | FG producer inventory in days and regional shipment rate | `浮法玻璃 库存天数 出库 发运` / Mysteel | Normalizes inventory across changing capacity. |
| P1 | Sodium silicate, detergent, bicarbonate, and lithium-carbonate soda demand | relevant operating-rate/output series / Mysteel or iFind | Captures the non-glass part of SA demand. |
| P1 | Regional natural gas, petroleum coke, coal tar, and coal prices | iFind daily; Mysteel | Required for plant-specific FG cash margins. |
| P1 | Property completion, glass deep-processing, auto production, and PV module schedules | iFind official macro/industry data | End-demand confirmation at slower frequency. |
| P1 | Regional freight and deliverable-location basis | `沙河 华中 华东 运费 玻璃 纯碱` / Mysteel | Important because both contracts are physically delivered. |
| P2 | Import/export orders and arbitrage | customs/iFind | Useful in high-price regimes but usually secondary. |

### Copper

Already present: LME price/curve/inventory, domestic spot and premium, bonded stocks, exchange receipts, TC/RC, Yangshan premium, import breakeven, refined-versus-scrap spreads, copper-rod fees, and concentrate-port stocks.

Add:

- P0: refined-copper output, smelter utilization/maintenance, and new capacity.
- P0: copper-concentrate imports and arrivals by origin; mine disruption/event flags.
- P0: refined-copper import/export and bonded-to-domestic flows.
- P0: operating rates and orders for wire rod, tube, plate/strip, foil, and cable.
- P1: scrap supply, imports, and tax-policy effects; secondary-copper utilization.
- P1: grid investment/delivery, air-conditioner output, EV/PV installation, and export orders.
- P1: sulfuric-acid and precious-metal by-product credits for smelter margin.

### Aluminum and alumina (`AL` / `AO`)

Already present: LME/A00 price and premium, exchange and social inventory, bauxite spot and port stocks, bauxite shipping proxies, alumina domestic/import prices and multiple inventory locations, alumina receipts, 6063 billet fees/stocks, and scrap prices.

Add:

- P0: electrolytic-aluminum output, effective capacity, utilization, maintenance, and power curtailment by province.
- P0: liquid-aluminum ratio and ingot casting ratio; these explain visible ingot inventory.
- P0: alumina output/utilization/maintenance by region and process.
- P0: bauxite arrivals/imports by origin, especially Guinea and Australia.
- P0: alumina cost inputs: caustic soda, coal/gas, lime, bauxite grade, and freight.
- P1: aluminum semis operating rates/orders for extrusion, plate/sheet/strip, foil, cable, and primary alloy.
- P1: semis exports, billet inventories, PV/auto demand, and regional spot premiums.

### Zinc

Already present: LME price/curve/inventory, domestic spot premiums, warehouse receipts, domestic/import concentrate TC, concentrate-port stocks, refined stocks, smelter finished-goods stocks, zinc-alloy fees, and galvanized-sheet stocks/prices.

Add:

- P0: refined-zinc output, smelter utilization, maintenance, and power constraints.
- P0: zinc-concentrate mine output/imports and port arrivals by origin.
- P0: galvanizing, die-casting alloy, brass, and zinc-oxide operating rates/orders.
- P1: refined-zinc import/export and bonded flows.
- P1: sulfuric-acid by-product price and regional power cost for smelter margins.
- P1: infrastructure, auto, appliance, and construction demand proxies.

### Lead

Already present: LME price/curve/inventory, domestic primary/recycled lead prices, primary-concentrate TC and prices, scrap-battery price proxies, refined stocks, warehouse receipts, recycled-lead profit, and recycled raw-material/finished-goods stocks.

Add:

- P0: primary and recycled refined-lead output/utilization/maintenance.
- P0: lead-acid battery operating rates, finished-goods stocks, orders, and exports.
- P0: waste-battery supply, regional price, collection volume, and tax-policy changes.
- P1: lead-concentrate imports and mine supply.
- P1: silver and sulfuric-acid by-product credits for primary smelters.
- P1: e-bike/auto replacement demand and battery scrap-return seasonality.

### Nickel

Already present: LME price/curve/inventory, domestic refined premiums, nickel ore prices and port stocks, NPI prices, sulfate prices, refined stocks, selected 300-series stainless inventories, and Indonesia NPI import profit.

Add:

- P0: Indonesia and China NPI output, capacity, and utilization.
- P0: Indonesia ore quota/RKAB approvals, mine output, and ore arrivals.
- P0: stainless output/utilization/orders and total social/mill inventory by grade.
- P0: refined-nickel output, imports/exports, and new deliverable brands.
- P1: MHP/high-matte output, imports, payables/discounts, and conversion margins.
- P1: battery precursor/cathode operating rates and sulfate production margins.
- P1: ferronickel/stainless import-export flows.

### Stainless steel and nickel (`SS` / `NI`)

Current coverage can support a first-pass stainless-versus-nickel model:

- 300-series social inventory for Foshan and Wuxi, plus 200/300/400-series and total stainless inventory series.
- Stainless warehouse receipts and Wuxi 304 scrap-stainless price.
- NPI spot prices and Indonesian NPI import profit.
- The nickel-chain series listed above, including refined nickel, ore, NPI, sulfate, stocks, premiums, and LME structure.

Add these series:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | 304 cold-rolled and hot-rolled spot prices and regional basis | `304 冷轧 热轧 无锡 佛山 现货 升贴水` / Mysteel | The futures contract cannot be compared reliably with one scrap or inventory series. |
| P0 | Stainless output, utilization, maintenance, and schedules by 200/300/400 series | `不锈钢 排产 产量 开工率 300系 检修` / Mysteel | Separates nickel-intensive 300-series supply from the total market. |
| P0 | Mill, trader, and social inventory by series and region; inventory days | `不锈钢 厂库 社库 库存天数 无锡 佛山` / Mysteel | Identifies whether a squeeze is in deliverable material rather than total tonnes. |
| P0 | Mill orders, shipment rate, and downstream purchasing | `不锈钢 订单 接单 发运 成交` / Mysteel | Demand confirmation for changes in inventory. |
| P0 | Stainless raw-material basket: NPI/FeNi, high-carbon ferrochrome, chrome ore, and scrap | Mysteel daily/weekly prices | A mill margin is more defensible than treating refined-NI futures as the only nickel input. |
| P1 | Indonesia stainless and NPI output, capacity, maintenance, and exports to China | `印尼 不锈钢 镍铁 产量 出口` / Mysteel or iFind customs | Indonesian integration can move NPI and stainless together while refined nickel diverges. |
| P1 | End-use indicators for appliances, auto, elevators, machinery, and petrochemicals | iFind industry data; Mysteel surveys | Distinguishes real demand from trader restocking. |
| P1 | Conversion costs, regional freight, and deliverable-brand/location basis | Mysteel | Needed to compare mill economics with the futures delivery system. |

The existing stainless spot-premium series ends in December 2025 and should be refreshed or replaced before it is used in a live basis signal.

### Tin

Already present: LME price/curve/inventory, domestic spot/premium, exchange receipts, concentrate price/TC, refined social stocks, and selected scrap prices.

Add:

- P0: Myanmar mine/wa-state shipment and policy data, Indonesian exports, and China concentrate imports.
- P0: refined-tin output, smelter utilization, maintenance, and raw-material days.
- P0: solder producer utilization/orders and electronics/semiconductor demand indicators.
- P1: refined import/export arbitrage and bonded stocks.
- P1: secondary tin/scrap supply and processing margins.

### Lithium carbonate (`LC`)

Current coverage includes battery- and industrial-grade domestic spot prices, selected CIF/FOB prices, exchange warehouse receipts, monthly operating rate, Mysteel production-profit series for spodumene/lepidolite/carbonation routes, and selected lithium-ore inventories.

Add these series:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | Weekly output, utilization, maintenance, and capacity by spodumene, lepidolite, brine, and recycling route | `碳酸锂 周产量 开工率 检修 锂辉石 云母 盐湖 回收` / Mysteel | Supply response and cost differ materially by process. |
| P0 | Producer, trader, downstream, warehouse, and deliverable-grade inventory; inventory days | `碳酸锂 厂库 社库 下游库存 可交割 库存天数` / Mysteel | Exchange receipts alone miss most physical inventory. |
| P0 | Spodumene 6% CIF, lepidolite by grade, brine cost, and ore payables | Mysteel or iFind | Required for route-specific margins and mine-to-salt transmission. |
| P0 | LFP/NCM cathode output, utilization, orders, and inventories | `磷酸铁锂 三元材料 排产 开工率 库存 订单` / Mysteel | High-frequency first-use demand for lithium salts. |
| P0 | Cell output/inventory and EV/ESS battery schedules | iFind industry data; Mysteel | Confirms whether cathode restocking reflects end demand. |
| P1 | Lithium hydroxide prices, inventories, and conversion spread | Mysteel/iFind | Captures product switching and divergent battery chemistry demand. |
| P1 | Lithium salt and ore imports by origin, customs clearance, and port arrivals | Chile, Argentina, Australia / iFind customs; Mysteel | Imports can overwhelm domestic production changes. |
| P1 | Recycling output, feed cost, and utilization | `锂电回收 碳酸锂 产量 开工率 成本` / Mysteel | Recycling is a price-sensitive marginal supply source. |
| P1 | Quality/specification basis and regional freight | Mysteel | Avoids mixing industrial grade, battery grade, and deliverable material. |

### Industrial silicon (`SI`)

Current coverage is thin: exchange warehouse receipts and one national social-inventory series, the latter ending in July 2025.

Add these series:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | Spot prices by grade and region, including 553, 441, 421, and 3303; deliverable basis | `工业硅 553 441 421 3303 新疆 云南 四川` / Mysteel | Grade and oxygen/non-oxygen specifications can move differently. |
| P0 | Weekly output, utilization, maintenance, and capacity by province | `工业硅 周产量 开工率 检修 新疆 云南 四川` / Mysteel | Supply is highly regional and seasonal. |
| P0 | Producer, port, warehouse, and social inventory; inventory days and receipt aging | `工业硅 厂库 港口 社库 仓单 库龄` / Mysteel/exchange | Separates broad surplus from deliverable tightness. |
| P0 | Regional industrial electricity prices, hydropower season, and curtailment flags | iFind tariffs; Mysteel plant survey | Power cost and Yunnan/Sichuan hydrology drive the marginal supply curve. |
| P0 | Silica, petroleum coke/coal/charcoal, and electrode prices | Mysteel/iFind | Completes auditable cash-cost curves. |
| P0 | Output/utilization and silicon consumption for aluminum alloy, organosilicon, and polysilicon | Mysteel | Maps the three principal demand channels. |
| P1 | Exports/imports, regional freight, new capacity, and environmental policy events | iFind customs; Mysteel | Important for medium-horizon balance and regional basis. |

### Polysilicon (`PS`)

Current coverage includes exchange warehouse receipts, monthly utilization, weekly N-type dense-material cost, and national weekly inventory.

Add these series:

| Priority | Indicator to source | Preferred search terms / likely source | Why it matters |
|---|---|---|---|
| P0 | Spot prices and basis by N-type dense, granular, P-type, and deliverable quality | `多晶硅 N型致密料 颗粒硅 P型 交割品` / Mysteel | Quality premia are essential for mapping spot economics to futures. |
| P0 | Weekly output, utilization, maintenance, and new capacity by producer/region | `多晶硅 周产量 开工率 检修 新增产能` / Mysteel | Monthly utilization is too slow for supply-turning points. |
| P0 | Producer, trader, and warehouse inventory; order books and contract-signing rate | `多晶硅 厂库 社库 订单 签单率` / Mysteel | Distinguishes forced producer accumulation from downstream restocking. |
| P0 | Industrial silicon, electricity, trichlorosilane/HCl, steam, and cash-cost/depreciation series | Mysteel/iFind | Required for a `PS-SI` processing margin and shutdown threshold. |
| P0 | Wafer output, utilization, inventories, and prices by N/P type and wafer size | `硅片 排产 开工率 库存 N型 P型 182 210` / Mysteel | The nearest downstream demand and margin signal. |
| P1 | Cell/module output, utilization, inventories, exports, tenders, and order schedules | Mysteel/iFind | Confirms whether wafer demand is sustainable. |
| P1 | Imports/exports, quality conversion, financing cost, and warehouse aging | iFind customs; exchange/Mysteel | Useful for delivery pressure and convergence. |

## Signal construction

### Data preparation common to all signals

1. Use the publication timestamp, not the observation-period label. A Friday survey released the following week must enter the backtest only at release.
2. Forward-fill weekly data only after release and cap the maximum stale age.
3. Convert inventories to days of demand where possible.
4. Seasonally adjust weekly series using week-of-year history, then calculate robust rolling z-scores using a median/MAD or winsorized mean/std.
5. Prefer surprises and changes to raw levels: latest minus seasonal expectation, four-week change, and year-on-year change.
6. Build separate supply, demand, inventory, margin, basis/curve, and flow sleeves. Do not allow ten correlated inventory series to dominate one demand series.
7. Smooth only after timestamp alignment. A 2-4 observation EWMA is a reasonable starting point for weekly data.

A generic flat-price score can be:

```text
inventory_tightness = -z(inventory_days) - 0.5*z(change_4w_inventory_days)
demand_impulse      =  z(downstream_output_surprise) + z(order_or_sales_surprise)
supply_impulse      =  z(production_surprise) + z(utilization_change) + z(net_import_surprise)
physical_tightness  =  z(spot_basis) + z(curve_backwardation)

fundamental_score = average(
    inventory_tightness,
    demand_impulse,
    -supply_impulse,
    physical_tightness
)
```

Enter only when at least three independent sleeves agree. Start with a 5-20 trading-day holding horizon for weekly physical data and a 1-5 day horizon for daily basis/auction/flow surprises. Fit weights only with walk-forward data; equal-weighted sleeves are the proper benchmark.

Producer margin is regime-dependent. High margins are normally bearish at a multi-week horizon because they invite output, while negative margins become bullish only after utilization, maintenance, or output confirms cuts. Treat margin as a supply-response gate, not a mechanically signed signal.

### `J-JM`: processing spread and relative-value signal

The physical starting point is:

```text
coke_cash_margin = coke_spot
                 - coal_ratio * blended_coking_coal_cost
                 - conversion_cost
                 + byproduct_credit
```

The DCE states that roughly 1.33 tonnes of coking coal are consumed per tonne of coke. Use this as an initial physical coefficient, but build `blended_coking_coal_cost` from the actual coal-quality mix rather than treating the JM deliverable price as the whole blend.

For futures:

```text
physical_spread_m = J_m - 1.33 * JM_m
```

The DCE contract sizes are 100 tonnes for J and 60 tonnes for JM. One J lot therefore corresponds to about 2.22 JM lots on a physical-input basis. In practice, trade integer baskets over multiple units or use a beta/volatility hedge; do not assume a one-lot-versus-one-lot position is a processing-margin hedge.

A more robust tradable residual is:

```text
J_m = intercept + beta_jm*JM_m + beta_i*I_m + beta_rb*RB_m + seasonal_terms + residual
rv_signal = -z(residual)
```

Estimate the regression on a rolling window using same-delivery-month or constant-maturity prices. Add iron ore/rebar only if they improve out-of-sample stability. Use the residual for entry and the physical margin/inventory balance for confirmation.

Long J / short JM is supported when:

- Coke stocks are falling and steel-mill coke days are low.
- Hot-metal output is firm or rising.
- Coker margins are compressed and confirmed output cuts are appearing.
- Coking-coal mine/import supply is improving, coal auctions weaken, or coal stocks rise.

Short J / long JM is supported by the reverse combination: rising coke output/stocks, weak hot metal, strong coker margins, and tightening coal supply or stronger auctions.

Main failure modes: changes in coal-blend quality, coke price increase/decrease rounds, policy-driven mine inspections, import restrictions, and mismatched delivery months/locations.

### `FG-SA`: physical cost spread and relative-value signal

An industry starting assumption is about 0.20 tonne of heavy soda ash per tonne of float glass. The plant cash margin is therefore:

```text
glass_cash_margin = FG_spot
                  - 0.20 * heavy_SA_spot
                  - fuel_cost
                  - silica_and_other_raw_materials
                  - conversion_and_freight
```

A first-pass futures spread is:

```text
physical_spread_m = FG_m - 0.20 * SA_m
```

Both CZCE contracts use 20 tonnes per lot. A physical raw-material basket is therefore approximately five FG lots against one SA lot. A 1:1 `FG-SA` price spread is a statistical pair, not a cost hedge. For trading, compare three versions: physical coefficient, rolling regression beta, and dollar-volatility-neutral beta.

Long FG / short SA is supported when:

- Float-glass inventory days fall, shipments and deep-processing orders rise.
- FG cold repairs increase or restarts are delayed.
- SA utilization/output rises, maintenance ends, and heavy-ash inventory/orders loosen.
- Non-glass SA demand is weak.

Short FG / long SA is supported when glass inventory builds and orders weaken while SA maintenance, new downstream demand, exports, or supply outages tighten heavy ash.

Important caveat: SA has large non-float-glass demand and FG has large fuel-cost exposure. A two-leg regression can break when light-ash demand, lithium-related demand, photovoltaic glass, or fuel prices move independently.

### `SM-SF`: common-demand relative value

SM and SF share steel demand and significant electricity exposure but have different dominant raw materials. The CZCE describes electricity as roughly 60-70% of SF cost, while manganese ore is roughly 60% of SM cost and electricity roughly 20-25%.

Construct:

```text
rv_residual = SM_m - beta_t * SF_m
```

Estimate `beta_t` from a rolling robust regression or use dollar-volatility neutrality. Confirm the residual with the difference between product-specific fundamental scores:

```text
relative_fundamental = score_SM - score_SF
```

Long SM / short SF is supported by tighter manganese ore/SM producer stocks, stronger SM tenders, or weaker SF magnesium demand. Short SM / long SF is supported by loose manganese ore/SM supply or power/production restrictions concentrated in SF regions. Do not rely on substitution demand: the exchange's industry material notes that actual substitution is limited.

### `SS-NI`: stainless margin and nickel-chain relative value

Start with the stainless mill input basket rather than a two-price identity:

```text
SS_mill_margin = SS_spot
               - k_npi * NPI_price
               - k_fecr * high_carbon_ferrochrome_price
               - k_scrap * stainless_scrap_price
               - conversion_cost
               - freight
```

Estimate the input coefficients from representative mill recipes and allow the nickel-unit source to vary. The 304 specification contains roughly 8-11% nickel, so one 5-tonne SS lot contains approximately 0.40-0.55 tonne of nickel on a chemistry basis. This is only a material-content reference. It is not a fixed hedge ratio against the 1-tonne NI contract because NPI, ferronickel, scrap, and refined nickel are not perfectly substitutable.

For a tradable pair, estimate a controlled residual:

```text
SS_m = intercept
     + beta_ni * NI_m
     + beta_fecr * ferrochrome
     + beta_npi * NPI
     + seasonal_terms
     + residual

rv_signal = -z(residual)
```

If NPI and ferrochrome are not available point-in-time, use `SS_m - beta_t*NI_m`, but require the direction to agree with the difference in product-level fundamentals:

```text
relative_fundamental = score_SS - score_refined_NI
```

Long SS / short NI is supported when 300-series stainless inventory falls, mill orders/shipments strengthen, and SS output is constrained while refined nickel or NPI supply improves. Short SS / long NI is supported when stainless output and inventories rise or downstream orders weaken while refined-nickel availability tightens.

Main failure modes are a Class-1-versus-NPI disconnect, Indonesian ore or export policy, ferrochrome shocks, changes in mill feed mix, and grade-specific stainless demand. Backtest physical-content, rolling-beta, and volatility-neutral sizing separately; do not hard-code a 0.40-0.55 NI-lot hedge as the trading optimum.

### `AL-AO`: smelting margin and relative value

Industry material indicates approximately 1.92 tonnes of alumina are required per tonne of primary aluminum. A simplified smelting margin is:

```text
AL_smelting_margin = AL_spot
                   - 1.92 * AO_spot
                   - electricity_cost
                   - carbon_anode_cost
                   - fluoride_and_other_cost
                   - freight
```

The first-pass futures spread is therefore:

```text
physical_spread_m = AL_m - 1.92 * AO_m
```

AL is 5 tonnes per lot and AO is 20 tonnes per lot. One AL lot therefore consumes about 9.6 tonnes, or 0.48 AO lot, on a physical-input basis; a practical integer basket is approximately two AL lots against one AO lot. Compare this with a rolling-beta and dollar-volatility-neutral basket because power, anodes, regional premiums, and capacity constraints cause the price relationship to change.

Long AL / short AO is supported when aluminum inventories and billet stocks fall, semis orders strengthen, and aluminum output is constrained while alumina utilization/output rises or bauxite supply improves. Short AL / long AO is supported when bauxite or alumina supply tightens and AO stocks fall while aluminum production/inventory rises or downstream demand weakens.

Use the margin as a supply-response signal: a compressed AL margin is bullish AL relative to AO only after smelter curtailment or reduced operating capacity is confirmed. Main failure modes are Guinea bauxite disruptions, hydropower or coal-price shocks, changes in liquid-aluminum/ingot ratios, import arbitrage, regional freight, and mismatched contract months.

### `LC`: flat-price, curve, and route-margin signals

Build the LC flat-price score with four distinct physical sleeves:

```text
LC_inventory = -z(total_inventory_days) - 0.5*z(change_4w_inventory_days)
LC_demand    =  z(LFP_NCM_output_surprise) + z(cell_schedule_surprise)
LC_supply    = -z(LC_output_surprise) - z(net_import_surprise)
LC_physical  =  z(battery_grade_basis) + z(curve_backwardation)

LC_score = average(LC_inventory, LC_demand, LC_supply, LC_physical)
```

Keep route economics separate:

```text
route_margin_r = LC_spot
               - ore_or_brine_requirement_r * feed_price_r
               - reagents_r
               - energy_r
               - conversion_and_freight_r
```

Do not use one universal ore coefficient: grade, recovery, process, and by-products differ across spodumene, lepidolite, brine, and recycling. A deeply negative marginal-route profit is bullish only after utilization, maintenance, or output confirms cuts; otherwise it can simply describe persistent oversupply.

Useful trades include outright LC from the composite score, calendar spreads from projected inventory and maintenance by delivery month, and battery-grade versus industrial-grade basis mean reversion. Lithium-hydroxide and cathode margins are confirmation variables rather than clean futures pairs. Watch for producer hedging, deliverable-quality conversion, import timing, downstream destocking, and policy-driven EV/ESS demand.

### `SI`: flat-price and hydropower-regime signals

Combine the common framework with a province-specific power and hydrology sleeve:

```text
SI_supply_pressure = z(output_surprise)
                   + z(utilization_change)
                   + z(restart_capacity)

SI_cost_support = z(regional_power_cost)
                + z(carbon_reductant_cost)
                + z(loss_making_capacity_share)

SI_score = average(
    -z(inventory_days),
    -SI_supply_pressure,
    z(weighted_downstream_output_surprise),
    SI_cost_support,
    z(spot_basis) + z(curve_backwardation)
)
```

Weight polysilicon, organosilicon, and aluminum-alloy demand using rolling observed consumption rather than fixed shares. Interact Yunnan/Sichuan output with hydropower season and electricity tariffs; a seasonal restart should not be scored as an unexpected bearish shock unless it differs from the historical calendar.

The cleanest expressions are SI outright and calendar spreads around wet-season restarts, dry-season cuts, maintenance, and new-capacity schedules. Require inventory and basis confirmation because high-cost support can coexist with prolonged surplus. Grade conversion, warehouse-receipt rules, regional freight, and environmental/power policy are the main regime-break risks.

### `PS-SI`: polysilicon processing spread and relative value

Use an auditable plant-cost formulation:

```text
PS_cash_margin = PS_spot
               - k_si * SI_spot
               - electricity_cost
               - trichlorosilane_and_chemicals
               - steam_and_other_conversion_cost
               - freight
```

Estimate `k_si` by representative technology and update it as recovery and recycling improve. Do not infer it from a short price regression. Normalize units before calculation: the existing PS cost series is quoted in yuan/kg while futures prices are yuan/tonne, so multiply yuan/kg inputs by 1,000 where required.

For futures:

```text
physical_spread_m = PS_m - k_si * SI_m
statistical_residual = PS_m - beta_t * SI_m
relative_fundamental = score_PS - score_SI
```

SI is 5 tonnes per lot and PS is 3 tonnes per lot. One PS lot requires `3*k_si/5` SI lots on an input basis. Build a larger integer basket or use rolling beta/volatility sizing; do not round a fractional physical requirement into a structurally biased one-lot pair.

Long PS / short SI is supported when PS inventories fall, wafer schedules/orders improve, and PS output is cut while SI output/inventory rises or non-PS SI demand weakens. Short PS / long SI is supported when PS capacity/output and inventories rise, wafer/cell margins weaken, and SI supply tightens through power, hydrology, or raw-material constraints.

The margin becomes bullish PS only after loss-making producers actually cut output. Major failure modes are technology-driven changes in silicon consumption, quality premia, producer self-supply, rapid capacity commissioning, wafer-industry destocking, and electricity-price divergence between SI and PS regions.

### Base-metal flat-price signals

Use the same sleeve architecture but adapt the economics:

- Inventory: SHFE/LME/COMEX where relevant, bonded/social/plant stocks, cancellations, and inventory days.
- Physical premium: domestic spot premium, import premium, bonded premium, cash-to-three-month spread.
- Raw-material tightness: TC/RC, ore/concentrate premium, scrap discount, or intermediate-product payable.
- Supply response: smelter/refinery utilization, maintenance, loss-making share, and new capacity.
- Demand: downstream operating rates/orders by first-use sector.
- Flows: imports/exports, port arrivals, cross-border arbitrage, and regional transfers.

Examples of useful directional interpretation:

- Copper: falling TC plus declining visible inventory and strong rod/cable orders is stronger than falling TC alone.
- Aluminum: low ingot stocks are less bullish if the liquid-aluminum ratio is falling and more metal is being cast into deliverable ingot.
- Zinc: low TC and smelter cuts are bullish only if galvanized/die-cast demand is not simultaneously weakening.
- Lead: recycled-lead margin and waste-battery availability often explain supply better than LME signals.
- Nickel: separate Class 1 refined nickel, NPI/stainless, and MHP/matte/battery chains; do not combine them into one undifferentiated inventory score.
- Tin: small visible inventories make mine/export disruptions important, but confirm them with smelter raw-material availability and solder demand.

### Base-metal relative value

Use two layers:

1. Build a comparable fundamental score for each metal.
2. Long the strongest and short the weakest while neutralizing broad metal beta, USD/CNY exposure, and ex-ante volatility.

Suitable comparisons include Cu/Al for broad cyclical and electrification demand, Zn/Al for construction/manufacturing divergence, and Pb/Zn for different downstream cycles sharing some smelting economics. Cross-metal pairs should be sized by rolling beta or volatility, not equal lots.

Processing and location spreads are generally cleaner than arbitrary cross-metal pairs:

- China/LME import arbitrage after FX, VAT, freight, financing, and premium.
- Aluminum minus alumina and power cost.
- Zinc/lead metal price versus concentrate cost, TC, energy, and by-products.
- Nickel futures versus NPI/stainless economics; sulfate versus MHP/matte economics.
- Refined metal versus scrap/recycled feedstock.

## Backtest and implementation checks

- Use same delivery month or a documented constant-maturity roll for both legs.
- Include contract multipliers in P&L and hedge sizing.
- Use point-in-time series with release timestamps and revision vintages where available.
- Test both level and change signals; many physical series are non-stationary.
- Remove predictable seasonality before z-scoring inventories and operating rates.
- Separate 2015-2019, 2020-2022, and later structural regimes where capacity or contract specifications changed.
- Charge bid/ask, commissions, slippage, and roll costs; use realistic limit-move and liquidity constraints.
- Require a minimum history and coverage ratio; block stale or missing inputs rather than replacing them with zero.
- Report turnover, hit rate, drawdown, performance by regime, and marginal contribution of each sleeve.
- Test incremental value over price momentum, carry, basis, and curve-only baselines.

## Observed data-quality risks in the current workbooks

- Nine indicator IDs are duplicated across sheets or workbooks. Deduplicate by indicator ID and choose one canonical frequency/source before building factors.
- Several useful base-metal series are stale: the existing SMM aluminum inventory series ends in March 2025, tin social inventory ends in July 2025, and nickel-bean six-region inventory ends in September 2024. The workbooks contain newer alternative inventory series, so stale columns should not be forward-filled indefinitely.
- The current stainless spot-premium series ends in December 2025, and the only SI social-inventory series ends in July 2025. Refresh them before using basis or inventory features; warehouse receipts are not substitutes for total social inventory.
- Some sheets label a series as daily even though the underlying survey is weekly or twice weekly. Preserve the actual release schedule from the source description.
- Week-ending labels can be later than the publication date. Treat these as period labels, not as timestamps at which the value was known.
- Some economic concepts have several overlapping definitions (exchange receipts, social inventory, bonded inventory, plant inventory). Keep them separate and document coverage; do not sum them unless the source definitions are mutually exclusive.
- Histories for several high-value Mysteel series begin only in 2020-2024. Use simpler models, stronger regularization, and shorter feature sets rather than fitting many parameters to limited regimes.
- Normalize units explicitly. In particular, PS cost data may be yuan/kg while GFEX PS futures are quoted in yuan/tonne; ore, contained-metal, and finished-product series may also use different grades or payable conventions.
- LC, SI, and PS contracts have short or structurally changing histories. Use event-aware walk-forward validation and avoid treating pre-listing spot relationships as equivalent to post-listing futures behavior.

## Suggested first implementation sequence

1. Build `J/JM` with hot metal, coker margin, coke/coal inventory days, coal imports, and auctions.
2. Build `FG/SA` with glass inventory days/orders, SA heavy-ash inventory/orders, line events, and both physical and regression hedge ratios.
3. Extend the already-strong `SM/SF` coverage with regional plant inventories, ore flows, tenders, and power events.
4. Build `AL/AO` next: the current workbooks already have strong spot, inventory, bauxite, and alumina coverage; add production/utilization, power, anodes, and downstream semis orders.
5. Build `SS/NI` after adding 300-series output/orders and the complete NPI-ferrochrome-scrap cost basket. Keep the mill-margin and refined-NI residual models separate.
6. Build `LC` as an outright/curve model using route-specific supply, total inventory, cathode schedules, imports, and marginal-route shutdown confirmation.
7. Build `SI`, then `PS/SI`, only after refreshing SI inventory and adding weekly regional production, power/hydrology, PS output, and wafer-side demand. The existing SI coverage is not sufficient for a robust backtest.
8. For the remaining base metals, add output/utilization/imports/downstream operating rates to the current price-premium-inventory framework. Copper is a strong starting point because existing market-microstructure coverage is rich.

## External references

- Dalian Commodity Exchange, coke and coking-coal factsheet: https://www.dce.com.cn/dce/file/2026-01-15/17684624156122c9a882b9ae6dcbb289019bc092f6fc1681.pdf
- Zhengzhou Commodity Exchange, glass and soda-ash contract comparison: https://www.czce.com.cn/cn/rootfiles/2021/09/03/1605597062175721-1605597062191026.pdf
- Zhengzhou Commodity Exchange, ferroalloy contract and industry guide: https://www.czce.com.cn/cn/rootfiles/2018/06/29/1531036139932593-1531036139954918.pdf
- Zhengzhou Commodity Exchange, ferroalloy industry fundamentals: https://www.czce.com.cn/cn/rootfiles/2022/06/16/1655814940219539-1655814940231752.pdf
- Shanghai Futures Exchange, nonferrous inventory data-system notice: https://www.shfe.com.cn/publicnotice/notice/202503/t20250327_824890.html
- Shanghai Futures Exchange, contract rules: https://www.shfe.com.cn/services/indexopt/contractrules/
- Shanghai Futures Exchange, stainless-steel futures manual: https://www.shfe.com.cn/products/futures/metal/ferrousandpreciousmetal/ss_f/manual/201909/P020240320684500937834.pdf
- Shanghai Futures Exchange, nickel futures contract: https://www.shfe.com.cn/products/futures/metal/nonferrousmetal/ni_f/standard_ni_f/202312/t20231205_313864.html
- Shanghai Futures Exchange, alumina futures manual: https://www.shfe.com.cn/products/futures/metal/nonferrousmetal/ao_f/manual/202306/P020240320711871173139.pdf
- Guangzhou Futures Exchange, lithium-carbonate contract: https://www.gfex.com.cn/gfex/sytslqhhy/202307/9ad927b8ec4c458594e172fd1ada2a9b.shtml
- Guangzhou Futures Exchange, industrial-silicon contract: https://www.gfex.com.cn/gfex/llbb/202402/73cd6f4cc26b4dd5b3d127b5462b59a7/files/%E5%B9%BF%E5%B7%9E%E6%9C%9F%E8%B4%A7%E4%BA%A4%E6%98%93%E6%89%80%E5%B7%A5%E4%B8%9A%E7%A1%85%E6%9C%9F%E8%B4%A7%E5%90%88%E7%BA%A6%EF%BC%882022%E5%B9%B412%E6%9C%8812%E6%97%A5%E7%89%88%EF%BC%89.pdf
- Guangzhou Futures Exchange, current polysilicon business-rule notice: https://www.gfex.com.cn/gfex/tzts/202604/8e2e876e29de41eeb6419f0dccdcdfe1.shtml
- Glass/soda-ash consumption reference: https://www.ccbfutures.com/upload/20211202/20211202143730891.pdf
