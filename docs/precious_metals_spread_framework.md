# Precious-Metals Arbitrage, Spread, and Premium Framework

Date: 2026-08-27

## 1. Scope

This framework covers gold, silver, platinum, and palladium. Rhodium is included
only as a low-liquidity industrial reference for the PGM complex.

The goal is to monitor five distinct metric families:

1. `location_premium`: the same metal at two physical locations;
2. `import_parity`: destination price minus fully landed and legally importable cost;
3. `futures_basis`: futures minus spot after carry and delivery normalization;
4. `form_quality_premium`: bar size, purity, brand, sponge/ingot, or allocated-status premium;
5. `intermetal_relative_value`: ratios or substitution spreads between metals.

A positive Shanghai-London price difference is a `headline_location_premium`.
It is not automatically an executable import arbitrage.

## 2. Common conventions

Constants:

```text
TROY_OZ_PER_KG = 32.1507465686
GRAMS_PER_TROY_OZ = 31.1034768
```

Conversions:

```text
cny_per_g_to_usd_per_oz(price, usdcny)
    = price * GRAMS_PER_TROY_OZ / usdcny

cny_per_kg_to_usd_per_oz(price, usdcny)
    = price / TROY_OZ_PER_KG / usdcny
```

Every metric must state:

- benchmark and timestamp;
- spot, deferred, or futures tenor;
- delivery location and vault system;
- allocated or unallocated metal;
- bar/plate form, weight, fineness, and acceptable refiner/brand;
- currency conversion timestamp or matching FX forward;
- tax basis and whether physical withdrawal occurs;
- freight, insurance, assay, refining/recasting, vault and financing cost;
- import/export eligibility or quota constraint;
- whether the metric is `headline`, `landed`, or `executable`.

## 3. Gold

### 3.1 Priority metrics

| Priority | ID | Core formula | Interpretation |
|---|---|---|---|
| P0 | `gold_shanghai_london_premium` | `SGE_Au9999_USD_oz - LBMA_Gold_USD_oz` | Mainland physical tightness versus Loco London |
| P0 | `gold_shanghai_london_premium_pct` | `gold_shanghai_london_premium / LBMA_Gold` | Scale-normalized Shanghai premium |
| P0 | `gold_sge_deferred_spot_basis` | `SGE_AuTD - SGE_Au9999` | Domestic deferred-market funding/physical signal |
| P0 | `gold_shfe_sge_basis` | `SHFE_AU_matched - SGE_spot_or_forward_equivalent` | Domestic futures carry and convergence |
| P0 | `gold_comex_london_efp` | `COMEX_GC - London_spot_forward_equivalent` | New York futures versus London OTC location/carry basis |
| P0 | `gold_forward_spot_basis` | `Gold_forward(T) - Gold_spot` | Funding, lease-rate and balance-sheet signal |
| P1 | `gold_shfe_comex_cross_market_basis` | `SHFE_AU_USD_oz - COMEX_GC`, maturity and FX-forward aligned | China versus New York futures relative value |
| P1 | `gold_sgei_mainboard_premium` | `SGE_main_board - SGE_international_board`, same form | Onshore versus bonded/international-board tightness |
| P1 | `gold_hongkong_shanghai_premium` | `SGE_main_board - SGE_HK_or_bonded_equivalent` | Mainland versus offshore Chinese gold |
| P1 | `gold_kilobar_lgd_premium` | `99.99% kilobar - 400oz London Good Delivery equivalent` | Asian fabrication and bar-form demand |
| P2 | `gold_retail_bar_coin_premium` | `dealer product price - fine-gold wholesale value` | Retail fabrication/distribution stress |
| P2 | `gold_etf_nav_premium` | `ETF_market_price / NAV - 1` | Fund plumbing/liquidity; not a metal import arbitrage |

### 3.2 Shanghai-London calculation layers

Headline premium:

```text
sge_gold_usd_oz = SGE_Au9999_cny_g * 31.1034768 / USDCNY_spot
headline_premium = sge_gold_usd_oz - LBMA_gold_spot_usd_oz
```

Indicative landed premium:

```text
landed_london_gold_cn = LBMA_gold
                      + kilobar_conversion
                      + freight_insurance
                      + vault_assay_handling
                      + financing_and_fx_hedge
                      + applicable_tax

landed_premium = sge_gold_usd_oz - landed_london_gold_cn
```

Executable arbitrage additionally requires an eligible importer, available
quota/approval, acceptable bar/refiner, a valid import and delivery channel,
matched timestamps, and sufficient bid/offer depth. Store this separately:

```text
executable_arb = landed_premium - execution_slippage - regulatory_scarcity_cost
```

Do not assign a constant value to `regulatory_scarcity_cost`; use an
`executable_flag` and reason code when it cannot be observed.

### 3.3 Gold forward and lease metrics

Approximate annualized implied lease rate:

```text
implied_gold_lease_rate(T)
    = usd_funding_rate(T) - ln(Gold_forward(T) / Gold_spot) / year_fraction(T)
```

This requires matched spot/forward timestamps, consistent settlement dates,
and the relevant secured/unsecured funding curve. Negative or extreme readings
can be caused by stale prices, date mismatch, balance-sheet constraints, or bar
location/form shortages rather than a clean lending opportunity.

Recommended related diagnostics:

- COMEX calendar spreads and annualized basis;
- SGE Au(T+D) deferred compensation/funding fields where available;
- SGE international-board quoted lease rates;
- COMEX registered/eligible stocks and London vault data;
- bid/ask width and cross-market trading-hour overlap.

## 4. Silver

### 4.1 Priority metrics

| Priority | ID | Core formula | Interpretation |
|---|---|---|---|
| P0 | `silver_shanghai_london_premium` | `SGE_Ag9999_USD_oz - LBMA_Silver_USD_oz` | China physical premium before landed-cost adjustment |
| P0 | `silver_shfe_london_basis` | `SHFE_AG_USD_oz - London_forward_equivalent` | China futures versus London silver |
| P0 | `silver_shfe_comex_basis` | `SHFE_AG_USD_oz - COMEX_SI`, maturity/FX aligned | Cross-exchange relative value |
| P0 | `silver_sge_deferred_spot_basis` | `SGE_AgTD - SGE_Ag9999` | Domestic funding and physical tightness |
| P0 | `silver_comex_london_efp` | `COMEX_SI - London_spot_forward_equivalent` | New York futures versus Loco London |
| P1 | `silver_import_parity_cn` | `China_domestic - fully_landed_London_or_origin_silver` | Import incentive after VAT/tax and logistics |
| P1 | `silver_forward_spot_basis` | `Silver_forward(T) - Silver_spot` | Silver funding/lease and inventory tightness |
| P1 | `silver_bar_form_premium` | `China_15kg_or_small_bar - London_1000oz_equivalent` | Recasting, fineness, and local-form demand |
| P2 | `silver_retail_premium` | `retail_bar_or_coin - wholesale_fine_silver` | Retail fabrication stress |

SHFE/SGE conversion:

```text
china_silver_usd_oz
    = silver_cny_kg / 32.1507465686 / USDCNY
```

Silver import parity is more tax-sensitive than the headline gold premium.
Domestic and overseas series must be compared on the same VAT basis, and tax
parameters must be effective-dated. Do not simply multiply every overseas
quote by the current headline VAT rate without checking the contract and
invoice treatment.

Useful physical diagnostics:

- SGE and SHFE inventories/warrants;
- COMEX registered versus eligible inventory;
- LBMA London vault holdings;
- photovoltaic and electronics demand proxies;
- gold/silver ratio, because silver combines monetary and industrial exposure.

## 5. Platinum, palladium, and the PGM complex

### 5.1 Platinum priority metrics

| Priority | ID | Core formula | Interpretation |
|---|---|---|---|
| P0 | `platinum_sge_london_premium` | `SGE_Pt9995_USD_oz - LPPM_Pt_USD_oz` | China platinum premium before landed costs |
| P0 | `platinum_nymex_lppm_basis` | `NYMEX_PL - LPPM_forward_equivalent` | New York futures versus London/Zurich physical metal |
| P0 | `platinum_tocom_lppm_basis` | `TOCOM_Pt_USD_oz - LPPM_forward_equivalent` | Japan futures/spot versus London/Zurich |
| P1 | `platinum_sge_tocom_basis` | `SGE_Pt9995 - TOCOM_Pt`, FX and form aligned | China versus Japan physical/derivatives market |
| P1 | `platinum_london_zurich_premium` | `Loco_London - Loco_Zurich` | PGM location and clearing tightness |
| P1 | `platinum_sponge_ingot_premium` | `Pt_sponge - Pt_ingot_equivalent` | Industrial form availability and fabrication |
| P2 | `platinum_jewelry_premium_cn` | `fabricated_product - Pt9995_fine_value` | Fabrication/retail demand, not wholesale arbitrage |

SGE Pt99.95 is quoted in CNY/gram, so:

```text
sge_platinum_usd_oz = Pt9995_cny_g * 31.1034768 / USDCNY
```

Platinum comparisons must specify Loco London or Loco Zurich. The LPPM Good
Delivery standard covers plates/ingots of 1-6 kg and minimum 99.95% fineness;
SGE Pt99.95 accepts 0.5-6 kg ingots and minimum 99.95% fineness. The overlap is
helpful, but refiner acceptance, vault, tax and import status still require
normalization.

### 5.2 Palladium and PGM substitution metrics

| Priority | ID | Formula | Interpretation |
|---|---|---|---|
| P0 | `platinum_palladium_spread` | `Platinum - Palladium` in USD/fine oz | Autocatalyst substitution and relative scarcity |
| P1 | `palladium_nymex_lppm_basis` | `NYMEX_PA - LPPM_Pd_forward_equivalent` | Futures versus London/Zurich physical metal |
| P1 | `palladium_london_zurich_premium` | `Loco_London - Loco_Zurich` | PGM location tightness |
| P1 | `palladium_sponge_ingot_premium` | `Pd_sponge - Pd_ingot_equivalent` | Industrial form tightness |
| P1 | `pgm_catalyst_basket_value` | `w_pt*Pt + w_pd*Pd + w_rh*Rh` | Technology-specific catalyst input value |
| P2 | `palladium_rhodium_ratio` | `Palladium / Rhodium` | Broader gasoline-catalyst relative value |

There is no equally liquid mainland Chinese palladium futures benchmark. A
China palladium premium will usually depend on dealer/industrial physical
quotes and should be flagged lower-confidence than exchange-backed gold,
silver, or SGE platinum metrics.

Rhodium should not be modeled like an exchange-traded futures asset. Use
assessed physical prices, wider stale-price tolerances, and explicit liquidity
flags.

## 6. Cross-metal relative-value metrics

| Priority | ID | Formula | Main interpretation |
|---|---|---|---|
| P0 | `gold_silver_ratio` | `Gold_USD_oz / Silver_USD_oz` | Monetary versus monetary-industrial regime |
| P0 | `gold_platinum_ratio` | `Gold_USD_oz / Platinum_USD_oz` | Defensive monetary demand versus cyclical PGM demand |
| P0 | `platinum_palladium_ratio` | `Platinum_USD_oz / Palladium_USD_oz` | Autocatalyst substitution/relative scarcity |
| P1 | `silver_platinum_ratio` | `Silver_USD_oz / Platinum_USD_oz` | Industrial precious-metal relative value |
| P1 | `precious_metals_dispersion` | cross-sectional z-score dispersion of Au/Ag/Pt/Pd | Relative-value opportunity/regime intensity |
| P1 | `monetary_vs_pgm_basket` | `Gold - beta*(Pt/Pd basket)` | Monetary versus industrial precious metals |

Ratios should use the same timestamp and location convention, preferably
Loco-London/LPPM USD per fine troy ounce or synchronized futures prices.
For signal generation, retain both the raw ratio and log ratio.

## 7. Product, location, and custody premiums

The benchmark price normally assumes a particular delivery location and Good
Delivery form. Track these premiums separately:

| Premium | Examples | Main drivers |
|---|---|---|
| Location | London-New York, London-Zurich, London-Shanghai, Shanghai-Hong Kong | freight, vault stocks, clearing, regulation, import access |
| Bar size | London gold 400oz versus Asian 1kg; London silver ~1000oz versus China 15kg | recasting capacity, fabrication demand, transport |
| Fineness | gold 995 versus 9999; silver 999 versus 9999; PGM 9995 | refining yield, accepted brands, end use |
| Form | PGM sponge versus ingot/plate; grain versus bar | industrial immediacy and fabrication bottlenecks |
| Custody | allocated versus unallocated; warranted versus eligible | credit, funding, vault and mobilization constraints |
| Brand | Good Delivery/approved refiner versus non-approved material | assay, upgrading, responsible-sourcing and acceptance risk |

## 8. Recommended implementation order

### Phase A: immediately computable from existing mappings

The repository already maps:

- SGE Au99.99 and Au(T+D);
- SGE Ag99.99 and Ag(T+D);
- London spot gold and silver;
- SGE silver inventory and gold futures warrants;
- a gold ETF holding series;
- China and US futures histories for gold and silver;
- US platinum and palladium futures histories in `bktest/data`.

Implement first:

1. `gold_shanghai_london_premium` and percentage premium;
2. `silver_shanghai_london_premium` with explicit VAT/tax-basis metadata;
3. SGE deferred-versus-spot bases for gold and silver;
4. SHFE-SGE domestic bases;
5. gold/silver, gold/platinum, gold/PGM, and platinum/palladium ratios;
6. futures calendar basis and annualized carry for AU, AG, GC, SI, PL and PA.

### Phase B: benchmark and physical-market enrichment

Add:

- LBMA Gold and Silver benchmark/spot or licensed redistributor series;
- LPPM platinum and palladium prices with explicit London/Zurich location;
- COMEX London Spot Spread/EFP or sufficient synchronized spot/futures inputs;
- SGE international-board and Hong Kong gold contracts;
- SGE Pt99.95 and TOCOM/JPX platinum;
- USD/CNY spot and tenor-matched FX forwards;
- benchmark interest rates, precious-metal forwards, vault and inventory data.

### Phase C: executable landed arbitrage

Build an effective-dated landed-cost and eligibility layer containing:

- importer/exporter eligibility and approval state;
- bar/refiner acceptance matrix by venue;
- tariff and VAT treatment by metal, venue, trade type and withdrawal status;
- freight, insurance, recasting, assay, vault, financing and FX hedge cost;
- delivery calendar, settlement lag and market-hours overlap;
- executable bid/offer depth and maximum practical size.

## 9. Required output fields

```text
metric_id
timestamp
metal
metric_family
headline_value
landed_value
executable_value
value_pct
source_market_1
source_market_2
location_1
location_2
form_1
form_2
fineness_1
fineness_2
fx_basis
tax_basis
tenor
bar_conversion_cost
freight_insurance_cost
financing_cost
other_cost
executable_flag
non_executable_reason
input_freshness
calculation_quality
```

## 10. Guardrails

1. Do not call a converted Shanghai-London price difference an import arbitrage
   without import eligibility and landed costs.
2. Do not compare London unallocated spot directly with a specific allocated
   bar without form and custody adjustments.
3. Do not mix spot and futures or different maturities without carry alignment.
4. Do not use spot USD/CNY to compare long-dated CNY and USD futures; use a
   tenor-matched FX forward or explicitly decompose the currency basis.
5. Do not apply one tax rule to gold, silver and PGMs. Tax treatment and
   physical-withdrawal status differ and can change.
6. Do not mix grams, kilograms and troy ounces without explicit conversion.
7. Do not treat COMEX eligible inventory as immediately deliverable registered
   inventory.
8. Do not interpret platinum-palladium substitution without accounting for
   catalyst technology, emissions rules, loadings, rhodium and qualification lag.
9. Do not forward-fill illiquid PGM and rhodium quotes indefinitely; attach
   stale-price and liquidity flags.

## 11. Primary source anchors

- LBMA, Loco London market and standard delivery location:
  https://www.lbma.org.uk/market-standards/about-loco-london
- LBMA, benchmark prices and auction timing:
  https://www.lbma.org.uk/prices-and-data/about-lbma-daily-auction-prices
- LBMA, price effects of location, bar size, form and fineness:
  https://www.lbma.org.uk/publications/the-otc-guide/the-price
- LBMA, gold and silver Good Delivery:
  https://www.lbma.org.uk/good-delivery/about-good-delivery
- CME, London Spot Spread link between London OTC and COMEX futures:
  https://www.cmegroup.com/education/articles-and-reports/spot-spread-faq
- CME, gold enhanced-delivery bar-size specifications:
  https://www.cmegroup.com/trading/metals/precious/faq-gold-enhanced-delivery-futures.html
- CME, silver futures size and fineness:
  https://www.cmegroup.com/trading/metals/files/fact-card-silver-futures-options.pdf
- CME, platinum contract overview:
  https://www.cmegroup.com/education/lessons/platinum-product-overview
- SGE, international-board iAu99.99 specifications:
  https://en.sge.com.cn/upload/file/201906/10/fRETGQZuzTvg5wrx.pdf
- SGE, Pt99.95 specifications:
  https://en.sge.com.cn/eng_trading_ProductsIntroduce_Physicaldetails?pro_id=943330789087645696
- SGE, international-board gold lease-rate product:
  https://en.sge.com.cn/h5_trading_ProductsIntroduce_details2?pro_id=943325762509254656
- SGE, Hong Kong-vault gold contracts launched in 2025:
  https://en.sge.com.cn/eng_news_Announcement/10002290
- SHFE, silver futures quotation basis:
  https://tsite.shfe.com.cn/eng/services/rules/shfe/911404889.html
- China State Taxation Administration, effective gold tax policy published in
  2025:
  https://shanghai.chinatax.gov.cn/zcfw/zcfgk/zzs/202511/t478143.html
- PBOC, gold import/export approval framework:
  https://dalian.pbc.gov.cn/dalian/123812/123830/123800/2025092315121774417/index.html
- LPPM, platinum and palladium London/Zurich Good Delivery standards:
  https://www.lppm.com/good-delivery/good-delivery-rules
- JPX, platinum rolling-spot contract specifications:
  https://www.jpx.co.jp/english/derivatives/products/precious-metals/platinum-rolling-spot-futures/01.html
