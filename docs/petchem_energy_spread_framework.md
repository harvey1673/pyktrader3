# Petrochemical and Energy Margin/Spread Framework

Date: 2026-08-27

Related framework: `docs/precious_metals_spread_framework.md`

## 1. Objective

Extend the commodity-processing project beyond workbook-reconciled base-metal
cost models. The target is a trading-oriented metric library covering:

1. `processing_margin`: product value minus feedstock and variable conversion cost;
2. `location_spread`: the same commodity at two locations, preferably net of freight;
3. `import_parity`: domestic price minus tax/FX/freight-adjusted import cost;
4. `product_premium`: grade, quality, or substitutable-product price difference;
5. `route_switch`: relative economics of two production or consumption routes.

These types must not be mixed. A simple price difference can be an excellent
trading factor without being a plant gross margin.

## 2. Common calculation standard

Every metric should declare:

- price basis: futures, FOB, CFR, ex-tank, ex-warehouse, or ex-works;
- currency and unit;
- tax basis;
- location and delivery window;
- contract/laycan alignment rule;
- feedstock coefficient source and effective date;
- freight, port, finance, FX, tariff, VAT, and loss assumptions;
- whether the output is `physical_economics`, `tradable_proxy`, or `relative_value`.

Recommended outputs:

```text
spread_abs       = product_value - adjusted_input_value
spread_rate      = spread_abs / abs(adjusted_input_value)
spread_z         = rolling_zscore(spread_abs)
spread_percentile = rolling_percentile(spread_abs)
```

For physical margins, keep both a cash-variable-cost version and a fuller
version including utilities and fixed processing fees. Do not subtract an
arbitrary fixed fee from a relative-value spread used only as a trading signal.

## 3. Priority metric set

### 3.1 Petrochemical processing margins

#### Priority 0: first production set

| ID | Metric | Core formula | Main use | Main caveat |
|---|---|---|---|---|
| `px_naphtha_spread_asia` | Asian PX-naphtha spread | `PX_CFR_TW_CN - Naphtha_CFR_Japan` | Aromatics-chain cash proxy | Not a full reformer/aromatics-complex margin; co-products matter |
| `pta_px_margin_cn` | PTA processing margin | `PTA_CN - 0.655 * PX_import_parity_CN - variable_fee` | PTA run-rate and TA signal | Match PX laycan to PTA date; handle FX/VAT consistently |
| `polyester_melt_margin_cn` | Polyester melt margin | `Polyester_product - 0.855*PTA - 0.335*MEG - variable_fee` | Polyester-chain demand and PF signal | Coefficients and fees vary by product and plant |
| `pp_mto_margin_cn` | MTO-to-PP margin | `PP - 3.0*Methanol - variable_fee` | Coal/methanol olefin economics | PP-only attribution ignores PE and other MTO products |
| `pp_pdh_margin_cn` | PDH-to-PP margin | `PP - 1.18*Propane_import_parity - variable_fee` | PDH operating incentive | Propylene and PP stages should also be stored separately |
| `styrene_feed_margin_asia` | Styrene feedstock margin | `SM - a_BZ*Benzene - a_Ethylene*Ethylene - variable_fee` | Styrene run-rate and EB signal | Default coefficients must be calibrated; PO/SM route is separate |
| `aliphatic_fuel_cracker_proxy` | Naphtha cracker cash proxy | `weighted_olefin_aromatic_output - Naphtha - variable_cost` | Cracker utilization/regime | A single ethylene-naphtha spread is not a full cracker margin |

#### Priority 1: route and chain extensions

| ID | Metric | Core formula | Main use |
|---|---|---|---|
| `ethylene_naphtha_spread_asia` | `Ethylene_CFR_NEA - Naphtha_CFR_Japan` | Simple olefin cash proxy |
| `propylene_naphtha_spread_asia` | `Propylene_CFR_China - Naphtha_CFR_Japan` | Propylene tightness versus cracker feed |
| `benzene_naphtha_spread_asia` | `Benzene_FOB_Korea - Naphtha_CFR_Japan` | Aromatics value versus feed |
| `px_mx_spread_asia` | `PX_CFR_TW_CN - MX_FOB_Korea` | PX conversion/premium indicator |
| `eg_ethylene_margin_asia` | `MEG_CFR_China - a*Ethylene_CFR_NEA - fee` | Oil-route MEG economics |
| `eg_coal_margin_cn` | `MEG_CN - coal_syngas_cost - fee` | Coal-route MEG economics |
| `pvc_carbide_margin_cn` | `PVC - a*Carbide - power_chloralkali_cost - fee` | Northwest-China carbide route |
| `pvc_ethylene_margin_cn` | `PVC - a*Ethylene_import_parity - chlorine_cost - fee` | Coastal ethylene route |
| `soda_ash_glass_margin_cn` | `Glass - a*SodaAsh - fuel_cost - fee` | Glass production economics |
| `methanol_coal_margin_cn` | `Methanol - a*Coal - utility_cost - fee` | Coal-to-methanol run-rate |
| `methanol_gas_margin_intl` | `Methanol - gas_intensity*Gas - fee` | Gas-route competitiveness |

### 3.2 Petrochemical location and import-parity spreads

| Priority | ID | Core formula | Interpretation |
|---|---|---|---|
| P0 | `px_cfr_fob_freight_spread` | `PX_CFR_TW_CN - PX_FOB_Korea` | Near-haul freight/import pull; compare with assessed freight |
| P0 | `px_cn_import_parity` | `PX_CN_ex_tank - landed(PX_CFR_TW_CN)` | China import incentive and PTA feed-cost basis |
| P0 | `pta_domestic_location_spread` | `PTA_East - PTA_other_region` | Domestic logistics/tank tightness |
| P0 | `meg_cn_import_parity` | `MEG_East_CN - landed(MEG_CFR_China)` | Import incentive and port inventory pressure |
| P0 | `styrene_cn_import_parity` | `SM_East_CN - landed(SM_CFR_China)` | Import incentive and East-China balance |
| P1 | `ethylene_nea_sea_spread` | `Ethylene_CFR_NEA - Ethylene_CFR_SEA` | Inter-Asian olefin pull |
| P1 | `propylene_cfr_fob_spread` | `Propylene_CFR_China - Propylene_FOB_Korea` | China import pull versus Korea export value |
| P1 | `benzene_cfr_fob_spread` | `Benzene_CFR_China - Benzene_FOB_Korea` | China aromatics import pull |
| P1 | `polyolefin_cn_import_parity` | `Domestic_PE_or_PP - landed(CFR_FE_Asia)` | Import window by grade |
| P1 | `methanol_coastal_inland_spread` | `East_coast_Methanol - Inland_Methanol` | Freight, port inventory, and MTO demand |
| P2 | `asia_europe_chemical_arb` | `Asia_marker - Europe_marker - freight - finance` | Long-haul trade-flow incentive |

Generic landed-cost function:

```text
landed_cny_t = usd_cfr_t * usdcny
             * tariff_multiplier
             * vat_multiplier
             + port_fee_cny_t
             + finance_cny_t
             + quality_adjustment_cny_t
```

The tariff/VAT implementation must be commodity- and date-specific. Futures
prices must first be reconciled to their quoted tax basis; applying VAT as a
generic multiplier can double count tax.

### 3.3 Petrochemical product and route premiums

| Priority | ID | Formula | Economic meaning |
|---|---|---|---|
| P0 | `pp_lldpe_spread_cn` | `PP - LLDPE` | Polymer substitution and relative supply |
| P0 | `pta_meg_relative_tightness` | normalized `PTA` versus normalized `MEG` | Relative polyester feed tightness; not a margin |
| P1 | `ethylene_propylene_premium` | `Ethylene_CFR_NEA - Propylene_CFR_China` | Olefin product mix/tightness |
| P1 | `pe_grade_premium` | `LDPE_or_HDPE - LLDPE`, matched location | Grade-specific tightness |
| P1 | `pp_copolymer_premium` | `PP_copolymer - PP_homopolymer` | Higher-value grade premium |
| P1 | `benzene_toluene_spread` | `Benzene - Toluene`, matched basis | Aromatics conversion/substitution signal |
| P1 | `pdh_vs_mto_route_advantage` | `PP_PDH_margin - PP_MTO_margin` | Marginal route competitiveness |
| P2 | `oil_vs_coal_olefin_advantage` | comparable cracker margin minus CTO/MTO margin | Feedstock regime |

## 4. Energy priority metric set

### 4.1 Crude location and quality spreads

| Priority | ID | Formula | Main use |
|---|---|---|---|
| P0 | `brent_wti_spread` | `Brent - WTI` | Atlantic-basin location/export economics |
| P0 | `brent_dubai_efs` | `Brent - Dubai`, matched forward month | West-of-Suez versus East-of-Suez crude value |
| P0 | `sc_dubai_import_parity` | `SC - landed(Dubai/Oman basket)` | China bonded crude relative value |
| P1 | `dubai_oman_spread` | `Dubai - Oman` | Middle-East sour benchmark relative value |
| P1 | `murban_dubai_premium` | `Murban - Dubai` | Light-sour quality premium |
| P1 | `dated_forward_structure` | `Dated Brent - Brent future` | Prompt physical tightness |

SC is quoted in CNY/barrel and represents medium-sour crude. Do not convert it
to CNY/ton before comparing it with another barrel-denominated crude marker.
For product cracks quoted per metric ton, use product-specific density rather
than a universal `7.33 barrels/ton` coefficient.

### 4.2 Refining and product cracks

| Priority | ID | Formula | Main use |
|---|---|---|---|
| P0 | `singapore_gasoline92_dubai_crack` | `Gasoline92_FOB_Singapore - Dubai` in USD/bbl | Asian gasoline margin |
| P0 | `singapore_gasoil_dubai_crack` | `Gasoil10ppm_FOB_Singapore - Dubai` | Middle-distillate margin |
| P0 | `singapore_jet_dubai_crack` | `JetKero_FOB_Singapore - Dubai` | Jet margin |
| P0 | `singapore_naphtha_dubai_crack` | `Naphtha_CFR_Japan_or_Singapore - Dubai` | Light-distillate/petchem feed value |
| P0 | `singapore_hsfo_dubai_crack` | `HSFO380_FOB_Singapore - Dubai` | Residual fuel margin |
| P0 | `singapore_vlsfo_dubai_crack` | `MarineFuel0.5_FOB_Singapore - Dubai` | Low-sulfur bunker margin |
| P0 | `cn_fu_sc_crack` | `FU - density_adjusted_SC` | China high-sulfur fuel-oil proxy |
| P0 | `cn_lu_sc_crack` | `LU - density_adjusted_SC` | China low-sulfur fuel-oil proxy |
| P0 | `cn_bu_sc_crack` | `BU - density_adjusted_SC` | China bitumen refinery proxy |
| P1 | `us_321_crack` | `(2*RBOB + 1*ULSD - 3*Crude)/3` | US composite refining-margin proxy |
| P1 | `asia_composite_crack` | yield-weighted Asian product slate minus Dubai | Asian refinery proxy |

The composite cracks are gross proxies. They exclude many products, refinery
fuel, losses, credits, variable operating cost, and fixed cost.

### 4.3 Energy location and product premiums

| Priority | ID | Formula | Interpretation |
|---|---|---|---|
| P0 | `vlsfo_hsfo_spread_singapore` | `MarineFuel0.5 - HSFO380` | Sulfur-compliance/sweet-residue premium |
| P0 | `cn_lu_fu_spread` | `LU - FU`, tax and delivery aligned | China low- versus high-sulfur premium |
| P0 | `jet_gasoil_regrade_asia` | `JetKero - Gasoil10ppm` | Refinery yield-switch incentive |
| P0 | `gasoline_octane_spread_asia` | `Gasoline95 - Gasoline92` | Octane/blending-component value |
| P1 | `gasoil_east_west` | `Singapore_Gasoil - NWE_Gasoil - freight` | East-West distillate flow incentive |
| P1 | `fuel_oil_singapore_rotterdam` | `Singapore - Rotterdam - freight` | Bunker/residual flow incentive |
| P1 | `gasoline_asia_usgc_arb` | destination value minus origin and freight | Gasoline trade-flow incentive |
| P1 | `sulfur_grade_gasoil_premium` | `10ppm - higher_sulfur_grade` | Desulfurization and specification premium |
| P2 | `regional_bunker_premium` | port bunker price minus Singapore benchmark | Port/location tightness |

### 4.4 Natural gas, LNG, coal, power, and carbon

| Priority | ID | Formula | Interpretation |
|---|---|---|---|
| P0 | `jkm_ttf_spread` | `JKM - TTF`, both USD/MMBtu | Atlantic-Pacific LNG pull before freight |
| P0 | `jkm_hh_spread` | `JKM - HenryHub` | Gross US-to-Asia LNG value before liquefaction/freight |
| P0 | `ttf_hh_spread` | `TTF - HenryHub` | Gross US-to-Europe LNG value |
| P1 | `us_lng_netback_asia` | `JKM - HH_feedgas_factor*HH - liquefaction - freight - losses` | USGC-to-Asia cargo margin |
| P1 | `us_lng_netback_europe` | `NWE_LNG_or_TTF - HH_feedgas_factor*HH - liquefaction - freight - losses` | USGC-to-Europe cargo margin |
| P1 | `jkm_seam_wim_location_spreads` | `JKM - SEAM` and `JKM - WIM` | Asian LNG destination pull |
| P1 | `api2_api4_coal_spread` | `API2 - API4`, energy-normalized | Atlantic versus South-African coal value |
| P1 | `newcastle_qhd_import_parity` | `QHD - landed(Newcastle/Indonesia coal)` | China seaborne import incentive |
| P2 | `clean_spark_spread` | `Power - heat_rate*Gas - emissions*Carbon` | Gas-fired power cash margin |
| P2 | `clean_dark_spread` | `Power - heat_rate*Coal - emissions*Carbon` | Coal-fired power cash margin |
| P2 | `gas_coal_switch_spread` | heat-rate-adjusted gas generation cost minus coal generation cost | Fuel switching |

JKM is a delivered Northeast-Asia LNG marker, while TTF and Henry Hub are gas
hub prices. `JKM-TTF` and `JKM-HH` are therefore useful relative-value spreads,
not cargo netbacks until freight, liquefaction/regasification, losses, and timing
are included.

## 5. Recommended production order

### Phase A: native and already-near-mapped metrics

1. Repair `FU-SC`, `LU-SC`, and `BU-SC` unit conversion.
2. Add `LU-FU`, `PP-MA`, `PP-L`, `PTA-PX`, polyester melt margin, and
   `PX-naphtha` using a shared metric schema.
3. Store absolute spread, rate, z-score, percentile, input freshness, and a
   calculation-quality flag.

### Phase B: Asian physical benchmarks and import parity

1. Add PX FOB Korea/CFR Taiwan-China, naphtha CFR Japan, olefin, aromatics,
   MEG, styrene, propane, and polyolefin benchmark series.
2. Build one reusable landed-cost engine for FX, tariff, VAT, port, freight,
   financing, and quality adjustments.
3. Align physical laycans before calculating location spreads.

### Phase C: global energy relative value

1. Add Brent-WTI, Brent-Dubai EFS, SC-Dubai/Oman parity, and the Asian product
   crack stack.
2. Add JKM-TTF-Henry Hub and freight-adjusted LNG netbacks.
3. Add East-West refined-product spreads, coal import parity, and eventually
   clean spark/dark spreads where reliable power and carbon data exist.

## 6. Repository gap assessment

Current useful coverage in `tests/index_map_full.py` includes Dubai spot, dated
Brent, PX FOB Korea, PX CFR Taiwan, PTA East-China spot, PX East-China spot,
PTA margin/cost, PX margin, PX-naphtha spread, and PX-MX spread. This is a good
start for the aromatics/polyester chain.

Material gaps:

- no common metric metadata or landed-cost engine;
- no systematic freight, FX, tax, laycan, density, or quality normalization;
- limited olefin, aromatics, polyolefin, LPG/propane, LNG, gas, and refined-
  product benchmark mapping;
- `tests/supply_chain_spreads.py` uses raw `FU-SC` despite different units and
  simplifies polyester margin to `PF-PTA`;
- no distinction between physical margin, trading proxy, and relative value;
- no input-freshness or calculation-quality flags.

## 7. Source anchors

Primary methodology and contract references:

- S&P Global, Asia-Pacific Chemicals specifications (March 2026):
  https://www.spglobal.com/content/dam/spglobal/ci/en/documents/platts/en/our-methodology/methodology-specifications/chemicals/chemicals-asia-pacific-specifications.pdf
- S&P Global, LNG market benchmarks and methodology:
  https://www.spglobal.com/commodity-insights/en/products-solutions/lng/lng-market-data
- S&P Global, benchmark statements for Asian refined products:
  https://www.spglobal.com/commodity-insights/en/pricing-benchmarks/our-methodology/methodology-specifications/benchmark-statements
- U.S. EIA, petroleum product prices and crack spreads:
  https://www.eia.gov/finance/markets/products/prices.php
- U.S. EIA, 3:2:1 crack-spread definition and limitations:
  https://www.eia.gov/todayinenergy/includes/crackspread_explain.php
- ICE, TTF as a global natural-gas reference:
  https://www.ice.com/global-natural-gas-futures/ttf
- INE, crude-oil contract and deliverable crude grades:
  https://www.ine.cn/eng/market/futures/energy/sc/contract/index.html
  and https://www.ine.cn/eng/services/delivery/goods/
- INE, low-sulfur fuel-oil delivery specification:
  https://tsite.shfe.com.cn/eng/market/futures/energy/lu/appendixes/
- SHFE, fuel-oil futures rules and quotation basis:
  https://tsite.shfe.com.cn/eng/services/rules/shfe/911404890.html

## 8. Guardrails

1. Do not label a two-price proxy as a plant margin.
2. Do not subtract prices with incompatible currencies, tax bases, units, or
   delivery periods.
3. Do not use a universal crude barrels-per-ton conversion for all products.
4. Do not treat CFR-minus-FOB as pure freight without checking credit, timing,
   port, and specification normalization.
5. Do not use front contracts from different seasonal specifications without
   an explicit roll/alignment rule.
6. Version all coefficients rather than embedding them in lambda functions.
7. Preserve raw benchmark spreads alongside fully adjusted economic spreads;
   both can be useful signals, but they answer different questions.
