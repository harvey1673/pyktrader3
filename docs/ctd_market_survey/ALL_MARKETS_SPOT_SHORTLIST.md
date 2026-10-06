# China futures physical-carry spot shortlist

Status: research specification, not a production mapping.  Start date target is
2016 where the futures contract and source history permit it.  The executable
catalog is in `tests/ctd_spot_catalog.py`.

## Selection rule

The objective is a stable physical-carry signal, not a perfect daily auction of
every deliverable brand.  For each futures code the shortlist therefore keeps:

1. one liquid, interpretable preferred spot price;
2. one fallback only when it extends history or captures a material location or
   grade alternative; and
3. an explicit transformation from the vendor quote to futures-equivalent cost.

Never take a minimum across prices with different tax, payment, delivery,
currency, wet/dry, bonded, or timing conventions before normalising them.  A
fallback is not automatically a second CTD candidate.

`local alias` below means an existing alias was previously identified in the
local price maps.  A blank value means "not yet confirmed locally", not proof
that the series is absent from the source workbook.

## Ferrous and industrial materials

| Code | Preferred spot | Fallback | Preferred source | Local alias | Main conversion |
|---|---|---|---|---|---|
| rb | Shanghai HRB400E 20mm rebar | Hangzhou same specification | Mysteel | `rebar_sh` | brand, theoretical/actual weight, warehouse |
| hc | Shanghai Q235B 4.75mm HRC | Tianjin same specification | Mysteel | `hrc_sh` | thickness, width, brand, warehouse |
| i | Qingdao PB fines cash price, wet tonne | BRBF/IOC6/SSF small basket | Mysteel | `io_ctd_spot` | contract-era brand and chemistry penalties; wet-to-dry |
| j | Rizhao quasi-grade-1 wet-quench coke, cash tax-included ex-warehouse | Luliang quasi-grade-1 dry-quench ex-works | Mysteel | `coke_sub_a_rz` | moisture, quality, freight; do not mix with port closing/FOB quote |
| jm | Tangshan pickup Mongolian No.5 washed coal, **ID01892939** | Jiexiu medium-sulfur prime coking coal; Xiaoyi low-sulfur for backfill | Mysteel | `ckc_outstock_ganqimaodu` | quality and delivery-location cost; Shaheyi ends in 2024 and is not the core leg |
| SM | Tianjin FeMn65Si17 | Ulanqab producer price | Mysteel | `SM_65s17_tj` | size, chemistry, payment term, freight |
| SF | Tianjin 72# qualified lump ferrosilicon | Ningxia Zhongwei producer price | Mysteel | `SF_72_ningxia` | size, impurities, payment term, freight |
| SA | Shahe heavy soda ash | Shandong heavy soda ash | Mysteel | `SA_heavy_shahe` | heavy grade only; ex-works/delivered and warehouse |
| FG | Shahe 5mm large-sheet float glass | Hubei same specification | Mysteel | `FG_5mm_shahe` | specification, weight convention, factory warehouse |
| v | East China carbide-process PVC SG-5 | North China SG-5 | Mysteel | `pvc_cac2_east` | deliverable producer/grade, pickup location |
| SH | Shandong 32% liquid caustic soda divided by 0.32 | Shandong 50% divided by 0.50 | Mysteel | `SH_ctd_spot` | dry-basis conversion, contract-era concentration/location adjustment |

For `j`, the user-located Rizhao cash ex-warehouse series is the core physical
carry input.  The Rizhao closing/FOB series is an acceptance diagnostic, not a
competing CTD price.  For `jm`, retain both a domestic Shanxi leg and Mongolian
No.5, but compare them only after delivered-to-exchange-location conversion.

## Metals and new-energy materials

| Code | Preferred spot | Fallback | Preferred source | Local alias | Main conversion |
|---|---|---|---|---|---|
| cu | Shanghai SMM #1 copper | Changjiang #1 copper | iFinD | `cu_smm1_spot` | registered brand and warehouse |
| al | Shanghai SMM A00 aluminium | Changjiang A00 | iFinD | `al_smm0_spot` | location, brand, warehouse |
| zn | Shanghai SMM #0 zinc | Shanghai #1 zinc | iFinD | `zn_smm0_spot` | #0 core; brand and warehouse |
| ni | Shanghai SMM #1 refined nickel | deliverable Jinchuan/Russian brand quote | iFinD | `ni_smm1_spot` | preserve brand premium |
| pb | Shanghai SMM #1 lead | East China 99.994% lead | iFinD | `pb_994_shmet_east` | primary/recycled, brand, warehouse |
| sn | Shanghai SMM #1 tin | Changjiang #1 tin | iFinD | `sn_smm1_spot` | registered brand and warehouse |
| ss | Wuxi Hongwang 304/2B, 2×1240×C | lowest Wuxi same-spec deliverable brand | Mysteel | `ss_304_gross_wuxi` | confirm mill edge; apply contract-era edge adjustment |
| ao | Shandong AO-1/registered-brand alumina | Shanxi or Henan same grade | Mysteel | `alumina_spot_qd` | registered brand, location and packaging |
| au | SGE Au99.99 physical close/weighted average | SGE Au99.95 | iFinD | `au_9999_sge_close` | unit and fineness |
| ag | Shanghai SMM #1 silver | SGE Ag99.99 physical | iFinD | — | use physical silver; do not substitute Ag(T+D) without labeling it |
| si | Sichuan 553 non-oxygen silicon | Sichuan 421 silicon | Mysteel | `si_ctd_spot` | transform each grade under the applicable contract rules, then minimise |
| lc | domestic battery-grade lithium carbonate | domestic industrial-grade | iFinD | `lc_ctd_spot` | contract-era grade premium, packaging, brand, location |
| ps | N-type dense/mixed block polysilicon | N-type granular polysilicon | iFinD | — | verify benchmark-grade transition for the exact contract before coding |

The stainless series supplied by the user is a better core input than a Wuxi
city aggregate because it preserves producer, finish, width and edge type.  The
exact vendor indicator code and history start still need recording.

## Rubber, pulp and fertilizer

| Code | Preferred spot | Fallback | Preferred source | Local alias | Main conversion |
|---|---|---|---|---|---|
| ru | East China registered SCR WF full-ribbed rubber | Kunming SCR WF | Mysteel | `ru_scrwf_kunming` | production year, region discount and freight |
| UR | Shandong small-granule urea | Henan small-granule urea | Mysteel | `UR_shandong_spot` | nitrogen, size, ex-works/delivered, factory warehouse |
| sp | Shandong imported softwood pulp, Silver Star brand | Jiangsu/Shanghai Silver Star | Mysteel | — | registered brand/origin and warehouse |
| nr | Qingdao bonded STR20/SIR20 USD spot | Qingdao non-bonded RMB TSR20 | Mysteel | — | FX, tax, financing, bonded warehouse, registered origin |
| br | Shandong BR9000 registered-brand spot | East China BR9000 | Mysteel | — | certified brand/grade, tax and warehouse |

## Petrochemicals and energy

| Code | Preferred spot | Fallback | Preferred source | Local alias | Main conversion |
|---|---|---|---|---|---|
| l | North China/Tianjin LLDPE 7042 deliverable producer | East China 7042 | Mysteel | `l_7042_tj` | producer, grade, tax, pickup location |
| pp | East China PP raffia T30S deliverable producer | Linyi/North China T30S | Mysteel | `pp_100ppi_spot` | replace generic index with producer/grade quote when available |
| TA | East China PTA spot pickup | main-port deliverable brand | Mysteel | `TA_east_spot` | payment, brand, warehouse |
| PX | East China domestic PX ex-warehouse | CFR China/Taiwan PX USD | Mysteel | — | imported landed cost: FX, duty/VAT, port, freight, financing |
| eg | East China main-port MEG tank spot | Zhangjiagang MEG | Mysteel | `eg_east_spot` | tank, payment, tax, grade |
| MA | Taicang/Jiangsu methanol ex-tank | southern Shandong ex-works | Mysteel | `MA_spot_jiangsu` | port preferred; inland freight and payment basis |
| eb | East China main-port styrene ex-tank | Jiangsu styrene | Mysteel | `eb_east_spot` | delivery month, tank, tax, grade |
| sc | landed-cost basket of deliverable medium-sour crude, including Oman/Dubai grades | INE bonded warrant/spot trade | iFinD | — | grade differential, FX, bbl/t, freight and bonded costs; Brent alone is unsuitable |
| lu | Singapore 0.5% LSFO cargo | Zhoushan bonded LSFO warrant/spot | iFinD | — | use cargo rather than bunker retail; FX, freight, bonded storage |
| bu | Shandong 70# heavy road bitumen, registered brand | East China same grade | Mysteel | `bu_heavy_shandong` | registered brand, ex-works/warehouse and location |
| fu | Singapore 380cst HSFO cargo | Zhoushan bonded HSFO warrant/spot | iFinD | `fo_180cst_xiamen` is only an old proxy | replace 180cst proxy; viscosity/sulfur, FX, freight, bonded storage |
| pg | Shandong civil LPG market index/arrival price | South China imported LPG landed cost | Mysteel | `pg_sd_spot_idx` | propane/butane composition, FX/tax/freight for import leg |

For `sc`, `lu`, and `fu`, the transformation is part of the data definition.
Store the original USD quote, FX fix, tax status and freight components rather
than only the final RMB number.  This prevents historical revisions in the
conversion assumptions from becoming invisible.

## Oils, meals, grains, soft commodities and livestock

| Code | Preferred spot | Fallback | Preferred source | Main conversion |
|---|---|---|---|---|
| m | Rizhao/East China 43% protein soybean meal | Dongguan 43% meal | iFinD | mill/brand, basis month, tax, warehouse |
| RM | Dongguan/Guangxi 36% protein rapeseed meal | Nantong 36% meal | iFinD | protein, moisture, region and warehouse |
| y | Zhangjiagang grade-1 soybean oil | Tianjin grade-1 soybean oil | iFinD | grade, bulk, tax and delivery warehouse |
| p | Guangzhou 24-degree palm olein | Zhangjiagang/Tianjin 24-degree | iFinD | 24-degree grade, port and warehouse |
| OI | East China grade-3 rapeseed oil | Guangxi grade-3 rapeseed oil | iFinD | contract-era grade, acid value, origin and warehouse |
| a | Heilongjiang domestic non-GMO grade-3 clean soybean | Beian/Harbin same grade | iFinD | protein, moisture, sound kernels, clean/raw grain, freight |
| b | Rizhao/Qingdao imported GMO soybean distribution price | Nantong imported soybean | iFinD | No.2 soybean quality, tax, port, quarantine and warehouse |
| c | Jinzhou Port grade-2 corn | Bayuquan grade-2 corn | iFinD | 14.5% moisture, test weight, mould/toxin and flat/warehouse basis |
| cs | Shandong grade-1 corn starch ex-works | Hebei grade-1 starch | iFinD | moisture, acidity, protein, specks, bag and freight |
| CJ | Xinjiang grade-1 grey jujube, Cangzhou market or delivery warehouse | Aksu/Ruoqiang origin | iFinD | count/kg, moisture, sugar, defective fruit, contract year |
| CF | CC Index 3128B | Xinjiang warehouse 3128B machine-picked cotton | iFinD | colour, length, micronaire, strength, preparation and warehouse |
| jd | Shandong Dezhou large fresh egg producer price | national producing-area average | iFinD | CNY/jin to CNY/500kg, packing, freshness and delivery cost |
| AP | Qixia 80# grade 1/2 bagged Fuji storage apple | Luochuan same grade | iFinD | diameter, colour, firmness, sugar, russet, cold-store cost |
| lh | Henan standard-weight external-three-way-cross live hog | national external-three-way-cross average | iFinD | weight, lean ratio, ex-farm/arrival and region |
| SR | Nanning grade-1 white sugar | Liuzhou grade-1 white sugar | iFinD | grade, tax, packaging, warehouse and producing/consuming region |
| PK | Henan oil-use peanut kernels | Shandong oil-use peanut kernels | iFinD | do not use food-grade Baisha quote; oil, moisture, acid value and impurities |

These agricultural prices should initially be treated as one-factor physical
benchmarks.  `jd`, `AP`, `CJ`, and `lh` have large non-storable or grading
components, so a CTD label would overstate precision.  Their first signal should
be named `physical_spot_basis` until delivery conversion is validated.

## Highest-priority series to request

The following series are the most valuable additions where no exact local alias
has yet been confirmed:

1. `j`: Luliang quasi-grade-1 **dry-quench** coke, cash tax-included ex-works;
   optionally Rizhao dry-quench if it has usable history.
2. `jm`: Jiexiu medium-sulfur prime coking coal plus the user-located Tangshan
   Mongolian No.5 **ID01892939**; retain the discontinued Shaheyi series only as
   a historical diagnostic.
3. `ss`: Wuxi Hongwang 304/2B 2×1240×C exact indicator code and history.
4. `SF`: Tianjin 72# qualified-lump spot, because the existing Ningxia series is
   a location fallback.
5. `ao`, `ps`, `sp`, `nr`, `br`: the exact registered-grade/brand series shown
   above.  Brand identity is material in these contracts.
6. `PX`, `sc`, `lu`, `fu`: both the raw international quote and every conversion
   component.  Do not source only a precomputed RMB series.
7. `ag`: SMM #1 physical silver; this avoids relying on a deferred product as
   spot.
8. The agricultural block from `m` through `PK`, starting with `CF 3128B`,
   Nanning grade-1 sugar, 43% soybean meal, grade-1 soybean oil, 24-degree palm
   oil, Jinzhou corn and Shandong corn starch.

## Source and implementation evidence

- SHFE explains that deliverable products across metals, steel, and energy
  chemicals use registered or recognised brands, so brand filtering is part of
  the price definition: [SHFE delivery information](https://www.shfe.com.cn/specialtopic/investor/delivery/).
- SHFE's standard-warrant guide confirms the warehouse-warrant framework across
  metals, precious metals, steel, rubber, pulp, fuel oil and crude oil:
  [standard warrant guide](https://www.shfe.com.cn/services/delivery/warehousewarrant1/).
- The SHFE BR rules require each warrant to contain one producer, brand, grade
  and package specification: [BR business rules](https://www.shfe.com.cn/regulation/exchangerules/historicalversion/202508/t20250807_828544.html).
- INE registrations show that 20-rubber is origin and factory specific; for
  example STR20 can be standard-price deliverable while separately approved
  TSR10 carries a discount: [STR20 registration](https://www.shfe.com.cn/publicnotice/notice/202412/t20241206_823600.html),
  [TSR10 discount example](https://www.shfe.com.cn/publicnotice/notice/202606/t20260601_831901.html).
- CZCE's published market overview confirms the listed physical-delivery
  contracts and contract units used in this scope: [CZCE market overview](https://www.czce.com.cn/cn/rootfiles/2025/07/04/1750859895384730-1750859895409197.pdf).

## Next validation pass

For every requested vendor indicator, record `indicator_id`, exact Chinese
label, unit, tax basis, payment basis, delivery basis, frequency, start date,
end date, and missing-day ratio.  Join sources by date only after those fields
match.  MySteel remains the preferred source when the same local alias exists
in both MySteel and iFinD, including days when the MySteel value is missing.

