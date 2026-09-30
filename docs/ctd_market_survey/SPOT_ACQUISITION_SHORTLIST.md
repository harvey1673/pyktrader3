# Sparse spot-price acquisition shortlist

22 September 2026. Scope: J/JM/SS/SM/SF, history requested from 2016 or the first genuine observation. Recommendations concern signal inputs, not a certified deliverable basket. Priority is an economic judgment; incremental signal performance has not been measured.

## Recommended requests

| Priority | Market | Request to provider | Gap and intended use |
|---|---|---|---|
| 1 | JM | 蒙5#精煤：沙河驿：到厂/到站现货价，含税，明确付款方式、标准煤质及其变更日期 | No explicitly identified Mongolian No.5 washed-coal series found. Existing Ganqimaodu quotes lack this identity and complete assay. Prefer one delivered quote to absorb the main inland transport effect; replace the generic Mongolia leg rather than add another leg. If unavailable, request 甘其毛都蒙5#精煤含税库提价 with equivalent metadata, and use a documented freight assumption. Do not request both initially. |
| 2 | SF | 天津72#硅铁合格块现货价：含税现汇，明确粒度及仓库自提/到货口径 | Not found. Current sources cover national72 and Ningxia/Inner Mongolia/Gansu exworks72. A Tianjin quote reduces dependence on the national-price interpretation and a guessed constant freight addition. Request 2016 onward if genuinely available; retain national72 for earlier unavailable history. |
| 3, conditional | SS | 无锡：宏旺304/2B冷轧卷：2.0mm：毛边：现货含税价，固定宽度并记录宽度变更 | No brand-specific price found. Prefer clarifying existing iFinD S004785205 first; if its definition already matches a stable representative 2mm quote, no purchase is needed. Otherwise replace the aggregate with one brand, not a multi-mill minimum. Start from SS listing in 2019; older history is optional context. |
| 4, optional | J | 日照港准一级干熄焦：贸易现汇含税出库价，明确水分和标准煤质 | No explicitly dry-quenched port quote found. This can complement the existing wet-quench port series, capturing a second meaningful route/quality spread. Defer until the existing cash ex-warehouse quote and its conversion are working. EN's dry-basis quotation is not proof of dry-quenching. |
| No new request | SM | Retain 天津FeMn65Si17, S002959498 | Already present, with Inner Mongolia and Gansu alternatives. No additional provinces needed for the initial sparse signal. |

For JM, acquire the underlying spot price, not a broker's calculated warehouse-receipt/盘面折算价, which may embed changing assumptions. A label such as 蒙5 is not by itself a complete deliverability specification; record its standard assay and changes without trying to model every batch.

## Existing prices to use or refresh before buying more

The additional workbook scan changes the earlier nine-sheet assessment. In `mysteel data (base ferrous).xlsx`, sheet `mysteel prices`:

| Column | Existing series | Cached numeric coverage from 2016 onward | Action |
|---|---|---|---|
| FU, dates FS | 日照港准一级焦出库价格指数；湿吨、含税、现金 | 2019-06-12 to 2026-09-01 | Strong candidate to replace the FOB quote in the recent J benchmark. It is already available; normalize costs before comparing/bridging. |
| FT, dates FS | 日照港准一级焦平仓价格指数；湿吨、承兑含税 | 2017-05-23 to 2026-09-01 | Existing overlap/reference. Payment and loading conventions differ from FU; do not splice raw levels. |
| N, dates M | 日照港准一级焦旧汇总价 | 2016-01-04 to 2022-04-29 | Historical reference; marked stopped. Inspect methodology before using as early history. |
| AF, dates AE | 柳林主焦煤 A10.5/S1.3/G75；出厂承兑含税 | 2018-01-15 to 2022-02-08 | Already present but stopped. Ask for successor/refresh if domestic mid-sulfur representation is desired; evaluate as a replacement for the domestic JM leg, not a third permanent leg. |
| EN, dates EM | 日照浩宇准一级焦 A13/S0.72/MT0/CSR60/CRI30；现金含税、干基 | 2017-12-21 to 2026-09-01 | Existing producer quote. Dry-basis pricing does not establish dry-quenching; confirm before using as the dry-quench alternative. |
| DT, dates DR | 无锡304/2B、2mm汇总价 | No numeric observations found | A header exists, but no usable cached series. Brand and edge definition also absent. |

The older `mysteel data.xlsx` repeats these candidates with shorter recent coverage for several blocks. Each Mysteel export block has its own date column; worksheet column A is not a shared date index. The scan uses block-local dates. Header-only entries and stopped series do not count as live inputs. Workbook contents are cached snapshots, not proof of vendor subscription entitlement or original publication vintages.

## Evidence and reproducibility

All sheets in the eight local iFinD/Mysteel workbooks were scanned for relevant readable metadata. Selected additional Mysteel candidates were then checked against their own date columns. Garbled labels were supplemented with the readable first-row names. No exact provider identifiers are invented for missing series; these are search/request specifications. Absence means not identified in these snapshots, not unavailable from the vendor.

- [Mysteel product directory](https://list1.m.mysteel.com/dh/114/l1.html) lists a Mongolian No.5 Shaheyi spot specification, supporting the request for an explicitly identified series; exact delivery/payment conventions still require confirmation.
- [Five Minmetals Futures, 15 June 2026](https://www.wkjyqh.com/ueditor/jsp/upload/file/20260615/1781483552317089224.pdf) uses Tianjin72 ferrosilicon in its spot/futures comparison.
- [Mysteel Wuxi stainless basis comparison, 27 April 2026](https://bxg.mysteel.com/a/26042715/B586C7915ACC8074.html) distinguishes 304/2B2.0 coil by mill, including Hongwang. This supports a precise single-brand request, not a claim that one brand is always cheapest.
- [Mysteel port coke quotation conventions](https://list1.mysteel.com/zhishi/jiaotangangkoujia.html) distinguish wet/dry-quench and trade cash ex-warehouse versus factory acceptance FOB prices.

Reproduce local checks with `audit_additional_workbooks.py`. Inspect `priority_spot_audit/all_workbook_headers.json` and `priority_spot_audit/additional_mysteel_profiles.json`. Source workbooks and production signals were not modified. Earlier audit totals remain limited to their stated nine-sheet scope; these additional series are not yet merged into the raw signal panel.

Acquisition sequence: JM and SF first; clarify SS metadata before purchase; use existing J/FU and refresh domestic JM only as needed. Aim to retain about six core inputs across five markets. Do not automatically expand the basket with every acquisition.
