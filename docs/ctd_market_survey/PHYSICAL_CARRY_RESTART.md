# Physical carry spot research: restart assessment

**Research implementation available:** [CTD_IMPLEMENTATION.md](CTD_IMPLEMENTATION.md) documents the test-only J/JM/SS/SM/SF sparse CTD functions. Production folders and signal mappings remain unchanged.

Reviewed 2026-09-22 against local code. This is a repository audit and proposed research specification, not a revalidation of exchange rules or a completed spot-data study.

Follow-up: `PRIORITY_MARKETS_2016_REVIEW.md` now contains the first workbook/rule assessment and supersedes this initial brief where findings differ. User clarification: prioritize a small number of spot series capturing the major carry effects; exhaustive delivery precision is not the objective. The revised selection uses six core series and two optional comparators across J/JM/SS/SM/SF.

## Objective

For each uncovered commodity, choose a defensible historical spot input for physical carry: a representative cash benchmark, a futures-equivalent adjusted benchmark, or a minimum across verified deliverable candidates. Full CTD is useful only where it improves comparability enough to justify its data requirements. Preserve the existing metals/construction signals as the baseline. User-confirmed scope: target history from 2016, with metals and industrial materials first. Start each product at the later of 2016, its futures listing, and defensible data/rule availability; explicitly report any gap rather than extrapolating modern rules backwards.

## What exists

- `tests/ctd_adjustments.py`: experimental candidate normalization and minimum selection for all 21 originally requested products. It is not connected to the production carry mapping.
- `tests/test_ctd_adjustments.py`: arithmetic and regime tests. These do not establish that the underlying rules, quote metadata, or historical coverage are correct.
- `tests/ctd_spot_requirements.py` and `tests/CTD_SPOT_PRICE_REQUIREMENTS.md`: acquisition shortlist; JM coverage was understated and is corrected in this restart.
- `docs/ctd_market_survey/china_futures_ctd_survey.html`: earlier rule survey and external source links. Treat numerical rules as unverified until their sources and effective periods are checked again.
- `tests/index_map_full.py`: Excel-derived ticker catalogue. An alias proves a mapping exists, not that observations are available or suitable.
- `pycmqlib3/strategy/signal_repo.py`, `commod_phycarry_dict`: existing production selections.
- `misc_scripts/fun_factor_update.py` and `misc_scripts/historical_signal_generator.py`: live and historical carry calculations.
- Source workbook location recorded by the mapping generator: `C:/Users/harve/Nutstore/1/Nutstore`. The daily, full-history, data, stock and weekly ifind workbooks are present; their contents were not re-audited in this pass.

## Existing signal convention

Both carry paths use approximately:

`carry = log((spot + adder) / unadjusted_C1) * 365 / days_to_expiry + smoothed_r007`

They reconstruct C1 from `close / exp(shift)`. Both hard-code SF +350 and SM +190. Thus these two products already have a basic adjustment, not simply raw spot inputs. Any replacement must remove those adders for normalized series, or it will double-count them. Both paths forward-fill inputs without a quote-age limit; the historical path masks zero days, while the live path divides directly by days.

For this signal, construct a **current cash spot equivalent** by converting currency, units, tax convention, moisture/weight and immediate location/quality differences. Keep financing to expiry and future storage outside that spot series: financing is already present in the signal, and adding the full cash-and-carry cost would change its economic meaning. A separate delivery-arbitrage cost series can include these costs.

`spot_equivalent(t, contract) = converted_cash_quote + immediate_conversion_cost - exchange_premium`

`ctd_spot(t, contract) = minimum of valid, fresh, eligible spot_equivalent candidates`

Record whether the result is a benchmark, adjusted benchmark, or verified CTD; never silently relabel a broad regional quote as CTD. The minimum depends on the candidate universe and must retain the selected candidate and available-candidate count.

## Product worklist

These are research priorities inferred from local mappings, not confirmed delivery recommendations.

| Product | Existing carry input | Next research decision |
|---|---|---|
| J | coke_sub_a_rz | Establish wet/dry weight and full assay; compare adjusted Rizhao benchmark with inland-to-port alternatives. |
| JM | ckc_outstock_ganqimaodu | Inspect Ganqimaodu stock/outstock and Lvliang/Tangshan assay-labelled quotes before buying new series. Do not infer Mongolian No. 5 or washed status from the aliases. |
| SS | ss_304_gross_wuxi | Confirm thickness, edge, brand composition and weight convention; determine if adjusted benchmark is sufficient. Treat ss_304_wuxi_phybasis as a basis, not an outright price. |
| SM | SM_65s17_tj + 190 | Validate historical Tianjin adjustment; compare producer locations only with exact pickup points and freight. |
| SF | SF_72_ningxia + 350 | Establish whether regional Ningxia price represents the relevant delivery point; validate contract-specific adjustment history. |
| EG | eg_east_spot | Audit domestic tank/warehouse benchmark, tax and timestamp first. |
| EB | eb_east_spot | Audit East-China quote basis before expanding regional/import panel. |
| TA | TA_east_spot | Confirm brand/grade and pickup convention; compare second East-China series. |
| MA | MA_spot_jiangsu | Audit Jiangsu benchmark, then compare Zhejiang and inland prices after conversion. |
| L | l_7042_tj | Compare Tianjin with East/North/South 7042; establish producer/grade composition. |
| PP | pp_100ppi_spot | Inspect aggregation and grade; compare Linyi/Wenzhou alternatives. |
| V | pvc_cac2_east | Audit grade/process and compare regional/ethylene-route candidates on common basis. |
| BU | bu_heavy_shandong | Confirm producer/grade and pickup basis; compare North/East prices. |
| RU | ru_scrwf_kunming | Validate brand, production year and location adjustment; compare Zhejiang/Jiangsu. |
| FU | fo_180cst_xiamen | Explicitly audit grade and domestic/bonded basis compatibility before treating it as a futures-comparable spot. |
| PX | No entry | Audit PX_exw_east_spot first; retain CFR/FOB conversion as a separate candidate path. |
| UR | No entry | Compare Henan/Shandong/North quotes with grade and pickup metadata. |
| SC | No entry | Separate grade, currency, unit and bonded/import basis workstream. |
| LU | No entry | Establish cargo versus bunker convention and bonded delivery comparability. |
| NR | No entry | Find outright deliverable-grade quotes; inventory/warrant series cannot substitute for price. |
| BR | No entry | Find BR9000 outright quotes and producer/brand history. |

## Prototype gaps to resolve before promotion

1. SS implements a stated date-based notice using expiry month. Store observation/receipt-rule dates separately from delivery-contract boundaries; verify the actual notice and transition mechanism.
2. General petrochemical handlers permit some missing quality metadata and default zero adjustments. Zero must mean a verified zero, not an unknown premium or conversion.
3. Candidate specifications are static mappings. Historical assays, brands, locations and eligibility require effective intervals or dated observations.
4. J rejects contracts before J2201. JM incorrectly applies its 2018 rule branch to all pre-JM2304 contracts, including pre-JM1907. The requested 2016 start requires recovery and verification of earlier delivery standards and transitions; the current prototype does not meet that requirement.
5. The audit output lacks rule/source identifiers and rejection reasons, and missing price columns are silently skipped. Add winner, quote age, candidate coverage and missing-input diagnostics.
6. Rule selection must distinguish announcement date, effective date and effective contract. Backtests must not use later-known specifications or revisions prematurely.
7. Require a historical FX/unit/tax convention for imported/bonded candidates, rather than allowing an unexplained additive cash-cost scalar to stand in for conversion.
8. Take minima only after timestamp alignment, stale-quote exclusion and basis normalization. Retain a fixed benchmark alongside CTD to diagnose selection switches and changing coverage.

## Execution and acceptance

Start with J/JM/SS/SM/SF, consistent with the confirmed industrial-material priority. For each product, produce a small decision record containing exact ticker and workbook description, actual observation coverage since 2016 or listing, quote convention, current benchmark, adjustment rule sources and effective boundaries, missing inputs, and a proposed output series. Then process EG/EB/TA/MA and the remaining petrochemicals; isolate SC/FU/LU and rubber where conversion or data acquisition dominates.

For each product compare raw benchmark, adjusted benchmark and CTD where feasible on the same dates and C1 contracts. Inspect quote age, coverage, roll/expiry jumps, rule-change discontinuities, winner switches, and basis behavior near delivery. Select the input for economic comparability and reproducible history before testing strategy performance. Do not choose a spot series solely for the best backtest.

Production acceptance requires verified rule provenance, supported historical periods, sufficient metadata, explicit missing-data behavior, and agreement between live and historical carry paths. Tests should cover real rule boundaries, conversion signs, unavailable/stale inputs, and removal of legacy adders. No production signal mapping was changed during this restart audit.
