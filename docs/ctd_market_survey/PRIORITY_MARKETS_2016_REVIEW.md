# Spot inputs for J, JM, SS, SM and SF physical carry

**Additional workbook discovery:** See [SPOT_ACQUISITION_SHORTLIST.md](SPOT_ACQUISITION_SHORTLIST.md) for the subsequent all-workbook header scan and targeted Mysteel checks. In particular, J cash ex-warehouse FU already exists through 2026-09-01; the earlier inventory below covered only its stated nine sheets. The acquisition shortlist supersedes earlier missing-input recommendations.

Research date: 22 September 2026. Scope: spot selection and historical delivery-rule requirements from 2016, using the local workbook snapshots and exchange documents or broker-hosted reproductions. This completes the first source/data assessment; it does not establish a production CTD series or validate signal returns.

## Small-basket recommendation

The user clarified that the goal is to capture the major contribution to physical carry with a small number of spot prices, not to reproduce every delivery option. The recommendation below supersedes the broader candidate-acquisition agenda: use **six core series across five markets**, plus two optional comparison/early-history series. Full delivery certification is not an acceptance requirement for a research signal proxy. Assumptions must be explicit, and large systematic conversion errors still need correction.

| Market | Core series | Optional series | Major effect to retain | Simplification |
|---|---|---|---|---|
| J | coke_sub_a_rz | coke_sub_a_tj for 2016–2017 and overlap checks | Wet/dry accounting, old quality discount and J2604 Mf effect | Model the quoted standard assay; use one documented port-cost convention. No daily multi-origin CTD. |
| JM | ckc_a10v24s08_lvliang + ckc_stock_ganqimaodu | None in the core model | Domestic versus Mongolian sourcing, quality-regime shifts and a coarse transport/location conversion | Two-source proxy minimum only after normalization; use Xiaoyi alone when the Mongolian observation is missing/stale. |
| SS | ss_304_gross_wuxi | None | Mill-edge conversion; thickness assumption | Use a representative 2.0mm 304/2B assumption provisionally if provider clarification is unavailable; no registered-brand panel. |
| SM | sm_65s17_tj | None | Historical 0/150/190 location conversion | Single representative grade and pickup convention; no provincial minimum. |
| SF | sf_72_shmet | sf_72_ningxia from 2019-07-15 | Persistent national/producer basis and freight | Use the national series for a consistent 2016 benchmark; compare Ningxia separately before deciding whether it improves the signal. |

For JM, the Mongolia description does not identify No.5 or a complete assay. A representative washed-coal assay and fixed or slowly varying transport cost are reasonable **explicit research assumptions**, subject to a sensitivity check. They should not be presented as observed daily delivery specifications. If the ranking depends heavily on that assumption, retain Xiaoyi as the baseline and treat Mongolia as a separate carry feature instead of forcing a minimum.

Prefer assumptions that can be specified before examining trading returns. Compare a few economically grounded cost/quality scenarios and inspect whether carry sign/rank is robust. Only acquire another series when it changes the interpretation materially. Historical contract changes and long quote gaps are still important even in this simpler model.

The machine-readable selection is in `sparse_spot_selection.json`; `sparse_raw_spot_panel.csv` contains its eight core/optional columns with original gaps preserved. This is an input panel, not yet a futures-normalized or backtested signal.

## Broader candidates considered

| Market | Recommended research baseline | History available in inspected files | Decision |
|---|---|---|---|
| J | Rizhao quasi-grade-1, with contract-specific moisture/quality conversion; retain Tianjin as a separate early-history benchmark | Rizhao from 2018-01-02; Tianjin from 2016-01-04 | Do not splice raw prices. Different assays and loading conventions matter. Full CTD needs inland candidates, freight and missing assays. |
| JM | Separate Xiaoyi and Lishi assay-labelled benchmarks, plus Mongolian washed-coal and Australian port candidates | Xiaoyi/Lishi/Australian port from 2016-01-04; Mongolian series have material gaps | Prefer a candidate panel over the current single Ganqimaodu input. Do not compute its minimum until quality and transport are normalized. |
| SS | Wuxi 304/2B mill-edge outright price, with a conditional +170 edge conversion | Spot from 2019-07-01; futures listed 2019-09-25 | Best simple adjusted-benchmark candidate after SM. Confirm thickness, registered brands, tax and weight convention. No futures carry before listing. |
| SM | Tianjin FeMn65Si17, with the historical Tianjin location schedule | From 2016-01-04 | First implementation priority. Use +0 / +150 / +190 according to contract, conditional on quote-to-warehouse comparability. |
| SF | National 72% benchmark for a continuous 2016 comparison; separate Ningxia 72% ex-works benchmark from July 2019 | National from 2016-01-04; Ningxia from 2019-07-15 | No justified continuous delivery-equivalent series yet. Obtain historical Tianjin/consumer-region 72% quotes or actual freight and pickup details. |

The recommendations are research judgments based on economic comparability and observed coverage. None of the five has a fully verified CTD panel in the inspected files. The output should retain both a stable benchmark and any later CTD estimate, with explicit labels.

## Workbook evidence

Read-only extraction covered nine sheets in four workbooks. The current and full-history daily workbooks overlap without conflicting numeric values for the inspected tickers. Combined results contain **32 distinct series and 72,641 distinct code/date observations** from 2016 through the as-of cutoff. These include context-only basis, scrap and monthly tender series; they are not 32 deliverable spot candidates. The exported research panel contains 13 raw outright-price series, without forward-filling, normalization or minimum selection.

The full-history workbook generally stops in June 2026; the current daily workbook extends the same series into September. Combining exact matching overlaps is defensible for this snapshot, but is not evidence that the history was available unrevised on each past date. Publication timestamps and historical metadata revisions are not available.

| Input | Code | Workbook / sheet / column | First observed since 2016 | Key description or limitation |
|---|---|---|---|---|
| coke_sub_a_rz | S004425298 | ifind_data.xlsx / const_d / B | 2018-01-02 | 日照港平仓含税; A13/S0.7/CSR60/MT7. Missing M40/M10/CRI/volatile/size/Mf. |
| coke_sub_a_tj | S004369291 | daily workbooks / ferrous_d / R | 2016-01-04 | 天津港平仓; A<12.5/S<0.7/M25>90/M10<7.5/Mt<7/CSR>62; M25 is not M40. |
| ckc_a10v24s08_lvliang | S002877257 | daily workbooks / ferrous_d / CQ | 2016-01-04 | 孝义车板含税; A10/V24/S0.8/G75/Y24/CSR63. Moisture and petrography absent. |
| ckc_a9v18s10_lvliang | S002877258 | daily workbooks / ferrous_d / CR | 2016-01-04 | 离石车板含税; A9/V18/S1/G80/Y18/CSR68. Moisture and petrography absent. |
| ckc_a10v24s10_ts | S002877299 | daily workbooks / ferrous_d / CG | 2016-01-04 | 唐山出厂含税; CSR46. Exclude from deliverable CTD under the standards reviewed. |
| ckc_stock_ganqimaodu | S004085268 | daily workbooks / ferrous_d / BS | 2016-01-04 | 蒙古焦煤精煤库提; washed coal is explicit, tax and No.5 identity are not. |
| ckc_outstock_ganqimaodu | S009785426 | daily workbooks / ferrous_d / AX | 2021-09-30 | 主焦(蒙古)库提含税; No.5 identity and full assay are not explicit. |
| Australian Jingtang port quote | S002858879 | daily workbooks / ferrous_d / AY | 2016-01-04 | 库提含税澳大利亚主焦煤; no exact brand/assay. |
| ss_304_gross_wuxi | S004785205 | daily workbooks / base_d / BL | 2019-07-01 | 304/2B卷、毛边; “gross” alias means mill edge here, not proof of gross-weight pricing. |
| sm_65s17_tj | S002959498 | daily workbooks / ferrous_d / AB | 2016-01-04 | 天津市场价 FeMn65Si17; exact pickup/tax/payment terms need confirmation. |
| sf_72_ningxia | S004789784 | daily workbooks / ferrous_d / CH | 2019-07-15 | 宁夏出厂含税72; not a named Zhongwei delivery warehouse quote. |
| sf_72_shmet | S005068030 | ifind_data.xlsx / base_d2 / Q | 2016-01-04 | 全国平均72; usable benchmark, but no unique delivery location. |

Exact headers, units, cells, per-source observation dates/counts and annual counts are in `priority_spot_audit/inventory.csv` and `yearly_coverage.csv`. `merged_profiles.csv` describes combined coverage. `observations.csv` preserves the originating cell for each numeric observation.

Important data findings:

- The older Ganqimaodu series has consecutive available observations on **2016-03-25 and 2017-08-02**, a 495-calendar-day gap. The newer series has a **2022-05-31 to 2023-01-03** gap of 217 days. These are gaps between observations, not counts of missed trading sessions. Unlimited forward-fill would create stale carry signals.
- `ss_304_wuxi_phybasis` ends on **2025-12-15** in both daily snapshots. It is a basis, not an outright price; negative/zero values are valid in principle. Do not place it in a spot-price minimum or forward-fill it into 2026.
- `ss_304_scrap_wuxi`, national SF75 and monthly Hegang procurement quotes are comparison inputs, not interchangeable deliverable prices. Month-end procurement period labels also do not establish publication dates.
- The supplied maps contain more aliases than the inspected source sheets. A mapped-but-unseen price must not be counted as acquired history. In particular, the earlier SM Guangxi and Australian FOB/CFR acquisition assumptions need actual observations located before use.
- Same-day publication time is unknown. The existing signal execution lag must be retained and checked rather than assuming every quote was known at the futures close.

## J: historical normalization

The 2011 delivery standard allows ash/sulfur discounts and a single combined strength/reactivity discount; moisture above 5% is weight-deducted. This differs materially from J2201 onward, which uses dry-basis pricing and full-moisture deduction. Sources: [2011 exchange text reproduced by Minmetals](https://www.minfutures.com/main/a/20170302/12902.shtml), [2021 exchange standard in Guolian's delivery manual, pp. 4–5](https://www.glqh.com/u/cms/www/202507/30110346mtf5.pdf), and [DCE factsheet confirming J2201/J2604 boundaries](https://www.dce.com.cn/dce/file/2026-01-15/17684624156122c9a882b9ae6dcbb289019bc092f6fc1681.pdf).

For the stated Rizhao A13/S0.7/CSR60/MT7 specification, an **illustrative** old-standard conversion is `P / 0.98 + 80`: ash discount 15, sulfur discount 15, CSR discount 50 and 2% excess-moisture deduction. This assumes all missing eligibility tests pass and no other quote-basis conversion is required. From J2201, the corresponding moisture-only conversion is `P / 0.93`; actual quality deductions still depend on missing tests. These equations demonstrate why the regimes cannot be joined with a fixed adjustment; they are not certified deliverable prices.

J2604 introduces a 110 yuan/t deduction when equilibrium moisture exceeds 1%, raising the equivalent futures-basis cost by 110. Full moisture and equilibrium moisture are separate quantities. Missing Mf must not default to zero; the delivery rules also specify treatment of warehouse reports without an Mf result. Sources: [J003-2024 standard, p. 3](https://www.glqh.com/u/cms/www/202503/281125304v18.pdf), [exchange business rules reproduced by Changjiang](https://www.cjfco.com.cn/ueditor/jsp/upload/file/20250328/1743143292092043647.pdf).

**Decision:** use Rizhao as the main adjusted benchmark from 2018, and keep Tianjin separate for 2016–2017 and overlap comparison. The Tianjin series has different inequality-based assays and cannot be assigned the exact Rizhao quality adjustment. Both are loaded-on-board quotes, so investigate whether loading/port charges should be removed to compare with warehouse delivery. Recover historical inland warehouse/location schedules before using inland-to-port candidates. The earlier survey's J2012 Shanxi change remains a source lead rather than a fully reconstructed warehouse history.

## JM: separate historical quality and location schedules

Use four quality periods: pre-JM1907 (2013 standard), JM1907–JM2303 (2018 standard), JM2304–JM2612 (2022 standard), JM2701 onward (2025 standard). The initial standard requires CSR>50; the 2018 replacement adds stricter strength/petrographic requirements, while the 2022 regime changes benchmark quality and geography. Sources: [2013 exchange standard and warehouse list](https://www.doto-futures.com/jysgg/4095.html), [DCE 2018 notice reproduced by Dadi](https://www.ddqh.com/news/info/757.html), [Huatai's 2022 comparison](https://www.htfc.com/wz_upload/png_upload/20220415/1650034398236fef0d6.pdf), [DCE current contract-period factsheet](https://www.dce.com.cn/dce/file/2026-01-15/17684624156122c9a882b9ae6dcbb289019bc092f6fc1681.pdf).

The 2022 moisture treatment is proportional: for moisture `m > 8%`, quoted wet-tonne cost is multiplied by `0.92 / (1-m)`. At 10% moisture this is 1.022222, whereas the prototype uses `1/(1-(m-0.08)) = 1.020408`. Earlier rules round the excess-moisture deduction. The difference is small per tonne but systematic. The prototype also accepts G>65 generally, although registration eligibility must distinguish the stricter inbound requirement from the outbound tolerance.

**Decision:** Xiaoyi and Lishi are the best-described continuous local candidate quotes, not proven CTD winners. Their quality premiums differ by contract regime; add actual transport/handling to a valid warehouse and subtract that warehouse's premium. Ganqimaodu can be a competing candidate only after exact coal identity, moisture, assays, tax convention and transport are established. Exclude the CSR46 Tangshan quote from CTD unless the provider corrects its specification. Australian Jingtang is useful as a market comparator but needs a brand/assay before eligibility can be inferred.

A geography history is essential: Shanxi was not always zero-premium. Huatai's 2022 review reports the transition from -200 to 0 for Shanxi and 0 to +170 for the named ports from JM2304, as well as an earlier -300/-200 change whose boundary remains to be recovered. JM2701 changes **Tangshan/Tianjin specifically** from +170 to +140; do not apply that change to every port. [DCE announcement 2025-106](https://www.cctd.com.cn/show-111-253136-1.html). Brand premiums, group-delivery self-quoted premiums and warehouse openings require separate dated overlays. The old prototype's entire pre-2304 location handling is therefore unsuitable for historical CTD.

## SS: a useful adjusted benchmark, not a brand minimum

The local series explicitly identifies 304/2B mill-edge coil. SHFE assigns mill edge a -170 yuan/t delivery differential, so a comparable eligible coil becomes `P + 170`, less any applicable thickness premium. Thickness, width, brand and weight/tax convention must be confirmed before this is more than a conditional adjusted benchmark. Do not infer 2.0mm or a registered brand from the alias. [SHFE premium schedule](https://www.shfe.com.cn/products/futures/metal/ferrousandpreciousmetal/ss_f/attach/201909/t20190918_795033.html).

The 0.7mm premium changes from +400 to +300 on **2026-07-20**, under a notice dated 2025-12-05. This is a date-based rule, not a July contract-month boundary. A delivery-cost model needs the applicable settlement/receipt event date; a current cash equivalent should state which date it uses. It has no direct effect on a confirmed 2.0mm quote. [SHFE announcement 2025-151 reproduced by the broker](https://www.btqh.com/index.php?a=show&c=index&catid=26&id=17535&m=content&pid=10).

**Decision:** retain the outright Wuxi series from SS listing onward, test the +170 edge normalization once metadata is confirmed, and obtain registered-brand Wuxi/Foshan quotes only if a brand CTD adds value. SS futures started **2019-09-25**, so a 2016 carry history cannot exist. [SHFE account of listing](https://edu.shfe.com.cn/LearningCorner/ServiceCases/59.html).

## SM and SF: historical location changes matter

The 2018 exchange notice gives different activation contracts: **SF1907** for the updated SF warehouse schedule and **SM1911** for SM. For the Tianjin SM warehouses it changes 0 to -150; the 2023 notice changes -150 to -190 from **SM2411**. Accordingly, a comparable Tianjin cash quote needs additions of **0 / 150 / 190**, not a constant 190 throughout history. Sources: [CZCE 2018 notice 39, printed p. 37](https://www.czce.com.cn/cn/rootfiles/2018/11/07/1538463921362256-1538463921378754.pdf), [CZCE 2023 notice 114, printed pp. 47–48](https://www.czce.com.cn/cn/rootfiles/2023/11/13/1699404501503421-1699404501519214.pdf). Earlier activation history is supported by the old/new schedule but still needs checking against the exact warehouse used by the quote.

The new producer regions activate from **SF2309/SM2309**, with warehouse business starting no earlier than 2023-08-20 and actual opening dates required: SF Zhongwei -350; SM Shizuishan -360, Ulanqab -400 and Qinzhou -350. These are not universal premiums for all Ningxia/Inner Mongolia/Guangxi quotes. [CZCE regional expansion and activation table, printed p. 44](https://www.czce.com.cn/cn/rootfiles/2022/10/14/1655825346982784-1655825347001972.pdf).

Announced future changes: Zhongwei SF becomes -280 from SF2707; SM Qinzhou/Ulanqab/Shizuishan become -250/-290/-340 from SM2711. [CZCE notice 2026-96 reproduced with the full table](https://www.btqh.com/index.php?a=show&c=index&catid=26&id=18075&m=content).

**SM decision:** Tianjin is the simplest first adjusted benchmark. Compare Inner Mongolia, Gansu and the national average as separate robustness series. Confirm cash versus acceptance-payment terms, taxes, exact pickup and any handling before treating the quote as warehouse-equivalent. A provincial minimum is not CTD.

**SF decision:** the national 72 series provides 2016 coverage but cannot receive a location premium. Ningxia is explicitly ex-works and starts in July 2019. Obtain historical Tianjin 72 delivery-grade prices; alternatively, document time-varying transport from a specific producer to an approved warehouse. Keep SF75 separate because it is a different grade rather than an automatically cheaper equivalent. The existing +350 may be a useful approximate freight/basis convention in some periods, but has not been justified as a universal exchange adjustment back to 2016.

## Implementation consequences and outstanding evidence

1. Follow the small-basket recommendation above. The broader candidate and exact-eligibility discussion is supporting research, not a requirement to acquire every missing field before evaluating a proxy. Preserve current production outputs as a comparison baseline. Build new named research outputs rather than silently redefining existing aliases.
2. Prioritize `SM_tianjin_adjusted_spot(t, contract)` and a conditional `ss_wuxi_adjusted_spot`; do not add SM's existing +190 again when passing a normalized spot to carry.
3. For J/JM, retain material legacy quality changes, moisture convention, delivery location and source. Representative missing assays can be assumed for a proxy and varied in sensitivity checks; only a deliverable-CTD claim requires complete eligibility evidence.
4. Separate announcement date, contract activation, warehouse opening, and receipt/settlement dates. Future rules can affect a listed future contract before the calendar reaches its delivery year, but must not leak into earlier contracts.
5. Add quote-age limits using an exchange calendar, keep missing history missing, and reconcile same-day observation availability. A 10–13 calendar-day holiday gap is not automatically stale; a 217-day gap cannot be carried through without justification.
6. Validate regional overlap, rule-boundary jumps and near-delivery futures basis before choosing a replacement signal. No futures join, convergence test or performance backtest was performed in this phase.

The previous research module has additional confirmed problems: no J pre-2201 implementation; silent use of 2018 JM rules before JM1907; incorrect 2022+ JM moisture conversion; unspecified pre-2304 JM geography; missing JM2701 Tangshan/Tianjin change; date-based SS change represented as expiry month; ferroalloy histories starting too late; and missing J Mf treated as zero. Its current tests cannot certify rule correctness because they largely assert its encoded assumptions.

## Reproduction and source limits

Run `audit_priority_spots.py --source-dir C:/Users/harve/Nutstore/1/Nutstore --as-of 2026-09-22`, then `summarize_priority_spots.py`, using the bundled Python runtime. Source workbooks are read-only and use cached numeric values; no iFinD refresh was attempted. The companion notebook reproduces the summaries and key checks. Source snapshot updates can change later runs.

Some exchange PDFs were available through indexed extracted text while direct downloads or page screenshots timed out. Those entries are identified in `priority_rule_evidence.json`; they are usable evidence for this research review, but not claimed to be a visually verified complete rule archive. Exact historic warehouses, freight, quote publication time and metadata vintages remain open. Proxy production use requires documented approximations, historical-rule tests and carry sensitivity checks; it does not require an exhaustive delivery model.
