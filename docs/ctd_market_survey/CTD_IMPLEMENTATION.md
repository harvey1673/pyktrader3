# Sparse CTD research implementation

The implementation remains isolated in `tests/ctd_adjustments.py`. It does not modify production mappings, factor generation, analytics, or utility code.

```python
from tests.ctd_adjustments import (
    j_ctd_basis, jm_ctd_basis, ss_ctd_basis, SM_ctd_basis, SF_ctd_basis,
)

series, audit = j_ctd_basis(spot_df, expiry, return_details=True)
```

Each function returns a futures-equivalent spot Series. With `return_details=True`, it also returns raw price, cash cost, quality, location and brand adjustments, weight multiplier, eligibility, and equivalent price for every candidate.

```text
equivalent = raw_price * weight_multiplier
             + cash_cost
             - quality_adjustment
             - location_adjustment
             - brand_adjustment
```

Positive exchange premiums lower the futures-equivalent price. Exchange discounts are negative and raise it. Freight and other cash costs belong in `CTDCandidate.cash_cost`.

## Default sparse basket

| Product | Research behavior |
|---|---|
| J | Rizhao quasi-grade-1 is primary; Tianjin only fills missing early history. From J2201 the new quality schedule applies; J2604 treats missing equilibrium moisture conservatively. Before J2201, the quote uses an explicitly labelled proxy for the major legacy moisture effect. |
| JM | Xiaoyi/Lvliang A10 V24 S0.8 is the default. Ganqimaodu is excluded from the automatic minimum until Mongolian No.5 identity and freight are supplied. JM2304 uses proportional dry-matter conversion above 8% moisture. |
| SS | Wuxi 304/2B 2.0mm mill-edge proxy with the -170 adjustment. The listing boundary is enforced and dated rule changes use the observation date. Registered status, thickness and width remain research assumptions. |
| SM | Tianjin 6517: zero before SM1911, -150 from SM1911, and -190 from SM2411. |
| SF | National grade-72 is treated as a Tianjin-basis proxy. It receives no automatic Zhongwei/Ningxia adjustment. |

Explicit `CTDCandidate` lists are minimized after normalization, allowing alternative prices and freight scenarios to be tested without changing production code.

## Validation

`tests/test_sparse_ctd_research.py` covers the sparse defaults and major regime boundaries. `tests/test_ctd_adjustments.py` retains the wider candidate-engine tests.

The audited sparse raw panel produces research histories beginning in 2016 for J, JM, SM and SF, and from the SS listing date for SS. This is an execution and coverage check, not signal-performance validation.
