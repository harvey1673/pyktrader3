# Price, volatility, aggregate OI, and volume research

## Scope and execution convention

- Inputs: `C:\dev\data\fut_d_20260930.parquet` and
  `C:\dev\data\fut_oi_volume_20260930.parquet`.
- Research period: 2010-01-01 through 2026-09-30 across 57 products.
- Signal families: time-series and cross-sectional demean only. `xs_score` is
  excluded because the paired study did not show a stable advantage over
  demeaning and it increased turnover.
- Library size: 102 recipes and 204 family-specific strategies.
- Execution: date-T close/OI/volume determines the target. The position is held
  from T+1 and the T+1 return receives the n305, then n310, then a1505 execution
  adjustment where available, matching the daily production convention.
- Costs: 2 bps per unit turnover.
- Selection: direction is chosen on 2010-2018; validation is 2019-2023;
  screening and portfolio weights use no data after 2023-12-31. The reported
  OOS period is 2024-01-01 through 2026-09-30.

The implementation is in `tests/price_oi_volume_broad_research.py`. Detailed
outputs are in `C:\dev\data\output\price_oi_volume_broad_research`.

## Correct cross-sectional construction

For a base signal `s`, product-specific position scaler `f`, and standalone
20-day volatility `v`, the refreshed cross-sectional construction is:

1. demean the base signal across products;
2. multiply by the product-specific scaler when present;
3. demean the completed scaled signal again; and
4. construct holdings from `signal / v`.

The second demeaning is essential because a product-specific scaler generally
breaks the neutrality created by the first demeaning. The resulting condition
is `sum_i(holding_i * vol20_i) = 0`: signed standalone-volatility risk is
neutral, although nominal holdings need not sum to zero. This is consistent
with the existing portfolio convention where `signal_df` is the signal to
trade and holdings are subsequently divided by `vol20`.

The two older research scripts already follow this convention. In
`oi_volume_trend_research.py`, each complete cross-sectional candidate is
demeaned before `evaluate_signal` divides it by `vol20`. The exploratory
`oi_volume_sharpe_search.py` reuses that evaluator. They therefore did not need
a mechanical rerun solely for this correction. The broad Oct 4-5 study did,
because its earlier implementation scaled and demeaned the already risk-scaled
weights, which targeted nominal rather than risk neutrality.

## Volatility factor

The refreshed broad study uses:

`vr_240 = vol20 / vol240`

`VF(k) = min(exp(-k * (vr_240 - 1)), 1)`

with `k = 5`. A separate sensitivity study tested `k = 0.5, 1, 2, 5, 10, 15`
while fixing signal direction from the unscaled 2010-2018 price anchors. The
equal-weight validation Sharpe was highest at `k=5` (0.818), versus 0.788 for
`k=1`, 0.807 for `k=10`, and 0.796 for `k=15`. OOS Sharpe was 1.260 for `k=5`,
1.197 for `k=1`, 1.269 for `k=10`, and 1.282 for `k=15`. Because OOS is not a
selection sample, `k=5` is the supported choice; the higher OOS values at
`k=10` and `k=15` are diagnostics, not a reason to retune.

Relative to the unscaled anchors, `k=5` increased mean time-series Sharpe by
0.102 in training, 0.039 in validation, and 0.082 OOS. For cross-sectional
demean, the corresponding changes were +0.036, -0.012, and +0.149. Therefore
the volatility factor is a broadly supported time-series overlay, while XS
use should remain recipe-specific and must pass pre-2024 validation rather
than being applied to every XS signal.

## Refreshed key conclusions

1. Slow cross-sectional price trends remain the strongest family. The OOS
   mean Sharpe across the full library is about 0.91 for XS demean versus 0.41
   for time-series, but individual variants remain highly correlated and must
   not be interpreted as independent evidence.

2. Efficiency-ratio scaling is most useful for slow XS trends. The leading
   OOS sleeves combine 240-day momentum, HLR, or regression t-stat signals with
   a 240-day efficiency factor; several also retain the `k=5` volatility
   factor after passing both training and validation screens.

3. OI and volume are more useful as confirmation or regime variables than as
   unrestricted directional forecasts. OI breakout confirmation helps some
   slow XS momentum signals. Standalone OI/volume states are unstable; notably,
   the selected `oi_momentum__time_series` sleeve has negative OOS Sharpe and
   is one reason not to deploy the raw optimizer unchanged.

4. High OI alone is not robust reversal evidence. Reversal behavior is more
   plausible when high OI or high volume coincides with price extremes,
   volatility stress, low efficiency, or concentration. These sleeves are
   diversifiers, not the main return engine, and many nominally labelled
   reversal recipes are inverted by the training sample and are economically
   continuation signals.

5. Removing `xs_score` is still supported. The refreshed library uses only
   time-series and XS demean; no conclusion depends on an XS-score variant.

## Portfolio results

Thirty sleeves pass the pre-2024 Sharpe and correlation screen. The table
below shows the refreshed portfolios in research risk-budget units, not
account-level capital returns.

| Portfolio | Validation Sharpe | OOS Sharpe | OOS annual return | OOS annual vol | OOS max drawdown | OOS mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equal weight | 1.029 | 1.342 | 2.30% | 1.71% | -1.36% | 0.0929 |
| Raw optimized | 1.337 | 1.824 | 2.05% | 1.13% | -0.77% | 0.0871 |
| Recommended 50% shrink | 1.172 | 1.571 | 2.18% | 1.39% | -0.90% | 0.0900 |

Compared with the superseded nominal-neutral, `k=1` run, the refreshed
equal-weight OOS Sharpe rises from 1.118 to 1.342 and the raw-optimized OOS
Sharpe rises from 1.642 to 1.824. OOS turnover and drawdown also improve. The
old artifacts are retained in the output directory with the
`superseded_nominal_k1_` prefix for auditability.

The raw optimizer maximizes pre-2024 net-PNL Sharpe subject to non-negative
weights, a 10% per-sleeve cap, a 30-sleeve limit, and an 0.85 pairwise
correlation screen. It has 15 nonzero sleeves and assigns about 80% to XS
demean. This is an in-sample constrained optimizer, not a statement that its
weights are known precisely.

The recommended portfolio is therefore the 50/50 shrinkage blend between the
equal-weight and raw-optimized vectors. It retains all 30 selected sleeves,
cuts each optimizer concentration toward 1/30, and gives up some headline
Sharpe for lower estimation risk. Its full-sample Sharpe is 1.557 and OOS
Sharpe is 1.571.

The detailed recommended weights and each sleeve's daily and annualized unit
standard deviation are in
`portfolio_components_with_unit_std.csv`. The largest recommended weights are
6.67% each for seven capped optimized sleeves: 240-day momentum with the
480-day volatility factor (XS); 240-day HLR with joint volatility/efficiency
scaling (XS); 120-day momentum with joint scaling (XS); 60-day regression
t-stat with joint scaling (XS); 20-day HLR both unscaled and with joint scaling
(XS); and 240-day regression t-stat with efficiency scaling (XS).

That file now also contains `unit_gross_pnl_daily_std_cny`. It is the
pre-2024 daily standard deviation of gross PnL in CNY for one research sleeve,
calculated from the research holdings, point price changes and each product's
contract multiplier. The complete strategy-level CNY series is saved in
`daily_gross_pnl_cny.parquet`. This is a risk diagnostic before trading costs;
it is not account risk until the research sleeve is mapped to a production
capital or contract scale.

## Recommended implementation path

- Use `k=5` as the default volatility factor for this research refresh.
- Apply a product-specific scaler to the signal, then demean the completed XS
  signal again, then let the existing engine construct holdings as
  `signal_df / vol20`.
- Start from the 50%-shrunk portfolio rather than the raw optimizer. Review the
  weak OOS standalone-OI sleeve and group risk by method and horizon before
  assigning production capital.
- Preserve OI breakout as a trend-confirmation candidate and keep reversal
  recipes in a small separate sleeve.
- Refit directions and portfolio weights only on a declared schedule; do not
  tune them from the 2024-2026 diagnostic period.

## Condensing composites into `signal_repo`

Do not put formulas into a feature string such as `px*vol20` or invent special
characters with parsing precedence. Keep existing 10-item lists working and
add a structured composite form for new signals. A minimal shape is:

```python
{
    "base": {"feature": "px", "func": "hlratio", "params": {"window": 20}},
    "scalers": [
        {"feature": "px", "func": "efficiency_ratio", "window": 20},
        {"feature": "px", "func": "vol_ratio_factor",
         "fast": 20, "slow": 240, "k": 5},
    ],
    "combine": "multiply",
    "cross_section": "demean",
    "risk_vol": {"feature": "px", "window": 20},
}
```

The signal builder should calculate the base, multiply the scalers, apply the
final XS transform, and return `signal_df`. The existing portfolio code remains
responsible for dividing by `vol20`. A small adapter can translate legacy list
specifications into the same internal representation, avoiding a broad rewrite
and keeping the current configurations backward compatible. This is a design
recommendation only; production `signal_repo.py` and `feature_config.py` were
not changed in this research refresh.

## Reproducibility and caveats

Run:

```powershell
D:\miniconda3\python.exe tests\price_oi_volume_broad_research.py
D:\miniconda3\python.exe tests\vol_ratio_k_sensitivity.py
```

The 2024+ segment is excluded mechanically from directions, screening, and
weights, but it is not a pristine external holdout because the research ideas
were formed after earlier results had been observed. The full-sample Sharpe
objective also creates research-selection bias. Treat this as a candidate
library and portfolio proposal, not a production performance forecast.
