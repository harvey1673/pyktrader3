# Signal Digger Enhancement Plan: Alphalens Integration

## Executive Summary
Incorporate Alphalens' full-featured factor analysis capabilities into `signal_digger.py` so the existing `analyze_feature_ic_ir_ts()` and `analyze_feature_ic_ir_xs()` functions become one-stop report generators for factor evaluation, including statistics, plots, and test outputs.

---

## 1. GAP ANALYSIS: Signal Digger vs Alphalens

### Current Signal Digger Capabilities ✅
| Capability | Status | Level |
|-----------|--------|-------|
| IC/IR Time-Series | ✅ | Basic (per-asset IC) |
| IC/IR Cross-Sectional | ✅ | Basic (per-date IC) |
| Rolling Beta Computation | ✅ | Advanced |
| Multi-Horizon Analysis | ✅ | Good |
| Asset/Sector Grouping | ✅ | Good |
| Forward Returns | ✅ | Good |
| Correlation Methods | ✅ | Good (Spearman, Pearson) |

### Missing Alphalens Features ❌
| Feature Category | Missing | Impact |
|-----------------|---------|--------|
| **IC Statistics** | IC Skewness, Kurtosis, Risk-Adjusted IC | Medium (statistical rigor) |
| **Returns Analysis** | Quantile-based returns, long-short spreads, mean returns by bucket | High (profitability assessment) |
| **Turnover Analysis** | Turnover tracking, transaction cost modeling | High (trading feasibility) |
| **Factor Stability** | Rank autocorrelation, factor persistence | Medium (robustness) |
| **Significance Testing** | t-statistics, p-values, multiple-period adjustment | Medium (rigor) |
| **Group-Neutral Returns** | Beta/sector-neutral portfolio construction | Medium (exposure control) |
| **Visualization** | Heatmaps, distribution plots, cumulative returns | Low (communication) |
| **Quantile Analysis** | Decile/quintile decomposition | Medium (ranking verification) |

---

## 2. IMPLEMENTATION ROADMAP (Phased)

### Phase 1: IC Statistics Enhancement (Core Metrics)
**File:** `signal_digger.py` → New functions
**Effort:** 2-3 hours | **Priority:** HIGH

```
compute_ic_statistics()
├── IC Mean (exists via mean IC)
├── IC Std (exists via std IC)
├── IC Skewness (NEW)
├── IC Kurtosis (NEW)
├── Risk-Adjusted IC = Mean IC / Std IC (exists but verify)
├── t-statistic (NEW)
└── p-value from t-stat (NEW)
```

**New Functions:**
- `compute_ic_stats_ts()` - Time-series IC statistics (per asset/period)
- `compute_ic_stats_xs()` - Cross-sectional IC statistics (per date)
- `compute_significance_stats()` - t-stat, p-value, CI

---

### Phase 2: Returns Analysis (Profitability Metrics)
**File:** `signal_digger.py` → New functions
**Effort:** 3-4 hours | **Priority:** HIGH

```
analyze_returns_by_quantile()
├── Quantile Decomposition (decile/quintile/custom)
├── Mean Return by Quantile (NEW)
├── Long-Short Spread (NEW)
├── Long-Short IR (NEW)
├── Cumulative Returns (NEW)
└── Sharpe Ratio by Quantile (NEW)
```

**New Functions:**
- `compute_quantile_returns()` - Group by factor quantiles, compute returns
- `compute_long_short_returns()` - Long top quantile / short bottom
- `compute_factor_returns()` - Time-series of factor portfolio returns
- `analyze_returns_ic_ir_xs()` - Enhanced XS analysis with returns breakdown

---

### Phase 3: Turnover & Transaction Analysis
**File:** `signal_digger.py` → New functions
**Effort:** 2-3 hours | **Priority:** MEDIUM

```
analyze_factor_turnover()
├── Factor Value Turnover (NEW)
├── Quantile Membership Turnover (NEW)
├── Portfolio Turnover (NEW)
├── Estimated Transaction Costs (NEW)
└── Turnover Impact on Returns (NEW)
```

**New Functions:**
- `compute_factor_turnover()` - Period-over-period factor value changes
- `compute_quantile_turnover()` - Membership changes between deciles
- `estimate_transaction_costs()` - Based on turnover + bid-ask spread
- `adjust_returns_for_costs()` - Long-short returns net of costs

---

### Phase 4: Factor Stability & Persistence
**File:** `signal_digger.py` → New functions
**Effort:** 2-3 hours | **Priority:** MEDIUM

```
analyze_factor_stability()
├── Rank Autocorrelation (NEW)
├── Factor Persistence (NEW)
├── IC Time-Series Clustering (NEW)
└── Drawdown Analysis (NEW)
```

**New Functions:**
- `compute_rank_autocorrelation()` - Spearman rank correlation of consecutive periods
- `compute_factor_persistence()` - How long factor effect lasts
- `compute_ic_drawdown()` - Max consecutive negative IC periods
- `compute_factor_decay()` - Forward returns decay over time

---

### Phase 5: Group-Neutral Analysis
**File:** `signal_digger.py` → Enhanced functions
**Effort:** 2-3 hours | **Priority:** MEDIUM

```
analyze_feature_ic_ir_group_neutral() [ENHANCED]
├── Sector-Neutral IC (NEW)
├── Sector-Neutral Returns (NEW)
├── Beta-Neutral Returns (NEW)
└── Exposure Analysis (NEW)
```

**New Functions:**
- `compute_group_neutral_returns()` - Long/short within groups
- `compute_beta_neutral_returns()` - Remove systematic beta exposure
- `compute_exposure_analysis()` - Asset/sector exposure breakdown

---

### Phase 6: Visualization & Reporting
**File:** `signal_digger.py` → New module (optional)
**Effort:** 3-4 hours | **Priority:** LOW

```
Create visualization layer (optional matplotlib/seaborn)
├── IC Heatmaps
├── Return Distribution Plots
├── Cumulative Returns
├── IC Time Series
├── Quantile Return Spread
└── Turnover Analysis Charts
```

**New Functions:**
- `plot_ic_heatmap()` - IC by period/asset
- `plot_returns_by_quantile()` - Box plots by decile
- `plot_cumulative_returns()` - Long-short cumulative PnL
- `plot_ic_timeseries()` - Rolling IC line chart
- `create_factor_tear_sheet()` - Combined PDF report (optional)

---

## 3. IMPLEMENTATION SEQUENCE (by dependency)

```
1. compute_ic_stats_ts/xs()              [Phase 1] - Core IC metrics
2. compute_quantile_returns()            [Phase 2] - Group returns
3. compute_long_short_returns()          [Phase 2] - LS spread
4. compute_factor_turnover()             [Phase 3] - Turnover basics
5. compute_rank_autocorrelation()        [Phase 4] - Stability
6. compute_group_neutral_returns()       [Phase 5] - Neutralization
7. Visualization functions               [Phase 6] - Optional
8. Integrate into analyze_feature_ic_ir_ts/xs() - Convert into report-producing wrappers

### Target User Experience
The goal is that users should be able to call either `analyze_feature_ic_ir_ts()` or `analyze_feature_ic_ir_xs()` and receive a single structured result that contains:
- summary statistics
- per-horizon IC/IR tables
- quantile return breakdowns
- turnover and persistence diagnostics
- plots or plot-ready artifacts
- test/statistical significance outputs

That keeps the current public entry points stable while adding the missing Alphalens-style reporting behind them.
```

---

## 4. SPECIFIC IMPLEMENTATION DETAILS

### Phase 1 Details: IC Statistics

```python
def compute_ic_stats_ts(panel: pd.DataFrame, group_col: str = "group"):
    """
    Compute IC statistics for time-series analysis (per asset/period)
    
    Returns dict with:
    - ic_mean: float
    - ic_std: float
    - ic_skew: float (NEW)
    - ic_kurt: float (NEW)
    - ic_ir: float (risk-adjusted IC)
    - t_stat: float (NEW)
    - p_value: float (NEW)
    - ci_lower: float (NEW)
    - ci_upper: float (NEW)
    """
    from scipy.stats import skew, kurtosis, t as t_dist
    
    # Group by horizon, compute IC for each asset
    ic_values = []
    for (asset, h), group in panel.groupby(['asset', 'horizon']):
        ic, _ = compute_corr(group['score'], group['forward_return'])
        ic_values.append(ic)
    
    ic_arr = np.array(ic_values)
    ic_arr = ic_arr[~np.isnan(ic_arr)]
    n = len(ic_arr)
    
    return {
        'ic_mean': np.mean(ic_arr),
        'ic_std': np.std(ic_arr),
        'ic_skew': skew(ic_arr),           # NEW
        'ic_kurt': kurtosis(ic_arr),       # NEW
        'ic_ir': np.mean(ic_arr) / np.std(ic_arr) if np.std(ic_arr) > 0 else np.nan,
        't_stat': np.mean(ic_arr) / (np.std(ic_arr) / np.sqrt(n)) if np.std(ic_arr) > 0 else np.nan,  # NEW
        'p_value': 1 - t_dist.cdf(abs(t_stat), n-1) * 2,  # Two-tailed, NEW
        'ci_lower': np.mean(ic_arr) - 1.96 * np.std(ic_arr) / np.sqrt(n),  # NEW
        'ci_upper': np.mean(ic_arr) + 1.96 * np.std(ic_arr) / np.sqrt(n),  # NEW
    }
```

### Phase 2 Details: Returns by Quantile

```python
def compute_quantile_returns(panel: pd.DataFrame, n_quantiles: int = 10):
    """
    Group assets by factor score into quantiles, compute returns for each.
    
    Returns:
    pd.DataFrame with columns [quantile, horizon, mean_return, std_return, sharpe, n_assets]
    """
    results = []
    for (h, date), group in panel.groupby(['horizon', 'date']):
        # Create quantiles within each date/horizon
        group['quantile'] = pd.qcut(group['score'], q=n_quantiles, labels=False, duplicates='drop')
        
        for q in group['quantile'].unique():
            q_data = group[group['quantile'] == q]
            mean_ret = q_data['forward_return'].mean()
            std_ret = q_data['forward_return'].std()
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            
            results.append({
                'horizon': h,
                'date': date,
                'quantile': q,
                'mean_return': mean_ret,
                'std_return': std_ret,
                'sharpe': sharpe,
                'n_assets': len(q_data)
            })
    
    return pd.DataFrame(results)

def compute_long_short_returns(quantile_returns: pd.DataFrame):
    """
    Compute long (top quantile) minus short (bottom quantile) spread.
    
    Returns:
    pd.DataFrame with long-short returns, sharpe, win_rate
    """
    results = []
    for (h, date), group in quantile_returns.groupby(['horizon', 'date']):
        long_ret = group[group['quantile'] == group['quantile'].max()]['mean_return'].iloc[0]
        short_ret = group[group['quantile'] == group['quantile'].min()]['mean_return'].iloc[0]
        ls_ret = long_ret - short_ret
        
        results.append({
            'horizon': h,
            'date': date,
            'long_return': long_ret,
            'short_return': short_ret,
            'ls_return': ls_ret,
            'ls_sharpe': ls_ret / quantile_returns['std_return'].mean() if quantile_returns['std_return'].mean() > 0 else 0
        })
    
    return pd.DataFrame(results)
```

### Phase 3 Details: Turnover

```python
def compute_factor_turnover(panel: pd.DataFrame, n_quantiles: int = 10):
    """
    Compute factor ranking turnover period-over-period.
    
    Returns:
    pd.Series with turnover percentage by date
    """
    turnover = {}
    for h in panel['horizon'].unique():
        h_data = panel[panel['horizon'] == h].sort_values(['date', 'asset'])
        
        for date in h_data['date'].unique():
            if date not in h_data['date'].unique():
                continue
            
            # Current period rankings
            current = h_data[h_data['date'] == date][['asset', 'score']].set_index('asset')
            current['quantile'] = pd.qcut(current['score'], q=n_quantiles, labels=False, duplicates='drop')
            
            # Previous period rankings
            prev_date = h_data[h_data['date'] < date]['date'].max()
            if pd.isna(prev_date):
                continue
            
            previous = h_data[h_data['date'] == prev_date][['asset', 'score']].set_index('asset')
            previous['quantile'] = pd.qcut(previous['score'], q=n_quantiles, labels=False, duplicates='drop')
            
            # Turnover = pct of assets that changed quantile
            common_assets = current.index.intersection(previous.index)
            if len(common_assets) > 0:
                changed = (current.loc[common_assets, 'quantile'] != 
                          previous.loc[common_assets, 'quantile']).sum()
                turnover[(h, date)] = changed / len(common_assets)
    
    return pd.Series(turnover)

def estimate_transaction_costs(turnover_series: pd.Series, bid_ask_spread: float = 0.001):
    """
    Estimate transaction costs based on turnover.
    Assumes round-trip cost = turnover * (bid-ask spread + market impact)
    """
    # Typical costs: 0.1% bid-ask + 0.05% market impact
    round_trip_cost = bid_ask_spread + 0.0005
    costs = turnover_series * round_trip_cost
    return costs
```

### Phase 4 Details: Stability

```python
def compute_rank_autocorrelation(panel: pd.DataFrame, max_lag: int = 5):
    """
    Compute Spearman rank autocorrelation of factor scores across periods.
    High autocorr = factor is persistent = more reliable
    
    Returns:
    pd.DataFrame with autocorrelation by lag and horizon
    """
    results = []
    for h in panel['horizon'].unique():
        h_data = panel[panel['horizon'] == h].sort_values(['date', 'asset'])
        
        for lag in range(1, max_lag + 1):
            dates = sorted(h_data['date'].unique())
            valid_pairs = []
            
            for i in range(len(dates) - lag):
                date1, date2 = dates[i], dates[i + lag]
                
                d1 = h_data[h_data['date'] == date1][['asset', 'score']].set_index('asset')
                d2 = h_data[h_data['date'] == date2][['asset', 'score']].set_index('asset')
                
                common = d1.index.intersection(d2.index)
                if len(common) >= 10:
                    corr, _ = compute_corr(d1.loc[common, 'score'], 
                                          d2.loc[common, 'score'], 
                                          method='spearman')
                    valid_pairs.append(corr)
            
            if valid_pairs:
                results.append({
                    'horizon': h,
                    'lag': lag,
                    'autocorr': np.mean(valid_pairs),
                    'autocorr_std': np.std(valid_pairs)
                })
    
    return pd.DataFrame(results)
```

### Phase 5 Details: Group-Neutral Returns

```python
def compute_group_neutral_returns(panel: pd.DataFrame, group_col: str = "group"):
    """
    Compute long-short returns neutral to group membership.
    Within each group: long top scores, short bottom scores
    Average across groups to get sector-neutral exposure
    
    Returns:
    pd.DataFrame with group-neutral returns by date/horizon
    """
    results = []
    for (h, date), group in panel.groupby(['horizon', 'date']):
        
        if group_col not in group.columns:
            continue
        
        group_ls = []
        for sector, sector_data in group.groupby(group_col):
            # Within-sector long-short
            if len(sector_data) >= 4:
                top_assets = sector_data.nlargest(max(1, len(sector_data)//2), 'score')
                bot_assets = sector_data.nsmallest(max(1, len(sector_data)//2), 'score')
                
                ls_ret = top_assets['forward_return'].mean() - bot_assets['forward_return'].mean()
                group_ls.append(ls_ret)
        
        if group_ls:
            results.append({
                'horizon': h,
                'date': date,
                'group_neutral_ls': np.mean(group_ls),
                'group_neutral_std': np.std(group_ls)
            })
    
    return pd.DataFrame(results)
```

---

## 5. INTEGRATION POINTS

### Enhanced `analyze_feature_ic_ir_ts()`
```python
# Add new parameters
def analyze_feature_ic_ir_ts(
    panel: pd.DataFrame,
    method: str = "spearman",
    sector_analysis: bool = True,
    group_col: str = "group",
    include_returns_analysis: bool = True,    # NEW
    include_turnover: bool = True,             # NEW
    include_stability: bool = True,            # NEW
    include_ic_stats: bool = True,             # NEW
    n_quantiles: int = 10                      # NEW
):
    # Existing IC/IR analysis
    existing_results = {...}  # Current implementation
    
    # NEW: IC Statistics (Phase 1)
    if include_ic_stats:
        ic_stats = compute_ic_stats_ts(panel, group_col)
        existing_results['ic_stats'] = ic_stats
    
    # NEW: Returns Analysis (Phase 2)
    if include_returns_analysis:
        quantile_rets = compute_quantile_returns(panel, n_quantiles)
        ls_rets = compute_long_short_returns(quantile_rets)
        existing_results['quantile_returns'] = quantile_rets
        existing_results['long_short_returns'] = ls_rets
    
    # NEW: Turnover (Phase 3)
    if include_turnover:
        turnover = compute_factor_turnover(panel, n_quantiles)
        costs = estimate_transaction_costs(turnover)
        existing_results['turnover'] = turnover
        existing_results['est_costs'] = costs
    
    # NEW: Stability (Phase 4)
    if include_stability:
        autocorr = compute_rank_autocorrelation(panel)
        existing_results['rank_autocorr'] = autocorr
    
    return existing_results
```

---

## 6. TESTING STRATEGY

### Unit Tests (per phase)
1. Test each new function with synthetic data
2. Verify numerical correctness (compare with numpy/scipy direct computations)
3. Edge case handling (NaN, single-asset, single-period)

### Integration Tests
1. Run existing `opt_test_notebook.ipynb` - ensure backward compatibility
2. Compare Phase 1 IC stats with Alphalens output
3. Compare Phase 2 quantile returns with Alphalens
4. Performance benchmarking (target: <1 second for 1000-asset analysis)

### Validation Against Alphalens
```python
# In a test notebook:
import alphalens as al

# Run through signal_digger
results_sd = analyze_feature_ic_ir_ts(panel, include_all=True)

# Run through alphalens
factor_data = al.utils.get_clean_factor_and_forward_returns(...)
results_al = al.tears.create_information_tear_sheet(factor_data)

# Compare results_sd vs results_al
assert np.isclose(results_sd['ic_stats']['ic_mean'], results_al['IC Mean'])
```

---

## 7. PRIORITY & TIME ESTIMATES

| Phase | Priority | Effort | Business Value | Status |
|-------|----------|--------|-----------------|--------|
| 1 | ⭐⭐⭐ HIGH | 2-3h | Essential metrics | TODO |
| 2 | ⭐⭐⭐ HIGH | 3-4h | Profitability | TODO |
| 3 | ⭐⭐ MEDIUM | 2-3h | Feasibility | TODO |
| 4 | ⭐⭐ MEDIUM | 2-3h | Robustness | TODO |
| 5 | ⭐⭐ MEDIUM | 2-3h | Risk control | TODO |
| 6 | ⭐ LOW | 3-4h | Communication | OPTIONAL |

**Total Estimated Effort:** 14-20 hours (2-3 full days)

---

## 8. DELIVERABLES

After implementation:
1. ✅ `signal_digger.py` with ~10 new functions (500+ lines)
2. ✅ Backward-compatible with existing code
3. ✅ Unit + integration tests
4. ✅ Docstrings + usage examples
5. ✅ Performance benchmarks
6. ✅ Update notebook with new features (opt_test_notebook.ipynb)
7. ✅ Optional: tear sheet visualization module

---

## 9. NEXT STEPS

**Immediate Actions:**
1. Review this plan with team
2. Prioritize phases based on business needs
3. Create feature branches for each phase
4. Start with Phase 1 (IC Statistics) - lowest complexity, high value
5. Use `ALPHALENS_RESEARCH.md` as reference throughout

**Questions to Address:**
- Do we need Alphalens still after Phase 2? (Can deprecate external dep)
- Should visualization (Phase 6) be separate module or inline?
- Performance: acceptable threshold for large-scale analysis?
- Backward compatibility: keep existing function signatures?

