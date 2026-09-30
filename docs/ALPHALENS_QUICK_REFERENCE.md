# Alphalens Quick Reference Guide

## Fast Lookup: Alphalens Functions & Capabilities

### 🔧 Core Workflow (in 3 steps)

```python
import alphalens as al
import pandas as pd

# 1. PREPARE: Get your factor and pricing data into shape
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor_ts,           # Series: (date, asset) -> factor values
    pricing_data,        # DataFrame: assets x dates with prices
    quantiles=5,         # Split factor into 5 quantiles (quintiles)
    periods=(1, 5, 20)   # Analyze 1, 5, 20 day forward returns
)

# 2. ANALYZE IC (Information Coefficient)
al.tears.create_information_tear_sheet(factor_data)

# 3. FULL ANALYSIS
al.tears.create_full_tear_sheet(factor_data)
```

---

## 📊 Key Metrics Cheat Sheet

| Metric | What It Means | Good Value |
|--------|------------|-----------|
| **IC Mean** | Avg correlation to future returns | > 0.02 |
| **IC Std** | Consistency of correlation | Lower is boring |
| **IC/Std** | Quality per unit of volatility | > 0.1 |
| **t-stat** | Statistical significance | > 1.96 |
| **p-value** | Probability IC = 0 | < 0.05 |
| **Return Q10-Q1** | Profit potential (top vs bottom quantile) | Higher is better |
| **Turnover** | % of portfolio changing each period | < 5% is low |
| **Skew** | Distribution bias (positive = usually good) | > 0 |

---

## 🎯 Functions by Purpose

### Data Ingestion
```python
# Main function (use this 90% of the time)
al.utils.get_clean_factor_and_forward_returns(factor, prices, quantiles=5, periods=(...))

# Alternative if you pre-computed returns
al.utils.get_clean_factor(factor, forward_returns, quantiles=5)

# Manual forward return computation
al.utils.compute_forward_returns(prices, periods=(...))
```

### Analysis & Metrics
```python
# All pre-computed in factor_data, but extract with:
al.performance.mean_return_by_quantile(factor_data, by_group=False)
al.performance.quantile_turnover(factor_data)
al.performance.factor_rank_auto_correlation(factor_data)
al.performance.ic(factor_data)  # Information Coefficient
```

### Visualization (Tear Sheets)
```python
# Information Coefficient analysis
al.tears.create_information_tear_sheet(factor_data)

# Complete analysis (IC + Returns + Turnover + Groups)
al.tears.create_full_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False)

# Just returns
al.tears.create_returns_tear_sheet(factor_data)

# Just turnover
al.tears.create_turnover_tear_sheet(factor_data)
```

### Individual Plots
```python
al.plotting.plot_quantile_returns_violin(factor_data)
al.plotting.plot_mean_quantile_returns_bar_plot(factor_data)
al.plotting.plot_mean_quantile_returns_spread_time_series(factor_data)
al.plotting.plot_ic_ts(factor_data)
al.plotting.plot_ic_hist(factor_data)
al.plotting.plot_turnover_time_series(factor_data)
al.plotting.plot_cumulative_returns_by_quantile(factor_data)
al.plotting.plot_factor_rank_auto_correlation(factor_data)
al.plotting.plot_monthly_returns_heatmap(factor_data)
```

---

## 📋 Input Data Format

### Factor Data Structure (what you pass in)
```python
# Factor: MultiIndex Series
factor = pd.Series(
    [0.5, -0.3, 0.1, ...],  # Factor values
    index=pd.MultiIndex.from_product([
        pd.date_range('2020-01-01', periods=100),  # Dates
        ['AAPL', 'MSFT', 'GOOGL', ...]             # Assets
    ], names=['date', 'asset'])
)

# Pricing: DataFrame
prices = pd.DataFrame({
    'AAPL': [100, 101, 102, ...],
    'MSFT': [200, 201, 202, ...],
    'GOOGL': [1500, 1501, 1502, ...],
}, index=pd.date_range('2020-01-01', periods=100))
```

### Optional: Groupby (e.g., sectors)
```python
# Add sector column to factor_data after creation
factor_data['sector'] = pd.Series({
    ('2020-01-02', 'AAPL'): 'Technology',
    ('2020-01-02', 'JPM'): 'Financials',
    ...
})
```

---

## 🔍 Output: What Tear Sheets Show

### Information Tear Sheet Outputs
- **Statistics Table**: IC Mean, IC Std, Risk-Adjusted IC, t-stat, p-value, skew, kurtosis
- **IC Time Series Plot**: How IC evolves over time
- **IC Distribution**: Histogram of IC values
- **IC by Quantile**: How each factor quantile contributes to IC

### Full Tear Sheet Outputs (sections)
1. **Summary Statistics Table** (all metrics)
2. **Returns Analysis**
   - Mean returns by quantile
   - Long-short return spread
   - Cumulative returns plot
3. **Information Coefficient Analysis** (as above)
4. **Turnover Analysis**
   - Turnover by quantile
   - Turnover vs holding period
5. **Sector/Group Analysis** (if groupby provided)
   - All above metrics per group

---

## ⚙️ Common Parameters

### quantiles vs bins
```python
quantiles=5      # Splits factor into 5 equal-population groups (quintiles)
bins=5           # Splits into 5 equal-width ranges (if factor is 0-100)
```

### periods
```python
periods=(1,)      # Only 1-day forward returns
periods=(1, 5, 10, 20)  # 1, 5, 10, 20 day forward returns (more comprehensive)
```

### long_short
```python
long_short=True   # Shows top quantile - bottom quantile
long_short=False  # Shows absolute returns per quantile
```

### group_neutral
```python
group_neutral=True   # Returns are neutralized within each group (removes sector exposure)
group_neutral=False  # Raw returns (includes sector effect)
```

### by_group
```python
by_group=True    # Separate tear sheet for each group
by_group=False   # Aggregate across groups
```

---

## 🎯 Interpretation Quick Guide

### Good Factor Indicators
- ✅ IC Mean > 0.02
- ✅ IC t-stat > 2 (significance)
- ✅ IC consistent over time (low IC Std ratio)
- ✅ Monotonic return spread (Q5 > Q4 > Q3 > Q2 > Q1 or vice versa)
- ✅ Low turnover relative to returns
- ✅ Positive skew in IC
- ✅ Stable ranking (high autocorrelation)

### Bad Factor Indicators
- ❌ IC Mean ≈ 0 or negative
- ❌ t-stat < 1.96
- ❌ IC highly variable (high Std)
- ❌ Non-monotonic returns (jumbled quantile returns)
- ❌ Extreme turnover
- ❌ Negative skew
- ❌ Factor rankings random (low autocorrelation)

---

## 🛠️ Troubleshooting

### Problem: "NaN in factor_data"
- **Cause**: Price data has NaNs or factor misaligned with prices
- **Fix**: `factor_data = get_clean_factor_and_forward_returns(..., max_loss=0.1)`

### Problem: "Not enough data in quantiles"
- **Cause**: Too few observations per quantile
- **Fix**: Use fewer quantiles: `quantiles=3` instead of `quantiles=10`

### Problem: "Tear sheet looks blank/error"
- **Cause**: Factor data has insufficient valid rows
- **Fix**: Check input sizes: at least 100 observations minimum

### Problem: "High turnover destroying returns"
- **Cause**: Normal if factor value changes rapidly
- **Fix**: Consider holding period longer or smoothing factor

---

## 📚 Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|------------|
| Observations | 100 | 1000+ |
| Assets | 5 | 20+ |
| Quantiles | depends | 5-10 |
| Periods | 1 | 3-5 |
| Frequency | Daily | Daily or higher |

---

## 🚀 Usage in pycmqlib3

All examples in codebase follow this pattern:

```python
# From bktest_daily_factor_analysis.ipynb
import alphalens as al

# Step 1: Prepare factor and prices
fac_ts = factor_series_multiindex  # (date, asset) indexed
pricing_data = price_dataframe     # assets as columns

# Step 2: Get factor data
factor_data = al.utils.get_clean_factor_and_forward_returns(
    fac_ts, pricing_data, 
    quantiles=10,        # Deciles
    periods=(1, 5, 10)   # 1-day, 5-day, 10-day returns
)

# Step 3: Analyze
al.tears.create_information_tear_sheet(factor_data)
al.tears.create_full_tear_sheet(factor_data)
```

---

## 🔗 Key Module Structure

```
alphalens
├── utils              # Data preparation
│   ├── get_clean_factor_and_forward_returns()
│   ├── get_clean_factor()
│   └── compute_forward_returns()
├── performance        # Metric computation
│   ├── mean_return_by_quantile()
│   ├── ic()
│   ├── quantile_turnover()
│   └── factor_rank_auto_correlation()
├── tears              # Integrated tear sheets
│   ├── create_full_tear_sheet()
│   ├── create_information_tear_sheet()
│   ├── create_returns_tear_sheet()
│   └── create_turnover_tear_sheet()
└── plotting           # Individual plots
    ├── plot_quantile_returns_*()
    ├── plot_ic_*()
    ├── plot_turnover_*()
    └── plot_cumulative_*()
```

---

*Quick Reference for Alphalens Factor Analysis Library*
*Generated: May 2026 | pycmqlib3 Research*
