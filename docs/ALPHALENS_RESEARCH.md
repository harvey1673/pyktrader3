# Alphalens Feature Research

Comprehensive analysis of Alphalens capabilities used in pycmqlib3 codebase and full feature set based on documentation research.

---

## 1. DATA PREPARATION & FACTOR INGESTION

### Core Data Preparation Functions

#### `al.utils.get_clean_factor_and_forward_returns()`
- **Purpose**: Main entry point for Alphalens; prepares factor data for analysis
- **Parameters**:
  - `factor`: Time-indexed factor series (date, asset, factor_value)
  - `prices`: Price data (usually close prices) used for computing returns
  - `quantiles`: Number of quantiles to split factor into (e.g., 5, 10)
  - `periods`: Forward return calculation periods, e.g., `(1, 5, 10)` for 1, 5, 10 day returns
  - `groupby` (optional): Column for grouping (e.g., sector, asset class)
  - `groupby_labels` (optional): Custom labels for groupby categories
  - `bins` (optional): Discrete bins instead of quantiles
  - `max_loss`: Maximum allowed NaN ratio before dropping observation
- **Returns**: Cleaned factor_data DataFrame with multi-index (date, asset) and columns:
  - Factor value
  - Forward returns for each period
  - Quantile/bin assignment
  - Optional groupby category
- **Usage in codebase**: 
  ```python
  factor_data = al.utils.get_clean_factor_and_forward_returns(
      fac_ts, pricing_data, 
      quantiles=10, 
      periods=(1,5,10)
  )
  ```

#### `al.utils.get_clean_factor()`
- **Purpose**: Similar to above but takes pre-computed forward returns
- **Parameters**:
  - `factor`: Factor series
  - `forward_returns`: Pre-computed forward returns
  - `quantiles`: Number of quantiles
  - `bins`: Alternative to quantiles
  - Similar optional parameters as `get_clean_factor_and_forward_returns()`
- **Usage**: For manual control over forward return computation
  ```python
  factor_data = al.utils.get_clean_factor(
      fac_ts, res, 
      quantiles=10, 
      bins=None
  )
  ```

#### `al.utils.compute_forward_returns()`
- **Purpose**: Compute forward returns from pricing data
- **Parameters**:
  - `prices`: Price series
  - `periods`: Periods for return calculation
- **Returns**: Forward returns DataFrame
- **Usage**: Intermediate step when pre-computing returns
  ```python
  res = al.utils.compute_forward_returns(fac_ts, pricing_data)
  ```

### Supporting Data Utilities
- Data validation and NaN handling
- Factor alignment with price data
- Multi-asset factor preparation
- Period standardization

---

## 2. STATISTICAL METRICS & ANALYSIS

### Information Coefficient (IC) Metrics

**Information Coefficient (IC)**
- Measures correlation between factor and forward returns
- Typically computed as rank correlation (Spearman's rho)
- Key statistic for factor quality assessment
- Variations:
  - **IC Mean**: Average IC across all periods
  - **IC Std**: Standard deviation of IC (volatility of predictive power)
  - **Risk-Adjusted IC (IC/Std or IR)**: Information Ratio - IC Mean / IC Std
  - **t-stat**: t-statistic of IC (IC Mean / Std(IC) / sqrt(N))
  - **p-value**: Statistical significance of IC
  - **Skew**: Skewness of IC distribution
  - **Kurtosis**: Excess kurtosis of IC distribution

### Quantile-Based Returns Analysis

**Mean Return by Quantile**
- Isolates factor effect by computing average returns per quantile
- Shows monotonic increase/decrease pattern if factor is effective
- Multi-period analysis: returns at 1-day, 5-day, 10-day, etc. horizons

**Quantile Return Spread**
- Long-Short spread (Q10 - Q1, or top quantile minus bottom quantile)
- Shows profit potential from going long top quantile, short bottom
- Computed with optional constraints:
  - `long_short`: Include long-short spread analysis
  - `group_neutral`: Neutralize sector/group exposures
  - `by_group`: Show results separately for each group

### Turnover & Transaction Costs

**Factor Turnover Metrics**
- Measures trading activity required to maintain quantile-based strategy
- Typically computed as percentage of portfolio turned over per period
- Higher turnover = higher transaction costs

**Specific Metrics**:
- Turnover by quantile
- Portfolio turnover across holding periods
- Top/bottom quantile turnover separately
- Rolling turnover analysis

### Returns Statistics

**Per-Period Returns Analysis**
- Compound returns over different holding periods
- Cumulative returns by quantile
- Log returns for theoretical properties

**Risk Metrics**
- Annualized volatility/standard deviation
- Sharpe ratio (if benchmark/rf rate provided)
- Maximum drawdown
- Calmar ratio

---

## 3. VISUALIZATION & PLOTTING FUNCTIONS

### Tear Sheet Functions (Integrated Analysis & Plotting)

#### `al.tears.create_information_tear_sheet()`
- **Purpose**: Generates IC analysis visualization and statistics table
- **Produces**:
  - IC mean/std table with statistical significance
  - IC over time plot (line chart)
  - IC distribution (histogram)
  - Breakdown by factor quantile
  - Optional groupby analysis
- **Parameters**:
  - `factor_data`: Cleaned factor data from `get_clean_factor_and_forward_returns()`
  - `by_group`: Show IC separately for each group
  - `group_neutral`: Compute group-neutral IC
- **Usage**:
  ```python
  al.tears.create_information_tear_sheet(factor_data)
  ```

#### `al.tears.create_full_tear_sheet()`
- **Purpose**: Comprehensive factor analysis combining all available analyses
- **Produces**: Multi-part analysis including:
  1. Returns Analysis
  2. Information Coefficient Analysis
  3. Turnover Analysis
  4. Grouped Analysis (if groupby provided)
- **Parameters**:
  - `factor_data`: Cleaned factor data
  - `long_short`: Include long-short analysis (default True)
  - `group_neutral`: Neutralize group effects
  - `by_group`: Show by-group breakdowns
- **Usage**:
  ```python
  al.tears.create_full_tear_sheet(factor_data)
  ```

#### `al.tears.create_returns_tear_sheet()`
- **Purpose**: Focuses on return-based analysis
- **Produces**:
  - Mean returns by quantile table
  - Returns over time (multiple subplots for different periods)
  - Cumulative returns by quantile
  - Long-short spread analysis
  - Optional sector analysis
- **Parameters**: Similar to other tear sheets
- **Status**: Commented out in codebase but available

#### `al.tears.create_turnover_tear_sheet()`
- **Purpose**: Dedicated turnover analysis
- **Produces**:
  - Turnover tables by quantile and period
  - Turnover evolution over time
  - Cost implications if transaction costs provided
- **Parameters**:
  - `factor_data`: Cleaned factor data
  - `turnover_periods`: Specific periods for turnover analysis

### Individual Plotting Functions

**Quantile Analysis Plots**
- `al.plotting.plot_quantile_returns_violin()`: Violin plot of returns by quantile
- `al.plotting.plot_mean_quantile_returns_spread_time_series()`: Long-short spread over time
- `al.plotting.plot_mean_quantile_returns_bar_plot()`: Bar chart of average returns by quantile

**Information Coefficient Plots**
- `al.plotting.plot_ic_ts()`: IC time series with significance markers
- `al.plotting.plot_ic_hist()`: IC distribution histogram
- `al.plotting.plot_ic_qq()`: Q-Q plot of IC against normal distribution

**Turnover Plots**
- `al.plotting.plot_turnover_time_series()`: Turnover evolution
- `al.plotting.plot_turnover_by_quantile()`: Turnover separated by quantile

**Factor Value Plots**
- `al.plotting.plot_factor_rank_auto_correlation()`: Factor autocorrelation
- `al.plotting.plot_monthly_returns_heatmap()`: Calendar heatmap of returns

**Cumulative Performance Plots**
- `al.plotting.plot_cumulative_returns_by_quantile()`: Cumulative returns comparison

---

## 4. QUANTILE & GROUP ANALYSIS

### Quantile-Based Analysis Functions

#### `al.performance.quantile_returns()`
- Mean returns for each quantile across periods
- Enables ranking factor efficacy

#### `al.performance.mean_return_by_quantile()`
- Detailed statistics by quantile
- Includes min, max, mean, std per quantile

#### `al.performance.quantile_turnover()`
- How much the portfolio membership in each quantile changes
- Important for understanding rebalancing costs

#### `al.performance.factor_rank_auto_correlation()`
- Measures how stable factor rankings are over time
- Auto-correlation of factor ranks between consecutive periods
- 0 = random, 1 = perfectly persistent

### Groupby Analysis

**Purpose**: Segment analysis (e.g., by sector, market capitalization, region)

**Key Functions**:
- `al.performance.performance_stats_by_group()`: Statistics broken down by group
- Group-specific IC analysis
- Returns analysis per group
- Turnover analysis per group

**Supports**:
- Multi-level grouping
- Custom groupby fields in factor_data
- Group-neutral (market-neutral) calculations

---

## 5. ADVANCED ANALYSIS CAPABILITIES

### Event-Based Analysis

#### `al.tears.create_event_tear_sheet()` (if available)
- Analyze factor around specific events (earnings, splits, etc.)
- Event window returns analysis
- Event-specific IC computation

### Performance Attribution

- **By Period**: Results aggregated over different time horizons
- **By Quantile**: Isolated quintile/decile performance
- **By Group**: Performance separated by classifications

### Risk Analysis

**Factor Exposure Risk**
- Correlation between factor and market returns
- Beta of factor-sorted portfolios
- Factor concentration risk

**Transaction Cost Modeling**
- Integrate turnover with assumed bid-ask spreads
- Net-of-costs Sharpe ratios
- Break-even transaction cost analysis

### Stability & Persistence

- **Rolling window analysis**: Performance consistency over time
- **Forward-test stability**: Period-by-period IC and returns
- **Drawdown analysis**: Max drawdown by quantile/group

---

## 6. FACTOR DATA STRUCTURE

### Input Format
```
factor_data = MultiIndex DataFrame with:
- Index: (date, asset)
- Columns:
  - factor: The factor value
  - <period>D: Forward returns (1D, 5D, 10D, etc.)
  - quantile: Quantile assignment (1-N)
  - group: Optional groupby category
```

### Data Requirements
- At least 100-200 observations per quantile (minimum)
- No missing values after cleaning (NaN removal)
- Consistent time frequency (daily, monthly, etc.)
- Asset-date alignment

---

## 7. COMPUTATIONAL DETAILS

### Correlation Methods
- **Spearman Rank Correlation**: Primary for IC (default)
- **Pearson Correlation**: Alternative (less common due to outlier sensitivity)
- **Kendall Tau**: Robust rank correlation option

### Period Return Computation
- Simple returns: (P_t+n / P_t) - 1
- Log returns: ln(P_t+n / P_t)
- Adjustments for splits, dividends (in equity context)

### Statistical Testing
- t-statistics for IC significance
- p-values under null hypothesis of zero IC
- Multiple testing corrections (if by-group analysis)

---

## 8. USAGE IN PYCMQLIB3 CODEBASE

### Notebooks Using Alphalens

1. **bktest_daily_factor_analysis.ipynb**
   - Uses: `al.utils.get_clean_factor_and_forward_returns()`
   - Uses: `al.tears.create_information_tear_sheet()`
   - Uses: `al.tears.create_full_tear_sheet()`
   - Commodity factor analysis (10 quantiles, 1/5/10 day periods)

2. **all_product_statistics.ipynb**
   - Similar comprehensive analysis
   - Multi-product statistics

3. **bktest_xsmom1_ret.ipynb** (old_backtest)
   - Uses: `al.utils.compute_forward_returns()`
   - Uses: `al.utils.get_clean_factor()`
   - Alternative factor data preparation approach

### Pattern in Codebase
```python
# Step 1: Prepare factor and pricing data
factor_ts = df[factor_column]  # (date, asset) indexed Series
pricing_data = df[price_columns]  # DataFrame of prices by asset

# Step 2: Create factor data
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor_ts, 
    pricing_data, 
    quantiles=10,  # Deciles
    periods=(1, 5, 10)  # Multi-period analysis
)

# Step 3: Analyze
al.tears.create_information_tear_sheet(factor_data)
al.tears.create_full_tear_sheet(factor_data)
```

---

## 9. KEY STATISTICS EXPLAINED

### Information Coefficient (IC)
- **Interpretation**: 
  - IC > 0: Factor positively predicts returns
  - IC < 0: Factor negatively predicts returns
  - |IC| > 0.02: Generally considered meaningful
  - |IC| > 0.05: Strong predictive power
- **Range**: -1 to +1 (for rank correlation)

### IC/Std Ratio (Information Ratio)
- Measures consistency of factor's predictive power
- Higher = more stable IC
- Similar to Sharpe ratio concept

### t-statistic
- Measures statistical significance of IC
- t-stat > 1.96 ≈ 95% confidence that IC ≠ 0
- t-stat > 2.576 ≈ 99% confidence

### Turnover
- Low turnover: Strategy is stable, lower costs
- High turnover: Frequent rebalancing needed, higher costs
- Typical threshold: < 5-10% monthly turnover considered "low"

### Skew & Kurtosis of IC
- Skew > 0: IC tends to be positive
- Skew < 0: IC tends to be negative
- High positive kurtosis: IC has more extreme values (both high and low)

---

## 10. LIMITATIONS & CONSIDERATIONS

### Look-Ahead Bias
- Ensure factor computation uses only past data
- Forward returns computed from future prices

### Survivorship Bias
- Factor data may exclude delisted assets
- Analysis subject to selection effects

### Multiple Testing
- Running many factors increases false discovery rate
- Requires multiple testing corrections for significance claims

### Data Quality
- NaN handling and cleaning important
- Quantile assignment requires sufficient observations per period

### Asset Class Specifics
- Commodity markets (used in codebase): different microstructure
- Equity: typical with sector groupby
- Futures: contiguous contracts or roll adjustments needed

---

## 11. SUMMARY TABLE: FUNCTION CATEGORIES

| Category | Function | Purpose |
|----------|----------|---------|
| **Data Prep** | get_clean_factor_and_forward_returns | Primary ingestion |
| **Data Prep** | get_clean_factor | Alternative ingestion |
| **Data Prep** | compute_forward_returns | Manual forward computation |
| **Metrics** | IC analysis functions | Correlation to returns |
| **Metrics** | Mean return by quantile | Ranking effectiveness |
| **Metrics** | Turnover analysis | Trading activity cost |
| **Metrics** | Factor autocorrelation | Ranking stability |
| **Plotting** | create_information_tear_sheet | IC visualization |
| **Plotting** | create_full_tear_sheet | Complete analysis |
| **Plotting** | create_returns_tear_sheet | Returns focus |
| **Plotting** | create_turnover_tear_sheet | Turnover focus |
| **Plotting** | plot_* functions | Individual plots |
| **Analysis** | Performance by quantile | Ranking breakdown |
| **Analysis** | Performance by group | Segment analysis |
| **Analysis** | Event tear sheet | Event window analysis |

---

## 12. RECOMMENDED ANALYSIS WORKFLOW

1. **Prepare Data**
   ```python
   factor_data = al.utils.get_clean_factor_and_forward_returns(...)
   ```

2. **Run Quick Information Check**
   ```python
   al.tears.create_information_tear_sheet(factor_data)
   ```

3. **Full Comprehensive Analysis**
   ```python
   al.tears.create_full_tear_sheet(factor_data)
   ```

4. **Custom Analysis (if needed)**
   - Extract specific metrics using performance functions
   - Create custom plots with plotting functions
   - Analyze by group with group-specific functions

---

*Research Date: May 2026*
*Based on: Quantopian Alphalens GitHub repository, pycmqlib3 notebooks, and observed usage patterns*
