# Alphalens: Detailed Metrics & Visualization Reference

Comprehensive guide to every metric and visualization Alphalens produces.

---

## SECTION 1: INFORMATION COEFFICIENT (IC) METRICS

### What is IC?

Information Coefficient measures the **rank correlation between factor values and forward returns**. It answers: "Does my factor predict future returns?"

### IC Metrics Explained

#### 1. **IC Mean**
- **Definition**: Average Information Coefficient across all time periods
- **Calculation**: Mean of IC values computed for each date
- **Range**: -1.0 to +1.0 (for rank correlation)
- **Interpretation**:
  - IC > 0: Factor positively predicts returns
  - IC < 0: Factor negatively predicts returns (but still useful if you short)
  - IC ≈ 0: No predictive power
- **Benchmark**: 
  - |IC| > 0.01: Barely meaningful
  - |IC| > 0.02: Meaningful in practice
  - |IC| > 0.05: Strong predictive signal
- **Example**: 
  - IC Mean = 0.03 means on average, high factor values correlate with 3% higher returns

#### 2. **IC Std (IC Standard Deviation)**
- **Definition**: Standard deviation of IC across all periods
- **Interpretation**: 
  - Low Std: IC is consistent/stable (predictable predictability)
  - High Std: IC varies wildly period-to-period (unstable)
- **Trade-off**: 
  - Low Std + High Mean = Best (consistent alpha generation)
  - High Std + High Mean = Risky (alpha is sporadic)
- **Impact on Strategy**:
  - High IC Std means you can't rely on consistent returns
  - Reduces practical value despite high mean IC

#### 3. **IC/Std Ratio (Information Ratio)**
- **Definition**: `IC Mean / IC Std` - quality per unit of volatility
- **Calculation**: Similar to Sharpe ratio but for IC stability
- **Interpretation**:
  - Ratio > 0.1: Good (consistent positive IC)
  - Ratio > 0.15: Very good
  - Ratio < 0.05: Questionable (too much noise)
- **Example**: 
  - Factor A: Mean=0.03, Std=0.05 → Ratio=0.6 (good)
  - Factor B: Mean=0.03, Std=0.10 → Ratio=0.3 (mediocre despite same mean)

#### 4. **t-statistic**
- **Definition**: Statistical significance test of IC
- **Formula**: `t-stat = IC Mean / (IC Std / sqrt(N_periods))`
- **Interpretation**:
  - t-stat > 1.96 ≈ 95% confidence IC ≠ 0 (2-tail test)
  - t-stat > 2.576 ≈ 99% confidence
  - t-stat > 3.29 ≈ 99.9% confidence
- **Usage**: Distinguishes lucky factors from truly predictive ones
- **Multiple Testing**: If testing 100 factors, adjust threshold (e.g., divide p-value by 100)

#### 5. **p-value**
- **Definition**: Probability that IC = 0 under null hypothesis
- **Interpretation**:
  - p < 0.05: Statistically significant at 95% confidence
  - p < 0.01: Statistically significant at 99% confidence
  - p > 0.10: Not significant (likely false positive)
- **Note**: p-value inversely related to t-stat

#### 6. **IC Skew**
- **Definition**: Asymmetry of IC distribution
- **Formula**: Third moment of IC distribution
- **Interpretation**:
  - Skew > 0 (positive): Right-tailed - outliers on positive side
    - Means IC is "usually" negative but occasionally very positive
    - Good: Means factor has occasional positive surprises
  - Skew < 0 (negative): Left-tailed - outliers on negative side
    - Bad: Factor has negative outliers/tail risks
  - Skew ≈ 0: Symmetric distribution
- **Quality Signal**: Positive skew is preferred (positive surprises)
- **Practical Impact**: Relates to downside risk in strategy returns

#### 7. **IC Kurtosis (Excess Kurtosis)**
- **Definition**: Peakedness and tail weight of IC distribution
- **Formula**: (Fourth moment / Std^4) - 3 [excess = subtract 3 from normal]
- **Interpretation**:
  - Kurtosis > 0 (positive): "Fat tails" - more extreme values
    - Higher peaks at mean, heavier tails
    - Indicates occasional very good/bad periods
  - Kurtosis < 0 (negative): "Light tails" - consistent IC
    - More uniform distribution
  - Normal distribution: Kurtosis ≈ 0
- **Strategy Impact**: 
  - High positive kurtosis = occasional large gains/losses
  - Suggests risk management needed

---

## SECTION 2: RETURNS ANALYSIS METRICS

### Quantile-Based Returns

#### **Mean Return by Quantile**
- **Definition**: Average forward return for each quantile group
- **Calculation**: Group factor into N quantiles (e.g., 5), compute average forward return per group
- **Example Output**:
  ```
  Quantile    1-Day Ret    5-Day Ret    10-Day Ret
      1         0.05%       0.15%        0.30%
      2         0.10%       0.25%        0.45%
      3         0.12%       0.30%        0.50%
      4         0.08%       0.20%        0.35%
      5         0.03%       0.10%        0.25%
  ```
- **Ideal Pattern**: Monotonic increase (1→5) or decrease (5→1)
  - Shows factor ranks accurately predict returns
  - Non-monotonic = questionable signal quality
- **Long-Short Spread**: Return(Q5) - Return(Q1) = profit potential

#### **Long-Short Return Spread**
- **Definition**: Return difference between top and bottom quantiles
- **Calculation**: `Mean_Return_Q5 - Mean_Return_Q1`
- **Interpretation**:
  - Positive spread: Factor is directionally correct
  - Negative spread: Factor is inverse predictor (invert it!)
  - Spread > Returns: Better than just buying average stock
- **Annualization**: Spread × √252 (for daily data) = annual potential
- **Example**:
  - Spread = 0.15% per day → 0.15% × √252 ≈ 2.4% annualized
- **Cost Consideration**: Subtract transaction costs from spread

#### **Cumulative Returns by Quantile**
- **Definition**: Compound return trajectory of each quantile portfolio over time
- **Graph**: Line plot showing growth of $1 invested in each quantile
- **Interpretation**:
  - Spread between Q1 and Q5: Shows compounding power
  - Volatility differences: Q1 vs Q5 risk profile
  - Drawdown periods: When strategy underperforms
- **Signal Quality**: Wide separation = stronger signal

### Alternative Period Analysis
- **1-Day Forward Returns**: Very short-term predictability (microstructure/momentum)
- **5-Day Forward Returns**: Short-term (mean reversion window)
- **20-Day Forward Returns**: Intermediate-term (typical rebalance period)
- **60-Day Forward Returns**: Medium-term (monthly portfolio perspective)

---

## SECTION 3: TURNOVER ANALYSIS METRICS

### Why Turnover Matters
- **Definition**: How much of portfolio changes between rebalancing periods
- **Impact**: High turnover = high transaction costs = reduced net returns
- **Practical Concern**: A 2% day return becomes -1% with 3% turnover costs

### Turnover Metrics

#### **Turnover by Quantile**
- **Definition**: How often assets enter/exit each quantile
- **Calculation**: `(|Position_Change|) / 2 × 100%`
- **Interpretation**:
  - Q1 turnover = % of Q1 positions replaced each period
  - Low for Q1: Assets stick in bottom quantile (sticky factor)
  - High for Q5: Assets rapidly enter/exit top quantile (volatile factor)
- **Pattern**: Usually:
  - Middle quantiles (Q2, Q3): Highest turnover
  - Extreme quantiles (Q1, Q5): Lower turnover

#### **Turnover by Holding Period**
- **Definition**: How turnover varies with holding period
- **Periods**: 1-day, 5-day, 20-day, etc.
- **Typical Pattern**: 
  - Turnover decreases with longer holding periods
  - Short-term: 50% daily turnover (fully replace portfolio)
  - Medium-term: 10% weekly turnover (replace 70% of portfolio per week)

#### **Net Turnover Impact**
- **Definition**: Turnover × Transaction Cost Ratio = Drag
- **Calculation Example**:
  - 10% daily turnover × 0.1% transaction cost = 1 bp drag
  - 10% daily turnover × 0.5% transaction cost = 5 bp drag (material!)
- **Break-Even**: What IC needs to offset turnover costs

#### **Turnover Persistence**
- **Definition**: How stable turnover is over time
- **Good Signal**: Consistent turnover (predictable costs)
- **Bad Signal**: Highly variable turnover (unpredictable costs)

---

## SECTION 4: FACTOR RANKING STABILITY

### **Factor Rank Autocorrelation**
- **Definition**: Correlation of factor rankings between consecutive periods
- **Interpretation**:
  - 1.0: Perfect persistence (rank exactly same next period)
  - 0.5: 50% rank correlation (moderate stability)
  - 0.0: Random persistence (factor changes completely)
  - -1.0: Perfect mean reversion (ranks exactly flip)
- **Calculation**: Spearman rank correlation of Factor_t vs Factor_t+1
- **Implications**:
  - High (>0.7): Stable, sticky factor (low turnover needed)
  - Medium (0.3-0.7): Some rebalancing
  - Low (<0.3): Highly dynamic factor (high turnover)
- **Strategy Design**:
  - High autocorr → Hold longer, lower rebalancing costs
  - Low autocorr → Need more frequent rebalancing

---

## SECTION 5: GROUP ANALYSIS METRICS

### When to Use
- **Segment Analysis**: Decompose performance by classification
- **Sector-Neutral Analysis**: Remove sector exposure to isolate factor
- **Robustness Check**: Does factor work across all groups equally?

### Group-Specific Metrics

#### **IC by Group**
- **Definition**: Information Coefficient computed within each group
- **Example**: IC for Tech stocks vs Financials vs Healthcare separately
- **Interpretation**:
  - Consistent IC across groups: Robust factor
  - IC high in one group only: Specific to that group
  - Divergent signs: Factor works opposite ways in different groups

#### **Returns by Group**
- **Definition**: Mean returns by quantile within each group
- **Example**: 
  ```
  SECTOR      Q1 Ret   Q5 Ret   Spread
  Tech        0.04%    0.12%    0.08%
  Finance     0.02%    0.05%    0.03%
  Energy      0.06%    0.15%    0.09%
  ```
- **Interpretation**: Does factor strength vary by group?

#### **Turnover by Group**
- **Definition**: Portfolio churn within each group
- **Practical**: If turnover high in one group, concentrated costs

#### **Group-Neutral Returns**
- **Definition**: Remove group effect from returns
- **Calculation**: Within each group, compute average return; long/short relative to group mean
- **Purpose**: Isolate pure factor effect from group effects
- **Example**: If Tech always outperforms, group-neutral removes that bias

---

## SECTION 6: VISUALIZATIONS PRODUCED

### A. INFORMATION COEFFICIENT PLOTS

#### 1. **IC Time Series Plot**
- **What**: Line chart of IC for each date
- **Axes**: X=Date, Y=IC value
- **Features**:
  - Horizontal line at IC=0
  - Shaded zero band (±1 std err)
  - Color: Green for positive IC, red for negative
- **Interpretation**:
  - Mostly above zero → Consistently predictive
  - Wide swings → Unstable factor
  - Trends: IC degrading over time?

#### 2. **IC Distribution Histogram**
- **What**: Frequency distribution of IC values
- **Features**:
  - X-axis: IC values
  - Y-axis: Frequency (number of periods)
  - Vertical line: Mean IC
  - Curve: Normal distribution overlay
- **Interpretation**:
  - Center > 0: Positively skewed predictive power
  - Narrow: Concentrated IC values (predictable)
  - Wide: Spread IC values (unpredictable)

#### 3. **IC Q-Q Plot**
- **What**: Quantile-quantile plot of IC vs normal distribution
- **Purpose**: Check if IC is normally distributed
- **Interpretation**:
  - Points on 45° line: Normal distribution
  - Points below line on tails: Fat-tailed distribution
  - Points above: Light-tailed distribution
- **Practical**: Normality affects statistical testing validity

#### 4. **IC by Quantile**
- **What**: Bar chart showing IC for each factor quantile separately
- **Interpretation**:
  - Equal height bars: IC consistent across quantiles
  - Unequal: IC driven by specific quantile (e.g., only top quintile predictive)

### B. RETURNS ANALYSIS PLOTS

#### 1. **Mean Return by Quantile (Bar Chart)**
- **What**: Bar chart of average forward returns per quantile
- **Axes**: X=Quantile (1-5), Y=Mean Return
- **Ideal Pattern**: Monotonic increasing/decreasing
- **Variations**: Shows results for different forward periods side-by-side

#### 2. **Cumulative Return by Quantile (Line Chart)**
- **What**: Compound wealth trajectory for each quantile
- **Example**: "$1 grows to $1.52 in Q5, $1.20 in Q1 over 5 years"
- **Interpretation**:
  - Vertical separation: Return differential
  - Trajectory smoothness: Consistent return generation
  - Drawdowns: Common stress periods

#### 3. **Long-Short Return Spread Over Time**
- **What**: Line chart of (Q5 - Q1) return spread for each period
- **Interpretation**:
  - Consistently > 0: Reliable long-short returns
  - Occasionally < 0: Period reversals (factor weakness)
  - Volatility: How stable the spread is

#### 4. **Quantile Returns Violin Plot**
- **What**: Distribution shape of returns for each quantile
- **Features**: Width shows frequency, center shows median, tails show extremes
- **Interpretation**:
  - Q5 wider but shifted right: Returns variable but mostly positive
  - Q1 narrow and centered on zero: Consistent near-zero returns

#### 5. **Monthly Returns Heatmap**
- **What**: Calendar view of returns by quantile and month
- **Axes**: X=Month/Year, Y=Quantile
- **Color**: Green=positive month, Red=negative month
- **Interpretation**:
  - Consistent green in Q5: Reliable generator
  - Red streaks: Factor weakness periods
  - Seasonal patterns: Does factor have seasons?

### C. TURNOVER PLOTS

#### 1. **Turnover Time Series**
- **What**: Line chart of turnover for each rebalancing period
- **Y-axis**: % of portfolio changed
- **Interpretation**:
  - High but stable: Predictable costs
  - Variable: Costs hard to estimate
  - Trending up: Factor becoming less sticky over time?

#### 2. **Turnover by Quantile**
- **What**: Separate time series for each quantile turnover
- **Interpretation**:
  - Q5 spiky: Top performers frequently replaced
  - Q1 smooth: Bottom performers stick around
  - Patterns: Understand rebalancing needs per quantile

#### 3. **Average Turnover vs Holding Period**
- **What**: Bar chart showing how turnover decreases with longer holding
- **Example**: 50% for 1-day, 20% for 5-day, 5% for 20-day
- **Use**: Optimize rebalancing frequency

---

## SECTION 7: INTERPRETATION FRAMEWORK

### The Complete Analysis Checklist

#### 1. **Is the Signal Real? (IC Check)**
- [ ] IC Mean > 0.02 (meaningful magnitude)
- [ ] t-stat > 1.96 (statistically significant)
- [ ] IC Std reasonable relative to mean (ratio > 0.05)
- [ ] Positive skew (occasional positive surprises)

#### 2. **Is it Consistent? (Stability Check)**
- [ ] IC roughly consistent period-to-period (IC time series smooth)
- [ ] Factor ranks persist (autocorrelation > 0.3)
- [ ] Works across multiple groups (if applicable)
- [ ] Signal doesn't degrade over time

#### 3. **Is it Profitable? (Returns Check)**
- [ ] Long-short spread > 0 (right direction)
- [ ] Cumulative returns monotonic by quantile
- [ ] Spread magnitude > potential costs
- [ ] No extreme outlier periods
- [ ] Comparable to benchmark strategies

#### 4. **Is it Practical? (Turnover Check)**
- [ ] Turnover low enough for transaction costs
- [ ] Break-even analysis: (Long-Short Spread) vs (Turnover × Cost)
- [ ] Holding periods feasible operationally
- [ ] Turnover stable/predictable

#### 5. **Is it Robust? (Robustness Check)**
- [ ] Results hold across all groups
- [ ] Multiple timeframes show consistent IC
- [ ] Forward vs holdout periods consistent
- [ ] Results not driven by outliers/specific assets

### Traffic Light Interpretation

| Metric | Red ⛔ | Yellow ⚠️ | Green ✅ |
|--------|--------|----------|----------|
| IC Mean | < 0.01 | 0.01-0.02 | > 0.02 |
| t-stat | < 1.96 | 1.96-2.5 | > 2.5 |
| IC Ratio | < 0.05 | 0.05-0.1 | > 0.1 |
| IC Stability | High Std | Moderate | Low Std |
| Return Spread | Negative | Flat | Positive |
| Turnover | >20% daily | 5-20% | <5% |
| Autocorr | < 0 | 0-0.3 | > 0.3 |

---

## SECTION 8: COMMON PITFALLS & INTERPRETATION ERRORS

### ❌ Mistake 1: High IC Mean with High Std
- **Error**: "IC = 0.04 is good"
- **Reality**: If Std = 0.06, Ratio = 0.67 (mediocre)
- **Fix**: Always check IC/Std, not just mean

### ❌ Mistake 2: Ignoring Turnover
- **Error**: "0.15% daily spread is great"
- **Reality**: At 50% turnover × 0.5% costs = 2.5 bps drag (shrinks spread to 0.125%)
- **Fix**: Always calculate net-of-costs returns

### ❌ Mistake 3: Non-Monotonic Returns Ignored
- **Error**: "Q5 > Q3 > Q1, even if Q2, Q4 are jumbled"
- **Reality**: Suggests factor is ranking only top/bottom, not overall
- **Fix**: Check entire quantile structure, not just extremes

### ❌ Mistake 4: Period-Specific IC
- **Error**: "IC is 0.05 but only for 10-day forward returns"
- **Reality**: Factor might not work for 1-day or 5-day
- **Fix**: Check consistency across periods; use multiple periods together

### ❌ Mistake 5: Ignoring Group Effects
- **Error**: "Overall IC = 0.03, robust factor"
- **Reality**: IC = 0.05 in Tech, 0.00 in Financials (not robust)
- **Fix**: Always decompose by groups if available

---

## SECTION 9: METRICS PRIORITIZATION

### If You Only Had Time for One Check:
1. **t-stat** → Is factor real or lucky?

### If You Had Three Checks:
1. **t-stat** → Real?
2. **IC Ratio** → Consistent?
3. **Long-Short Spread** → Profitable?

### Full Tier-1 Analysis (Minimal):
1. IC Mean/Std/t-stat
2. Return Spread (Q5-Q1)
3. Turnover costs
4. IC by group (if groups exist)

### Complete Tier-2 Analysis (Best Practice):
- All Tier-1 plus:
- IC autocorrelation (factor stability)
- Monthly returns heatmap (seasonal patterns)
- Turnover time series (cost predictability)
- Forward period variations (1D vs 5D vs 20D)

---

*Reference: Complete Metrics Guide for Alphalens Analysis*
*pycmqlib3 Research | May 2026*
