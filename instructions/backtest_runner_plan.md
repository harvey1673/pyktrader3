# Plan: Standalone Backtest Runner to Replace Notebook

## Goal
Create a standalone Python script (`tools/run_prod_backtest.py`) that completely replaces `bktest/bktest_prod_daily_run.ipynb`:
- Loads data (prices, spreads, factors)
- Runs all strategy backtests
- Calculates metrics
- Generates interactive dashboard HTML
- NO notebook needed!

## Architecture

```
run_prod_backtest.py
├── 1. Data Loading Module
│   ├── Load futures prices (load_saved_fut)
│   ├── Load spreads
│   ├── Load factors
│   └── Prepare DataFrames
├── 2. Strategy Execution Module
│   ├── Loop through strategy groups
│   ├── Generate signals for each strategy
│   ├── Calculate holdings
│   ├── Create MetricsBase objects
│   └── Store in dicts (bt_dict, pnl_dict, signal_dict, holding_dict)
├── 3. Portfolio Aggregation Module
│   ├── Aggregate signals across strategies
│   ├── Calculate portfolio PnL
│   ├── Calculate strategy group PnL
│   └── Create pnl_by_signal DataFrame
├── 4. Dashboard Generation Module
│   ├── Create Overview tab (portfolio + groups)
│   ├── Create 12 strategy group tabs
│   └── Save HTML with lazy-loaded charts
└── 5. CLI Interface
    ├── Arguments: --date, --output, --cache-folder
    └── Progress reporting
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Create `backtest_runner.py` module
  - [ ] `DataLoader` class: Load prices, spreads, factors
  - [ ] `StrategyExecutor` class: Run strategies, generate signals
  - [ ] `PortfolioAggregator` class: Combine results
- [ ] Create strategy configuration (weights, parameters)
- [ ] Test with 2-3 strategies to verify workflow

### Phase 2: Strategy Execution
- [ ] Implement signal generation for each strategy type:
  - [ ] Premium strategies (momentum, carry, basis)
  - [ ] Metal strategies (inventory, momentum, PBC)
  - [ ] Ferrous strategies
  - [ ] Spread strategies
  - [ ] Macro strategies
  - [ ] Seasonal strategies
  - [ ] Bond strategies
- [ ] Calculate holdings from signals
- [ ] Create MetricsBase for each strategy
- [ ] Store results in dicts

### Phase 3: Portfolio Metrics
- [ ] Aggregate portfolio PnL
- [ ] Calculate strategy group PnL
- [ ] Compute correlations between groups
- [ ] Calculate Sharpe by tenors (3m, 6m, 1y, etc.)
- [ ] Calculate std dev by tenors
- [ ] Get pnl_per_trade and turnover

### Phase 4: Dashboard Updates
- [ ] Update `BacktestDashboard` class for new requirements:
  - [ ] **Overview Tab**:
    - [ ] Portfolio cumulative PnL (plotly)
    - [ ] Asset cumulative PnL (plotly)
    - [ ] Strategy group cumulative PnL (plotly, all groups on one chart)
    - [ ] Correlation matrix of strategy groups (heatmap)
    - [ ] Sharpe ratio by tenors table (portfolio + groups)
    - [ ] Standard deviation by tenors table
    - [ ] PnL per trade by asset (table)
    - [ ] Turnover by asset (table)
  - [ ] **Strategy Group Tabs** (12 tabs):
    - [ ] For each signal in the group:
      - [ ] Cumulative PnL plot (plotly)
      - [ ] Asset-level PnL plot (plotly, expandable)
      - [ ] Turnover by asset (table)
      - [ ] PnL per trade by asset (table)
    - [ ] Lazy loading for charts

### Phase 5: Integration & Testing
- [ ] Integrate all modules into `run_prod_backtest.py`
- [ ] Add CLI with arguments
- [ ] Test with full date range
- [ ] Compare output with notebook results
- [ ] Verify metrics match

### Phase 6: Optimization & Documentation
- [ ] Add caching for intermediate results
- [ ] Add progress bars
- [ ] Create user documentation
- [ ] Add error handling and validation

## Data Flow

```
1. CLI invocation
   └─> run_prod_backtest.py --date 2026-02-13 --output dashboard.html

2. Data Loading
   ├─> Load futures prices from database
   ├─> Load spread data
   ├─> Load factor data
   └─> Align dates, create DataFrames

3. Strategy Execution (loop for each strategy)
   ├─> Get strategy config (name, weight, parameters)
   ├─> Generate signal (call strategy function)
   ├─> Scale signal by weight
   ├─> Calculate holdings
   ├─> Create MetricsBase(holdings, returns)
   ├─> Calculate pnl_stats (pnl, pnl_per_trade, turnover)
   └─> Store in dicts

4. Portfolio Aggregation
   ├─> Sum signals across all strategies
   ├─> Calculate portfolio holdings
   ├─> Calculate portfolio PnL
   ├─> Group by strategy groups
   ├─> Calculate group PnLs
   └─> Create pnl_by_signal DataFrame

5. Metrics Calculation
   ├─> Sharpe by tenors (portfolio + groups)
   ├─> Std dev by tenors
   ├─> Correlation matrix of groups
   ├─> PnL per trade by asset
   └─> Turnover by asset

6. Dashboard Generation
   ├─> Create Overview tab with all portfolio charts
   ├─> Create 12 strategy group tabs
   ├─> Add lazy loading for charts
   └─> Save HTML

7. Output
   └─> dashboard.html (lightweight, organized, interactive)
```

## Key Modules to Create

### 1. `backtest_runner/data_loader.py`
```python
class DataLoader:
    def __init__(self, start_date, end_date):
        ...
    
    def load_futures_prices(self, assets):
        """Load futures price data"""
        
    def load_spreads(self, spread_config):
        """Load spread data"""
        
    def load_factors(self, factor_list):
        """Load factor data"""
        
    def prepare_dataframes(self):
        """Align dates, create returns, etc."""
```

### 2. `backtest_runner/strategy_executor.py`
```python
class StrategyExecutor:
    def __init__(self, df, df_pxchg, cost_dict):
        ...
    
    def execute_strategy(self, strategy_name, weight, params):
        """Execute single strategy, return bt_metrics"""
        
    def execute_all(self, strategy_groups):
        """Execute all strategies, return dicts"""
```

### 3. `backtest_runner/portfolio_aggregator.py`
```python
class PortfolioAggregator:
    def __init__(self, bt_dict, pnl_dict):
        ...
    
    def aggregate_portfolio(self):
        """Calculate total portfolio PnL"""
        
    def aggregate_groups(self, strategy_groups):
        """Calculate strategy group PnLs"""
        
    def calculate_correlations(self):
        """Group correlation matrix"""
```

### 4. Update `tools/backtest_dashboard/dashboard.py`
```python
class BacktestDashboard:
    def create_overview_tab(self):
        # Add:
        # - Strategy group cumulative PnL (all on one chart)
        # - Correlation matrix
        # - Sharpe by tenors (portfolio + groups)
        # - Std dev by tenors
        # - PnL per trade by asset
        # - Turnover by asset
        
    def create_group_tab(self, group_name, signals):
        # For each signal:
        # - Cumulative PnL chart
        # - Asset PnL chart (expandable)
        # - Turnover table
        # - PnL per trade table
```

## Strategy Catalog (from notebook)

Based on the notebook, we need to implement signal generation for:

1. **Premium Strategies**: momentum-based (mom_momma240, mom_hlr_st, cclr_mom_sgnma, colr_mom_sgnma)
2. **Metal Strategies**: PBC, momentum, inventory-based (metal_pbc_ema, metal_mom_hlrhys, metal_inv_hlr)
3. **Ferrous Strategies**: PBC-based (ferrous_pbc_ema, ferrous_pbc_zs)
4. **Ferrous Spread**: Rolling spread strategies (rbi_spd_froll_mt, hci_spd_froll_mt)
5. **Base Strategies**: PBC, inventory (base_pbc_ema, base_inv_hlr)
6. **Energy Strategies**: Momentum-based (energy_mom_zsa)
7. **Macro Strategies**: Credit, PMI factors (macro_cn_credit_zsa, macro_cn_pmi_zs)
8. **Macro2 Strategies**: Inflation (macro_us_infl_zsa)
9. **Seasonal Strategies**: Volatility-based (seazn_hr_vol_st_raw, seazn_ll_vol_mt_raw)
10. **Bond Strategies**: Spread momentum (bond_tr_tfspd_mom_zs)
11. **AU Spread**: Gold/silver spreads (auag_cme_wratio_zs, auag_vix_zsa_mt, etc.)
12. **Miscellaneous**: IV, fund flows (misc_fut_ivol_zsa, misc_cmfund_ret_zsa)

All signal functions should already exist in `pycmqlib3.strategy.signal_repo` or `misc_scripts.fun_factor_update`.

## Timeline Estimate

- **Phase 1**: 2-3 hours (infrastructure)
- **Phase 2**: 4-5 hours (strategy execution - can reuse existing functions)
- **Phase 3**: 1-2 hours (portfolio metrics)
- **Phase 4**: 3-4 hours (dashboard updates)
- **Phase 5**: 2-3 hours (testing & integration)
- **Phase 6**: 1-2 hours (docs & polish)

**Total**: ~13-19 hours of development work

## Success Criteria

✓ Script runs without Jupyter notebook
✓ Generates same PnL as notebook (verify on known date)
✓ Dashboard HTML < 5MB
✓ All 12 strategy groups appear correctly
✓ Charts load properly with lazy loading
✓ Metrics match notebook output (Sharpe, turnover, etc.)
✓ Runtime < 10 minutes for full backtest

## Next Steps

1. Confirm this plan matches your requirements
2. Start Phase 1: Create core infrastructure modules
3. Build incrementally, testing each phase
4. Deliver working standalone script

---

**Ready to proceed with implementation?**
