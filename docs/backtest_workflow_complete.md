# Production Backtest to Dashboard - Complete Workflow

## Overview

This workflow replaces manually exporting the Jupyter notebook to HTML. Instead:
1. Run your backtest notebook as usual
2. Save results at the end
3. Run a Python script to generate an organized, interactive dashboard

## Step-by-Step Instructions

### Step 1: Add Save Cell to Notebook

At the **END** of `bktest/bktest_prod_daily_run.ipynb`, add a new cell with the code from `bktest/save_results_cell.py`:

```python
# Save Results for Dashboard
import pickle

results = {
    'bt_dict': bt_dict,
    'pnl_dict': pnl_dict,
    'signal_dict': signal_dict,
    'holding_dict': holding_dict,
    'port_pnl': port_pnl,
    'pnl_by_signal': pnl_by_signal,
    'strategy_groups': {
        'Premium': [s[0] for s in prem_strats],
        'Metal': [s[0] for s in metal_strats],
        'Miscellaneous': [s[0] for s in misc_strats],
        'Ferrous': [s[0] for s in ferrous_strats],
        'Ferrous Spread': [s[0] for s in ferrous_spd_strats],
        'Base': [s[0] for s in base_strats],
        'Energy': [s[0] for s in energy_strats],
        'Macro': [s[0] for s in macro_strats],
        'Macro 2': [s[0] for s in macro2_strats],
        'Seasonal': [s[0] for s in seazn_strats],
        'Bond': [s[0] for s in bond_strats],
        'AU Spread': [s[0] for s in auspd_strats],
    }
}

date_str = tday.strftime('%Y%m%d')
cache_file = f'C:/dev/data/backtest_results_{date_str}.pkl'
with open(cache_file, 'wb') as f:
    pickle.dump(results, f)

print(f"✓ Results saved to {cache_file}")
```

### Step 2: Run Your Notebook

Execute the notebook as usual. When done, the save cell will create a pickle file with all results.

### Step 3: Generate Dashboard

Run the dashboard generator script:

```bash
# Using today's date (default)
python tools/run_prod_backtest.py --output C:/dev/data/dashboard.html

# Or specify a date
python tools/run_prod_backtest.py --date 2026-02-13 --output C:/dev/data/dashboard.html

# Or load from specific cache file
python tools/run_prod_backtest.py --load-cached C:/dev/data/my_results.pkl --output dashboard.html
```

### Step 4: View Dashboard

Open the generated HTML file in your browser. The dashboard includes:

- **Overview Tab**
  - Portfolio-level KPIs (Total Return, Sharpe, Max DD, Volatility, Win Rate)
  - Combined equity curve with plotly (interactive)
  - Strategy summary table (sortable)
  - Monthly returns heatmap

- **12 Strategy Group Tabs**
  - Premium, Metal, Miscellaneous, Ferrous, Ferrous Spread, Base, Energy, Macro, Macro 2, Seasonal, Bond, AU Spread
  - Group-level metrics
  - Group equity curve
  - Expandable panels for each strategy with:
    - Equity curve
    - Drawdown chart
    - Returns histogram
    - Rolling Sharpe
  - **Lazy loading**: Charts load only when expanded

- **All Strategies Tab**
  - Sortable table of all strategies
  - Filterable by group

## Benefits Over Notebook HTML Export

1. **Organized**: Strategies grouped into tabs instead of one long page
2. **Smaller file**: Typically < 1MB vs 10+ MB for notebook HTML
3. **Interactive**: Better charts with plotly/hvplot
4. **Faster**: Lazy loading means charts load on demand
5. **Consistent**: Same metrics calculation as notebook (uses MetricsBase)
6. **No notebook needed**: Just run the script, no Jupyter required

## File Locations

- **Notebook**: `c:\dev\pyktrader3\bktest\bktest_prod_daily_run.ipynb`
- **Save cell code**: `c:\dev\pyktrader3\bktest\save_results_cell.py`
- **Dashboard generator**: `c:\dev\pyktrader3\tools\run_prod_backtest.py`
- **Cached results**: `C:\dev\data\backtest_results_YYYYMMDD.pkl`
- **Output dashboard**: `C:\dev\data\dashboard.html` (or your choice)

## Metrics Included

The dashboard uses `MetricsBase.calculate_pnl_stats()` which provides:
- asset_pnl: Asset-level PnL
- portfolio_pnl: Total portfolio PnL
- asset_cumpnl: Cumulative asset PnL
- portfolio_cumpnl: Cumulative portfolio PnL
- **pnl_per_trade**: Average PnL per trade (in bps)
- **turnover**: Portfolio turnover
- asset_sharpe_stats: Sharpe ratios
- Plus: sortino, calmar, win_rate, max_drawdown

## Troubleshooting

**Error: Cache file not found**
- Make sure you ran the notebook and added the save cell
- Check the file path matches the date

**Missing strategies in dashboard**
- Verify strategy names match between notebook variables and pnl_by_signal columns
- Check strategy_groups mapping is correct

**Dashboard looks empty**
- Ensure pnl_by_signal has data
- Check console for errors when generating

## Advanced: Serve Mode (Future)

For very large datasets, you can run a Panel server instead of HTML export:

```python
# In development - not yet implemented
python tools/run_prod_backtest.py --serve --port 5006
```

This will launch an interactive server at `http://localhost:5006` with full interactivity and faster loading for large datasets.

---

**Status**: ✓ Ready to use
**Last Updated**: 2026-03-01
