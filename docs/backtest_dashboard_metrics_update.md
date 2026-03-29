# Backtest Dashboard - Metrics Update

## Changes Made

Updated the dashboard to use the existing `MetricsBase` class from `pycmqlib3.analytics.btmetrics` instead of custom metric calculations.

### What Changed

**Before:**
- Custom calculations for CAGR, Sharpe, max drawdown, volatility, win rate
- Manual implementation of each metric

**After:**
- Uses `MetricsBase.calculate_pnl_stats()` which returns:
  - `asset_pnl`: Asset-level PnL
  - `portfolio_pnl`: Portfolio total PnL  
  - `asset_cumpnl`: Cumulative asset PnL
  - `portfolio_cumpnl`: Cumulative portfolio PnL
  - `pnl_per_trade`: Average PnL per trade (in bps)
  - `turnover`: Portfolio turnover statistics
  - `asset_sharpe_stats`: Sharpe ratios by asset
  - Plus any additional performance metrics (sortino, calmar, win_rate, max_drawdown)

### Functions Updated

1. **`calculate_strategy_metrics()`** - Now creates a MetricsBase instance internally
2. **`extract_metrics_summary()`** - NEW function to extract display-friendly metrics from pnl_stats
3. **Dashboard components** - Updated to use `extract_metrics_summary()` for KPIs

### Integration with Notebook

In your notebook, if you already have bt_metrics objects from MetricsBase:

```python
from tools.backtest_dashboard.metrics import calculate_strategy_metrics_from_bt

# If you have a MetricsBase instance:
metrics = calculate_strategy_metrics_from_bt(bt_metrics)

# This gives you:
# - metrics['pnl_stats']: Full stats dict
# - metrics['asset_pnl']: Asset-level PnL DataFrame  
# - metrics['pnl_per_trade']: PnL per trade Series
# - metrics['turnover']: Turnover Series
```

### Benefits

1. **Consistency**: Uses same metric calculations as your existing backtests
2. **Completeness**: Gets all metrics from `calculate_pnl_stats()` including pnl_per_trade and turnover
3. **Maintainability**: Single source of truth for metric calculations

### No Breaking Changes

The dashboard still works the same way from the notebook - just pass `port_pnl` and `pnl_by_signal` as before. The metrics are now calculated using your existing MetricsBase class internally.
