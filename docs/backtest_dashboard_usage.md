# Backtest Dashboard - Usage Example

## Add this cell at the end of your bktest_prod_daily_run.ipynb notebook:

```python
# ========== Backtest Dashboard ==========
import sys
sys.path.insert(0, 'C:/dev/pyktrader3')
from tools.run_backtest_dashboard import run_dashboard

# Map strategy group names to your strategy lists
strategy_groups_mapping = {
    'Premium': prem_strats,
    'Metal': metal_strats,
    'Miscellaneous': misc_strats,
    'Ferrous': ferrous_strats,
    'Ferrous Spread': ferrous_spd_strats,
    'Base': base_strats,
    'Energy': energy_strats,
    'Macro': macro_strats,
    'Macro 2': macro2_strats,
    'Seasonal': seazn_strats,
    'Bond': bond_strats,
    'AU Spread': auspd_strats,
}

# Option 1: Save to HTML file
run_dashboard(
    port_pnl=port_pnl,
    pnl_by_signal=pnl_by_signal,
    strategy_groups_mapping=strategy_groups_mapping,
    output_html='C:/dev/data/backtest_dashboard.html'
)

# Option 2: Run interactive server (opens in browser)
# run_dashboard(
#     port_pnl=port_pnl,
#     pnl_by_signal=pnl_by_signal,
#     strategy_groups_mapping=strategy_groups_mapping,
#     serve=True,
#     port=5006
# )
```

## Features:
- **Overview Tab**: Portfolio-level KPIs, equity curve, strategy summary table, monthly heatmap
- **12 Strategy Group Tabs**: Each with group metrics, equity curve, and expandable strategy panels
- **All Strategies Tab**: Sortable table of all strategies with key metrics
- **Interactive Charts**: 
  - Portfolio and asset PnL using plotly (iplot-style)
  - Equity curves with hvplot
  - Drawdown charts
  - Returns histograms
  - Rolling metrics
- **Lazy Loading**: Charts load only when expanded to save memory
- **Responsive**: Works on different screen sizes

## Dependencies:
Make sure you have installed:
```bash
pip install panel bokeh hvplot plotly pandas numpy
```

## Tips:
1. For large datasets, use `serve=True` instead of HTML export
2. HTML export may be large if you have many strategies
3. Server mode allows real-time interaction and faster loading
4. Access server at: http://localhost:5006
