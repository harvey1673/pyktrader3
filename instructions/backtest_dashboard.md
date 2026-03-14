# Backtest Dashboard Plan

## Overview
Reproduce the backtest/plotting in `bktest/bktest_prod_daily_run.ipynb` and output the chart in a lightweight, organized interactive dashboard using Panel. Currently, the notebook result is converted to HTML in C:/dev/data/html/prod_portfolio.html. The new dashboard will provide multiple tabs (The total Portfolio, grouped strategy tabs) with lazy-loaded charts.

## Strategy Groups
Based on notebook analysis (12 groups total):
1. **Premium Strategies** (`prem_strats`): Premium-based strategies
2. **Metal Strategies** (`metal_strats`): Metal commodity strategies
3. **Miscellaneous Strategies** (`misc_strats`): Other miscellaneous strategies
4. **Ferrous Strategies** (`ferrous_strats`): Ferrous metal strategies
5. **Ferrous Spread Strategies** (`ferrous_spd_strats`): Ferrous spread strategies
6. **Base Strategies** (`base_strats`): Base metal strategies
7. **Energy Strategies** (`energy_strats`): Energy commodity strategies
8. **Macro Strategies** (`macro_strats`): Macro factor strategies
9. **Macro 2 Strategies** (`macro2_strats`): Secondary macro strategies
10. **Seasonal Strategies** (`seazn_strats`): Seasonal pattern strategies
11. **Bond Strategies** (`bond_strats`): Bond-related strategies
12. **Gold/Silver Spread Strategies** (`auspd_strats`): Australia spread strategies

## Steps
- run backtest in the same way as the notebook, save the results such as signal_dict, holding_dict, pnl_dict the dict for signal, holding and pnl by asset for each signal, can save the results into a pickle file
-  calculate the full portfolio pnl, and strategy group, plot the cumulative pnl for strategy group in one chart, as well as the correlation of the strategy groups, the SR performance by tenors for the strategy groups and total portfolio in the total tab
- for each strategy group, create one tab in which it has all the cumulative pnl chart for each signal in the strategy group, 

## UI Structure

### Tab 1: Overview
- Portfolio-level Chart:
  - Portfolio cumulative PNL plot
  - asset cumulative PNL plot
  - Sharpe ratio by tenors for the portfolio and for the strategy group
  - pnl per trade and turnover by asset
  - standard deviation by tenors for strategy groups and the total portfolio

### Tab 2-13: Strategy Group Tabs (12 groups)
Each tab contains:
- cumulative PNL plot, and asset-level PNL plot for each signal
- turnover and pnl per trade for each asset
- Lazy load: Charts render only when expanded
