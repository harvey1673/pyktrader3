"""
Test script for backtest dashboard with sample data.
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'C:/dev/pyktrader3')

from tools.backtest_dashboard import BacktestDashboard, load_from_notebook_vars

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

# Sample strategies
strategies = {
    'prem_ryield': np.random.randn(len(dates)) * 100 + 50,
    'prem_basmom': np.random.randn(len(dates)) * 80 + 30,
    'metal_cu': np.random.randn(len(dates)) * 120 + 40,
    'ferrous_rb': np.random.randn(len(dates)) * 90 + 35,
}

# Create DataFrames
pnl_by_signal = pd.DataFrame(strategies, index=dates)
port_pnl = pnl_by_signal.copy()

# Strategy groups
strategy_groups_mapping = {
    'Premium': ['prem_ryield', 'prem_basmom'],
    'Metal': ['metal_cu'],
    'Ferrous': ['ferrous_rb'],
}

print("Creating test dashboard...")
print(f"Data shape: {pnl_by_signal.shape}")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Strategies: {list(strategies.keys())}")

# Load data
data = load_from_notebook_vars(
    port_pnl=port_pnl,
    pnl_by_signal=pnl_by_signal,
    strategy_groups=strategy_groups_mapping
)

# Create dashboard
dashboard = BacktestDashboard(data, strategy_groups_mapping)

# Build and save
print("\nBuilding dashboard...")
dashboard.build()
print("✓ Dashboard built successfully")

# Save to HTML
output_path = 'C:/dev/data/test_dashboard.html'
print(f"\nSaving to {output_path}...")
dashboard.save_html(output_path, title="Test Backtest Dashboard")
print("✓ Test completed successfully!")
