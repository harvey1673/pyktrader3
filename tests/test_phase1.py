"""
Test script for Phase 1: Core Infrastructure

Tests DataLoader, StrategyExecutor, and PortfolioAggregator with simple data.
"""
import datetime
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, r'c:\dev\pyktrader3')

from backtest_runner import DataLoader, StrategyExecutor, PortfolioAggregator


def test_phase1():
    """Test Phase 1 core infrastructure."""

    print("="*70)
    print("PHASE 1 TEST: Core Infrastructure")
    print("="*70)

    # Test parameters
    start_date = datetime.date(2026, 1, 1)
    end_date = datetime.date(2026, 3, 4)
    test_assets = ['rb', 'hc', 'i']  # Test with 3 ferrous metals

    # Step 1: Test DataLoader
    print("\n" + "="*70)
    print("STEP 1: Testing DataLoader")
    print("="*70)

    try:
        loader = DataLoader(
            start_date=start_date,
            end_date=end_date,
            cache_folder=r"C:\dev\data\data_cache\\"
        )

        price_df, returns_df = loader.prepare_dataframes(
            assets=test_assets,
            roll_name='hot'
        )

        print(f"\n✓ DataLoader test passed")
        print(f"  Loaded {len(test_assets)} assets")
        print(f"  Price data: {price_df.shape}")
        print(f"  Returns data: {returns_df.shape}")

    except Exception as e:
        print(f"\n✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Test StrategyExecutor
    print("\n" + "="*70)
    print("STEP 2: Testing StrategyExecutor")
    print("="*70)

    try:
        executor = StrategyExecutor(
            price_df=price_df,
            returns_df=returns_df,
            trd_cost=2e-4
        )

        # Test with simple strategy config using real strategies from signal_store
        test_config = {
            'test_group': [
                ['ryield_ema', 0.5],          # Real strategy from signal_store
                ['mom_hlr_st', 0.3],           # Real strategy from signal_store
            ]
        }

        bt_dict, signal_dict, holding_dict, pnl_dict = executor.execute_all_strategies(
            strategy_config=test_config
        )

        print(f"\n✓ StrategyExecutor test passed")
        print(f"  Executed {len(bt_dict.get('test_group', {}))} strategies")

    except Exception as e:
        print(f"\n✗ StrategyExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Test PortfolioAggregator
    print("\n" + "="*70)
    print("STEP 3: Testing PortfolioAggregator")
    print("="*70)

    try:
        aggregator = PortfolioAggregator(returns_df=returns_df)

        results = aggregator.aggregate_all_metrics(
            bt_dict=bt_dict,
            pnl_dict=pnl_dict
        )

        print(f"\n✓ PortfolioAggregator test passed")
        print(f"  Calculated metrics: {list(results.keys())}")

        if 'portfolio_pnl' in results and len(results['portfolio_pnl']) > 0:
            pnl = results['portfolio_pnl']
            print(f"  Portfolio total PnL: {pnl.sum():.2f}")
            print(f"  Portfolio Sharpe: {pnl.mean() / pnl.std() * np.sqrt(252):.2f}")

    except Exception as e:
        print(f"\n✗ PortfolioAggregator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*70)
    print("PHASE 1 TEST COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n✓ All Phase 1 components working:")
    print("  - DataLoader: Load prices and calculate returns")
    print("  - StrategyExecutor: Execute strategies and create metrics")
    print("  - PortfolioAggregator: Aggregate portfolio-level metrics")
    print("\nReady for Phase 2: Strategy Implementation")

    return True


if __name__ == '__main__':
    success = test_phase1()
    sys.exit(0 if success else 1)
