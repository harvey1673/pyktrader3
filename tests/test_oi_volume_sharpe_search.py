import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.oi_volume_sharpe_search import (  # noqa: E402
    build_signal_library,
    optimize_nonnegative_sharpe,
)


class OiVolumeSharpeSearchTests(unittest.TestCase):
    def test_signal_library_is_point_in_time(self):
        index = pd.bdate_range("2018-01-01", periods=620)
        products = ["rb", "cu", "m"]
        step = np.arange(len(index), dtype=float)
        close = pd.DataFrame(
            {product: 100 + (i + 1) * step * 0.03 + np.sin(step / (9 + i))
             for i, product in enumerate(products)},
            index=index,
        )
        returns = close.pct_change(fill_method=None)
        features = {
            "returns": returns,
            "aggregate_oi": pd.DataFrame(
                {product: 10_000 + (i + 2) * step for i, product in enumerate(products)},
                index=index,
            ),
            "aggregate_volume": pd.DataFrame(
                {product: 2_000 + (step % (25 + i)) * 20 for i, product in enumerate(products)},
                index=index,
            ),
        }
        condition_names = (
            "vol_regime", "vol_zscore", "vol_hlratio", "vol_regt",
            "oi_momentum", "oi_breakout", "oi_level_zscore", "oi_regt",
            "volume_activity", "volume_breakout", "volume_regt",
            "price_oi_corr", "price_volume_corr", "front_concentration",
            "c1_roll_concentration", "turnover_qtl", "price_oi_divergence",
            "price_volume_divergence",
        )
        for i, name in enumerate(condition_names):
            features[name] = pd.DataFrame(
                {product: np.sin(step / (20 + i)) for product in products},
                index=index,
            )

        full, _ = build_signal_library(close, features)
        cutoff = index[-40]
        truncated_features = {name: frame.loc[:cutoff] for name, frame in features.items()}
        truncated, _ = build_signal_library(close.loc[:cutoff], truncated_features)
        for name in (
            "price_mom_60d",
            "oi_regt_60d",
            "price_mom_60d__cross__oi_regt",
        ):
            pd.testing.assert_frame_equal(full[name].loc[:cutoff], truncated[name])

    def test_nonnegative_optimizer_respects_simplex_and_cap(self):
        index = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(7)
        pnl = pd.DataFrame(
            {
                "strong": 0.0010 + rng.normal(0, 0.01, len(index)),
                "medium": 0.0005 + rng.normal(0, 0.01, len(index)),
                "weak": rng.normal(0, 0.01, len(index)),
            },
            index=index,
        )
        weights, sharpe, success = optimize_nonnegative_sharpe(pnl, 0.60)
        self.assertTrue(success)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue((weights >= 0).all())
        self.assertLessEqual(weights.max(), 0.60 + 1e-8)
        self.assertGreater(sharpe, 0)


if __name__ == "__main__":
    unittest.main()
