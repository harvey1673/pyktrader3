import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.price_oi_volume_broad_research import (  # noqa: E402
    FAMILIES,
    VOL_FACTOR_K,
    SignalRecipe,
    efficiency_ratio,
    evaluate_broad_signal,
    volatility_position_factor,
)


class PriceOiVolumeBroadResearchTests(unittest.TestCase):
    def test_volatility_position_factor_matches_requested_curve(self):
        ratios = pd.DataFrame({"rb": [0.8, 1.0, 1.5, 2.0]})
        actual = volatility_position_factor(ratios)["rb"]
        expected = pd.Series(
            [1.0, 1.0, np.exp(-0.5), np.exp(-1.0)], name="rb"
        )
        pd.testing.assert_series_equal(actual, expected)

    def test_volatility_position_factor_supports_steeper_k(self):
        ratios = pd.DataFrame({"rb": [0.8, 1.0, 1.1, 1.2]})
        actual = volatility_position_factor(ratios, k=10.0)["rb"]
        expected = pd.Series(
            [1.0, 1.0, np.exp(-1.0), np.exp(-2.0)], name="rb"
        )
        pd.testing.assert_series_equal(actual, expected)

    def test_efficiency_ratio_is_one_for_monotonic_path(self):
        index = pd.bdate_range("2024-01-01", periods=40)
        log_price = pd.DataFrame({"rb": np.arange(40, dtype=float)}, index=index)
        actual = efficiency_ratio(log_price, 20)["rb"].dropna()
        self.assertTrue(np.allclose(actual, 1.0))

    def test_position_scale_reduces_gross_and_keeps_xs_neutral(self):
        index = pd.bdate_range("2024-01-01", periods=50)
        columns = ["rb", "cu", "m"]
        signal = pd.DataFrame(
            np.tile([1.0, -0.5, 0.2], (len(index), 1)),
            index=index,
            columns=columns,
        )
        scale = pd.DataFrame(
            np.tile([0.5, 0.8, 0.2], (len(index), 1)),
            index=index,
            columns=columns,
        )
        close = pd.DataFrame(
            np.tile([100.0, 200.0, 150.0], (len(index), 1)),
            index=index,
            columns=columns,
        )
        features = {
            "returns": close.pct_change(fill_method=None),
            "vol20": pd.DataFrame(
                np.tile([0.1, 0.2, 0.4], (len(index), 1)),
                index=index,
                columns=columns,
            ),
            "close": close,
            "execution_price": close,
        }
        result = evaluate_broad_signal(
            SignalRecipe(signal, "test", "test", "test", scale),
            features,
            "xs_demean",
            index[0],
            index[-1],
        )
        self.assertLessEqual(result["gross_exposure"].max(), 2.0 + 1e-12)
        self.assertTrue(
            np.allclose(
                (result["weights"] * features["vol20"]).sum(axis=1),
                0.0,
                atol=1e-12,
            )
        )
        self.assertFalse(
            np.allclose(result["weights"].sum(axis=1), 0.0, atol=1e-12)
        )
        pd.testing.assert_frame_equal(
            result["holdings"], result["weights"].shift(1)
        )

    def test_research_uses_only_time_series_and_xs_demean(self):
        self.assertEqual(FAMILIES, ("time_series", "xs_demean"))
        self.assertEqual(VOL_FACTOR_K, 5.0)

    def test_local_pnl_uses_contract_multiplier(self):
        index = pd.bdate_range("2024-01-01", periods=5)
        close = pd.DataFrame({"rb": np.arange(100.0, 105.0)}, index=index)
        features = {
            "returns": close.pct_change(fill_method=None),
            "vol20": pd.DataFrame({"rb": 0.2}, index=index),
            "close": close,
            "execution_price": close,
        }
        result = evaluate_broad_signal(
            SignalRecipe(
                pd.DataFrame({"rb": 1.0}, index=index),
                "test",
                "test",
                "test",
            ),
            features,
            "time_series",
            index[0],
            index[-1],
        )
        self.assertTrue(np.allclose(result["gross_pnl_cny"].dropna(), 10.0))


if __name__ == "__main__":
    unittest.main()
