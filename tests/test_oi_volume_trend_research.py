import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.oi_volume_trend_research import (  # noqa: E402
    build_aggregate_panel,
    build_candidate_signals,
    build_feature_panels,
    cost_sensitivity_metrics,
    evaluate_signal,
    normalize_aggregate_panel,
)
from pycmqlib3.analytics import tstool  # noqa: E402


class OiVolumeTrendResearchTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2020-01-01", periods=620)
        self.products = ["rb", "cu", "m"]
        base = np.arange(len(self.index), dtype=float)
        close = pd.DataFrame(
            {
                "rb": 100 + base * 0.08 + np.sin(base / 9),
                "cu": 200 + base * 0.04 + np.cos(base / 11),
                "m": 150 - base * 0.02 + np.sin(base / 13),
            },
            index=self.index,
        )
        self.front = {
            "close": close,
            "c1_oi": pd.DataFrame(
                {product: 5000 + base * (i + 1) for i, product in enumerate(self.products)},
                index=self.index,
            ),
            "c2_oi": pd.DataFrame(
                {product: 3000 + base * (i + 0.5) for i, product in enumerate(self.products)},
                index=self.index,
            ),
            "c1_volume": pd.DataFrame(
                {product: 1000 + (base % (30 + i * 5)) * 10 for i, product in enumerate(self.products)},
                index=self.index,
            ),
            "c2_volume": pd.DataFrame(
                {product: 700 + (base % (25 + i * 5)) * 8 for i, product in enumerate(self.products)},
                index=self.index,
            ),
        }
        self.front["front_oi"] = self.front["c1_oi"] + self.front["c2_oi"]
        self.front["front_volume"] = self.front["c1_volume"] + self.front["c2_volume"]
        columns = {}
        for i, product in enumerate(self.products):
            columns[(product, "openInterest")] = 10000 + base * (i + 2)
            columns[(product, "volume")] = 2000 + (base % (35 + i * 3)) * 20
        self.aggregate = pd.DataFrame(columns, index=self.index)
        self.aggregate.columns = pd.MultiIndex.from_tuples(self.aggregate.columns)

    def test_aggregate_loader_preserves_product_field_grain_and_failures(self):
        def loader(product, **_kwargs):
            if product == "bad":
                raise ValueError("missing")
            return pd.DataFrame(
                {"volume": [10.0, 20.0], "openInterest": [30.0, 40.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            )

        panel, failures = build_aggregate_panel(
            ["rb", "bad"],
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            loader=loader,
        )
        self.assertEqual(panel.columns.names, ["product", "field"])
        self.assertEqual(panel.loc[pd.Timestamp("2024-01-03"), ("rb", "openInterest")], 40.0)
        self.assertEqual(failures.iloc[0]["product"], "bad")

    def test_dated_aggregate_export_schema_is_normalized(self):
        exported = pd.DataFrame(
            {
                ("rbc1", "agg_vol"): [10.0, 20.0],
                ("rbc1", "agg_oi"): [30.0, 40.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        normalized = normalize_aggregate_panel(exported)
        self.assertEqual(normalized.columns.names, ["product", "field"])
        self.assertEqual(normalized.loc["2024-01-03", ("rb", "volume")], 20.0)
        self.assertEqual(
            normalized.loc["2024-01-03", ("rb", "openInterest")], 40.0
        )

    def test_feature_transforms_match_tstool_and_do_not_use_future_rows(self):
        features = build_feature_panels(self.front, self.aggregate)
        expected = tstool.pct_score(features["vol20"]["rb"].dropna(), 252)
        actual_rank = features["vol_regime"]["rb"]
        common = expected.dropna().index[:20]
        # The 1y component is present exactly; the combined 1y/2y score equals
        # it before a 2y history becomes available.
        pd.testing.assert_series_equal(
            actual_rank.loc[common], expected.loc[common], check_names=False
        )

        cutoff = self.index[-30]
        truncated_front = {name: frame.loc[:cutoff] for name, frame in self.front.items()}
        truncated_features = build_feature_panels(
            truncated_front, self.aggregate.loc[:cutoff]
        )
        for name in ("vol_regime", "oi_momentum", "volume_activity", "front_concentration"):
            pd.testing.assert_frame_equal(
                features[name].loc[:cutoff], truncated_features[name]
            )

    def test_cross_sectional_candidates_are_daily_demeaned(self):
        features = build_feature_panels(self.front, self.aggregate)
        baseline = pd.DataFrame(
            {product: np.sin(np.arange(len(self.index)) / (7 + i)) for i, product in enumerate(self.products)},
            index=self.index,
        )
        candidates, definitions = build_candidate_signals(
            {"mom_test": baseline, "mom_test_xdemean": tstool.xs_demean(baseline)},
            features,
        )
        xs_names = definitions.loc[definitions["family"] == "cross_sectional", "signal"]
        for name in xs_names:
            row_means = candidates[name].mean(axis=1).dropna()
            self.assertTrue(np.allclose(row_means, 0.0, atol=1e-12))

    def test_backtest_lags_signal_before_return(self):
        index = pd.bdate_range("2024-01-01", periods=40)
        returns = pd.DataFrame({"rb": [0.0] * 20 + [0.1] + [0.0] * 19}, index=index)
        signal = pd.DataFrame({"rb": [0.0] * 20 + [1.0] + [0.0] * 19}, index=index)
        vol = pd.DataFrame({"rb": 0.2}, index=index)
        result = evaluate_signal(signal, returns, vol, cost_bps=0.0, family="time_series")
        # Same-day signal cannot earn the return; the next day holding is long.
        self.assertEqual(result["net_pnl"].iloc[20], 0.0)
        self.assertEqual(result["holdings"].iloc[21, 0], 1.0)

    def test_next_day_execution_price_matches_notebook_pnl_adjustment(self):
        index = pd.bdate_range("2024-01-01", periods=5)
        close = pd.DataFrame({"rb": [100.0, 100.0, 110.0, 110.0, 110.0]}, index=index)
        execution = pd.DataFrame(
            {"rb": [100.0, 100.0, 105.0, 110.0, 110.0]}, index=index
        )
        signal = pd.DataFrame({"rb": [0.0, 1.0, 1.0, 0.0, 0.0]}, index=index)
        vol = pd.DataFrame({"rb": 0.2}, index=index)
        result = evaluate_signal(
            signal,
            close.pct_change(fill_method=None),
            vol,
            cost_bps=0.0,
            family="time_series",
            close_prices=close,
            execution_prices=execution,
        )
        expected = (110.0 / 100.0 - 1.0) + (110.0 / 105.0 - 1.0)
        self.assertAlmostEqual(result["gross_asset_pnl"].iloc[2, 0], expected)
        self.assertEqual(result["holdings"].iloc[2, 0], 1.0)

    def test_cost_sensitivity_reprices_same_turnover(self):
        index = pd.bdate_range("2018-01-01", "2024-03-01")
        pnl_at_two_bps = pd.DataFrame({"signal": 0.001}, index=index)
        turnover = pd.DataFrame({"signal": 0.5}, index=index)
        metrics = cost_sensitivity_metrics(
            pnl_at_two_bps,
            turnover,
            configured_cost_bps=2.0,
            validation_start=pd.Timestamp("2020-01-01"),
            oos_start=pd.Timestamp("2023-01-01"),
            costs_bps=(0.0, 2.0),
        )
        train = metrics[metrics["split"] == "train"].set_index("cost_bps")
        expected_annual_difference = 0.5 * 2.0 / 10_000.0 * tstool.PNL_BDAYS
        self.assertAlmostEqual(
            train.loc[0.0, "annual_return"] - train.loc[2.0, "annual_return"],
            expected_annual_difference,
        )


if __name__ == "__main__":
    unittest.main()
