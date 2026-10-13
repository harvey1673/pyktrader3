import datetime as dt
import math
import unittest

import pandas as pd

from misc_scripts.futures_roll_monitor import (
    active_products_for_position_pair,
    build_email_html,
    build_monitor,
    build_product_snapshot,
    classify_roll_status,
    normalize_hotmap_contract,
    resolve_position_pair,
)


class FuturesRollMonitorTests(unittest.TestCase):
    def setUp(self):
        dates = [dt.date(2026, 10, day) for day in (6, 7, 8, 9, 12)]
        contracts = {
            "TA2610": (dt.date(2026, 10, 20), [100, 101, 102, 103, 105], 1000, 800),
            "TA2611": (dt.date(2026, 11, 20), [110, 111, 112, 113, 115], 1200, 500),
            "TA2701": (dt.date(2027, 1, 20), [120, 121, 122, 123, 126], 1500, 600),
            "TA2705": (dt.date(2027, 5, 20), [130, 131, 132, 133, 134], 900, 700),
            "TA2609": (dt.date(2026, 9, 20), [90, 91, 92, 93, 94], 5000, 5000),
        }
        rows = []
        for contract, (expiry, closes, oi, volume) in contracts.items():
            for index, (date, close) in enumerate(zip(dates, closes)):
                rows.append(
                    {
                        "instID": contract,
                        "date": date,
                        "expiry": expiry,
                        "close": float(close),
                        "settle": float(close) + 0.5,
                        "volume": float(volume + index),
                        "openInterest": float(oi + index),
                        "diff_oi": 1.0,
                    }
                )
        self.history = pd.DataFrame(rows)

    def test_normalizes_czce_three_digit_contract_across_year_end(self):
        ref_date = dt.date(2026, 10, 12)
        self.assertEqual(normalize_hotmap_contract("CZCE", "TA611", ref_date), "TA2611")
        self.assertEqual(normalize_hotmap_contract("CZCE", "TA701", ref_date), "TA2701")
        self.assertEqual(normalize_hotmap_contract("DCE", "m2701", ref_date), "m2701")

    def test_resolves_current_and_new_positions_around_market_date(self):
        history = {
            dt.date(2026, 10, 9): {"rb": 2.0},
            dt.date(2026, 10, 12): {"rb": 3.0},
            dt.date(2026, 10, 13): {"rb": -4.0},
        }
        current, current_date, new, new_date = resolve_position_pair(
            history, dt.date(2026, 10, 12), "rb"
        )
        self.assertEqual((current, current_date), (3.0, dt.date(2026, 10, 12)))
        self.assertEqual((new, new_date), (-4.0, dt.date(2026, 10, 13)))

    def test_active_products_include_listed_zero_positions(self):
        position_history = {
            dt.date(2026, 10, 12): {"TA": 3.0, "rb": 0.0},
            dt.date(2026, 10, 13): {"TA": 0.0, "rb": -2.0, "IF": 0.0},
        }
        self.assertEqual(
            active_products_for_position_pair(
                position_history, dt.date(2026, 10, 12)
            ),
            {"TA", "rb", "IF"},
        )

    def test_monitor_only_loads_products_traded_in_position_pair(self):
        loaded_products = []

        def loader(exchange, product, start_date, end_date, data_root):
            loaded_products.append(product)
            history = self.history.copy()
            history["instID"] = history["instID"].str.replace("TA", product, regex=False)
            return history

        hotmap_rows = [
            {"exchange": "DCE", "product": product, "hotmap_contract": f"{product}2610"}
            for product in ("TA", "rb", "IF", "IH")
        ]
        detail, summary, errors = build_monitor(
            hotmap_rows,
            requested_date=dt.date(2026, 10, 12),
            loader=loader,
            position_history={
                dt.date(2026, 10, 12): {"TA": 3.0, "IF": 0.0},
                dt.date(2026, 10, 13): {"rb": -2.0},
            },
        )
        self.assertEqual(loaded_products, ["TA", "rb", "IF"])
        self.assertEqual(set(summary["product"]), {"TA", "rb", "IF"})
        self.assertEqual(set(detail["product"]), {"TA", "rb", "IF"})
        self.assertTrue(errors.empty)

    def test_selects_only_later_expiries_and_ranks_by_open_interest(self):
        detail, summary = build_product_snapshot(
            self.history,
            exchange="CZCE",
            product="TA",
            hotmap_contract="TA610",
            requested_date=dt.date(2026, 10, 12),
        )
        self.assertEqual(detail["contract"].tolist(), ["TA2610", "TA2701", "TA2611"])
        self.assertNotIn("TA2609", detail["contract"].tolist())
        self.assertEqual(summary["recommended_contract"], "TA2701")
        self.assertEqual(summary["roll_status"], "READY")

    def test_price_changes_use_same_contract_trading_observations(self):
        detail, _ = build_product_snapshot(
            self.history,
            exchange="CZCE",
            product="TA",
            hotmap_contract="TA610",
            requested_date=dt.date(2026, 10, 12),
        )
        c1 = detail.loc[detail["role"] == "c1"].iloc[0]
        self.assertEqual(c1["price_change_1d"], 2.0)
        self.assertEqual(c1["price_change_2d"], 3.0)
        self.assertEqual(c1["price_change_3d"], 4.0)
        self.assertEqual(c1["price_change_5d"], None)
        self.assertAlmostEqual(c1["price_return_1d_pct"], (105.0 / 103.0 - 1.0) * 100.0)

    def test_summary_shows_two_candidate_ratios_and_spread_changes(self):
        _, summary = build_product_snapshot(
            self.history,
            exchange="CZCE",
            product="TA",
            hotmap_contract="TA610",
            requested_date=dt.date(2026, 10, 12),
        )
        self.assertEqual(summary["candidate_1"], "TA2701")
        self.assertAlmostEqual(summary["candidate_1_oi_vs_c1"], 1504.0 / 1004.0)
        self.assertAlmostEqual(summary["candidate_1_volume_vs_c1"], 604.0 / 804.0)
        self.assertEqual(
            summary["candidate_1_c1_minus_candidate_spread_change_1d"], -1.0
        )
        self.assertEqual(
            summary["candidate_1_c1_minus_candidate_spread_change_2d"], -1.0
        )
        self.assertEqual(
            summary["candidate_1_c1_minus_candidate_spread_change_3d"], -1.0
        )
        self.assertEqual(summary["candidate_2"], "TA2611")
        self.assertAlmostEqual(summary["candidate_2_oi_vs_c1"], 1204.0 / 1004.0)
        self.assertAlmostEqual(summary["candidate_2_volume_vs_c1"], 504.0 / 804.0)
        self.assertEqual(
            summary["candidate_2_c1_minus_candidate_spread_change_1d"], 0.0
        )
        self.assertEqual(
            summary["candidate_2_c1_minus_candidate_spread_change_2d"], 0.0
        )
        self.assertEqual(
            summary["candidate_2_c1_minus_candidate_spread_change_3d"], 0.0
        )

    def test_uses_latest_c1_observation_no_later_than_requested_date(self):
        detail, summary = build_product_snapshot(
            self.history,
            exchange="CZCE",
            product="TA",
            hotmap_contract="TA610",
            requested_date=dt.date(2026, 10, 11),
        )
        self.assertEqual(summary["data_date"], dt.date(2026, 10, 9))
        self.assertTrue((detail["data_date"] == dt.date(2026, 10, 9)).all())

    def test_roll_status_thresholds_are_explicit(self):
        self.assertEqual(classify_roll_status(1.1, 0.6)[0], "READY")
        self.assertEqual(classify_roll_status(0.85, 0.2)[0], "WATCH")
        self.assertEqual(classify_roll_status(0.4, 1.1)[0], "WATCH")
        self.assertEqual(classify_roll_status(0.4, 0.2)[0], "HOLD")
        self.assertEqual(
            classify_roll_status(
                0.4,
                0.2,
                position_roll_yield_ann_pct=8.0,
                position_side="LONG",
                yield_candidate_oi_ratio=0.4,
                yield_candidate_volume_ratio=0.2,
            )[0],
            "YIELD_WATCH_LONG",
        )
        self.assertEqual(
            classify_roll_status(
                0.4,
                0.2,
                position_roll_yield_ann_pct=8.0,
                position_side="SHORT",
                yield_candidate_oi_ratio=0.4,
                yield_candidate_volume_ratio=0.2,
            )[0],
            "YIELD_WATCH_SHORT",
        )

    def test_roll_yield_is_direction_specific_and_annualized(self):
        detail, summary = build_product_snapshot(
            self.history,
            exchange="CZCE",
            product="TA",
            hotmap_contract="TA610",
            requested_date=dt.date(2026, 10, 12),
            position_history={
                dt.date(2026, 10, 12): {"TA": 5},
                dt.date(2026, 10, 13): {"TA": -5},
            },
        )
        candidate = detail.loc[detail["contract"] == "TA2701"].iloc[0]
        expected = math.log(105.0 / 126.0) * 365.0 / 92.0 * 100.0
        expected_unannualized = math.log(105.0 / 126.0) * 100.0
        self.assertAlmostEqual(candidate["long_roll_yield_pct"], expected_unannualized)
        self.assertAlmostEqual(candidate["long_roll_yield_ann_pct"], expected)
        self.assertAlmostEqual(
            candidate["short_roll_yield_ann_pct"], -expected
        )
        self.assertIn(summary["best_long_roll_contract"], {"TA2611", "TA2701"})
        self.assertIn(summary["best_short_roll_contract"], {"TA2611", "TA2701"})
        self.assertEqual(summary["current_position_lots"], 5.0)
        self.assertEqual(summary["new_position_lots"], -5.0)
        self.assertEqual(summary["position_change_lots"], -10.0)
        self.assertEqual(summary["position_lots"], -5.0)
        self.assertEqual(summary["position_side"], "SHORT")
        self.assertGreater(summary["position_roll_yield_ann_pct"], 0.0)

    def test_email_html_highlights_actionable_products(self):
        detail, summary_row = build_product_snapshot(
            self.history,
            exchange="CZCE",
            product="TA",
            hotmap_contract="TA610",
            requested_date=dt.date(2026, 10, 12),
        )
        html = build_email_html(
            pd.DataFrame([summary_row]), detail, pd.DataFrame(), dt.date(2026, 10, 12)
        )
        self.assertIn("Futures roll monitor", html)
        self.assertIn("TA2701", html)
        self.assertIn("candidate_2_oi_vs_c1", html)
        self.assertIn("READY", html)


if __name__ == "__main__":
    unittest.main()
