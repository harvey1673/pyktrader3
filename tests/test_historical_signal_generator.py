import unittest

import numpy as np
import pandas as pd

from misc_scripts.historical_signal_generator import HistoricalSignalGenerator
from misc_scripts.strategy_scenario_backtest import SignalSpec, _execution_signal_name


def _recipe(post_func=""):
    parameters = [None] * 10
    parameters[7] = post_func
    return [None, parameters]


class HistoricalSignalGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2022-01-03", periods=320, freq="B")
        columns = {}
        for asset, offset in (("cu", 0.0), ("al", 20.0)):
            columns[(f"{asset}c1", "close")] = np.arange(100.0, 420.0) + offset
            columns[(f"{asset}d1", "close")] = np.arange(110.0, 430.0) + offset
        self.prices = pd.DataFrame(columns, index=self.dates)
        self.prices.columns = pd.MultiIndex.from_tuples(self.prices.columns)
        self.spot = pd.DataFrame({"feature": np.arange(320.0)}, index=self.dates)

    def _generator(self, assets=None, **routes):
        assets = ["cu", "al"] if assets is None else list(assets)
        names = {
            name
            for registry in routes.values()
            for name in registry
        }
        store = {name: _recipe("raw|buf0.2") for name in names}

        def get_signal(spot_df, signal_name, **kwargs):
            asset = kwargs.get("asset")
            scale = 2.0 if asset == "al" else 1.0
            return pd.Series(scale, index=spot_df.index, name=signal_name)

        defaults = {
            "factors_by_asset": {},
            "single_factors": {},
            "factors_by_spread": {},
            "factors_by_spread2": {},
            "factors_by_beta_neutral": {},
            "factors_by_func": {},
        }
        defaults.update(routes)
        prices = self.prices.copy()
        for offset, asset in enumerate(assets, start=1):
            if (f"{asset}c1", "close") not in prices.columns:
                prices[(f"{asset}c1", "close")] = (
                    np.arange(100.0, 420.0) + offset * 10.0
                )
            if (f"{asset}d1", "close") not in prices.columns:
                prices[(f"{asset}d1", "close")] = (
                    np.arange(110.0, 430.0) + offset * 10.0
                )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns)
        return HistoricalSignalGenerator(
            prices,
            self.spot,
            assets,
            signal_store=store,
            get_signal=get_signal,
            spread_config={"cu_al": [[("cu", 1.0), ("al", -1.0)], 5, "hot", 1]},
            execution_name=_execution_signal_name,
            vol_window=20,
            **defaults,
        )

    def test_routes_asset_single_spread_and_function_signals(self):
        generator = self._generator(
            factors_by_asset={"asset_sig": ["cu", "al"]},
            single_factors={"single_sig": ["cu", "al"]},
            factors_by_spread={"spread_sig": [("cu", 1.0), ("al", -1.0)]},
            factors_by_func={
                "func_sig": {
                    "func": lambda price, spot: pd.DataFrame(
                        {"cu": 3.0, "al": -3.0}, index=spot.index
                    ),
                    "args": {},
                }
            },
        )
        specs = [
            SignalSpec(name, name, "ts", 1.0)
            for name in ("asset_sig", "single_sig", "spread_sig", "func_sig")
        ]

        bundle = generator.generate(specs)

        self.assertEqual(bundle.routes["asset_sig"], "factors_by_asset")
        self.assertEqual(bundle.factor_values["asset_sig"].iloc[0].tolist(), [1.0, 2.0])
        self.assertEqual(bundle.factor_values["single_sig"].iloc[0].tolist(), [1.0, 1.0])
        self.assertEqual(bundle.factor_values["spread_sig"].iloc[0].tolist(), [1.0, -1.0])
        self.assertEqual(bundle.factor_values["func_sig"].iloc[0].tolist(), [3.0, -3.0])
        self.assertEqual(bundle.post_funcs["asset_sig"], "raw|buf0.2")

    def test_function_signal_does_not_require_signal_store_recipe(self):
        generator = self._generator(
            factors_by_func={
                "func_only": {
                    "func": lambda price, spot: pd.DataFrame(
                        {"cu": 2.0}, index=spot.index
                    ),
                    "args": {},
                }
            }
        )
        generator.signal_store.clear()

        bundle = generator.generate(
            [SignalSpec("factor", "func_only", "pos", 1.0)]
        )

        self.assertEqual(bundle.routes["factor"], "factors_by_func")
        self.assertEqual(bundle.factor_values["factor"].iloc[0, 0], 2.0)
        self.assertNotIn("func_only", bundle.post_funcs)

    def test_spread2_generates_price_difference_backtest_metadata(self):
        generator = self._generator(factors_by_spread2={"spread2_sig": ["cu_al"]})

        bundle = generator.generate(
            [SignalSpec("factor", "spread2_sig", "ts", 1.0)]
        )

        self.assertEqual(bundle.pnl_modes["factor"], "px")
        self.assertEqual(
            bundle.execution_overrides["spread2_sig"], {"win": "close", "lag": 1}
        )
        self.assertEqual(list(bundle.traded_price_overrides["factor"]), ["cu", "al"])
        self.assertEqual(list(bundle.volatility_overrides["factor"]), ["cu", "al"])

    def test_beta_neutral_route_generates_both_legs(self):
        generator = self._generator(
            factors_by_beta_neutral={"beta_sig": [("cu", "al", 1.0)]}
        )

        bundle = generator.generate([SignalSpec("factor", "beta_sig", "ts", 1.0)])

        frame = bundle.factor_values["factor"]
        self.assertEqual(list(frame.columns), ["cu", "al"])
        self.assertTrue(frame.iloc[-1].notna().all())

    def test_beta_neutral_route_keeps_only_complete_strategy_pairs(self):
        generator = self._generator(
            factors_by_beta_neutral={
                "beta_sig": [("cu", "al", 1.0), ("zn", "pb", 1.0)]
            }
        )

        bundle = generator.generate([SignalSpec("factor", "beta_sig", "ts", 1.0)])

        self.assertEqual(list(bundle.factor_values["factor"].columns), ["cu", "al"])

    def test_route_without_allowable_strategy_underliers_fails(self):
        generator = self._generator(
            factors_by_beta_neutral={"beta_sig": [("zn", "pb", 1.0)]}
        )

        with self.assertRaisesRegex(ValueError, "no allowable underlying products"):
            generator.generate([SignalSpec("factor", "beta_sig", "ts", 1.0)])

    def test_incomplete_spread_is_not_backtested_as_one_leg(self):
        generator = self._generator(
            factors_by_spread={"spread_sig": [("cu", 1.0), ("zn", -1.0)]}
        )

        with self.assertRaisesRegex(ValueError, "no allowable underlying products"):
            generator.generate([SignalSpec("factor", "spread_sig", "ts", 1.0)])

    def test_function_route_is_restricted_to_strategy_underliers(self):
        generator = self._generator(
            factors_by_func={
                "func_sig": {
                    "func": lambda price, spot: pd.DataFrame(
                        {"cu": 1.0, "al": 2.0, "zn": 3.0}, index=spot.index
                    ),
                    "args": {},
                }
            }
        )

        bundle = generator.generate([SignalSpec("factor", "func_sig", "pos", 1.0)])

        self.assertEqual(list(bundle.factor_values["factor"].columns), ["cu", "al"])

    def test_shared_coal_signals_follow_each_strategy_universe(self):
        routes = {
            "single_factors": {"coal_mom_st_hlr": ["SF", "j", "jm"]},
            "factors_by_beta_neutral": {
                "coal_mom_spd_st": [
                    ("SF", "SM", 1.0),
                    ("jm", "i", 1.0),
                    ("j", "i", 1.0),
                ]
            },
        }
        specs = [
            SignalSpec("level", "coal_mom_st_hlr", "pos", 1.0),
            SignalSpec("spread", "coal_mom_spd_st", "pos", 1.0),
        ]

        funfer = self._generator(assets=["rb", "hc", "i", "j", "jm"], **routes)
        smsfspd = self._generator(assets=["SM", "SF"], **routes)

        funfer_bundle = funfer.generate(specs)
        smsfspd_bundle = smsfspd.generate(specs)
        self.assertEqual(
            list(funfer_bundle.factor_values["level"].columns), ["j", "jm"]
        )
        self.assertEqual(
            list(funfer_bundle.factor_values["spread"].columns), ["i", "j", "jm"]
        )
        self.assertEqual(
            list(smsfspd_bundle.factor_values["level"].columns), ["SF"]
        )
        self.assertEqual(
            list(smsfspd_bundle.factor_values["spread"].columns), ["SM", "SF"]
        )

    def test_missing_route_or_recipe_fails_preflight(self):
        generator = self._generator(factors_by_asset={"known": ["cu"]})
        with self.assertRaisesRegex(KeyError, "not configured"):
            generator.generate([SignalSpec("factor", "unknown", "ts", 1.0)])

        generator.signal_store.clear()
        with self.assertRaisesRegex(KeyError, "no recipe in signal_store"):
            generator.generate([SignalSpec("factor", "known", "ts", 1.0)])


if __name__ == "__main__":
    unittest.main()
