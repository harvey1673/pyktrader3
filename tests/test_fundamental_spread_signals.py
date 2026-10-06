import unittest

import pandas as pd

from tests.fundamental_spread_signals import SPECS, build_fundamental_spreads


class FundamentalSpreadSignalTests(unittest.TestCase):
    def test_builds_available_formulas_and_reports_missing_ones(self):
        frame = pd.DataFrame({
            "rebar_sh": [4000.0, 4100.0],
            "hrc_sh": [3900.0, 4050.0],
            "SH_50_spot_sdjl_shandong": [1500.0, 1550.0],
            "SH_32_spot_sdjl_shandong": [900.0, 920.0],
        }, index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        spreads, audit = build_fundamental_spreads(frame)
        self.assertEqual(spreads["rb_hc_spot_spread"].tolist(), [100.0, 50.0])
        self.assertAlmostEqual(spreads["SH_50_32_dry_basis"].iloc[0], 187.5)
        status = audit.set_index("signal")["status"]
        self.assertEqual(status["rb_hc_spot_spread"], "available")
        self.assertEqual(status["l_7042_tj_sh_location"], "missing_inputs")

    def test_registry_names_are_unique_and_have_explicit_inputs(self):
        self.assertEqual(len(SPECS), len(set(SPECS)))
        self.assertTrue(all(spec.legs for spec in SPECS.values()))


if __name__ == "__main__":
    unittest.main()
