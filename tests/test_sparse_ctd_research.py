import unittest

import numpy as np
import pandas as pd

from tests.ctd_adjustments import (
    CTDCandidate,
    SF_ctd_basis,
    SM_ctd_basis,
    j_ctd_basis,
    jm_ctd_basis,
    ss_ctd_basis,
)


class SparseCTDResearchTests(unittest.TestCase):
    def frame(self, **columns):
        return pd.DataFrame(columns, index=pd.to_datetime(["2026-01-02"]))

    def expiry(self, value):
        return pd.Series(pd.Timestamp(value), index=pd.to_datetime(["2026-01-02"]))

    def test_j_default_uses_rizhao_then_tianjin_as_fallback(self):
        spot = self.frame(coke_sub_a_rz=[2000.0], coke_sub_a_tj=[1.0])
        result = j_ctd_basis(spot, self.expiry("2025-09-01"))
        self.assertAlmostEqual(result.iloc[0], 2000.0 / .93)

        spot["coke_sub_a_rz"] = np.nan
        fallback = j_ctd_basis(spot, self.expiry("2025-09-01"))
        self.assertAlmostEqual(fallback.iloc[0], 1.0 / .93 - 15.0)

    def test_j_legacy_proxy_is_available_for_2016_history(self):
        spot = self.frame(coke_sub_a_tj=[1960.0])
        result, details = j_ctd_basis(spot, self.expiry("2016-05-01"), return_details=True)
        self.assertAlmostEqual(result.iloc[0], 2000.0)
        self.assertTrue(details[("tianjin_early_history", "eligible")].iloc[0])

    def test_j2604_missing_mf_is_penalized(self):
        spot = self.frame(coke_sub_a_rz=[2000.0])
        before = j_ctd_basis(spot, self.expiry("2026-03-01"))
        after = j_ctd_basis(spot, self.expiry("2026-04-01"))
        self.assertEqual(after.iloc[0] - before.iloc[0], 110.0)

    def test_jm_default_does_not_use_unconverted_border_quote(self):
        spot = self.frame(ckc_a10v24s08_lvliang=[1800.0], ckc_stock_ganqimaodu=[1.0])
        result = jm_ctd_basis(spot, self.expiry("2026-09-01"))
        self.assertGreater(result.iloc[0], 1000.0)

    def test_jm2304_uses_dry_matter_moisture_conversion(self):
        candidate = CTDCandidate(
            "wet", "spot", location="shanxi",
            spec={"ash": 10.5, "sulfur": 1.3, "volatile": 24, "g": 75,
                  "y": 14, "csr": 62, "moisture": 10},
        )
        result = jm_ctd_basis(self.frame(spot=[1800.0]), self.expiry("2023-04-01"), [candidate])
        self.assertAlmostEqual(result.iloc[0], 1800.0 * .92 / .90)

    def test_ss_schedule_uses_observation_date_not_expiry_month(self):
        index = pd.to_datetime(["2026-07-17", "2026-07-21"])
        spot = pd.DataFrame({"spot": [14000.0, 14000.0]}, index=index)
        expiry = pd.Series(pd.Timestamp("2026-09-01"), index=index)
        candidate = CTDCandidate(
            "registered", "spot",
            spec={"registered": True, "grade": "304", "surface": "2B",
                  "thickness_mm": .7, "width_mm": 1219, "edge": "mill"},
        )
        result = ss_ctd_basis(spot, expiry, [candidate])
        self.assertEqual(result.iloc[1] - result.iloc[0], 100.0)

    def test_sm_tianjin_contract_boundaries(self):
        spot = self.frame(sm_65s17_tj=[6000.0])
        self.assertEqual(SM_ctd_basis(spot, self.expiry("2019-10-01")).iloc[0], 6000.0)
        self.assertEqual(SM_ctd_basis(spot, self.expiry("2019-11-01")).iloc[0], 6150.0)
        self.assertEqual(SM_ctd_basis(spot, self.expiry("2024-11-01")).iloc[0], 6190.0)

    def test_sf_national_proxy_has_no_automatic_zhongwei_adder(self):
        spot = self.frame(sf_72_shmet=[6000.0])
        self.assertEqual(SF_ctd_basis(spot, self.expiry("2026-09-01")).iloc[0], 6000.0)


if __name__ == "__main__":
    unittest.main()
