import unittest

import numpy as np
import pandas as pd

from tests.ctd_phycarry_adapter import (
    CTD_PHYCARRY_MAP,
    add_priority_ctd_phycarry,
    add_priority_ctd_spots,
    add_fx_converted_energy_spots,
)


class CTDPhycarryAdapterTests(unittest.TestCase):
    def price_frame(self, asset, dates, close, expiry, shift=0.0):
        return pd.DataFrame(
            {
                (f"{asset}c1", "close"): close,
                (f"{asset}c1", "shift"): shift,
                (f"{asset}c1", "expiry"): pd.Timestamp(expiry),
            },
            index=pd.to_datetime(dates),
        )

    def test_map_matches_existing_phycarry_column_contract(self):
        self.assertEqual(
            {key: CTD_PHYCARRY_MAP[key] for key in ["j", "jm", "ss", "SM", "SF"]},
            {
                "j": "j_ctd_spot",
                "jm": "jm_ctd_spot",
                "ss": "ss_ctd_spot",
                "SM": "SM_ctd_spot",
                "SF": "SF_ctd_spot",
            },
        )

    def test_spot_is_forward_filled_without_lookahead(self):
        spot = pd.DataFrame(
            {"SF_72_tj": [6000.0, 6200.0]},
            index=pd.to_datetime(["2026-01-01", "2026-01-04"]),
        )
        prices = self.price_frame(
            "SF",
            ["2026-01-02", "2026-01-05"],
            [6100.0, 6300.0],
            "2026-05-01",
        )
        result = add_priority_ctd_spots(prices, spot, products=["SF"])
        self.assertEqual(result.loc["2026-01-02", "SF_ctd_spot"], 6000.0)
        self.assertEqual(result.loc["2026-01-05", "SF_ctd_spot"], 6200.0)

    def test_normalized_sf_spot_does_not_receive_legacy_350_adder(self):
        date = pd.Timestamp("2026-01-02")
        expiry = pd.Timestamp("2026-02-01")
        spot = pd.DataFrame(
            {"SF_72_tj": [6000.0], "r007_cn": [0.0], "SF_phycarry": [99.0]},
            index=[date],
        )
        prices = self.price_frame("SF", [date], [6200.0], expiry)
        result = add_priority_ctd_phycarry(prices, spot, products=["SF"])
        expected = (np.log(6000.0) - np.log(6200.0)) / 30.0 * 365.0
        self.assertAlmostEqual(result.loc[date, "SF_phycarry"], expected)
        legacy_double_adjusted = (np.log(6350.0) - np.log(6200.0)) / 30.0 * 365.0
        self.assertNotAlmostEqual(result.loc[date, "SF_phycarry"], legacy_double_adjusted)

    def test_unrequested_existing_phycarry_is_preserved(self):
        date = pd.Timestamp("2026-01-02")
        spot = pd.DataFrame(
            {"SF_72_tj": [6000.0], "r007_cn": [0.0], "j_phycarry": [1.25]},
            index=[date],
        )
        prices = self.price_frame("SF", [date], [6200.0], "2026-02-01")
        result = add_priority_ctd_phycarry(prices, spot, products=["SF"])
        self.assertEqual(result.loc[date, "j_phycarry"], 1.25)

    def test_energy_usd_quotes_use_offshore_fx_with_onshore_fallback(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        spot = pd.DataFrame(
            {
                "fo_380cst_zhoushan": [600.0, 610.0],
                "propane_cfr_south": [500.0, 510.0],
                "usdcnh_spot": [7.0, np.nan],
                "usdcny_spot": [6.9, 6.8],
            },
            index=dates,
        )
        result = add_fx_converted_energy_spots(spot)
        self.assertEqual(result.loc[dates[0], "fo_380cst_zhoushan_cny"], 4200.0)
        self.assertEqual(result.loc[dates[1], "fo_380cst_zhoushan_cny"], 4148.0)
        self.assertAlmostEqual(
            result.loc[dates[0], "propane_cfr_south_cny_vat"], 3955.0
        )
        self.assertAlmostEqual(
            result.loc[dates[1], "propane_cfr_south_cny_vat"],
            510.0 * 6.8 * 1.13,
        )


if __name__ == "__main__":
    unittest.main()
