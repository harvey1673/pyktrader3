import unittest

import pandas as pd

from ctd_adjustments import (
    CTDCandidate,
    MA_ctd_basis,
    SF_ctd_basis,
    SM_ctd_basis,
    UR_ctd_basis,
    adjusted_candidates,
    ctd_basis,
    j_ctd_basis,
    jm_ctd_basis,
    l_ctd_basis,
    nr_ctd_basis,
    sc_ctd_basis,
    ss_ctd_basis,
)
from ctd_spot_requirements import CTD_SPOT_REQUIREMENTS, requirements_for


class CTDAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.to_datetime(["2026-01-02"])

    def expiry(self, value):
        return pd.Series(pd.Timestamp(value), index=self.index)

    def spot(self, **columns):
        return pd.DataFrame(columns, index=self.index)

    def test_j_quality_location_and_moisture(self):
        candidate = CTDCandidate(
            "rizhao_sub_a",
            "j_spot",
            location="rizhao",
            spec={"ash": 13.0, "sulfur": 0.70, "m40": 80, "m10": 7.5,
                  "cri": 28, "csr": 62, "moisture": 7, "equilibrium_moisture": 0.8},
        )
        ctd, details = j_ctd_basis(
            self.spot(j_spot=[2000.0]), self.expiry("2026-04-01"), [candidate], True
        )
        self.assertAlmostEqual(ctd.iloc[0], 2000 / 0.93)
        self.assertTrue(details[("rizhao_sub_a", "eligible")].iloc[0])

    def test_j003_equilibrium_moisture_penalty_increases_equivalent(self):
        candidate = CTDCandidate(
            "wet_quench", "j_spot", location="rizhao",
            spec={"ash": 13, "sulfur": .70, "m40": 80, "m10": 7.5,
                  "cri": 28, "csr": 62, "moisture": 0, "equilibrium_moisture": 1.2},
        )
        old = j_ctd_basis(self.spot(j_spot=[2000]), self.expiry("2026-03-01"), [candidate])
        new = j_ctd_basis(self.spot(j_spot=[2000]), self.expiry("2026-04-01"), [candidate])
        self.assertEqual(new.iloc[0] - old.iloc[0], 110)

    def test_jm003_quality_location_and_brand_overlay(self):
        candidate = CTDCandidate(
            "mongol_5", "jm_spot", location="jingtang", brand="shanjiao_rizhao_no_1",
            spec={"ash": 10, "sulfur": 1.0, "volatile": 24, "g": 80,
                  "y": 14, "csr": 62, "moisture": 8},
        )
        ctd, details = jm_ctd_basis(
            self.spot(jm_spot=[1800]), self.expiry("2026-01-01"), [candidate], True
        )
        # +30 ash, +75 sulfur, +170 port, +80 brand are all exchange premiums.
        self.assertEqual(ctd.iloc[0], 1800 - 30 - 75 - 170 - 80)
        self.assertEqual(details[("mongol_5", "brand_adj")].iloc[0], 80)

    def test_ss_thickness_and_edge_schedule(self):
        candidate = CTDCandidate(
            "registered_07_mill", "ss_spot", brand="example_registered",
            spec={"registered": True, "grade": "304", "surface": "2B",
                  "thickness_mm": 0.7, "width_mm": 1219, "edge": "mill"},
        )
        index = pd.to_datetime(["2026-07-17", "2026-07-21"])
        spot = pd.DataFrame({"ss_spot": [14000, 14000]}, index=index)
        expiry = pd.Series(pd.Timestamp("2026-09-01"), index=index)
        result = ss_ctd_basis(spot, expiry, [candidate])
        self.assertEqual(result.iloc[0], 14000 - (400 - 170))
        self.assertEqual(result.iloc[1], 14000 - (300 - 170))

    def test_sm_and_sf_future_location_boundaries(self):
        sm = CTDCandidate("ulanqab", "sm", location="ulanqab", spec={"grade": "6517"})
        sf = CTDCandidate("zhongwei", "sf", location="zhongwei", spec={"grade": "72"})
        sm_old = SM_ctd_basis(self.spot(sm=[6000]), self.expiry("2027-10-01"), [sm])
        sm_new = SM_ctd_basis(self.spot(sm=[6000]), self.expiry("2027-11-01"), [sm])
        sf_old = SF_ctd_basis(self.spot(sf=[6000]), self.expiry("2027-06-01"), [sf])
        sf_new = SF_ctd_basis(self.spot(sf=[6000]), self.expiry("2027-07-01"), [sf])
        self.assertEqual(sm_old.iloc[0], 6400)
        self.assertEqual(sm_new.iloc[0], 6290)
        self.assertEqual(sf_old.iloc[0], 6350)
        self.assertEqual(sf_new.iloc[0], 6280)

    def test_l_qualified_substitute_and_ctd_minimum(self):
        standard = CTDCandidate("standard", "l_std", spec={"registered": True, "quality": "standard"})
        substitute = CTDCandidate("qualified", "l_sub", spec={"registered": True, "quality": "qualified"})
        ctd = l_ctd_basis(
            self.spot(l_std=[8000], l_sub=[7970]), self.expiry("2026-09-01"),
            [standard, substitute],
        )
        # Qualified material receives -20, so its futures-equivalent is 7990.
        self.assertEqual(ctd.iloc[0], 7990)

    def test_ur_and_nr_quality_alternatives(self):
        ur = CTDCandidate("large", "ur", location="liaoning", spec={"quality": "large_granule"})
        nr = CTDCandidate("tsr10", "nr", brand="registered", spec={"registered": True, "quality": "tsr10"})
        self.assertEqual(UR_ctd_basis(self.spot(ur=[2000]), self.expiry("2027-03-01"), [ur]).iloc[0], 1980)
        self.assertEqual(nr_ctd_basis(self.spot(nr=[12000]), self.expiry("2026-09-01"), [nr]).iloc[0], 12400)

    def test_dynamic_products_accept_explicit_adjustments_and_costs(self):
        crude = CTDCandidate(
            "oman", "oman_usd_converted", spec={"quality": "standard"},
            cash_cost=85, quality_adj_override=20, location_adj_override=-10,
        )
        ctd, details = sc_ctd_basis(
            self.spot(oman_usd_converted=[4000]), self.expiry("2026-09-01"), [crude], True
        )
        self.assertEqual(ctd.iloc[0], 4000 + 85 - 20 - (-10))
        self.assertEqual(details[("oman", "cash_cost")].iloc[0], 85)

    def test_missing_or_ineligible_candidates_do_not_win(self):
        good = CTDCandidate("good", "good", spec={"quality": "standard"})
        bad = CTDCandidate("bad", "bad", spec={"quality": "standard"}, eligible=False)
        details = adjusted_candidates("MA", self.spot(good=[2500], bad=[1]), self.expiry("2026-09-01"), [good, bad])
        self.assertTrue(pd.isna(details[("bad", "equivalent")].iloc[0]))
        self.assertEqual(MA_ctd_basis(self.spot(good=[2500], bad=[1]), self.expiry("2026-09-01"), [good, bad]).iloc[0], 2500)

    def test_recommended_spot_inventory_covers_requested_products(self):
        requested = {"j", "jm", "ss", "sm", "sf", "l", "pp", "v", "eg", "eb",
                     "ta", "px", "ma", "sc", "fu", "lu", "bu", "ur", "ru", "nr", "br"}
        covered = set()
        for row in CTD_SPOT_REQUIREMENTS:
            covered.update(row["product"].lower().split("/"))
        self.assertTrue(requested.issubset(covered))
        self.assertTrue(any("Mongolian No. 5" in row["candidate"] for row in requirements_for("jm")))

    def test_all_second_wave_products_have_an_executable_path(self):
        products = ["l", "pp", "v", "eg", "eb", "TA", "PX", "MA", "sc",
                    "fu", "lu", "bu", "UR", "ru", "nr", "br"]
        for product in products:
            spec = {"quality": "standard", "registered": True}
            if product == "ru":
                spec["grade"] = "scr_wf"
            if product == "nr":
                spec["quality"] = "tsr20"
            candidate = CTDCandidate(product, "spot", location="jiangsu", spec=spec)
            result = ctd_basis(product, self.spot(spot=[100]), self.expiry("2026-09-01"), [candidate])
            self.assertEqual(result.iloc[0], 100, product)


if __name__ == "__main__":
    unittest.main()
