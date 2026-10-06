import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CZCE_PREFIXES = (
    "AP", "CF", "CJ", "FG", "MA", "OI", "PF", "PK", "PR", "PX",
    "RM", "SA", "SF", "SH", "SM", "SR", "TA", "UR",
)


def load_literal_dict(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
            continue
        return {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(node.value.keys, node.value.values)
        }
    raise KeyError(name)


class SpotIndexMapReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production_path = ROOT / "pycmqlib3/utility/spot_idx_map.py"
        cls.ifind = load_literal_dict(cls.production_path, "index_map")
        cls.ifind_full = load_literal_dict(
            ROOT / "tests/index_map_full.py", "index_map_full"
        )
        cls.mysteel = load_literal_dict(cls.production_path, "mysteel_index_map")
        cls.mysteel_catalog = load_literal_dict(
            ROOT / "tests/mysteel_ticker_map.py", "ticker_name"
        )

    def test_ifind_production_map_matches_full_catalog(self):
        self.assertEqual(self.ifind, self.ifind_full)
        self.assertEqual(len(self.ifind), 1098)

    def test_mysteel_production_map_matches_catalog_keys(self):
        self.assertEqual(set(self.mysteel), set(self.mysteel_catalog))
        self.assertEqual(len(self.mysteel), 111)

    def test_each_source_has_unique_aliases(self):
        self.assertEqual(len(self.ifind.values()), len(set(self.ifind.values())))
        self.assertEqual(len(self.mysteel.values()), len(set(self.mysteel.values())))

    def test_providers_do_not_share_aliases(self):
        self.assertFalse(set(self.ifind.values()) & set(self.mysteel.values()))

    def test_czce_aliases_use_exchange_code_case(self):
        aliases = list(self.ifind.values()) + list(self.mysteel.values())
        invalid_prefixes = tuple(f"{code.lower()}_" for code in CZCE_PREFIXES)
        self.assertFalse(any(alias.startswith(invalid_prefixes) for alias in aliases))
        self.assertFalse(any(alias.startswith(("pta_", "PTA_")) for alias in aliases))

    def test_synthetic_duplicate_codes_are_not_loaded(self):
        self.assertFalse(any(code.endswith(".1") for code in self.ifind))

    def test_production_czce_references_use_canonical_aliases(self):
        from misc_scripts.update_fun_data import PROD_FULL_HIST_TICKERS
        from pycmqlib3.strategy.feature_config import METAL_INV_FEATURES
        from pycmqlib3.strategy.signal_repo import (
            commod_phycarry_dict,
            feature_to_feature_key_mapping,
            signal_store,
        )

        expected_inventory = {
            "SM": "SM_stockdays",
            "SF": "SF_inv_mill",
            "FG": "FG_inv_mill",
            "SA": "SA_inv_mill_all",
            "SH": "SH_inv_mill_all",
        }
        for product, alias in expected_inventory.items():
            self.assertEqual(METAL_INV_FEATURES[product], alias)
            self.assertIn(alias, PROD_FULL_HIST_TICKERS)

        expected_carry = {
            "FG": "FG_5mm_shahe",
            "SM": "SM_65s17_tj",
            "SF": "SF_72_ningxia",
            "SA": "SA_heavy_shahe",
            "MA": "MA_spot_jiangsu",
            "TA": "TA_east_spot",
            "PF": "PF_fujian_spot",
            "pp": "pp_t30s_shaoxing_hz",
            "pg": "propane_cfr_south_cny_vat",
            "fu": "fo_380cst_zhoushan_cny",
        }
        for product, alias in expected_carry.items():
            self.assertEqual(commod_phycarry_dict[product], alias)

        self.assertEqual(
            feature_to_feature_key_mapping["smsf_prodcost"],
            {"SM": "SM_neimeng_cost", "SF": "SF_neimeng_cost"},
        )
        expected_signal_fields = {
            "TA_margin_st_zs": "TA_margin_cn_d",
            "TA_arb_ma": "TA_cfr_dom_ratio",
            "TA_minv_hys": "TA_invdays_mill",
            "fgsa_margin_mom_st": "FG_margin_petcoke",
            "FG_margin_hlr_1y": "FG_margin_avg",
            "FG_util_mom_st": "FG_util_adj",
            "FG_util_mom_spd_st": "FG_util_adj",
        }
        for signal_name, field in expected_signal_fields.items():
            self.assertEqual(signal_store[signal_name][1][0], field)

    def test_new_signal_families_are_exposed(self):
        expected = {
            "pp_t30s_shaoxing_hz",
            "eg_east_spot_mid",
            "eb_jiangsu_n1",
            "propane_prem_east_cfr",
            "SF_hesteel_purchase_qty",
            "ferroalloy_power_px_neimeng",
        }
        self.assertTrue(expected.issubset(set(self.ifind.values())))

        mysteel_expected = {
            "pp_inv_mill_cn_w_ms",
            "sp_inv_ports_cn_w_ms",
            "bu_inv_social_104_cn_w_ms",
            "TA_feedstock_invdays_polyester_w_ms",
        }
        self.assertTrue(mysteel_expected.issubset(set(self.mysteel.values())))

    def test_every_production_dependency_is_annotated(self):
        from misc_scripts.data_dependency_monitor import (
            build_dependency_rows,
            collect_production_index_codes,
        )

        used = collect_production_index_codes(build_dependency_rows())
        text = self.production_path.read_text(encoding="utf-8")
        self.assertEqual(text.count("used in prod"), len(used))
        for qualified_code in used:
            _, code = qualified_code.split(":", 1)
            line = next(line for line in text.splitlines() if f"{code!r}:" in line)
            self.assertIn("used in prod", line)


if __name__ == "__main__":
    unittest.main()
