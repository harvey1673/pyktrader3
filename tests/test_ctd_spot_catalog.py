import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("ctd_spot_catalog.py")
SPEC = importlib.util.spec_from_file_location("local_ctd_spot_catalog", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXPECTED_CODES = MODULE.EXPECTED_CODES
SPOT_CATALOG = MODULE.SPOT_CATALOG
get_spot_spec = MODULE.get_spot_spec


class TestCtdSpotCatalog(unittest.TestCase):
    def test_catalog_covers_requested_codes_once(self):
        self.assertEqual(len(EXPECTED_CODES), len(set(EXPECTED_CODES)))
        self.assertEqual(set(EXPECTED_CODES), set(SPOT_CATALOG))

    def test_each_spec_has_a_small_auditable_price_set(self):
        for code, spec in SPOT_CATALOG.items():
            with self.subTest(code=code):
                self.assertTrue(spec["preferred"])
                self.assertIn(spec["provider"], {"mysteel", "ifind"})
                self.assertTrue(spec["adjustment"])
                self.assertTrue(spec["fallback"])

    def test_lookup_returns_a_copy(self):
        spec = get_spot_spec("j")
        spec["preferred"] = "changed"
        self.assertNotEqual(spec["preferred"], SPOT_CATALOG["j"]["preferred"])

    def test_unknown_code_is_explicit(self):
        with self.assertRaisesRegex(KeyError, "No CTD spot"):
            get_spot_spec("unknown")


if __name__ == "__main__":
    unittest.main()
