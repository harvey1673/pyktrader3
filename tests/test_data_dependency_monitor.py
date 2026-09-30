import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

if "C:/dev/wtpy" not in sys.path:
    sys.path.append("C:/dev/wtpy")
from misc_scripts import data_dependency_monitor as monitor


class SourceMappingTests(unittest.TestCase):
    def test_all_production_routes_are_included(self):
        with patch.object(monitor, "factors_by_spread3", {"new_spread": []}), patch.object(
            monitor, "factors_by_func", {"new_function": {}}
        ):
            names = monitor.get_production_signal_list()
            self.assertIn("new_spread", names)
            self.assertIn("new_function", names)
            self.assertEqual(len(names), len(set(names)))

    def test_missing_recipe_is_reported_not_dropped(self):
        with patch.object(monitor, "single_factors", {"missing_example": []}):
            rows = monitor.build_dependency_rows(["missing_example"])
        self.assertEqual(rows[0]["in_spot_df"], "missing_recipe")
        self.assertEqual(rows[0]["route"], "single_factors")
        self.assertEqual(monitor.summarize_rows(rows)["missing_recipe_count"], 1)

    def test_function_without_spot_keys_is_explicitly_marked_for_review(self):
        with patch.object(monitor, "factors_by_func", {
            "function_example": {"func": monitor.get_production_signal_list, "args": {}}
        }), patch.object(monitor, "build_function_spot_df_dependency_set", return_value=set()):
            rows = monitor.build_dependency_rows(["function_example"])
        self.assertEqual(rows[0]["in_spot_df"], "function_requires_review")
        self.assertEqual(monitor.summarize_rows(rows)["function_review_count"], 1)

    def test_overlapping_codes_and_alias_precedence(self):
        with patch.object(monitor, "index_map", {"SAME": "ifind_only", "OLD": "shared"}), patch.object(
            monitor, "mysteel_index_map", {"SAME": "steel_only", "NEW": "shared"}
        ):
            mapping = monitor.effective_source_index_map()
            self.assertEqual(mapping, {
                "ifind:SAME": "ifind_only", "mysteel:SAME": "steel_only",
                "mysteel:NEW": "shared",
            })
            rows = [{"index_codes": "ifind:SAME|mysteel:NEW",
                     "transitive_index_codes": "mysteel:SAME"}]
            self.assertEqual(monitor.collect_production_index_codes(rows), mapping)

    def test_freshness_queries_sources_separately(self):
        def load(codes, source, column_name):
            self.assertEqual(codes, ["SAME"])
            value = 10 if source == "ifind" else 20
            return pd.DataFrame({"SAME": [value, value + 1]},
                                index=pd.to_datetime(["2026-09-01", "2026-09-02"]))
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            monitor, "load_codes_from_edb", side_effect=load
        ) as loader:
            path = Path(tmp) / "freshness.csv"
            monitor.generate_data_freshness_report(
                {"ifind:SAME": "first", "mysteel:SAME": "second"}, path
            )
            result = pd.read_csv(path)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(result["source"].tolist(), ["ifind", "mysteel"])
            self.assertEqual(result["last_value"].tolist(), [11, 21])
            self.assertEqual(result["prev_value"].tolist(), [10, 20])
            monitor.generate_data_freshness_report(
                {"ifind:SAME": "first", "mysteel:SAME": "second"}, path,
                source=["mysteel"],
            )
            self.assertEqual(pd.read_csv(path)["source"].tolist(), ["mysteel"])

    def test_direct_and_transitive_mysteel_dependencies(self):
        with patch.object(monitor, "index_map", {"OLD": "shared"}), patch.object(
            monitor, "mysteel_index_map", {"NEW": "shared"}
        ), patch.object(monitor, "signal_store", {"example": []}), patch.object(
            monitor, "extract_required_keys_for_signal",
            return_value=({"shared", "derived"}, set(), "shared", False)
        ), patch.object(monitor, "build_spot_df_column_universe", return_value={"shared", "derived"}), patch.object(
            monitor, "build_process_spot_formula_dependency_map", return_value={"derived": {"shared"}}
        ), patch.object(monitor, "build_ctd_basis_formula_dependency_map", return_value={}), patch.object(
            monitor, "build_update_db_factor_formula_dependency_map", return_value={}
        ), patch.object(monitor, "get_process_spot_formula_deps", return_value=[]):
            rows = {r["required_key"]: r for r in monitor.build_dependency_rows(["example"])}
            self.assertEqual(rows["shared"]["index_codes"], "mysteel:NEW")
            self.assertEqual(rows["derived"]["transitive_index_codes"], "mysteel:NEW")


if __name__ == "__main__":
    unittest.main()
