import ast
import json
import tempfile
import unittest
from pathlib import Path


FACTOR_UPDATE = (
    Path(__file__).resolve().parents[1] / "misc_scripts" / "factor_data_update.py"
)


def load_sync_function(port_config):
    tree = ast.parse(FACTOR_UPDATE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sync_port_pos_scalers"
    )
    namespace = {"json": json, "port_pos_config": port_config}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(FACTOR_UPDATE), "exec"), namespace)
    return namespace["sync_port_pos_scalers"]


class SyncPortPosScalersTest(unittest.TestCase):
    def test_syncs_json_and_ignores_non_json_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = root / "settings"
            settings.mkdir()
            strategy = settings / "STRAT.json"
            strategy.write_text(
                '{\n  "config": {"name": "test", "pos_scaler": 1, "weight": 2}\n}\n',
                encoding="utf-8",
            )
            sync_port_pos_scalers = load_sync_function(
                {
                    "PORT": {
                        "pos_loc": str(root),
                        "strat_list": [("STRAT.json", 42000), ("manual.csv", 1)],
                    }
                }
            )

            updated = sync_port_pos_scalers()

            self.assertEqual([Path(path) for path in updated], [strategy])
            self.assertEqual(
                json.loads(strategy.read_text(encoding="utf-8"))["config"]["pos_scaler"],
                42000,
            )
            self.assertIn('"weight": 2', strategy.read_text(encoding="utf-8"))

    def test_prevalidation_prevents_partial_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = root / "settings"
            settings.mkdir()
            strategy = settings / "STRAT.json"
            original = '{"config": {"pos_scaler": 1}}'
            strategy.write_text(original, encoding="utf-8")
            sync_port_pos_scalers = load_sync_function(
                {
                    "PORT": {
                        "pos_loc": str(root),
                        "strat_list": [("STRAT.json", 2), ("MISSING.json", 3)],
                    }
                }
            )

            with self.assertRaises(FileNotFoundError):
                sync_port_pos_scalers()

            self.assertEqual(strategy.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
