import ast
import argparse
import json
import tempfile
import unittest
from pathlib import Path


FACTOR_UPDATE = (
    Path(__file__).resolve().parents[1] / "misc_scripts" / "factor_data_update.py"
)
PORT_POSITION_UPDATE = (
    Path(__file__).resolve().parents[1] / "misc_scripts" / "port_position_update.py"
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


def load_cli_parser(path, email_notify):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_cli_args"
    )
    namespace = {"argparse": argparse, "EMAIL_NOTIFY": email_notify}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["parse_cli_args"]


class SyncPortPosScalersTest(unittest.TestCase):
    def test_factor_cli_email_flags_override_shared_default(self):
        parse_args = load_cli_parser(FACTOR_UPDATE, email_notify=False)
        self.assertTrue(parse_args(["20261013", "--email"]).email_notify)
        self.assertFalse(parse_args(["20261013", "--no-email"]).email_notify)
        self.assertFalse(parse_args(["20261013"]).email_notify)
        self.assertEqual(parse_args(["--sync-scalers", "PORT"]).sync_scalers, "PORT")

    def test_port_position_cli_email_flags_override_shared_default(self):
        parse_args = load_cli_parser(PORT_POSITION_UPDATE, email_notify=True)
        self.assertTrue(parse_args(["20261013"]).email_notify)
        self.assertTrue(parse_args(["20261013", "--email"]).email_notify)
        self.assertFalse(parse_args(["20261013", "--no-email"]).email_notify)

    def test_factor_position_email_uses_run_date_and_email_notify(self):
        tree = ast.parse(FACTOR_UPDATE.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_port_position"
        )
        self.assertIsInstance(function.args.defaults[-1], ast.Name)
        self.assertEqual(function.args.defaults[-1].id, "EMAIL_NOTIFY")
        email_block = next(
            node
            for node in function.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "email_notify"
        )
        names = {node.id for node in ast.walk(email_block) if isinstance(node, ast.Name)}
        calls = {
            node.func.id
            for node in ast.walk(email_block)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("run_date", names)
        self.assertNotIn("tday", names)
        self.assertIn("send_html_by_smtp", calls)

    def test_port_position_wrapper_keeps_api_and_forwards_email_flag(self):
        tree = ast.parse(PORT_POSITION_UPDATE.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_port_pos"
        )
        factor_call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "update_port_position"
        )
        keywords = {keyword.arg: keyword.value for keyword in factor_call.keywords}
        self.assertIsInstance(keywords["email_notify"], ast.Name)
        self.assertEqual(keywords["email_notify"].id, "email_notify")

    def test_position_update_syncs_json_scalers_first(self):
        tree = ast.parse(FACTOR_UPDATE.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_port_position"
        )

        first_statement = function.body[0]
        self.assertIsInstance(first_statement, ast.Assign)
        self.assertIsInstance(first_statement.value, ast.Call)
        self.assertEqual(first_statement.value.func.id, "sync_port_pos_scalers")

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

    def test_can_sync_only_one_selected_strategy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = root / "settings"
            settings.mkdir()
            strategy = settings / "STRAT.json"
            strategy.write_text(
                '{"config": {"pos_scaler": 1}}',
                encoding="utf-8",
            )
            sync_port_pos_scalers = load_sync_function(
                {
                    "PORT": {
                        "pos_loc": str(root),
                        "strat_list": [("STRAT.json", 2), ("MISSING.json", 3)],
                    }
                }
            )

            updated = sync_port_pos_scalers(strategy_file="STRAT.json")

            self.assertEqual([Path(path) for path in updated], [strategy])
            self.assertEqual(
                json.loads(strategy.read_text(encoding="utf-8"))["config"]["pos_scaler"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
