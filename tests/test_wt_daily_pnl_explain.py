from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - depends on the local test runtime
    pd = None

if pd is not None:
    from misc_scripts.wt_daily_pnl_explain import (
        ExplainPaths,
        build_code_pnl_detail,
        build_event_trading_date_mapper,
        build_product_pnl_detail,
        build_strategy_attribution,
        load_daily_funds_detail,
        load_rtdata_current_state,
        reconcile_product_pnl,
        resolve_default_run_date,
        resolve_latest_completed_run_date,
        send_pnl_email,
    )


@unittest.skipIf(pd is None, "pandas is required for PnL explain tests")
class DailyPnlExplainTest(unittest.TestCase):
    def make_paths(self, root: Path) -> "ExplainPaths":
        return ExplainPaths(
            port_dir=root,
            group_dir=root,
            port_file="PORT",
            output_strategy="OUTPUT",
            mode="eod",
            run_date_str="20260106",
            prev_date_str="20260105",
            port_path=root / "PORT_20260106.json",
            prev_port_path=root / "PORT_20260105.json",
            strat_path=root / "pos_by_strat_PORT_20260106.json",
            prev_strat_path=root / "pos_by_strat_PORT_20260105.json",
            positions_csv=root / "positions.csv",
            funds_csv=root / "funds.csv",
            closes_csv=root / "closes.csv",
            trades_csv=root / "trades.csv",
            rtdata_json=root / "rtdata.json",
            stradata_json=root / "stradata.json",
        )

    def test_night_session_maps_to_next_available_trading_date(self) -> None:
        mapper = build_event_trading_date_mapper(
            ["20260105", "20260106", "20260109", "20260112"]
        )
        self.assertEqual(mapper(202601060905), "20260106")
        self.assertEqual(mapper(202601092105), "20260112")
        self.assertIsNone(mapper(202601122105))
        self.assertIsNone(mapper("bad timestamp"))

    def test_default_run_date_cutoff_and_weekend_rules(self) -> None:
        self.assertEqual(
            resolve_default_run_date(datetime(2026, 9, 18, 20, 59)),
            datetime(2026, 9, 18).date(),
        )

    def test_latest_completed_date_requires_both_eod_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            positions = root / "positions.csv"
            funds = root / "funds.csv"
            pd.DataFrame(
                [{"date": 20260921}, {"date": 20260922}]
            ).to_csv(positions, index=False)
            pd.DataFrame(
                [{"date": 20260918}, {"date": 20260921}]
            ).to_csv(funds, index=False)
            self.assertEqual(
                resolve_latest_completed_run_date(
                    positions, funds, "20260922"
                ),
                "20260921",
            )
        self.assertEqual(
            resolve_default_run_date(datetime(2026, 9, 18, 21, 1)),
            datetime(2026, 9, 21).date(),
        )
        self.assertEqual(
            resolve_default_run_date(datetime(2026, 9, 19, 12, 0)),
            datetime(2026, 9, 18).date(),
        )

    def test_roll_safe_ledger_uses_closes_dynprofit_and_fees(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self.make_paths(root)
            pd.DataFrame(
                [
                    {
                        "date": 20260105,
                        "code": "DCE.j.2605",
                        "volume": 1,
                        "closeprofit": 0,
                        "dynprofit": 100,
                    },
                    {
                        "date": 20260106,
                        "code": "DCE.j.2609",
                        "volume": 1,
                        "closeprofit": 0,
                        "dynprofit": 20,
                    },
                ]
            ).to_csv(paths.positions_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "code": "DCE.j.2605",
                        "closetime": 202601052105,
                        "qty": 1,
                        "profit": 130,
                    }
                ]
            ).to_csv(paths.closes_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "code": "DCE.j.2605",
                        "time": 202601052105,
                        "price": 1000,
                        "qty": 1,
                        "fee": 3,
                    }
                ]
            ).to_csv(paths.trades_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "date": 20260105,
                        "closeprofit": 0,
                        "positionprofit": 100,
                        "dynbalance": 100,
                        "fee": 0,
                    },
                    {
                        "date": 20260106,
                        "closeprofit": 130,
                        "positionprofit": 20,
                        "dynbalance": 147,
                        "fee": 3,
                    },
                ]
            ).to_csv(paths.funds_csv, index=False)

            code_detail = build_code_pnl_detail(paths)
            product_detail = build_product_pnl_detail(code_detail)
            funds_detail = load_daily_funds_detail(paths)
            product_detail, residual = reconcile_product_pnl(
                product_detail, funds_detail["daily_equity_pnl"]
            )

            self.assertAlmostEqual(code_detail["realized_pnl"].sum(), 130.0)
            self.assertAlmostEqual(code_detail["unrealized_pnl_change"].sum(), -80.0)
            self.assertAlmostEqual(code_detail["fee"].sum(), 3.0)
            self.assertAlmostEqual(
                code_detail["night_avg_executed_price"].dropna().iloc[0], 1000.0
            )
            self.assertAlmostEqual(code_detail["night_executed_volume"].sum(), 1.0)
            self.assertAlmostEqual(code_detail["day_executed_volume"].sum(), 0.0)
            self.assertAlmostEqual(product_detail["daily_pnl"].sum(), 47.0)
            self.assertAlmostEqual(residual, 0.0)

    def test_intraday_uses_live_stradata_without_current_eod_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self.make_paths(root)
            object.__setattr__(paths, "mode", "intraday")
            pd.DataFrame(
                [
                    {
                        "date": 20260105,
                        "code": "DCE.j.2605",
                        "volume": 1,
                        "closeprofit": 0,
                        "dynprofit": 100,
                    }
                ]
            ).to_csv(paths.positions_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "date": 20260105,
                        "closeprofit": 0,
                        "positionprofit": 100,
                        "dynbalance": 100,
                        "fee": 0,
                    }
                ]
            ).to_csv(paths.funds_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "code": "DCE.j.2605",
                        "closetime": 202601052105,
                        "qty": 1,
                        "profit": 130,
                    }
                ]
            ).to_csv(paths.closes_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "code": "DCE.j.2605",
                        "time": 202601052105,
                        "price": 1000,
                        "qty": 1,
                        "fee": 3,
                    }
                ]
            ).to_csv(paths.trades_csv, index=False)
            paths.stradata_json.write_text(
                json.dumps(
                    {
                        "fund": {
                            "tdate": 20260106,
                            "total_profit": 130,
                            "total_dynprofit": 20,
                            "total_fees": 3,
                        },
                        "positions": [
                            {
                                "code": "DCE.j.2609",
                                "volume": 1,
                                "dynprofit": 20,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            code_detail = build_code_pnl_detail(paths)
            funds_detail = load_daily_funds_detail(paths)

            self.assertAlmostEqual(code_detail["daily_pnl"].sum(), 47.0)
            self.assertAlmostEqual(funds_detail["daily_equity_pnl"], 47.0)

    def test_rtdata_account_and_positions_are_loaded_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rtdata.json"
            path.write_text(
                json.dumps(
                    {
                        "funds": {
                            "CNY": {
                                "prebalance": 1000,
                                "balance": 1100,
                                "margin": 220,
                                "available": 880,
                                "closeprofit": 50,
                                "dynprofit": 60,
                                "fee": 10,
                            }
                        },
                        "positions": [
                            {
                                "code": "DCE.j.2609",
                                "long": {
                                    "newvol": 2,
                                    "newavail": 1,
                                    "prevol": 3,
                                    "preavail": 3,
                                },
                                "short": {
                                    "newvol": 1,
                                    "newavail": 1,
                                    "prevol": 0,
                                    "preavail": 0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            account, positions, metadata = load_rtdata_current_state(path)

            self.assertEqual(account.loc[0, "balance"], 1100)
            self.assertAlmostEqual(account.loc[0, "margin_to_balance"], 0.2)
            self.assertEqual(positions.loc[0, "contract"], "DCE.j.2609")
            self.assertEqual(positions.loc[0, "net_position"], 4)
            self.assertEqual(metadata["rtdata_position_count"], 1)

    @patch("pycmqlib3.utility.email_tool.send_html_by_smtp")
    def test_email_uses_configured_smtp_and_full_report_tables(self, smtp_mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self.make_paths(Path(temp_dir))
            code_detail = pd.DataFrame(
                [
                    {
                        "contract": "DCE.j.2609",
                        "product": "j",
                        "daily_pnl": 12.5,
                        "night_avg_executed_price": 1500,
                        "night_executed_volume": 2,
                    }
                ]
            )
            product_detail = pd.DataFrame(
                [{"product": "j", "daily_pnl": 12.5}]
            )

            sent = send_pnl_email(
                paths,
                {"daily_equity_pnl": 12.5, "daily_fee": 1.0},
                code_detail,
                product_detail,
                pd.DataFrame([{"strategy": "TEST", "attributed_pnl": 12.5}]),
                pd.DataFrame(),
                pd.DataFrame([{"currency": "CNY", "balance": 1000}]),
                pd.DataFrame([{"contract": "DCE.j.2609", "net_position": 2}]),
                {"rtdata_last_modified": "2026-01-06T15:00:00+08:00"},
                email_notify=True,
            )

            self.assertTrue(sent)
            smtp_mock.assert_called_once()
            subject = smtp_mock.call_args.args[2]
            html = smtp_mock.call_args.args[3]
            self.assertIn("2026.01.06", subject)
            self.assertIn("Contract PnL and Exchange Executions", html)
            self.assertIn("DCE.j.2609", html)
            self.assertIn("Current Exchange-Reported Account", html)

    def test_signed_attribution_and_unallocated_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self.make_paths(root)
            paths.prev_port_path.write_text(
                json.dumps({"X": 1, "Y": 0, "Z": 1}), encoding="utf-8"
            )
            paths.port_path.write_text(
                json.dumps({"X": 1, "Y": 0, "Z": 1}), encoding="utf-8"
            )
            previous_positions = {
                "LONG.json": {"X": 2, "Y": 1, "Z": float("nan")},
                "SHORT.json": {"X": -1, "Y": -1},
            }
            current_positions = {
                "LONG.json": {"X": 2, "Y": 1, "Z": 1},
                "SHORT.json": {"X": -1, "Y": -1},
            }
            paths.prev_strat_path.write_text(
                json.dumps(previous_positions), encoding="utf-8"
            )
            paths.strat_path.write_text(
                json.dumps(current_positions), encoding="utf-8"
            )
            product_detail = pd.DataFrame(
                [
                    {"product": "X", "daily_pnl": 100.0, "prev_volume": 1, "curr_volume": 1},
                    {"product": "Y", "daily_pnl": 50.0, "prev_volume": 0, "curr_volume": 0},
                    {"product": "Z", "daily_pnl": 25.0, "prev_volume": 1, "curr_volume": 1},
                ]
            )

            detail, summary, issues = build_strategy_attribution(paths, product_detail)

            x_rows = detail[detail["product"] == "X"].set_index("strategy_file")
            self.assertAlmostEqual(x_rows.loc["LONG.json", "attributed_pnl"], 200.0)
            self.assertAlmostEqual(x_rows.loc["SHORT.json", "attributed_pnl"], -100.0)
            self.assertEqual(
                detail.loc[detail["product"] == "Y", "strategy_file"].iloc[0],
                "UNALLOCATED_OFFSETTING",
            )
            self.assertEqual(
                detail.loc[detail["product"] == "Z", "strategy_file"].iloc[0],
                "UNALLOCATED_INVALID_INPUT",
            )
            self.assertAlmostEqual(summary["attributed_pnl"].sum(), 175.0)
            self.assertEqual(len(issues), 1)

    def test_runtime_position_mismatch_is_not_leveraged_to_force_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self.make_paths(root)
            paths.prev_port_path.write_text(json.dumps({"X": 1}), encoding="utf-8")
            paths.port_path.write_text(json.dumps({"X": 1}), encoding="utf-8")
            positions = {"ONE.json": {"X": 1}}
            paths.prev_strat_path.write_text(json.dumps(positions), encoding="utf-8")
            paths.strat_path.write_text(json.dumps(positions), encoding="utf-8")
            product_detail = pd.DataFrame(
                [
                    {
                        "product": "X",
                        "daily_pnl": 100.0,
                        "prev_volume": 10.0,
                        "curr_volume": 10.0,
                    }
                ]
            )

            detail, summary, _ = build_strategy_attribution(paths, product_detail)

            strategy_pnl = detail.loc[
                detail["strategy_file"] == "ONE.json", "attributed_pnl"
            ].iloc[0]
            unallocated_pnl = detail.loc[
                detail["strategy_file"] == "UNALLOCATED_POSITION_MISMATCH",
                "attributed_pnl",
            ].iloc[0]
            self.assertAlmostEqual(strategy_pnl, 10.0)
            self.assertAlmostEqual(unallocated_pnl, 90.0)
            self.assertAlmostEqual(summary["attributed_pnl"].sum(), 100.0)


if __name__ == "__main__":
    unittest.main()
