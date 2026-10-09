import json
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from misc_scripts.portfolio_weight_backtest import (
    _iplot_figure,
    _split_strategy_cumulative_pnl,
    aggregate_strategy_results,
    run_full_portfolio_backtest,
    workbook_new_weight_strategies,
    write_full_portfolio_html,
)
from misc_scripts.strategy_scenario_backtest import SignalBacktestResult


class _Provider:
    def __init__(self, dates):
        self.dates = dates
        self.signal_source = "unit-test"

    def backtest(self, _spec):
        return SignalBacktestResult(
            holdings=pd.DataFrame({"cu": 1.0}, index=self.dates),
            gross_asset_pnl=pd.DataFrame({"cu": 1.0}, index=self.dates),
            cost_rates=pd.Series({"cu": 0.0}),
        )


def _write_workbook(path, rows):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "signal_weights"
    worksheet.append(
        [
            "strategy",
            "factor_name",
            "signal_name",
            "type",
            "curr_weight",
            "new_weight",
        ]
    )
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()


def _write_strategy(path, scaler):
    path.write_text(
        json.dumps(
            {
                "config": {
                    "pos_scaler": scaler,
                    "factor_repo": {
                        "factor": {
                            "name": "signal",
                            "type": "asset",
                            "weight": 99,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


class PortfolioWeightBacktestTests(unittest.TestCase):
    def test_strategy_chart_uses_nearest_point_hover(self):
        class FakeScatter(dict):
            def __init__(self, **kwargs):
                super().__init__(kwargs)

        class FakeFigure:
            def __init__(self):
                self.data = []
                self.layout = {}

            def add_trace(self, trace):
                self.data.append(trace)

            def add_annotation(self, **kwargs):
                self.layout.setdefault("annotations", []).append(kwargs)

            def update_layout(self, **kwargs):
                self.layout.update(kwargs)

        fake_go = types.SimpleNamespace(Figure=FakeFigure, Scatter=FakeScatter)
        frame = pd.DataFrame(
            {"strategy_a": [1.0, 2.0]},
            index=pd.date_range("2026-01-01", periods=2),
        )

        figure = _iplot_figure(
            frame,
            "Strategy PNL",
            fake_go,
            hovermode="closest",
        )

        self.assertEqual(figure.layout["hovermode"], "closest")
        self.assertIn("%{x|%Y-%m-%d}", figure.data[0]["hovertemplate"])
        self.assertIn("%{y:,.0f}", figure.data[0]["hovertemplate"])

    def test_strategy_curves_split_by_ending_cumulative_pnl(self):
        cumulative = pd.DataFrame(
            {
                "high": [10_000_000.0, 31_000_000.0],
                "equal": [15_000_000.0, 30_000_000.0],
                "low": [5_000_000.0, 20_000_000.0],
            },
            index=pd.date_range("2026-01-01", periods=2),
        )

        above, below = _split_strategy_cumulative_pnl(
            cumulative,
            ["high", "equal", "low"],
        )

        self.assertEqual(list(above.columns), ["high"])
        self.assertEqual(list(below.columns), ["equal", "low"])

    def test_workbook_strategy_discovery_uses_nonzero_new_weight(self):
        with tempfile.TemporaryDirectory() as temp:
            workbook = Path(temp) / "weights.xlsx"
            _write_workbook(
                workbook,
                [
                    ("B.json", "b", "sig_b", "asset", 1, 0),
                    ("A.json", "a", "sig_a", "asset", 0, 2),
                    ("a.json", "a2", "sig_a2", "asset", 0, -1),
                ],
            )
            self.assertEqual(workbook_new_weight_strategies(workbook), ("A.json",))

    def test_portfolio_signal_keys_use_strategy_and_factor_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = root / "settings"
            settings.mkdir()
            (settings / "A.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "pos_scaler": 1,
                            "factor_repo": {
                                "shared.ts": {
                                    "name": "shared",
                                    "type": "ts",
                                    "weight": 1,
                                },
                                "shared.xs": {
                                    "name": "shared",
                                    "type": "xs-demean",
                                    "weight": 1,
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            workbook = root / "weights.xlsx"
            _write_workbook(
                workbook,
                [
                    ("A.json", "shared.ts", "shared", "ts", 1, 1),
                    ("A.json", "shared.xs", "shared", "xs-demean", 1, 1),
                ],
            )
            dates = pd.date_range("2024-01-01", periods=5, freq="B")
            report = run_full_portfolio_backtest(
                settings,
                workbook,
                start_date=dates[0].date(),
                end_date=dates[-1].date(),
                as_of=dates[-1].date(),
                provider_builder=lambda _scenarios, **_kwargs: _Provider(dates),
                progress=None,
            )

            self.assertEqual(
                set(report.total.signal_holdings),
                {"A:shared.ts", "A:shared.xs"},
            )

    def test_full_run_uses_new_weights_and_json_scalers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = root / "settings"
            settings.mkdir()
            _write_strategy(settings / "A.json", 10)
            _write_strategy(settings / "B.json", 20)
            workbook = root / "weights.xlsx"
            _write_workbook(
                workbook,
                [
                    ("A.json", "factor", "signal", "asset", 100, 2),
                    ("B.json", "factor", "signal", "asset", 100, 3),
                ],
            )
            dates = pd.date_range("2024-01-01", periods=20, freq="B")

            def builder(_scenarios, **_kwargs):
                return _Provider(dates)

            report = run_full_portfolio_backtest(
                settings,
                workbook,
                start_date=dates[0].date(),
                end_date=dates[-1].date(),
                as_of=dates[-1].date(),
                tenors=("6m", "1y"),
                provider_builder=builder,
                progress=None,
            )
            self.assertEqual(set(report.strategies), {"A.json", "B.json"})
            self.assertTrue((report.daily_pnl["A.json"] == 20).all())
            self.assertTrue((report.daily_pnl["B.json"] == 60).all())
            self.assertTrue((report.daily_pnl["full_portfolio"] == 80).all())
            self.assertEqual(list(report.total_metrics.index), ["full", "6m", "1y"])
            self.assertEqual(
                report.strategy_metrics.index.names, ["strategy", "tenor"]
            )
            self.assertEqual(
                report.signal_metrics.index.names,
                ["strategy", "factor_name", "tenor"],
            )
            self.assertEqual(
                list(report.signal_metrics.columns),
                [
                    "signal_name",
                    "type",
                    "new_weight",
                    "sharpe",
                    "daily_std",
                    "sortino",
                    "calmar",
                    "max_drawdown",
                    "annualized_pnl",
                    "total_pnl",
                    "turnover_pct",
                    "pnl_per_trade_bps",
                ],
            )

            combined = aggregate_strategy_results(report.strategies)
            pd.testing.assert_series_equal(
                combined.portfolio_pnl,
                report.daily_pnl["full_portfolio"].rename("full_portfolio"),
            )
            self.assertEqual(
                set(combined.signal_holdings),
                {"A:factor", "B:factor"},
            )

            html_path = write_full_portfolio_html(report, root / "report.html")
            document = html_path.read_text(encoding="utf-8")
            self.assertIn("Full portfolio cumulative PNL", document)
            self.assertIn("Cumulative PNL by strategy", document)
            self.assertIn("ending above 30M", document)
            self.assertIn("ending at or below 30M", document)
            self.assertIn('"hovermode":"closest"', document)
            self.assertIn("Strategy daily-PNL correlation", document)
            self.assertIn("Strategy Sharpe by tenor", document)
            self.assertIn("Strategy daily standard deviation by tenor", document)
            self.assertIn("report_strategies/A.html", document)
            self.assertIn("A.json", document)
            self.assertTrue((root / "report_assets" / "plotly.min.js").is_file())
            strategy_document = (
                root / "report_strategies" / "A.html"
            ).read_text(encoding="utf-8")
            self.assertNotIn("updatemenus", strategy_document)
            self.assertIn("factor cumulative PNL", strategy_document)
            self.assertIn("factor cumulative PNL by asset", strategy_document)
            self.assertIn("Performance by tenor", strategy_document)
            self.assertIn("calmar", strategy_document)
            self.assertIn("pnl_per_trade_bps", strategy_document)
            self.assertLess(
                strategy_document.index(">full<"),
                strategy_document.index(">6m<"),
            )
            self.assertLess(
                strategy_document.index(">6m<"),
                strategy_document.index(">1y<"),
            )


if __name__ == "__main__":
    unittest.main()
