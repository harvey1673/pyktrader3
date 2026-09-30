# Daily PnL Explain Tool

`misc_scripts/wt_daily_pnl_explain.py` builds a day-level PnL explain report for the
paper portfolio using two local sources:

- `process/paper_sim1/*` daily target files
- `wtdev/deploy/cta_prod/generated/outputs/PTSIM1_FACTPORT1/*` runtime PnL
  snapshots

## Inputs

- `PTSIM1_FACTPORT1_hot_YYYYMMDD.json`: total paper target by product
- `pos_by_strat_PTSIM1_FACTPORT1_hot_YYYYMMDD.json`: product target by strategy
- `generated/outputs/PTSIM1_FACTPORT1/positions.csv`: code-level PnL state
- `generated/outputs/PTSIM1_FACTPORT1/closes.csv`: realized close PnL
- `generated/outputs/PTSIM1_FACTPORT1/trades.csv`: executions and fees
- `generated/outputs/PTSIM1_FACTPORT1/funds.csv`: portfolio-level daily PnL
- `generated/traders/cyqh_ctp/rtdata.json`: current exchange-reported account
  funds and positions; the report reads this file and never modifies it

`PTSIM1_MANUEL_TRADING.csv` is already represented inside `pos_by_strat_*`, so
manual trading is included automatically in the attribution output.

## What The Tool Produces

- `code_pnl_detail.csv`: code-level daily explain from two-day snapshot changes
- `product_pnl_detail.csv`: product-level daily PnL
- `strategy_pnl_detail.csv`: product PnL allocated to strategies
- `strategy_pnl_summary.csv`: total attributed PnL by strategy
- `input_issues.csv`: invalid or non-finite strategy-position inputs
- `current_account.csv`: exchange-reported balance, available funds, margin,
  fees and related account fields from `rtdata.json`
- `current_positions.csv`: current long, short, net and available positions by
  contract from `rtdata.json`
- `report.md`: short markdown summary
- `summary.json`: headline totals, including manual trading contribution

## Attribution Logic

- Gross code daily PnL is realized PnL from `closes.csv` plus the change in
  EOD `dynprofit` from `positions.csv`.
- Net code/product PnL subtracts execution fees from `trades.csv`.
- Contract detail includes night-session and day-session executed volume and
  volume-weighted average execution price from `trades.csv`. Product and
  contract tables are both shown in full in the Markdown report.
- Night-session trades and closes are assigned to the next available EOD
  trading date. For example, Friday 21:05 activity belongs to Monday.
- The sum of product PnL is reconciled to the authoritative equity change in
  `funds.csv`. Any remaining state reset or cash adjustment is retained as
  `UNEXPLAINED_ADJUSTMENT`.
- Strategy attribution uses signed previous-day `pos_by_strat_*` product lots
  when available and scales their estimated per-lot PnL using the actual prior
  EOD runtime position. If a product is newly opened, the current-day strategy
  and runtime positions are used instead.
- Any difference caused by paper-target versus runtime-position mismatch is
  retained as `UNALLOCATED_POSITION_MISMATCH`; it is never amplified across
  offsetting strategies merely to force the strategy sum to reconcile.
- Strategy weights remain signed. Products with offsetting zero-net positions,
  missing position bases, or non-finite inputs are left explicitly unallocated
  instead of being forced through absolute exposure weights.

## Example

```powershell
python misc_scripts/wt_daily_pnl_explain.py 20260529 `
  --port-dir C:/dev/pyktrader3/process/paper_sim1 `
  --group-dir C:/dev/wtdev/deploy/cta_prod `
  --port-file PTSIM1_FACTPORT1_hot `
  --output-strategy PTSIM1_FACTPORT1
```

Omit the positional date for the scheduled/default date rule:

- on a China working day before 21:00: current working date
- on a China working day at or after 21:00: next working date
- on a weekend or China holiday: previous working date

Email uses the same `EMAIL_QQ`, `NOTIFIERS`, `LOCAL_PC_NAME`, `EMAIL_NOTIFY`
and `send_html_by_smtp` configuration as `port_position_update.py`. Use
`--email` to force a send when the shared `EMAIL_NOTIFY` switch is disabled, or
`--no-email` for a report-only run.

The default `--mode auto` uses completed EOD data when both `positions.csv` and
`funds.csv` contain the selected date. Before those 15:15 snapshots are
available, it uses live `stradata`, exchange `trades.csv`/`closes.csv`, and
`rtdata.json`, with the latest completed EOD as the opening baseline. Use
`--mode eod` or `--mode intraday` to force a mode. Intraday mode does not require
the current date to exist in either EOD CSV; if the current strategy target
files are not available yet, it falls back to the previous target snapshot.

Default output directory:

`C:/dev/data/analytics/PTSIM1_FACTPORT1_hot_20260529/`
