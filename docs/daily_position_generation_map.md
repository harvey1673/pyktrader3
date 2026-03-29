# Daily Position Generation Map

## Scope
This document maps the production path that generates daily target position files for WT execution.

Main goal:
- pyktrader3 computes factor signals and portfolio target positions
- wtdev PortfolioTrader loads the generated daily JSON and executes toward those targets

## Deployment Context
- This local machine is not used for live trading.
- Live trading runs on a separate cloud server with similar code.
- Changes here should be treated as research/simulation or pre-production unless explicitly marked for cloud deployment.

## End-to-End Flow
1. `misc_scripts/port_position_update.py` orchestrates the daily update sequence.
2. `misc_scripts/auto_update_data_xl.py` refreshes and loads fundamental/EDB data from Excel-based sources.
3. `misc_scripts/fun_factor_update.py` computes and stores factor series into `fut_fact_data` (MySQL).
4. `misc_scripts/factor_data_update.py::update_port_position()` reads strategy configs and generates target positions.
5. Output JSON files are written under `process/paper_sim1/`.
6. `wtdev/Strategies/PortfolioTrader.py` loads `{strategy}_{postfix}_{trading_date}.json` and adjusts live positions.

## Key Entrypoints
- Position orchestration:
  - `misc_scripts/port_position_update.py`
  - Runs `fetch_sgx_eod`, `fun_data_xl_loading`, `fun_factor_update`, `fact_pos_file`
- Factor computation and DB writing:
  - `misc_scripts/fun_factor_update.py::update_db_factor`
- Position generation:
  - `misc_scripts/factor_data_update.py::update_port_position`
- Optional broader daily batch:
  - `misc_scripts/daily_update_job.py::run_update`

## Position Generation Contract
In `misc_scripts/factor_data_update.py`:
- `port_pos_config` defines portfolios and strategy list with scaler.
- For each strategy setting JSON in `{pos_loc}/settings/*.json`:
  - Parse `config.assets`, `factor_repo`, `roll_label`, `freq`, `repo_type`, `hist_fact_lookback`.
  - Call `pycmqlib3/strategy/strat_util.py::generate_strat_position`.
- Aggregate strategy-level target positions into portfolio-level `target_pos`.
- Apply product-specific lot step rounding rules.
- Write files:
  - `{pos_loc}/{port_name}_{yyyymmdd}.json` (final target position)
  - `{pos_loc}/curr_signal_{yyyymmdd}.json` (state for signal buffering)
  - `{pos_loc}/pos_by_strat_{port_name}_{yyyymmdd}.json` (strategy-level decomposition)

Current active portfolio config points to:
- `process/paper_sim1`
- Portfolio name: `PTSIM1_FACTPORT1_hot`

## Example Produced Files
Under `process/paper_sim1/`:
- `PTSIM1_FACTPORT1_hot_20260309.json`: product -> integer target lots
- `curr_signal_20260309.json`: factor key -> per-product signal values
- `pos_by_strat_PTSIM1_FACTPORT1_hot_20260309.json`: strategy file -> per-product contribution

## WT Execution Side
`wtdev/Strategies/PortfolioTrader.py` (`StraPortTrader`):
- Loads strategy universe from `{pos_loc}/settings/{strategy_name}.json`.
- On session begin and configured reload times, reads:
  - `{pos_loc}/{strategy_name}_{strat_postfix}_{cur_date}.json`
- For each product/contract:
  - Compare current position vs target
  - Apply minimum-open threshold guard (`min_open_rule`) for specific products
  - Call `stra_set_position` to adjust

This means the position file naming must match the strategy runtime args:
- strategy name (example: `PTSIM1_FACTPORT1`)
- postfix (example: `hot`)
- trading date from WT context

## Signal and Factor Plumbing
- Signal definitions and transforms are centralized in:
  - `pycmqlib3/strategy/signal_repo.py`
- Buffer behavior is controlled by:
  - `signal_buffer_config` in `signal_repo.py`
- Factor generation categories are in:
  - `single_factors`, `factors_by_asset`, `factors_by_spread`, etc. in `fun_factor_update.py`

Important integration points:
- `fun_factor_update.py::update_db_factor` computes factors and writes to MySQL table `fut_fact_data`.
- `strat_util.py::generate_strat_position` reads factor values from DB, applies transforms and weights from strategy JSON, scales by volatility and lot size, and returns target positions.

## Strategy Settings File Shape
Typical strategy file location:
- `process/paper_sim1/settings/PTSIM1_*.json`

Expected structure:
- top-level `class`
- `config` object with:
  - `name`, `freq`, `hist_fact_lookback`, `roll_label`, `repo_type`
  - `assets` with `underliers` and optional `prev_underliers`
  - `factor_repo` mapping of factor alias -> config (`name`, `type`, `rebal`, `weight`, etc.)

## How To Add A New Signal (Operational Checklist)
1. Define or verify signal formula in `pycmqlib3/strategy/signal_repo.py`.
2. Add it to the relevant factor collection in `misc_scripts/fun_factor_update.py` so it gets computed and written to `fut_fact_data`.
3. If buffering is needed, update `signal_buffer_config` in `signal_repo.py`.
4. Add the signal entry into a strategy settings JSON under `process/paper_sim1/settings/` with weight and type.
5. Ensure the strategy JSON is listed in `port_pos_config` in `misc_scripts/factor_data_update.py` (with scaler).
6. Run `misc_scripts/port_position_update.py` for a trading day and verify:
  - Position file exists
  - `pos_by_strat_*` includes the new strategy contribution
  - `curr_signal_*` includes the new signal key
7. Confirm WT strategy args (`name`, `strat_postfix`, `pos_loc`) match the generated filename.

## Known Risks and Maintenance Notes
- Missing/manual strategy file risk:
  - Previously this file was missing locally; after importing production files, `process/paper_sim1/settings/PTSIM1_MANUEL_TRADING.csv` is now present and active.
- DB dependency:
  - Position generation depends on up-to-date `fut_fact_data` and successful `update_db_factor` run.
- Naming tight coupling:
  - Any mismatch between generated filename and WT runtime naming causes execution to use stale/missing targets.
- Product step rounding:
  - Product-specific rounding in `update_port_position` can materially change final lots.

## Production Import Observations (2026-03-28)
- New latest daily outputs are present through `20260330`:
  - `process/paper_sim1/PTSIM1_FACTPORT1_hot_20260330.json`
  - `process/paper_sim1/curr_signal_20260330.json`
  - `process/paper_sim1/pos_by_strat_PTSIM1_FACTPORT1_hot_20260330.json`
- `PTSIM1_MANUEL_TRADING.csv` is included in settings and appears as an active component in `pos_by_strat_*` output.
- Core strategy setting files differ materially from older local backups:
  - `PTSIM1_FACTPORT1.json`, `PTSIM1_FUNFER.json`, `PTSIM1_FUNBASE.json`, `PTSIM1_FUNMTL.json`, `PTSIM1_LL2MR.json`, `PTSIM1_MR1Y.json`
- Example structural drift seen in imported settings:
  - broader asset universe in `PTSIM1_FACTPORT1.json`
  - changed `pos_scaler` values across multiple strategy configs
  - updated underlier month mappings and `prev_underliers`

## Practical Commands
- Generate/update positions for a date:
  - `python misc_scripts/port_position_update.py 20260309`
- Direct position-only generation (without orchestrator extras):
  - `python misc_scripts/factor_data_update.py 20260309`

## References
- `misc_scripts/port_position_update.py`
- `misc_scripts/factor_data_update.py`
- `misc_scripts/fun_factor_update.py`
- `pycmqlib3/strategy/strat_util.py`
- `pycmqlib3/strategy/signal_repo.py`
- `wtdev/Strategies/PortfolioTrader.py`
- `process/paper_sim1/settings/PTSIM1_FACTPORT1.json`
- `process/paper_sim1/PTSIM1_FACTPORT1_hot_20260309.json`
- `process/paper_sim1/curr_signal_20260309.json`
- `process/paper_sim1/pos_by_strat_PTSIM1_FACTPORT1_hot_20260309.json`
