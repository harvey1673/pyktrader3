# Signal Onboarding Process

This note defines the exact workflow to add a new signal into the daily position pipeline.

## Goal
Add one new tradable signal end-to-end so it can:
1. be defined in signal generation logic,
2. be persisted into factor DB,
3. be consumed by strategy JSON,
4. contribute to daily target position files.

## Step 1: Add Signal in signal_repo
File:
- `pycmqlib3/strategy/signal_repo.py`

Action:
- Add a new key in `signal_store` with:
  - universe/list of assets or spread names,
  - feature key,
  - transform function,
  - params,
  - post-processing (optional, e.g. `buf0.2`).

Notes:
- If this signal needs buffering between days, add it to `signal_buffer_config`.
- If signal execution timing/lag is special, add to `signal_execution_config`.
- If it needs a custom feature mapping, ensure `feature_to_feature_key_mapping` supports it.

## Step 2: Route Signal in fun_factor_update
File:
- `misc_scripts/fun_factor_update.py`

Choose exactly one (or more if intentional) route by signal shape:

1. `single_factors`
- Use when one signal time series is generated and then copied to a fixed list of assets.
- Runtime path: loop over `single_factors`, call `get_funda_signal_from_store(...)`, then `save_signal_to_db(...)` per asset.

2. `factors_by_asset`
- Use when signal is naturally asset-specific and computed per asset.
- Runtime path: loop `(factor_name, asset)` and compute per asset.
- Supports ts/xs naming patterns via `*_xdemean`, `*_xscore`, `*_xrank` conventions.

3. `factors_by_spread`
- Use when one spread signal should be linearly projected to legs with static weights.
- Example pattern: `[('au', 1), ('ag', -1)]`.

4. `factors_by_spread2`
- Use when spread signal is tied to a named spread config in `spread_config` and requires spread-vol persistence (`spd_vol`) plus leg-level projection.

5. `factors_by_spread3`
- Use for pair-constructed spread risk scaling where volatility is derived from pair return differences.

6. `factors_by_beta_neutral`
- Use when signal must be translated into beta-neutral legs.
- Runtime uses precomputed `beta_dict` (`trade_leg`, `index_leg`) and writes per-asset factors.

7. `factors_by_func`
- Use if signal is easier as a custom function that returns a full DataFrame of signals by asset.

## Step 3: Ensure DB Factor Exists
The daily updater (`update_db_factor`) must write this factor into `fut_fact_data`.

Validation:
- confirm no empty series for targeted assets,
- confirm factor rows are inserted for expected dates,
- confirm naming in DB matches strategy JSON `factor_repo[*].name`.

## Step 4: Add Signal to Strategy JSON
Files:
- typically under `process/paper_sim1/settings/*.json`

Action:
- Add a new `factor_repo` entry in relevant strategy JSON.
- Required fields per factor entry:
  - `name`: DB factor name (must match saved factor name),
  - `type`: `ts`, `xs-*`, or `pos`,
  - `rebal`, `weight`, `threshold`, `param`, optional `exec_assets`.

Important:
- `name` is the lookup key used by `generate_strat_position` when loading from DB.
- `type` controls transform behavior in `strat_util.generate_strat_position`.
- `weight` directly scales contribution to final `target_pos`.

## Step 5: Ensure Portfolio Includes the Strategy File
File:
- `misc_scripts/factor_data_update.py` (`port_pos_config`)

Action:
- Ensure the strategy JSON is listed in the active portfolio `strat_list` with proper scaler.
- If using manual overlay, keep `PTSIM1_MANUEL_TRADING.csv` behavior in mind.

## Step 6: Run and Verify Outputs
Run:
- `python misc_scripts/port_position_update.py YYYYMMDD`

Check generated artifacts:
- `process/paper_sim1/PTSIM1_FACTPORT1_hot_YYYYMMDD.json`
- `process/paper_sim1/curr_signal_YYYYMMDD.json`
- `process/paper_sim1/pos_by_strat_PTSIM1_FACTPORT1_hot_YYYYMMDD.json`

Verify:
1. New factor appears in `curr_signal_YYYYMMDD.json` (if buffered/current signal tracked).
2. Strategy contribution reflects new signal in `pos_by_strat_*`.
3. Aggregate target lot changes propagate to final portfolio position JSON.

## Decision Quick Guide
- One signal -> many assets fixed list: `single_factors`
- Per-asset independent signal: `factors_by_asset`
- One spread signal -> fixed leg weights: `factors_by_spread`
- Spread-config-driven with spread vol persistence: `factors_by_spread2`
- Pair-return-vol based spread normalization: `factors_by_spread3`
- Beta-neutral legging: `factors_by_beta_neutral`
- Bespoke composite logic returning DataFrame: `factors_by_func`

## Common Failure Modes
1. Name mismatch:
- `signal_repo` key, DB factor name, and strategy JSON `name` differ.

2. Wrong routing dict:
- Signal added to `single_factors` while logic requires per-asset or spread handling.

3. Universe mismatch:
- Asset list in routing dict does not align with strategy assets.

4. Missing portfolio inclusion:
- Strategy JSON updated but not included in `port_pos_config`.

5. Buffer expectations not configured:
- Signal intended to be buffered but missing from `signal_buffer_config`.

6. Roll/freq mismatch:
- Strategy `roll_label`/`freq` not aligned with factor data being written.

## Recommended Safe Rollout Pattern
1. Add signal in code and route it.
2. Add to one strategy JSON with small weight.
3. Run one trading date locally.
4. Inspect `curr_signal`, `pos_by_strat`, final position JSON.
5. Scale weight after sanity checks.

## Local vs Production Context
- This workstation is non-live and used for research/simulation/pre-production.
- Live execution is on separate cloud infrastructure.
- Treat local onboarding as a dry run before cloud promotion.
