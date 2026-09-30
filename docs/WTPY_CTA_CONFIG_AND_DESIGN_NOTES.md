# WTPY CTA Config And Design Notes (2026-05-29)

## Scope

This document summarizes:

1. How cta_prod YAML configuration maps to runtime behavior
2. How to switch execution algorithm behavior
3. How to enforce position matching vs allow temporary non-matching
4. What files under generated/ are used for, and safer recovery operations
5. Root-cause analysis for the MA609 repeated 3-lot reject loop and post-restart oscillation
6. Existing controls and tradeoffs for limit-up/limit-down execution churn

Environment note: this workstation is research/pre-production only; live trading runs on a separate cloud host.

---

## 1) cta_prod YAML mapping to runtime behavior

### config.yaml (top-level wiring)

Main role: wire all sub-config files and runtime modules.

Key paths and effects:

- env.filters -> filters.yaml
  - runtime trading filters (strategy/code/executer filters)
- env.bspolicy -> actpolicy.yaml
  - action policy (close/open ordering and behavior)
- executers -> executers.yaml
  - local executer definitions and execution policy
- parsers -> tdparsers.yaml
  - market data adapters
- traders -> tdtraders.yaml
  - trading channel adapters and risk monitor limits
- notifier -> IPC / event notifications

### executers.yaml

Main role: execution behavior and per-instrument execution unit parameters.

Observed important settings:

- strict_sync: false
  - false: only instruments with strategy target are controlled strictly
  - true: unmanaged residual positions can also be cleaned/synchronized
- policy.default
  - default execution unit settings
- policy.<product>/policy.<stdCode>
  - instrument/product overrides, e.g. lots, expire, and algorithm params

### actpolicy.yaml

Main role: open/close action ordering policy (how actions are sequenced).

### tdtraders.yaml

Main role: trader channel and risk monitor thresholds.

Observed important settings:

- savedata: true
  - runtime state persisted into generated/traders/<trader_id>/rtdata.json
- riskmon limits
  - order/cancel throttles and safety controls

### run.py

Main role: start ET_CTA engine, load configs, bind strategy, run schedule.

### Strategy (StraPortTrader / PortfolioTrader)

Main role: generate target positions and send them into execution pipeline.

Observed important behavior:

- min_open_rule includes MA:4 in strategy-side rule logic
- at schedule points, target changes call set_position path
- threshold logic only skips some small same-direction increases; not a universal lot-multiple guard

---

## 2) How to change execution algorithms

Execution algorithm behavior is controlled by executers.yaml policy entries.

Two practical levels:

1. Global defaults
- modify policy.default for baseline behavior

2. Product/instrument override
- add/modify policy.<product> or policy.<stdCode>
- this is preferred for exchange-specific constraints such as minimum opening lot multiples

For this incident class (MA minimum new position 4 lots), explicitly define minopenlots (and/or lots) for MA in executers.yaml policy.

Operational guidance:

- If exchange rule is strict multiple (e.g. 4), do not rely only on strategy threshold.
- Enforce at execution-unit level via executers policy so recalc/reorder loops cannot submit invalid residuals.

### Executor unit survey (vs default WtMinImpactExeUnit)

Baseline default in this stack: WtMinImpactExeUnit.

Baseline objective:

- Chase target with low implementation shortfall using quote-book aware child orders and periodic recalc.

Baseline policy fields (WtMinImpactExeUnit):

- offset: integer tick offset added to selected reference price.
- expire: seconds before managed live order is considered stale and canceled.
- pricemode: price selection mode. -1=best, 0=latest, 1=opponent, 2=auto.
- span: minimum milliseconds between two placements/re-calcs.
- byrate: if true, use rate as dynamic sizing basis from opposite book; if false, use lots.
- lots: fixed child order quantity when byrate=false.
- rate: proportional child order factor when byrate=true.
- minopenlots: optional open-side minimum. If computed opening child size is smaller than this threshold, quantity is lifted to this value.

The following units are also implemented in the same executor factory:

### 2.1 WtTWapExeUnit

What it tries to achieve:

- Time-slice toward target along a schedule (TWAP-style), with periodic order refresh/cancel.

Pros vs WtMinImpactExeUnit:

- More predictable pacing and participation over a defined window.
- Easier to cap short-term aggressiveness using schedule and single-shot size.

Cons vs WtMinImpactExeUnit:

- Less reactive to order-book liquidity state than MinImpact.
- More schedule tuning required; wrong window settings can delay convergence.

Available policy fields and meaning:

- ord_sticky: pending-order timeout (seconds) before cancel/reprice cycle.
- begin_time: execution window start (HHMM).
- end_time: execution window end (HHMM).
- total_secs: configured execution duration in seconds, but current code recomputes total duration from begin_time/end_time and overwrites this value.
- tail_secs: reserved tail buffer (seconds) for final catch-up orders.
- total_times: planned number of slices across the active window.
- price_mode: price selection mode used by this unit.
- price_offset: integer tick offset from selected reference price.
- lots: child order quantity per fire.
- minopenlots: optional open-side minimum child size.

### 2.2 WtVWapExeUnit

What it tries to achieve:

- Time-profile execution toward a VWAP participation curve read from an external profile file.

Pros vs WtMinImpactExeUnit:

- Better alignment with expected intraday volume curve when profile is accurate.
- Can reduce concentration risk compared with purely reactive chasing.

Cons vs WtMinImpactExeUnit:

- Depends on external VWAP profile file (Vwap_<commodity_name>.txt).
- If profile is stale/wrong, execution quality can degrade significantly.

Available policy fields and meaning:

- begin_time: execution window start (HHMM).
- end_time: execution window end (HHMM).
- ord_sticky: pending-order timeout (seconds).
- tail_secs: tail reserve window (seconds) for catch-up.
- total_times: planned number of decision slices.
- price_mode: price selection mode.
- price_offset: integer tick offset from selected reference price.
- lots: child order quantity per fire.
- minopenlots: optional open-side minimum child size.

### 2.3 WtStockVWapExeUnit

What it tries to achieve:

- Stock-oriented VWAP execution with stock market microstructure handling (including board/min-order adaptation).

Pros vs WtMinImpactExeUnit:

- Better suited for stock session/lot constraints than generic futures-oriented MinImpact.
- VWAP curve mode can smooth intraday participation for equities.

Cons vs WtMinImpactExeUnit:

- More moving parts (VWAP profile + stock-specific min-order rules).
- Field naming differs from WtVWapExeUnit for price offset, increasing config error risk.

Available policy fields and meaning:

- begin_time: execution window start (HHMM).
- end_time: execution window end (HHMM).
- ord_sticky: pending-order timeout (seconds).
- tail_secs: tail reserve window (seconds).
- total_times: planned number of slices.
- price_mode: price selection mode.
- offset: integer tick offset from selected reference price.
- lots: child order quantity per fire.
- minopenlots: optional minimum order size (then adjusted by stock board/min-order logic internally).

### 2.4 WtStockMinImpactExeUnit

What it tries to achieve:

- MinImpact logic adapted for stocks, including account-amount basis and stock min-order handling.

Pros vs WtMinImpactExeUnit:

- Adds stock-specific controls (capital-based sizing, unmanaged-order handling toggles).
- Better fit for stock lot/board constraints and T0/T1 mode differences.

Cons vs WtMinImpactExeUnit:

- More complex behavior and more optional knobs; easier to misconfigure.
- Some semantics (for min_order and board adjustment) require careful validation per market.

Available policy fields and meaning:

- offset: integer tick offset from selected reference price.
- expire: seconds before live order is considered stale.
- pricemode: price selection mode. -1=best, 0=latest, 1=opponent, 2=auto.
- span: minimum milliseconds between placements/re-calcs.
- byrate: true uses rate-based dynamic sizing, false uses lots.
- lots: fixed child size when byrate=false.
- rate: proportional child size factor when byrate=true.
- total_money: optional notional amount budget used by executor for ratio-style targets; if omitted/non-positive, runtime available funds are used.
- is_cancel_unmanaged_order: optional boolean; whether to cancel unmanaged live orders on channel ready.
- max_cancel_time: optional integer cap for repeated cancel/retry cycles.
- min_order: optional minimum order quantity for stock order placement (later adjusted by stock board/min-order constraints).

### 2.5 WtDiffMinImpactExeUnit

What it tries to achieve:

- Execute toward a target difference (spread/diff style) while tracking leftover diff from trades.

Pros vs WtMinImpactExeUnit:

- Better fit for diff-style execution flows where the managed state is a remaining difference rather than absolute position.
- Keeps dedicated left-diff bookkeeping updated by trade callbacks.

Cons vs WtMinImpactExeUnit:

- Not created from normal createExeUnit path; intended for diff execution path only.
- Fewer safeguards/knobs than StockMinImpact for market-specific constraints.

Available policy fields and meaning:

- offset: integer tick offset from selected reference price.
- expire: seconds before live order is considered stale.
- pricemode: price selection mode. -1=best, 0=latest, 1=opponent, 2=auto.
- span: minimum milliseconds between placements/re-calcs.
- byrate: true uses rate-based dynamic sizing, false uses lots.
- lots: fixed child size when byrate=false.
- rate: proportional child size factor when byrate=true.

Operational note:

- For CZCE MA minimum-open constraints, WtMinImpactExeUnit remains the most direct and least surprising default in CTA futures flow; use explicit per-product/per-code lots and minopenlots to avoid invalid residual submissions.

### Limit-up / limit-down churn control

Problem shape:

- If a contract is pinned at upper limit while the executor still wants to buy, or pinned at lower limit while the executor still wants to sell, repeated place/cancel/recalc behavior can trigger exchange warnings.

Observed built-in mechanisms:

1. Executor-side price clipping in WtMinImpactExeUnit
- If computed buy price is above upper limit, it is clipped down to upper limit.
- If computed sell price is below lower limit, it is clipped up to lower limit.
- When clipping happens through this path, the order is marked as non-cancelable in local order monitor logic, so normal expire-driven cancel loop will skip that order and leave it queued.

2. Trader-side flow control in tdtraders.yaml riskmon
- order_stat_timespan / order_times_boundary limit order frequency within a time window.
- cancel_stat_timespan / cancel_times_boundary limit cancel frequency within a time window.
- order_total_limits / cancel_total_limits cap daily totals.
- If a symbol breaches configured thresholds, TraderAdapter adds that symbol to an internal exclude list and further order/cancel requests for that symbol are blocked.

Important caveat:

- The no-cancel behavior only triggers when computed price crosses beyond the limit and then gets clipped.
- If computed price lands exactly on the limit without crossing it, current code does not necessarily mark the order as non-cancelable.
- In that exact-at-limit case, repeated expire/cancel/reorder behavior can still occur.

Practical options without code changes:

### Option 1: Leave order queued at limit and avoid auto-cancel

Goal:

- If the market is locked at limit, place at limit and keep queue priority instead of repeated cancel/reorder.

How:

- Use WtMinImpactExeUnit with pricemode: 1 and a small positive offset such as 1.
- This intentionally pushes computed aggressive price beyond the limit when the market is pinned, causing price clipping path to trigger.
- Result: order is left at limit price and local expire logic will not cancel it.

Tradeoff:

- A positive offset can increase execution cost in normal, non-limit conditions because it makes pricing more aggressive.

Use-case recommendation:

- Only enable this on products/codes where limit-lock churn is a recurrent operational problem.
- Prefer product/code override in executers.yaml rather than changing global default behavior.

### Option 2: Keep offset conservative and rely on channel throttles

Goal:

- Avoid increasing normal execution cost, while still preventing exchange-warning-level churn.

How:

- Keep offset unchanged.
- Tighten tdtraders.yaml riskmon thresholds for affected products or default channel policy.
- When thresholds are breached, the symbol is blocked at trader adapter level.

Tradeoff:

- This is a downstream brake, not a graceful queue-at-limit mechanism.
- It stops further requests only after frequency thresholds are hit.

### Option 3: Emergency operational stop for one symbol

Goal:

- Stop one locked-limit symbol immediately without changing execution cost model.

How:

- Add a code filter with action: ignore for the affected symbol or product.
- This is the safest operational control if exchange-warning risk is already building.

Recommendation for current preference:

- If execution-cost sensitivity is more important than queue priority at limit, do not add offset broadly.
- Keep current pricing behavior, tighten riskmon thresholds, and use code filter ignore as the emergency stop.
- If one or two products frequently lock limit and produce warning risk, consider a targeted per-product override using small positive offset only for those products.

What does not exist today as a simple config flag:

- There is no native executers.yaml field that means stop execution immediately when symbol is at price limit.
- A true halt-on-limit behavior would require a small executor code enhancement, for example a flag such as halt_on_price_limit: true.

---

## 3) How to force matching vs allow non-matching

### A. Force matching behavior

- Keep strategy active and emitting targets
- Keep executer enabled
- Prefer strict_sync=true if you want stronger cleanup/synchronization of unmanaged residuals

Result:
- runtime continuously drives actual position toward target position

### B. Temporarily allow non-matching (do not auto-chase)

Three common controls:

1. Code filter in filters.yaml
- add code_filters entry with action: ignore for specific instrument or product
- effect: target update for those codes is ignored by filter manager

2. Strategy filter in filters.yaml
- add strategy_filters entry with action: ignore
- effect: suppress all target updates from that strategy

3. Executer filter in filters.yaml
- disable an executer id in executer_filters
- effect: that executer will not process tasks temporarily

Use-case recommendation:

- For one-symbol emergency stop (e.g. MA only): code filter is least disruptive.
- For full strategy freeze: strategy filter.
- For full channel stop: executer filter (broadest impact).

Emergency MA2609 example:

- If the goal is to stop the rejected 3-lot chase immediately, add a code filter for CZCE.MA.2609 with action: ignore.
- If you want MA to stay active but pin the target to the current live position, use action: redirect and set target to the broker truth, such as 41 during the incident.

Runtime behavior:

- filters.yaml is read during runtime.
- The CTA engine reloads filters on schedule, and the filter manager only applies the new file when the timestamp changes.
- In practice, saving the file takes effect on the next schedule cycle, so you usually do not need a full restart.
- There can still be a short delay of up to one schedule interval.

---

## 4) generated/ files and ownership

### generated/traders/<trader_id>/rtdata.json

Owner: TraderAdapter persistence.
Contains:

- broker-observed positions
- undone quantities
- channel runtime state snapshots

### generated/stradata/<strategy>.json

Owner: strategy context persistence.
Contains:

- strategy-side position cache
- signal cache/history used during replay/restart

### generated/portfolio/datas.json

Owner: portfolio/engine state persistence.
Contains:

- target/cache state for portfolio layer

### generated/marker.json

Owner: engine marker/recovery metadata.
Contains:

- run/session markers used during restart/recovery flow

Important:
- Manual edits can fix emergency mismatch, but increase risk of inconsistent fields.
- Prefer controlled script-based edits with backup + dry-run.

---

## 5) Incident timeline and root cause (MA609)

## Timeline (key evidence)

1. Around 09:03 strategy target moved MA from 40 to 44
- strategy log shows: adjust position for CZCE.MA.2609 from 40.0 to 44

2. Execution layer then repeatedly tried residual buy quantity=3
- repeated buy/cancel/recalc cycle observed roughly every 5s

3. Exchange rejected those orders due to minimum new-open multiple constraint
- trader log shows repeated order rejects for MA quantity=3 with exchange minimum-volume related error text

4. Around 09:20 restart occurred (re-login + re-query + re-subscribe)
- then broad reconciliation orders fired across many symbols
- user-observed behavior (sell then immediate buy) is consistent with restart-time reconciliation under stale/misaligned state snapshots and pending execution tasks

## Root cause summary

Primary cause:

- residual delta reached 3 lots while exchange minimum new open required 4 lots
- execution unit kept recalculating and resubmitting because min lot multiple was not enforced at execution policy level for that code/path

Contributing factors:

- strategy threshold/min_open_rule did not fully prevent this path in all recalc states
- restart during live session triggered aggressive synchronization/reconciliation
- generated state mismatch (strategy cache vs broker runtime vs pending tasks) amplified oscillation risk

---

## 6) Immediate mitigations

1. Add explicit MA execution-unit minimum lot policy in executers.yaml
- enforce minopenlots/lots for MA to valid exchange multiple

2. Add emergency code filter playbook
- filters.yaml code_filters for rapid per-instrument freeze

3. Restart safety checklist
- before restart, align strategy state to broker state for affected symbols
- clear stale signals for affected symbols
- restart only after mismatch is reduced or intentionally frozen

4. Add reject-loop monitor
- detect repeated same-symbol same-reason rejects in short window and auto-trigger filter/freeze alert

---

## 7) Automation helper added

A helper script was added to avoid manual JSON edits:

- tools/wtdev_generated_state_tool.py

What it does:

- inspect broker net positions from generated/traders/<trader_id>/rtdata.json
- align generated/stradata/*.json selected instrument volume to broker net position
- optionally clear stale signals for selected instruments
- default dry-run, with file backup on --apply

Example usage:

1. inspect MA broker net:

python tools/wtdev_generated_state_tool.py inspect --code CZCE.MA.2609

2. dry-run align for MA and clear stale signals:

python tools/wtdev_generated_state_tool.py align --code CZCE.MA.2609 --clear-signals

3. apply align:

python tools/wtdev_generated_state_tool.py align --code CZCE.MA.2609 --clear-signals --apply

---

## 8) Recommended next hardening steps

1. Policy hardening
- maintain per-product/per-code minimum lot/multiple constraints in executers.yaml

2. Pre-restart guard command
- create one-click script: snapshot -> validate mismatch -> align stradata -> optional freeze filters -> restart

3. Post-restart gradual release
- unfreeze in stages by product group instead of all-at-once

4. Monitoring

---

## 9) filters.yaml templates by scenario

Important note:

- Monitor UI "过滤" for strategy/code writes redirect->0 (flatten intent), not ignore.
- If you want to pause trading and keep current positions unchanged, use action: ignore manually in filters.yaml.

### Scenario A: Pause one contract and keep existing position (recommended for MA609 loop stop)

Goal:

- Stop new target execution for one contract
- Do not force flatten existing position

Template:

  code_filters:
    CZCE.MA.2609:
      action: ignore

### Scenario B: Pause all contracts of one product and keep existing positions

Goal:

- Stop all MA-family target execution
- Keep existing positions

Template:

  code_filters:
    CZCE.MA:
      action: ignore

### Scenario C: Pause whole strategy and keep existing positions

Goal:

- Stop target updates from one strategy only

Template:

  strategy_filters:
    PTSIM1_FACTPORT1:
      action: ignore

### Scenario D: Freeze entire execution channel

Goal:

- No new execution through one executer/channel

Template:

  executer_filters:
    exec: true

Note:

- exec id must match executer name in executers.yaml.

### Scenario E: Intentionally flatten one contract

Goal:

- Force target to zero for one contract

Template:

  code_filters:
    CZCE.MA.2609:
      action: redirect
      target: 0

### Scenario F: Hold one contract at current live broker position

Goal:

- Keep MA at broker truth temporarily (for controlled restart/recovery)

Template:

  code_filters:
    CZCE.MA.2609:
      action: redirect
      target: 41

Update target to actual live position number during incident.

### Scenario G: Resume normal execution

Goal:

- Remove temporary controls after incident stabilizes

Template:

  strategy_filters: {}
  code_filters: {}
  executer_filters: {}

### MA609 emergency order of operations

1. Add Scenario A (ignore MA2609) to stop reject-loop churn.
2. Verify no new MA orders are sent.
3. Align strategy cache with broker truth if mismatch exists.
4. Remove ignore and switch to desired target workflow when safe.
5. Ensure executers policy has MA min lot rule to prevent recurrence.
- alert on repeated reject loop pattern (same code + same reason + repeated within N minutes)

---

## 10) Source-verified risk monitor options (WonderTrader/WTPY)

This section is based on source-level inspection of WonderTrader C++ runtime
paths and is intended to complement the config-level notes above.

### 10.1 Trading channel risk monitor (tdtraders.yaml)

Location:

- traders[].riskmon in tdtraders.yaml

Available keys:

- active
- policy
  - default (recommended, see note below)
  - <product_key> entries, where product_key is standard commodity id,
    for example CFFEX.IF or SHFE.RB

Per-policy limits:

- order_stat_timespan
- order_times_boundary
- order_total_limits
- cancel_stat_timespan
- cancel_times_boundary
- cancel_total_limits

Matching and precedence:

1. Runtime converts stdCode to standard product id.
2. It first looks for product-specific policy.
3. If not found, it falls back to default.

Behavior details:

- Breach of either frequency or daily-total thresholds moves that stdCode
  into an internal exclusion set.
- Excluded stdCode is blocked for subsequent order/cancel checks.
- Frequency trigger uses "times > boundary" logic (strictly greater).
- Exclusion is not automatically cleared during runtime.

Operational recommendation:

- Always define policy.default.
- Then add stricter per-product overrides only where needed.

Example:

  traders:
  - id: cyqh_ctp
    riskmon:
      active: true
      policy:
        default:
          order_stat_timespan: 10
          order_times_boundary: 20
          order_total_limits: 1000
          cancel_stat_timespan: 10
          cancel_times_boundary: 20
          cancel_total_limits: 470
        CFFEX.IF:
          order_stat_timespan: 10
          order_times_boundary: 10
          order_total_limits: 300
          cancel_stat_timespan: 10
          cancel_times_boundary: 10
          cancel_total_limits: 120

Per-symbol override note:

- Native policy granularity is product-level, not single contract-level.
- If true per-contract limits are required, use custom extension logic,
  or split routing/channel controls as a workaround.

### 10.2 Portfolio risk monitor (env.riskmon in config.yaml)

Location:

- env.riskmon in config.yaml

Built-in factory and monitor:

- module: WtRiskMonFact
- name: SimpleRiskMon

Available keys for SimpleRiskMon:

- active
- module
- name
- base_amount
- basic_ratio
- calc_span
- inner_day_active
- inner_day_fd
- multi_day_active
- multi_day_fd
- risk_scale
- risk_span

Core logic summary:

- Monitor runs periodically every calc_span seconds while trading is active.
- Intraday branch:
  - checks drawdown from intraday dynamic-equity peak
  - requires prior profit threshold via basic_ratio
  - if trigger reached within risk_span window, applies risk_scale
- Multi-day branch:
  - checks drawdown from multi-day max dynamic equity
  - on trigger, scales position factor to 0.0

How it affects CTA execution:

- Risk monitor calls setVolScale(scale).
- CTA engine uses this scale when committing target positions for
  execution on current trading day.

Example:

  env:
    riskmon:
      active: true
      module: WtRiskMonFact
      name: SimpleRiskMon
      base_amount: 18000000
      basic_ratio: 110
      calc_span: 5
      inner_day_active: true
      inner_day_fd: 20.0
      multi_day_active: true
      multi_day_fd: 60.0
      risk_scale: 0.3
      risk_span: 30

Current cta_prod status (as observed in this workspace):

- tdtraders channel riskmon: active
- env portfolio riskmon: inactive

---

## 11) process_wt_data.py merge/backfill notes (with periods)

This section documents the operational data-repair helpers in
pycmqlib3/utility/process_wt_data.py and ties them to runtime config
recovery workflows.

### 11.1 Main merge/update helpers

combine_bars_wt_store(src_folder, dst_folder, target_folder, cutoff=None)

- Scope: bars only (day/min1/min5)
- Typical use: merge two historical stores with cutoff splice
- Merge logic:
  - keep src data up to cutoff
  - keep dst data after cutoff
  - concatenate and write to target store

combine_data_wt_store(..., periods=["day", "min1", "min5", "ticks"])

- Scope: selective window replacement for bars and ticks
- Key parameter: periods controls which domains are processed
  - supported values: day, min1, min5, ticks
- For bars:
  - replace records in [time_range[0], time_range[1]] using dst source
  - keep outside-window records from src
- For ticks:
  - same window replacement logic on tick time
  - per trading date folder

update_wt_store(base_folder, update_folder, cutoff=None)

- Calls combine_bars_wt_store for bar domains
- Then copies ticks and snapshot trees from update folder to base folder
- Intended for practical store refresh/backfill after data repair

### 11.2 Recommended repair workflow

1. Backup base store before merge.
2. Run combine_data_wt_store for narrow incident windows first.
3. Validate merged data in target store.
4. Run update_wt_store only after validation passes.
5. Restart engine and verify generated state consistency if live runtime
   snapshots are also involved.

### 11.3 Config linkage with earlier sections

- Data merge fixes market data/input quality, not execution/risk policy.
- If incident involved order churn/reject loops, pair data repair with:
  - executer policy hardening (lots/minopenlots)
  - channel riskmon threshold review
  - temporary filters.yaml controls during stabilization

This is why data-plane repair and config-plane controls should be tracked
together in the same operational playbook.
