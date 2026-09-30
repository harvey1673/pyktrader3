# pysystemtrade historical futures export

Inspected 15 September 2026. Source: `C:/dev/pysystemtrade/data/futures`.
Reference: `C:/dev/data/fut_d_20260914.parquet` (not double underscore).

## What the folders contain

| Folder | Files | Meaning |
|---|---:|---|
| adjusted_prices_csv | 252 | Continuous prices, additively back-adjusted for rolls (Panama method). |
| multiple_prices_csv | 252 | PRICE and PRICE_CONTRACT identify the held contract; FORWARD and FORWARD_CONTRACT identify the next contract in the roll sequence; CARRY and CARRY_CONTRACT provide a comparison contract for carry. These are not full individual-contract histories. |
| fx_prices_csv | 12 | Currency conversion series against USD: AUD, CAD, CHF, CNH, EUR, GBP, HKD, JPY, KRW, MXP, SEK, SGD. JPYUSD means USD per JPY, not the usual USDJPY quotation. Preserve the source's MXP code. |
| roll_calendars_csv | 279 | Dates with current, next and carry contract identifiers; this broader universe is not proof that every market has prices. |
| crypto_spread_roll_calendars_csv | 8 | Separate roll calendars; not eight additional price datasets. |
| csvconfig | 3 | Instrument descriptions, point values, currencies, asset classes and cost parameters; roll parameters; spread costs. These are current configuration snapshots, not historical effective-dated specifications. |

The additive method is verified in `sysobjects/adjusted_prices.py`: at a roll,
the prior FORWARD minus prior PRICE is added to all earlier adjusted prices.
An identifier such as 19821200 denotes December 1982; `00` is not an expiry day.
The active series need not be the nearest listed maturity, and FORWARD is not
necessarily the exchange's second nearest contract. Instrument codes are
pysystemtrade identifiers, not validated IB execution symbols.

## Built files

Under `C:/dev/pyktrader3/output/pysystemtrade_history/`:

- `fut_d_pysystemtrade_20260914.parquet`: 14,998 dates, 252 markets, 2,268 columns.
- `fx_pysystemtrade_20260914.parquet`: 12 currency conversion series.
- `instrument_metadata.csv`: source configuration, keyed by Instrument.
- `source_profile.csv`: per-source first/last timestamps, rows, intraday consolidation and null-cell counts.
- `manifest.json`: source hashes, transformation policy, output shape and coverage.

Futures coverage across the union of markets is 1969-12-02 through **2024-03-29**.
FX also ends on **2024-03-29**. Individual markets have different start/end dates;
consult the profile. The filename date is a requested cutoff, not evidence of
2026 observations. No new IB data was downloaded.

## Schema and compatibility

Like the reference file, futures has a date DatetimeIndex and two-level columns.
Example: `('SP500c1', 'close')`. The c1 suffix is a compatibility naming convention
for the source's active roll series, not a guarantee of first-nearby maturity.

| Field | Meaning |
|---|---|
| close | Supplied additive adjusted price, suitable for price differences. |
| raw_close | Unadjusted PRICE of the active contract. |
| contract | Source PRICE_CONTRACT as a nullable string. |
| contmth | YYYYMM integer derived from contract; not exact expiry. |
| carry / carry_contract | Source comparison price and contract identifier. |
| forward / forward_contract | Source next-roll price and contract identifier. |
| adjustment | close minus raw_close; additive, not the reference file's logarithmic shift. |

Open/high/low, volume, open interest, settlement, exact expiry, intraday TWAPs
and multiplicative shift are intentionally absent because the CSVs do not
establish them. Existing code requiring these fields needs a dedicated adapter
or additional data. Do not silently substitute close for open or settlement.

The source contains multiple observations on many dates. Each CSV is sorted,
and the last actual row on each source calendar date is retained. This avoids
mixing cells from different intraday rows or contracts. No timezone conversion,
exchange-session relabeling, forward filling or backward filling is performed.
These daily dates are research labels, not verified synchronized global closes.
Adjusted and multiple prices are reduced independently and joined by date;
their source timestamps are available in the original CSVs. Validate timestamp
alignment before using adjustment as a precise intraday roll diagnostic.

## Use in backtests

```python
import pandas as pd
root = 'C:/dev/pyktrader3/output/pysystemtrade_history/'
data = pd.read_parquet(root + 'fut_d_pysystemtrade_20260914.parquet')
fx = pd.read_parquet(root + 'fx_pysystemtrade_20260914.parquet')
meta = pd.read_csv(root + 'instrument_metadata.csv', index_col='Instrument')
sp = data['SP500c1'].dropna(subset=['close'])
point_changes = sp['close'].diff()
# For a USD instrument, gross one-contract P&L in USD:
one_contract_pnl = point_changes * meta.loc['SP500', 'Pointsize']
# Strategy P&L additionally requires lagged contract positions and costs.
```

**Do not apply pct_change() to this adjusted close as a futures return.**
Additive price levels depend on later roll adjustments and can be negative.
Several existing pyktrader3 analytics call pct_change(), so structural
compatibility alone is insufficient. Use adjusted price differences with
contract positions and point values. Portfolio returns require an explicit
capital denominator. A raw-price percentage change introduces roll jumps.
For non-USD P&L, apply the appropriate USD-per-local-currency conversion with
an explicit timestamp/staleness policy. USD instruments use a conversion of 1.

Carry should use raw PRICE and CARRY with their signed contract-month spacing;
do not assume the comparison contract is later or exactly one month away.
Exact expiry-based carry requires additional expiry metadata. Costs, historical
specification changes, tradability filters and execution timing remain part of
the backtest implementation. This export is a source snapshot, not a
point-in-time vintage database. Signals based on additive price levels need
particular care about dependence on future roll adjustments.

## Rebuild and validation

With pandas and pyarrow installed:

```powershell
& D:/miniconda3/python.exe misc_scripts/build_pysystemtrade_history.py --as-of 2026-09-14
```

The converter fails on duplicate source timestamps, invalid timestamps,
non-integral contract IDs and duplicate metadata keys. It verifies unique,
sorted output dates and exact dataframe equality after Parquet write/read.
All 252 exported instruments matched metadata. The profile counts source nulls;
these are retained, not treated as zeros. Price-series reconstruction, historical
cost accuracy and IB contract mappings have not been independently validated.

Compatibility repair: the initial PyArrow 25.0.1 exports triggered
`Repetition level histogram size mismatch` in the user's PyArrow 19.0.0.
Disabling writer statistics did not resolve it. Both files were rebuilt with
the consuming Miniconda environment (pandas 2.2.3, PyArrow 19.0.0) and passed
full reads there. Use that environment for rebuilds; no package upgrade is
required. New manifests record the writer versions. Temporary files are
validated before replacing exports.

The inspectable transformation and profiling code is
`misc_scripts/build_pysystemtrade_history.py`; hashes in the manifest identify
the precise inputs used. Source files and the domestic cache were not modified.
