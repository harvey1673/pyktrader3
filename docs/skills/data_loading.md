# Skill: Data Loading

## Purpose

Load market data, factor/fundamental data, and related datasets using the project's existing helper functions. Data is sourced from two backends: **wtpy DSB binary files** (for OHLCV bar data) and a **MySQL database named `blueshale`** (for factor signals, spot/macro data, and EDB time-series). Pre-built parquet caches are available for the most common use case — continuous nearby contracts.

---

## When To Use

- Retrieving **daily or intraday OHLCV** for one or more futures products
- Loading a **continuous nearby contract** series with price adjustment (for backtesting)
- Loading **specific contract data** by exact contract code (e.g. `rb2605`)
- Retrieving **computed factor signals** (e.g. `ryield`, `basmom`, inventory) stored in `fut_fact_data`
- Retrieving **spot / macro / EDB time-series** (e.g. iFind codes) from the `edb` MySQL table

---

## Functions Used

| Function | File | Description |
|---|---|---|
| `load_hist_fut_prices()` | `misc_scripts/fun_factor_update.py` | Loads continuous nearby contracts for a list of products, pivots to MultiIndex DataFrame, saves/reads parquet cache |
| `dataseries.nearby()` | `pycmqlib3/utility/dataseries.py` | Loads nth nearby continuous contract for a single product using roll schedule from `C:/dev/wtdev/config/` JSON files |
| `dataseries.nearby_wt()` | `pycmqlib3/utility/dataseries.py` | Alternative nearby loader using calendar-based roll rules instead of JSON roll files |
| `load_hist_bars_to_df()` | `pycmqlib3/utility/process_wt_data.py` | Loads OHLCV bars for a **single specific contract** directly from a wtpy DSB binary file |
| `load_bars_by_code()` | `pycmqlib3/utility/process_wt_data.py` | Similar to above; used inside `nearby()` for each contract segment |
| `load_fut_by_product()` (wtpy) | `pycmqlib3/utility/process_wt_data.py` | Scans all DSB files for a product and returns a long-format DataFrame of all contracts |
| `load_fut_by_product()` (MySQL) | `pycmqlib3/utility/dbaccess.py` | Queries `fut_daily` or `fut_min` MySQL table; slower than DSB path |
| `load_factor_data()` | `pycmqlib3/utility/dbaccess.py` | Loads pre-computed factor values from `fut_fact_data` MySQL table |
| `load_codes_from_edb()` | `pycmqlib3/utility/dbaccess.py` | Loads spot/macro EDB time-series from the `edb` MySQL table (iFind or other sources) |
| `load_int_stock_daily()` | `pycmqlib3/utility/dbaccess.py` | Loads international equity OHLCV from `int_stock_daily` MySQL table |

---

## Inputs

### `dataseries.nearby()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | — | Product code e.g. `'rb'`, or exchange-qualified e.g. `'SHFE.rb'` |
| `n` | `int` | `1` | Nth nearby (1 = front, 2 = second, …) |
| `start_date` | `datetime.date` | 1 year ago | Start of date range |
| `end_date` | `datetime.date` | today | End of date range |
| `freq` | `str` | `'d'` | `'d'` for daily, `'m5'` for 5-min |
| `shift_mode` | `int` | `0` | `0` = no adjust, `1` = additive, `2` = multiplicative |
| `roll_name` | `str` | `'hot'` | Roll schedule name; looks up `C:/dev/wtdev/config/{roll_name}{n}.json` |

### `load_hist_bars_to_df()` (single contract)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | — | Exchange-qualified contract code e.g. `'SHFE.rb2605'` |
| `start_date` | `datetime.date` | `None` | Start of date range |
| `end_date` | `datetime.date` | `None` | End of date range |
| `freq` | `str` | `'d'` | `'d'`, `'m1'`, or `'m5'` |
| `index_col` | `str` | `'date'` | Column to use as index; use `'datetime'` for intraday |

### `load_factor_data()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `product_list` | `list[str]` | — | Product codes e.g. `['rb', 'hc', 'i']` |
| `factor_list` | `list[str]` | `[]` | Factor names e.g. `['ryield', 'basmom20']`; empty = all |
| `roll_label` | `str` | `'CAL_30b'` | Roll label used when factors were computed |
| `start` | `datetime.date` | today | Start date |
| `end` | `datetime.date` | today | End date |
| `freq` | `str` | `'d'` | Frequency |

### `load_codes_from_edb()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code_list` | `list[str]` or `str` | — | iFind or other source codes |
| `source` | `list[str]` | `['ifind']` | Data source filter |
| `start_date` | `datetime.date` | `None` | Start date |
| `end_date` | `datetime.date` | `None` | End date |
| `column_name` | `str` | `'index_name'` | Column to use as pivot column headers |

---

## Outputs

### Continuous nearby — `dataseries.nearby()`

`pd.DataFrame` with DatetimeIndex (`date`), one row per trading day.

Columns: `contract`, `open`, `high`, `low`, `close`, `volume`, `openInterest`, `diff_oi`, `shift`

```
            contract     open     high      low    close  volume  shift
date
2025-01-02  rb2503   3682.0   3720.0   3660.0   3698.0  234521    0.0
2025-01-03  rb2503   3700.0   3730.0   3685.0   3712.0  198763    0.0
```

### Multi-product parquet / `load_hist_fut_prices()`

`pd.DataFrame` with DatetimeIndex, **MultiIndex columns** `(product, field)` where product is `rbc1`, `hcc1`, etc.

```python
df.columns  # MultiIndex
# ('rbc1', 'close'), ('rbc1', 'open'), ..., ('hcc1', 'close'), ...

# Access:
df[('rbc1', 'close')]          # Series
df.xs('close', axis=1, level=1)  # DataFrame of close prices for all products
```

### Single contract — `load_hist_bars_to_df()`

`pd.DataFrame` with DatetimeIndex (`date` for daily, `datetime` for intraday).

Columns: `open`, `high`, `low`, `close`, `volume`, `openInterest`

### Factor data — `load_factor_data()`

`pd.DataFrame` in long format with columns: `product_code`, `date`, `serial_no`, `serial_key`, `fact_name`, `fact_val`

Pivot to wide format when needed:
```python
df.pivot(index='date', columns=['product_code', 'fact_name'], values='fact_val')
```

### EDB data — `load_codes_from_edb()`

`pd.DataFrame` with DatetimeIndex (`date`), columns = `index_name` values (one column per EDB code).

---

## Example Usage

### Load continuous nearby (daily, multiplicative adjust)

```python
import datetime
from pycmqlib3.utility import dataseries

df = dataseries.nearby(
    'rb',
    n=1,
    start_date=datetime.date(2020, 1, 1),
    end_date=datetime.date(2026, 3, 1),
    shift_mode=2,   # multiplicative price adjustment
    freq='d',
    roll_name='hot'
)
close = df['close']
```

### Load multiple products using parquet cache

```python
import pandas as pd
import datetime

tday = datetime.date.today()
df = pd.read_parquet(f"C:/dev/data/fut_d_{tday.strftime('%Y%m%d')}.parquet")

assets = ['rb', 'hc', 'i', 'cu', 'al']
close = pd.DataFrame({
    asset: df[(asset + 'c1', 'close')]
    for asset in assets
    if (asset + 'c1', 'close') in df.columns
}).dropna(how='all').ffill()
```

### Load a specific contract by code

```python
import datetime
from pycmqlib3.utility.process_wt_data import load_hist_bars_to_df

df = load_hist_bars_to_df(
    'SHFE.rb2605',
    start_date=datetime.date(2025, 1, 1),
    end_date=datetime.date(2025, 12, 31),
    freq='d'
)
```

### Load all contracts for a product (wide scan)

```python
import datetime
from pycmqlib3.utility.process_wt_data import load_fut_by_product

df = load_fut_by_product(
    'SHFE.rb',
    start_date=datetime.date(2024, 1, 1),
    end_date=datetime.date(2025, 12, 31),
    freq='d'
)
# Long format: columns include instID, date, open, high, low, close, ...
```

### Load pre-computed factor signals from MySQL

```python
import datetime
from pycmqlib3.utility.dbaccess import load_factor_data

df = load_factor_data(
    product_list=['rb', 'hc', 'i'],
    factor_list=['ryield'],
    roll_label='CAL_30b',
    start=datetime.date(2022, 1, 1),
    end=datetime.date(2026, 3, 1),
    freq='d'
)
# Pivot to wide:
wide = df.pivot(index='date', columns=['product_code', 'fact_name'], values='fact_val')
```

### Load spot/macro EDB time-series

```python
import datetime
from pycmqlib3.utility.dbaccess import load_codes_from_edb

spot_df = load_codes_from_edb(
    code_list=['S0031400', 'M0066571'],
    source=['ifind'],
    start_date=datetime.date(2018, 1, 1),
    end_date=datetime.date(2026, 3, 1)
)
# Returns: DatetimeIndex df, columns = index_name values
```

### Compute price changes over multiple periods

```python
import pandas as pd
import datetime

tday = datetime.date.today()
df = pd.read_parquet(f"C:/dev/data/fut_d_{tday.strftime('%Y%m%d')}.parquet")

assets = ['rb', 'hc', 'i', 'cu', 'al', 'au']
close = pd.DataFrame({
    asset: df[(asset + 'c1', 'close')]
    for asset in assets
    if (asset + 'c1', 'close') in df.columns
}).dropna(how='all').ffill()

pct_chg = pd.DataFrame({
    label: close.pct_change(n).iloc[-1]
    for label, n in {'1d': 1, '3d': 3, '5d': 5, '10d': 10}.items()
})
```

---

## Implementation Notes

- **Exchange qualification**: wtpy functions require exchange-prefixed codes like `'SHFE.rb'` or `'SHFE.rb2605'`. The mapping from product code to exchange is in `pycmqlib3/utility/misc.py` via `prod2exch()`.
- **CZCE naming**: Modern CZCE contracts (2013+) use **4-digit year** codes matching other exchanges, e.g. `RM2605`, `CF2605`. Older contracts (pre-2013) used 3-digit year, e.g. `RM1305`. Pass the full code as-is: `'CZCE.RM2605'`. `load_fut_by_product` and `wtcode_to_code()` handle normalisation internally.
- **MA / ZC aliases**: `MA` was historically called `ME` and `ZC` was called `TC`. Both `load_fut_by_product` functions handle these aliases internally.
- **DSB storage path**: `C:/dev/wtdev/storage/his/day/{EXCH}/{contract}.dsb` for daily; `C:/dev/wtdev/storage/his/min5/{EXCH}/{contract}.dsb` for 5-min.
- **Roll schedule files**: `C:/dev/wtdev/config/hot1.json`, `hot2.json`, etc. Used by `dataseries.nearby()`. If the file is missing for a product, `nearby()` returns an empty DataFrame silently.
- **Parquet cache files**: `C:/dev/data/fut_d_YYYYMMDD.parquet` for EOD data. `C:/dev/data/fut_eod_YYYYMMDD.parquet` for settlement. Generated by `load_hist_fut_prices()` / `update_db_factor()`.
- **DataFrame index**: All functions return a DatetimeIndex. For daily data the index name is `date`; for intraday it is `datetime`.
- **Price adjustment on roll**: `shift_mode=2` (multiplicative) is the standard for trend-following backtests. The `shift` column records the cumulative log-adjustment applied.
- **MySQL connection config**: Stored in `pycmqlib3/utility/dbaccess.py` as `dbconfig`. Database name is `blueshale`.
- **`ffill()` after loading**: Spot and EDB data often have gaps on non-trading days. Always `.ffill()` before aligning with futures price data.
- **`fillna` deprecation**: pandas 2.2+ removed the `method` keyword from `fillna()`. Use `.ffill()` / `.bfill()` directly.

---

## Common Patterns

### Get close prices from the cached parquet (most common)

```python
df = pd.read_parquet(f"C:/dev/data/fut_d_{tday.strftime('%Y%m%d')}.parquet")
close = df.xs('close', axis=1, level=1)          # all products, close prices
close.columns = [c[:-2] for c in close.columns]  # strip 'c1' suffix → 'rb', 'hc', ...
close = close[asset_list].dropna(how='all').ffill()
```

### Compute returns

```python
returns = close.pct_change().fillna(0)
```

### Load nearby for a single product and get daily returns

```python
df = dataseries.nearby('cu', n=1, start_date=start, end_date=end, shift_mode=2)
returns = df['close'].pct_change()
```

### Load factor from DB, pivot, and align with price index

```python
factor_raw = load_factor_data(['rb'], factor_list=['ryield'], start=start, end=end)
factor_wide = factor_raw.pivot(index='date', columns='product_code', values='fact_val')
factor_wide.index = pd.to_datetime(factor_wide.index)
factor_aligned = factor_wide.reindex(close.index).ffill()
```

---

## Related Skills

- [Signal Generation](signal_generation.md) — using loaded data to generate trading signals via `signal_store`
- [Backtest Execution](backtest_execution.md) — running `MetricsBase` with loaded prices and signals
