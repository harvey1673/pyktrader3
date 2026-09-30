# iFinD EDB API Notes (Python)

This note summarizes EDB-related iFinD API commands and gives practical
examples for this repository.

Official example page used for reference:
https://quantapi.10jqka.com.cn/gwstatic/static/ds_web/quantapi-web/example.html

## 1. Login

```python
import iFinDPy as ifind

ret = ifind.THS_iFinDLogin("your_user", "your_password")
if ret not in (0, -201):
    raise RuntimeError(f"iFinD login failed: {ret}")
```

`0` and `-201` are treated as login success in official examples.

## 2. EDB Commands

### 2.1 THS_EDBQuery

Signature from installed `iFinDPy.py`:

```python
THS_EDBQuery(indicators, begintime, endtime, outflag=False)
```

Command meaning:
- `indicators`: EDB indicator ID(s), comma-separated string
- `begintime`: start date (`YYYY-MM-DD`)
- `endtime`: end date (`YYYY-MM-DD`)
- returns a dict-like JSON payload when `outflag=False`

Example from this repository notebook:

```python
ths_data = ifind.THS_EDBQuery("S002837338", "2021-01-01", "2026-04-24")
print(ths_data)
```

Reference command style from official examples (C#/Matlab docs on same page):

```text
THS_EDBQuery("M001620247", "2020-01-01", "2020-12-31")
```

### 2.2 THS_EDB

Signature from installed `iFinDPy.py`:

```python
THS_EDB(indicators, param, begintime, endtime, format="format:dataframe")
```

Command meaning:
- `indicators`: EDB indicator ID(s), comma-separated string
- `param`: optional parameter string (empty string is common)
- `begintime`: start date (`YYYY-MM-DD`)
- `endtime`: end date (`YYYY-MM-DD`)
- `format`: default is `format:dataframe`, returning `THSData`
  object with `.data` as pandas DataFrame

Example from this repository notebook:

```python
ths_df = ifind.THS_EDB("S016936188", "", "2025-04-25", "2026-04-25")
print(ths_df.data)
```

## 3. Which One Should You Use?

- Use `THS_EDB` when you want a stable `THSData` result object and direct
  DataFrame output (`result.data`).
- Use `THS_EDBQuery` when you want the raw payload and full control over
  custom parsing.

## 4. New Utility Function in This Repo

File:
- `pycmqlib3/utility/ifind_utils.py`

Function:

```python
read_edb_data(indicators, start_date, end_date, param="", method="THS_EDB")
```

Usage examples:

```python
from pycmqlib3.utility.ifind_utils import read_edb_data

# Default path: THS_EDB -> DataFrame
df = read_edb_data("S016936188", "2025-04-25", "2026-04-25")

# Multiple indicators
df_multi = read_edb_data(
    ["S016936188", "S002837338"],
    "2025-01-01",
    "2026-01-01",
)

# Raw-query path, then normalized to DataFrame
df_q = read_edb_data(
    "S002837338",
    "2021-01-01",
    "2026-04-24",
    method="THS_EDBQuery",
)
```

## 5. Optional Follow-Up Command

You can inspect account usage (includes EDB quota bucket) with:

```python
stats = ifind.THS_DataStatistics()
print(stats)
```
