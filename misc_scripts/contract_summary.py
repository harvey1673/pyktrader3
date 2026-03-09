"""
Contract Summary Table
======================
For a given run_date, produce a combined table for c1/c2 contracts with:
  - MultiIndex: (product, contract)
  - pct change (%) over 1d, 2d, 5d, 10d, 20d
  - 20-day annualised vol
  - 1-std daily PnL move for 1 lot (20d window): std(price_diff) * volscale
  - individual OI per contract
  - OI change (c1+c2 combined) over 1d, 2d, 5d, 10d, 20d

Usage:
    python misc_scripts/contract_summary.py [YYYYMMDD]

If no date argument, uses today.
"""

import sys
import json
import datetime
import pandas as pd

if __name__ == '__main__':
    sys.path.insert(0, r'c:\dev\pyktrader3')
    sys.path.insert(0, r'c:\dev\wtpy')

from pycmqlib3.utility.sec_bits import EMAIL_QQ, NOTIFIERS, LOCAL_PC_NAME, EMAIL_NOTIFY
from pycmqlib3.utility.email_tool import send_html_by_smtp

from pycmqlib3.utility.misc import prod2exch
from pycmqlib3.utility.process_wt_data import load_hist_bars_to_df


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOTMAP  = r'C:\dev\wtdev\deploy\hotpicker\hotmap.json'
SECMAP  = r'C:\dev\wtdev\deploy\hotpicker\secmap.json'
COMMODITIES_JSON = r'C:\dev\wtdev\common\commodities.json'

ASSET_LIST = [
    'rb', 'hc', 'i', 'j', 'jm', 'FG', 'SA', 'SH', 'v',
    'SM', 'SF', 'ru', 'nr', 'br',
    'cu', 'al', 'zn', 'ni', 'pb', 'sn', 'ss', 'ao',
    'au', 'ag', 'bc', 'lc', 'si', 'ps', 'sp', 'ec',
    'l', 'pp', 'TA', 'MA', 'sc', 'eb', 'eg', 'pg',
    'UR', 'lu', 'bu', 'fu', 'PX', 'PF',
    'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'b',
    'CF', 'jd', 'AP', 'lh', 'CJ', 'PK',
    'T', 'TF', 'TL',
]

PERIODS     = {'1d': 1, '2d': 2, '5d': 5, '10d': 10, '20d': 20}
STD_WIN     = 20
LOOKBACK_DAYS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_contract_map(path: str) -> dict:
    raw = json.load(open(path, encoding='utf-8'))
    result = {}
    for exch_dict in raw.values():
        result.update(exch_dict)
    return result


def load_volscale_map(path: str) -> dict:
    raw = json.load(open(path, encoding='utf-8'))
    result = {}
    for exch_dict in raw.values():
        result.update(exch_dict)
    return result


def get_volscale(prod: str, volscale_map: dict):
    info = volscale_map.get(prod) or volscale_map.get(prod.lower()) or volscale_map.get(prod.upper())
    return info.get('volscale') if isinstance(info, dict) else None


def load_bars(contract: str, end_date: datetime.date, n_days: int) -> pd.DataFrame:
    start_date = end_date - datetime.timedelta(days=n_days * 2)
    exch = prod2exch(contract.rstrip('0123456789'))
    df = load_hist_bars_to_df(f'{exch}.{contract}', start_date=start_date, end_date=end_date, freq='d')
    return df if df is not None and len(df) > 0 else pd.DataFrame()


def contract_stats(contract: str, end_date: datetime.date, volscale_map: dict) -> dict:
    """Return price changes, vol and OI for one contract."""
    df = load_bars(contract, end_date, LOOKBACK_DAYS)
    row = {}

    if df.empty:
        for p in PERIODS:
            row[f'pct_{p}'] = None
        row['vol_20d'] = None
        row['risk'] = None
        row['oi']      = None
        return row

    close = df['close'].dropna()
    oi    = df['openInterest'].dropna()
    pct   = close.pct_change()
    prod  = contract.rstrip('0123456789')
    volscale = get_volscale(prod, volscale_map)

    # Pct price changes
    for label, n in PERIODS.items():
        row[f'pct_{label}'] = round((close.iloc[-1] / close.iloc[-1 - n] - 1) * 100, 2) if len(close) > n else None

    # 20-day annualised vol (%)
    row['vol_20d'] = round(pct.iloc[-STD_WIN:].std() * (252**0.5) * 100, 2) if len(pct.dropna()) >= STD_WIN else None
    price_diff = close.diff().dropna()
    row['risk'] = (
        round(price_diff.iloc[-STD_WIN:].std() * volscale, 2)
        if (len(price_diff) >= STD_WIN and volscale is not None)
        else None
    )

    # Individual OI
    row['oi'] = int(oi.iloc[-1]) if len(oi) else None

    # OI changes (raw, for combining later)
    row['_oi_series'] = oi  # kept temporarily for combined OI chg

    return row


def build_summary(asset_list, hot_map, sec_map, run_date, volscale_map) -> pd.DataFrame:
    rows = []
    total = len(asset_list)

    for idx, asset in enumerate(asset_list):
        c1_contract = hot_map.get(asset)
        c2_contract = sec_map.get(asset)
        print(f"  [{idx+1}/{total}] {asset}: c1={c1_contract}  c2={c2_contract}")

        c1_stats = contract_stats(c1_contract, run_date, volscale_map) if c1_contract else {}
        c2_stats = contract_stats(c2_contract, run_date, volscale_map) if c2_contract else {}

        # Combined OI change (c1 + c2)
        c1_oi = c1_stats.pop('_oi_series', pd.Series(dtype=float))
        c2_oi = c2_stats.pop('_oi_series', pd.Series(dtype=float))
        combined_oi = c1_oi.add(c2_oi, fill_value=0)

        def oi_chg(n):
            if len(combined_oi) > n and combined_oi.iloc[-1 - n] != 0:
                return round((combined_oi.iloc[-1] / combined_oi.iloc[-1 - n] - 1) * 100, 2)
            return None

        for contract, stats, label in [(c1_contract, c1_stats, 'c1'), (c2_contract, c2_stats, 'c2')]:
            if not contract:
                continue
            row = {
                'product':  asset,
                'contract': contract,
                'label':    label,
            }
            row.update({k: v for k, v in stats.items()})
            for p_label, n in PERIODS.items():
                row[f'oi_chg_{p_label}'] = oi_chg(n) if label == 'c1' else None
            rows.append(row)

    df = pd.DataFrame(rows).set_index(['product', 'contract'])
    df = df.drop(columns=['label'])

    # Reorder columns
    pct_cols    = [f'pct_{p}'    for p in PERIODS]
    oi_chg_cols = [f'oi_chg_{p}' for p in PERIODS]
    df = df[pct_cols + ['vol_20d', 'risk', 'oi'] + oi_chg_cols]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_date: datetime.date, email_notify: bool = EMAIL_NOTIFY):
    hot_map = load_contract_map(HOTMAP)
    sec_map = load_contract_map(SECMAP)
    volscale_map = load_volscale_map(COMMODITIES_JSON)

    print(f"\nBuilding contract summary for {run_date} ...")
    summary = build_summary(ASSET_LIST, hot_map, sec_map, run_date, volscale_map)

    print(f"\n{'='*110}")
    print(f"Contract Summary — {run_date}")
    print(f"{'='*110}")
    print(summary.to_string())

    if email_notify:
        sub = '%s contract summary <%s>' % (LOCAL_PC_NAME, run_date.strftime('%Y.%m.%d'))
        html = (
            "<html><head><style>"
            "table { border-collapse: collapse; font-size: 12px; }"
            "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }"
            "th { background-color: #f2f2f2; }"
            "</style></head><body>"
            f"<p>Contract Summary for <b>{run_date}</b>:</p>"
            + summary.to_html()
            + "</body></html>"
        )
        send_html_by_smtp(EMAIL_QQ, NOTIFIERS, sub, html)
        print("Email sent.")

    return summary


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d').date()
    else:
        run_date = datetime.date.today()

    main(run_date, email_notify=EMAIL_NOTIFY)
