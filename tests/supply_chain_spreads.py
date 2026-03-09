"""
Supply Chain Spread Analysis
=============================
Expresses physical supply chain relationships in Chinese futures as
price spreads/ratios. For each spread, shows:
  - Time series plot
  - Rolling z-score (mean-reversion signal)
  - Summary stats: mean, std, current value, current z-score

Physical conversion ratios used:
  - 1 ton rebar  ← ~1.6t iron ore + ~0.5t coke
  - 1 ton coke   ← ~1.5t coking coal
  - 1 ton soybean → ~0.785t meal + ~0.165t soy oil
  - 1 ton PTA    ← ~0.655t PX (paraxylene)
  - 1 ton aluminum ← ~2.0t alumina
  - 1 ton glass  ← ~0.20t soda ash (+ silica sand, not traded)

Usage:
    python tests/supply_chain_spreads.py [YYYYMMDD]
"""

import sys
import os
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

sys.path.insert(0, r'c:\dev\pyktrader3')
sys.path.insert(0, r'c:\dev\wtpy')

OUT_DIR   = r'C:\dev\data\spreads'
ZSCORE_WIN = 252   # rolling window for z-score normalisation


# ---------------------------------------------------------------------------
# Spread definitions  (all prices in CNY/ton unless noted)
# ---------------------------------------------------------------------------

SPREADS = {
    # ---- Ferrous / Steel making chain ----
    # Input cost basis: coking coal price per ton of coke produced
    'coke_margin': {
        'desc':      'Coke margin: Coke − 1.5× Coking Coal',
        'sector':    'ferrous',
        'assets':    ['j', 'jm'],
        'formula':   lambda p: p['j'] - 1.5 * p['jm'],
        'normalize': lambda p: 1.5 * p['jm'],   # input cost per ton of coke
        'unit':      'CNY/t coke',
    },
    'bf_margin': {
        'desc':      'Blast-furnace steel margin: Rebar − 1.6× Iron Ore − 0.5× Coke',
        'sector':    'ferrous',
        'assets':    ['rb', 'i', 'j'],
        'formula':   lambda p: p['rb'] - 1.6 * p['i'] - 0.5 * p['j'],
        'normalize': lambda p: 1.6 * p['i'] + 0.5 * p['j'],
        'unit':      'CNY/t rebar',
    },
    'bf_full_margin': {
        'desc':      'Full BF margin (incl. coal): Rebar − 1.6× Iron Ore − 0.6× Coking Coal',
        'sector':    'ferrous',
        'assets':    ['rb', 'i', 'jm'],
        'formula':   lambda p: p['rb'] - 1.6 * p['i'] - 0.6 * p['jm'],
        'normalize': lambda p: 1.6 * p['i'] + 0.6 * p['jm'],
        'unit':      'CNY/t rebar',
    },
    'ore_coke_ratio': {
        'desc':      'Iron ore / Coke price ratio (raw material balance)',
        'sector':    'ferrous',
        'assets':    ['i', 'j'],
        'formula':   lambda p: p['i'] / p['j'],
        'normalize': None,
        'unit':      'ratio',
    },
    'rb_hc_spread': {
        'desc':      'Rebar − Hot-rolled coil spread (product mix premium)',
        'sector':    'ferrous',
        'assets':    ['rb', 'hc'],
        'formula':   lambda p: p['rb'] - p['hc'],
        'normalize': lambda p: p['hc'],   # express as % premium of rb over hc
        'unit':      'CNY/t',
    },

    # ---- Soybean crushing chain ----
    # 1 ton soybeans → 0.785t meal + 0.165t oil (crush yield)
    'soy_crush': {
        'desc':      'Soy crush margin: 0.785× Meal + 0.165× Soy Oil − Soybean',
        'sector':    'agri_crush',
        'assets':    ['m', 'y', 'a'],
        'formula':   lambda p: 0.785 * p['m'] + 0.165 * p['y'] - p['a'],
        'normalize': lambda p: p['a'],   # per ton of soybean input
        'unit':      'CNY/t soybean',
    },
    'rape_crush': {
        'desc':      'Rapeseed crush margin: 0.36× Rapeseed Oil + 0.60× Rapeseed Meal − Rapeseed',
        'sector':    'agri_crush',
        'assets':    ['OI', 'RM', 'b'],
        'formula':   lambda p: 0.36 * p['OI'] + 0.60 * p['RM'] - p['b'],
        'normalize': lambda p: p['b'],
        'unit':      'CNY/t rapeseed',
    },
    'meal_oil_ratio': {
        'desc':      'Soybean meal / soy oil price ratio (protein vs fat premium)',
        'sector':    'agri_crush',
        'assets':    ['m', 'y'],
        'formula':   lambda p: p['m'] / p['y'],
        'normalize': None,
        'unit':      'ratio',
    },

    # ---- Vegetable oil complex (substitution margins) ----
    'soy_palm_spread': {
        'desc':      'Soy oil − Palm oil spread (quality/substitution premium)',
        'sector':    'veg_oil',
        'assets':    ['y', 'p'],
        'formula':   lambda p: p['y'] - p['p'],
        'normalize': lambda p: p['p'],
        'unit':      'CNY/t',
    },
    'soy_rape_spread': {
        'desc':      'Soy oil − Rapeseed oil spread',
        'sector':    'veg_oil',
        'assets':    ['y', 'OI'],
        'formula':   lambda p: p['y'] - p['OI'],
        'normalize': lambda p: p['OI'],
        'unit':      'CNY/t',
    },
    'palm_rape_spread': {
        'desc':      'Palm oil − Rapeseed oil spread',
        'sector':    'veg_oil',
        'assets':    ['p', 'OI'],
        'formula':   lambda p: p['p'] - p['OI'],
        'normalize': lambda p: p['OI'],
        'unit':      'CNY/t',
    },
    'soy_rape_meal_spread': {
        'desc':      'Soybean meal − Rapeseed meal spread (protein substitution)',
        'sector':    'veg_oil',
        'assets':    ['m', 'RM'],
        'formula':   lambda p: p['m'] - p['RM'],
        'normalize': lambda p: p['RM'],
        'unit':      'CNY/t',
    },

    # ---- Petrochemical chain ----
    # PX → PTA: 0.655t PX per ton of PTA
    'pta_px_margin': {
        'desc':      'PTA processing margin: PTA − 0.655× PX',
        'sector':    'petrochem',
        'assets':    ['TA', 'PX'],
        'formula':   lambda p: p['TA'] - 0.655 * p['PX'],
        'normalize': lambda p: 0.655 * p['PX'],
        'unit':      'CNY/t PTA',
    },
    # PTA → Polyester fiber
    'polyester_margin': {
        'desc':      'Polyester fiber margin: PF − PTA',
        'sector':    'petrochem',
        'assets':    ['PF', 'TA'],
        'formula':   lambda p: p['PF'] - p['TA'],
        'normalize': lambda p: p['TA'],
        'unit':      'CNY/t PF',
    },
    # Crude → Fuel oil crack
    'fuel_crude_crack': {
        'desc':      'Fuel oil crack: Fuel Oil − Crude Oil',
        'sector':    'petrochem',
        'assets':    ['fu', 'sc'],
        'formula':   lambda p: p['fu'] - p['sc'],
        'normalize': lambda p: p['sc'],
        'unit':      'CNY/t',
    },
    'lo_hi_sulfur': {
        'desc':      'Low-sulfur premium: Low Sulfur Fuel Oil − High Sulfur Fuel Oil',
        'sector':    'petrochem',
        'assets':    ['lu', 'fu'],
        'formula':   lambda p: p['lu'] - p['fu'],
        'normalize': lambda p: p['fu'],
        'unit':      'CNY/t',
    },
    'pp_l_spread': {
        'desc':      'PP − LLDPE spread (propylene vs ethylene feedstock premium)',
        'sector':    'petrochem',
        'assets':    ['pp', 'l'],
        'formula':   lambda p: p['pp'] - p['l'],
        'normalize': lambda p: p['l'],
        'unit':      'CNY/t',
    },

    # ---- Metals chain ----
    # Alumina → Aluminum: 2.0t alumina per ton of aluminum
    'al_ao_margin': {
        'desc':      'Aluminum smelting margin: Aluminum − 2.0× Alumina',
        'sector':    'metals',
        'assets':    ['al', 'ao'],
        'formula':   lambda p: p['al'] - 2.0 * p['ao'],
        'normalize': lambda p: 2.0 * p['ao'],
        'unit':      'CNY/t aluminum',
    },
    'cu_al_spread': {
        'desc':      'Copper − Aluminum spread (relative industrial metal premium)',
        'sector':    'metals',
        'assets':    ['cu', 'al'],
        'formula':   lambda p: p['cu'] - p['al'],
        'normalize': lambda p: p['al'],
        'unit':      'CNY/t',
    },
    'au_ag_ratio': {
        'desc':      'Gold / Silver price ratio',
        'sector':    'metals',
        'assets':    ['au', 'ag'],
        'formula':   lambda p: p['au'] / (p['ag'] / 1000),
        'normalize': None,
        'unit':      'ratio',
    },

    # ---- Building materials ----
    # Soda ash is ~20% of glass input cost by weight
    'glass_sodaash_margin': {
        'desc':      'Glass margin: Glass − 0.20× Soda Ash',
        'sector':    'building',
        'assets':    ['FG', 'SA'],
        'formula':   lambda p: p['FG'] - 0.20 * p['SA'],
        'normalize': lambda p: 0.20 * p['SA'],
        'unit':      'CNY/t glass',
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices(tday: datetime.date,
                start_date: datetime.date = datetime.date(2015, 1, 1)) -> pd.DataFrame:
    """Load c1 close prices from parquet, return wide DataFrame (asset × date)."""
    parquet = f'C:/dev/data/fut_d_{tday.strftime("%Y%m%d")}.parquet'
    df = pd.read_parquet(parquet)

    c1_cols = [c for c in df.columns.get_level_values(0).unique() if c.endswith('c1')]
    close = (df.loc[:, df.columns.get_level_values(0).isin(c1_cols) &
                    (df.columns.get_level_values(1) == 'close')]
               .droplevel(1, axis=1))
    close.columns = [c[:-2] for c in close.columns]
    close = close.loc[pd.to_datetime(start_date):pd.to_datetime(tday)]
    return close.ffill()


# ---------------------------------------------------------------------------
# Spread calculation
# ---------------------------------------------------------------------------

def compute_spreads(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all defined spread time series.
    Returns (spreads_abs, spreads_rate) where spreads_rate is margin / input_cost * 100.
    Ratio-type spreads (normalize=None) are returned as-is in both DataFrames.
    """
    abs_results  = {}
    rate_results = {}
    for name, spec in SPREADS.items():
        assets = spec['assets']
        missing = [a for a in assets if a not in prices.columns]
        if missing:
            print(f'  Skipping {name}: missing {missing}')
            continue
        try:
            spread = spec['formula'](prices)
            abs_results[name] = spread
            if spec['normalize'] is not None:
                rate_results[name] = spread / spec['normalize'](prices) * 100
            else:
                rate_results[name] = spread   # ratio: keep as-is
        except Exception as e:
            print(f'  Error computing {name}: {e}')
    return pd.DataFrame(abs_results), pd.DataFrame(rate_results)


# ---------------------------------------------------------------------------
# Analysis & plotting
# ---------------------------------------------------------------------------

def spread_stats(spreads_abs: pd.DataFrame, spreads_rate: pd.DataFrame,
                 zscore_win: int = ZSCORE_WIN) -> pd.DataFrame:
    """Compute summary stats + rolling z-score for each spread (absolute and margin rate)."""
    rows = []
    for col in spreads_abs.columns:
        s     = spreads_abs[col].dropna()
        s_pct = spreads_rate[col].dropna()
        if len(s) < 30:
            continue
        is_ratio = SPREADS[col]['normalize'] is None

        # Z-score on the margin rate (% terms) for comparable signal strength
        roll_mean = s_pct.rolling(zscore_win, min_periods=60).mean()
        roll_std  = s_pct.rolling(zscore_win, min_periods=60).std()
        zscore    = (s_pct - roll_mean) / roll_std

        rows.append({
            'spread':            col,
            'sector':            SPREADS[col]['sector'],
            'unit':              SPREADS[col]['unit'],
            'desc':              SPREADS[col]['desc'],
            'margin_abs':        round(s.iloc[-1], 2),
            'margin_rate_%':     None if is_ratio else round(s_pct.iloc[-1], 2),
            'mean_rate_%':       None if is_ratio else round(s_pct.mean(), 2),
            'std_rate_%':        None if is_ratio else round(s_pct.std(), 2),
            'mean_rate_1y_%':    None if (is_ratio or len(s_pct) < 252) else round(s_pct.iloc[-252:].mean(), 2),
            'zscore_current':    round(zscore.iloc[-1], 2) if not np.isnan(zscore.iloc[-1]) else None,
            'pct_rank':          round((s_pct < s_pct.iloc[-1]).mean() * 100, 1),
            'n_days':            len(s),
        })

    return pd.DataFrame(rows).set_index('spread')


def plot_spread(name: str, s_abs: pd.Series, s_rate: pd.Series,
                zscore_win: int = ZSCORE_WIN, out_dir: str = OUT_DIR):
    """Plot spread: top=absolute level, middle=margin rate %, bottom=z-score."""
    is_ratio = SPREADS[name]['normalize'] is None

    roll_mean = s_rate.rolling(zscore_win, min_periods=60).mean()
    roll_std  = s_rate.rolling(zscore_win, min_periods=60).std()
    zscore    = (s_rate - roll_mean) / roll_std

    n_rows = 2 if is_ratio else 3
    fig = plt.figure(figsize=(14, 3 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 1,
                            height_ratios=[2, 1.5, 1][:n_rows], hspace=0.08)

    # Panel 1: absolute spread
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(s_abs.index, s_abs.values, color='steelblue', linewidth=1)
    ax1.set_title(f'{name}  |  {SPREADS[name]["desc"]}', fontsize=11)
    ax1.set_ylabel(SPREADS[name]['unit'])
    ax1.axhline(s_abs.iloc[-1], color='red', linewidth=0.7, linestyle=':')
    ax1.set_xticklabels([])

    if not is_ratio:
        # Panel 2: margin rate %
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(s_rate.index, s_rate.values, color='darkorange', linewidth=1)
        ax2.plot(s_rate.index, roll_mean.values, color='grey', linewidth=0.8,
                 linestyle='--', label=f'{zscore_win}d mean')
        ax2.fill_between(s_rate.index,
                         (roll_mean - roll_std).values,
                         (roll_mean + roll_std).values,
                         alpha=0.15, color='grey')
        ax2.axhline(s_rate.iloc[-1], color='red', linewidth=0.7, linestyle=':')
        ax2.set_ylabel('Margin rate (%)')
        ax2.legend(fontsize=7)
        ax2.set_xticklabels([])
        ax_z = fig.add_subplot(gs[2], sharex=ax1)
    else:
        ax_z = fig.add_subplot(gs[1], sharex=ax1)

    # Last panel: z-score
    ax_z.plot(zscore.index, zscore.values, color='darkgreen', linewidth=1)
    ax_z.axhline(0,  color='black', linewidth=0.5)
    ax_z.axhline(2,  color='red',   linewidth=0.5, linestyle='--')
    ax_z.axhline(-2, color='red',   linewidth=0.5, linestyle='--')
    ax_z.fill_between(zscore.index, zscore.values, 0,
                      where=(zscore > 2),  color='red',   alpha=0.3)
    ax_z.fill_between(zscore.index, zscore.values, 0,
                      where=(zscore < -2), color='green', alpha=0.3)
    label = 'Z-score (ratio)' if is_ratio else 'Z-score (rate %)'
    ax_z.set_ylabel(label)
    ax_z.set_ylim(-4, 4)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=120)
    plt.close(fig)


def plot_sector_overview(spreads: pd.DataFrame, stats: pd.DataFrame,
                         out_dir: str = OUT_DIR):
    """Bar chart of current z-scores for all spreads, coloured by sector."""
    valid = stats[stats['zscore_current'].notna()].copy()
    valid = valid.sort_values('sector')

    sectors = valid['sector'].unique()
    sector_colors = {s: c for s, c in zip(sectors,
        ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628'])}
    colors = [sector_colors[s] for s in valid['sector']]

    fig, ax = plt.subplots(figsize=(max(14, len(valid)), 6))
    bars = ax.bar(range(len(valid)), valid['zscore_current'], color=colors)
    ax.axhline(0,  color='black', linewidth=0.8)
    ax.axhline(2,  color='red',   linewidth=0.7, linestyle='--', alpha=0.7)
    ax.axhline(-2, color='red',   linewidth=0.7, linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Z-score (rolling {ZSCORE_WIN}d)')
    ax.set_title('Supply Chain Spread Z-scores — Current Snapshot')

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=s) for s, c in sector_colors.items()
                      if s in valid['sector'].values]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'spread_zscore_overview.png'), dpi=130)
    plt.close(fig)
    print('Z-score overview saved')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_date: datetime.date):
    print(f'\nLoading prices up to {run_date} ...')
    prices = load_prices(run_date)

    print('Computing spreads ...')
    spreads_abs, spreads_rate = compute_spreads(prices)

    print(f'\nComputed {len(spreads_abs.columns)} spreads. Calculating stats ...')
    stats = spread_stats(spreads_abs, spreads_rate)

    # Print summary table
    print('\n' + '='*120)
    print('Supply Chain Processing Margin Summary')
    print('='*120)
    display_cols = ['sector', 'unit', 'margin_abs', 'margin_rate_%',
                    'mean_rate_%', 'std_rate_%', 'zscore_current', 'pct_rank', 'n_days']
    print(stats[display_cols].to_string())

    # Flag extreme z-scores
    extreme = stats[stats['zscore_current'].abs() > 1.5].sort_values('zscore_current')
    if len(extreme):
        print('\n--- Margins at extremes (|z| > 1.5) ---')
        print(extreme[['sector', 'desc', 'margin_abs', 'margin_rate_%',
                        'zscore_current', 'pct_rank']].to_string())

    # Plot each spread
    print('\nPlotting individual spreads ...')
    for name in spreads_abs.columns:
        plot_spread(name, spreads_abs[name].dropna(), spreads_rate[name].dropna())

    # Overview z-score chart
    plot_sector_overview(spreads_rate, stats)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    stats.to_csv(os.path.join(OUT_DIR, 'spread_stats.csv'))
    spreads_abs.to_csv(os.path.join(OUT_DIR, 'spread_timeseries_abs.csv'))
    spreads_rate.to_csv(os.path.join(OUT_DIR, 'spread_timeseries_rate.csv'))
    print(f'\nOutputs saved to {OUT_DIR}')

    return spreads_abs, spreads_rate, stats


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d').date()
    else:
        run_date = datetime.date.today()

    spreads_abs, spreads_rate, stats = main(run_date)
