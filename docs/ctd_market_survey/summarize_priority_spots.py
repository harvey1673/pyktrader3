"""Summarize the cached-value audit without filling missing observations."""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).parent / 'priority_spot_audit'


def main():
    obs = pd.read_csv(BASE / 'observations.csv', keep_default_na=False)
    inv = pd.read_csv(BASE / 'inventory.csv', keep_default_na=False)
    groups = obs.groupby(['code', 'date']).value.nunique()
    if (groups > 1).any():
        raise ValueError('Conflicting source values: resolve before producing merged profiles')
    # Exact matching numeric overlaps may be combined. This is not an as-known history.
    unique = obs.drop_duplicates(['code', 'date']).copy()
    unique['date'] = pd.to_datetime(unique.date)
    summaries, gaps = [], []
    for code, frame in unique.groupby('code'):
        frame = frame.sort_values('date')
        meta = inv[inv.code == code].iloc[0]
        role = ('basis_only' if '升贴水' in meta.description else
                'scrap_not_deliverable_coil' if '废不锈钢' in meta.description else
                'monthly_tender_not_daily_spot' if '采购价' in meta.description else
                'grade75_separate_benchmark' if meta.alias == 'sf_75_shmet' else
                'outright_spot_candidate')
        delta = frame.date.diff().dt.days
        summaries.append({'product': meta['product'], 'code': code, 'alias': meta.alias,
            'description': meta.description, 'role': role,
            'first': frame.date.min().date().isoformat(),
            'last': frame.date.max().date().isoformat(), 'observations': len(frame),
            'nonpositive_count': int((frame.value <= 0).sum()),
            'max_gap_calendar_days': int(delta.max()) if len(frame)>1 else 0,
            'sources': '; '.join(sorted(set(inv[inv.code==code].workbook + '/' + inv[inv.code==code].sheet)))})
        frame['previous_date'] = frame.date.shift()
        for _, row in frame[delta > 20].iterrows():
            gaps.append({'product': meta['product'], 'code': code, 'alias': meta.alias,
                'previous_observation': row.previous_date.date().isoformat(),
                'next_observation': row.date.date().isoformat(),
                'gap_calendar_days': (row.date-row.previous_date).days})
    pd.DataFrame(summaries).sort_values(['product','alias']).to_csv(BASE/'merged_profiles.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(gaps).to_csv(BASE/'long_gaps.csv', index=False, encoding='utf-8-sig')
    selected = ['coke_sub_a_rz','coke_sub_a_tj','ckc_stock_ganqimaodu','ckc_outstock_ganqimaodu',
        'ckc_a10v24s08_lvliang','ckc_a9v18s10_lvliang','sm_65s17_tj','sm_65s17_neimeng',
        'sf_72_shmet','sf_72_ningxia','sf_72_neimeng','sf_72_gansu','ss_304_gross_wuxi']
    panel = unique[unique.alias.isin(selected)].pivot(index='date', columns='alias', values='value').sort_index()
    panel.to_csv(BASE/'raw_benchmark_panel.csv', encoding='utf-8-sig')
    selection = json.loads((BASE.parent/'sparse_spot_selection.json').read_text(encoding='utf-8'))
    sparse_aliases = [alias for spec in selection['products'].values()
                      for role in ['core','optional'] for alias in spec[role]]
    panel[sparse_aliases].to_csv(BASE/'sparse_raw_spot_panel.csv', encoding='utf-8-sig')
    # A source-specific annual count avoids confusing weekends with missing trading sessions.
    stats = {'unique_tickers':len(summaries),'unique_code_dates':len(unique),
             'conflicting_numeric_overlaps':0,'panel_series':len(panel.columns),
             'sparse_panel_series':len(sparse_aliases),
             'panel_semantics':'Raw quoted prices; no quality/freight conversion, filling, CTD minimum or futures join.'}
    (BASE/'profile_summary.json').write_text(json.dumps(stats,indent=2), encoding='utf-8')
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
