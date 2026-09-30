"""Create the small reproducible notebook and source receipt for this assessment."""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).parent


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell('markdown', '# Priority commodity spot audit\n\nRead-only cached Excel observations from 2016 through 2026-09-22. No source refresh, filling, futures join or CTD certification. Six core inputs plus two optional comparators are selected for a parsimonious carry-proxy study.\n'),
    cell('code', "from pathlib import Path\nimport json\nimport pandas as pd\nBASE = Path.cwd()\nif not (BASE / 'priority_spot_audit').exists():\n    BASE = BASE / 'docs' / 'ctd_market_survey'\nDATA = BASE / 'priority_spot_audit'\nprofiles = pd.read_csv(DATA / 'merged_profiles.csv')\nprofiles[['product','alias','role','first','last','observations','max_gap_calendar_days']]\n"),
    cell('code', "obs = pd.read_csv(DATA / 'observations.csv', keep_default_na=False)\nconflicts = obs.groupby(['code','date']).value.nunique()\nassert not (conflicts > 1).any()\nunique = obs.drop_duplicates(['code','date'])\nassert len(unique) == 72641  # This recorded snapshot; update if source workbooks change.\nassert unique.code.nunique() == 32\npd.read_csv(DATA / 'long_gaps.csv')\n"),
    cell('code', "selection = json.loads((BASE / 'sparse_spot_selection.json').read_text(encoding='utf-8'))\ncore = [x for v in selection['products'].values() for x in v['core']]\noptional = [x for v in selection['products'].values() for x in v['optional']]\nassert len(core) == 6 and len(optional) == 2\npanel = pd.read_csv(DATA / 'sparse_raw_spot_panel.csv', index_col='date', parse_dates=True)\nassert set(panel.columns) == set(core + optional)\nassert panel.loc[panel.index < '2018-01-02','coke_sub_a_rz'].isna().all()\nassert panel.loc[panel.index < '2019-07-15','sf_72_ningxia'].isna().all()\npanel.notna().sum().to_frame('observed_rows')\n"),
    cell('markdown', '## Re-extracting a later snapshot\n\nUse the bundled Python runtime to run `audit_priority_spots.py --source-dir C:/Users/harve/Nutstore/1/Nutstore --as-of YYYY-MM-DD`, then `summarize_priority_spots.py`. The notebook assertions above document this snapshot rather than immutable provider history. Source descriptions are in `inventory.csv`; annual counts use dated worksheet rows, not an exchange trading calendar. See `PRIORITY_MARKETS_2016_REVIEW.md` for the economic interpretation and rule-source limitations.\n')
]
notebook = {'cells':cells, 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):
    c['id'] = f'priority-audit-{i}'
(BASE/'priority_spot_audit.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')

profiles = pd.read_csv(BASE/'priority_spot_audit/merged_profiles.csv')
aliases = ['coke_sub_a_rz','ckc_a10v24s08_lvliang','ckc_stock_ganqimaodu','ss_304_gross_wuxi','sm_65s17_tj','sf_72_shmet']
rows = profiles[profiles.alias.isin(aliases)][['product','alias','first','last','observations','max_gap_calendar_days']].to_dict('records')
receipt = {'schemaVersion':1,'items':[{'id':'sparse-spot-inputs','title':'Coverage of the six core spot inputs',
    'queries':[{'id':'local-workbook-profile','source':{'label':'iFinD workbook snapshots',
        'filters':['Observation dates: 2016-01-01 through 2026-09-22','Cached finite numeric values only'],
        'caveats':['Historical values are current snapshots, not point-in-time vintages.',
                   'Price gaps are retained. These are raw benchmark inputs, not normalized CTD prices.',
                   'Maximum gap measures calendar days between observations, not missed trading sessions.']},
        'columns':[{'field':'product','label':'Market'},{'field':'alias','label':'Spot input'},
                   {'field':'first','label':'First observation'},{'field':'last','label':'Last observation'},
                   {'field':'observations','label':'Observations'},{'field':'max_gap_calendar_days','label':'Largest gap (calendar days)'}],
        'rows':rows,'reportingPeriod':'2016-01-01 through 2026-09-22',
        'preview':{'kind':'partial','note':'Six selected core series from the 32-series profile.','totalRows':32},
        'methods':[{'language':'python','code':"conflicts = obs.groupby(['code', 'date']).value.nunique()\nassert not (conflicts > 1).any()\nunique = obs.drop_duplicates(['code', 'date'])\n# Per code: sort dates; report min/max, observation count and maximum date difference."}]}]}]}
(BASE/'priority_sources_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('Wrote notebook and reviewed source receipt payload')
