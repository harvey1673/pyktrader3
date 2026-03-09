import json

with open(r'c:\dev\pyktrader3\bktest\bktest_prod_daily_run.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Find cells with strategy execution keywords
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and cell.get('source'):
        source = ''.join(cell['source'])

        # Look for strategy execution patterns
        if any(kw in source for kw in ['strat_group', 'sig_name, wgt', 'custom_funda_signal', 'MetricsBase']):
            if len(source) > 50 and len(source) < 5000:  # Reasonable size
                print(f"\n{'='*70}")
                print(f"Cell {i}:")
                print('='*70)
                print(source[:2000])  # Limit output
