import datetime
import pandas as pd
import json
from pathlib import Path
from pycmqlib3.utility import misc
from pycmqlib3.utility.process_wt_data import *
from pycmqlib3.utility.dataseries import load_processed_fut_by_product, rolling_fut_cont

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'ZC', 'SM', "SF", 'nr']
base_metal_mkts = ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss']
precious_metal_mkts = ['au', 'ag']
ind_metal_mkts = ferrous_products_mkts + ferrous_mixed_mkts + base_metal_mkts
petro_chem_mkts = ['l', 'pp', 'v', 'TA', 'MA', 'bu', 'sc', 'fu', 'eg', 'eb', 'lu', 'pg', 'PF']
ind_all_mkts = ind_metal_mkts + petro_chem_mkts
ags_oil_mkts = ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'b'] #, 'b']
ags_soft_mkts = ['CF', 'SR', 'jd', 'AP', 'sp', 'CJ', 'UR', 'SA', 'lh', 'PK',] # 'CY',]
ags_all_mkts = ags_oil_mkts + ags_soft_mkts
eq_fut_mkts = ['IF', 'IH', 'IC', 'IM',]
bond_fut_mkts = ['T', 'TF', 'TS']
fin_all_mkts = eq_fut_mkts + bond_fut_mkts
commod_all_mkts = ind_all_mkts + ags_all_mkts + precious_metal_mkts
all_markets = commod_all_mkts + fin_all_mkts


def main_cont_filter(df, prodcode):
    product_cont_map = {
        'ni': [datetime.date(2019, 1, 1), [1, 5, 9], ],
        'sn': [datetime.date(2020, 1, 1), [1, 5, 9]],
        'bu': [datetime.date(2018, 1, 1), [6, 9, 12]],
        'b': [datetime.date(2019, 4, 1), [1, 5, 9]],
    }

    if prodcode in base_metal_mkts + ['sc', 'nr', 'eb', 'lu', 'IF', 'IC', 'IH', 'IM', 'b', 'bu']:
        cont_list = [i for i in range(1, 13)]
    elif prodcode in ['T', 'TF', 'TS']:
        cont_list = [3, 6, 9, 12]
    elif prodcode in ['rb', 'hc', 'AP']:
        cont_list = [1, 5, 10]
    elif prodcode in ['PK']:
        cont_list = [1, 4, 10]
    elif prodcode in ['au', 'ag']:
        cont_list = [6, 12]
    elif prodcode in ['bu']:
        cont_list = [6, 12]
    else:
        cont_list = [1, 5, 9]
    if prodcode in product_cont_map:
        flag = ((df['date'] < product_cont_map[prodcode][0]) & (df['month'].isin(product_cont_map[prodcode][1])))
        flag = flag | ((df['date'] >= product_cont_map[prodcode][0]) & (df['month'].isin(cont_list)))
    else:
        flag = df['month'].isin(cont_list)

    product_starts = {
        'fu': datetime.date(2018, 9, 1),
        'b': datetime.date(2018, 1, 3),
        'nr': datetime.date(2020, 2, 1),
    }
    if prodcode in product_starts:
        flag = flag & (df['date'] >= product_starts[prodcode])
    return flag


def update_expiry_roll(start_date=datetime.date(2020, 1, 1),
                       end_date=datetime.date.today(),
                       cutoff=misc.day_shift(datetime.date.today(), '-1b', misc.CHN_Holidays),
                       roll_name='troll',
                       nb_cont=2,
                       folder="C:/dev/wtdev/config/roll",
                       skip_exists=True):
    roll_mode = 0
    roll_win = 1
    cont_thres = 0
    cont_ratio = [1.0, 1.0]
    contract_filter = main_cont_filter
    for prodcode in all_markets:
        roll_cutoff = '-30b'
        if prodcode in ['cu', 'al', 'zn', 'pb', 'sn', 'ss', ]:
            roll_cutoff = '-25b'
        elif prodcode in ['ni', 'jd', 'lh', ]:
            roll_cutoff = '-35b'
        elif prodcode in ['sc', 'eb'] + bond_fut_mkts:
            roll_cutoff = '-20b'
        elif prodcode in ['lu']:
            roll_cutoff = '-45b'
        elif prodcode in precious_metal_mkts:
            roll_cutoff = '-15b'
        elif prodcode in eq_fut_mkts:
            roll_cutoff = '-0b'

        filename = "%s/%s_%s.csv" % (folder, prodcode, roll_name)
        file = Path(filename)
        if file.exists():
            if skip_exists:
                continue
            curr_roll = pd.read_csv(file, parse_dates=['date'])
            curr_roll['date'] = curr_roll['date'].dt.date
        else:
            curr_roll = pd.DataFrame()
        roll_kwargs = {'roll_win': roll_win, 'roll_cutoff': roll_cutoff, 'cont_ratio': cont_ratio,
                       'contract_filter': contract_filter, 'min_thres': 0}
        df = load_processed_fut_by_product(prodcode,
                                           start_date=start_date,
                                           end_date=end_date,
                                           freq='d',
                                           **roll_kwargs)
        roll_map, daily_cont = rolling_fut_cont(df, nb_cont=nb_cont,
                                                cont_thres=cont_thres,
                                                roll_mode=roll_mode,
                                                curr_roll=curr_roll)
        if len(roll_map) == 0:
            print(f"empty roll map, skipping {prodcode}\n")
            continue
        roll_map = roll_map.reset_index()
        roll_map.columns = [str(col) for col in roll_map.columns]
        if len(curr_roll) > 0:
            last_roll = min(curr_roll['date'].iloc[-1], cutoff)
            roll_map = curr_roll.append(roll_map[roll_map['date'] > last_roll], ignore_index=True)
        if roll_map['1'].iloc[-1] == None:
            roll_map['1'].iloc[-1] = misc.default_next_main_contract(roll_map['0'].iloc[-1], start_date, end_date)
        flag = roll_map['1'].isna()
        roll_map['1'].loc[flag] = roll_map['0'].loc[flag.shift(1).fillna(False)].values
        roll_map = roll_map.set_index('date')
        roll_map.to_csv(filename)


def update_main_roll(start_date=datetime.date(2020, 1, 1),
                    end_date=datetime.date.today(),
                    cutoff=misc.day_shift(datetime.date.today(), '-1b', misc.CHN_Holidays),
                    roll_name='nroll',
                    folder="C:/dev/wtdev/config/roll",
                    skip_exists=True,
                    nb_cont=2,
                    cont_ratio=[1.0, 0.0],
                    min_thres=7500):
    roll_mode = 0
    roll_win = 1
    for prodcode in all_markets:
        roll_cutoff = '-20b'
        contract_filter = None
        if prodcode in ['IF', 'IC', 'IH', 'IM', 'sc', ]:
            min_thres = 0.1 * min_thres
        elif prodcode in ['j', 'ni', 'sn', 'cu', 'bc', 'lh']:
            min_thres = 0.2 * min_thres
        elif prodcode in ['ru', 'nr', 'al', 'zn', 'pb', 'jm', 'ss', 'PK', 'b', ]:
            min_thres = 0.5 * min_thres
        elif prodcode in ['SM', 'SF', 'FG', 'i', 'AP', 'eb', ]:
            min_thres = 0.8 * min_thres
        elif prodcode in ['T', 'TF', 'TS']:
            min_thres = 0
        if prodcode in ['IF', 'IC', 'IH', 'IM', ]:
            roll_cutoff = '0b'

        filename = "%s/%s_%s.csv" % (folder, prodcode, roll_name)
        file = Path(filename)
        if file.exists():
            if skip_exists:
                continue
            curr_roll = pd.read_csv(file, parse_dates=['date'])
            curr_roll['date'] = curr_roll['date'].dt.date
        else:
            curr_roll = pd.DataFrame()
        roll_kwargs = {'roll_win': roll_win, 'roll_cutoff': roll_cutoff, 'cont_ratio': cont_ratio,
                       'contract_filter': contract_filter, 'min_thres': min_thres}

        df = load_processed_fut_by_product(prodcode, start_date=start_date, end_date=end_date, freq='d', **roll_kwargs)
        cont_thres = 50_000
        roll_map, daily_cont = rolling_fut_cont(df, nb_cont=nb_cont, cont_thres=cont_thres, roll_mode=roll_mode)
        if len(roll_map) == 0:
            print(f"empty roll map, skipping {prodcode}\n")
            continue
        roll_map = roll_map.reset_index()
        roll_map.columns = [str(col) for col in roll_map.columns]
        if len(curr_roll) > 0:
            last_roll = min(curr_roll['date'].iloc[-1], cutoff)
            roll_map = curr_roll.append(roll_map[roll_map['date'] > last_roll], ignore_index=True)
        if roll_map['1'].iloc[-1] is None:
            roll_map['1'].iloc[-1] = misc.default_next_main_contract(roll_map['0'].iloc[-1], start_date, end_date)
        flag = roll_map['1'].isna()
        roll_map['1'].loc[flag] = roll_map['0'].loc[flag.shift(1).fillna(False)].values
        roll_map = roll_map.set_index('date')
        roll_map.to_csv(filename)


def generate_daily_roll(folder="C:/dev/wtdev/config/roll",
                        roll_list=[('nroll', 'nearby'), ('troll', 'main')]):
    for roll_name, outfile_prefix in roll_list:
        out_dict = {
            '0': dict([(exch, {}) for exch in ["DCE", "CZCE", "SHFE", "INE", "CFFEX"]]),
            '1': dict([(exch, {}) for exch in ["DCE", "CZCE", "SHFE", "INE", "CFFEX"]]),
        }

        key_map = {
            '0': f'{outfile_prefix}1',
            '1': f'{outfile_prefix}2',
        }
        for prodcode in all_markets:
            filename = "%s/%s_%s.csv" % (folder, prodcode, roll_name)
            exch = misc.prod2exch(prodcode)
            roll_map = pd.read_csv(filename, parse_dates=['date'])
            roll_map['date'] = roll_map['date'].dt.date
            roll_map['roll_date'] = roll_map['date'].apply(lambda x: misc.day_shift(x, '1b', misc.CHN_Holidays))
            xdf = load_fut_by_product(prodcode, exch, start_date, end_date, freq='d')
            for nb in ['0', '1']:
                roll_map[f'{nb}_prev'] = roll_map[nb].shift(1)
                for key in [nb, f'{nb}_prev']:
                    roll_map = roll_map.merge(xdf[['instID', 'date', 'close', 'volume', 'openInterest']], how='left',
                                              left_on=[key, 'date'],
                                              right_on=['instID', 'date']).drop(columns=['instID'])
                    roll_map = roll_map.rename(columns={'close': f'close_{key}',
                                                        'volume': f'vol_{key}',
                                                        'openInterest': f'oi_{key}'})
                nb_df = roll_map[['roll_date', f'{nb}_prev', nb, f'close_{nb}_prev', f'close_{nb}']]
                nb_df.columns = ['date', 'from', 'to', 'oldclose', 'newclose']
                nb_df['date'] = nb_df['date'].apply(lambda d: misc.day_shift(d, '1b', misc.CHN_Holidays))
                nb_df['date'] = nb_df['date'].astype('datetime64').dt.strftime('%Y%m%d').astype('int64')
                nb_df.fillna({'from': '', 'oldclose': 0.0}, inplace = True)
                out_dict[nb][exch][prodcode] = nb_df.to_dict('records')
        for nb in ['0', '1']:
            fname = '%s/%s.json' % (folder, key_map[nb])
            with open(fname, 'w') as ofile:
                json.dump(out_dict[nb], ofile, indent=4)


if __name__ == "__main__":
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date.today()
    folder = 'C:/dev/wtdev/config/roll'
    roll_list = [('nroll', 'nearby'), ('troll', 'main')]

    update_expiry_roll(start_date=start_date,
                       end_date=end_date,
                       cutoff=misc.day_shift(end_date, '0b', misc.CHN_Holidays),
                       roll_name='troll',
                       folder=folder, skip_exists=True)
    update_main_roll(start_date=start_date,
                    end_date=end_date,
                    cutoff=misc.day_shift(end_date, '-1b', misc.CHN_Holidays),
                     roll_name='volroll',
                     folder=folder,
                     skip_exists=False,
                     cont_ratio=[1.0, 0.0],
                     min_thres=7500)

    generate_daily_roll(folder="C:/dev/wtdev/config/roll", roll_list=roll_list)



