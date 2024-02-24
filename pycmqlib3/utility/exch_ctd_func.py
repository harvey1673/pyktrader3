import datetime
import pandas as pd
import numpy as np

io_spec_dict = {
    # band: [fe, si, al, p, s, moisure]
    'iocj': [65, 2, 1.37, 0.06, 0.01, 8.5],
    'brbf': [62.5, 5.2, 1.8, 0.07, 0.0, 8],
    'ssf': [56.5, 6.2, 3, 0.05, 0.045, 9],
    'pbf': [61.5, 3.57, 2.31, 0.09, 0.02, 8.5],
    'macf': [60.8, 4.4, 2.3, 0.08, 0.02, 7.5],
    'jmb': [60.5, 4.5, 2.9, 0.12, 0.02, 8.2],
    'nmf': [62.7, 4.11, 2.2, 0.1, 0.06, 7.8],
}

io_brand_dict = {
    'pbf': 'pbf_qd',
    'nmf': 'nmf_qd',
    'macf': 'macf_qd',
    'jmb': 'jmb_qd',
    'ssf': 'ssf_qd',
    'brbf': 'brbf_qd',
    'iocj': 'iocj_qd'
}


def io_brand_adj(brand_name):
    if brand_name not in io_spec_dict:
        return pd.Series()
    fe, si, al, p, s, m = io_spec_dict[brand_name]
    prem_df = pd.Series(np.nan,
                        index=pd.date_range(start='20131014', end=datetime.date.today(), freq='D'),
                        name=f'{brand_name}_adj')
    if (fe <= 65) and (fe >= 60) and (al + si <= 10) and (p <= 0.15) and (s <= 0.2):
        fe_adj = min(max(fe - 62, 0), 3) * 10 + min(fe - 62, 0) * 10 * 1.5
        si_adj = -max(si - 4.0, 0) * 10
        al_adj = -max(al - 2.5, 0) * 10
        p_adj = -max(p - 0.07, 0) * 100 - max(p - 0.1, 0) * 200
        s_adj = -max(s - 0.05, 0) * 100
        prem_df[:'20180815'] = fe_adj + si_adj + al_adj + p_adj + s_adj

    if (fe >= 60) and (al + si <= 8.5) and (si <= 6.5) and (al <= 3.5) and (p <= 0.15) and (s <= 0.2):
        fe_adj = min(max(fe - 62, 0), 3) * 10 + min(fe - 62, 0) * 10 * 1.5
        si_adj = -max(si - 4.0, 0) * 10 - min(max(si - 4.5, 0), 2) * 10
        al_adj = -max(al - 2.5, 0) * 15 - min(max(al - 3, 0), 0.5) * 15
        p_adj = -max(p - 0.07, 0) * 100 - max(p - 0.1, 0) * 200
        s_adj = -max(s - 0.03, 0) * 100
        prem_df['20180815':'20200815'] = fe_adj + si_adj + al_adj + p_adj + s_adj

    fe_adj = (fe - 62) * 10
    si_adj = -max(si - 5.0, 0) * 10
    al_adj = -max(al - 2.5, 0) * 10
    p_adj = -max(p - 1.0, 0) * 500
    s_adj = -max(s - 0.03, 0) * 100
    if brand_name in ['macf', 'rhf']:
        brand_prem = -20
    elif brand_name == 'jmb':
        brand_prem = -25
    elif brand_name == 'brbf':
        brand_prem = 20
    elif brand_name == 'ssf':
        brand_prem = -90
    elif brand_name == 'iocj':
        brand_prem = 35
    else:
        brand_prem = 0
    prem_df['20200815':'20220118'] = fe_adj + si_adj + al_adj + p_adj + s_adj + brand_prem

    if brand_name in ['macf', 'rhf']:
        brand_prem = -30
    elif brand_name == 'jmb':
        brand_prem = -35
    elif brand_name == 'brbf':
        brand_prem = 20
    elif brand_name == 'ssf':
        brand_prem = -80
    elif brand_name == 'iocj':
        brand_prem = 90
    else:
        brand_prem = 0
    prem_df['20220118':'20220419'] = fe_adj + si_adj + al_adj + p_adj + s_adj + brand_prem

    if (fe >= 56) and (si <= 8.5) and (al <= 3.5) and (p <= 0.15) and (s <= 0.2):
        X = 1.5
        fe_adj = (fe - 61) * 10 * X + min(fe - 60, 0) * 15 + max(fe - 63.5, 0) * 10
        si_adj = -max(si - 4.5, 0) * 10 - max(si - 6.5, 0) * 5 - min(si - 4.5, 0) * 5
        al_adj = -max(al - 2.5, 0) * 30 - max(min(al - 2.5, 0), -1.5) * 20
        p_adj = -max(p - 0.1, 0) * 1000 - max(p - 0.12, 0) * 500
        s_adj = -max(s - 0.03, 0) * 100 - max(s - 0.1, 0) * 400
        if brand_name in ['pbf', 'iocj', 'brbf']:
            brand_prem = 15
        else:
            brand_prem = 0
        prem_df['20220419':] = fe_adj + si_adj + al_adj + p_adj + s_adj + brand_prem
    return prem_df


def io_ctd_basis(spot_df, c1_expiries, brand_list=['pbf', 'iocj', 'macf', 'jmb', 'ssf']):
    c1_expiries = c1_expiries.dropna()
    data_df = pd.DataFrame(index=c1_expiries.index)
    data_df['expiry'] = pd.to_datetime(c1_expiries)
    skipped_list = []
    for brand_name in brand_list:
        spec = io_spec_dict[brand_name]
        moisture = spec[5]
        spot_name = io_brand_dict[brand_name]
        if spot_name not in spot_df.columns:
            skipped_list.append(brand_name)
            continue
        adj_df = io_brand_adj(brand_name)
        data_df = pd.merge(data_df, adj_df, how='left', left_on='expiry', right_index=True)
        data_df[f'{brand_name}_adj'] = data_df[f'{brand_name}_adj'].ffill()
        data_df = pd.merge(data_df, spot_df[spot_name], how='left', left_index=True, right_index=True)
        data_df[f'{brand_name}'] = data_df[spot_name]/(1-moisture/100) - data_df[f'{brand_name}_adj']
    data_df['ctd'] = data_df[[brand for brand in brand_list if brand not in skipped_list]].min(axis=1)
    return data_df['ctd']
