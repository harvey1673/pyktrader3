import sys
from pycmqlib3.strategy.signal_repo import signal_store, funda_signal_by_name
from pycmqlib3.utility.spot_idx_map import index_map, process_spot_df
from pycmqlib3.utility.dbaccess import load_codes_from_edb, load_factor_data
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, prod2exch, is_workday, nearby
from pycmqlib3.analytics.tstool import *
from misc_scripts.factor_data_update import update_factor_db


def long_hol_signal(spot_df, days=2, gaps=7):
    sig_ts = spot_df.index.map(lambda x:
                               1 if ((day_shift(x.date(), f'{days}b', CHN_Holidays) - x.date()).days >= gaps) or
                                    ((x.date() - day_shift(x.date(), f'-{days}b', CHN_Holidays)).days >= gaps)
                               else 0)
    sig_ts = pd.Series(sig_ts, index=spot_df.index)
    return sig_ts


single_factors = {
    'steel_margin_lvl_fast': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'strip_hsec_lvl_mid': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_removal_lvl': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_removal_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_millinv_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_invdays_lvl': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_invdays_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'steel_sinv_lyoy_zs': ['rb', 'hc', 'i', 'FG', 'v'],
    'steel_sinv_lyoy_mds': ['rb', 'hc', 'i', 'FG', 'v'],
    'fef_c1_c2_ratio_or_qtl': ['rb', 'hc', 'j'],
    'cu_prem_usd_zsa': ['cu'],
    'cu_prem_usd_md': ['cu'],
    'cu_phybasis_zsa': ['cu'],
    'cu_phybasis_hlr': ['cu'],
}

factors_by_asset = {
    'lme_base_ts_mds': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'lme_base_ts_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'base_phybas_carry_ma': ['cu', 'al', 'zn', 'ni', 'sn'],
    'base_inv_mds': ['cu', 'al', 'zn', 'ni'],
}

factors_by_spread = {
    'rbhc_dmd_mds': [('rb', 1), ('hc', -1)],
    'rbhc_dmd_lyoy_mds': [('rb', 1), ('hc', -1)],
    'rbhc_sinv_mds': [('rb', 1), ('hc', -1)],
    'rbhc_sinv_lyoy_mds': [('rb', 1), ('hc', -1)],
}

factors_by_beta_neutral = {
    'io_pinv31_lvl_zsa': [('rb', 'i', 1), ('hc', 'i', 1)],
    'io_pinv45_lvl_hlr': [('rb', 'i', 1), ('hc', 'i', 1)],
    'fef_c1_c2_ratio_spd_qtl': [('rb', 'i', 1), ('hc', 'i', 1), ('j', 'i', 1)],
}

factors_by_func = {
    "long_hol_2b": {
        'assets': ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'SM', 'SF',
                   'cu', 'zn', 'sn', 'ni', 'ss', 'pb', 'al',
                   'l', 'pp', 'v', 'TA', 'sc', 'eb', 'eg', 'UR', 'lu',
                   'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', 'AP', 'lh'],
        'func': long_hol_signal,
        'args': {'days': 2, 'gaps': 7}
    }
}


def get_fun_data(start_date, end_date):
    cdate_rng = pd.date_range(start=start_date, end=end_date, freq='D', name='date')
    data_df = load_codes_from_edb(index_map.keys(), source='ifind', column_name='index_code')
    data_df = data_df.rename(columns=index_map)
    spot_df = data_df.dropna(how='all').copy(deep=True)
    spot_df = spot_df.reindex(index=cdate_rng)

    for col in [
        'io_inv_imp_mill(64)',
        'io_inv_dom_mill(64)',
        'io_invdays_imp_mill(64)'
    ]:
        spot_df[col] = spot_df[col].shift(-3).ffill().reindex(
            index=pd.date_range(start=spot_df.index[0], end=spot_df.index[-1], freq='W-Fri'))

    for col in [
        'rebar_inv_mill', 'wirerod_inv_mill', 'hrc_inv_mill', 'crc_inv_mill', 'plate_inv_mill',
        'rebar_inv_social', 'wirerod_inv_social', 'hrc_inv_social', 'crc_inv_social', 'plate_inv_social',
        'steel_inv_social', 'rebar_inv_all', 'rebar_prod_all', 'wirerod_prod_all', 'wirerod_inv_all',
        'hrc_prod_all', 'hrc_inv_all', 'crc_prod_all', 'crc_inv_all',
    ]:
        spot_df[col] = spot_df[col].shift(-1)
    spot_df = process_spot_df(spot_df)
    fef_nb1 = nearby('FEF', n=2,
                     start_date=start_date,
                     end_date=end_date,
                     roll_rule='-2b', freq='d', shift_mode=0)
    fef_nb2 = nearby('FEF', n=3,
                     start_date=start_date,
                     end_date=end_date,
                     roll_rule='-2b', freq='d', shift_mode=0)
    fef_data = pd.concat([fef_nb1['settle'].to_frame('FEFc1'), fef_nb2['settle'].to_frame('FEFc2')], axis=1).dropna()
    fef_data['FEF_c1_c2_ratio'] = fef_data['FEFc1']/fef_data['FEFc2']
    spot_df['FEF_c1_c2_ratio'] = fef_data['FEF_c1_c2_ratio']
    return spot_df


def save_signal_to_db(asset, factor_name, signal_ts, run_date, roll_label='hot', freq='d1', flavor='mysql'):
    fact_config = {'roll_label': roll_label, 'freq': freq,
                   'serial_key': '0', 'serial_no': 0,
                   'product_code': asset, 'exch': prod2exch(asset)}
    asset_df = pd.DataFrame(signal_ts.to_frame(factor_name), index=signal_ts.index)
    asset_df.index = asset_df.index.map(lambda x: x.date())
    asset_df.index.name = 'date'
    update_factor_db(asset_df, factor_name, fact_config,
                     start_date=signal_ts.index[0].date(),
                     end_date=run_date,
                     flavor=flavor)


def update_fun_factor(run_date=datetime.date.today(), flavor='mysql'):
    start_date = day_shift(run_date, '-3y')
    update_start = day_shift(run_date, '-60b', CHN_Holidays)
    spot_df = get_fun_data(start_date, run_date)

    for factor_name in factors_by_asset.keys():
        for asset in factors_by_asset[factor_name]:
            signal_ts = funda_signal_by_name(spot_df, factor_name, price_df=None, signal_cap=None, asset=asset)
            save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=run_date, flavor=flavor)

    for factor_name in single_factors:
        signal_ts = funda_signal_by_name(spot_df, factor_name, price_df=None, signal_cap=None)
        for asset in single_factors[factor_name]:
            save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=run_date, flavor=flavor)

    for factor_name in factors_by_func:
        func = factors_by_func[factor_name]['func']
        func_args = factors_by_func[factor_name]['args']
        signal_ts = func(spot_df, **func_args)
        for asset in factors_by_func[factor_name]['assets']:
            save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=run_date, flavor=flavor)

    for factor_name in factors_by_spread.keys():
        signal_ts = funda_signal_by_name(spot_df, factor_name, price_df=None, signal_cap=None)
        for asset, weight in factors_by_spread[factor_name]:
            save_signal_to_db(asset, factor_name, weight*signal_ts[update_start:], run_date=run_date, flavor=flavor)

    for factor_name in factors_by_beta_neutral.keys():
        signal_ts = funda_signal_by_name(spot_df, factor_name, price_df=None, signal_cap=None)
        signal_df = pd.DataFrame(index=signal_ts.index)
        signal_df['raw_sig'] = signal_ts
        asset_list = []
        for trade_asset, index_asset, weight in factors_by_beta_neutral[factor_name]:
            asset_list = list(set(asset_list + [trade_asset, index_asset]))
            if trade_asset not in signal_df.columns:
                signal_df[trade_asset] = 0
            if index_asset not in signal_df.columns:
                signal_df[index_asset] = 0
            key = '_'.join([trade_asset, index_asset, 'beta'])
            beta_df = load_factor_data([key], factor_list=['trade_leg', 'index_leg'],
                                       roll_label='hot',
                                       start=start_date,
                                       end=run_date,
                                       freq='d1')
            beta_df = pd.pivot_table(beta_df, index='date', columns=['fact_name'], values='fact_val', aggfunc='last')
            signal_df['trade_ratio'] = beta_df['trade_leg']
            signal_df['index_ratio'] = beta_df['index_leg']
            signal_df = signal_df.ffill()
            signal_df[trade_asset] += signal_df['trade_ratio'] * signal_df['raw_sig'] * weight
            signal_df[index_asset] += signal_df['index_ratio'] * signal_df['raw_sig'] * weight
        for asset in asset_list:
            save_signal_to_db(asset, factor_name, signal_df[asset][update_start:], run_date=run_date, flavor=flavor)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        now = datetime.datetime.now()
        tday = now.date()
        if (~is_workday(tday, 'CHN')) or (now.time() < datetime.time(14, 59, 0)):
            tday = day_shift(tday, '-1b', CHN_Holidays)
    print("running for %s" % str(tday))
    update_fun_factor(run_date=tday)
