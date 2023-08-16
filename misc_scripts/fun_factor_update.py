import sys
from pycmqlib3.strategy.signal_repo import signal_store, funda_signal_by_name
from pycmqlib3.utility.spot_idx_map import index_map, process_spot_df
from pycmqlib3.utility.dbaccess import load_codes_from_edb
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, prod2exch, is_workday
from pycmqlib3.analytics.tstool import *
from misc_scripts.factor_data_update import update_factor_db

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
    'cu_prem_usd_zsa': ['cu'],
    'cu_prem_usd_md': ['cu'],
    'cu_phybasis_zsa': ['cu'],
    'cu_phybasis_hlr': ['cu'],
}

factors_by_asset = {
    'lme_base_ts_mds': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'lme_base_ts_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
}

factors_by_spread = {
    'rbhc_dmd_mds': [('rb', 1), ('hc', -1)],
    'rbhc_dmd_lyoy_mds': [('rb', 1), ('hc', -1)],
    'rbhc_sinv_mds': [('rb', 1), ('hc', -1)],
    'rbhc_sinv_lyoy_mds': [('rb', 1), ('hc', -1)],
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

    for factor_name in factors_by_spread.keys():
        signal_ts = funda_signal_by_name(spot_df, factor_name, price_df=None, signal_cap=None)
        for asset, weight in factors_by_spread[factor_name]:
            save_signal_to_db(asset, factor_name, weight*signal_ts[update_start:], run_date=run_date, flavor=flavor)


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
