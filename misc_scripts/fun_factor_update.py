import sys

import pandas as pd

from pycmqlib3.strategy.signal_repo import get_funda_signal_from_store
from pycmqlib3.utility.spot_idx_map import index_map, process_spot_df
from pycmqlib3.utility.dbaccess import load_codes_from_edb, load_factor_data
from pycmqlib3.utility import dataseries
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, prod2exch, is_workday, \
    nearby, contract_expiry, inst2contmth
from pycmqlib3.analytics.tstool import *
from misc_scripts.factor_data_update import update_factor_db


def long_hol_signal(spot_df, days=2, gaps=7):
    idx = pd.date_range(spot_df.index[0])
    sig_ts = spot_df.index.map(lambda x:
                               1 if ((day_shift(x.date(), f'{days}b', CHN_Holidays) - x.date()).days >= gaps) or
                                    ((x.date() - day_shift(x.date(), f'-{days}b', CHN_Holidays)).days >= gaps)
                               else 0)
    sig_ts = pd.Series(sig_ts, index=spot_df.index)
    return sig_ts


def cnc_hol_seasonality(df_pxchg, pre_days=2, post_days=2):
    sig_ts = pd.Series(0, index=pd.date_range(start=df_pxchg.index[0],
                                              end=df_pxchg.index[-1] + pd.DateOffset(days=30), freq='B'))
    for yr in range(sig_ts.index[0].year, sig_ts.index[-1].year+1):
        cny_date = pd.Timestamp(lunardate.LunarDate(yr, 1, 1).toSolarDate())
        for (evt_d, befdays, aftdays) in \
                [(cny_date, pre_days, post_days),
                 (pd.Timestamp(datetime.date(yr, 5, 1)), 2, 2),
                 (pd.Timestamp(datetime.date(yr, 10, 1)), 2, 2)]:
            flag = pd.Series(sig_ts.index < evt_d, index=sig_ts.index) & pd.Series(
                sig_ts.index.map(lambda d: pd.Timestamp(day_shift(d.date(), f'{befdays}b', CHN_Holidays)) > evt_d),
                index=sig_ts.index)
            sig_ts[flag] = 1
            flag = pd.Series(sig_ts.index >= evt_d, index=sig_ts.index) & pd.Series(
                sig_ts.index.map(lambda d: pd.Timestamp(day_shift(d.date(), f'-{aftdays}b', CHN_Holidays)) < evt_d),
                index=sig_ts.index)
            sig_ts[flag] = 1
    sig_ts = sig_ts.reindex(index=df_pxchg.index).ffill()
    return sig_ts


single_factors = {
    'steel_margin_lvl_fast': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'strip_hsec_lvl_mid': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_removal_lvl': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_removal_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_millinv_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_invdays_lvl': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_invdays_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_inv_rmv_ratio_1y': ['i'],
    'ioarb_px_hlr': ['rb', 'hc', 'i'],
    'ioarb_px_hlrhys': ['rb', 'hc', 'i'],
    'steel_sinv_lyoy_zs': ['rb', 'hc', 'i', 'FG', 'v'],
    'steel_sinv_lyoy_mds': ['rb', 'hc', 'i', 'FG', 'v'],
    'rbsales_lyoy_mom_lt': ['rb'],
    'rb_sales_inv_ratio_lyoy': ['rb'],
    'fef_c1_c2_ratio_or_qtl': ['rb', 'hc', 'j'],
    'fef_fly_ratio_or_qtl': ['rb', 'hc', 'j'],
    'fef_basmom_or_qtl': ['rb', 'hc'],
    'fef_basmom5_or_qtl': ['rb', 'hc'],
    "al_alumina_qtl": ['al'],
    "al_alumina_yoy_qtl": ['al'],
    "al_coal_qtl": ['al'],
    "ni_nis_mom_qtl": ['ni'],
    "ni_ore_qtl": ['ni'],
    "sn_conc_spot_hlr": ['sn'],
    'cu_prem_usd_zsa': ['cu'],
    'cu_prem_usd_md': ['cu'],
    'cu_phybasis_zsa': ['cu'],
    'cu_phybasis_hlr': ['cu'],
    "base_etf_mom_zsa": ["cu", "al", "zn", "pb", "ni", "sn"],
    "base_etf_mom_ewm": ["cu", "al", "zn", "pb", "ni", "sn"],
    "const_etf_mom_zsa": ["rb", "i", "j", "FG", "v"],
    "const_etf_mom_ewm": ["rb", "i", "j", "FG", "v"],
    "prop_etf_mom_dbth_zs": ["rb", "i", "FG", "v"],
    "prop_etf_mom_dbth_qtl": ["rb", "i", "FG", "v"],
    "prop_etf_mom_dbth_qtl2": ["rb", "i", "FG", "v"],
}

factors_by_asset = {
    'lme_base_ts_mds': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'lme_base_ts_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'sn'],
    'base_phybas_carry_ma': ['cu', 'al', 'zn', 'ni', 'sn', 'pb'],
    'base_inv_mds': ['cu', 'al', 'zn', 'ni', 'sn', 'pb', 'ss', 'si', 'ao'],
    'base_tc_1y_zs': ['cu', 'pb', 'zn'],
    'base_tc_2y_zs': ['cu', 'pb', 'sn'],
    'base_cifprem_1y_zs': ['cu', 'al', 'zn', 'ni'],
    'base_phybasmom_1m_zs': ['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
    'base_phybasmom_1y_zs': ['cu', 'al', 'zn', 'ni', 'pb', 'sn'],
    'metal_pbc_ema': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
                      'rb', 'hc', 'i', 'SM', 'SF', 'v', 'FG', 'SA'],
    'metal_mom_hlrhys': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
                         'rb', 'hc', 'i', 'j', 'jm', 'SM', 'SF', 'v', 'FG', 'SA'],
    # 'metal_pbc_ema_xdemean': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
    #                           'rb', 'hc', 'i', 'j', 'jm', 'SM', 'SF', 'v', 'FG', 'SA'],
    'metal_inv_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
                      'rb', 'hc', 'i', 'SM', 'SF', 'v', 'FG', 'SA'],
    'metal_inv_lyoy_hlr': ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
                           'rb', 'hc', 'i', 'SM', 'SF', 'v', 'FG', 'SA'],
}

factors_by_spread = {
    'rbhc_dmd_mds': [('rb', 1), ('hc', -1)],
    'rbhc_dmd_lyoy_mds': [('rb', 1), ('hc', -1)],
    'rbhc_sinv_mds': [('rb', 1), ('hc', -1)],
    'rbhc_sinv_lyoy_mds': [('rb', 1), ('hc', -1)],
    'rbsales_lyoy_spd_st': [('rb', 1), ('hc', -1)],
}

factors_by_beta_neutral = {
    'io_pinv31_lvl_zsa': [('rb', 'i', 1), ('hc', 'i', 1)],
    'io_pinv45_lvl_hlr': [('rb', 'i', 1), ('hc', 'i', 1)],
    'ioarb_spd_qtl_1y': [('rb', 'i', 1), ('hc', 'i', 1)],
    'fef_c1_c2_ratio_spd_qtl': [('rb', 'i', 1), ('hc', 'i', 1)],
    'fef_basmom5_spd_qtl': [('rb', 'i', 1), ('hc', 'i', 1)],
}

factors_by_func = {
    "long_hol_2b": {
        'assets': ['rb', 'hc', 'i', 'j', 'jm', 'FG',
                   'cu', 'zn', 'sn', 'ni', 'ss', 'al',
                   'l', 'pp', 'v', 'TA', 'sc', 'eb', 'eg', 'lu',
                   'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF'],
        'func': cnc_hol_seasonality,
        'args': {'pre_days': 10, 'post_days': 5}
    }
}


def get_fun_data(start_date, run_date):
    e_date = day_shift(run_date, '5b', CHN_Holidays)
    cdate_rng = pd.date_range(start=start_date, end=e_date, freq='D', name='date')
    data_df = load_codes_from_edb(index_map.keys(), source='ifind', column_name='index_code')
    data_df = data_df.rename(columns=index_map)
    spot_df = data_df.dropna(how='all').copy(deep=True)
    spot_df = spot_df.reindex(index=cdate_rng)
    # for col in [
    #     'io_inv_imp_mill(64)',
    #     'io_inv_dom_mill(64)',
    #     'io_invdays_imp_mill(64)'
    # ]:
    #     spot_df[col] = spot_df[col].shift(-3).ffill().reindex(
    #         index=pd.date_range(start=spot_df.index[0], end=spot_df.index[-1], freq='W-Fri'))
    #
    # for col in [
    #     'rebar_inv_mill', 'wirerod_inv_mill', 'hrc_inv_mill', 'crc_inv_mill', 'plate_inv_mill',
    #     'rebar_inv_social', 'wirerod_inv_social', 'hrc_inv_social', 'crc_inv_social', 'plate_inv_social',
    #     'steel_inv_social', 'rebar_inv_all', 'rebar_prod_all', 'wirerod_prod_all', 'wirerod_inv_all',
    #     'hrc_prod_all', 'hrc_inv_all', 'crc_prod_all', 'crc_inv_all',
    # ]:
    #     spot_df[col] = spot_df[col].shift(-1)
    spot_df = process_spot_df(spot_df, adjust_time=True)
    fef_list = []
    for nb in [2, 3, 4]:
        fef_nb = nearby('FEF', n=nb,
                        start_date=max(start_date, datetime.date(2016, 7, 1)),
                        end_date=run_date,
                        roll_rule='-3b', freq='d', shift_mode=2)
        fef_nb.loc[fef_nb['settle'] <= 0, 'settle'] = np.nan
        fef_nb.loc[fef_nb['close'] <= 0, 'close'] = np.nan
        fef_list.append(fef_nb['settle'].to_frame(f'FEFc{nb-1}'))
        fef_list.append(fef_nb['close'].to_frame(f'FEFc{nb-1}_close'))
        fef_list.append(fef_nb['shift'].to_frame(f'FEFc{nb-1}_shift'))
    fef_data = pd.concat(fef_list, axis=1)
    fef_data.index = pd.to_datetime(fef_data.index)
    spot_df = pd.concat([spot_df, fef_data], axis=1)
    spot_df['FEF_c1_c2_ratio'] = (spot_df['FEFc1']/np.exp(spot_df['FEFc1_shift'])) / \
                                 (spot_df['FEFc2']/np.exp(spot_df['FEFc2_shift']))
    spot_df['FEF_c123fly_ratio'] = spot_df['FEFc1'] * spot_df['FEFc3'] / \
                                   (spot_df['FEFc2'] * spot_df['FEFc2']) * \
                                   np.exp(2 * spot_df['FEFc2_shift'] - spot_df['FEFc1_shift'] - spot_df['FEFc3_shift'])
    spot_df['FEF_ryield'] = (np.log(spot_df['FEFc1'] / np.exp(spot_df['FEFc1_shift'])) -
                             np.log(spot_df['FEFc2'] / np.exp(spot_df['FEFc2_shift']))) * 12
    spot_df['FEF_basmom'] = np.log(1 + spot_df['FEFc1'].dropna().pct_change()) - \
                            np.log(1 + spot_df['FEFc2'].dropna().pct_change())
    spot_df['FEF_basmom10'] = spot_df['FEF_basmom'].dropna().rolling(10).sum()
    spot_df['FEF_basmom5'] = spot_df['FEF_basmom'].dropna().rolling(5).sum()
    spot_df = spot_df.dropna(how='all')
    return spot_df


def save_signal_to_db(asset, factor_name, signal_ts, run_date, roll_label='hot', freq='d1', flavor='mysql'):
    fact_config = {'roll_label': roll_label, 'freq': freq,
                   'serial_key': '0', 'serial_no': 0,
                   'product_code': asset, 'exch': prod2exch(asset)}
    if len(signal_ts) == 0:
        print(f"{factor_name}: {asset} for run_date={run_date} is empty")
        return
    asset_df = pd.DataFrame(signal_ts.to_frame(factor_name), index=signal_ts.index)
    asset_df.index = asset_df.index.map(lambda x: x.date())
    asset_df.index.name = 'date'
    update_factor_db(asset_df, factor_name, fact_config,
                     start_date=signal_ts.index[0].date(),
                     end_date=run_date,
                     flavor=flavor)


def load_hist_fut_prices(markets, start_date, end_date,
                         shift_mode=2, roll_name='hot', nb_cont=1, freq='d1'):
    fields = ['contract', 'open', 'high', 'low', 'close', 'volume', 'openInterest', 'diff_oi', 'expiry', 'mth', 'shift']
    data_df = pd.DataFrame()
    for prodcode in markets:
        for nb in range(nb_cont):
            xdf = dataseries.nearby(prodcode,
                                    nb + 1,
                                    start_date=start_date,
                                    end_date=end_date,
                                    shift_mode=shift_mode,
                                    freq=freq,
                                    roll_name=roll_name)
            xdf['expiry'] = xdf['contract'].map(contract_expiry)
            xdf['contmth'] = xdf['contract'].map(inst2contmth)
            xdf['mth'] = xdf['contmth'].apply(lambda x: x // 100 * 12 + x % 100)
            xdf['product'] = prodcode
            xdf['code'] = f'c{nb + 1}'
            data_df = pd.concat([data_df, xdf])
    df = pd.pivot_table(data_df.reset_index(), index='date', columns=['product', 'code'], values=list(fields),
                        aggfunc='last')
    df = df.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
    df.columns.rename(['product', 'code', 'field', ], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def update_fun_factor(run_date=datetime.date.today(), flavor='mysql'):
    start_date = day_shift(run_date, '-4y')
    update_start = day_shift(run_date, '-120b', CHN_Holidays)
    markets = ['cu', 'al', 'zn', 'pb', 'ni', 'ss', 'sn', 'ao', 'si',
               'rb', 'hc', 'i', 'j', 'jm', 'SM', 'SF', 'v', 'FG', 'SA']
    price_df = load_hist_fut_prices(markets, start_date=start_date, end_date=run_date)
    cutoff_date = day_shift(day_shift(run_date, '1b', CHN_Holidays), '-1d')
    spot_df = get_fun_data(start_date, run_date)

    fact_config = {'roll_label': 'hot', 'freq': 'd1', 'serial_key': 0, 'serial_no': 0}
    vol_win = 20
    for asset in price_df.columns.get_level_values(0).unique():
        local_df = pd.DataFrame(index=price_df.index)
        local_df['close'] = price_df[(asset, 'c1', 'close')]
        local_df['pct_chg'] = price_df[(asset, 'c1', 'close')].pct_change()
        local_df['pct_vol'] = local_df['close'] * local_df['pct_chg'].rolling(vol_win).std()
        local_df.index.name = 'date'
        fact_config['product_code'] = asset
        fact_config['exch'] = prod2exch(asset)
        update_factor_db(local_df, 'pct_vol', fact_config,
                         start_date=pd.to_datetime(update_start),
                         end_date=pd.to_datetime(run_date), flavor=flavor)

    asset_factors = []
    for factor_name in factors_by_asset.keys():
        if factor_name[-8:] == '_xdemean':
            db_fact_name = factor_name[:-8]
        elif factor_name[-7:] == '_xscore':
            db_fact_name = factor_name[:-7]
        elif factor_name[-6:] == '_xrank':
            db_fact_name = factor_name[:-6]
        else:
            db_fact_name = factor_name
        if db_fact_name in asset_factors:
            continue
        for asset in factors_by_asset[factor_name]:
            signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                    price_df=price_df,
                                                    signal_cap=[-2, 2],
                                                    asset=asset,
                                                    curr_date=run_date)
            save_signal_to_db(asset, db_fact_name, signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)
        asset_factors.append(db_fact_name)

    for factor_name in single_factors:
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                signal_cap=[-2, 2],
                                                curr_date=run_date)
        for asset in single_factors[factor_name]:
            save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_func:
        func = factors_by_func[factor_name]['func']
        func_args = factors_by_func[factor_name]['args']
        signal_ts = func(spot_df, **func_args)
        for asset in factors_by_func[factor_name]['assets']:
            save_signal_to_db(asset, factor_name, signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_spread.keys():
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                signal_cap=[-2, 2],
                                                curr_date=run_date)
        for asset, weight in factors_by_spread[factor_name]:
            save_signal_to_db(asset, factor_name, weight*signal_ts[update_start:], run_date=cutoff_date, flavor=flavor)

    for factor_name in factors_by_beta_neutral.keys():
        signal_ts = get_funda_signal_from_store(spot_df, factor_name,
                                                price_df=price_df,
                                                signal_cap=[-2, 2],
                                                curr_date=run_date)
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
            save_signal_to_db(asset, factor_name, signal_df[asset][update_start:], run_date=cutoff_date, flavor=flavor)


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
