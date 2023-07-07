import sys
from pycmqlib3.strategy.signal_repo import signal_store
from pycmqlib3.utility.spot_idx_map import index_map, process_spot_df
from pycmqlib3.utility.dbaccess import load_codes_from_edb
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, prod2exch
from pycmqlib3.analytics.tstool import *
from misc_scripts.factor_data_update import update_factor_db

update_factors = {
    'steel_margin_lvl_fast': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'strip_hsec_lvl_mid': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_removal_lvl': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_removal_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_millinv_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_invdays_lvl': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
    'io_invdays_lyoy': ['rb', 'hc', 'i', 'j', 'jm', 'FG', 'SF', 'v', 'al', 'SM'],
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
    spot_df = process_spot_df(spot_df)
    return spot_df


def update_fun_factor(run_date=datetime.date.today(), flavor='mysql'):
    start_date = day_shift(run_date, '-3y')
    update_start = day_shift(run_date, '-20b', CHN_Holidays)
    cdate_rng = pd.date_range(start=start_date, end=run_date, freq='D', name='date')
    bdate_rng = pd.bdate_range(start=start_date, end=run_date, freq='C', holidays=CHN_Holidays, name='date')
    spot_df = get_fun_data(start_date, run_date)

    for factor_name in update_factors.keys():
        feature, signal_func, param_rng, proc_func, chg_func, bullish, freq = signal_store[factor_name]
        feature_ts = spot_df[feature].reindex(index=cdate_rng)
        if freq == 'price':
            feature_ts = feature_ts.ffill().reindex(index=bdate_rng)
        elif len(freq) > 0:
            feature_ts = feature_ts.ffill().reindex(index=pd.date_range(start=start_date,
                                                                        end=run_date,
                                                                        freq=freq)).ffill()
        else:
            feature_ts = feature_ts.dropna()

        if 'yoy' in proc_func:
            if 'lunar' in proc_func:
                label_func = lunar_label
                label_args = {}
            else:
                label_func = calendar_label
                label_args = {'anchor_date': {'month': 1, 'day': 1}}
            if '_wk' in proc_func:
                group_col = 'label_wk'
            else:
                group_col = 'label_day'
            feature_ts = yoy_generic(feature_ts, label_func=label_func, group_col=group_col, func=chg_func,
                                     label_args=label_args)
        elif 'df' in proc_func:
            n_diff = int(proc_func[2:])
            feature_ts = getattr(feature_ts, chg_func)(n_diff)

        if signal_func == 'seasonal_score_w':
            signal_ts = seasonal_score(feature_ts.to_frame(), backward=10, forward=10, rolling_years=3,
                                       min_obs=10).reindex(index=bdate_rng).ffill()
        elif signal_func == 'seasonal_score_d':
            signal_ts = seasonal_score(feature_ts.to_frame(), backward=15, forward=15, rolling_years=3, min_obs=30)
        elif len(signal_func) > 0:
            signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng,
                                         signal_cap=None)
        else:
            signal_ts = feature_ts

        if not bullish:
            signal_ts = -signal_ts
        signal_ts = signal_ts
        fact_config = {
            'roll_label': 'hot',
            'freq': 'd1',
            'serial_key': '0',
            'serial_no': 0,
        }
        #print(factor_name, signal_ts.tail(10))
        for asset in update_factors[factor_name]:
            fact_config['product_code'] = asset
            fact_config['exch'] = prod2exch(asset)
            asset_df = pd.DataFrame(signal_ts.to_frame(factor_name), index=signal_ts.index)
            asset_df.index = asset_df.index.map(lambda x: x.date())
            update_factor_db(asset_df, factor_name, fact_config,
                             start_date=update_start,
                             end_date=run_date,
                             flavor=flavor)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        tday = datetime.date.today()
    update_fun_factor(run_date=tday)
