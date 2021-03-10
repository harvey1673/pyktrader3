import bsopt
import copy
import dateutil
import pyktlib as qlib
import pandas as pd
import numpy as np
import math
import dbaccess
import data_handler as dh
from scipy.stats import norm
from misc import *

SOLVER_ERROR_EPSILON = 1e-5
ITERATION_NUM = 100
ITERATION_STEP = 0.001
YEARLY_DAYS = 365.25
# Cash flow calculation for delta hedging.
# Inside the period, Vol is constant and hedging frequency is once per ndays
# bussinessDays is number of business days from the startD to expiryT

def delta_cashflow(df, vol, option_input, rehedge_period = 1, column = 'close'):
    CF = 0.0
    strike = option_input['strike']
    otype = option_input.get('otype', True)
    expiry = option_input['expiry']
    ir = option_input['ir']
    dfunc_name = option_input.get('delta_func', 'bsopt.BSDelta')
    delta_func = eval(dfunc_name)
    is_dtime = option_input.get('is_dtime', False)
    day_frac = 1.0
    nlen = len(df.index)
    for pidx in range(int(nlen/rehedge_period)):
        idx = pidx * rehedge_period
        nxt_idx = min((pidx + 1) * rehedge_period, nlen)
        if nxt_idx >= nlen -1:
            break
        if is_dtime:
            day_frac = 1.0 - qlib.GetDayFraction(datetime2xl(df['datetime'].iat[idx]), 'COMN1')
        tau = max((expiry.date() - df['date'].iat[idx].date()).days, day_frac)/YEARLY_DAYS
        opt_delta = delta_func(otype, df[column].iat[idx], strike, vol, tau, ir)
        CF = CF + opt_delta * (df[column].iat[nxt_idx] - df[column].iat[idx])
    return CF

def breakeven_vol(df, option_input, calib_input, column = 'close'):
    strike = option_input['strike']
    otype = option_input.get('otype', True)
    expiry = option_input['expiry']
    ir = option_input['ir']
    ref_vol = calib_input.get('ref_vol', 0.5)
    opt_payoff = calib_input.get('opt_payoff', 0.0)
    rehedge_period = calib_input.get('rehedge_period', 1)
    fwd = df[column].iat[0]
    pricer_func = eval(option_input.get('pricer_func', 'bsopt.BSOpt'))
    if expiry.date() < df['date'].iat[-1].date():
        raise ValueError('Expiry time must be no earlier than the end of the time series')
    numTries = 0
    diff = 1000.0
    start_d = df['date'].iat[0].date()
    tau = max((expiry.date() - start_d).days, 1.0)/YEARLY_DAYS
    vol = ref_vol
    def func(x):
        return pricer_func(otype, fwd, strike, x, tau, ir) + delta_cashflow(df, x, option_input, rehedge_period, column) - opt_payoff

    while diff >= SOLVER_ERROR_EPSILON and numTries <= ITERATION_NUM:
        current = func(vol)
        high = func(vol + ITERATION_STEP)
        low = func(vol - ITERATION_STEP)
        if high == low:
            volnext = max(vol -ITERATION_STEP, 1e-2)
        else:
            volnext = vol - 2* ITERATION_STEP * current/(high-low)
            if volnext < 1e-2:
                volnext = vol/2.0

        diff = abs(volnext - vol)
        vol = volnext
        numTries += 1

    if diff >= SOLVER_ERROR_EPSILON or numTries > ITERATION_NUM:
        return None
    else :
        return vol

def bs_delta_to_strike(fwd, delta, vol, texp):
    ys = norm.ppf(delta)
    return fwd/math.exp((ys - 0.5 * vol * math.sqrt(texp)) * vol * math.sqrt(texp))

def bachelier_delta_to_strike(fwd, delta, vol, texp):
    return fwd - norm.ppf(delta) * vol * math.sqrt(texp)

def realized_termstruct(option_input, data):
    is_dtime = data.get('is_dtime', False)
    column = data.get('data_column', 'close')
    xs = data.get('xs', [0.5])
    xs_cols = data.get('xs_names', ['atm'])
    use_bootstrap = data.get('use_bootstrap', False)
    xs_func = data.get('xs_func', 'bs_delta_to_strike')
    xs_func = eval(xs_func)
    term_tenor = data.get('term_tenor', '-1m')
    df = data['dataframe']
    calib_input = {}
    calib_input['rehedge_period'] = data.get('rehedge_period', 1)
    expiry = option_input['expiry']
    otype = option_input.get('otype', True)
    ref_vol = option_input.get('ref_vol', 0.5)
    ir = option_input['ir']
    end_vol = option_input.get('end_vol', 0.0)
    vol = end_vol
    pricer_func = eval(option_input.get('pricer_func', 'bsopt.BSFwd'))
    if is_dtime:
        datelist = df['date']
        dexp = expiry.date()
    else:
        datelist = df['date']
        dexp = expiry
    xdf = df[datelist <= dexp]
    datelist = datelist[datelist <= dexp]
    end_d  = datelist.iat[-1].date()
    start_d = end_d
    final_value = 0.0
    vol_ts = pd.DataFrame(columns = xs_cols )
    roll_idx = 0
    while start_d > datelist.iat[0].date():
        if use_bootstrap:
            end_vol = vol
            end_d = start_d
            start_d = day_shift(end_d, term_tenor)
        else:
            end_vol = 0.0
            start_d = day_shift(start_d, term_tenor)
        roll_idx += 1
        sub_df = xdf[(datelist <= end_d) & (datelist > start_d)]
        if len(sub_df) < 2:
            break
        vols = []
        for idx, x in enumerate(xs):
            strike = sub_df[column].iat[0]
            texp = (expiry.date() - start_d).days/YEARLY_DAYS
            if idx > 0:
                strike = xs_func(strike, xs[idx], vols[0], texp)
            option_input['strike'] = strike
            if end_vol > 0:
                tau = max((expiry.date() - end_d).days, 1.0)/YEARLY_DAYS
                final_value = pricer_func(otype, sub_df[column].iat[-1], strike, end_vol, tau, ir)
                ref_vol = end_vol
            elif end_vol == 0:
                if otype:
                    final_value = max((sub_df[column].iat[-1] - strike), 0)
                else:
                    final_value = max((strike - sub_df[column].iat[-1]), 0)
            elif end_vol == None:
                raise ValueError('no vol is found to match PnL')
            calib_input['ref_vol'] = 0.5
            calib_input['opt_payoff'] = final_value
            vol = breakeven_vol(sub_df, option_input, calib_input, column)
            vols.append(vol)
        tenor_str = str(roll_idx * int(term_tenor[-2])) + term_tenor[-1]
        vol_ts.ix[tenor_str, :] = vols
    return vol_ts

def breakeven_vol_by_product(prodcode, start_d, end_d, periods = 12, tenor = '-1m', writeDB = False):
    cont_mth, exch = dbaccess.prod_main_cont_exch(prodcode)
    contlist, _ = contract_range(prodcode, exch, cont_mth, start_d, end_d)
    exp_dates = [get_opt_expiry(cont, inst2contmth(cont)) for cont in contlist]
    data = {'is_dtime': True,
            'data_column': 'close',
            'data_freq': '30min',
            'xs': [0.5, 0.25, 0.75],
            'xs_names': ['atm', 'v25', 'v75'],
            'xs_func': 'bs_delta_to_strike',
            'rehedge_period': 1,
            'term_tenor': tenor,
            'database': 'hist_data'
            }
    option_input = {'otype': True,
                    'ir': 0.0,
                    'end_vol': 0.0,
                    'ref_vol': 0.5,
                    'pricer_func': 'bsopt.BSOpt',
                    'delta_func': 'bsopt.BSDelta',
                    'is_dtime': data['is_dtime'],
                    }
    freq = data['data_freq']
    for cont, expiry in zip(contlist, exp_dates):
        expiry_d = expiry.date()
        if expiry_d > end_d:
            break
        p_str = '-' + str(int(tenor[1:-1]) * periods) + tenor[-1]
        d_start = day_shift(expiry_d, p_str)
        cnx = dbaccess.connect(**dbaccess.dbconfig)
        if freq == 'd':
            df = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', cont, d_start, expiry_d, index_col = None)
        else:
            mdf = dbaccess.load_min_data_to_df(cnx, 'fut_min', cont, d_start, expiry_d, minid_start=300,
                                                  minid_end=2115, index_col = None)
            mdf = cleanup_mindata(mdf, prodcode, index_col = None)
            mdf['bar_id'] = dh.bar_conv_func2(mdf['min_id'])
            df = dh.conv_ohlc_freq(mdf, freq, bar_func=dh.bar_conv_func2, extra_cols=['bar_id'], index_col = None)
        cnx.close()
        option_input['expiry'] = expiry
        data['dataframe'] = df
        vol_df = realized_termstruct(option_input, data)
        print(cont, expiry_d, vol_df)

def hist_cso_by_product(prodcode, start_d, end_d, periods = 24, tenor = '-1w', max_spd = 2, writeDB = False, mode = 'n'):
    cont_mth, exch = dbaccess.prod_main_cont_exch(prodcode)
    contlist, _ = contract_range(prodcode, exch, cont_mth, start_d, end_d)
    exp_dates = [get_opt_expiry(cont, inst2contmth(cont)) for cont in contlist]
    if mode == 'n':
        xs_func = 'bachelier_delta_to_strike'
        pricer_func = 'bsopt.BSFwdNormal'
        delta_func = 'bsopt.BSFwdNormalDelta'
    else:
        xs_func = 'bs_delta_to_strike'
        pricer_func = 'bsopt.BSOpt'
        delta_func = 'bsopt.BSDelta'
    data = {'is_dtime': True,
            'data_column': 'close',
            'data_freq': '30min',
            'xs': [0.5, 0.25, 0.75],
            'xs_names': ['atm', 'v25', 'v75'],
            'xs_func': xs_func,
            'rehedge_period': 1,
            'term_tenor': tenor,
            'database': 'hist_data'
            }
    option_input = {'otype': True,
                    'ir': 0.0,
                    'end_vol': 0.0,
                    'ref_vol': 0.5,
                    'pricer_func': pricer_func,
                    'delta_func': delta_func,
                    'is_dtime': data['is_dtime'],
                    }
    freq = data['data_freq']
    dbconfig = copy.deepcopy(**dbaccess.hist_dbconfig)
    dbconfig['database'] = data['database']
    cnx = dbaccess.connect(**dbconfig)
    for cont, expiry in zip(contlist, exp_dates):
        expiry_d = expiry.date()
        if expiry_d > end_d:
            break
        p_str = '-' + str(int(tenor[1:-1]) * periods) + tenor[-1]
        d_start = day_shift(expiry_d, p_str)
        if freq == 'd':
            df = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', cont, d_start, expiry_d, index_col = None)
        else:
            mdf = dbaccess.load_min_data_to_df(cnx, 'fut_min', cont, d_start, expiry_d, minid_start=300,
                                                  minid_end=2115, index_col = None)
            mdf = cleanup_mindata(mdf, prodcode, index_col = None)
            mdf['bar_id'] = dh.bar_conv_func2(mdf['min_id'])
            df = dh.conv_ohlc_freq(mdf, freq, bar_func=dh.bar_conv_func2, extra_cols=['bar_id'], index_col = None)
        cnx.close()
        option_input['expiry'] = expiry
        data['dataframe'] = df
        vol_df = realized_termstruct(option_input, data)
        print(cont, expiry_d, vol_df)

def volgrid_hist_slice(underlier, strike_list, curr_date, tick_id, accr = 'COMN1', ir = 0.03, ostyle = 'EU', \
                       iv_tol=1e-5, iv_steps = 100):
    if ostyle == 'EU':
        iv_func = qlib.BlackImpliedVol
    else:
        iv_func = qlib.AmericanImpliedVol
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    df = dbaccess.load_tick_to_df(cnx, 'fut_tick', underlier, curr_date, curr_date, start_tick=300000, end_tick = tick_id)
    slice_tick = df['tick_id'].iat[-1]
    under_bid = df['bidPrice1'].iat[-1]
    under_ask = df['askPrice1'].iat[-1]
    under_mid = (under_bid + under_ask)/2.0
    opt_expiry = get_opt_expiry(underlier, inst2contmth(underlier), inst2exch(underlier))
    nBusDays = qlib.NumBusDays(date2xl(curr_date), date2xl(opt_expiry.date()), qlib.CHN_Holidays)
    day_frac = qlib.GetDayFraction(min2time(tick_id / 1000), accr)
    time2exp = (nBusDays - day_frac)/BDAYS_PER_YEAR
    res = {}
    for strike in strike_list:
        res[strike] = {}
        for otype in ['C', 'P']:
            opt_inst = get_opt_name(underlier, otype, strike)
            xdf = dbaccess.load_tick_to_df(cnx, 'fut_tick', opt_inst, curr_date, curr_date, start_tick=300000,
                                             end_tick=slice_tick)
            if len(xdf) == 0:
                print(opt_inst)
            else:
                for lbl, tag in zip(['bidPrice1', 'askPrice1'], ['bvol', 'avol']):
                    quote_p = xdf[lbl].iat[-1]
                    iv_args = (quote_p, under_mid, strike, ir, time2exp, otype, iv_tol)
                    if ostyle == 'AM':
                        iv_args = tuple(list(iv_args) + [iv_steps])
                    ivol = iv_func(*iv_args)
                    key = otype + '-' + tag
                    res[strike][key] = ivol
    cnx.close()
    return pd.DataFrame.from_dict(res, orient = 'index')[['C-bvol','C-avol','P-bvol','P-avol']]

def variance_ratio(ts, freqs):
    data = ts.values
    nlen = len(data)
    res = {'n': [], 'ln':[]}
    var1 = np.var(data[1:] - data[:-1])
    lnvar1 = np.var(np.log(data[1:]/data[:-1]))
    for freq in freqs:
        nrow = nlen/freq
        nsize = freq * nrow
        shaped_arr = np.reshape(data[:nsize], (nrow, freq))
        diff = shaped_arr[1:,freq-1] - shaped_arr[:-1,freq-1]
        res['n'].append(np.var(diff)/freq/var1)
        ln_diff = np.log(shaped_arr[1:,freq-1]/shaped_arr[:-1,freq-1])
        res['ln'].append(np.var(ln_diff)/freq/lnvar1)
    return res

def validate_db_data(tday, filter = False):
    all_insts = filter_main_cont(tday, filter)
    data_count = {}
    inst_list = {'min': [], 'daily': [] }
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    for instID in all_insts:
        df = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', instID, tday, tday)
        if len(df) <= 0:
            inst_list['daily'].append(instID)
        elif (df.close[-1] == 0) or (df.high[-1] == 0) or (df.low[-1] == 0) or df.open[-1] == 0:
            inst_list['daily'].append(instID)
        df = dbaccess.load_min_data_to_df(cnx, 'fut_min', instID, tday, tday, minid_start=300, minid_end=2115, database='blueshale')
        if len(df) <= 100:
            output = instID + ':' + str(len(df))
            inst_list['min'].append(output)
        elif df.min_id < 2055:
            output = instID + ': end earlier'
            inst_list['min'].append(output)
    cnx.close()
    print(inst_list)

def breakeven_vol_by_spot(spotID, start_d, end_d, periods = 1, tenor = '-1m'):
    data = {'is_dtime': False,
            'data_column': 'close',
            'data_freq': 'd',
            'xs': [0.5],
            'xs_names': ['atm'],
            'xs_func': 'bs_delta_to_strike',
            'rehedge_period': 1,
            'term_tenor': tenor,
            }
    option_input = {'otype': True,
                    'ir': 0.0,
                    'end_vol': 0.0,
                    'ref_vol': 0.4,
                    'pricer_func': 'bsopt.BSFwd',
                    'delta_func': 'bsopt.BSFwdDelta',
                    'is_dtime': data['is_dtime'],
                    }
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    p_str = '-' + str(int(tenor[1:-1]) * periods) + tenor[-1]
    d_end = end_d
    res = {}
    df = dbaccess.load_daily_data_to_df(cnx, 'spot_daily', spotID, day_shift(start_d, p_str), end_d, index_col=None, field = 'spotID')
    df['date'] = pd.to_datetime(df['date'])
    while d_end > start_d:
        d_start = day_shift(d_end, p_str)
        xdf = df[(df['date'] <= d_end) & (df['date'] >= d_start)]
        option_input['expiry'] = datetime.datetime.combine(d_end, datetime.time(0, 0))
        data['dataframe'] = xdf
        vol_df = realized_termstruct(option_input, data)
        res[d_start] = vol_df['atm'].values
        d_end = day_shift(d_end, tenor)
    cnx.close()
    df = pd.DataFrame.from_dict(res, orient = 'index').sort_index()
    return df

def spd_ratiovol_by_product(products, start_d, end_d, periods = 12, tenor = '-1m'):
    cont_mth, exch = dbaccess.prod_main_cont_exch(products)
    contlist, _ = contract_range(products, exch, cont_mth, start_d, end_d)
    exp_dates = [get_opt_expiry(cont, inst2contmth(cont)) for cont in contlist]
    data = {'is_dtime': True,
            'data_column': 'close',
            'data_freq': '30min',
            'xs': [0.5, 0.25, 0.75],
            'xs_names': ['atm', 'v25', 'v75'],
            'xs_func': 'bs_delta_to_strike',
            'rehedge_period': 1,
            'term_tenor': tenor,
            'database': 'hist_data'
            }
    option_input = {'otype': True,
                    'ir': 0.0,
                    'end_vol': 0.0,
                    'ref_vol': 0.5,
                    'pricer_func': 'bsopt.BSOpt',
                    'delta_func': 'bsopt.BSDelta',
                    'is_dtime': data['is_dtime'],
                    }
    freq = data['data_freq']
    for cont, expiry in zip(contlist, exp_dates):
        expiry_d = expiry.date()
        if expiry_d > end_d:
            break
        p_str = '-' + str(int(tenor[1:-1]) * periods) + tenor[-1]
        d_start = day_shift(expiry_d, p_str)
        cnx = dbaccess.connect(**dbaccess.dbconfig)
        if freq == 'd':
            df = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', cont, d_start, expiry_d, index_col = None)
        else:
            mdf = dbaccess.load_min_data_to_df(cnx, 'fut_min', cont, d_start, expiry_d, minid_start=300,
                                                  minid_end=2115, index_col = None)
            mdf = cleanup_mindata(mdf, products, index_col = None)
            mdf['bar_id'] = dh.bar_conv_func2(mdf['min_id'])
            df = dh.conv_ohlc_freq(mdf, freq, bar_func=dh.bar_conv_func2, extra_cols=['bar_id'], index_col = None)
        cnx.close()
        option_input['expiry'] = expiry
        data['dataframe'] = df
        vol_df = realized_termstruct(option_input, data)
        print(cont, expiry_d, vol_df)