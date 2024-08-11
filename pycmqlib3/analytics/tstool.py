import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import datetime
import math
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
import itertools
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.regression.rolling as RollingOLS
from statsmodels.tools.tools import add_constant
import scipy.stats as stats
from pykalman import KalmanFilter
import seaborn as sns
import lunardate
from matplotlib import font_manager
from .stats_test import test_mean_reverting, half_life
from statsmodels.tsa.stattools import coint, adfuller
from pycmqlib3.utility.misc import invert_dict, CHN_Holidays
import plotly.graph_objs as go
from plotly.subplots import make_subplots

font = font_manager.FontProperties(fname='C:\\windows\\fonts\\simsun.ttc')
PNL_BDAYS = 244


def multislice_many(df, label_map):
    idx_label_map = {idx: label_map[label] for idx, label in enumerate(df.columns.names) if label in label_map}
    num_levels = len(df.columns.names)
    idx_slice = tuple([idx_label_map.get(i, slice(None)) for i in range(num_levels)])
    return df.loc[:, idx_slice]


def apply_vat(df, field_list=None, index_col=None, direction=1, with_ret=True):
    if direction == 1:
        vat_fac = [1.17, 1.16, 1.13]
    else:
        vat_fac = [1/1.17, 1/1.16, 1/1.13]
    if field_list is None:
        field_list = [col for col in df.columns if col != index_col]
    if index_col is None:
        idx = df.index
    else:
        idx = df[index_col]
    cutoff_dates = [datetime.date(1980, 1, 1), datetime.date(2018, 5, 1), datetime.date(2019, 4, 1), datetime.date(2100, 1, 1)]
    if type(idx[-1]).__name__ == 'Timestamp':
        cutoff_dates = [ pd.Timestamp(d) for d in cutoff_dates]
    if with_ret:
        xdf = df.copy()
    else:
        xdf = df
    for sd, ed, vat in zip(cutoff_dates[:-1], cutoff_dates[1:], vat_fac):
        ind = (idx < ed) & (idx >= sd)
        for field in field_list:
            xdf[field][ind] = xdf[field][ind]/vat
    if with_ret:
        return xdf


def vat_adj(ts, direction=1):
    vat_fac = [1.17, 1.16, 1.13]
    if direction != 1:
        vat_fac = [1/1.17, 1/1.16, 1/1.13]
    cutoff_dates = ['1970-01-01', '2018-05-01', '2019-04-01', '2100-01-01']
    cutoff_dates = [pd.Timestamp(d) for d in cutoff_dates]
    xts = ts.copy()
    for sd, ed, vat in zip(cutoff_dates[:-1], cutoff_dates[1:], vat_fac):
        ind = (ts.index < ed) & (ts.index >= sd)
        xts[ind] = xts[ind]/vat
    return xts


def rolling_percentile(ts, win=100, direction='max'):
    ts = ts.dropna().copy(deep=True)
    data = ts.to_numpy()
    sw = sliding_window_view(data, win, axis=0).T
    scores_np = np.empty(len(ts))
    scores_np.fill(np.nan)
    scores_np[(win-1):] = ((sw <= sw[-1:, ...]).sum(axis=0).T / sw.shape[0]).flatten()
    scores_np_ts = (pd.Series(scores_np, index = ts.index) - 1/win) / (1 - 1/win)
    if direction == 'min':
        scores_np_ts = 1 - scores_np_ts
    return scores_np_ts


def zscore_roll(ts, win):
    return (ts - ts.rolling(win).mean())/ts.rolling(win).std()


def zscore_adj_roll(ts, win):
    return (ts - ts.rolling(win).mean())/ts.diff().rolling(win).std()/np.sqrt(win)


def zscore_ewm(ts, win):
    return (ts - ts.ewm(halflife=win, min_periods=win).mean())/ts.ewm(halflife=win, min_periods=win).std()


def pct_score(ts: pd.Series, win: int):
    if len(ts) < win:
        res = pd.Series(0, index=ts.index)
    else:
        res = rolling_percentile(ts, win) * 2 -1.0
    return res


def ewmac(ts, win_s, win_l=None, ls_ratio=4, vol_win=0):
    if win_l is None:
        win_l = ls_ratio * win_s
    s1 = ts.ewm(halflife=win_s, min_periods=win_s).mean()
    s2 = ts.ewm(halflife=win_l, min_periods=win_s).mean()
    if vol_win == 0:
        sig = s1 - s2
    else:
        sig = risk_normalized(s1-s2, vol_win)
    return sig


def conv_ewm(ts, h1s: list, h2s: list):
    h1_rg = list(range(*h1s))
    h2_rg = list(range(*h2s))
    combinations = itertools.product(h1_rg, h2_rg)
    collection = []
    for h1, h2 in combinations:
        collection.append(ewmac(ts, win_s=h1, win_l=h2).dropna())
    conv = pd.concat(collection, axis=1).mean(axis=1)
    return conv


def risk_normalized(ts, win=252):
    return ts/ts.ewm(halflife=win, min_periods=win, ignore_na=True).std()


def norm_ewm(ts, win=80):
    xs = ts.ewm(halflife=win, min_periods=win, ignore_na=True).mean()
    vs = ts.ewm(halflife=win, min_periods=win, ignore_na=True).std()
    return xs/vs


def hlratio(ts, win=80):
    ll = ts.rolling(win).min()
    hh = ts.rolling(win).max()
    return (ts - ll) / (hh - ll) * 2 - 1.0


def response_curve(y, response='linear', param=1):
    ''' response curve to apply to a signal, either string or a 1D function f(x)'''
    if not isinstance(response, str):  # 1D interpolation function
        out = response(y)
    elif response == 'reverting':
        scale = (1 + 2 / param ** 2) ** 0.75
        out = scale * y * np.exp(-0.5 * (y / param) ** 2)  # min/max on param
    elif response == 'absorbing':
        scale = 0.258198 * (1 + 6 / param ** 2) ** 1.75
        out = scale * y ** 3 * np.exp(-1.5 * (y / param) ** 2)
    elif response == 'sigmoid':
        # no closed form as a function of the parameter for the 2 below?
        # out = y*0+scale*(erf(y/param/np.sqrt(2))) # y*0 to maintain pandas shape through scipy
        # out = y*0+scale*(2/(1+np.exp(-y/param/np.sqrt(2)))-1) # y*0 to maintain pandas shape through scipy
        scale = 1 / np.sqrt(1 - np.sqrt(np.pi / 2) * param * np.exp(param ** 2 / 2) * math.erfc(param / np.sqrt(2)))
        out = scale * y / np.sqrt(param ** 2 + y ** 2)
    elif response == 'linear':
        out = y
    elif response == 'sign':
        out = 1.0 if y >= 0 else -1.0
    elif response == 'semilinear':
        scale = 1 / np.sqrt(
            param ** 2 + (1 - param ** 2) * math.erf(param / np.sqrt(2)) - 0.797885 * param * np.exp(-0.5 * param ** 2))
        out = scale * np.minimum(param, np.maximum(-param, y))
    elif response == 'buffer':
        scale = 1 / np.sqrt(2 * (-param * stats.norm.pdf(param) + (1 + param ** 2) * stats.norm.cdf(-param)))
        out = scale * (np.maximum(y - param, 0) + np.minimum(y + param, 0))
    elif response == 'band':
        scale = 1 / np.sqrt(1 - math.erf(param / np.sqrt(2)) + 0.797885 * param * np.exp(-0.5 * param ** 2))
        out = y * (np.abs(y) > param)
        out = out * scale
    else:
        raise Exception('unknown response curve')
    return out


def calc_conv_signal(feature_ts, signal_func, param_rng, signal_cap=2.5, vol_win=120):
    sig_list = []
    for win in range(*param_rng):
        if len(feature_ts) <= win:
            continue
        if signal_func == 'ma':
            signal_ts = feature_ts.rolling(win, min_periods=win//2).mean() \
                        / feature_ts.abs().rolling(vol_win, min_periods=vol_win//2).mean()
        elif signal_func == 'ma_sgn':
            signal_ts = np.sign(feature_ts.rolling(win).mean())
        elif signal_func == 'ema':
            signal_ts = feature_ts.ewm(win).mean()/feature_ts.abs().ewm(vol_win).mean()
        elif signal_func == 'ema_sgn':
            signal_ts = np.sign(feature_ts.ewm(win).mean())
        elif signal_func == 'ema_dff':
            signal_ts = feature_ts - feature_ts.ewm(win).mean()
            signal_ts = risk_normalized(signal_ts, win=vol_win)
        elif signal_func == 'ema_dff_sgn':
            signal_ts = np.sign(feature_ts - feature_ts.ewm(win).mean())
        elif signal_func == 'ma_dff':
            signal_ts = feature_ts - feature_ts.rolling(win).mean()
            signal_ts = risk_normalized(signal_ts, win=vol_win)
        elif signal_func == 'ma_dff_sgn':
            signal_ts = np.sign(feature_ts - feature_ts.rolling(win).mean())
        elif signal_func == 'ewmac':
            signal_ts = ewmac(feature_ts, win_s=win, ls_ratio=4, vol_win=vol_win)
            signal_ts = risk_normalized(signal_ts, win=vol_win)
        elif signal_func == 'zscore_biased':
            signal_ts = feature_ts/feature_ts.rolling(win).std()
        elif signal_func == 'zscore':
            signal_ts = zscore_roll(feature_ts, win=win)
        elif signal_func == 'zscore_adj':
            signal_ts = zscore_adj_roll(feature_ts, win=win)
        elif signal_func == 'qtl':
            signal_ts = pct_score(feature_ts.dropna(), win=win)
        elif signal_func == 'hlratio':
            signal_ts = hlratio(feature_ts, win=win)
        else:
            continue
        if signal_cap:
            if type(signal_cap) == list:
                signal_ts = cap(signal_ts, signal_cap[0], signal_cap[1])
            else:
                signal_ts = cap(signal_ts, -signal_cap, signal_cap)
        sig_list.append(signal_ts)
    if len(sig_list) > 0:
        conv_signal = pd.concat(sig_list, axis=1).mean(axis=1)
    else:
        conv_signal = pd.Series()
    return conv_signal


def signal_hysteresis(signal_ts, enter_threshold, exit_gap, use_sgn=True):
    sig_np = signal_ts.to_numpy()
    pos_np = np.zeros(len(sig_np))
    for idx in range(1, len(sig_np)):
        score = sig_np[idx]
        if use_sgn:
            pos = np.sign(score)
        else:
            pos = score
        if pos_np[idx-1] == 0 and abs(score) > enter_threshold:
            pos_np[idx] = pos
        if pos_np[idx-1] > 0:
            if score <= enter_threshold - exit_gap and abs(score) <= enter_threshold:
                pos_np[idx] = 0
            elif score <= enter_threshold - exit_gap and abs(score) > enter_threshold:
                pos_np[idx] = pos
            else:
                pos_np[idx] = pos_np[idx-1]
        elif pos_np[idx-1] < 0:
            if score >= - (enter_threshold - exit_gap) and abs(score) <= enter_threshold:
                pos_np[idx] = 0
            elif score >= -(enter_threshold - exit_gap) and abs(score) > enter_threshold:
                pos_np[idx] = pos
            else:
                pos_np[idx] = pos_np[idx-1]
    return pd.Series(pos_np, index=signal_ts.index)


def signal_hump(signal_ts, gap=0.2):
    sig_np = signal_ts.to_numpy()
    pos_np = np.zeros(len(sig_np))
    if ~np.isnan(sig_np[0]):
        pos_np[0] = sig_np[0]
    for idx in range(1, len(sig_np)):
        if (~np.isnan(sig_np[idx])) and (abs(pos_np[idx-1] - sig_np[idx]) >= gap):
            pos_np[idx] = sig_np[idx]
        else:
            pos_np[idx] = pos_np[idx-1]
    return pd.Series(pos_np, index=signal_ts.index)


def signal_buffer(signal_ts, gap=0.2):
    if type(gap) != pd.Series:
        gap = pd.Series(gap, index=signal_ts.index)
    else:
        gap = gap.reindex(index=signal_ts.index).ffill()
    if type(signal_ts) == pd.Series:
        sig_df = signal_ts.to_frame('asset')
    elif type(signal_ts) == pd.DataFrame:
        sig_df = signal_ts.copy()
    else:
        raise ValueError("First input param has to be either Series or DataFrame!")
    sig_buffered = pd.DataFrame(index=sig_df.index, columns=sig_df.columns)
    for col in sig_df.columns:
        sig_ts = sig_df[col]
        sig_np = sig_ts.to_numpy()
        gap_np = gap.to_numpy()
        pos_np = np.zeros(len(sig_np))
        if ~np.isnan(sig_np[0]) and (abs(sig_np[0]) >= gap_np[0]):
            pos_np[0] = sig_np[0]
        else:
            pos_np[0] = 0
        for idx in range(1, len(sig_np)):
            if ~np.isnan(sig_np[idx]):
                if (sig_np[idx] - gap_np[idx] - pos_np[idx-1]) >= 0:
                    pos_np[idx] = sig_np[idx] - gap[idx]
                elif (sig_np[idx] + gap_np[idx] - pos_np[idx-1]) < 0:
                    pos_np[idx] = sig_np[idx] + gap[idx]
                else:
                    pos_np[idx] = pos_np[idx-1]
            else:
                pos_np[idx] = pos_np[idx - 1]
        sig_buffered[col] = pd.Series(pos_np, index=sig_ts.index)
    if type(signal_ts) == pd.Series:
        sig_buffered = sig_buffered['asset']
    return sig_buffered


def signal_cost_optim(signal_df, hump, vol_df, cost_dict, turnover_dict={}, power=3):
    asset_list = signal_df.columns
    cost_df = pd.DataFrame.from_dict(dict(**{'date': signal_df.index}, **cost_dict)).set_index('date')
    cost_df = cost_df.reindex_like(signal_df).fillna(0)
    turnover_df = pd.DataFrame.from_dict(dict(**{'date': signal_df.index}, **turnover_dict)).set_index('date')
    turnover_df = cost_df.reindex_like(turnover_df).fillna(1)
    vol_df = vol_df.reindex_like(signal_df).ffill()
    if power == 3:
        gap_df = hump * (3*cost_df.mul(turnover_df).div(vol_df)).pow(1/power)
    elif power == 1:
        gap_df = hump * cost_df.mul(turnover_df).div(vol_df)
    signal_buffered = pd.DataFrame(index=signal_df.index, columns=signal_df.columns)
    for asset in asset_list:
        signal_buffered[asset] = signal_buffer(signal_df[asset], gap=gap_df[asset])
    return signal_buffered


def make_seasonal_df(ser, limit=1, fill=False, weekly_dense=False):
    df = ser.to_frame('data')
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    elif isinstance(df.index, pd.Index):
        df.index = pd.to_datetime(df.index)
    else:
        pass

    df['year'] = df.index.year

    if weekly_dense and isinstance(ser.index, pd.PeriodIndex) and ser.index.freqstr.staretswith('W'):
        start = pd.datetime.today() - pd.offsets.YearBegin()
        end = pd.datetime.today() + pd.offsets.YearEnd()
        df['date_s'] = df.index.week
        pr_df = pd.period_range(start, end, freq=ser.index.freq).to_frame()
        pr_df['week'] = pr_df.index.week
        pr_df.index = pr_df.index.end_time.to_period('D')
        df['date_s'] = df['date_s'].map(invert_dict(pr_df['week'].to_dict(), return_flat=True))
    else:
        df['date_s'] = df.index.map(lambda t: t.replace(year=2020))
    df = pd.pivot_table(df, values='data', index='date_s', columns='year', aggfunc=np.sum)

    if fill:
        df = df.fillna(method='ffill', limit=limit)

    if type(ser.index) == pd.PeriodIndex and ser.index.freqstr[0] == 'W':
        df = df.ffill(limit=4)

    return df


def make_lunar_df(ser, limit=1, fill=False, group_col='label_day'):
    df = ser.to_frame('data')
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    elif isinstance(df.index, pd.Index):
        df.index = pd.to_datetime(df.index)
    else:
        pass
    tdf = lunar_label(df)
    df = pd.pivot_table(tdf, values='data', index=group_col, columns='label_yr', aggfunc='last')
    if fill:
        df = df.fillna(method='ffill', limit=limit)
    if type(ser.index) == pd.PeriodIndex and ser.index.freqstr[0] == 'W':
        df = df.ffill(limit=4)
    return df


def colored_scatter(ts_a, ts_b, ts_c):
    points = plt.scatter(ts_a, ts_b, c = [float((d-ts_c.min()).days) for d in ts_c], s=20, cmap='jet')
    cb = plt.colorbar(points)
    cb.ax.set_yticklabels([str(x) for x in ts_c[::len(ts_c)//7]])
    plt.show()


def plot_signal_pnl(cumpnl, signal=None, asset_price=None, is_cum=True, figsize=(16, 10), title=''):
    if not is_cum:
        cumpnl = cumpnl.cumsum()
    dd = cumpnl.expanding().max() - cumpnl
    pnl_df = pd.concat([cumpnl, dd], axis=1)
    pnl_df.columns = ['pnl', 'dd']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(211)
    ax.plot(cumpnl.index, cumpnl.values, label='cumpnl')
    if asset_price is not None:
        asset_price = asset_price.reindex(cumpnl.index).ffill()
        ax2 = ax.twinx()
        ax2.plot(asset_price.index, asset_price.values, linestyle=':', color='y', label='price')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(True)
    plt.title(title, fontproperties=font)

    ax = fig.add_subplot(212)
    ax.plot(dd.index, dd.values, label='drawdown')
    if signal is not None:
        signal = signal.reindex(cumpnl.index).ffill()
        ax2 = ax.twinx()
        ax2.plot(signal.index, signal.values, linestyle=':', color='y', label='signal')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(True)
    plt.show()


def plot_lunar_df(ts, cutoff=None, title='', group_col='label_day', convert_seasonal=True):
    if convert_seasonal:
        xdf = make_lunar_df(ts[cutoff:], group_col=group_col)
    else:
        xdf = ts.copy()
    curr_yr = max(xdf.columns)
    fig, ax = plt.subplots()
    for yr in xdf.columns:
        if yr == curr_yr:
            marker = 'o'
            linestyle = '-'
        else:
            marker = '.'
            linestyle = '--'
        xts = xdf[yr]
        ts_mask = np.isfinite(xts)
        plt.plot(xts.index[ts_mask], xts.values[ts_mask], linestyle=linestyle, marker=marker, label=yr)

        if yr == curr_yr:
            ax.text(xts.index[ts_mask][-1], xts.values[ts_mask][-1],
                    "%s: %.1fs" % (xts.index[ts_mask][-1], xts.values[ts_mask][-1]))
    plt.title(title, fontproperties=font)
    ax.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def iplot(df,
          xaxis_title="date", yaxis_title="",
          yaxis2_title="", title="",
          width=900, height=600, secondary_y=None):
    """
    Plots an interactive plot using Plotly with an optional secondary y-axis.

    Parameters:
    df (pd.DataFrame): Data to plot.
    secondary_y (list or None): List of column names to plot on the secondary y-axis.
    """
    # Define the subplots with a secondary y-axis if needed
    fig = make_subplots(specs=[[{"secondary_y": bool(secondary_y)}]])

    # Add traces for each column in the DataFrame
    for col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col),
            secondary_y=(col in secondary_y) if secondary_y else False
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis2=dict(title=yaxis2_title) if secondary_y else None,
        width=width,
        height=height
    )

    # Show the plot
    fig.show()


def iplot_seasonal_df(ts, title='', width=900, height=500, convert_seasonal=True, cutoff=None):
    if convert_seasonal:
        xdf = make_seasonal_df(ts[cutoff:])
    else:
        xdf = ts.copy()
    fig = make_subplots()
    for yr in xdf.columns:
        xts = xdf[yr]
        ts_mask = np.isfinite(xts)
        fig.add_trace(
            go.Scatter(x=xts.index[ts_mask], y=xts.values[ts_mask],
                       name=yr, line=dict(shape='linear'),
                       marker=dict(size=6), yaxis='y')
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='date',
        yaxis_title='values',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        width=width,
        height=height
    )

    # Show the plot
    fig.show()


def iplot_lunar_df(ts, cutoff=None, title='', group_col='label_day', width=900, height=500, convert_seasonal=True):
    if convert_seasonal:
        xdf = make_lunar_df(ts[cutoff:], group_col=group_col)
    else:
        xdf = ts.copy()
    fig = make_subplots()
    for yr in xdf.columns:
        xts = xdf[yr]
        ts_mask = np.isfinite(xts)
        fig.add_trace(
            go.Scatter(x=xts.index[ts_mask], y=xts.values[ts_mask],
                       name=yr, line=dict(shape='linear'),
                       marker=dict(size=6), yaxis='y')
        )
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='date',
        yaxis_title='values',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        width=width,
        height=height
    )
    # Show the plot
    fig.show()


def plot_df_on_2ax(df, left_on=[], right_on=[], left_style='-', right_style=':'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for col in left_on:
        ts = df[col]
        ts_mask = np.isfinite(ts)
        ax.plot(ts.index[ts_mask], ts.values[ts_mask], left_style, label=col)
    ax2 = ax.twinx()
    for col in right_on:
        ts = df[col]
        ts_mask = np.isfinite(ts)
        ax2.plot(ts.index[ts_mask], ts.values[ts_mask], right_style, label=col)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid()
    plt.show()


def lunar_label(df: pd.DataFrame, copy=True):
    def find_nearest_cny(d):
        curr_yr = int(d.year)
        next_yr = int(d.year+1)
        cny_date = cny_date_map[curr_yr]
        next_cny_date = cny_date_map[next_yr]
        if abs((d-cny_date).days) <= abs((d-next_cny_date).days):
            days_to_cny = (d-cny_date).days
            cny_yr = curr_yr
        else:
            days_to_cny = (d-next_cny_date).days
            cny_yr = next_yr
        return (cny_yr, days_to_cny)
    if copy:
        data_daily = df.copy()
    else:
        data_daily = df
    bds = data_daily.index
    cny_date_map = dict([(yr, pd.Timestamp(lunardate.LunarDate(yr, 1, 1).toSolarDate()))
                         for yr in range(bds[0].year-1, bds[-1].year+2)])
    days_to_cny = pd.DataFrame(data_daily.index.map(find_nearest_cny).tolist(),
                               index=data_daily.index,
                               columns=['cny', 'days_to_cny'])
    data_daily[['label_yr', 'label_day']] = days_to_cny
    data_daily['label_wk'] = data_daily['label_day'] // 7
    return data_daily


def lunar_yoy(ts, group_col='label_day', func='diff'):
    ts_name = ts.name
    tdf = ts.dropna().to_frame(ts_name)
    tdf.index.name = 'date'
    tdf = lunar_label(tdf)
    ddf = pd.pivot_table(tdf, columns=['label_yr'], index=group_col, values=[ts_name], aggfunc='last')
    ddf = ddf.interpolate(method='linear', limit_direction='both')
    ddf = getattr(ddf, func)(axis=1)

    rdf = pd.melt(ddf.reset_index(),
                  id_vars=[(group_col, '')],
                  value_vars=[col for col in ddf.columns if col[0] == ts_name]).rename(
        columns={(group_col, ''): group_col, 'value': 'yoy'}
    )
    tdf = tdf.reset_index().merge(rdf, how='left', left_on=['label_yr', group_col], right_on=['label_yr', group_col])
    tdf = tdf.set_index('date')
    tdf = tdf[(tdf['label_yr'] > tdf['label_yr'].iloc[0] + 1) |
              ((tdf['label_yr'] == tdf['label_yr'].iloc[0] + 1) & (tdf[group_col] >= tdf[group_col].iloc[0] + 1))]
    return tdf['yoy'].to_frame(ts_name)


def calendar_label(data_df: pd.DataFrame, copy=True): # anchor_date={'month': 1, 'day': 1},
    if copy:
        data_daily = data_df.copy()
    else:
        data_daily = data_df
    data_daily.index = pd.to_datetime(data_daily.index)
    data_daily['label_yr'] = data_daily.index.map(lambda d:d.year)
    data_daily['label_mth'] = data_daily.index.map(lambda d: d.month)
    data_daily['label_wk'] = data_daily.index.map(lambda d: d.week)
    data_daily['label_day'] = data_daily.index.map(lambda d: d.dayofyear)
    # data_daily['anchor_dates'] = data_daily.index.map(lambda d: datetime.date(d.year, **anchor_date)
    #                                                   if d.date() >= datetime.date(d.year, **anchor_date)
    #                                                   else datetime.date(d.year-1, **anchor_date))
    # data_daily['label_yr'] = data_daily['anchor_dates'].map(lambda d: d.year)
    # data_daily['label_day'] = (data_daily.index - pd.to_datetime(data_daily['anchor_dates'])).map(lambda x: x.days)
    # data_daily['label_wk'] = data_daily['label_day'] // 7
    # data_daily = data_daily.drop(columns=['anchor_dates'])
    return data_daily


def yoy_generic(ts, label_func=lunar_label, func='diff', interpolate=False, group_col='label_day', label_args={}):
    ts_name = ts.name
    tdf = ts.dropna().to_frame(ts_name)
    tdf.index.name = 'date'
    tdf = label_func(tdf, **label_args)
    ddf = pd.pivot_table(tdf, columns=['label_yr'], index=group_col, values=[ts_name], aggfunc='last')
    if interpolate:
        ddf = ddf.interpolate(method='linear', limit_direction='both')
    else:
        ddf = ddf.ffill().bfill()
    ddf = getattr(ddf, func)(axis=1)
    rdf = pd.melt(ddf.reset_index(),
                  id_vars=[(group_col, '')],
                  value_vars=[col for col in ddf.columns if col[0] == ts_name]).rename(
        columns={(group_col, ''): group_col, 'value': 'yoy'}
    )
    tdf = tdf.reset_index().merge(rdf, how='left', left_on=['label_yr', group_col],
                                  right_on=['label_yr', group_col])
    tdf = tdf.set_index('date')
    tdf = tdf[(tdf['label_yr'] > tdf['label_yr'].iloc[0] + 1) |
              ((tdf['label_yr'] == tdf['label_yr'].iloc[0] + 1) & (tdf[group_col] >= tdf[group_col].iloc[0] + 1))]
    return tdf['yoy'].to_frame(ts_name)


def stitched_yoy(ts, func='diff', interpolate=False, group_col='label_day',
                 switch_period=[(12, 1, 12, 31), (3, 15, 4, 14)]):
    ts_name = ts.name
    dts = ts.dropna()
    lyoy_ts = yoy_generic(dts, label_func=lunar_label, func=func, interpolate=interpolate, group_col=group_col)
    cyoy_ts = yoy_generic(dts, label_func=calendar_label, func=func, interpolate=interpolate, group_col=group_col)
    data_df = pd.concat([lyoy_ts[ts_name].to_frame('lyoy'), cyoy_ts[ts_name].to_frame('cyoy')], axis=1).ffill()

    weight_df = pd.DataFrame(index=pd.date_range(start=datetime.date(dts.index[0].year-1, 12, 1),
                                                 end=datetime.date(dts.index[-1].year, 12, 31), freq='D'))
    weight_df['mth'] = weight_df.index.map(lambda d: d.month)
    weight_df['day'] = weight_df.index.map(lambda d: d.day)
    weight_df['w'] = np.nan
    flag = ((weight_df['mth'] == switch_period[0][0]) & (weight_df['day'] == switch_period[0][1])) | \
           ((weight_df['mth'] == switch_period[1][2]) & (weight_df['day'] == switch_period[1][3]))
    weight_df.loc[flag, 'w'] = 1
    flag = ((weight_df['mth'] == switch_period[0][2]) & (weight_df['day'] == switch_period[0][3])) | \
           ((weight_df['mth'] == switch_period[1][0]) & (weight_df['day'] == switch_period[1][1]))
    weight_df.loc[flag, 'w'] = 0
    wdf = (weight_df[['w']].interpolate(method='linear', limit_area='inside').ffill().bfill())\
        .reindex(index=data_df.index)
    data_df['w_c'] = wdf['w']
    data_df['yoy'] = data_df['cyoy'] * data_df['w_c'] + data_df['lyoy'] * (1-data_df['w_c'])
    return data_df['yoy'].to_frame(ts_name)


def stitched_yoy2(ts, func='diff', interpolate=False, group_col='label_day',
                 switch_period=[-75, -45, 30, 50]):
    ts_name = ts.name
    dts = ts.dropna()
    lyoy_ts = yoy_generic(dts, label_func=lunar_label, func=func, interpolate=interpolate, group_col=group_col)
    cyoy_ts = yoy_generic(dts, label_func=calendar_label, func=func, interpolate=interpolate, group_col=group_col)
    data_df = pd.concat([lyoy_ts[ts_name].to_frame('lyoy'), cyoy_ts[ts_name].to_frame('cyoy')], axis=1).ffill()

    weight_df = lunar_label(pd.DataFrame(index=data_df.index))
    weight_df['w'] = np.nan
    flag = (weight_df['label_day'] <= switch_period[0]) | (weight_df['label_day'] >= switch_period[3])
    weight_df.loc[flag, 'w'] = 1
    flag = (weight_df['label_day'] >= switch_period[1]) & (weight_df['label_day'] <= switch_period[2])
    weight_df.loc[flag, 'w'] = 0
    wdf = (weight_df[['w']].interpolate(method='linear', limit_area='inside').ffill().bfill())\
        .reindex(index=data_df.index)
    data_df['w_c'] = wdf['w']
    data_df['yoy'] = data_df['cyoy'] * data_df['w_c'] + data_df['lyoy'] * (1-data_df['w_c'])
    return data_df['yoy'].to_frame(ts_name)


def calendar_aggregation(df_in, period='monthly', how='returns'):
    if period == 'weekly':
        period_code = 'W-FRI'
    elif period == 'monthly':
        period_code = 'M'
    elif period == 'quarterly':
        period_code = 'Q'
    elif period == 'annual':
        period_code = 'A'
    else:
        raise ValueError("Don't recognise period")
    
    if how in ['first', 'last', 'sum', 'mean']:
        df_out = df_in.resample(period_code)
        f = getattr(df_out, how)
        df_out = f()
    elif how == 'returns':
        df_out_log = df_in.resample(period_code).sum()
        df_out = np.exp(df_out_log) - 1
    else:
        raise ValueError("Don't recognize method")
    return df_out


def cap(df_in, min_val, max_val):
    df_out = df_in.copy()
    df_out[df_out < min_val] = min_val
    df_out[df_out > max_val] = max_val
    return df_out


def lag(df_in, lag):
    df_out = df_in.shift(lag)
    return df_out


def filldown(df_in, maxfill=1):
    df_out = df_in.fillna(method='ffill', limit=maxfill)
    return df_out


def fillup(df_in, maxfill=1):
    df_out = df_in.fillna(method='bfill', limit=maxfill)
    return df_out


def diff(df_in, window=1, skipna=True):
    if skipna:
        nd_in = df_in.values
        nd_out = np_diff(nd_in, window=window)
        if type(df_in) == pd.DataFrame:
            df_out = pd.DataFrame(nd_out, index=df_in.index, columns=df_in.columns)
        elif type(df_in) == pd.Series:
            df_out = pd.Series(nd_out, index=df_in.index)
        else:
            raise ValueError("Don't recognize type")
    else:
        df_out = df_in.diff(periods=window, axis=0)
    return df_out


def np_diff(nd_in, window=1):
    shape_original = nd_in.shape
    if len(nd_in.shape) == 1:
        nd_in = nd_in.copy()
        nd_in.shape = (nd_in.shape[0], 1)
    
    nd_out = np.empty(nd_in.shape) * np.NaN
    for j in range(nd_in.shape[1]):
        j_non_nan_index = ~np.isnan(nd_in[:,j].astype(float))
        j_non_nan_values = nd_in[j_non_nan_index, j]

        j_differenced_values = np.empty(j_non_nan_values.shape) * np.NaN
        for t in range(window, len(j_non_nan_values)):
            j_differenced_values[t] = j_non_nan_values[t] - j_non_nan_values[t-window]
        
        nd_out[j_non_nan_index, j] = j_differenced_values
    nd_out.shape = shape_original
    return nd_out

    
def exp_smooth(df_in, hl, min_obs=0, fill_backward=True):
    df_out = df_in.ewm(halflife=hl,  min_periods=min_obs).mean()
    if fill_backward:
        df_out = df_out.fillna(method='bfill')
    return df_out


def ts_demean(df_in, hl, min_obs=0, fill_backward=True):
    means = exp_smooth(df_in, hl, min_obs=min_obs, fill_backward = fill_backward)
    df_out = df_in - means
    return df_out


def ts_scale(df_in, hl, min_obs=0, fill_backward=True):
    vars = df_in.pow(2.0)
    vars_sm = exp_smooth(vars, hl, min_obs=min_obs, fill_backward=fill_backward)
    vols_sm = vars_sm.pow(0.5)
    df_out = df_in/vols_sm
    return df_out


def ts_score(df_in, hl_mean, hl_vol, min_obs_mean=0, fill_backward_mean=True, min_obs_vol=0, fill_backward_vol=True):
    df_demean = ts_demean(df_in, hl_mean, min_obs=min_obs_mean, fill_backward=fill_backward_mean)
    df_out = ts_scale(df_demean, hl_vol, min_obs=min_obs_vol, fill_backward=fill_backward_vol)
    return df_out


def np_nanmean_nowarning(nd_in, axis=0, keepdims=True):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        means = np.nanmean(nd_in, axis=axis, keepdims=keepdims)
    return means


def clip_by_rolling_std(df_in, std_cut=3, **kwargs):
    rolling_std = df_in.rolling(**kwargs).std()
    rolling_mean = df_in.rolling(**kwargs).mean()
    upper = rolling_mean + std_cut * rolling_std
    lower = rolling_mean - std_cut * rolling_std
    df_in = df_in.reindex(upper.dropna().index)
    return df_in.clip(lower = lower.dropna(), upper = upper.dropna(), axis = 0)


def xs_mean(df_in):
    means = np_nanmean_nowarning(df_in.values, axis=1)
    series_out = pd.Series(means, index=df_in.index)
    return series_out


def xs_mean_repeat(df_in):
    means = np_nanmean_nowarning(df_in.values, axis=1)
    mean_repeat = np.repeat(means, len(df_in.columns), axis=1)
    mean_repeat[np.isnan(df_in.values)] = np.NaN
    df_out = pd.DataFrame(mean_repeat, index=df_in.index, columns=df_in.columns)
    return df_out


def xs_demean(df_in):
    df_out = df_in - xs_mean_repeat(df_in)
    return df_out


def xs_score(df_in, demean=True, hl=None):
    if demean:
        df_demeaned = xs_demean(df_in)
    else:
        df_demeaned = df_in
    vols_raw = df_demeaned.std(axis=1)
    if hl is None:
        vols = vols_raw
    else:
        vols = exp_smooth(vols_raw, hl)
        
    data_scored = df_demeaned.values/np.repeat(vols.values.reshape([len(vols.index), 1]),
                                               len(df_in.columns), axis=1)
    df_out = pd.DataFrame(data_scored, index=df_in.index, columns=df_in.columns)
    return df_out


def xs_rank(df_in, cutoff=0.5):
    prod_count = df_in.apply(lambda x: x.count if x.count() > 0 else np.nan, axis=1)
    rank_df = df_in.rank(axis=1)
    median_ts = rank_df.quantile(0.5, axis=1)
    df_out = rank_df.sub(median_ts, axis=0).div(prod_count - 1, axis=0) * 2.0
    if cutoff < 0.5:
        df_out = df_out[(df_out <= -1 + cutoff * 2) | (df_out >= 1 - cutoff * 2)].fillna(0)
    df_out = df_out.div(df_out.abs().sum(axis=1)/2, axis=0)
    return df_out


def seasonal_helper(df_in, func, date_range=None, min_obs=0,
                    backward=30, forward=30, rolling_years=100, **kwargs):
    import datetime as dt
    import copy
    df_in = copy.deepcopy(df_in)
    results = {}
    if date_range is None:
        date_range = df_in.index
        
    for t_date in date_range:
        try:
            past_years = set(df_in[:t_date].index.year)
            mask = []
            
            for y in past_years:
                if t_date.year - y > rolling_years:
                    continue
                
                if t_date.month == 2 and t_date.day == 29:
                    start = t_date.replace(year=y, day=28) - dt.timedelta(days = backward)
                else:
                    start = t_date.replace(year=y) - dt.timedelta(days = backward)
                    
                if y == t_date.year:
                    end = t_date
                else:
                    end = start + dt.timedelta(days = backward) + dt.timedelta(days = forward)
                mask.append((df_in.index >= start) & (df_in.index <= end))
            sample_period = df_in.loc[reduce(np.logical_or, mask)]
            if sample_period.empty or (min_obs is not None and len(sample_period) < min_obs): continue
            results[t_date] = func(sample_period, **kwargs)
        except Exception:
            raise
    return results


def seasonal_score(signal_df, **kwargs):
    def agg_func(sample_df):
        return (sample_df.iloc[-1] - sample_df.mean())/sample_df.std()
    df = seasonal_helper(df_in=signal_df, func=agg_func, **kwargs)
    return pd.DataFrame(df).T.reindex_like(signal_df)


def seasonal_group_helper(df_in, func, score_cols, yr_col='label_yr', group_col='label_wk',
                          min_obs=0, backward=1, forward=1, split_zero=True, yr_shift=0,
                          rolling_years=100, **kwargs):
    df_in = df_in.copy(deep=True)
    if type(score_cols) == str:
        score_cols = [score_cols]
    results = {}
    group_max = df_in[group_col].max()
    group_min = df_in[group_col].min()
    for t_date in df_in.index:
        curr_yr = df_in.loc[t_date, yr_col]
        curr_grp = df_in.loc[t_date, group_col]
        mask = (df_in.index <= t_date) & (df_in[yr_col] >= curr_yr - rolling_years) & \
               (df_in[yr_col] <= curr_yr - yr_shift)
        grp_rng = []
        for grp_id in range(curr_grp - backward, curr_grp + forward+1):
            gid = grp_id
            if gid < group_min:
                gid += (group_max - group_min + 1)
            elif gid > group_max:
                gid -= (group_max - group_min + 1)
            grp_rng.append(gid)
        if split_zero:
            if curr_grp >=0:
                grp_rng = [gid for gid in grp_rng if gid >= 0]
            else:
                grp_rng = [gid for gid in grp_rng if gid < 0]
        mask = mask & (df_in[group_col].isin(grp_rng))
        sample_period = df_in[mask]
        if sample_period.empty or (min_obs is not None and len(sample_period) < min_obs):
            continue
        results[t_date] = func(sample_period[score_cols], **kwargs)
    return results


def seasonal_group_score(signal_df, score_cols, **kwargs):
    def agg_func(sample_df):
        return (sample_df.iloc[-1] - sample_df.mean()) / sample_df.std()
    df = seasonal_group_helper(df_in=signal_df, func=agg_func, score_cols=score_cols, **kwargs)
    return pd.DataFrame(df).T


def calc_funda_signal(spot_df, feature, signal_func, param_rng,
                      proc_func='', chg_func='diff', bullish=True,
                      freq='price', signal_cap=[-2, 2], bdates=None,
                      post_func='', vol_win=120, curr_date=None):
    feature_ts = spot_df[feature].dropna()
    start_date = feature_ts.index[0]
    end_date = feature_ts.index[-1]
    if curr_date is not None:
        curr_date = pd.to_datetime(curr_date)
        end_date = max(curr_date, end_date)
    cdates = pd.date_range(start=start_date, end=end_date, freq='D')
    if bdates is None:
        bdates = pd.bdate_range(start=start_date, end=end_date, freq='C')
    if len(freq) > 0 and 'W-' in freq:
        feature_ts = spot_df[feature].reindex(index=cdates).ffill().reindex(
            index=pd.date_range(start=start_date, end=end_date, freq=freq)).ffill()

    if '|' in proc_func:
        proc_func_list = proc_func.split('|')
    else:
        proc_func_list = [proc_func]

    for pfunc in proc_func_list:
        if pfunc == 'beta_resid':
            if len(param_rng) > 3:
                beta_win = param_rng[3]
            else:
                beta_win = 250
            if len(chg_func) == 0:
                chg_func = 'pct_change'
            feature_ts = beta_residual(feature_ts, spot_df[freq].dropna(),
                                       beta_win=beta_win,
                                       chg_func=chg_func)
        if 'yoy' in pfunc:
            if 'lunar' in pfunc:
                label_func = lunar_label
                label_args = {}
            elif 'cal' in pfunc:
                label_func = calendar_label
                label_args = {}
            else:
                label_func = None
            if '_wk' in pfunc:
                group_col = 'label_wk'
            else:
                group_col = 'label_day'
            if label_func is None:
                feature_ts = stitched_yoy(feature_ts, group_col=group_col, func=chg_func)
            else:
                feature_ts = yoy_generic(feature_ts,
                                         label_func=label_func,
                                         group_col=group_col,
                                         func=chg_func,
                                         label_args=label_args)
            feature_ts = feature_ts[feature]
            if freq == 'price':
                feature_ts = feature_ts.reindex(index=cdates).ffill().reindex(index=bdates)
        else:
            if freq == 'price':
                feature_ts = feature_ts.reindex(index=cdates).ffill().reindex(index=bdates)
            if 'df' in pfunc:
                n_diff = int(pfunc[2:])
                feature_ts = getattr(feature_ts, chg_func)(n_diff)
            elif 'sma' in pfunc:
                n_days = int(pfunc[3:])
                feature_ts = feature_ts.rolling(n_days).mean()
            elif 'ema' in pfunc:
                n_days = int(pfunc[3:])
                feature_ts = feature_ts.ewm(n_days).mean()
            elif '_lr' in pfunc:
                feature_ts = np.log(1+feature_ts)
            elif 'csum' in pfunc:
                feature_ts = feature_ts.cumsum()
            elif 'floor' == pfunc:
                feature_ts = feature_ts.apply(lambda x: max(x - param_rng[0], 0) / param_rng[1])

    if signal_func == 'seasonal_score_w':
        signal_ts = seasonal_score(feature_ts.to_frame(),
                                   backward=param_rng[0],
                                   forward=param_rng[1],
                                   rolling_years=param_rng[2],
                                   min_obs=10)[feature]
    elif signal_func == 'seasonal_score_d':
        signal_ts = seasonal_score(feature_ts.to_frame(),
                                   backward=param_rng[0],
                                   forward=param_rng[1],
                                   rolling_years=param_rng[2],
                                   min_obs=40)[feature]
    elif signal_func == 'rng':
        signal_ts = feature_ts.apply(lambda x: max(min(x-param_rng[0], param_rng[1]), -param_rng[1])/param_rng[2])
    elif signal_func == 'dbl_th':
        if len(chg_func) > 0:
            signal_ts = eval(chg_func)(feature_ts, param_rng[1])
        else:
            signal_ts = feature_ts
        signal_ts = signal_ts.apply(lambda x: np.sign(x) if abs(x) >= param_rng[0] else 0)
    elif signal_func == 'hysteresis':
        if len(chg_func) > 0:
            signal_ts = eval(chg_func)(feature_ts, param_rng[1])
        else:
            signal_ts = feature_ts
        if len(param_rng) > 3:
            use_sgn = (param_rng[3] > 0)
        else:
            use_sgn = True
        signal_ts = signal_hysteresis(signal_ts, param_rng[0], param_rng[2], use_sgn=use_sgn)
    elif len(signal_func) > 0:
        signal_ts = calc_conv_signal(feature_ts, signal_func=signal_func, param_rng=param_rng,
                                     signal_cap=signal_cap, vol_win=vol_win)
    else:
        signal_ts = feature_ts

    if not bullish:
        signal_ts = -signal_ts

    if len(post_func) > 0:
        signal_ts = signal_ts.reindex(index=cdates).ffill().reindex(index=bdates)
        if '|' in post_func:
            post_func_list = post_func.split('|')
        else:
            post_func_list = [post_func]
        for pfunc in post_func_list:
            if pfunc[:3] == 'ema':
                n_win = int(pfunc[3])
                signal_ts = signal_ts.ewm(n_win, ignore_na=True).mean()
            elif pfunc[:3] == 'sma':
                n_win = int(pfunc[3:])
                signal_ts = signal_ts.rolling(n_win).mean()
            elif pfunc[:3] == 'hmp':
                hump_lvl = float(pfunc[3:])
                signal_ts = signal_hump(signal_ts, hump_lvl)
            elif pfunc[:3] == 'pos':
                signal_ts = signal_ts.apply(lambda x: x if x > 0 else 0)
            elif pfunc[:3] == 'neg':
                signal_ts = signal_ts.apply(lambda x: x if x < 0 else 0)
            elif pfunc[:3] == 'hys':
                params = pfunc.split('_')
                signal_ts = signal_hysteresis(signal_ts, float(params[1]), float(params[2]))
    return signal_ts


def rolling_deseasonal(raw_df, **kwargs):
    def agg_func(sample_df):
        return sample_df.iloc[-1] - sample_df.mean()
    df = seasonal_helper(df_in=raw_df, func=agg_func, **kwargs)
    return pd.DataFrame(df).T.reindex_like(raw_df)


def get_sharpe(pnl):
    return np.nanmean(pnl)/np.nanstd(pnl) * np.sqrt(PNL_BDAYS)


def get_success_rate(pnl):
    valid_pnl = pnl[pnl.abs() > 0].dropna()
    return len(valid_pnl[valid_pnl > 0]) / len(valid_pnl)


def get_ema_diff_signal(price_adj, span_fast, span_slow, quantile_window=250):
    diff = price_adj.ewm(span=span_fast, min_periods=span_fast).mean() \
            - price_adj.ewm(span=span_slow, min_periods=span_fast).mean()
    diff_quantile = get_rolling_percentiles(diff, window=quantile_window)
    signal = (diff_quantile - 0.5) * 2
    return signal


def percentile(x, vector):
    if np.isnan(x):
        return np.nan
    vector = np.array(vector)
    vector = vector[~np.isnan(vector)]
    return np.sum(vector < x) / len(vector)


def get_rolling_percentiles(vector, window=252, min_periods=None, use_abs=False):
    if min_periods is None:
        min_periods = window // 3 + 1
    if not isinstance(vector, pd.Series):
        vector = pd.Series(vector)
        
    if use_abs:
        percentiles = vector.abs().rolling(window + 1, 
                                           min_peridos=min_periods).apply(lambda s: percentile(s[-1], s[:-1]), raw=True)
        percentiles *= np.sign(vector)
    else:
        percentiles = vector.rolling(window + 1, 
                                     min_peridos=min_periods).apply(lambda s: percentile(s[-1], s[:-1]), raw=True)
    return percentiles


def get_scored_signal(signal, hl_smooth=20, hl_score=252, demean_signal=True, signal_cap=2):
    sig_smooth = exp_smooth(signal, hl=hl_smooth)
    if demean_signal:
        sig_scored = ts_score(sig_smooth, hl_vol=hl_score, hl_mean=hl_score)
    else:
        sig_scored = ts_scale(sig_smooth, hl=hl_score)
    score_capped = cap(sig_scored, -signal_cap, signal_cap)
    score_filled = filldown(score_capped, 2)
    return score_filled


def beta_hedge_ratio(target_data, index_data, beta_win=245, beta_rng=[0, 1e+5], corr_step=5):
    target_data = target_data.copy()
    if type(target_data).__name__ == 'Series':
        target_data = target_data.to_frame('port')
        index_data = index_data.to_frame('port')
    beta = pd.DataFrame(index=target_data.index)
    for col in target_data.columns:
        local_df = pd.concat([target_data[col].to_frame('target'), index_data[col].to_frame('index')],
                             axis=1).dropna(how='any')
        local_df = local_df.rolling(corr_step).mean()
        beta[col] = local_df['index'].rolling(beta_win).cov(local_df['target'])/local_df['index'].rolling(beta_win).var()
    beta = beta.clip(beta_rng[0], beta_rng[1]).ffill()
    if type(target_data).__name__ == 'Series':
        beta = beta['port']
    return beta


def beta_residual(feature_ts, index_ts, beta_win=250, chg_func='pct_change', corr_step=5):
    data_df = pd.concat([feature_ts.to_frame('feature'), index_ts.to_frame('index')], axis=1).ffill()
    betas = beta_hedge_ratio(getattr(data_df['feature'], chg_func)(corr_step),
                             getattr(data_df['index'], chg_func)(corr_step),
                             beta_win=beta_win,
                             corr_step=corr_step)
    ts = (getattr(data_df['feature'], chg_func)(1) - betas['port'] * getattr(data_df['index'], chg_func)(1)).cumsum()
    return ts


def beta_hedged_position(strat_holding, index_holding, df_pxchg,
                         beta_win=240,
                         beta_rng=[0, 1e+5],
                         hedge_mode='port'):
    strat_holding = strat_holding.reindex_like(df_pxchg).ffill()
    strat_pnl = strat_holding.shift(1).fillna(0).multiply(df_pxchg)
    index_holding = index_holding.reindex_like(df_pxchg).ffill()
    index_pnl = index_holding.shift(1).fillna(0).multiply(df_pxchg)
    if hedge_mode == 'port':
        if type(strat_pnl).__name__ == 'DataFrame':
            strat_pnl = strat_pnl.sum(axis=1)
            index_pnl = index_pnl.sum(axis=1)
        beta = beta_hedge_ratio(strat_pnl, index_pnl, beta_win=beta_win, beta_rng=beta_rng)['port']
        neutral_holding = strat_holding - index_holding.mul(beta, axis=0)
    else:
        beta = pd.DataFrame(columns=strat_pnl.columns, index=strat_pnl.index)
        neutral_holding = pd.DataFrame(columns=strat_pnl.columns, index=strat_pnl.index)
        for asset in strat_pnl.columns:
            beta[asset] = beta_hedge_ratio(strat_pnl[asset], index_pnl[asset],
                                           beta_win=beta_win, beta_rng=beta_rng)
            neutral_holding[asset] = strat_holding[asset] - index_holding[asset].mul(beta[asset], axis=0)
    hedge_results = {
        'beta': beta,
        'beta_neutral': neutral_holding,
        'index_pnl': index_pnl,
        'strat_pnl': strat_pnl,
        'hedged_pnl': strat_pnl - index_pnl.mul(beta, axis=0)
    }
    return hedge_results


def rolling_regress(ts_y, ts_x, beta_win=250, corr_step=5, add_const=True, drop_type='all'):
    var_y = ts_y.name
    var_x = ts_x.name
    reg_df = pd.concat([ts_y.to_frame(var_y), ts_x.to_frame(var_x)], axis=1).dropna(how=drop_type).ffill()
    reg_df = reg_df.rolling(corr_step).mean()
    exog = reg_df[[var_x]]
    if add_const:
        exog['const'] = 1
    rolling_ols = RollingOLS(reg_df[var_y], exog, window=beta_win, expanding=False)
    fit = rolling_ols.fit(params_only=False)
    for col in fit.params.columns:
        reg_df[f"{col}_coeff"] = fit.params[col]
    reg_df['rsquared'] = fit.rsquared
    reg_df[f"{var_y}_hat"] = reg_df[f"{var_x}"] * reg_df[f"{var_x}_coeff"]
    if add_const:
        reg_df[f"{var_y}_hat"] += reg_df["const_coeff"]
    reg_df['residuals'] = reg_df[f"{var_y}"] - reg_df[f"{var_y}_hat"]
    return reg_df


def generate_signal_sensitivity_report(signals, pnls, quantiles=None, nb_bins=6, p=0.7, return_fig=False):
    with sns.plotting_context('notebook'):
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5]
        fig, axarray = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.25)
        
        colors1 = sns.color_palatte("Set1", len(quantiles) + 2)
        colors2 = sns.color_palatte("Set2", 3)
        
        signals = signals.loc[pnls.index].copy()
        
        Q = pd.DataFrame(index=signals.index)
        Q['100th Perc'] = pnls
        for q in quantiles:
            to_keep = signals[signals.abs() > signals.abs().quantile(q)]
            Q[f'{(1 - q) * 100}th Perc'] = pnls[to_keep.index].copy()
        unique_signals = signals.unique()
        unique_signals.sort()
        unique_signals = [unique_signals[0] - 1e-6] + unique_signals.tolist()
        
        if nb_bins < len(unique_signals):
            binned = pd.qcut(signals, nb_bins, duplicates='drop')
            bin_delta = 0
            while len(binned.unique()) < nb_bins:
                bin_delta += 1
                binned = pd.qcut(signals, nb_bins + bin_delta, duplicates='drop')
        else:
            binned = pd.qcut(signals, unique_signals, duplicates='drop')
        
        df = pnls.copy().to_frame()
        df.columns = ['PnL']
        df['binned'] = binned
        PnL_per_bin = df.groupby('binned').mean()['PNL']
        Q25_per_bin = df.groupby('binned').quantile(0.25)['PNL']
        Q75_per_bin = df.groupby('binned').quantile(0.75)['PNL']
        bins = list(binned.cat.categories)
        
        ax = axarray[0, 0]
        for i, col in enumerate(Q.columns):
            sharpe = get_sharpe(Q[col])
            sr = get_success_rate(Q[col])
            cum_pnl = Q[col].cumsum().dropna()
            if len(cum_pnl) > 0:
                cum_pnl.plot(ax=ax, color=colors1[i], lavel=f'{col}, sharpe = {sharpe:.2f} - SR = {sr:.2f}')
        ax.legend(loc='best', frameon=False)
        ax.set_title('cumulative PnL signal dependence')
        ax.set_ylabel('cumulative PnL')
        
        last_signal = signals.iloc[-1]
        
        ax = axarray[0, 1]
        ax.bar(
            x=np.arrange(len(PnL_per_bin)),
            height=PnL_per_bin.values,
            tick_label=[pd.Interval(np.round(i.left, 2), np.round(i.right, 2)) for i in bins],
            yerr=(Q75_per_bin - Q25_per_bin).values,
            color=[colors2[2] if last_signal in i else colors2[0] for i in bins]
        )
        ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=70, fontsize=9)
        ax.set_ylabel('average PnL')
        
        pvals_sharpe = df.groupby('binned')['PnL'].apply(get_sharpe)
        for i, bbb in enumerate(ax.patches):
            ax.annotate(f'{pvals_sharpe.iloc[i]:.2f}', (bbb.get_x(), PnL_per_bin.values[i] * 1.5), fontsize=12)
        
        sub_strats = []
        ns = int(p + len(pnls))
        for _ in range(50):
            sub_strats += [sklearn.utils.resample(pnls, n_samples=ns).sort_index().cumsum()]
        
        ax = axarray[1, 0]
        PnL_cols = sns.cubehelix_palette(len(sub_strats))
        ax.set_ylabel('subsample cumulative PnL', fontsize=12)
        
        for cum_pnl, col in zip(sub_strats, PnL_cols):
            ax.plot(cum_pnl, color=col, linewidth=0.6)
            
        ax.set_xlim(pnls.index[0], pnls.index[-1])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax = axarray[1, 1]
        abs_pnl = pnls.abs()
        if len(pnls) > 0:
            pnls.cumsum().plot(ax=ax, color='black', label=f'full PnL', linewidth = 1)
            for q_min, q_max in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
                abs_pnl_min = abs_pnl[abs_pnl > 0].quantile(q_min/100)
                abs_pnl_max = abs_pnl[abs_pnl > 0].quantile(q_max/100)
                pnl_filtered = pnls[(abs_pnl_min < abs_pnl) & (abs_pnl <= abs_pnl_max)]
                pnl_filtered.cumsum().plot(ax=ax, label=f'abs(PnL) {q_min}-{q_max}%', linewidth=1)
        ax.set_ylabel('cumulative PnL', fontsize=12)
        ax.legend(loc='best', frameon=False)
        plt.tight_layout()
    
    if return_fig:
        return fig


def split_df(df, date_list, split_col='date'):
    output = []
    if len(date_list) == 0:
        output.append(df)
        return output
    if split_col == 'index':
        ts = df.index
    else:
        ts = df[split_col]
    index_list = [ts[0]] + date_list + [ts[-1]]
    for sdate, edate in zip(index_list[:-1], index_list[1:]):
        output.append(df[(ts <= edate) & (ts >= sdate)])
    return output


class Regression(object):
    def __init__(self, df, dependent=None, independent=None):
        """
        Initialize the class object
        Pre-condition:
            dependent - column name
            independent - list of column names
        """
        if not dependent:
            dependent = df.columns[1]
        if not independent:
            independent = [df.columns[2], ]

        formula = '{} ~ '.format(dependent)
        first = True
        for element in independent:
            if first:
                formula += element
                first = False
            else:
                formula += ' + {}'.format(element)

        self.df = df
        self.dependent = dependent
        self.independent = independent
        self.result = smf.ols(formula, df).fit()

    def summary(self):
        """
        Return linear regression summary
        """
        return self.result.summary()

    def plot_all(self):
        """
        Plot all dependent and independent variables against time. To visualize
        there relations
        """
        df = self.df
        independent = self.independent
        dependent = self.dependent

        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[dependent], label=dependent)
        for indep in independent:
            plt.plot(df.index, df[indep], label=indep)
        plt.xticks(rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot2D(self, rotation=False):
        """
        Print scatter plot and the best fit line
        Pre-condition:
            graph must be of 2D
        """
        if len(self.independent) > 1:
            raise ValueError("Not a single independent variable regression")
        params = self.result.params
        df = self.df
        k = params[1]
        b = params[0]
        independent = self.independent[0]
        dependent = self.dependent
        model = k * df[independent] + b

        plt.figure(figsize=(10, 5))
        plt.plot(df[independent], df[dependent], 'o')
        plt.plot(df[independent], model)
        plt.xlabel(independent)
        plt.ylabel(dependent)
        plt.title(dependent + ' vs. ' + independent)
        if rotation:
            plt.xticks(rotation='vertical')
        plt.show()

    def residual(self):
        """
        Return a pandas Series of residual
        Pre-condition:
            There should be no NAN in data. Hence length of date is equal to length
            of data
        """
        df = self.result.resid
        df.index = self.df.index
        return df

    def residual_plot(self, std_line=2, rotation=True):
        """
        Plot the residual against time
        Pre-condition:
            std_line - plot n std band. Set to zero to disable the feature.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.df.index, self.result.resid, label='residual')
        if rotation:
            plt.xticks(rotation='vertical')
        plt.title('residual plot')
        if std_line != 0:
            df = self.df
            std = self.residual().describe()['std']
            mean = self.residual().describe()['mean']
            num = len(df.index)
            plt.plot(df.index, std_line * std * np.ones(num) + mean, 'r--')
            plt.plot(df.index, -std_line * std * np.ones(num) + mean, 'r--')
            plt.title('residual plot ({} STD band)'.format(std_line))
        plt.show()

    def residual_vs_fit(self, colorbar=True):
        if colorbar:
            df = self.df
            y_predict = self.result.predict(df[self.independent])
            colored_scatter(y_predict, self.result.resid, df.index)
        else:
            residual = self.residual()
            df = self.df
            y_predict = self.result.predict(df[self.independent])
            plt.plot(y_predict, residual, 'o')
            plt.plot(y_predict, np.zeros(len(residual)), 'r--')
            plt.xlabel("predict")
            plt.ylabel('residual')
            plt.title('Residual vs fit')
            plt.show()

    def cross_validation(self, split_dates, split_col = 'index'):
        if type(self.df.index[0]).__name__ == 'Timestamp' and type(split_dates[0]).__name__ != 'Timestamp':
            split_dates = [pd.to_datetime(idx) for idx in split_dates]
        data_set = split_df(self.df, split_dates, split_col = split_col)
        for idx, train in enumerate(data_set):
            reg_train = Regression(train, self.dependent, self.independent)
            string = []
            for indep in reg_train.independent:
                string.append("%.4f * %s" % (reg_train.result.params[indep], indep))
            print("Train set %s: %s = %s + %.4f\t\nR-sqr: %.2f\tResid std: %.4f" % (idx,
                reg_train.dependent, ' + '.join(string), reg_train.result.params[0],
                reg_train.result.rsquared, reg_train.result.resid.std()))
            for idy in range(len(data_set)):
                if idx != idy:
                    test_sum = 0
                    for indep in self.independent:
                        test_sum += data_set[idy][indep] * reg_train.result.params[indep]
                    test_resid = data_set[idy][self.dependent] - test_sum
                    print(("Test set %s: Resid std: %.4f\tResid mean: %.4f" % (idy, test_resid.std(), test_resid.mean(),)))

    def run_all(self):
        """
        Lazy ass's ultimate solution. Run all available analysis
        Pre-condition:
            There should be only one independent variable
        """
        _2D = len(self.independent) == 1
        print()
        self.plot_all()
        print()
        print(self.summary())
        if _2D:
            self.plot2D()
        print()
        print('Error statistics')
        print(self.residual().describe())
        print()
        self.residual_vs_fit()
        self.residual_plot()
        residual = self.residual()
        test_mean_reverting(residual)
        print()
        print('Halflife = ', half_life(residual))

    def summarize_all(self):
        if len(self.independent) == 1:
            dependent = self.dependent
            independent = self.independent[0]
            params = self.result.params
            result = self.result
            k = params[1]
            b = params[0]
            conf = result.conf_int()
            cadf = adfuller(result.resid)
            if cadf[0] <= cadf[4]['5%']:
                boolean = 'likely'
            else:
                boolean = 'unlikely'
            print()
            print(("{:^40}".format("{} vs {}".format(dependent.upper(), independent.upper()))))
            print(("%20s %s = %.4f * %s + %.4f" % ("Model:", dependent, k, independent, b)))
            print(("%20s %.4f" % ("R square:", result.rsquared)))
            print(("%20s [%.4f, %.4f]" % ("Confidence interval:", conf.iloc[1, 0], conf.iloc[1, 1])))
            print(("%20s %.4f" % ("Model error:", result.resid.std())))
            print(("%20s %s" % ("Mean reverting:", boolean)))
            print(("%20s %d" % ("Half life:", half_life(result.resid))))
        else:
            dependent = self.dependent
            independent = self.independent  # list
            params = self.result.params
            result = self.result
            b = params[0]
            conf = result.conf_int()  # pandas
            cadf = adfuller(result.resid)
            if cadf[0] <= cadf[4]['5%']:
                boolean = 'likely'
            else:
                boolean = 'unlikely'
            print()
            print(("{:^40}".format("{} vs {}".format(dependent.upper(), (', '.join(independent)).upper()))))
            string = []
            for i in range(len(independent)):
                string.append("%.4f * %s" % (params[independent[i]], independent[i]))
            print(("%20s %s = %s + %.4f" % ("Model:", dependent, ' + '.join(string), b)))
            print(("%20s %.4f" % ("R square:", result.rsquared)))
            string = []
            for i in range(len(independent)):
                string.append("[%.4f, %.4f]" % (conf.loc[independent[i], 0], conf.loc[independent[i], 1]))
            print(("%20s %s" % ("Confidence interval:", ' , '.join(string))))
            print(("%20s %.4f" % ("Model error:", result.resid.std())))
            print(("%20s %s" % ("Mean reverting:", boolean)))
            print(("%20s %d" % ("Half life:", half_life(result.resid))))


class KalmanRegression(object):
    def __init__(self, df, dependent=None, independent=None, delta=None, trans_cov=None, obs_cov=None):
        if not dependent:
            dependent = df.columns[1]
        if not independent:
            independent = df.columns[2]

        self.x = df[independent]
        self.x.index = df.index
        self.y = df[dependent]
        self.y.index = df.index
        self.dependent = dependent
        self.independent = independent

        self.delta = delta or 1e-5
        self.trans_cov = trans_cov or self.delta / (1 - self.delta) * np.eye(2)
        self.obs_mat = np.expand_dims(
            np.vstack([[self.x.values], [np.ones(len(self.x))]]).T,
            axis=1
        )
        self.obs_cov = obs_cov or 1
        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                               initial_state_mean=np.zeros(2),
                               initial_state_covariance=np.ones((2, 2)),
                               transition_matrices=np.eye(2),
                               observation_matrices=self.obs_mat,
                               observation_covariance=self.obs_cov,
                               transition_covariance=self.trans_cov)
        self.state_means, self.state_covs = self.kf.filter(self.y.values)

    def slope(self):
        state_means = self.state_means
        return pd.Series(state_means[:, 0], index=self.x.index)

    def plot_params(self):
        state_means = self.state_means
        x = self.x
        _, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(x.index, state_means[:, 0], label='slope')
        axarr[0].legend()
        axarr[1].plot(x.index, state_means[:, 1], label='intercept')
        axarr[1].legend()
        plt.tight_layout()
        plt.show()
        return state_means[:, 0]

    def plot2D(self):
        x = self.x
        y = self.y
        state_means = self.state_means

        cm = plt.get_cmap('jet')
        colors = np.linspace(0.1, 1, len(x))
        # Plot data points using colormap
        sc = plt.scatter(x, y, s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
        cb = plt.colorbar(sc)
        cb.ax.set_yticklabels([str(p.date()) for p in x[::len(x) // 9].index])

        # Plot every fifth line
        step = 100
        xi = np.linspace(x.min() - 5, x.max() + 5, 2)
        colors_l = np.linspace(0.1, 1, len(state_means[::step]))
        for i, beta in enumerate(state_means[::step]):
            plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[i]))

        # Plot the OLS regression line
        plt.plot(xi, np.poly1d(np.polyfit(x, y, 1))(xi), '0.4')

        plt.title(self.dependent + ' vs. ' + self.independent)
        plt.show()

    def run_all(self):
        self.plot_params()
        self.plot2D()

