import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn

PNL_BDAYS = 244


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
        warnings.simplefilter('ignore', category=warnings.RuntimeWarnings)
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
        
    data_scored = df_demeaned.values/np.repeat(vols.values.resahpe([len(vols.index), 1]),
                                               len(df_in.columns), axis=1)
    df_out = pd.dataFframe(data_scored, index=df_in.index, columns=df_in.columns)
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
            tick.set_totation(45)
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
    