from .tstool import *


def agg_mean(sample_df):
    return sample_df.mean()


def agg_sr(sample_df):
    return sample_df.mean()/sample_df.std()*np.sqrt(245)


def cal_seasonal_value(ts, agg_func=agg_sr, min_obs=60, backward=0, forward=30, rolling_years=5):
    df_in = ts.to_frame(ts.name)
    seasonal_df = seasonal_helper(df_in=df_in,
                                  func=agg_func,
                                  min_obs=min_obs,
                                  backward=backward,
                                  forward=forward,
                                  rolling_years=rolling_years)
    seasonal_df = pd.DataFrame(seasonal_df).T.reindex_like(df_in)
    return seasonal_df[ts.name]


def seasonal_group_value(ts, label_func=calendar_label,
                         agg_func=agg_sr,
                         group_col='label_wk',
                         min_obs=60,
                         backward=0,
                         forward=30,
                         rolling_years=5,
                         min_val=-177,
                         max_val=177):
    df_in = ts.to_frame(ts.name)
    df_in = label_func(df_in)
    df_in[group_col] = df_in[group_col].clip(min_val, max_val)
    seasonal_df = seasonal_group_helper(df_in=df_in, func=agg_func, score_cols=[ts.name],
                                        group_col=group_col,
                                        min_obs=min_obs,
                                        backward=backward,
                                        forward=forward,
                                        rolling_years=rolling_years, split_zero=False)
    seasonal_df = pd.DataFrame(seasonal_df).T.reindex_like(df_in)
    return seasonal_df[ts.name]
