import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.api import OLS, add_constant
from scipy.stats import spearmanr, pearsonr


def compute_corr(x, y, method="spearman"):
    """Compute correlation and p-value safely."""
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan, mask.sum()
    x, y = x[mask], y[mask]
    if method == "spearman":
        c, p = spearmanr(x, y)
    else:
        c, p = pearsonr(x, y)
    return c, p


def rolling_beta(y, x, window=60):
    """
    Compute rolling regression beta of y on x.
    Returns Series of betas aligned with y.index.
    """
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(window, len(y)):
        y_sub = y.iloc[i-window:i]
        x_sub = x.iloc[i-window:i]
        if y_sub.isna().any() or x_sub.isna().any():
            continue
        model = OLS(y_sub, add_constant(x_sub)).fit()
        betas.iloc[i] = model.params[1] if len(model.params) > 1 else np.nan
    return betas


def compute_forward_returns(px_df, horizons=(1, 3, 5), shift=0):
    """
    Compute forward returns starting from next day (t+1) to (t+h).
    """
    fwd_returns = {}
    for h in horizons:
        fwd_returns[h] = px_df.shift(-shift - h) / px_df.shift(-shift) - 1
    return fwd_returns


def build_feature_return_panel(
        px_df: pd.DataFrame,
        spot_df: pd.DataFrame,
        feature_name: str,
        forward_windows=(1, 3, 5),
        beta_window=60,
        default_group: str = "all"
        ):
    """
    Given price DataFrame (cols = assets) and feature Series (index = dates),
    aligns them, computes forward returns at feature dates,
    and builds a long panel with [date, asset, feature, fwd_return_h].
    
    If feature_name is common (in spot_df.columns), compute rolling beta
    of each asset's return to that feature, and use beta * feature as score.

    Returns:
        panel DataFrame with columns:
        ['date', 'asset', 'feature', 'score', 'horizon', 'forward_return']
    """
    if isinstance(px_df.columns, pd.MultiIndex):
        col_index = px_df.columns
    else:
        col_index = pd.MultiIndex.from_product(
            [px_df.columns, [default_group]], names=["asset", "group"]
        )
        px_df.columns = col_index

    assets = col_index.get_level_values("asset").unique()
    groups = dict(col_index)

    # --- Identify feature type ---
    if feature_name in spot_df:
        feature_type = "common"
        feature_ts = spot_df[feature_name].dropna()
    else:
        feature_type = "asset_specific"
        asset_cols = [
            f"{asset}_{feature_name}" for asset in assets if f"{asset}_{feature_name}" in spot_df.columns
        ]
        feature_ts = spot_df[asset_cols].dropna(how='all')

    # --- Align data ---
    px_aligned = px_df.reindex(index=feature_ts.index, method='ffill').sort_index()
    ret_df = px_aligned.pct_change()

    # --- Forward returns ---
    fwd_returns = {}
    for h in forward_windows:
        fwd_returns[h] = px_aligned.shift(-h) / px_aligned - 1.0
    fwd_df = pd.concat(fwd_returns, axis=1)  # MultiIndex columns: (horizon, asset)

    # --- Rolling beta computation for common feature ---
    if feature_type == "common" and beta_window > 0:
        f = feature_ts.loc[ret_df.index]
        f_mean = f.rolling(beta_window).mean()
        f_var = f.rolling(beta_window).var()

        betas = pd.DataFrame(index=ret_df.index, columns=assets, dtype=float)
        for (asset, group) in ret_df.columns:
            r = ret_df[(asset, group)]
            cov_rf = (r * f).rolling(beta_window).mean() - r.rolling(beta_window).mean() * f_mean
            betas[(asset, group)] = cov_rf / f_var

        # Compute score = beta * feature value
        score_df = betas.mul(f, axis=0)

    # --- Build panel ---
    panels = []
    for h in forward_windows:
        fwd_tmp = (
            fwd_df[h]
            .stack(level=["asset", "group"])
            .rename("forward_return")
            .reset_index()
            .rename(columns={"level_0": "date"})
        )
        fwd_tmp["horizon"] = h

        if feature_type == "common":
            fwd_tmp["feature"] = feature_ts.reindex(fwd_tmp["date"]).values
            if beta_window > 0:
                fwd_tmp["beta"] = [
                    score_df.at[dt, (asset, group)] / feature_ts.at[dt] 
                    if (dt in score_df.index and feature_ts.at[dt] != 0) 
                    else np.nan
                    for dt, asset, group in zip(
                        fwd_tmp["date"], fwd_tmp["asset"], fwd_tmp["group"]
                    )
                ]
                fwd_tmp["score"] = [
                    score_df.at[dt, (asset, group)] 
                    if dt in score_df.index 
                    else np.nan
                    for dt, asset, group in zip(
                        fwd_tmp["date"], fwd_tmp["asset"], fwd_tmp["group"]
                    )
                ]
            else:
                fwd_tmp["beta"] = np.nan
                fwd_tmp["score"] = fwd_tmp["feature"]
        else:
            # Asset-specific feature
            fwd_tmp["feature"] = [
                feature_ts.at[dt, f"{asset}_{feature_name}"] 
                if dt in feature_ts.index and f"{asset}_{feature_name}" in feature_ts.columns
                else np.nan
                for dt, asset in zip(fwd_tmp["date"], fwd_tmp["asset"])
            ]
            fwd_tmp["score"] = fwd_tmp["feature"]
            fwd_tmp["beta"] = np.nan

        panels.append(fwd_tmp)

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna(subset=["score", "forward_return"]).sort_values(["date", "asset"])
    return panel


def analyze_feature_ic_ir_ts(
        panel: pd.DataFrame, 
        method: str ="spearman",
        sector_analysis: bool = True,
        group_col: str = "group"
        ):

    results, asset_list, sector_list = [], [], []
    corr_func = spearmanr if method == "spearman" else pearsonr

    for h, dfh in panel.groupby("horizon"):
        asset_corrs = []
        for asset, sub in dfh.groupby("asset"):
            mask = sub["score"].notna() & sub["forward_return"].notna()
            if mask.sum() < 10:
                continue
            c, p = corr_func(sub.loc[mask, "score"], sub.loc[mask, "forward_return"])
            asset_corrs.append({"asset": asset, "horizon": h, "ic": c, "pval": p})
        per_asset_df = pd.DataFrame(asset_corrs)
        asset_list.append(per_asset_df)

        if sector_analysis and group_col in dfh.columns:
            sector_corrs = []
            for sector, sub in dfh.groupby(group_col):
                sector_ts = (
                    sub.groupby("date")[["score", "forward_return"]].mean().dropna()
                )
                if len(sector_ts) < 10:
                    continue
                c, p = corr_func(sector_ts["score"], sector_ts["forward_return"])
                sector_corrs.append({"sector": sector, "horizon": h, "ic": c, "pval": p})
            per_sector_df = pd.DataFrame(sector_corrs)
            sector_list.append(per_sector_df)
        else:
            per_sector_df = pd.DataFrame()

        asset_mean_ic = per_asset_df["ic"].mean()
        asset_ir = (
            asset_mean_ic / per_asset_df['ic'].std()
            if per_asset_df['ic'].std() > 0
            else np.nan
        )
        if not per_asset_df.empty:
            sector_mean_ic = per_sector_df['ic'].mean()
            sector_ir = (
                sector_mean_ic / per_sector_df['ic'].std()
                if per_asset_df['ic'].std() > 0
                else np.nan
            )
        else:
            sector_mean_ic, sector_ir = np.nan, np.nan

        results.append({
            "horizon": h,
            "asset_mean_ic": asset_mean_ic,
            "asset_ir": asset_ir,
            "sector_mean_id": sector_mean_ic,
            "sector_ir": sector_ir,
            "n_assets": per_asset_df["asset"].nunique(),
            "n_sectors": per_sector_df["sector"].nunique() if not per_sector_df.empty else 0,
        })

    return {
        "summary": pd.DataFrame(results),
        "per_asset_ic": pd.concat(asset_list, ignore_index=True),
        "per_sector_ic": pd.concat(sector_list, ignore_index=True) if sector_list else pd.DataFrame(),
    }


def analyze_feature_ic_ir_xs(
        panel: pd.DataFrame, 
        method="spearman",
        group_col: str = "group"
        ):
    results, date_list, sector_list = [], [], []
    corr_func = spearmanr if method == "spearman" else pearsonr

    for h, dfh in panel.groupby("horizon"):
        date_corrs = []
        for date, sub in dfh.groupby("asset"):
            mask = sub["score"].notna() & sub["forward_return"].notna()
            if sub.loc[mask, "score"].nunique() <= 1:
                continue
            c, p = corr_func(sub.loc[mask, "score"], sub.loc[mask, "forward_return"])
            date_corrs.append({"date": date, "horizon": h, "ic": c, "pval": p})
        per_date_df = pd.DataFrame(date_corrs)
        date_list.append(per_date_df)

        sector_corrs = []
        if group_col in dfh.columns:
            for (date, sector), sub in dfh.groupby(["date", group_col]):
                mask = sub["score"].notna() & sub["forward_return"].notna()
                if sub.loc[mask, "score"].nunique() <= 1:
                    continue
                c, p = corr_func(sub.loc[mask, "score"], sub.loc[mask, "forward_return"])
                sector_corrs.append({
                    "date": date,
                    "sector": sector,
                    "horizon": h,
                    "ic": c,
                    "pval": p
                })
            per_sector_df = pd.DataFrame(sector_corrs)
            sector_list.append(per_sector_df)
        else:
            per_sector_df = pd.DataFrame()

        mean_ic = per_date_df["ic"].mean()
        ir = mean_ic / per_date_df["ic"].std() if per_date_df["ic"].std() > 0 else np.nan

        if not per_sector_df.empty:
            sector_mean_ic = per_sector_df.groupby("sector")["ic"].mean().mean()
            sector_ir = per_sector_df.groupby("sector").apply(
                lambda x: x["ic"].mean() / x["ic"].std() if x["ic"].std() > 0 else np.nan
            ).mean()
        else:
            sector_mean_ic, sector_ir = np.nan, np.nan

        results.append({
            "horizon": h,
            "global_mean_ic": mean_ic,
            "global_ir": ir,
            "sector_mean_ic": sector_mean_ic,
            "sector_ir": sector_ir,
            "n_dates": len(per_date_df),
            "n_sectors": per_sector_df["sector"].nunique() if not per_sector_df.empty else 0,
        })

    return {
        "summary": pd.DataFrame(results),
        "per_date_ic": pd.concat(date_list, ignore_index=True),
        "per_sector_ic": pd.concat(sector_list, ignore_index=True) if sector_list else pd.DataFrame()
    }


if __name__ == "__main__":
    feature_name = "us_diesel_price"
    #f1_px.columns.name = "asset"
    #feature_df = spot_df[[feature_name]].dropna().copy()
    #feature_df = feature_df.diff() # - feature_df.rolling(8).mean() #.diff() #.pct_change(20) #.diff() # - feature_df.rolling(20).mean() #.diff() #.ewm(10).mean()
    #panel_df = build_feature_return_panel(f1_px, feature_df, feature_name, forward_windows=[1, 2, 3], beta_window=0)

    # res = analyze_feature_ic_ir(panel_df, method="spearman")
    # summary_df = res['summary']
    # per_asset_df = res['per_asset_ic']
    # per_date_df = res['per_date_ic']
    # display(summary_df)
    # display(pd.pivot_table(per_asset_df, index='horizon', columns='asset', values='ic', aggfunc='last'))
    # display(pd.pivot_table(per_asset_df, index='horizon', columns='asset', values='pval', aggfunc='last'))

    # iplot(pd.pivot_table(per_date_df, index='date', columns='horizon', values='ic', aggfunc='last').rolling(20).mean())
    #sns.heatmap(per_asset_df.pivot(index="asset", columns="horizon", values="ic"), cmap="RdBu_r", center=0)
    #plt.show()