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


def build_feature_return_panel(px_df: pd.DataFrame,
                               spot_df: pd.DataFrame,
                               feature_name: str,
                               forward_windows=(1, 3, 5),
                               beta_window=60):
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
    assets = list(px_df.columns)

    # --- Identify feature type ---
    if feature_name in spot_df:
        feature_type = "common"
        feature_ts = spot_df[feature_name].dropna()
    else:
        feature_type = "asset_specific"
        feature_ts = spot_df[[f"{asset}_{feature_name}" for asset in px_df.columns]].dropna(how='all')

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
        for asset in assets:
            r = ret_df[asset]
            cov_rf = (r * f).rolling(beta_window).mean() - r.rolling(beta_window).mean() * f_mean
            betas[asset] = cov_rf / f_var

        # Compute score = beta * feature value
        score_df = betas.mul(f, axis=0)

    # --- Build panel ---
    panels = []
    for h in forward_windows:
        fwd_tmp = fwd_df[h].stack().rename("forward_return").reset_index()
        fwd_tmp["horizon"] = h

        if feature_type == "common":
            fwd_tmp["feature"] = feature_ts.reindex(fwd_tmp["date"]).values
            if beta_window > 0:
                fwd_tmp["beta"] = [
                    score_df.at[dt, asset] / feature_ts.at[dt] if dt in score_df.index and feature_ts.at[dt] != 0 else np.nan
                    for dt, asset in zip(fwd_tmp["date"], fwd_tmp["asset"])
                ]
                fwd_tmp["score"] = [
                    score_df.at[dt, asset] if dt in score_df.index else np.nan
                    for dt, asset in zip(fwd_tmp["date"], fwd_tmp["asset"])
                ]
            else:
                fwd_tmp["beta"] = np.nan
                fwd_tmp["score"] = fwd_tmp["feature"]

        else:
            # Asset-specific feature
            fwd_tmp["feature"] = [
                feature_ts.at[dt, asset] if dt in feature_ts.index else np.nan
                for dt, asset in zip(fwd_tmp["date"], fwd_tmp["asset"])
            ]
            fwd_tmp["score"] = fwd_tmp["feature"]
            fwd_tmp["beta"] = np.nan

        panels.append(fwd_tmp)

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna(subset=["score", "forward_return"])
    return panel


def analyze_feature_ic_ir(panel: pd.DataFrame, method="spearman"):
    """
    Analyze feature predictive power via Information Coefficient (IC) and Information Ratio (IR).

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ['date', 'asset', 'feature', 'forward_return', 'horizon'].
    method : str
        'spearman' (rank IC) or 'pearson' (linear IC).

    Returns
    -------
    results : dict of DataFrames
        {
            'summary': IC/IR summary per horizon,
            'per_asset_ic': IC per asset per horizon,
            'per_date_ic': IC per date per horizon
        }
    """
    results = []
    per_asset_list = []
    per_date_list = []

    for h, dfh in panel.groupby("horizon"):
        # === (1) IC per asset (time-series ICs)
        asset_corrs, asset_pvals = [], []
        for asset, sub in dfh.groupby("asset"):
            c, p = compute_corr(sub["score"], sub["forward_return"], method)
            asset_corrs.append({"asset": asset, "horizon": h, "ic": c, "pval": p})
        per_asset_df = pd.DataFrame(asset_corrs)
        per_asset_list.append(per_asset_df)

        # === (2) IC per date (cross-sectional ICs)
        date_corrs = []
        for date, sub in dfh.groupby("date"):
            c, p = compute_corr(sub["score"], sub["forward_return"], method)
            date_corrs.append({"date": date, "horizon": h, "ic": c, "pval": p})
        per_date_df = pd.DataFrame(date_corrs)
        per_date_list.append(per_date_df)

        # === (3) Compute IR stats from cross-sectional ICs (per-date)
        mean_ic = per_date_df["ic"].mean()
        std_ic = per_date_df["ic"].std()
        ir = mean_ic / std_ic if std_ic > 0 else np.nan

        results.append({
            "horizon": h,
            "mean_ic": mean_ic,
            "median_ic": per_date_df["ic"].median(),
            "ic_std": std_ic,
            "ic_ir": ir,
            "n_dates": per_date_df["ic"].notna().sum(),
            "n_assets": per_asset_df["asset"].nunique()
        })

    return {
        "summary": pd.DataFrame(results),
        "per_asset_ic": pd.concat(per_asset_list, ignore_index=True),
        "per_date_ic": pd.concat(per_date_list, ignore_index=True),
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