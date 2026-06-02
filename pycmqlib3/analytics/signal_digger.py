import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from typing import Any, Dict, List, Optional
from statsmodels.api import OLS, add_constant
from scipy.stats import spearmanr, pearsonr, skew, kurtosis, t


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio and guard against divide-by-zero."""
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _safe_tstat_pvalue(values: pd.Series) -> Dict[str, float]:
    """Compute one-sample t-statistics and p-value for a value series."""
    clean = values.dropna().astype(float)
    n_obs = len(clean)
    if n_obs < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "n_obs": n_obs}

    mean_val = clean.mean()
    std_val = clean.std(ddof=1)
    t_stat = _safe_ratio(mean_val, std_val / np.sqrt(n_obs))
    if pd.isna(t_stat):
        p_value = np.nan
    else:
        p_value = 2.0 * (1.0 - t.cdf(abs(t_stat), df=n_obs - 1))
    return {"t_stat": t_stat, "p_value": p_value, "n_obs": n_obs}


def _compute_ic_stats_by_horizon(
        ic_df: pd.DataFrame,
        value_col: str = "ic") -> pd.DataFrame:
    """Compute Alphalens-style IC statistics by horizon."""
    if ic_df.empty or value_col not in ic_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []
    for horizon, sub in ic_df.groupby("horizon"):
        vals = sub[value_col].dropna()
        if vals.empty:
            rows.append({
                "horizon": horizon,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "risk_adjusted_ic": np.nan,
                "ic_skew": np.nan,
                "ic_kurtosis": np.nan,
                "t_stat_ic": np.nan,
                "p_value_ic": np.nan,
                "n_obs": 0,
            })
            continue

        test_stats = _safe_tstat_pvalue(vals)
        std_val = vals.std(ddof=1)
        rows.append({
            "horizon": horizon,
            "ic_mean": vals.mean(),
            "ic_std": std_val,
            "risk_adjusted_ic": _safe_ratio(vals.mean(), std_val),
            "ic_skew": skew(vals) if len(vals) > 2 else np.nan,
            "ic_kurtosis": kurtosis(vals) if len(vals) > 3 else np.nan,
            "t_stat_ic": test_stats["t_stat"],
            "p_value_ic": test_stats["p_value"],
            "n_obs": test_stats["n_obs"],
        })
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def _compute_quantile_returns(
        panel: pd.DataFrame,
        n_quantiles: int = 5) -> Dict[str, pd.DataFrame]:
    """Compute quantile return tables and long-short spread diagnostics."""
    if panel.empty:
        return {
            "quantile_returns": pd.DataFrame(),
            "quantile_summary": pd.DataFrame(),
            "long_short": pd.DataFrame(),
            "long_short_summary": pd.DataFrame(),
        }

    q_rows: List[Dict[str, Any]] = []
    ls_rows: List[Dict[str, Any]] = []

    for (horizon, date), sub in panel.groupby(["horizon", "date"]):
        work = sub[["asset", "score", "forward_return"]].dropna().copy()
        if len(work) < max(4, n_quantiles):
            continue

        try:
            work["quantile"] = pd.qcut(
                work["score"],
                q=n_quantiles,
                labels=False,
                duplicates="drop",
            ) + 1
        except ValueError:
            continue

        for quantile, q_sub in work.groupby("quantile"):
            q_rows.append({
                "horizon": horizon,
                "date": date,
                "quantile": int(quantile),
                "mean_return": q_sub["forward_return"].mean(),
                "std_return": q_sub["forward_return"].std(ddof=1),
                "n_assets": len(q_sub),
            })

        low_q = work["quantile"].min()
        high_q = work["quantile"].max()
        low_ret = work.loc[work["quantile"] == low_q, "forward_return"].mean()
        high_ret = work.loc[
            work["quantile"] == high_q, "forward_return"
        ].mean()
        ls_rows.append({
            "horizon": horizon,
            "date": date,
            "high_quantile": int(high_q),
            "low_quantile": int(low_q),
            "long_short_return": high_ret - low_ret,
        })

    quantile_returns = pd.DataFrame(q_rows)
    long_short = pd.DataFrame(ls_rows)

    if quantile_returns.empty:
        quantile_summary = pd.DataFrame()
    else:
        quantile_summary = (
            quantile_returns
            .groupby(["horizon", "quantile"])
            .agg(
                mean_return=("mean_return", "mean"),
                std_return=("mean_return", "std"),
                n_dates=("date", "nunique"),
            )
            .reset_index()
        )
        quantile_summary["return_ir"] = quantile_summary.apply(
            lambda r: _safe_ratio(r["mean_return"], r["std_return"]),
            axis=1,
        )

    if long_short.empty:
        long_short_summary = pd.DataFrame()
    else:
        ls_sum = (
            long_short
            .groupby("horizon")
            .agg(
                mean_return=("long_short_return", "mean"),
                std_return=("long_short_return", "std"),
                n_dates=("date", "nunique"),
                win_rate=("long_short_return", lambda x: (x > 0).mean()),
            )
            .reset_index()
        )
        test_rows: List[Dict[str, Any]] = []
        for horizon, vals in long_short.groupby("horizon")["long_short_return"]:
            stats = _safe_tstat_pvalue(vals)
            stats["horizon"] = horizon
            test_rows.append(stats)
        tests = pd.DataFrame(test_rows)
        if not tests.empty:
            tests = tests[["horizon", "t_stat", "p_value"]]
        long_short_summary = ls_sum.merge(tests, on="horizon", how="left")
        long_short_summary["return_ir"] = long_short_summary.apply(
            lambda r: _safe_ratio(r["mean_return"], r["std_return"]),
            axis=1,
        )

    return {
        "quantile_returns": quantile_returns,
        "quantile_summary": quantile_summary,
        "long_short": long_short,
        "long_short_summary": long_short_summary,
    }


def _compute_turnover_stats(
        panel: pd.DataFrame,
        n_quantiles: int = 5) -> Dict[str, pd.DataFrame]:
    """Compute quantile membership turnover and rank autocorrelation."""
    if panel.empty:
        return {
            "turnover_detail": pd.DataFrame(),
            "turnover_summary": pd.DataFrame(),
            "rank_autocorr": pd.DataFrame(),
        }

    turnover_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []

    for horizon, h_sub in panel.groupby("horizon"):
        date_frames: Dict[Any, pd.DataFrame] = {}
        for date, d_sub in h_sub.groupby("date"):
            work = d_sub[["asset", "score"]].dropna().copy()
            if len(work) < max(4, n_quantiles):
                continue
            try:
                work["quantile"] = pd.qcut(
                    work["score"],
                    q=n_quantiles,
                    labels=False,
                    duplicates="drop",
                ) + 1
            except ValueError:
                continue
            date_frames[date] = work.set_index("asset")

        dates = sorted(date_frames.keys())
        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]
            prev_df = date_frames[prev_date]
            curr_df = date_frames[curr_date]

            common_assets = prev_df.index.intersection(curr_df.index)
            if len(common_assets) < 3:
                continue

            prev_q = prev_df.loc[common_assets, "quantile"]
            curr_q = curr_df.loc[common_assets, "quantile"]
            changed = (prev_q != curr_q).sum()
            turnover_rows.append({
                "horizon": horizon,
                "date": curr_date,
                "turnover": changed / len(common_assets),
                "n_assets": len(common_assets),
            })

            rank_ic, _ = spearmanr(
                prev_df.loc[common_assets, "score"],
                curr_df.loc[common_assets, "score"],
            )
            rank_rows.append({
                "horizon": horizon,
                "date": curr_date,
                "rank_autocorr": rank_ic,
                "n_assets": len(common_assets),
            })

    turnover_detail = pd.DataFrame(turnover_rows)
    rank_autocorr = pd.DataFrame(rank_rows)

    if turnover_detail.empty:
        turnover_summary = pd.DataFrame()
    else:
        turnover_summary = (
            turnover_detail
            .groupby("horizon")
            .agg(
                turnover_mean=("turnover", "mean"),
                turnover_std=("turnover", "std"),
                n_dates=("date", "nunique"),
            )
            .reset_index()
        )
    if not rank_autocorr.empty:
        rank_summary = (
            rank_autocorr
            .groupby("horizon")["rank_autocorr"]
            .mean()
            .reset_index(name="rank_autocorr_mean")
        )
        turnover_summary = turnover_summary.merge(
            rank_summary, on="horizon", how="left"
        ) if not turnover_summary.empty else rank_summary

    return {
        "turnover_detail": turnover_detail,
        "turnover_summary": turnover_summary,
        "rank_autocorr": rank_autocorr,
    }


def _create_visuals(
        mode: str,
        summary_df: pd.DataFrame,
        quantile_summary: pd.DataFrame,
        long_short_df: pd.DataFrame,
        turnover_summary: pd.DataFrame) -> Dict[str, Any]:
    """Create lightweight matplotlib figures for report diagnostics."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.warn("matplotlib is unavailable; skipping figures")
        return {}

    figures: Dict[str, Any] = {}

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = summary_df["horizon"]
        ic_col = (
            "asset_mean_ic" if mode == "ts" else "global_mean_ic"
        )
        ax.plot(x, summary_df[ic_col], marker="o", label="Mean IC")
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{mode.upper()} Mean IC by Horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean IC")
        ax.legend()
        figures["mean_ic_by_horizon"] = fig

    if not quantile_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        first_h = sorted(quantile_summary["horizon"].unique())[0]
        q_sub = quantile_summary[quantile_summary["horizon"] == first_h]
        ax.bar(q_sub["quantile"].astype(str), q_sub["mean_return"])
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"Mean Quantile Return (h={first_h})")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Mean Return")
        figures["quantile_returns_bar"] = fig

    if not long_short_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for horizon, sub in long_short_df.groupby("horizon"):
            csum = sub.sort_values("date")["long_short_return"].cumsum()
            ax.plot(
                sub.sort_values("date")["date"],
                csum,
                label=f"h={horizon}",
            )
        ax.set_title("Long-Short Cumulative Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        figures["long_short_cumulative"] = fig

    if not turnover_summary.empty and "turnover_mean" in turnover_summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            turnover_summary["horizon"],
            turnover_summary["turnover_mean"],
            marker="o",
        )
        ax.set_title("Mean Quantile Turnover by Horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Turnover")
        figures["turnover_by_horizon"] = fig

    return figures


def _build_text_report(
        mode: str,
        summary_df: pd.DataFrame,
        ic_stats: pd.DataFrame,
        long_short_summary: pd.DataFrame,
        turnover_summary: pd.DataFrame) -> str:
    """Build a compact markdown report from computed diagnostics."""
    lines: List[str] = [f"# {mode.upper()} Factor Report"]
    if summary_df.empty:
        lines.append("No valid observations were available.")
        return "\n".join(lines)

    best_row = summary_df.iloc[summary_df.iloc[:, 0].index.min()]
    lines.append("## Core IC/IR")
    lines.append(f"- Horizons analyzed: {summary_df['horizon'].nunique()}")
    if mode == "ts":
        lines.append(
            "- Mean asset IC range: "
            f"{summary_df['asset_mean_ic'].min():.4f} to "
            f"{summary_df['asset_mean_ic'].max():.4f}"
        )
    else:
        lines.append(
            "- Mean cross-sectional IC range: "
            f"{summary_df['global_mean_ic'].min():.4f} to "
            f"{summary_df['global_mean_ic'].max():.4f}"
        )

    if not ic_stats.empty:
        sig_cnt = int((ic_stats["p_value_ic"] < 0.05).sum())
        lines.append("## IC Significance")
        lines.append(
            f"- Significant horizons (p<0.05): {sig_cnt}/"
            f"{len(ic_stats)}"
        )

    if not long_short_summary.empty:
        top = long_short_summary.sort_values(
            "return_ir", ascending=False
        ).iloc[0]
        lines.append("## Long-Short Spread")
        lines.append(
            f"- Best horizon by return IR: {int(top['horizon'])} "
            f"(IR={top['return_ir']:.3f}, p={top['p_value']:.4f})"
        )

    if not turnover_summary.empty and "turnover_mean" in turnover_summary:
        lines.append("## Turnover")
        lines.append(
            "- Average turnover range: "
            f"{turnover_summary['turnover_mean'].min():.3f} to "
            f"{turnover_summary['turnover_mean'].max():.3f}"
        )

    return "\n".join(lines)


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
        group_col: str = "group",
        n_quantiles: int = 5,
        include_visuals: bool = True
        ):
    """Analyze time-series IC/IR and return a full factor report package."""

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
            "asset_risk_adjusted_ic": asset_ir,
            "sector_mean_id": sector_mean_ic,
            "sector_mean_ic": sector_mean_ic,
            "sector_ir": sector_ir,
            "n_assets": per_asset_df["asset"].nunique(),
            "n_sectors": per_sector_df["sector"].nunique() if not per_sector_df.empty else 0,
        })

    per_asset_ic = pd.concat(asset_list, ignore_index=True) if asset_list else pd.DataFrame()
    per_sector_ic = (
        pd.concat(sector_list, ignore_index=True) if sector_list else pd.DataFrame()
    )
    summary_df = pd.DataFrame(results)

    ic_stats = _compute_ic_stats_by_horizon(per_asset_ic, value_col="ic")
    quantile_block = _compute_quantile_returns(panel, n_quantiles=n_quantiles)
    turnover_block = _compute_turnover_stats(panel, n_quantiles=n_quantiles)
    figures = _create_visuals(
        mode="ts",
        summary_df=summary_df,
        quantile_summary=quantile_block["quantile_summary"],
        long_short_df=quantile_block["long_short"],
        turnover_summary=turnover_block["turnover_summary"],
    ) if include_visuals else {}

    tests_df = ic_stats[["horizon", "t_stat_ic", "p_value_ic", "n_obs"]].copy() \
        if not ic_stats.empty else pd.DataFrame()

    report_text = _build_text_report(
        mode="ts",
        summary_df=summary_df,
        ic_stats=ic_stats,
        long_short_summary=quantile_block["long_short_summary"],
        turnover_summary=turnover_block["turnover_summary"],
    )

    return {
        "summary": summary_df,
        "per_asset_ic": per_asset_ic,
        "per_sector_ic": per_sector_ic,
        "ic_stats": ic_stats,
        "quantile_returns": quantile_block["quantile_returns"],
        "quantile_summary": quantile_block["quantile_summary"],
        "long_short": quantile_block["long_short"],
        "long_short_summary": quantile_block["long_short_summary"],
        "turnover_detail": turnover_block["turnover_detail"],
        "turnover_summary": turnover_block["turnover_summary"],
        "rank_autocorr": turnover_block["rank_autocorr"],
        "tests": tests_df,
        "figures": figures,
        "report_text": report_text,
    }


def analyze_feature_ic_ir_xs(
        panel: pd.DataFrame,
        method="spearman",
        group_col: str = "group",
        n_quantiles: int = 5,
        include_visuals: bool = True
        ):
    """Analyze cross-sectional IC/IR and return a full factor report package."""
    results, date_list, sector_list = [], [], []
    corr_func = spearmanr if method == "spearman" else pearsonr

    for h, dfh in panel.groupby("horizon"):
        date_corrs = []
        for date, sub in dfh.groupby("date"):
            mask = sub["score"].notna() & sub["forward_return"].notna()
            if mask.sum() < 4:
                continue
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
            "global_risk_adjusted_ic": ir,
            "sector_mean_ic": sector_mean_ic,
            "sector_ir": sector_ir,
            "n_dates": len(per_date_df),
            "n_sectors": per_sector_df["sector"].nunique() if not per_sector_df.empty else 0,
        })

    per_date_ic = pd.concat(date_list, ignore_index=True) if date_list else pd.DataFrame()
    per_sector_ic = (
        pd.concat(sector_list, ignore_index=True) if sector_list else pd.DataFrame()
    )
    summary_df = pd.DataFrame(results)

    ic_stats = _compute_ic_stats_by_horizon(per_date_ic, value_col="ic")
    quantile_block = _compute_quantile_returns(panel, n_quantiles=n_quantiles)
    turnover_block = _compute_turnover_stats(panel, n_quantiles=n_quantiles)
    figures = _create_visuals(
        mode="xs",
        summary_df=summary_df,
        quantile_summary=quantile_block["quantile_summary"],
        long_short_df=quantile_block["long_short"],
        turnover_summary=turnover_block["turnover_summary"],
    ) if include_visuals else {}

    tests_df = ic_stats[["horizon", "t_stat_ic", "p_value_ic", "n_obs"]].copy() \
        if not ic_stats.empty else pd.DataFrame()

    report_text = _build_text_report(
        mode="xs",
        summary_df=summary_df,
        ic_stats=ic_stats,
        long_short_summary=quantile_block["long_short_summary"],
        turnover_summary=turnover_block["turnover_summary"],
    )

    return {
        "summary": summary_df,
        "per_date_ic": per_date_ic,
        "per_sector_ic": per_sector_ic,
        "ic_stats": ic_stats,
        "quantile_returns": quantile_block["quantile_returns"],
        "quantile_summary": quantile_block["quantile_summary"],
        "long_short": quantile_block["long_short"],
        "long_short_summary": quantile_block["long_short_summary"],
        "turnover_detail": turnover_block["turnover_detail"],
        "turnover_summary": turnover_block["turnover_summary"],
        "rank_autocorr": turnover_block["rank_autocorr"],
        "tests": tests_df,
        "figures": figures,
        "report_text": report_text,
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
