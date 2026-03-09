"""
Commodity Correlation, PCA & Clustering Analytics
===================================================
Loads c1 continuous contract daily log returns and runs a full suite of
clustering and structural analytics:

  1. PCA (scree + loadings)
  2. Correlation heatmap
  3. Hierarchical clustering with dendrogram (multi-level cutoffs)
  4. Graph-based Louvain community detection (no k required)
  5. Minimum Spanning Tree (MST) — core sector structure
  6. Rolling correlation stability — which pairs are reliably correlated
  7. DBSCAN — detects outlier/unclustered assets

Usage:
    python tests/commodity_clustering.py [YYYYMMDD] [--show]

Outputs (all saved to C:\\dev\\data\\clustering\\):
    pca_scree.png, pca_loadings.png
    correlation_heatmap.png
    dendrogram.png
    mst.png
    rolling_corr_stability.png
    cluster_assignments.csv        (hierarchical + louvain + dbscan)
"""

import sys
import os
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')

sys.path.insert(0, r'c:\dev\pyktrader3')
sys.path.insert(0, r'c:\dev\wtpy')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import networkx as nx
import community as community_louvain  # python-louvain

# ---------------------------------------------------------------------------
# Sector definitions  (mirrors notebook cell 3)
# ---------------------------------------------------------------------------

SECTORS = {
    'ferrous':        ['rb', 'hc', 'i', 'j', 'jm'],
    'ferrous_mixed':  ['FG', 'SM', 'SF', 'ru', 'SA', 'UR'],
    'base_metal':     ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'ss', 'ao'],
    'precious_metal': ['au', 'ag'],
    'petro_chem':     ['sc', 'fu', 'lu', 'l', 'pp', 'v', 'TA', 'PX', 'MA', 'eg', 'eb', 'bu', 'PF'],
    'agri_oil':       ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'b'],
    'agri_soft':      ['CF', 'SR', 'jd', 'AP', 'lh', 'PK', 'CJ'],
    'bond_fut':       ['T', 'TF', 'TL'],
}

# Default commodity universe for analysis (excludes financials + ps which has <2y history)
DEFAULT_UNIVERSE = [
    'au', 'ag',
    'rb', 'hc', 'i', 'j', 'jm', 'SM', 'SF', 'FG', 'SA', 'UR', 'ru', 'SH',
    'cu', 'al', 'zn', 'pb', 'sn', 'ni', 'ao', 'ss', 'lc', 'si',
    'l', 'pp', 'v', 'TA', 'PX', 'MA', 'sc', 'bu', 'fu', 'eg', 'eb', 'lu', 'PF',
    'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs', 'b',
    'CF', 'jd', 'SR', 'AP', 'CJ', 'lh', 'PK', 'sp',
]

# Assets with < CORE_MIN_YEARS history are treated as "new" — projected onto
# core cluster structure rather than used to build it.
CORE_MIN_YEARS = 10.0

# Clustering cutoffs: {label: distance_threshold}
CLUSTER_CUTOFFS = {
    'broad':  1.6,   # ~4-5 clusters
    'medium': 1.2,   # ~8-10 clusters
    'fine':   0.8,   # ~15+ clusters
}

OUT_DIR = r'C:\dev\data\clustering'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_log_returns(tday: datetime.date,
                     start_date: datetime.date = datetime.date(2008, 1, 1)) -> pd.DataFrame:
    """Load c1 close prices from parquet and return log returns."""
    parquet = f'C:/dev/data/fut_d_{tday.strftime("%Y%m%d")}.parquet'
    df = pd.read_parquet(parquet)

    # Keep c1 close columns only
    c1_cols = [col for col in df.columns.get_level_values(0).unique() if col.endswith('c1')]
    close = (df.loc[:, df.columns.get_level_values(0).isin(c1_cols) &
                    (df.columns.get_level_values(1) == 'close')]
               .droplevel(1, axis=1))
    close.columns = [c[:-2] for c in close.columns]  # strip 'c1' suffix

    # Filter date range
    close = close.loc[pd.to_datetime(start_date):pd.to_datetime(tday)]

    logret = np.log(1 + close.pct_change(fill_method=None))
    return logret


def split_by_history(logret: pd.DataFrame, universe: list,
                     min_years: float = CORE_MIN_YEARS) -> tuple[list, list]:
    """
    Split universe into core (>=min_years history) and new (<min_years).
    Returns (core_assets, new_assets).
    """
    core, new = [], []
    for a in universe:
        if a not in logret.columns:
            continue
        n_days = logret[a].dropna().shape[0]
        if n_days / 252 >= min_years:
            core.append(a)
        else:
            new.append(a)
    print(f'\nHistory split (threshold={min_years}y):')
    print(f'  Core ({len(core)}): {core}')
    print(f'  New  ({len(new)}): {new}')
    return core, new


def project_new_assets(logret: pd.DataFrame, core_assets: list, new_assets: list,
                       core_clusters: pd.DataFrame, n_components: int = 10,
                       out_dir: str = OUT_DIR) -> pd.DataFrame:
    """
    Fit PCA on core assets (using their full history).
    For each new asset, use the overlapping period to project it onto core PCA space,
    then assign it to the cluster whose centroid it is nearest to.
    Returns a DataFrame with cluster assignments for new assets.
    """
    if not new_assets:
        return pd.DataFrame()

    # Fit scaler + PCA on core, using only rows where all core assets have data
    core_data = logret[core_assets].dropna()
    scaler = StandardScaler().fit(core_data)
    core_std = scaler.transform(core_data)
    pca = PCA(n_components=min(n_components, len(core_assets)))
    pca.fit(core_std)

    # Core asset coordinates in PCA space
    core_coords = pd.DataFrame(pca.transform(core_std),
                                index=core_data.index, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

    # Compute cluster centroids (mean PCA coords per cluster label)
    result_rows = []
    for new_asset in new_assets:
        if new_asset not in logret.columns:
            continue
        # Get overlapping period with core
        overlap = logret[core_assets + [new_asset]].dropna()
        if len(overlap) < 60:
            print(f'  {new_asset}: insufficient overlap ({len(overlap)} days), skipping')
            continue

        # Project new asset's contemporaneous core returns into PCA space
        overlap_core_std = scaler.transform(overlap[core_assets])
        overlap_pca_coords = pca.transform(overlap_core_std)  # shape (T, n_pc)

        # Correlation between new asset returns and each PC time series
        new_ret = overlap[new_asset].values
        pc_loadings = np.array([
            np.corrcoef(new_ret, overlap_pca_coords[:, k])[0, 1]
            for k in range(pca.n_components_)
        ])

        # Assign to cluster with most similar PC loading profile (nearest centroid)
        row = {'asset': new_asset, 'n_overlap_days': len(overlap)}
        for col in core_clusters.columns:
            # Centroid = mean PC loadings of assets in each cluster
            cluster_ids = core_clusters[col].unique()
            best_cluster, best_sim = None, -np.inf
            for cid in sorted(cluster_ids):
                members = core_clusters[core_clusters[col] == cid].index.tolist()
                members_in_core = [m for m in members if m in core_assets]
                if not members_in_core:
                    continue
                # Each core asset's loading = correlation with each PC
                member_loadings = np.array([
                    [np.corrcoef(core_data[m].values, core_std[:, k])[0, 1]
                     for k in range(pca.n_components_)]
                    for m in members_in_core
                ])
                centroid = member_loadings.mean(axis=0)
                # Cosine similarity
                sim = np.dot(pc_loadings, centroid) / (
                    np.linalg.norm(pc_loadings) * np.linalg.norm(centroid) + 1e-9)
                if sim > best_sim:
                    best_sim, best_cluster = sim, cid
            row[col] = best_cluster
            row[f'{col}_sim'] = round(best_sim, 3)
        result_rows.append(row)

    if not result_rows:
        return pd.DataFrame()

    proj_df = pd.DataFrame(result_rows).set_index('asset')
    print('\n--- New asset cluster projections ---')
    print(proj_df.to_string())

    os.makedirs(out_dir, exist_ok=True)
    proj_df.to_csv(os.path.join(out_dir, 'new_asset_cluster_projection.csv'))
    return proj_df


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(logret: pd.DataFrame, universe: list, n_components: int = 10,
            out_dir: str = OUT_DIR) -> tuple:
    data = logret[universe].dropna()
    x_std = StandardScaler().fit_transform(data)

    pca = PCA(n_components=n_components)
    pca.fit(x_std)

    os.makedirs(out_dir, exist_ok=True)

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(pca.explained_variance_ratio_,
              index=[f'PC{i+1}' for i in range(n_components)]).plot.bar(ax=ax)
    ax.set_title('PCA Explained Variance Ratio')
    ax.set_ylabel('Proportion')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_scree.png'), dpi=120)
    plt.close(fig)

    # Loadings per PC
    n_cols = 2
    n_rows = (n_components + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    for n in range(n_components):
        loadings = pd.Series(pca.components_[n], index=universe).sort_values()
        loadings.plot.bar(ax=axes[n], color=['red' if v < 0 else 'steelblue' for v in loadings])
        axes[n].set_title(f'PC{n+1} ({pca.explained_variance_ratio_[n]*100:.1f}%)')
        axes[n].axhline(0, color='black', linewidth=0.5)
    for ax in axes[n_components:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_loadings.png'), dpi=120)
    plt.close(fig)

    # ---- Cumulative PCA factor returns ----
    # For each PC, the factor return on day t = dot(standardised_returns[t], loadings[n])
    # This is the return of a long/short portfolio weighted by the eigenvector.
    # We normalise weights so they sum to 1 in absolute value (unit-weight portfolio).
    scaler = StandardScaler().fit(data)

    # Use the FULL logret history (not just the training window) so we can
    # see how each factor evolved over time including before fitting period.
    all_data   = logret[universe].dropna()
    all_x_std  = scaler.transform(all_data)

    pc_returns = pd.DataFrame(
        index=all_data.index,
        columns=[f'PC{n+1}' for n in range(n_components)],
        dtype=float,
    )
    for n in range(n_components):
        weights = pca.components_[n]                   # raw eigenvector
        weights_norm = weights / np.abs(weights).sum() # unit abs-weight
        pc_returns[f'PC{n+1}'] = all_x_std @ weights_norm

    cum_returns = pc_returns.cumsum()

    # ---- Energy crisis periods ----
    CRISIS_PERIODS = {
        'China Coal Crisis\n(Jul-Dec 2021)': (
            pd.Timestamp('2021-07-01'), pd.Timestamp('2021-12-31'),
            '#FF9900', 0.15),
        'Russia-Ukraine\nEnergy Crisis\n(Feb-Aug 2022)': (
            pd.Timestamp('2022-02-24'), pd.Timestamp('2022-08-31'),
            '#CC0000', 0.12),
    }

    def shade_crises(ax, ylims=None):
        """Add shaded rectangles, boundary lines, and labels for each crisis period."""
        for label, (t0, t1, color, _) in CRISIS_PERIODS.items():
            t0 = max(t0, cum_returns.index[0])
            t1 = min(t1, cum_returns.index[-1])
            if t0 >= t1:
                continue
            ax.axvspan(t0, t1, alpha=0.25, color=color, zorder=0)
            ax.axvline(t0, color=color, linewidth=1.2, linestyle='--', zorder=1, alpha=0.8)
            ax.axvline(t1, color=color, linewidth=1.2, linestyle='--', zorder=1, alpha=0.8)
            if ylims:
                y_range = ylims[1] - ylims[0]
                y_pos   = ylims[1] - y_range * 0.01
                ax.text(t0 + (t1 - t0) / 2, y_pos, label,
                        fontsize=6.5, ha='center', va='top',
                        color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color,
                                  alpha=0.7, linewidth=0.8))

    # ---- Crisis performance table ----
    def crisis_perf_table(pc_returns: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for label, (t0, t1, _, _) in CRISIS_PERIODS.items():
            window = pc_returns[(pc_returns.index >= t0) & (pc_returns.index <= t1)]
            if len(window) < 3:
                continue
            perf = window.sum()
            perf.name = label.replace('\n', ' ')
            rows.append(perf)
        return pd.DataFrame(rows)

    perf_table = crisis_perf_table(pc_returns)
    print('\n--- PCA Factor Performance During Energy Crises ---')
    print(perf_table.round(3).to_string())
    perf_table.round(3).to_csv(os.path.join(out_dir, 'pca_crisis_performance.csv'))

    # Plot: all PCs on one overview with crisis shading
    fig, ax = plt.subplots(figsize=(16, 7))
    for n in range(n_components):
        col = f'PC{n+1}'
        ax.plot(cum_returns.index, cum_returns[col],
                label=f'{col} ({pca.explained_variance_ratio_[n]*100:.1f}%)',
                linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.5)
    ylims = ax.get_ylim()
    shade_crises(ax, ylims)
    ax.set_title('Cumulative PCA Factor Returns — with Energy Crisis periods highlighted')
    ax.set_ylabel('Cumulative log-return')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_factor_cumret_overview.png'), dpi=120)
    plt.close(fig)

    # Individual panels with crisis shading + bar chart of crisis returns
    n_cols = 2
    n_rows = (n_components + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()
    for n in range(n_components):
        col     = f'PC{n+1}'
        cr      = cum_returns[col]
        loadings_sorted = pd.Series(pca.components_[n], index=universe).sort_values()

        ax_l = axes[n]
        ax_l.plot(cr.index, cr.values, color='steelblue', linewidth=1.2, zorder=2)
        ax_l.axhline(0, color='black', linewidth=0.4)

        ylims = ax_l.get_ylim()
        shade_crises(ax_l, ylims)

        # Crisis-period cumulative return annotations
        if len(perf_table):
            crisis_vals = perf_table[col] if col in perf_table.columns else pd.Series()
            annot_lines = '  '.join([
                f"{name.split('(')[0].strip()}: {v:+.2f}"
                for name, v in crisis_vals.items()
            ])
        else:
            annot_lines = ''

        top_long  = loadings_sorted.tail(4).index.tolist()
        top_short = loadings_sorted.head(4).index.tolist()
        var_pct   = pca.explained_variance_ratio_[n] * 100
        ax_l.set_title(
            f'{col} ({var_pct:.1f}%)\n'
            f'Long: {top_long}  Short: {top_short}\n'
            f'{annot_lines}',
            fontsize=7.5)
        ax_l.set_ylabel('Cum. log-return', fontsize=8)

    for ax in axes[n_components:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_factor_cumret_individual.png'), dpi=130)
    plt.close(fig)

    print(f'PCA: top {n_components} PCs explain '
          f'{pca.explained_variance_ratio_[:5].sum()*100:.1f}% (first 5)')
    print(f'PCA factor cumulative return plots saved')
    return pca, pc_returns


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation(logret: pd.DataFrame, universe: list,
                     out_dir: str = OUT_DIR) -> pd.DataFrame:
    # Use pairwise correlation (min_periods=120) to avoid dropna bias from short-history assets
    data = logret[universe]
    corr = data.corr(min_periods=120)

    os.makedirs(out_dir, exist_ok=True)
    size = max(12, len(universe) // 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(corr, cmap='coolwarm', vmin=-0.2, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title('Commodity Correlation Matrix', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'correlation_heatmap.png'), dpi=120)
    plt.close(fig)

    print(f'Correlation matrix saved ({len(universe)} assets)')
    return corr


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------

def cluster_commodities(corr: pd.DataFrame,
                        cutoffs: dict = CLUSTER_CUTOFFS,
                        out_dir: str = OUT_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # Use Euclidean distance on rows of correlation matrix (matches notebook)
    dist = pdist(corr)
    linkage = sch.linkage(dist, method='average')

    # Dendrogram
    fig, ax = plt.subplots(figsize=(max(18, len(corr) // 2), 8))
    sch.dendrogram(linkage, labels=corr.columns.tolist(),
                   leaf_rotation=90, leaf_font_size=9, ax=ax)
    ax.set_title('Hierarchical Clustering — China Futures', fontsize=14)
    ax.set_xlabel('Product')
    ax.set_ylabel('Distance (Euclidean on correlation rows)')
    # Draw cutoff lines
    colors = ['red', 'orange', 'green']
    for (label, threshold), color in zip(cutoffs.items(), colors):
        ax.axhline(threshold, color=color, linestyle='--', linewidth=1.2,
                   label=f'{label} cutoff={threshold}')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dendrogram.png'), dpi=120)
    plt.close(fig)

    # Cluster assignments at each cutoff
    result = pd.DataFrame(index=corr.columns)
    for label, threshold in cutoffs.items():
        labels = sch.fcluster(linkage, threshold, criterion='distance')
        # Re-label clusters by first occurrence (more intuitive ordering)
        seen = {}
        relabeled = []
        counter = 1
        for lbl in labels:
            if lbl not in seen:
                seen[lbl] = counter
                counter += 1
            relabeled.append(seen[lbl])
        result[label] = relabeled
        n_clusters = len(set(relabeled))
        print(f'  {label:8s} (cutoff={threshold}): {n_clusters} clusters')

    # Print cluster membership
    for label in cutoffs:
        print(f'\n--- {label.upper()} clusters ---')
        for cluster_id in sorted(result[label].unique()):
            members = result[result[label] == cluster_id].index.tolist()
            print(f'  Cluster {cluster_id}: {members}')

    # Save
    out_csv = os.path.join(out_dir, 'cluster_assignments.csv')
    result.to_csv(out_csv)
    print(f'\nCluster assignments saved to {out_csv}')

    return result

# ---------------------------------------------------------------------------
# Graph-based Louvain community detection
# ---------------------------------------------------------------------------

def louvain_clusters(corr: pd.DataFrame, min_edge_weight: float = 0.2,
                     out_dir: str = OUT_DIR) -> pd.Series:
    """
    Build a weighted correlation graph (edges = corr > threshold) and apply
    Louvain community detection. No k or cutoff required — finds natural
    communities by maximising modularity.
    """
    assets = corr.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if i >= j:
                continue
            w = corr.loc[a, b]
            if w > min_edge_weight:
                G.add_edge(a, b, weight=w)

    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    result = pd.Series(partition, name='louvain')

    # Re-label by first appearance
    seen, counter, relabeled = {}, 1, {}
    for asset in assets:
        lbl = partition[asset]
        if lbl not in seen:
            seen[lbl] = counter
            counter += 1
        relabeled[asset] = seen[lbl]
    result = pd.Series(relabeled, name='louvain')

    n_clusters = len(set(relabeled.values()))
    print(f'\nLouvain community detection: {n_clusters} communities '
          f'(min_edge_weight={min_edge_weight})')
    for cid in sorted(set(relabeled.values())):
        members = [a for a, v in relabeled.items() if v == cid]
        print(f'  Community {cid}: {members}')

    return result


# ---------------------------------------------------------------------------
# Minimum Spanning Tree
# ---------------------------------------------------------------------------

def plot_mst(corr: pd.DataFrame, cluster_labels: pd.Series = None,
             out_dir: str = OUT_DIR):
    """
    Build the Minimum Spanning Tree on correlation-distance graph.
    Prunes to the most structurally important pairwise links.
    Node colour = cluster label if provided.
    """
    assets = corr.columns.tolist()
    dist_matrix = 1 - corr.values  # correlation distance

    G = nx.Graph()
    G.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if i < j:
                G.add_edge(a, b, weight=dist_matrix[i, j])

    mst = nx.minimum_spanning_tree(G, weight='weight')

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(mst, seed=42, k=2.5)

    if cluster_labels is not None:
        n_clusters = cluster_labels.max()
        cmap = cm.get_cmap('tab20', n_clusters)
        node_colors = [cmap(cluster_labels.get(a, 0) - 1) for a in mst.nodes()]
    else:
        node_colors = 'steelblue'

    nx.draw_networkx(mst, pos=pos, ax=ax,
                     node_color=node_colors, node_size=600,
                     font_size=9, font_weight='bold',
                     edge_color='grey', width=1.2)

    # Label edges with correlation (not distance)
    edge_labels = {(u, v): f"{corr.loc[u, v]:.2f}"
                   for u, v in mst.edges()}
    nx.draw_networkx_edge_labels(mst, pos=pos, edge_labels=edge_labels,
                                 font_size=7, ax=ax)

    ax.set_title('Minimum Spanning Tree — Commodity Correlation Structure', fontsize=13)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mst.png'), dpi=130)
    plt.close(fig)
    print('MST saved')


# ---------------------------------------------------------------------------
# Rolling correlation stability
# ---------------------------------------------------------------------------

def rolling_corr_stability(logret: pd.DataFrame, universe: list,
                            window: int = 60, threshold: float = 0.3,
                            out_dir: str = OUT_DIR) -> pd.DataFrame:
    """
    For each pair, compute the fraction of rolling windows where corr > threshold.
    High stability (> 0.8) means the pair is reliably correlated — safe to cluster.
    Low stability means the correlation is regime-dependent.
    """
    data = logret[universe].dropna()
    n = len(universe)
    stability = pd.DataFrame(0.0, index=universe, columns=universe)

    roll_corrs = []
    for end in range(window, len(data) + 1):
        chunk = data.iloc[end - window: end]
        roll_corrs.append(chunk.corr().values)

    roll_corrs = np.array(roll_corrs)  # shape: (n_windows, n_assets, n_assets)
    stability_matrix = (roll_corrs > threshold).mean(axis=0)
    stability = pd.DataFrame(stability_matrix, index=universe, columns=universe)

    os.makedirs(out_dir, exist_ok=True)
    size = max(12, n // 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(stability, cmap='RdYlGn', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046,
                 label=f'Fraction of {window}d windows with corr > {threshold}')
    ax.set_xticks(range(n))
    ax.set_xticklabels(universe, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(universe, fontsize=8)
    ax.set_title(f'Rolling Correlation Stability ({window}d window, threshold={threshold})',
                 pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rolling_corr_stability.png'), dpi=120)
    plt.close(fig)

    # Print most/least stable pairs
    pairs = []
    for i, a in enumerate(universe):
        for j, b in enumerate(universe):
            if i < j:
                pairs.append((a, b, stability.loc[a, b]))
    pairs_df = pd.DataFrame(pairs, columns=['a', 'b', 'stability']).sort_values('stability', ascending=False)
    print(f'\nMost stable pairs (corr > {threshold} in >{threshold*100:.0f}% of {window}d windows):')
    print(pairs_df.head(10).to_string(index=False))
    print(f'\nLeast stable pairs:')
    print(pairs_df.tail(10).to_string(index=False))

    return stability


# ---------------------------------------------------------------------------
# DBSCAN — outlier detection
# ---------------------------------------------------------------------------

def dbscan_clusters(corr: pd.DataFrame, eps: float = 0.5, min_samples: int = 2,
                    out_dir: str = OUT_DIR) -> pd.Series:
    """
    DBSCAN on correlation-distance matrix. Assets labelled -1 are outliers
    that don't belong to any stable cluster.
    """
    dist_matrix = squareform(pdist(corr.values, metric='correlation'))
    np.fill_diagonal(dist_matrix, 0)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)
    result = pd.Series(labels, index=corr.columns, name='dbscan')

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    print(f'\nDBSCAN (eps={eps}): {n_clusters} clusters, {n_outliers} outliers')
    outliers = corr.columns[labels == -1].tolist()
    if outliers:
        print(f'  Outlier assets: {outliers}')
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        members = corr.columns[labels == cid].tolist()
        print(f'  Cluster {cid+1}: {members}')

    return result



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_date: datetime.date, universe: list = None, sector: str = None,
         core_min_years: float = CORE_MIN_YEARS, show_plots: bool = False):
    # Determine universe and output folder
    if sector is not None:
        if sector not in SECTORS:
            raise ValueError(f'Unknown sector "{sector}". Choose from: {list(SECTORS.keys())}')
        universe = SECTORS[sector]
        out_dir  = os.path.join(OUT_DIR, sector)
        # For a sector analysis use a lower core threshold — assets within a
        # sector often launched together; use min history that captures the
        # majority (default 5y for sector runs)
        if core_min_years == CORE_MIN_YEARS:
            core_min_years = 5.0
        print(f'\nSector analysis: {sector}  ({len(universe)} assets)')
    else:
        if universe is None:
            universe = DEFAULT_UNIVERSE
        out_dir = OUT_DIR

    print(f'Loading data up to {run_date} ...')
    logret = load_log_returns(run_date)

    available = [a for a in universe if a in logret.columns]
    missing = [a for a in universe if a not in logret.columns]
    if missing:
        print(f'Warning: not found in parquet: {missing}')

    # Split into core and new based on history length
    core_assets, new_assets = split_by_history(logret, available, min_years=core_min_years)

    n_pca = min(10, len(core_assets) - 1)

    # ---- All analytics run on CORE assets only ----

    # 1. PCA
    print(f'\n--- PCA (core: {len(core_assets)} assets) ---')
    pca, pc_returns = run_pca(logret, core_assets, n_components=n_pca, out_dir=out_dir)

    # 2. Correlation matrix
    print('\n--- Correlation Heatmap (core) ---')
    corr = plot_correlation(logret, core_assets, out_dir=out_dir)

    # 3. Hierarchical clustering
    print('\n--- Hierarchical Clustering (core) ---')
    hier_clusters = cluster_commodities(corr, out_dir=out_dir)

    # 4. Louvain
    print('\n--- Louvain Community Detection (core) ---')
    louvain = louvain_clusters(corr, min_edge_weight=0.2, out_dir=out_dir)

    # 5. MST
    print('\n--- Minimum Spanning Tree (core) ---')
    plot_mst(corr, cluster_labels=hier_clusters['medium'], out_dir=out_dir)

    # 6. Rolling correlation stability
    print('\n--- Rolling Correlation Stability (core) ---')
    rolling_corr_stability(logret, core_assets, window=60, threshold=0.3, out_dir=out_dir)

    # 7. DBSCAN
    print('\n--- DBSCAN (core) ---')
    dbscan = dbscan_clusters(corr, eps=0.5, min_samples=2, out_dir=out_dir)

    # Merge core cluster assignments
    core_clusters = hier_clusters.copy()
    core_clusters['louvain'] = louvain
    core_clusters['dbscan'] = dbscan
    out_csv = os.path.join(out_dir, 'cluster_assignments_core.csv')
    os.makedirs(out_dir, exist_ok=True)
    core_clusters.to_csv(out_csv)
    print(f'\nCore cluster assignments saved to {out_csv}')
    print(core_clusters.to_string())

    # 8. Project new assets onto core cluster structure
    if new_assets:
        print(f'\n--- Projecting new assets onto core clusters ---')
        project_new_assets(logret, core_assets, new_assets, core_clusters,
                           n_components=n_pca, out_dir=out_dir)

    if show_plots:
        import subprocess
        for png in ['pca_scree.png', 'pca_loadings.png', 'pca_factor_cumret_overview.png',
                    'correlation_heatmap.png', 'dendrogram.png', 'mst.png',
                    'rolling_corr_stability.png']:
            path = os.path.join(out_dir, png)
            if os.path.exists(path):
                subprocess.Popen(['start', path], shell=True)

    return corr, core_clusters


if __name__ == '__main__':
    args = sys.argv[1:]
    show   = '--show' in args
    sector = next((a.split('=')[1] for a in args if a.startswith('--sector=')), None)
    date_args = [a for a in args if not a.startswith('--')]
    run_date = (datetime.datetime.strptime(date_args[0], '%Y%m%d').date()
                if date_args else datetime.date.today())

    corr, clusters = main(run_date, sector=sector, show_plots=show)
    print('\nDone. Outputs in', os.path.join(OUT_DIR, sector) if sector else OUT_DIR)
