import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_cluster_feature_profiles(df, feature_cols, cluster_col='cluster'):
    """
    Compute standardized (z-score) feature profiles per cluster.

    For each feature and cluster:
      z = (cluster_mean - global_mean) / global_std

    Returns:
        profiles_df: long-form DataFrame with:
            [cluster, feature, cluster_mean, global_mean, global_std, z_score]
    """
    # global stats
    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std().replace(0, np.nan)

    rows = []
    for cluster_id, group in df.groupby(cluster_col):
        cluster_means = group[feature_cols].mean()
        for feat in feature_cols:
            gm = global_means[feat]
            gs = global_stds[feat]
            cm = cluster_means[feat]
            z = (cm - gm) / gs if pd.notna(gs) else 0.0
            rows.append({
                'cluster': cluster_id,
                'feature': feat,
                'cluster_mean': cm,
                'global_mean': gm,
                'global_std': gs,
                'z_score': z
            })
    profiles_df = pd.DataFrame(rows)
    return profiles_df


def print_cluster_summaries(profiles_df, top_n=5):
    """
    Print a human-readable interpretation of clusters based on z-scores.

    For each cluster:
      - top_n most positive features
      - top_n most negative features
    """
    for cluster_id, group in profiles_df.groupby('cluster'):
        print(f"\n=== Cluster {cluster_id} ===")

        # Top features where this cluster is above average
        top_pos = group.sort_values('z_score', ascending=False).head(top_n)

        # Features where this cluster is below average
        top_neg = group.sort_values('z_score', ascending=True).head(top_n)

        print("  Most ABOVE-average features:")
        for _, row in top_pos.iterrows():
            print(f"    {row['feature']}: z = {row['z_score']:.2f} "
                  f"(cluster_mean={row['cluster_mean']:.3f}, global_mean={row['global_mean']:.3f})")

        print("  Most BELOW-average features:")
        for _, row in top_neg.iterrows():
            print(f"    {row['feature']}: z = {row['z_score']:.2f} "
                  f"(cluster_mean={row['cluster_mean']:.3f}, global_mean={row['global_mean']:.3f})")


def plot_cluster_injury_rate(df, cluster_col='cluster', injury_col='injured_any'):
    """
    Plot injury rate (fraction of injured players) for each cluster.
    """
    summary = (
        df
        .groupby(cluster_col)
        .agg(
            n_players=('gsis_id', 'nunique'),
            injury_rate=(injury_col, 'mean')
        )
        .reset_index()
    )

    x = summary[cluster_col].values
    y = summary['injury_rate'].values

    plt.figure()
    plt.bar(x, y)
    plt.xlabel('Cluster')
    plt.ylabel('Injury Rate')
    plt.title('Injury Rate by Cluster')
    plt.xticks(x)
    plt.tight_layout()
    plt.show()


def plot_cluster_injury_severity(df, cluster_col='cluster',
                                 weeks_injury_col='weeks_with_injury',
                                 weeks_out_col='weeks_out'):
    """
    Plot mean weeks_with_injury and weeks_out per cluster (two separate bar charts).
    """
    summary = (
        df
        .groupby(cluster_col)
        .agg(
            mean_weeks_with_injury=(weeks_injury_col, 'mean'),
            mean_weeks_out=(weeks_out_col, 'mean')
        )
        .reset_index()
    )

    x = summary[cluster_col].values

    # Mean weeks with injury
    plt.figure()
    plt.bar(x, summary['mean_weeks_with_injury'].values)
    plt.xlabel('Cluster')
    plt.ylabel('Mean weeks with injury')
    plt.title('Mean Weeks with Injury by Cluster')
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

    # Mean weeks out
    plt.figure()
    plt.bar(x, summary['mean_weeks_out'].values)
    plt.xlabel('Cluster')
    plt.ylabel('Mean weeks out')
    plt.title('Mean Weeks Out by Cluster')
    plt.xticks(x)
    plt.tight_layout()
    plt.show()


def plot_pca_clusters(X_pca, labels):
    """
    Scatter plot of the first two PCA components, colored by cluster label.
    """
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Player Stats Colored by Cluster')
    plt.tight_layout()
    plt.show()


def plot_cluster_feature_heatmap(profiles_df, features_to_show=None):
    """
    Plot a heatmap of z-scores for clusters vs features.

    If features_to_show is None, use the top-10 most variable features.
    """
    # Pivot to matrix: rows = clusters, cols = features, values = z_score
    pivot = profiles_df.pivot(index='cluster', columns='feature', values='z_score')

    # Optionally select subset of features
    if features_to_show is None:
        # choose features with highest std of z-score across clusters
        stds = pivot.std(axis=0).sort_values(ascending=False)
        features_to_show = stds.head(10).index.tolist()

    pivot_sub = pivot[features_to_show]

    plt.figure(figsize=(1 + 0.6 * len(features_to_show), 4))
    plt.imshow(pivot_sub.values, aspect='auto')
    plt.colorbar(label='z-score')
    plt.xticks(ticks=range(len(features_to_show)), labels=features_to_show, rotation=45, ha='right')
    plt.yticks(ticks=range(len(pivot_sub.index)), labels=pivot_sub.index)
    plt.title('Cluster Feature Z-Score Heatmap')
    plt.tight_layout()
    plt.show()


def describe_cluster(profiles_df, cluster_id, top_n=5):
    """
    Print a human-readable interpretation of one cluster based on z-scores.
    Groups features by 'type' (passing, rushing, receiving, physical, misc).
    """
    group = profiles_df[profiles_df['cluster'] == cluster_id].copy()
    if group.empty:
        print(f"No data for cluster {cluster_id}")
        return

    # Sort by z_score to find above/below average features
    top_pos = group.sort_values('z_score', ascending=False).head(top_n)
    top_neg = group.sort_values('z_score', ascending=True).head(top_n)

    print(f"\n============================")
    print(f"Cluster {cluster_id} summary")
    print(f"============================")

    def classify_feature(feat):
        f = feat.lower()
        if 'pass' in f:
            return 'passing'
        if 'rush' in f or 'carry' in f:
            return 'rushing'
        if 'receiv' in f or 'target' in f:
            return 'receiving'
        if f in ['height', 'weight', 'age', 'experience']:
            return 'physical'
        if 'sack' in f or 'fumble' in f or 'penalt' in f:
            return 'risk/ball_security'
        return 'other'

    print("\nMost ABOVE-average stats (z > 0):")
    for _, row in top_pos.iterrows():
        feat = row['feature']
        z = row['z_score']
        ftype = classify_feature(feat)
        print(f"  - {feat} [{ftype}]: z = {z:.2f} "
              f"(cluster_mean={row['cluster_mean']:.3f}, global_mean={row['global_mean']:.3f})")

    print("\nMost BELOW-average stats (z < 0):")
    for _, row in top_neg.iterrows():
        feat = row['feature']
        z = row['z_score']
        ftype = classify_feature(feat)
        print(f"  - {feat} [{ftype}]: z = {z:.2f} "
              f"(cluster_mean={row['cluster_mean']:.3f}, global_mean={row['global_mean']:.3f})")

    print("\nInterpretation guide:")
    print("  • High positive z in 'rushing' → heavy runners / high rushing workload.")
    print("  • High positive z in 'passing' → high-volume passers.")
    print("  • High positive z in 'physical' → older, heavier, taller, more experienced.")
    print("  • High positive z in 'risk/ball_security' → more sacks, fumbles, penalties, etc.")


def compare_clusters_means(df, feature_cols, clusters_to_compare):
    """
    Show mean of each feature for selected clusters side-by-side.
    """
    sub = df[df['cluster'].isin(clusters_to_compare)]
    summary = (
        sub
        .groupby('cluster')[feature_cols]
        .mean()
        .T  # features as rows
    )
    return summary




def plot_pca_3d(X_pca, labels=None, title='First 3 Principal Components (3D)', 
                elev=30, azim=45):
    """
    Interactive-style 3D scatter plot of the first three principal components.
    
    Parameters
    ----------
    X_pca : np.ndarray
        Full PCA-transformed feature matrix (n_samples, n_components).
        The function will use the first 3 columns.
    labels : array-like, optional
        Cluster labels to color points by. If None, all points will be the same color.
    title : str
        Title for the plot.
    elev : float
        Elevation angle (degrees) for the 3D view.
    azim : float
        Azimuthal angle (degrees) for the 3D view.
    """
    X_3d = X_pca[:, :3]
    # print("3D: ", X_3d)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                             c=labels, cmap='tab10', alpha=0.7, s=20)
        plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.6)
    else:
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                   alpha=0.7, s=20, color='steelblue')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    plt.show()
