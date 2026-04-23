# merged_df has one row per (gsis_id, season)
# with at least:
#   'cluster'  (0..5)
#   'injured_any'  (0/1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cluster_odds_ratios(df, cluster_col='cluster', injury_col='injured_any'):
    """
    Compute injury rate, odds, and odds ratio vs a reference cluster for each cluster.

    Returns:
        stats_df with columns:
          cluster, n, injured, not_injured, rate, odds,
          ref_cluster, odds_ratio, or_ci_low, or_ci_high
    """
    # aggregate counts
    agg = (
        df.groupby(cluster_col)[injury_col]
        .agg(['sum', 'count'])
        .reset_index()
        .rename(columns={'sum': 'injured', 'count': 'n'})
    )
    agg['not_injured'] = agg['n'] - agg['injured']
    agg['rate'] = agg['injured'] / agg['n']
    agg['odds'] = agg['injured'] / agg['not_injured']

    # choose reference cluster: lowest injury rate
    ref_row = agg.loc[agg['rate'].idxmin()]
    ref_cluster = int(ref_row[cluster_col])
    a_ref = ref_row['injured']
    b_ref = ref_row['not_injured']
    odds_ref = ref_row['odds']

    # compute OR + 95% CI
    or_list = []
    or_low_list = []
    or_high_list = []
    ref_list = []

    for _, row in agg.iterrows():
        a = row['injured']
        b = row['not_injured']

        # odds ratio vs reference
        or_val = (a / b) / odds_ref

        # log(OR) standard error: sqrt(1/a + 1/b + 1/a_ref + 1/b_ref)
        # add a tiny epsilon to avoid division-by-zero if any cell is 0
        eps = 1e-9
        se_log_or = np.sqrt(
            1.0 / (a + eps) +
            1.0 / (b + eps) +
            1.0 / (a_ref + eps) +
            1.0 / (b_ref + eps)
        )

        z = 1.96  # 95% CI
        log_or = np.log(or_val)
        ci_low = np.exp(log_or - z * se_log_or)
        ci_high = np.exp(log_or + z * se_log_or)

        or_list.append(or_val)
        or_low_list.append(ci_low)
        or_high_list.append(ci_high)
        ref_list.append(ref_cluster)

    agg['ref_cluster'] = ref_list
    agg['odds_ratio'] = or_list
    agg['or_ci_low'] = or_low_list
    agg['or_ci_high'] = or_high_list

    return agg


def plot_injury_rate_diff_from_mean(df, cluster_col='cluster', injury_col='injured_any'):
    summary = (
        df.groupby(cluster_col)[injury_col]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'rate', 'count': 'n'})
    )

    global_rate = df[injury_col].mean()
    summary['rate_diff'] = summary['rate'] - global_rate

    x = summary[cluster_col].values
    y = summary['rate_diff'].values

    plt.figure()
    plt.axhline(0.0)  # global mean
    plt.bar(x, y)
    plt.xlabel('Cluster')
    plt.ylabel('Injury rate - global mean')
    plt.title('Injury Rate Difference from Global Mean by Cluster')
    plt.xticks(x)
    plt.tight_layout()
    plt.show()


def plot_cluster_odds_ratio_forest(or_stats, cluster_col='cluster'):
    """
    or_stats: output of compute_cluster_odds_ratios
    """
    # sort by cluster index for nicer plotting
    stats = or_stats.sort_values(cluster_col)

    x = stats['odds_ratio'].values
    x_low = stats['or_ci_low'].values
    x_high = stats['or_ci_high'].values
    clusters = stats[cluster_col].values
    ref_cluster = int(stats['ref_cluster'].iloc[0])

    # positions on y-axis
    y_pos = np.arange(len(clusters))

    plt.figure(figsize=(8, 4))
    # vertical line at OR = 1
    plt.axvline(1.0, linestyle='--')

    # central points
    plt.scatter(x, y_pos)

    # CI error bars
    for i in range(len(clusters)):
        plt.plot([x_low[i], x_high[i]], [y_pos[i], y_pos[i]])

    plt.yticks(y_pos, [f"Cluster {c}" + (" (ref)" if c == ref_cluster else "") for c in clusters])
    plt.xlabel('Odds Ratio (injury vs ref cluster)')
    plt.title('Injury Odds Ratio by Cluster')
    plt.tight_layout()
    plt.show()


def plot_real_vs_synthetic(real_player, synthetic_player, player_idx, training_year):
    """
    Plot real vs synthetic player injury trajectories on the same graph.

    Parameters
    ----------
    real_player : array-like
        Shape (n_weeks,), real player's injury data
    synthetic_player : array-like
        Shape (n_weeks,), synthetic player's data
    player_idx : int, optional
        Player index for labeling
    """

    real = np.asarray(real_player)
    synth = np.asarray(synthetic_player)

    if real.shape != synth.shape:
        raise ValueError("real_player and synthetic_player must have the same shape")

    weeks = np.arange(1, len(real) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(weeks, real, marker='o', linewidth=2, label="Real Player")
    plt.plot(weeks, synth, marker='s', linestyle='--', linewidth=2, label="Synthetic Player")

    plt.axvline(x=training_year, color='black', linestyle='--', linewidth=2, label='Train/Test Split')

    plt.xlabel("Week")
    plt.ylabel("Injury Severity / Games Missed")
    
    if player_idx is not None:
        plt.title(f"Player {player_idx}: Real vs Synthetic Injury Trajectory")
    else:
        plt.title("Real vs Synthetic Player")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
