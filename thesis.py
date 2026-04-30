import nflreadpy as nfl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=1000)

from format import (
    append_player_bio,
    attach_snap_counts,
    average_player_stats,
    average_snap_count_fields,
    collect_player_stats,
)
from chat_method import (
    aggregate_injury_flags,
    chi_square_cluster_injury_test,
    cluster_on_first_3_pcs,
    drop_irrelevant_stats,
    prepare_feature_matrix,
    run_kmeans_clusters,
    scale_and_pca,
    stats_dict_to_df,
    explain_pca,
)
from interpret import (
    compare_clusters_means,
    compute_cluster_feature_profiles,
    describe_cluster,
    plot_cluster_feature_heatmap,
    plot_cluster_injury_rate,
    plot_cluster_injury_severity,
    plot_pca_clusters,
    plot_pca_3d,  # Add this
    print_cluster_summaries,
    plot_pca_clusters_3d,
)

from more_visuals import (
    compute_cluster_odds_ratios,
    plot_cluster_odds_ratio_forest,
    plot_injury_rate_diff_from_mean,
    plot_real_vs_synthetic,
)
from injury_probability import (
    append_career_year_index,
    fit_injury_logistic_model,
    predict_player_injury_history,
    plot_injury_trajectory,
)

from synthetic_control import build_synthetic_player
from qb_absence_matrix import build_qb_weeks_out_matrix, compute_qb_matrix_svd_pca, plot_svd_vectors, calc_pc_from_vt, build_qb_injury_severity_matrix

injury_ranks = {'Limited Participation in Practice': 1, 
                'Full Participation in Practice': 0, 
                'Did Not Participate In Practice': 2, 
                'Out (Definitely Will Not Play)': 5}

    

def get_stats(years=None):
    if years is None:
        years = [2020]

    player_stats = nfl.load_player_stats(years)
    pbp_player = player_stats.to_pandas()
    injuries = nfl.load_injuries(years)
    pbp_injuries = injuries.to_pandas()
    snap_counts_polars = nfl.load_snap_counts(years)
    snap_counts = snap_counts_polars.to_pandas()
    players_polars = nfl.load_players()
    players = players_polars.to_pandas()
    return pbp_player, pbp_injuries, snap_counts, players

def plot_cluster_sizes(labels, save_path="cluster_sizes.png"):
    """
    Creates a bar chart showing number of players in each cluster.
    """

    clusters = [0, 1, 2, 3]
    values = [447, 422, 345, 403]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, values)

    # Add labels on top of bars
    # for bar in bars:
    #     plt.text(
    #         bar.get_x() + bar.get_width()/2,
    #         bar.get_height(),
    #         "400",
    #         ha='center',
    #         va='bottom'
    #     )

    plt.xlabel("Cluster")
    plt.ylabel("Number of Players")
    plt.title("Cluster Sizes")

    plt.xticks(clusters)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def run_clustering_workflow(final_vector, injuries):
    # 1. Dict → DataFrame
    player_df = stats_dict_to_df(final_vector)

    # 2. Clean → X
    clean_df, X, feature_cols = prepare_feature_matrix(player_df)

    # 3. Scale + PCA
    feature_names = list(player_df.columns)
    X_scaled, scaler, X_pca, pca = scale_and_pca(X, var_explained=0.9)
    explain_pca(pca, feature_names)

    # 4. Cluster
    labels, kmeans_model = run_kmeans_clusters(X_pca, n_clusters=4)
    clean_df['cluster'] = labels

    # 5. Injury flags (injuries is your pbp_injury df)
    injury_df = aggregate_injury_flags(injuries)

    # 6. Merge clusters with injury info
    merged_df = clean_df.merge(
        injury_df,
        on=['gsis_id', 'season'],
        how='left'
    )


    # Fill NaNs for players with no injury record
    for col in ['injured_any', 'weeks_with_injury', 'weeks_out', 'weeks_questionable', 'weeks_dnp_injury']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0).astype(int)

    cluster_summary = (
        merged_df
        .groupby('cluster')
        .agg(
            n_players=('gsis_id', 'nunique'),
            injured_players=('injured_any', 'sum'),
            injury_rate=('injured_any', 'mean')
        )
        .reset_index()
    )

    print(cluster_summary)
    # 7. Significance test
    chi2, p, dof, expected, table = chi_square_cluster_injury_test(merged_df)
    print(table)
    print(f"chi2 = {chi2:.3f}, p = {p:.5f}")

    severity_summary = (
        merged_df
        .groupby('cluster')
        .agg(
            mean_weeks_with_injury=('weeks_with_injury', 'mean'),
            mean_weeks_out=('weeks_out', 'mean'),
        )
    )
    print(severity_summary)

    plot_cluster_sizes(labels)

    return clean_df, merged_df, feature_cols, X_pca, labels


def view(merged_df, X_pca, clean_df, profiles_df):
    # After you have merged_df and profiles_df

    plot_cluster_injury_rate(merged_df)
    plot_cluster_injury_severity(merged_df)
    plot_pca_clusters(X_pca, clean_df['cluster'].values)
    plot_pca_clusters_3d(X_pca, clean_df['cluster'].values)
    plot_cluster_feature_heatmap(profiles_df)


def cluster_and_analyze(final_vector, injuries):
    clean_df, merged_df, feature_cols, X_pca, labels = run_clustering_workflow(final_vector, injuries)
    profiles_df = compute_cluster_feature_profiles(clean_df, feature_cols, cluster_col='cluster')
    print_cluster_summaries(profiles_df, top_n=5)
    view(merged_df, X_pca, clean_df, profiles_df)
    # High injury clusters
    describe_cluster(profiles_df, cluster_id=5, top_n=8)
    describe_cluster(profiles_df, cluster_id=0, top_n=8)

    # Low injury clusters
    describe_cluster(profiles_df, cluster_id=4, top_n=8)
    describe_cluster(profiles_df, cluster_id=1, top_n=8)

    # Example: high vs low
    high_vs_low = compare_clusters_means(clean_df, feature_cols, clusters_to_compare=[5, 0, 4, 1])
    print(high_vs_low.head(20))  # first 20 features

    # Odds ratios and other visualizations
    or_stats = compute_cluster_odds_ratios(merged_df, cluster_col='cluster', injury_col='injured_any')
    print(or_stats)
    or_stats = compute_cluster_odds_ratios(merged_df)
    plot_injury_rate_diff_from_mean(merged_df)
    plot_cluster_odds_ratio_forest(or_stats)

def plot_player_injury_likelihood(final_vector, injuries, target_gsis_id):
    """
    Fit a simple injury model and plot predicted injury probability over a player's career.
    """
    stats_df = stats_dict_to_df(final_vector)

    injury_flags = aggregate_injury_flags(injuries)
    model_df = stats_df.merge(
        injury_flags[['gsis_id', 'season', 'injured_any']],
        on=['gsis_id', 'season'],
        how='left'
    )
    model_df['injured_any'] = model_df['injured_any'].fillna(0).astype(int)

    model_df = append_career_year_index(model_df, player_col='gsis_id', season_col='season', out_col='career_year')

    logit_fit = fit_injury_logistic_model(
        model_df,
        formula='injured_any ~ career_year',
        cluster_se=True,
        cluster_col='gsis_id'
    )

    player_traj = predict_player_injury_history(
        logit_fit,
        model_df,
        player_id=target_gsis_id,
        player_col='gsis_id',
        season_col='season',
        career_year_col='career_year',
        pred_col='pred_injury_prob'
    )

    plot_injury_trajectory(
        player_traj,
        x_col='career_year',
        y_col='pred_injury_prob',
        title=f'Predicted injury likelihood: {target_gsis_id}'
    )

    return player_traj


def build_player_injury_history(injuries, pbp_player, target_gsis_id):
    """Return per-season injury metrics (including severity) for one player."""
    injury_flags = aggregate_injury_flags(injuries)

    seasons_in_stats = (
        pbp_player.loc[pbp_player["player_id"] == target_gsis_id, "season"]
        .dropna()
        .astype(int)
        .unique()
    )
    stats_df = pd.DataFrame({"season": seasons_in_stats, "gsis_id": target_gsis_id})

    player_hist = stats_df.merge(
        injury_flags,
        on=["gsis_id", "season"],
        how="left",
    ).fillna(0)

    if player_hist.empty:
        print(f"No seasons found for player_id={target_gsis_id} in player stats")
        return player_hist

    player_hist = player_hist.sort_values("season")
    player_hist["career_year"] = range(1, len(player_hist) + 1)
    player_hist["injury_severity"] = player_hist["weeks_out"] * 2 + player_hist["weeks_with_injury"]
    return player_hist


def plot_player_injury_history(player_hist: pd.DataFrame, target_gsis_id: str):
    """Plot weeks out as a line by career year."""
    if player_hist.empty:
        return

    fig1, ax1 = plt.subplots()
    ax1.plot(
        player_hist["career_year"],
        player_hist["weeks_out"],
        color="crimson",
        marker="o",
        label="Weeks Out",
    )
    ax1.set_ylabel("Weeks Out (games missed)")
    ax1.set_xlabel("Career Year")
    ax1.legend(loc="upper right")

    plt.title(f"Weeks out by season for {target_gsis_id}")
    plt.tight_layout()
    plt.show()


def plot_injury_severity(player_hist: pd.DataFrame, target_gsis_id: str):
    """Plot severity line = 2*weeks_out + weeks_with_injury over career year."""
    if player_hist.empty:
        return

    fig2, ax2 = plt.subplots()
    ax2.plot(
        player_hist["career_year"],
        player_hist["injury_severity"],
        color="firebrick",
        marker="o",
        label="Injury severity = 2*weeks_out + weeks_with_injury",
    )
    ax2.set_xlabel("Career Year")
    ax2.set_ylabel("Injury Severity (weighted)")
    ax2.legend(loc="upper right")
    plt.title(f"Injury severity over career for {target_gsis_id}")
    plt.tight_layout()
    plt.show()


def plot_multi_player_severity(histories: dict):
    """
    Plot weeks out for multiple players on the same axes.

    histories: dict {gsis_id: player_hist_df}
    """
    fig, ax = plt.subplots()
    for pid, hist in histories.items():
        if hist.empty:
            continue
        ax.plot(
            hist["career_year"],
            hist["weeks_out"],
            marker="o",
            label=pid,
        )
    ax.set_xlabel("Career Year")
    ax.set_ylabel("Weeks Out (games missed)")
    ax.legend(loc="upper right")
    plt.title("Weeks out across players")
    plt.tight_layout()
    plt.show()

def find_gsis_id(names, pbp_player):
    ids=[]
    for name in names:
        target_id = None
        for _, row in pbp_player.iterrows():
            if str(row.get("player_name", "")).strip() == name:
                target_id = row.get('player_id')
                print(f"{name} player_id: {target_id}")
                break
        if target_id != None:
            ids.append(target_id)
    return ids

def print_svd_pca(qb_matrix, svd_pca, s=True, u=True, vt=True):
    if not qb_matrix.empty:
        if s:
            print("Singular values:", svd_pca["S"])
        if u:
            print("U = ", svd_pca["U"])
        if vt:
            print("Vt = ", svd_pca['Vt'])

def plot_feature_distributions(pbp_player, save_path="feature_distributions.png"):
    """
    Creates a 4x4 grid of histograms for key player features used in clustering.
    """

    df = pbp_player.copy()

    features = [
        "passing_yards",
        "sacks_suffered",
        "carries",
        "targets",
    ]

    # Keep only features that exist
    features = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        data = df[feature].dropna()

        axes[i].hist(data, bins=30, edgecolor="black")
        axes[i].set_title(feature.replace("_", " ").title())
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

        # Log scale helps with skewed distributions
        axes[i].set_yscale("log")

    # Hide unused plots if fewer than 16 features
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_pre_treatment(real_player, synthetic_player, training_year):
    """
    Plots real vs synthetic player only up to the training year (pre-treatment).

    Parameters:
        real_player (array-like): real player trajectory
        synthetic_player (array-like): synthetic control trajectory
        training_year (int): index where treatment starts (not included in plot)
    """

    real = np.asarray(real_player)
    synth = np.asarray(synthetic_player)

    if real.shape != synth.shape:
        raise ValueError("real_player and synthetic_player must have the same shape")

    real_pre = real[:training_year]
    synth_pre = synth[:training_year]

    t = np.arange(1, len(real_pre) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(t, real_pre, marker='o', linewidth=2, label="Real Player")
    plt.plot(t, synth_pre, linestyle='--', linewidth=2, label="Synthetic Player")

    plt.xlabel("Time")
    plt.ylabel("Weeks out")
    plt.title("Pre-Treatment Fit: Real vs Synthetic Player (Weeks out)")

    plt.legend()
    plt.grid(True, alpha=0.3)

    # 🔥 Force y-axis scale
    plt.ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.show()

def plot_clustering_sensitivity(ATE):
    """
    ATE format:
    [k3_out, k3_burden, k4_out, k4_burden, k5_out, k5_burden, k6_out, k6_burden]
    """

    k_values = [3, 4, 5, 6]

    # Split data
    weeks_out = ATE[0::2]
    injury_burden = ATE[1::2]

    x = np.arange(len(k_values))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width/2, weeks_out, width, label="Weeks Out")
    plt.bar(x + width/2, injury_burden, width, label="Injury Burden")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Treatment Effect")
    plt.title("Sensitivity of Treatment Effects to Clustering (k-means)")

    plt.xticks(x, k_values)
    plt.legend()

    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("sensitivity_clusters.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    years = []
    for year in range(2012, 2020):
        years.append(year)
    pbp_player, pbp_injury, snap_counts, players = get_stats(years)

    # plot_feature_distributions(pbp_player)
    # positions = []

    # names = pbp_player.loc[pbp_player["team"] == "NE", "player_display_name"].tolist()
    # for player in pbp_player:
        # if pd.notna(pos) and pos not in positions:
        #     positions.append(pos)
        
    # print(positions)
    

    training_years = 5
    positions = ['K', 'CB', 'FS', 'MLB', 'ILB', 'TE', 'QB', 'DE', 'P', 'WR', 'C', 'LS', 'OLB', 'DT', 'NT', 'G', 'S', 'LB', 'OT', 'RB', 'FB', 'DB', 'DL', 'SAF', 'OL']
    player="Nate Solder"
    # Build 8×N QB weeks-out matrix (N = QBs with >=8 seasons)
    qb_matrix, qb_meta = build_qb_weeks_out_matrix(pbp_player, pbp_injury, positions, career_length=len(years))
    bruh, _ = build_qb_injury_severity_matrix(pbp_player, pbp_injury, positions=positions, career_length=len(years))
    matricies = [qb_matrix, bruh]
    # matricies = []
    # print(qb_matrix.columns.tolist())
    # print(bruh.columns.tolist())

    # Probably Massaged:
    # Devin McCourty, Nate Solder
    # Meh, prolly not
    # Chandler Jones
    time = 0
    for matrix in matricies:

        df_copy = matrix.iloc[:training_years, :].copy()
        # print("Matrix: ", matrix)
        # print("Shape: ", matrix.shape)
        # print("QB weeks-out matrix shape:", qb_matrix.shape)
        # pd.set_option("display.max_rows", None)
        # pd.set_option("display.max_columns", None)
        # print("\nFull QB weeks-out matrix:\n", qb_matrix.to_string())
        # print("QB column metadata (first 5):")
        # print(qb_meta.head())

        # Compute SVD/PCA on the QB matrix
        svd_pca = compute_qb_matrix_svd_pca(df_copy, n_components=3)
        # print_svd_pca(matrix, svd_pca, False, False, False)
        principal_comp = calc_pc_from_vt(df_copy, svd_pca["Vt"])
        # print(principal_comp.shape)
        # print(principal_comp)
        # plot_svd_vectors(matrix, single_plot_per_fig=True)

        # After you compute principal_comp from the SVD
        # Plot first 3 PCs in 3D (no cluster colors yet)
        # plot_pca_3d(principal_comp, labels=None, title='First 3 PCs Before Clustering')
        # print(principal_comp)
        # Then cluster based on the 3 PCs

        # Nothing changes for strictly weeks out
        # Best clustering amount adding weeks injured was 5
        labels_3d, kmeans_3d, X_3d = cluster_on_first_3_pcs(principal_comp, n_clusters=6)

        # Plot again with cluster colors
        # plot_pca_3d(principal_comp, labels=labels_3d, title='First 3 PCs Colored by Cluster')

        player_idx = -1
        names = matrix.columns.tolist()
        for name_idx in range(len(names)):
            if player == names[name_idx]:
                player_idx = name_idx
                break
        if player_idx == -1:
            print(f"Player {player} not found")
            return
        synthetic, weights, donor_indices, result = build_synthetic_player(df_copy.T, matrix.T, labels_3d, player_idx)


        # print("Target player:", player_idx)
        # print("Donor players:", len(donor_indices))
        # print("Weights:", weights)
        # print("Synthetic player:", synthetic)

        real_player = matrix.T.iloc[player_idx].to_numpy()

        # print("Real player:     ", real_player)
        # print("Synthetic player:", synthetic)
        # print("Reconstruction error:", np.sum((real_player[:training_years] - synthetic[:training_years]) ** 2))
        # plot_pre_treatment(real_player, synthetic, training_years)
        plot_real_vs_synthetic(real_player, synthetic, player_idx, training_years, player, time)
        time += 1
        print(f"ATE{np.mean(real_player - synthetic)}")



    # columns: out k=4, out+ k=4, k=3, k=3, k=5, k=5, k=6, k=6
    ATE = [0.175, 0.485, 0.175, 0.476, 0.175, 1.228, 0.175, 1.228]
    # plot_clustering_sensitivity(ATE)

    # Iterate through player stats and print player_id for T.Brady if present
    # names = ["E.Manning", "D.Brees", "T.Brady", "R.Wilson", "N.Foles", "K.Cousins"]
    # target_ids = find_gsis_id(names, pbp_player)

    # # Build per-player history, then plot individual and multi-player severity
    # histories = {}
    # for target_id in target_ids:
    #     target_hist = build_player_injury_history(pbp_injury, pbp_player, target_id)
    #     plot_player_injury_history(target_hist, target_id)
    #     plot_injury_severity(target_hist, target_id)
    #     histories[target_id] = target_hist

    # Example: add more player IDs to overlay severity curves
    # extra_ids = ["00-0022793", "00-0030000"]
    # for pid in extra_ids:
    #     hist = build_player_injury_history(pbp_injury, pbp_player, pid)
    #     histories[pid] = hist
    # plot_multi_player_severity(histories)

    # all_stats = collect_player_stats(pbp_player)
    # formatted_stats = average_player_stats(all_stats)
    # attach_snap_counts(formatted_stats, snap_counts)
    # stats_with_snap = average_snap_count_fields(formatted_stats)
    # final_vector = append_player_bio(stats_with_snap, players, years)
    # final_vector = drop_irrelevant_stats(final_vector)
    # cluster_and_analyze(final_vector, pbp_injury)
        

if __name__ == "__main__":
    main()
