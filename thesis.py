import nflreadpy as nfl
import matplotlib.pyplot as plt
import pandas as pd

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
    drop_irrelevant_stats,
    prepare_feature_matrix,
    run_kmeans_clusters,
    scale_and_pca,
    stats_dict_to_df,
)
from interpret import (
    compare_clusters_means,
    compute_cluster_feature_profiles,
    describe_cluster,
    plot_cluster_feature_heatmap,
    plot_cluster_injury_rate,
    plot_cluster_injury_severity,
    plot_pca_clusters,
    print_cluster_summaries,
)
from more_visuals import (
    compute_cluster_odds_ratios,
    plot_cluster_odds_ratio_forest,
    plot_injury_rate_diff_from_mean,
)
from injury_probability import (
    append_career_year_index,
    fit_injury_logistic_model,
    predict_player_injury_history,
    plot_injury_trajectory,
)

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

def run_clustering_workflow(final_vector, injuries):
    # 1. Dict → DataFrame
    player_df = stats_dict_to_df(final_vector)

    # 2. Clean → X
    clean_df, X, feature_cols = prepare_feature_matrix(player_df)

    # 3. Scale + PCA
    X_scaled, scaler, X_pca, pca = scale_and_pca(X, var_explained=0.9)

    # 4. Cluster
    labels, kmeans_model = run_kmeans_clusters(X_pca, n_clusters=6)
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

    return clean_df, merged_df, feature_cols, X_pca, labels


def view(merged_df, X_pca, clean_df, profiles_df):
    # After you have merged_df and profiles_df

    plot_cluster_injury_rate(merged_df)
    plot_cluster_injury_severity(merged_df)
    plot_pca_clusters(X_pca, clean_df['cluster'].values)
    plot_cluster_feature_heatmap(profiles_df)


def chat(final_vector, injuries):
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


def plot_actual_injury_history(injuries, pbp_player, target_gsis_id):
    """
    Plot realized injury severity over seasons for a specific player (no modeling).

    Severity proxy:
      - weeks_out (bars) shows how many games were missed
      - weeks_with_injury (line) shows how many weeks any injury was reported
    Larger values reflect more severe/longer injuries.
    """
    injury_flags = aggregate_injury_flags(injuries)

    # Build seasons present in the raw player stats for this player
    seasons_in_stats = (
        pbp_player.loc[pbp_player["player_id"] == target_gsis_id, "season"]
        .dropna()
        .astype(int)
        .unique()
    )
    stats_df = pd.DataFrame({"season": seasons_in_stats, "gsis_id": target_gsis_id})

    # Left-join injury info; missing rows mean no injuries that season
    player_hist = stats_df.merge(
        injury_flags,
        on=["gsis_id", "season"],
        how="left",
    ).fillna(0)

    if player_hist.empty:
        print(f"No seasons found for player_id={target_gsis_id} in final_vector")
        return player_hist

    player_hist = player_hist.sort_values("season")
    # Map seasons to career year order (1, 2, 3, ...)
    player_hist["career_year"] = range(1, len(player_hist) + 1)

    fig, ax1 = plt.subplots()
    ax1.bar(player_hist["career_year"], player_hist["weeks_out"], color="crimson", alpha=0.6, label="Weeks Out")
    ax1.set_ylabel("Weeks Out (games missed)")
    ax1.set_xlabel("Career Year")

    ax2 = ax1.twinx()
    ax2.plot(
        player_hist["career_year"],
        player_hist["weeks_with_injury"],
        color="navy",
        marker="o",
        label="Weeks with Injury",
    )
    ax2.set_ylabel("Weeks with Injury (any listing)")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.title(f"Real injury history for {target_gsis_id}")
    plt.tight_layout()
    plt.show()

    return player_hist


    

def main():
    years = []
    for year in range(2012, 2020):
        years.append(year)
    pbp_player, pbp_injury, snap_counts, players = get_stats(years)
    # Iterate through player stats and print player_id for T.Brady if present
    name = "D.Brees"
    for _, row in pbp_player.iterrows():
        if str(row.get("player_name", "")).strip() == name:
            target_id = row.get('player_id')
            print(f"{name} player_id: {target_id}")
            break

    plot_actual_injury_history(pbp_injury, pbp_player, target_id)

    # all_stats = collect_player_stats(pbp_player)
    # formatted_stats = average_player_stats(all_stats)
    # attach_snap_counts(formatted_stats, snap_counts)
    # stats_with_snap = average_snap_count_fields(formatted_stats)
    # final_vector = append_player_bio(stats_with_snap, players, years)
    # final_vector = drop_irrelevant_stats(final_vector)
    # chat(final_vector, pbp_injury)
        

if __name__ == "__main__":
    main()
