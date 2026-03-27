import nflreadpy as nfl


from format import *
from chat_method import *
from interpret import *
from more_visuals import *
from injury_probability import *

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

def chat_method(final_vector, injuries):
    # 1. Dict → DataFrame
    player_df = dict_to_player_df(final_vector)

    # 2. Clean → X
    clean_df, X, feature_cols = prepare_feature_matrix(player_df)

    # 3. Scale + PCA
    X_scaled, scaler, X_pca, pca = scale_and_pca(X, var_explained=0.9)

    # 4. Cluster
    labels, kmeans_model = run_kmeans(X_pca, n_clusters=6)
    clean_df['cluster'] = labels

    # 5. Injury flags (injuries is your pbp_injury df)
    injury_df = build_injury_flags(injuries)

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
    print('buh')

    # 7. Significance test
    chi2, p, dof, expected, table = chi_square_cluster_injury(merged_df)
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
    clean_df, merged_df, feature_cols, X_pca, labels = chat_method(final_vector, injuries)
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


    

def main():
    years = []
    for year in range(2012, 2020):
        years.append(year)
    pbp_player, pbp_injury, snap_counts, players = get_stats(years)
    all_stats = format(pbp_player)
    formated_stats = average_stats(all_stats)
    add_snap_count(formated_stats, snap_counts)
    format_with_snap = average_snap_counts(formated_stats)
    final_vector = add_personal_info(format_with_snap, players, years)
    final_vector = remove_unecessary_stats(final_vector)
    chat(final_vector, pbp_injury)

        

if __name__ == "__main__":
    main()