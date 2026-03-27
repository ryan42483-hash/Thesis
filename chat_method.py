import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from scipy.stats import chi2_contingency


stat_exclusions = {'position', 'player_display_name', 'position_group', 'team', 'season', 'total_snaps', 'offense_snaps', 'defense_snaps', 'special_snaps', 'offense_pct', 'defense_pct', 'special_pct'}

def remove_unecessary_stats(player_stats):
    cleaned_stats = {}
    for key in player_stats.keys():
        for stat_name in player_stats[key]:
            if stat_name not in stat_exclusions:
                cleaned_stats.setdefault(key, {})[stat_name] = player_stats[key][stat_name]
    return cleaned_stats


def dict_to_player_df(stats_dict):
    """
    Convert { (gsis_id, season): stats_dict } into a pandas DataFrame.

    Returns:
        df: DataFrame with columns ['gsis_id', 'season', ...stats...]
    """
    rows = []
    for (gsis_id, season), stats in stats_dict.items():
        row = {'gsis_id': gsis_id, 'season': season}
        row.update(stats)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def prepare_feature_matrix(player_df, drop_cols=None):
    """
    Takes the player_df and returns:
      - cleaned_df (with id columns kept)
      - X (numpy feature matrix suitable for scaling/PCA/clustering)
      - feature_cols (list of column names used in X)
    """
    df = player_df.copy()

    # ensure id columns exist
    id_cols = ['gsis_id', 'season']
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"Required id column '{col}' not found in DataFrame.")

    # columns to drop if present
    default_drop_cols = [
        # any leftovers you don't want in clustering
        'def_tackles_solo', 'def_tackles_with_assist', 'def_tackle_assists',
        'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced',
        'def_sacks', 'def_sack_yards', 'def_qb_hits', 'def_interceptions',
        'def_interception_yards', 'def_pass_defended', 'def_tds', 'def_fumbles',
        'def_safeties', 'misc_yards', 'wopr', 'target_share', 'air_yards_share'
    ]

    if drop_cols is not None:
        default_drop_cols.extend(drop_cols)

    # drop only if present
    default_drop_cols = [c for c in default_drop_cols if c in df.columns]
    df = df.drop(columns=default_drop_cols, errors='ignore')

    # numeric feature columns = all non-id columns
    feature_cols = [c for c in df.columns if c not in id_cols]

    # fill NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values.astype(float)
    return df, X, feature_cols


def scale_and_pca(X, var_explained=0.9):
    """
    Standardizes X then runs PCA to retain `var_explained` fraction
    of total variance.

    Returns:
        X_scaled
        scaler (StandardScaler)
        X_pca
        pca (PCA)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=var_explained)
    X_pca = pca.fit_transform(X_scaled)

    return X_scaled, scaler, X_pca, pca


def run_kmeans(X_pca, n_clusters, random_state=42):
    """
    Fit KMeans on PCA-transformed data.

    Returns:
        labels: cluster labels for each row
        model: fitted KMeans object
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)
    return labels, kmeans


def build_injury_flags(pbp_injury: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly injury data into season-level flags per (gsis_id, season).

    pbp_injury columns expected (based on your printout):
      - season
      - week
      - gsis_id
      - report_primary_injury
      - report_secondary_injury
      - report_status
      - practice_primary_injury
      - practice_secondary_injury
      - practice_status
      - game_type (REG, CON, SB, etc.)  # can be kept or filtered

    Returns:
        injury_df with:
          - gsis_id
          - season (int)
          - weeks_with_injury
          - weeks_out
          - weeks_questionable
          - weeks_dnp_injury
          - injured_any (1 if any real injury in season, else 0)
    """
    df = pbp_injury.copy()

    # Convert season/week to int so that they match your (gsis_id, year) keys
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # OPTIONAL: if you only want regular season injuries, uncomment:
    # df = df[df['game_type'] == 'REG']

    # Helper to normalize injury text fields
    def norm_injury_str(s):
        if pd.isna(s):
            return "None"
        return str(s).strip()

    for col in [
        'report_primary_injury', 'report_secondary_injury',
        'practice_primary_injury', 'practice_secondary_injury',
        'report_status', 'practice_status'
    ]:
        df[col] = df[col].apply(norm_injury_str)

    # Define what counts as "no real injury"
    non_injury_labels = {"None", "Not Injury Related"}

    # Row-level: does this week contain any actual injury (non-None, non-NIR)?
    df['has_real_injury_text'] = df.apply(
        lambda row: any(
            row[col] not in non_injury_labels
            for col in [
                'report_primary_injury',
                'report_secondary_injury',
                'practice_primary_injury',
                'practice_secondary_injury'
            ]
        ),
        axis=1
    )

    # Weekly injury flag: at least one real injury listed anywhere
    df['injury_week_flag'] = df['has_real_injury_text'].astype(int)

    # Game status flags
    df['out_week_flag'] = (df['report_status'] == 'Out').astype(int)
    df['questionable_week_flag'] = (df['report_status'] == 'Questionable').astype(int)
    # If you have 'Doubtful', you could add that too:
    # df['doubtful_week_flag'] = (df['report_status'] == 'Doubtful').astype(int)

    # Practice "Did Not Participate" that is actually injury-related
    df['dnp_injury_flag'] = (
        (df['practice_status'] == 'Did Not Participate In Practice') &
        (df['has_real_injury_text'])
    ).astype(int)

    # Aggregate by (gsis_id, season)
    grouped = df.groupby(['gsis_id', 'season'], as_index=False).agg(
        weeks_with_injury=('injury_week_flag', 'sum'),
        weeks_out=('out_week_flag', 'sum'),
        weeks_questionable=('questionable_week_flag', 'sum'),
        weeks_dnp_injury=('dnp_injury_flag', 'sum'),
    )

    # Binary: did this player have *any* injury this season?
    grouped['injured_any'] = (grouped['weeks_with_injury'] > 0).astype(int)

    return grouped


def merge_clusters_injuries(clean_df, injury_df):
    """
    Merge player cluster assignments with injury flags
    on (gsis_id, season).
    """
    merged = clean_df.merge(
        injury_df,
        on=['gsis_id', 'season'],
        how='left'
    )

    # players with no record in injury table → assume no injury
    merged['injured_any'] = merged['injured_any'].fillna(0).astype(int)
    merged['weeks_with_injury'] = merged['weeks_with_injury'].fillna(0).astype(int)
    merged['weeks_out'] = merged['weeks_out'].fillna(0).astype(int)

    return merged


def chi_square_cluster_injury(merged_df, cluster_col='cluster', injury_col='injured_any'):
    """
    Run chi-square test to see if injury rate differs by cluster.

    Returns:
        chi2, p, dof, expected, contingency_table
    """
    contingency = pd.crosstab(merged_df[cluster_col], merged_df[injury_col])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return chi2, p, dof, expected, contingency
