"""Clustering and injury flag utilities used by the thesis notebook."""

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Stats to drop before modeling
STAT_EXCLUSIONS = {
    "position",
    "player_display_name",
    "position_group",
    "team",
    "season",
    "total_snaps",
    "offense_snaps",
    "defense_snaps",
    "special_snaps",
    "offense_pct",
    "defense_pct",
    "special_pct",
}


# --------------------------------------------------------------------------- #
# Data cleaning helpers
# --------------------------------------------------------------------------- #

def drop_irrelevant_stats(player_stats: dict) -> dict:
    """Remove non-modeling stats while preserving the existing dict structure."""

    filtered_stats = {}
    for player_key, stats in player_stats.items():
        for stat_name, value in stats.items():
            if stat_name not in STAT_EXCLUSIONS:
                filtered_stats.setdefault(player_key, {})[stat_name] = value
    return filtered_stats


def stats_dict_to_df(stats_dict: dict) -> pd.DataFrame:
    """Convert {(gsis_id, season): stats} into a tidy DataFrame."""

    rows = [{"gsis_id": pid, "season": season, **stats} for (pid, season), stats in stats_dict.items()]
    return pd.DataFrame(rows)


def prepare_feature_matrix(player_df: pd.DataFrame, drop_cols=None):
    """Return cleaned df, numeric feature matrix, and feature column names."""

    df = player_df.copy()

    id_cols = ["gsis_id", "season"]
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"Required id column '{col}' not found in DataFrame.")

    default_drop_cols = [
        "def_tackles_solo",
        "def_tackles_with_assist",
        "def_tackle_assists",
        "def_tackles_for_loss",
        "def_tackles_for_loss_yards",
        "def_fumbles_forced",
        "def_sacks",
        "def_sack_yards",
        "def_qb_hits",
        "def_interceptions",
        "def_interception_yards",
        "def_pass_defended",
        "def_tds",
        "def_fumbles",
        "def_safeties",
        "misc_yards",
        "wopr",
        "target_share",
        "air_yards_share",
    ]

    if drop_cols is not None:
        default_drop_cols.extend(drop_cols)

    default_drop_cols = [c for c in default_drop_cols if c in df.columns]
    df = df.drop(columns=default_drop_cols, errors="ignore")

    feature_cols = [c for c in df.columns if c not in id_cols]
    df[feature_cols] = df[feature_cols].fillna(0)

    feature_matrix = df[feature_cols].values.astype(float)
    return df, feature_matrix, feature_cols


# --------------------------------------------------------------------------- #
# Modeling helpers
# --------------------------------------------------------------------------- #

def scale_and_pca(feature_matrix, var_explained=0.9):
    """Standardize features then run PCA retaining `var_explained` variance."""

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=var_explained)
    projected = pca.fit_transform(scaled)

    return scaled, scaler, projected, pca


def run_kmeans_clusters(X_pca, n_clusters, random_state=42):
    """Fit KMeans on PCA-transformed data and return labels + model."""

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)
    return labels, kmeans


# --------------------------------------------------------------------------- #
# Injury aggregation
# --------------------------------------------------------------------------- #

def aggregate_injury_flags(pbp_injury: pd.DataFrame) -> pd.DataFrame:
    """Roll up weekly injury data to season-level flags per player."""

    df = pbp_injury.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    def normalize_injury_text(value):
        if pd.isna(value):
            return "None"
        return str(value).strip()

    for col in [
        "report_primary_injury",
        "report_secondary_injury",
        "practice_primary_injury",
        "practice_secondary_injury",
        "report_status",
        "practice_status",
    ]:
        df[col] = df[col].apply(normalize_injury_text)

    non_injury_labels = {"None", "Not Injury Related"}

    df["has_real_injury_text"] = df.apply(
        lambda row: any(
            row[col] not in non_injury_labels
            for col in [
                "report_primary_injury",
                "report_secondary_injury",
                "practice_primary_injury",
                "practice_secondary_injury",
            ]
        ),
        axis=1,
    )

    df["injury_week_flag"] = df["has_real_injury_text"].astype(int)
    df["out_week_flag"] = (df["report_status"] == "Out").astype(int)
    df["questionable_week_flag"] = (df["report_status"] == "Questionable").astype(int)
    df["dnp_injury_flag"] = (
        (df["practice_status"] == "Did Not Participate In Practice")
        & (df["has_real_injury_text"])
    ).astype(int)

    grouped = df.groupby(["gsis_id", "season"], as_index=False).agg(
        weeks_with_injury=("injury_week_flag", "sum"),
        weeks_out=("out_week_flag", "sum"),
        weeks_questionable=("questionable_week_flag", "sum"),
        weeks_dnp_injury=("dnp_injury_flag", "sum"),
    )

    grouped["injured_any"] = (grouped["weeks_with_injury"] > 0).astype(int)
    return grouped


def merge_clusters_with_injuries(clean_df: pd.DataFrame, injury_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cluster assignments with injury flags on (gsis_id, season)."""

    merged = clean_df.merge(injury_df, on=["gsis_id", "season"], how="left")
    merged["injured_any"] = merged["injured_any"].fillna(0).astype(int)
    merged["weeks_with_injury"] = merged["weeks_with_injury"].fillna(0).astype(int)
    merged["weeks_out"] = merged["weeks_out"].fillna(0).astype(int)
    return merged


def chi_square_cluster_injury_test(
    merged_df: pd.DataFrame, cluster_col="cluster", injury_col="injured_any"
):
    """Run chi-square test to check if injury rate differs by cluster."""

    contingency = pd.crosstab(merged_df[cluster_col], merged_df[injury_col])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    return chi2, p_val, dof, expected, contingency



# --------------------------------------------------------------------------- #
# 3D PCA Clustering
# --------------------------------------------------------------------------- #

def cluster_on_first_3_pcs(X_pca, n_clusters=6, random_state=42):
    """
    Cluster players based on only the first 3 principal components.

    Parameters
    ----------
    X_pca : np.ndarray
        Full PCA-transformed feature matrix (n_samples, n_components).
    n_clusters : int, default=6
        Number of KMeans clusters.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Cluster label for each player (n_samples,).
    kmeans_model : KMeans
        Fitted KMeans model.
    X_3d : np.ndarray
        The first 3 principal components used for clustering (n_samples, 3).
    """
    # Extract the first 3 principal components
    X_3d = X_pca[:, :3]

    # Cluster on the 3D vectors
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans_model.fit_predict(X_3d)

    return labels, kmeans_model, X_3d
