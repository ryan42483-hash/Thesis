"""Utilities to build an 8×N quarterback injury matrix.

Each column corresponds to one quarterback (only those with at least eight
recorded seasons) and each row is the number of weeks the QB was out in that
career year (years 1–8). Missing injury records default to 0.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

from chat_method import aggregate_injury_flags


def qbs_with_min_seasons(
    pbp_player: pd.DataFrame, min_seasons: int = 8, position = "QB"
) -> pd.DataFrame:
    """Return quarterbacks with at least ``min_seasons`` unique seasons.

    The input ``pbp_player`` should come from ``nfl.load_player_stats`` and
    include ``player_id``, ``season``, ``position``, and optionally
    ``player_display_name``. Duplicate season rows (e.g., multiple teams) are
    collapsed to one season per player.
    """

    required_cols = {"player_id", "season", "position"}
    missing = required_cols - set(pbp_player.columns)
    if missing:
        raise ValueError(f"pbp_player is missing required columns: {sorted(missing)}")

    qb_seasons = (
        pbp_player[pbp_player["position"] == position][["player_id", "season", "player_display_name"]]
        .dropna(subset=["player_id", "season"])
        .assign(season=lambda df: df["season"].astype(int))
        .drop_duplicates(subset=["player_id", "season"])
    )

    counts = (
        qb_seasons.groupby(["player_id", "player_display_name"], dropna=False)["season"]
        .nunique()
        .reset_index(name="num_seasons")
    )

    return counts[counts["num_seasons"] >= min_seasons].reset_index(drop=True)


def build_qb_weeks_out_matrix(
    pbp_player: pd.DataFrame,
    pbp_injury: pd.DataFrame,
    *,
    career_length: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an ``career_length`` × N matrix of weeks-out for veteran QBs.

    Parameters
    ----------
    pbp_player : DataFrame
        Output of ``nfl.load_player_stats`` containing QB seasons.
    pbp_injury : DataFrame
        Weekly injury data (``nfl.load_injuries``) used to compute weeks out.
    career_length : int, default 8
        Number of career years to include; columns are limited to QBs with at
        least this many seasons. Only the earliest ``career_length`` seasons
        per QB are used to populate the matrix.

    Returns
    -------
    matrix_df : DataFrame
        Index = career years 1..``career_length``; columns = quarterback labels
        (display name when available, otherwise ``player_id``); values = weeks
        listed as out in that season. Missing injury rows are filled with 0.
    qb_meta : DataFrame
        Mapping of column labels to player_id and the seasons used.
    """

    eligible_qbs = qbs_with_min_seasons(pbp_player, min_seasons=career_length)
    if eligible_qbs.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Pre-aggregate injury data to season level.
    injury_rollup = aggregate_injury_flags(pbp_injury)

    matrix_columns: dict[str, list[int]] = {}
    metadata_rows: list[dict[str, object]] = []

    for _, qb_row in eligible_qbs.iterrows():
        qb_id = qb_row["player_id"]
        qb_name = qb_row.get("player_display_name")
        label = str(qb_name) if qb_name and qb_name == qb_name else str(qb_id)

        qb_seasons = (
            pbp_player.loc[pbp_player["player_id"] == qb_id, "season"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        # Use the earliest `career_length` seasons to create a consistent vector.
        seasons_used = qb_seasons[:career_length]
        if len(seasons_used) < career_length:
            # Skip if data is unexpectedly short after filtering duplicates.
            continue

        weeks_out_by_year: list[int] = []
        for season in seasons_used:
            row = injury_rollup[
                (injury_rollup["gsis_id"] == qb_id) & (injury_rollup["season"] == season)
            ]
            weeks_out = int(row["weeks_out"].iloc[0]) if not row.empty else 0
            weeks_out_by_year.append(weeks_out)

        matrix_columns[label] = weeks_out_by_year[:career_length]
        metadata_rows.append(
            {
                "column_label": label,
                "player_id": qb_id,
                "player_display_name": qb_name,
                "seasons_used": seasons_used[:career_length],
            }
        )

    if not matrix_columns:
        return pd.DataFrame(), pd.DataFrame()

    matrix_df = pd.DataFrame(matrix_columns)
    matrix_df.index = range(1, career_length + 1)
    matrix_df.index.name = "career_year"

    qb_meta = pd.DataFrame(metadata_rows)
    return matrix_df, qb_meta


__all__ = ["qbs_with_min_seasons", "build_qb_weeks_out_matrix"]


def compute_qb_matrix_svd_pca(
    qb_matrix: pd.DataFrame,
    n_components: int | None = None,
) -> dict:
    """Compute SVD and PCA on the QB weeks-out matrix.

    Parameters
    ----------
    qb_matrix : DataFrame
        Output from ``build_qb_weeks_out_matrix`` (shape = 8×N).
    n_components : int | None, default None
        Number of principal components to keep; defaults to full rank.

    Returns
    -------
    dict with keys:
      - ``U``, ``S``, ``Vt``: numpy arrays from ``np.linalg.svd`` (full matrix).
      - ``pca_model``: fitted ``sklearn.decomposition.PCA`` instance.
      - ``pca_components``: PCA component loadings (shape = k×N).
      - ``explained_variance``: array of per-component explained variance ratios.
    """

    if qb_matrix.empty:
        raise ValueError("qb_matrix is empty; build it before computing SVD/PCA.")

    matrix_values = qb_matrix.to_numpy(dtype=float)

    # Full SVD (no economy mode) for clarity downstream
    U, S, Vt = np.linalg.svd(matrix_values, full_matrices=False)

    # PCA on the same data
    pca = PCA(n_components=n_components)
    pca.fit(matrix_values)

    return {
        "U": U,
        "S": S,
        "Vt": Vt,
        "pca_model": pca,
        "pca_components": pca.components_,
        "explained_variance": pca.explained_variance_ratio_,
    }


__all__.extend(["compute_qb_matrix_svd_pca"])


def plot_svd_vectors(qb_matrix: pd.DataFrame, single_plot_per_fig: bool = False) -> None:
    """Plot each row of ``U`` and each column of ``Vt`` from the QB matrix SVD.

    Two figures are produced:
      1) One subplot per row of ``U`` (rows correspond to career years).
      2) One subplot per column of ``Vt`` (columns correspond to quarterbacks).

    Parameters
    ----------
    qb_matrix : DataFrame
        Input 8×N QB matrix.
    single_plot_per_fig : bool, default False
        When True, create one figure per vector (one per U row and one per Vt column);
        when False, use the compact multi-plot grids.
    """

    if qb_matrix.empty:
        print("qb_matrix is empty; nothing to plot.")
        return

    svd_pca = compute_qb_matrix_svd_pca(qb_matrix)
    U = svd_pca["U"]
    Vt = svd_pca["Vt"]

    # Plot U vectors
    num_u_rows = U.shape[0]
    if single_plot_per_fig:
        for idx in range(num_u_rows):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(U[idx, :], marker="o")
            ax.set_title(f"U row {idx+1}")
            ax.set_xlabel("Component index")
            ax.set_ylabel("value")
            fig.tight_layout()
    else:
        u_cols = 2
        u_rows = math.ceil(num_u_rows / u_cols)
        fig_u, axes_u = plt.subplots(
            u_rows, u_cols, figsize=(u_cols * 4.5, u_rows * 3), sharex=True
        )
        axes_u = np.atleast_2d(axes_u)

        for idx in range(num_u_rows):
            r, c = divmod(idx, u_cols)
            ax = axes_u[r, c]
            ax.plot(U[idx, :], marker="o")
            ax.set_title(f"U row {idx+1}")
            ax.set_ylabel("value")
        # Hide any unused U subplots
        for idx in range(num_u_rows, u_rows * u_cols):
            r, c = divmod(idx, u_cols)
            axes_u[r, c].axis("off")

        axes_u[-1, 0].set_xlabel("Component index")
        fig_u.tight_layout()

    # Plot Vt vectors
    num_v_cols = Vt.shape[1]
    if single_plot_per_fig:
        for idx in range(num_v_cols):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(Vt[:, idx], marker="o")
            ax.set_title(f"Vt col {idx+1}")
            ax.set_xlabel("Component index")
            ax.set_ylabel("value")
            fig.tight_layout()
    else:
        max_cols = 3
        n_rows = math.ceil(num_v_cols / max_cols)
        fig_v, axes_v = plt.subplots(
            n_rows, max_cols, figsize=(max_cols * 4, n_rows * 3), sharex=True
        )
        axes_v = np.atleast_2d(axes_v)

        for idx in range(num_v_cols):
            r, c = divmod(idx, max_cols)
            ax = axes_v[r, c]
            ax.plot(Vt[:, idx], marker="o")
            ax.set_title(f"Vt col {idx+1}")
            ax.set_ylabel("value")
        # Hide any unused subplots
        for idx in range(num_v_cols, n_rows * max_cols):
            r, c = divmod(idx, max_cols)
            axes_v[r, c].axis("off")

        axes_v[-1, 0].set_xlabel("Component index")
        fig_v.tight_layout()

    plt.show()


__all__.extend(["plot_svd_vectors"])
