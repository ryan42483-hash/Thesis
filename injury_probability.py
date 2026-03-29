"""Lightweight helpers for injury probability modeling.

The functions below intentionally avoid any side effects so they remain easy to
read, test, and drop into notebooks. Outputs are unchanged; names and
docstrings are clarified.
"""

import matplotlib.pyplot as plt
import pandas as pd
import patsy
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# 1) Panel construction helpers
# ---------------------------------------------------------------------------

def append_career_year_index(
    dataframe: pd.DataFrame,
    player_col: str = "player_id",
    season_col: str = "season",
    debut_season_col: str | None = None,
    out_col: str = "career_year",
) -> pd.DataFrame:
    """Add a sequential `career_year` per player.

    If `debut_season_col` is provided and present, use
    `season - debut_season + 1`; otherwise rank seasons per player. The
    numeric values (and therefore downstream outputs) match the previous
    implementation—only names are clearer.
    """

    panel = dataframe.copy()

    if debut_season_col is not None and debut_season_col in panel.columns:
        panel[out_col] = (panel[season_col] - panel[debut_season_col] + 1).astype(int)
    else:
        panel[out_col] = (
            panel.sort_values([player_col, season_col])
            .groupby(player_col)
            .cumcount()
            + 1
        ).astype(int)

    return panel


def add_injury_flag(
    dataframe: pd.DataFrame,
    games_missed_col: str = "games_missed",
    threshold: int = 3,
    out_col: str = "injured",
) -> pd.DataFrame:
    """Create a binary injury indicator.

    The logic is unchanged: `injured = 1` when `games_missed` meets or exceeds
    `threshold`, else 0. Missing values default to 0 games missed.
    """

    frame_with_flag = dataframe.copy()
    frame_with_flag[out_col] = (frame_with_flag[games_missed_col].fillna(0) >= threshold).astype(int)
    return frame_with_flag


def ensure_required_columns(dataframe: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Drop rows missing required fields and fail fast on absent columns."""

    cleaned = dataframe.copy()
    missing_cols = [col for col in required_cols if col not in cleaned.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return cleaned.dropna(subset=required_cols).copy()


# ---------------------------------------------------------------------------
# 2) Model fitting
# ---------------------------------------------------------------------------

def fit_injury_logistic_model(
    dataframe: pd.DataFrame,
    formula: str,
    cluster_se: bool = True,
    cluster_col: str = "player_id",
):
    """Fit a logistic regression using a patsy/statsmodels formula.

    Examples (unchanged):
      - No player fixed effects: 'injured ~ career_year + C(cluster) + usage'
      - Player fixed effects:    'injured ~ career_year + C(cluster) + usage + C(player_id)'

    When `cluster_se` is True, cluster-robust standard errors are computed at
    the `cluster_col` level.
    """

    y_matrix, design_matrix = patsy.dmatrices(formula, data=dataframe, return_type="dataframe")

    model = sm.GLM(y_matrix, design_matrix, family=sm.families.Binomial())
    if cluster_se:
        if cluster_col not in dataframe.columns:
            raise ValueError(f"cluster_col '{cluster_col}' not in dataframe")
        fit_result = model.fit(cov_type="cluster", cov_kwds={"groups": dataframe[cluster_col]})
    else:
        fit_result = model.fit()

    # Keep return structure the same for downstream compatibility
    return {"result": fit_result, "X_columns": design_matrix.columns, "formula": formula}


# ---------------------------------------------------------------------------
# 3) Player trajectory prediction
# ---------------------------------------------------------------------------

def predict_player_injury_history(
    fitted_model: dict,
    dataframe: pd.DataFrame,
    player_id,
    player_col: str = "player_id",
    season_col: str = "season",
    career_year_col: str = "career_year",
    pred_col: str = "pred_injury_prob",
) -> pd.DataFrame:
    """Return per-season injury probabilities for one player.

    Uses the stored formula so categorical terms (e.g., `C(player_id)`) are
    handled identically to training. Behavior matches the previous function;
    only names and variable clarity have improved.
    """

    model_result = fitted_model["result"]
    formula = fitted_model["formula"]

    player_rows = dataframe[dataframe[player_col] == player_id].copy()
    if player_rows.empty:
        raise ValueError(f"No rows found for player_id={player_id}")

    # Rebuild design matrix for this player's rows using the training formula
    _, player_design = patsy.dmatrices(formula, data=player_rows, return_type="dataframe")
    player_rows[pred_col] = model_result.predict(player_design)

    # Sort for time-series plotting / inspection
    player_rows = player_rows.sort_values([season_col, career_year_col]).reset_index(drop=True)

    ordered_columns = [player_col, season_col, career_year_col, pred_col]
    supplemental_cols = [col for col in player_rows.columns if col in ("cluster", "usage", "injured")]
    return player_rows[ordered_columns + supplemental_cols]


def plot_injury_trajectory(
    player_trajectory: pd.DataFrame,
    x_col: str = "career_year",
    y_col: str = "pred_injury_prob",
    title: str | None = None,
):
    """Plot predicted injury probability vs. career year."""

    plt.figure()
    plt.plot(player_trajectory[x_col], player_trajectory[y_col], marker="o")
    plt.xlabel("Year in League")
    plt.ylabel("Predicted injury probability")
    if title:
        plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# 4) (Optional) Smoothed trajectory
# ---------------------------------------------------------------------------

def smooth_injury_probability_curve(
    player_trajectory: pd.DataFrame,
    prob_col: str = "pred_injury_prob",
    window: int = 3,
    out_col: str = "pred_injury_prob_smooth",
) -> pd.DataFrame:
    """Apply a centered rolling mean to smooth predicted probabilities."""

    smoothed = player_trajectory.copy()
    smoothed[out_col] = (
        smoothed[prob_col]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    return smoothed
