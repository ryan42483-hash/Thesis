import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt


# ----------------------------
# 1) Panel construction helpers
# ----------------------------

def add_career_year(df: pd.DataFrame,
                    player_col: str = "player_id",
                    season_col: str = "season",
                    debut_season_col: str | None = None,
                    out_col: str = "career_year") -> pd.DataFrame:
    """
    Adds career_year = 1,2,3,... for each player by season ordering.
    If debut_season_col is provided, uses season - debut_season + 1.
    Otherwise uses rank within each player's seasons.
    """
    df = df.copy()
    if debut_season_col is not None and debut_season_col in df.columns:
        df[out_col] = (df[season_col] - df[debut_season_col] + 1).astype(int)
    else:
        df[out_col] = (
            df.sort_values([player_col, season_col])
              .groupby(player_col)
              .cumcount() + 1
        ).astype(int)
    return df


def make_injury_binary(df: pd.DataFrame,
                       games_missed_col: str = "games_missed",
                       threshold: int = 3,
                       out_col: str = "injured") -> pd.DataFrame:
    """
    Converts games missed into a binary injury indicator:
        injured = 1 if games_missed >= threshold else 0
    """
    df = df.copy()
    df[out_col] = (df[games_missed_col].fillna(0) >= threshold).astype(int)
    return df


def sanitize_panel(df: pd.DataFrame,
                   required_cols: list[str]) -> pd.DataFrame:
    """
    Drops rows with missing required fields and enforces reasonable types.
    """
    df = df.copy()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    return df


# ----------------------------
# 2) Model fitting (Option A)
# ----------------------------

def fit_injury_logit_panel(df: pd.DataFrame,
                           formula: str,
                           cluster_se: bool = True,
                           cluster_col: str = "player_id"):
    """
    Fits a logistic regression using a patsy/statsmodels formula.

    Example formula (no player FE):
      'injured ~ career_year + C(cluster) + usage'

    Example formula (player fixed effects):
      'injured ~ career_year + C(cluster) + usage + C(player_id)'

    cluster_se=True gives cluster-robust SEs at the player level.
    """
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    if cluster_se:
        if cluster_col not in df.columns:
            raise ValueError(f"cluster_col '{cluster_col}' not in df")
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    else:
        res = model.fit()

    return {"result": res, "X_columns": X.columns, "formula": formula}


# ----------------------------
# 3) Player trajectory prediction
# ----------------------------

def predict_player_injury_trajectory(fit: dict,
                                    df: pd.DataFrame,
                                    player_id,
                                    player_col: str = "player_id",
                                    season_col: str = "season",
                                    career_year_col: str = "career_year",
                                    pred_col: str = "pred_injury_prob") -> pd.DataFrame:
    """
    Returns the rows for a single player with predicted injury probability per season.
    Uses the model formula so C(player_id), C(cluster), etc. are handled correctly.

    Important:
    - If you included C(player_id) in the formula, you can only predict for players
      seen in training (which is what you want for tracking a real player).
    """
    res = fit["result"]
    formula = fit["formula"]

    player_df = df[df[player_col] == player_id].copy()
    if player_df.empty:
        raise ValueError(f"No rows found for player_id={player_id}")

    # Build design matrices for this player's rows using the same formula
    y_p, X_p = patsy.dmatrices(formula, data=player_df, return_type="dataframe")
    player_df[pred_col] = res.predict(X_p)

    # Sort nicely for time-series use
    player_df = player_df.sort_values([season_col, career_year_col]).reset_index(drop=True)
    return player_df[[player_col, season_col, career_year_col, pred_col] + 
                     [c for c in player_df.columns if c in ("cluster", "usage", "injured")]]


def plot_player_trajectory(player_traj: pd.DataFrame,
                           x_col: str = "career_year",
                           y_col: str = "pred_injury_prob",
                           title: str | None = None):
    """
    Simple plot of predicted injury probability vs career year.
    """
    plt.figure()
    plt.plot(player_traj[x_col], player_traj[y_col], marker="o")
    plt.xlabel("Year in League")
    plt.ylabel("Predicted injury probability")
    if title:
        plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


# ----------------------------
# 4) (Optional) Smoothed trajectory
# ----------------------------

def smooth_probability_curve(player_traj: pd.DataFrame,
                             prob_col: str = "pred_injury_prob",
                             window: int = 3,
                             out_col: str = "pred_injury_prob_smooth") -> pd.DataFrame:
    """
    Rolling mean smoothing over career years (useful for presentation).
    """
    player_traj = player_traj.copy()
    player_traj[out_col] = (
        player_traj[prob_col]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    return player_traj