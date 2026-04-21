import numpy as np
from scipy.optimize import minimize

def build_synthetic_player(X, labels, player_idx):
    """
    Build a synthetic version of one player using other players in the same cluster.

    Parameters
    ----------
    X : np.ndarray
        Original data matrix of shape (n_players, n_weeks).
        Each row is one player's injury time series.
    labels : np.ndarray
        Cluster label for each player, shape (n_players,).
    player_idx : int
        Index of the player to synthesize.

    Returns
    -------
    synthetic : np.ndarray
        Synthetic player's time series, shape (n_weeks,).
    weights : np.ndarray
        Weights assigned to each donor player in the same cluster.
    donor_indices : np.ndarray
        Indices of donor players used to build the synthetic player.
    result : OptimizeResult
        Full scipy optimization result object.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_players, n_weeks).")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if len(labels) != X.shape[0]:
        raise ValueError("labels length must match number of rows in X.")
    if not (0 <= player_idx < X.shape[0]):
        raise IndexError("player_idx is out of bounds.")

    # Find donors in same cluster, excluding target player
    cluster_id = labels[player_idx]
    donor_indices = np.where(labels == cluster_id)[0]
    donor_indices = donor_indices[donor_indices != player_idx]

    if len(donor_indices) == 0:
        raise ValueError("No donor players available in the same cluster.")

    target = X[player_idx]                  # shape (n_weeks,)
    donor_matrix = X[donor_indices]         # shape (n_donors, n_weeks)

    # Objective: minimize squared reconstruction error
    def loss(w):
        synthetic = w @ donor_matrix
        return np.sum((target - synthetic) ** 2)

    # Constraints: weights sum to 1, all weights between 0 and 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(len(donor_indices))]

    # Initial guess: equal weights
    w0 = np.ones(len(donor_indices)) / len(donor_indices)

    result = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    synthetic = weights @ donor_matrix

    return synthetic, weights, donor_indices, result