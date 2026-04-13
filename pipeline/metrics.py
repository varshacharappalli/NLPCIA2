"""
metrics.py
Evaluation metrics for sentence ordering: pairwise accuracy,
sequence accuracy, Kendall Tau, and helper utilities.
"""

import numpy as np
from scipy.stats import kendalltau as scipy_kendalltau



def kendall_tau(pred_order, true_order):
    """Kendall Tau correlation between predicted and true orderings."""
    if len(pred_order) <= 1:
        return 1.0
    tau, _ = scipy_kendalltau(pred_order, true_order)
    if np.isnan(tau):
        return 0.0
    return float(tau)



def tournament_to_order(score_matrix):
    """
    Convert pairwise score matrix to a total ordering.
    score_matrix[i][j] = probability/score that sentence i comes before j.
    Returns: list of sentence indices sorted by descending row-sum score.
    """
    n = score_matrix.shape[0]
    row_scores = score_matrix.sum(axis=1)
    return list(np.argsort(-row_scores))
