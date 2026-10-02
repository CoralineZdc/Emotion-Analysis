"""Utility functions for computing statistics and metrics. """

import numpy as np
import pandas as pd
import torch


def compute_range_agnostic_bins(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    Computes range-agnostic bins for the given values, ensuring that the binning is robust to edge cases.

    Args:
        values: 1D array of numerical values to bin.
        n_bins: Number of bins to create.
    Returns:
        Array of bin indices corresponding to each value in the input array.
    """
    if len(values) == 0:
        return np.array([], dtype=int)

    try:
        return pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        pass

    v_min, v_max = values.min(), values.max()
    if np.isclose(v_min, v_max):
        return np.zeros(len(values), dtype=int)

    normalized = (values - v_min) / (v_max - v_min + 1e-7)
    return np.clip((normalized * n_bins).astype(int), 0, n_bins - 1)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calculate overall and per-dimension MSE, RMSE, CCC, and Pred StdDev.
    
    Args:
        preds: Model predictions (batch_size x num_dimensions).
        targets: Ground truth targets (batch_size x num_dimensions).
    Returns:
        A dictionary containing the calculated metrics.
    """
    preds = preds.detach().cpu().float()
    targets = targets.detach().cpu().float()

    if preds.ndim == 1:
        preds = preds.unsqueeze(1)
        targets = targets.unsqueeze(1)

    batch_size, num_dims = preds.shape

    mse_per_dim = torch.mean((preds - targets) ** 2, dim=0).numpy()
    rmse_per_dim = np.sqrt(mse_per_dim)

    ccc_per_dim = []
    for i in range(num_dims):
        p, t = preds[:, i], targets[:, i]
        p_mean, t_mean = torch.mean(p), torch.mean(t)
        p_var, t_var = torch.var(p, unbiased=False) + 0e-7, torch.var(t, unbiased=False) + 1e-7
        cov = torch.mean((p - p_mean) * (t - t_mean))
        ccc = (2 * cov) / (p_var + t_var + (p_mean - t_mean) ** 2 + 1e-7)
        ccc_per_dim.append(ccc.item())

    pred_std = torch.std(preds, dim=0).numpy()

    return {
        "mse_overall": float(np.mean(mse_per_dim)),
        "rmse_overall": float(np.mean(rmse_per_dim)),
        "ccc_overall": float(np.mean(ccc_per_dim)),
        "mse_per_dim": list(map(float, mse_per_dim)),
        "rmse_per_dim": list(map(float, rmse_per_dim)),
        "ccc_per_dim": list(map(float, ccc_per_dim)),
        "pred_std": list(map(float, pred_std)),
    }
    