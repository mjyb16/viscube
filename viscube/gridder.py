import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, Callable, Sequence, Union
from numpy.typing import ArrayLike
from scipy.special import i0  # Modified Bessel I0
try:
    from scipy.special import pro_ang1
    _HAVE_PRO_ANG1 = True
except ImportError:
    _HAVE_PRO_ANG1 = False


# Main gridding code
def bin_data(u, v, values, weights, bins,
             window_fn: Callable,
             truncation_radius,
             uv_tree: cKDTree,
             grid_tree: cKDTree,
             pairs: Sequence[Sequence[int]],
             statistics_fn="mean",
             verbose=1,
             window_kwargs: Optional[Dict] = None):
    """
    Parameters
    ----------
    window_fn : callable
        Should accept (u_array, center) ONLY (other params captured via partial or LUTWindow).
        Must return non-negative weights, zero outside support.
    window_kwargs : dict, optional
        (Not used if you already froze params via partial/LUTWindow; kept for backwards compat.)
    """
    u_edges, v_edges = bins
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    grid = np.zeros((Nu, Nv), dtype=float)

    n_coarse = 0
    # Iterate per grid center
    for k, data_indices in enumerate(pairs):
        if not data_indices:
            continue

        u_center, v_center = grid_tree.data[k]
        # 1D separable window
        wu = window_fn(u[data_indices], u_center)
        wv = window_fn(v[data_indices], v_center)
        w = weights[data_indices] * wu * wv

        if w.sum() <= 0:
            continue

        val = values[data_indices]
        i, j = divmod(k, Nv)   # careful with ordering (Nu major?)

        if statistics_fn == "mean":
            grid[j, i] = np.sum(val * w) / np.sum(w)

        elif statistics_fn == "std":
            # Expand adaptively like your original version. Start with given support m=1 concept.
            # We'll mimic your adaptive m logic by gradually enlarging search radius (L1).
            indices = data_indices
            local_w = w
            effective = (local_w > 0).sum()
            expand = 1.0
            while effective < 5:
                expand += 0.1
                indices = uv_tree.query_ball_point([u_center, v_center],
                                                   expand * truncation_radius,
                                                   p=1, workers=6)
                val = values[indices]
                wu = window_fn(u[indices], u_center)
                wv = window_fn(v[indices], v_center)
                local_w = weights[indices] * wu * wv
                effective = (local_w > 0).sum()
            if expand > 1.0:
                n_coarse += 1
            # Effective sample size
            imp = wu * wv
            n_eff = (imp.sum() ** 2) / (np.sum(imp**2) + 1e-12)
            # Weighted variance
            mean_val = np.sum(val * local_w) / np.sum(local_w)
            var = np.sum(local_w * (val - mean_val)**2) / np.sum(local_w)
            # Unbiased-ish scaling with effective n
            grid[j, i] = np.sqrt(var) * np.sqrt(n_eff / (max(n_eff - 1, 1))) * (1 / np.sqrt(n_eff))

        elif statistics_fn == "count":
            grid[j, i] = (w > 0).sum()

        elif callable(statistics_fn):
            grid[j, i] = statistics_fn(val, w)

    if verbose:
        print(f"Number of coarsened pixels: {n_coarse}")
    return grid

