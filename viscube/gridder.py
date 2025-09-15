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
             verbose=0,
             window_kwargs: Optional[Dict] = None,
             # New: std-only controls (defaults preserve old behavior)
             std_p: int = 1,
             std_workers: int = 6,
             std_min_effective: int = 5,
             std_expand_step: float = 0.1,
             # New: return n_coarse for tqdm display when True
             collect_stats: bool = False):
    """
    Parameters
    ----------
    window_fn : callable
        Accepts (u_array, center). Other params captured via closure/partial.
    window_kwargs : dict, optional
        (Kept for backwards compat; not used when window_fn is already bound.)
    std_p : int
        `p` metric for cKDTree.query_ball_point during std expansion (default 1).
    std_workers : int
        `workers` for cKDTree.query_ball_point during std expansion (default 6).
    std_min_effective : int
        Minimum effective sample count before stopping expansion (default 5).
    std_expand_step : float
        Multiplicative radius increment per expansion step (default 0.1).
    collect_stats : bool
        If True, returns (grid, n_coarse). Otherwise returns grid only.
    """
    u_edges, v_edges = bins
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    grid = np.zeros((Nu, Nv), dtype=float)

    n_coarse = 0
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
        i, j = divmod(k, Nv)   # Nu-major ordering outside; fill grid[j, i] (unchanged)

        if statistics_fn == "mean":
            grid[j, i] = np.sum(val * w) / np.sum(w)

        elif statistics_fn == "std":
            indices = data_indices
            local_w = w
            effective = (local_w > 0).sum()
            expand = 1.0
            while effective < std_min_effective:
                expand += std_expand_step
                indices = uv_tree.query_ball_point([u_center, v_center],
                                                   expand * truncation_radius,
                                                   p=std_p, workers=std_workers)
                val = values[indices]
                wu = window_fn(u[indices], u_center)
                wv = window_fn(v[indices], v_center)
                local_w = weights[indices] * wu * wv
                effective = (local_w > 0).sum()
            if expand > 1.0:
                n_coarse += 1

            # Effective sample size & SE of the mean
            imp = wu * wv
            n_eff = (imp.sum() ** 2) / (np.sum(imp**2) + 1e-12)
            mean_val = np.sum(val * local_w) / np.sum(local_w)
            var = np.sum(local_w * (val - mean_val)**2) / np.sum(local_w)
            grid[j, i] = np.sqrt(var) * np.sqrt(n_eff / max(n_eff - 1, 1)) * (1.0 / np.sqrt(n_eff))

        elif statistics_fn == "count":
            grid[j, i] = (w > 0).sum()

        elif callable(statistics_fn):
            grid[j, i] = statistics_fn(val, w)

    # `verbose` is deprecated in favor of tqdm in the caller.
    if collect_stats:
        return grid, n_coarse
    return grid

