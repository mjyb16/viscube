import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, Callable, Sequence, Union
from scipy.special import i0  # Modified Bessel I0
try:
    from scipy.special import pro_ang1
    _HAVE_PRO_ANG1 = True
except ImportError:
    _HAVE_PRO_ANG1 = False


def bin_data(
    u, v, values, weights, invvar_group, bins,
    window_fn: Callable,
    truncation_radius,
    uv_tree: cKDTree,
    grid_tree: cKDTree,
    pairs: Sequence[Sequence[int]],
    statistics_fn="mean",
    verbose=0,
    window_kwargs: Optional[Dict] = None,
    std_p: int = 1,
    std_workers: int = 6,
    std_min_effective: int = 5,
    std_expand_step: float = 0.1,
    collect_stats: bool = False,
):
    """
    Hybrid std behavior:
      - Normal pixels: empirical within-pixel scatter -> SE(mean)
      - Low-info pixels: propagated SE(mean) using invvar_group (per-visibility inverse variance)

    Parameters
    ----------
    invvar_group : ndarray or None
        Per-visibility inverse variance aligned with `values` (same length as u/v/values).
        Used ONLY in low-info std fallback.
    """
    u_edges, v_edges = bins
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    # Keep your current shape convention (we can revisit later)
    grid = np.zeros((Nu, Nv), dtype=float)

    n_fallback = 0

    for k, data_indices in enumerate(pairs):
        if not data_indices:
            continue

        u_center, v_center = grid_tree.data[k]

        # kernel
        wu = window_fn(u[data_indices], u_center)
        wv = window_fn(v[data_indices], v_center)
        imp = wu * wv

        # weighted mean/count weights use your original weights
        w = weights[data_indices] * imp
        if w.sum() <= 0:
            continue

        val = values[data_indices]
        i, j = divmod(k, Nv)

        if statistics_fn == "mean":
            grid[i, j] = np.sum(val * w) / np.sum(w)

        elif statistics_fn == "std":
            indices = data_indices

            # Recompute kernel on `indices` (same as above, but explicit for clarity)
            wu = window_fn(u[indices], u_center)
            wv = window_fn(v[indices], v_center)
            imp = wu * wv

            # "Information" criterion (same spirit as your original)
            effective = (imp > 0).sum()

            # ---------------------------
            # LOW-INFO FALLBACK: propagate SE(mean) from group invvar
            # ---------------------------
            if effective < std_min_effective:
                n_fallback += 1

                if invvar_group is None:
                    grid[i, j] = np.nan
                    continue

                invv = np.asarray(invvar_group[indices], dtype=float)

                ok = np.isfinite(invv) & (invv > 0) & np.isfinite(imp) & (imp > 0)
                if not np.any(ok):
                    grid[i, j] = np.nan
                    continue

                imp_ok = imp[ok]
                invv_ok = invv[ok]

                # Var(mu_hat) = sum(imp^2 * invvar) / (sum(imp * invvar))^2
                den = np.sum(imp_ok * invv_ok)
                num = np.sum((imp_ok**2) * invv_ok)

                grid[i, j] = np.sqrt(num) / (den + 1e-30)
                continue

            # ---------------------------
            # NORMAL CASE: empirical within-pixel scatter -> SE(mean)
            # ---------------------------
            local_w = weights[indices] * imp
            if np.sum(local_w) <= 0:
                grid[i, j] = np.nan
                continue

            val = values[indices]
            mean_val = np.sum(val * local_w) / np.sum(local_w)
            var = np.sum(local_w * (val - mean_val)**2) / np.sum(local_w)

            # kernel-only effective sample size (your existing choice)
            n_eff = (imp.sum() ** 2) / (np.sum(imp**2) + 1e-12)

            if n_eff <= 1:
                grid[i, j] = np.nan
            else:
                grid[i, j] = np.sqrt(var) * np.sqrt(n_eff / (n_eff - 1.0)) * (1.0 / np.sqrt(n_eff))

        elif statistics_fn == "count":
            grid[i, j] = (w > 0).sum()

        elif callable(statistics_fn):
            grid[i, j] = statistics_fn(val, w)

    if collect_stats:
        return grid, n_fallback
    return grid