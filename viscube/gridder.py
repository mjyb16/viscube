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

def calibrated_bin_data(
    u, v, values, weights, bins,
    window_fn: Callable,
    truncation_radius,
    uv_tree,
    grid_tree,
    pairs: Sequence[Sequence[int]],
    statistics_fn="mean",
    verbose=0,
    window_kwargs: Optional[Dict] = None,

    # --- std controls ---
    std_min_neff: float = 5.0,     # threshold on effective sample size
    std_floor: float = 0.0,        # optional floor on sigma
    collect_stats: bool = False,   # if True, return (grid, stats_dict)
):
    """
    statistics_fn="std":
      - Returns sigma for the *weighted mean* in each cell (SE-of-mean).
      - Uses two-pass approach:
          Pass 1: compute per-cell sigma^2 (high-n_eff cells) and Q shape term,
                  estimate C via median(sigma^2/Q).
          Pass 2: fill grid with within-cell sigma where available; otherwise
                  use fallback sigma = sqrt(C)*sqrt(Q).

    Here:
      g_k = wu_k * wv_k
      a_k = weights_k * g_k
      Q   = (sum weights*g^2) / (sum weights*g)^2
      n_eff = (sum a)^2 / sum(a^2)
    """

    u_edges, v_edges = bins
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    grid = np.zeros((Nu, Nv), dtype=float)

    # Fast paths for mean/count stay single-pass
    if statistics_fn in ("mean", "count") or callable(statistics_fn):
        for k, data_indices in enumerate(pairs):
            if not data_indices:
                continue
            u_center, v_center = grid_tree.data[k]

            wu = window_fn(u[data_indices], u_center)
            wv = window_fn(v[data_indices], v_center)
            g = wu * wv

            # effective weights for mean/count
            a = weights[data_indices] * g
            a = np.where(a > 0, a, 0.0)
            sw = float(a.sum())
            if sw <= 0:
                continue

            i, j = divmod(k, Nv)

            if statistics_fn == "mean":
                val = values[data_indices]
                grid[j, i] = np.sum(val * a) / sw

            elif statistics_fn == "count":
                grid[j, i] = float((a > 0).sum())

            else:  # custom callable
                val = values[data_indices]
                grid[j, i] = statistics_fn(val, a)

        if collect_stats:
            return grid, {"n_fallback": 0, "C_hat": np.nan}
        return grid

    # --- std case: two-pass ---
    if statistics_fn != "std":
        raise ValueError(f"Unknown statistics_fn={statistics_fn!r}")

    # Storage for pass 1 results
    sigma2_hi = np.full((Nu, Nv), np.nan, dtype=float)   # sigma^2 (SE mean) for high-n_eff cells
    Q_cell    = np.full((Nu, Nv), np.nan, dtype=float)   # Q for all cells (we'll fill where possible)
    neff_cell = np.full((Nu, Nv), 0.0, dtype=float)      # for diagnostics / thresholding

    # Pass 1: compute sigma^2 for high-n_eff cells, compute Q everywhere possible
    C_samples = []  # gather sigma^2 / Q for robust C estimate

    for k, data_indices in enumerate(pairs):
        if not data_indices:
            continue

        u_center, v_center = grid_tree.data[k]
        wu = window_fn(u[data_indices], u_center)
        wv = window_fn(v[data_indices], v_center)
        g  = wu * wv

        w  = weights[data_indices]
        # clip weights to avoid negative issues (shouldn't happen with MS WEIGHT, but safe)
        w  = np.where(w > 0, w, 0.0)

        # effective weights for the estimator
        a = w * g
        a = np.where(a > 0, a, 0.0)

        sw = float(a.sum())
        if sw <= 0:
            continue

        # location
        i, j = divmod(k, Nv)

        # Q term (uses w and g, not a, because a includes g once)
        S1 = float(np.sum(w * g))
        S2 = float(np.sum(w * g * g))
        if S1 > 0 and S2 > 0:
            Q = S2 / (S1 * S1 + 1e-30)
            Q_cell[j, i] = Q  # note: grid indexing is [j,i] in your convention

        # effective sample size
        sa2 = float(np.sum(a * a))
        n_eff = (sw * sw) / (sa2 + 1e-12)
        neff_cell[j, i] = n_eff

        # If enough information, compute within-cell SE-of-mean variance
        if n_eff >= std_min_neff:
            val = values[data_indices]
            mean_val = np.sum(val * a) / sw

            var = np.sum(a * (val - mean_val) ** 2) / sw
            if var < 0:
                var = 0.0

            # small-sample correction using n_eff
            if n_eff > 1.0:
                var *= n_eff / (n_eff - 1.0)

            # SE-of-mean variance:
            sigma2 = var / (n_eff + 1e-30)
            sigma2_hi[j, i] = sigma2

            # C sample if Q is available
            if np.isfinite(Q_cell[j, i]) and Q_cell[j, i] > 0:
                C_samples.append(sigma2 / Q_cell[j, i])

    # Robust C estimate
    if len(C_samples) > 0:
        C_hat = float(np.median(C_samples))
        if not np.isfinite(C_hat) or C_hat <= 0:
            C_hat = 1.0
    else:
        # If there are no high-n_eff cells in this subset, we cannot learn scale here.
        # Choose 1.0 (equivalent to assuming the MS weight scale is already correct).
        C_hat = 1.0

    # Pass 2: fill output grid using within-cell where available; fallback otherwise
    n_fallback = 0
    for j in range(Nu):
        for i in range(Nv):
            sigma2 = sigma2_hi[j, i]
            if np.isfinite(sigma2):
                sigma = float(np.sqrt(max(sigma2, 0.0)))
            else:
                Q = Q_cell[j, i]
                if np.isfinite(Q) and Q > 0:
                    n_fallback += 1
                    sigma = float(np.sqrt(C_hat * Q))
                else:
                    # no data / no weights -> leave as 0 (or np.nan if you prefer)
                    sigma = 0.0

            if std_floor > 0:
                sigma = max(sigma, std_floor)
            grid[j, i] = sigma

    if collect_stats:
        return grid, {"n_fallback": int(n_fallback), "C_hat": float(C_hat)}
    return grid


# Old main gridding code
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
            if var < 0:
                print("Error: negative weights")
                print(f"local_w values: {local_w}")
                print(f"wu values: {wu}")
                print(f"wv values: {wv}")
                print("Window base:", getattr(window_fn, "_window_base", window_fn))
                print("Window kwargs:", getattr(window_fn, "_window_kwargs", None))
                sys.exit()

        elif statistics_fn == "count":
            grid[j, i] = (w > 0).sum()

        elif callable(statistics_fn):
            grid[j, i] = statistics_fn(val, w)

    # `verbose` is deprecated in favor of tqdm in the caller.
    if collect_stats:
        return grid, n_coarse
    return grid

