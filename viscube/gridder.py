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

from .windows import kb_kernel_1d


def bin_channel_nonoverlap(
    u, v, values, weights, invvar_group,
    u_edges, v_edges,
    *,
    m: int = 1,
    beta: float = 2.0,
    std_min_effective: int = 5,
    n_eff_mode: str = "both",
    dc_dedup_from: Optional[int] = None,
):
    """
    Vectorized NON-OVERLAPPING binner for one channel: each visibility
    contributes to exactly one uv cell (its containing bin, found with
    searchsorted on the edges), with within-bin weighting
    ``weights * KB(du) * KB(dv)`` (``beta = 0`` gives a pillbox). Replaces
    the KDTree kernel-overlap path of `bin_data` while reproducing its
    per-cell statistics exactly:

      - mean : normalized weighted mean per cell
      - std  : SE of the weighted mean with the n_eff small-sample
               correction; low-info cells (n_eff < std_min_effective)
               fall back to the propagated invvar SE; n_eff <= 1 -> NaN
      - counts : number of contributing samples per cell

    Parameters mirror `bin_data`; additionally

    dc_dedup_from : int, optional
        If given, samples with index >= dc_dedup_from (i.e. the
        hermitian-augmented copies, when the first ``dc_dedup_from``
        entries are the originals) whose target cell is the DC cell are
        DROPPED. Without this, a visibility with |u|,|v| < delta_u/2 and
        its augmented conjugate both land in the DC cell (imag forced to
        ~0, weight doubled, SE understated by sqrt(2)).

    Returns
    -------
    mean, std, counts : ndarray (Nu, Nv)
    stats : dict with n_fallback, n_dropped (outside grid), n_dc_dropped
    """
    if n_eff_mode not in {"geometric", "both"}:
        raise ValueError(f"n_eff_mode must be 'geometric' or 'both', got {n_eff_mode!r}")

    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()

    u_edges = np.asarray(u_edges, dtype=float)
    v_edges = np.asarray(v_edges, dtype=float)
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    ncell = Nu * Nv

    iu = np.searchsorted(u_edges, u, side="right") - 1
    jv = np.searchsorted(v_edges, v, side="right") - 1
    valid = (iu >= 0) & (iu < Nu) & (jv >= 0) & (jv < Nv)
    n_dropped = int(np.count_nonzero(~valid))

    # DC-cell dedup of hermitian-augmented copies
    n_dc_dropped = 0
    if dc_dedup_from is not None:
        iu0 = int(np.searchsorted(u_edges, 0.0, side="right") - 1)
        jv0 = int(np.searchsorted(v_edges, 0.0, side="right") - 1)
        idx = np.arange(u.size)
        dc_dup = valid & (idx >= int(dc_dedup_from)) & (iu == iu0) & (jv == jv0)
        n_dc_dropped = int(np.count_nonzero(dc_dup))
        valid &= ~dc_dup

    valid &= np.isfinite(weights) & (weights > 0)

    # within-bin kernel weights (offsets from cell centers)
    du_cell = float(u_edges[1] - u_edges[0])
    dv_cell = float(v_edges[1] - v_edges[0])
    u_ctr = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_ctr = 0.5 * (v_edges[:-1] + v_edges[1:])

    iu_v = iu[valid]
    jv_v = jv[valid]
    imp = (kb_kernel_1d(u[valid] - u_ctr[iu_v], delta_u=du_cell, m=m, beta=beta)
           * kb_kernel_1d(v[valid] - v_ctr[jv_v], delta_u=dv_cell, m=m, beta=beta))
    w = weights[valid] * imp
    val = values[valid]
    cell = iu_v.astype(np.int64) * Nv + jv_v.astype(np.int64)

    sum_w = np.bincount(cell, weights=w, minlength=ncell)
    sum_wv = np.bincount(cell, weights=w * val, minlength=ncell)
    occupied = sum_w > 0

    mean = np.zeros(ncell, dtype=np.float64)
    mean[occupied] = sum_wv[occupied] / sum_w[occupied]

    # two-pass weighted variance (avoids catastrophic cancellation)
    dev2 = (val - mean[cell]) ** 2
    sum_wd2 = np.bincount(cell, weights=w * dev2, minlength=ncell)
    var_w = np.zeros(ncell, dtype=np.float64)
    var_w[occupied] = sum_wd2[occupied] / sum_w[occupied]

    # effective sample size (same definitions as bin_data)
    sum_w2 = np.bincount(cell, weights=w * w, minlength=ncell)
    sum_imp = np.bincount(cell, weights=imp, minlength=ncell)
    sum_imp2 = np.bincount(cell, weights=imp * imp, minlength=ncell)
    if n_eff_mode == "geometric":
        n_eff = np.zeros(ncell, dtype=np.float64)
        n_eff[occupied] = sum_imp[occupied] ** 2 / (sum_imp2[occupied] + 1e-12)
    else:
        n_eff = np.zeros(ncell, dtype=np.float64)
        n_eff[occupied] = sum_w[occupied] ** 2 / (sum_w2[occupied] + 1e-12)

    # normal-case SE of the weighted mean
    std = np.zeros(ncell, dtype=np.float64)
    normal = occupied & (n_eff >= std_min_effective)
    gt1 = normal & (n_eff > 1)
    std[gt1] = (np.sqrt(var_w[gt1])
                * np.sqrt(n_eff[gt1] / (n_eff[gt1] - 1.0))
                / np.sqrt(n_eff[gt1]))
    std[normal & ~gt1] = np.nan

    # low-info fallback: propagated SE from per-visibility inverse variance
    fallback = occupied & (n_eff < std_min_effective)
    n_fallback = int(np.count_nonzero(fallback))
    if n_fallback:
        if invvar_group is None:
            std[fallback] = np.nan
        else:
            invv = np.asarray(invvar_group, dtype=float).ravel()[valid]
            ok = np.isfinite(invv) & (invv > 0) & np.isfinite(imp) & (imp > 0)
            invv_ok = np.where(ok, invv, 0.0)
            imp_ok = np.where(ok, imp, 0.0)
            den = np.bincount(cell, weights=imp_ok * invv_ok, minlength=ncell)
            num = np.bincount(cell, weights=imp_ok ** 2 * invv_ok, minlength=ncell)
            fb_ok = fallback & (den > 0)
            std[fb_ok] = np.sqrt(num[fb_ok]) / (den[fb_ok] + 1e-30)
            std[fallback & ~fb_ok] = np.nan

    counts = np.bincount(cell[w > 0], minlength=ncell).astype(np.float64)

    stats = {"n_fallback": n_fallback,
             "n_dropped": n_dropped,
             "n_dc_dropped": n_dc_dropped}
    return (mean.reshape(Nu, Nv), std.reshape(Nu, Nv),
            counts.reshape(Nu, Nv), stats)


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
    n_eff_mode: str = "both",
):
    """
    Hybrid std behavior:
      - Normal pixels: empirical within-pixel scatter -> SE(mean)
      - Low-info pixels: propagated SE(mean) using invvar_group
        (per-visibility inverse variance)

    Parameters
    ----------
    invvar_group : ndarray or None
        Per-visibility inverse variance aligned with `values`
        (same length as u/v/values).
        Used ONLY in low-info std fallback.

    n_eff_mode : {"geometric", "both"}
        Choice of effective sample size definition used consistently
        for both:
          1) the fallback trigger
          2) the normal-case SE(mean) correction

        - "geometric":
            n_eff = (sum imp)^2 / sum(imp^2)
            Uses kernel-only support / geometric weighting.

        - "both":
            n_eff = (sum local_w)^2 / sum(local_w^2)
            Uses full weights * kernel, i.e. incorporates both geometric
            interpolation weighting and measurement weighting.

    Returns
    -------
    grid : ndarray
        Output gridded statistic.

    If collect_stats is True:
        returns (grid, n_fallback)
    """
    allowed_modes = {"geometric", "both"}
    if n_eff_mode not in allowed_modes:
        raise ValueError(
            f"n_eff_mode must be one of {allowed_modes}, got {n_eff_mode!r}"
        )

    u_edges, v_edges = bins
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    grid = np.zeros((Nu, Nv), dtype=float)
    n_fallback = 0

    for k, data_indices in enumerate(pairs):
        if not data_indices:
            continue

        u_center, v_center = grid_tree.data[k]
        i, j = divmod(k, Nv)

        # Kernel weights
        wu = window_fn(u[data_indices], u_center)
        wv = window_fn(v[data_indices], v_center)
        imp = wu * wv

        # Combined measurement * interpolation weights
        local_w = weights[data_indices] * imp
        if np.sum(local_w) <= 0:
            continue

        val = values[data_indices]

        if statistics_fn == "mean":
            grid[i, j] = np.sum(val * local_w) / np.sum(local_w)

        elif statistics_fn == "std":
            # Unified N_eff: used for both fallback trigger and SE correction
            if n_eff_mode == "geometric":
                n_eff = (imp.sum() ** 2) / (np.sum(imp**2) + 1e-12)
            else:  # n_eff_mode == "both"
                n_eff = (local_w.sum() ** 2) / (np.sum(local_w**2) + 1e-12)

            # ---------------------------
            # LOW-INFO FALLBACK
            # ---------------------------
            if n_eff < std_min_effective:
                n_fallback += 1

                if invvar_group is None:
                    grid[i, j] = np.nan
                    continue

                invv = np.asarray(invvar_group[data_indices], dtype=float)

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
            # NORMAL CASE
            # ---------------------------
            mean_val = np.sum(val * local_w) / np.sum(local_w)
            var = np.sum(local_w * (val - mean_val)**2) / np.sum(local_w)

            if n_eff <= 1:
                grid[i, j] = np.nan
            else:
                grid[i, j] = (
                    np.sqrt(var)
                    * np.sqrt(n_eff / (n_eff - 1.0))
                    / np.sqrt(n_eff)
                )

        elif statistics_fn == "count":
            grid[i, j] = (local_w > 0).sum()

        elif callable(statistics_fn):
            grid[i, j] = statistics_fn(val, local_w)

    if collect_stats:
        return grid, n_fallback
    return grid