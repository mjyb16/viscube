from __future__ import annotations
from typing import Callable, Tuple, Sequence
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree


def load_and_mask(
    frequencies: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    vis: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply per-channel mask and compact arrays (exactly like your loop).

    Returns
    -------
    freq : (F,)
    u0, v0 : (F, Nmasked)
    vis0 : (F, Nmasked) complex128
    w0   : (F, Nmasked) float64
    """
    F = len(frequencies)
    Nmasked = int(mask[0].sum())
    u0 = np.zeros((F, Nmasked), dtype=np.float64)
    v0 = np.zeros((F, Nmasked), dtype=np.float64)
    vis0 = np.zeros((F, Nmasked), dtype=np.complex128)
    w0 = np.zeros((F, Nmasked), dtype=np.float64)
    for i in range(F):
        mi = mask[i]
        u0[i] = uu[i][mi]
        v0[i] = vv[i][mi]
        vis0[i] = vis[i][mi]
        w0[i] = weight[i][mi]
    return frequencies, u0, v0, vis0, w0


def hermitian_augment(
    u0: np.ndarray, v0: np.ndarray, vis0: np.ndarray, w0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hermitian augmentation exactly as in your script:
    (u, v, Re, Im, w) -> concat with (-u, -v, +Re, -Im, w)

    Returns
    -------
    uu, vv : (F, 2*N)
    vis_re, vis_imag : (F, 2*N)
    w : (F, 2*N)
    """
    uu = np.concatenate([u0, -u0], axis=1)
    vv = np.concatenate([v0, -v0], axis=1)
    vis_re = np.concatenate([vis0.real, vis0.real], axis=1)
    vis_imag = np.concatenate([vis0.imag, -vis0.imag], axis=1)
    w = np.concatenate([w0, w0], axis=1)
    return uu, vv, vis_re, vis_imag, w


def make_uv_grid(
    uu: np.ndarray, vv: np.ndarray, npix: int, pad_uv: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build symmetric square uv grid (unchanged logic).

    Returns
    -------
    u_edges, v_edges : (npix+1,)
    delta_u : float
    truncation_radius : float (== delta_u)
    """
    maxuv = max(np.abs(uu).max(), np.abs(vv).max())
    u_min = -maxuv * (1.0 + pad_uv)
    u_max = +maxuv * (1.0 + pad_uv)
    u_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
    v_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
    delta_u = float(u_edges[1] - u_edges[0])
    truncation_radius = delta_u  # L1 radius, same as your code
    return u_edges, v_edges, delta_u, truncation_radius


def build_grid_centers(u_edges: np.ndarray, v_edges: np.ndarray) -> np.ndarray:
    """
    Reproduce your center ordering EXACTLY:
    outer loop over u bins, inner loop over v bins.

    This preserves the downstream `i, j = divmod(k, Nv)` with `grid[j, i]` in your bin_data.
    """
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    centers = np.array(
        [
            ((u_edges[k] + u_edges[k + 1]) / 2.0, (v_edges[j] + v_edges[j + 1]) / 2.0)
            for k in range(Nu)
            for j in range(Nv)
        ],
        dtype=float,
    )
    return centers


def precompute_pairs(
    uu_i: np.ndarray,
    vv_i: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    centers: np.ndarray,
    truncation_radius: float,
    *,
    p_metric: int = 1,
    workers: int = 6,
) -> Tuple[cKDTree, cKDTree, Sequence[Sequence[int]]]:
    """
    Build KD-trees and query neighbor pairs for a single channel (same as your loop).
    """
    uv_points = np.vstack((uu_i.ravel(), vv_i.ravel())).T
    uv_tree = cKDTree(uv_points)
    grid_tree = cKDTree(centers)
    pairs = grid_tree.query_ball_tree(uv_tree, truncation_radius, p=p_metric, workers=workers)
    return uv_tree, grid_tree, pairs


def grid_channel(
    uu_i: np.ndarray,
    vv_i: np.ndarray,
    vis_re_i: np.ndarray,
    vis_imag_i: np.ndarray,
    w_i: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    window_fn: Callable[[ArrayLike, float], np.ndarray],
    truncation_radius: float,
    uv_tree: cKDTree,
    grid_tree: cKDTree,
    pairs: Sequence[Sequence[int]],
    *,
    bin_data: Callable = None,
    verbose_mean: int = 1,
    verbose_std: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid one frequency channel using your existing `bin_data`.

    Notes
    -----
    - Keeps your statistics functions and verbosity defaults:
        mean -> verbose=1, std -> verbose=2, count -> verbose=1
    - Does NOT alter the `grid[j, i]` behavior inside `bin_data`.
    """
    if bin_data is None:
        raise ValueError("Please pass your existing bin_data via the `bin_data` argument.")

    bins = (u_edges, v_edges)
    params = (uu_i, vv_i, w_i, bins, window_fn, truncation_radius, uv_tree, grid_tree, pairs)

    vis_bin_re   = bin_data(uu_i, vv_i, vis_re_i, *params[2:], statistics_fn="mean",  verbose=verbose_mean)
    std_bin_re   = bin_data(uu_i, vv_i, vis_re_i, *params[2:], statistics_fn="std",   verbose=verbose_std)
    vis_bin_imag = bin_data(uu_i, vv_i, vis_imag_i, *params[2:], statistics_fn="mean", verbose=verbose_mean)
    std_bin_imag = bin_data(uu_i, vv_i, vis_imag_i, *params[2:], statistics_fn="std",  verbose=verbose_std)
    counts       = bin_data(uu_i, vv_i, vis_re_i,  *params[2:], statistics_fn="count", verbose=verbose_mean)

    return vis_bin_re, std_bin_re, vis_bin_imag, std_bin_imag, counts


def grid_all_channels(
    uu: np.ndarray,
    vv: np.ndarray,
    vis_re: np.ndarray,
    vis_imag: np.ndarray,
    w: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    centers: np.ndarray,
    window_fn: Callable[[ArrayLike, float], np.ndarray],
    truncation_radius: float,
    *,
    bin_data: Callable,
    workers: int = 6,
    p_metric: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loop over channels and grid each one with the same behavior as your current script.

    Returns
    -------
    mean_re, std_re, mean_im, std_im, counts : arrays with shape (F, Nu, Nv)
    """
    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    mean_re = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nu, Nv), dtype=np.float64)

    for i in range(F):
        uv_tree, grid_tree, pairs = precompute_pairs(
            uu[i], vv[i], u_edges, v_edges, centers, truncation_radius, p_metric=p_metric, workers=workers
        )
        vb_re, sb_re, vb_im, sb_im, cnt = grid_channel(
            uu[i], vv[i], vis_re[i], vis_imag[i], w[i],
            u_edges, v_edges, window_fn, truncation_radius,
            uv_tree, grid_tree, pairs, bin_data=bin_data
        )
        mean_re[i] = vb_re
        std_re[i]  = sb_re
        mean_im[i] = vb_im
        std_im[i]  = sb_im
        counts[i]  = cnt

    return mean_re, std_re, mean_im, std_im, counts
