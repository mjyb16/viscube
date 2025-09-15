from __future__ import annotations
from typing import Callable, Tuple, Sequence, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

# Use your existing implementations
from .gridder import bin_data
from .windows import (
    kaiser_bessel_window,
    casa_pswf_window,
    pillbox_window,
    sinc_window,
)

# -----------------------
# Low-level utilities (unchanged behavior)
# -----------------------

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
    (u, v, Re, Im, w) -> concat with (-u, -v, +Re, -Im, w)
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
    Build symmetric square uv grid; truncation_radius == delta_u.
    """
    maxuv = max(np.abs(uu).max(), np.abs(vv).max())
    u_min = -maxuv * (1.0 + pad_uv)
    u_max = +maxuv * (1.0 + pad_uv)
    u_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
    v_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
    delta_u = float(u_edges[1] - u_edges[0])
    truncation_radius = delta_u
    return u_edges, v_edges, delta_u, truncation_radius


def build_grid_centers(u_edges: np.ndarray, v_edges: np.ndarray) -> np.ndarray:
    """
    Reproduce your center ordering EXACTLY:
    outer loop over u bins, inner loop over v bins.

    This preserves downstream `i, j = divmod(k, Nv)` with `grid[j, i]` in bin_data.
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
    centers: np.ndarray,
    truncation_radius: float,
    *,
    p_metric: int = 1,
    workers: int = 6,
) -> Tuple[cKDTree, cKDTree, Sequence[Sequence[int]]]:
    """
    Build KD-trees and query neighbor pairs for a single channel.
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
    verbose_mean: int = 1,
    verbose_std: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid one frequency channel using your existing bin_data.
    """
    bins = (u_edges, v_edges)
    params = (uu_i, vv_i, w_i, bins, window_fn, truncation_radius, uv_tree, grid_tree, pairs)

    vis_bin_re   = bin_data(uu_i, vv_i, vis_re_i, *params[2:], statistics_fn="mean",  verbose=verbose_mean)
    std_bin_re   = bin_data(uu_i, vv_i, vis_re_i, *params[2:], statistics_fn="std",   verbose=verbose_std)
    vis_bin_imag = bin_data(uu_i, vv_i, vis_imag_i, *params[2:], statistics_fn="mean", verbose=verbose_mean)
    std_bin_imag = bin_data(uu_i, vv_i, vis_imag_i, *params[2:], statistics_fn="std",  verbose=verbose_std)
    counts       = bin_data(uu_i, vv_i, vis_re_i,  *params[2:], statistics_fn="count", verbose=verbose_mean)

    return vis_bin_re, std_bin_re, vis_bin_imag, std_bin_imag, counts


# -----------------------
# User-facing helpers
# -----------------------

def _window_from_name(name: str,
                      *,
                      pixel_size: float,
                      m: int = 6,
                      beta: Optional[float] = None
                      ) -> Callable[[ArrayLike, float], np.ndarray]:
    """
    Convenience: build a window(u, center) callable from a string and args.
    Note: pixel_size is bound here (computed from the grid).
    """
    key = name.lower()
    if key in {"kb", "kaiser", "kaiser_bessel", "kaiser-bessel"}:
        return lambda u, c: kaiser_bessel_window(u, c, pixel_size=pixel_size, m=m, beta=beta)
    if key in {"pswf", "casa", "spheroidal"}:
        return lambda u, c: casa_pswf_window(u, c, pixel_size=pixel_size, m=m)
    if key in {"pillbox", "boxcar"}:
        return lambda u, c: pillbox_window(u, c, pixel_size=pixel_size, m=m)
    if key in {"sinc"}:
        return lambda u, c: sinc_window(u, c, pixel_size=pixel_size, m=m)
    raise ValueError(f"Unknown window name: {name!r}")


def grid_cube_simple(
    *,
    # Required observational inputs:
    frequencies: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    vis: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    # Grid config:
    npix: int = 501,
    pad_uv: float = 0.0,
    # Window config (choose either window_name OR pass a ready-made window_fn):
    window_name: Optional[str] = "kaiser_bessel",
    m: int = 6,
    beta: Optional[float] = None,
    window_fn: Optional[Callable[[ArrayLike, float], np.ndarray]] = None,
    # KD-tree config:
    workers: int = 6,
    p_metric: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level API: users provide raw UV data + window choice; get back grids.

    Returns
    -------
    mean_re, mean_im, std_re, std_im, counts, u_edges, v_edges
        Each grid has shape (F, Nu, Nv).
    """
    # 1) Mask + compact
    _, u0, v0, vis0, w0 = load_and_mask(frequencies, uu, vv, vis, weight, mask)

    # 2) Hermitian augment
    U, V, RE, IM, W = hermitian_augment(u0, v0, vis0, w0)

    # 3) Build grid (pixel_size == delta_u)
    u_edges, v_edges, delta_u, trunc_r = make_uv_grid(U, V, npix=npix, pad_uv=pad_uv)
    centers = build_grid_centers(u_edges, v_edges)

    # 4) Build window callable if not provided; bind pixel_size here
    if window_fn is None:
        if window_name is None:
            raise ValueError("Provide either window_name or a ready-made window_fn.")
        window_fn = _window_from_name(window_name, pixel_size=delta_u, m=m, beta=beta)

    # 5) Loop over channels (uses your bin_data internally)
    F = U.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    mean_re = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nu, Nv), dtype=np.float64)

    for i in range(F):
        uv_tree, grid_tree, pairs = precompute_pairs(U[i], V[i], centers, trunc_r, p_metric=p_metric, workers=workers)
        vb_re, sb_re, vb_im, sb_im, cnt = grid_channel(
            U[i], V[i], RE[i], IM[i], W[i],
            u_edges, v_edges, window_fn, trunc_r,
            uv_tree, grid_tree, pairs,
            # keep your default verbosities:
            verbose_mean=1, verbose_std=2,
        )
        mean_re[i] = vb_re
        std_re[i]  = sb_re
        mean_im[i] = vb_im
        std_im[i]  = sb_im
        counts[i]  = cnt

    return mean_re, mean_im, std_re, std_im, counts, u_edges, v_edges


def save_gridded_npz(
    out_path: str,
    mean_re: np.ndarray,
    mean_im: np.ndarray,
    std_re: np.ndarray,
    std_im: np.ndarray,
    counts: np.ndarray,
) -> None:
    """
    Save NPZ exactly like your notebook snippet:
        - flip axis=1 (u-axis) for each grid
        - mask := flipped(counts) > 0
    """
    np.savez(
        out_path,
        vis_bin_re   = np.flip(np.asarray(mean_re), axis=1),
        vis_bin_imag = np.flip(np.asarray(mean_im), axis=1),
        std_bin_re   = np.flip(np.asarray(std_re),  axis=1),
        std_bin_imag = np.flip(np.asarray(std_im),  axis=1),
        mask         = np.flip(np.asarray(counts) > 0, axis=1),
    )
