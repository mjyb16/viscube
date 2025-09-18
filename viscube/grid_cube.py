from __future__ import annotations
from typing import Callable, Tuple, Sequence, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import inspect
from tqdm import tqdm

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
    Apply per-channel mask and compact arrays. 
    Returns frequencies, u0, v0, vis0, w0.
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
    Returns uu, vv, vis_re, vis_imag, w
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
    Measurement Set conventions for grid centers.
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
    p_metric: int = 1
) -> Tuple[cKDTree, cKDTree, Sequence[Sequence[int]]]:
    """
    Build KD-trees and query neighbor pairs for a single channel.
    """
    uv_points = np.vstack((uu_i.ravel(), vv_i.ravel())).T
    uv_tree = cKDTree(uv_points)
    grid_tree = cKDTree(centers)
    pairs = grid_tree.query_ball_tree(uv_tree, truncation_radius, p=p_metric)
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

def _bind_window(fn: Callable, pixel_size: float, window_kwargs: Optional[dict]) -> Callable[[ArrayLike, float], np.ndarray]:
    """
    Return a callable window(u, center) with kwargs safely bound.
    Only passes arguments that `fn` actually accepts.
    Always passes pixel_size if `fn` accepts it and it's not already provided.
    """
    params = inspect.signature(fn).parameters
    kw = dict(window_kwargs or {})
    if "pixel_size" in params and "pixel_size" not in kw:
        kw["pixel_size"] = pixel_size

    # keep it minimal: caller can pass m/beta/normalize/etc in window_kwargs
    return lambda u, c, _fn=fn, _kw=kw: _fn(u, c, **_kw)


def _window_from_name(name: str,
                      *,
                      pixel_size: float,
                      window_kwargs: Optional[dict] = None
                      ) -> Callable[[ArrayLike, float], np.ndarray]:
    """
    Build a window(u, center) callable from a string and a kwargs dict.
    No assumptions about which kwargs exist; only forwards what the window accepts.
    """
    key = name.lower()
    if key in {"kb", "kaiser", "kaiser_bessel", "kaiser-bessel"}:
        base = kaiser_bessel_window
    elif key in {"pswf", "casa", "spheroidal"}:
        base = casa_pswf_window
    elif key in {"pillbox", "boxcar"}:
        base = pillbox_window
    elif key == "sinc":
        base = sinc_window
    else:
        raise ValueError(f"Unknown window name: {name!r}")

    return _bind_window(base, pixel_size=pixel_size, window_kwargs=window_kwargs)


def grid_cube_all_stats(
    *,
    # Required observational inputs:
    frequencies: np.ndarray,     # kept for API symmetry; unused here
    uu: np.ndarray,
    vv: np.ndarray,
    vis_re: np.ndarray,
    vis_imag: np.ndarray,
    weight: np.ndarray,
    # Grid config:
    npix: int = 501,
    pad_uv: float = 0.0,
    # Window config (choose either window_name OR pass a ready-made window_fn):
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable[[ArrayLike, float], np.ndarray]] = None,
    # KD-tree config:
    p_metric: int = 1,
    # New: std-only expansion controls passed into bin_data
    std_workers: int = 6,
    std_min_effective: int = 5,
    std_expand_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level API: provide raw UV data + a window choice/kwargs; get back gridded data.

    Returns
    -------
    mean_re, mean_im, std_re, std_im, counts, u_edges, v_edges
        Each grid has shape (F, Nu, Nv).
    """
    # 1) Build grid (pixel_size == delta_u)
    u_edges, v_edges, delta_u, trunc_r = make_uv_grid(uu, vv, npix=npix, pad_uv=pad_uv)
    centers = build_grid_centers(u_edges, v_edges)

    # 2) Build/bind window callable
    if window_fn is not None:
        window = _bind_window(window_fn, pixel_size=delta_u, window_kwargs=window_kwargs)
    else:
        if window_name is None:
            raise ValueError("Provide either window_name or a ready-made window_fn.")
        window = _window_from_name(window_name, pixel_size=delta_u, window_kwargs=window_kwargs)

    # 3) Allocate outputs
    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    mean_re = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nu, Nv), dtype=np.float64)

    # 4) Loop with tqdm + postfix showing coarsened pixels for std
    pbar = tqdm(range(F), unit="channel")
    for i in pbar:
        uv_tree, grid_tree, pairs = precompute_pairs(uu[i], vv[i], centers, trunc_r, p_metric=p_metric)

        # mean (Re/Im)
        vb_re = bin_data(uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
                         window, trunc_r, uv_tree, grid_tree, pairs,
                         statistics_fn="mean", verbose=0)
        vb_im = bin_data(uu[i], vv[i], vis_imag[i], weight[i], (u_edges, v_edges),
                         window, trunc_r, uv_tree, grid_tree, pairs,
                         statistics_fn="mean", verbose=0)

        # std (Re/Im) with stats collected
        sb_re, n_coarse_re = bin_data(uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
                                      window, trunc_r, uv_tree, grid_tree, pairs,
                                      statistics_fn="std", verbose=0,
                                      std_p=p_metric,
                                      std_workers=std_workers,
                                      std_min_effective=std_min_effective,
                                      std_expand_step=std_expand_step,
                                      collect_stats=True)
        sb_im, n_coarse_im = bin_data(uu[i], vv[i], vis_imag[i], weight[i], (u_edges, v_edges),
                                      window, trunc_r, uv_tree, grid_tree, pairs,
                                      statistics_fn="std", verbose=0,
                                      std_p=p_metric,
                                      std_workers=std_workers,
                                      std_min_effective=std_min_effective,
                                      std_expand_step=std_expand_step,
                                      collect_stats=True)

        # counts
        cnt = bin_data(uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
                       window, trunc_r, uv_tree, grid_tree, pairs,
                       statistics_fn="count", verbose=0)

        # store
        mean_re[i] = vb_re
        mean_im[i] = vb_im
        std_re[i]  = sb_re
        std_im[i]  = sb_im
        counts[i]  = cnt

        # tqdm postfix with std coarsening info
        pbar.set_postfix(coarse_std_re=int(n_coarse_re), coarse_std_im=int(n_coarse_im))

    # flip u-axis (axis=1) to match your NPZ saving convention
    return (np.flip(np.asarray(mean_re), axis=1),
            np.flip(np.asarray(mean_im), axis=1),
            np.flip(np.asarray(std_re),  axis=1),
            np.flip(np.asarray(std_im),  axis=1),
            np.flip(np.asarray(counts),  axis=1),
            u_edges, v_edges)

