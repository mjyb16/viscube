from __future__ import annotations
from typing import Callable, Tuple, Sequence, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import inspect
from tqdm import tqdm
from functools import wraps

# Use your existing implementations
from .gridder import bin_data, calibrated_bin_data
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

def _bind_window(fn, pixel_size, window_kwargs):
    """
    Return a callable window(u, center) with kwargs safely bound.
    Only passes arguments that `fn` actually accepts.
    Always passes pixel_size if `fn` accepts it and it's not already provided.
    """
    params = inspect.signature(fn).parameters
    kw = dict(window_kwargs or {})
    if "pixel_size" in params and "pixel_size" not in kw:
        kw["pixel_size"] = pixel_size

    @wraps(fn)
    def bound(u, c):
        return fn(u, c, **kw)

    bound._window_base = fn
    bound._window_kwargs = kw
    return bound


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
    frequencies: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    vis_re: np.ndarray,
    vis_imag: np.ndarray,
    weight: np.ndarray,
    npix: int = 501,
    pad_uv: float = 0.0,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable[[ArrayLike, float], np.ndarray]] = None,
    p_metric: int = 1,
    std_min_effective: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    u_edges, v_edges, delta_u, trunc_r = make_uv_grid(uu, vv, npix=npix, pad_uv=pad_uv)
    centers = build_grid_centers(u_edges, v_edges)

    if window_fn is not None:
        window = _bind_window(window_fn, pixel_size=delta_u, window_kwargs=window_kwargs)
    else:
        if window_name is None:
            raise ValueError("Provide either window_name or a ready-made window_fn.")
        window = _window_from_name(window_name, pixel_size=delta_u, window_kwargs=window_kwargs)

    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    mean_re = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nu, Nv), dtype=np.float64)

    pbar = tqdm(range(F), unit="channel",  ncols=200)
    for i in pbar:
        uv_tree, grid_tree, pairs = precompute_pairs(uu[i], vv[i], centers, trunc_r, p_metric=p_metric)

        vb_re = calibrated_bin_data(
            uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="mean", verbose=0
        )
        vb_im = calibrated_bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="mean", verbose=0
        )

        # std (Re/Im) + stats
        sb_re, stats_re = calibrated_bin_data(
            uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_neff=std_min_effective,
            collect_stats=True
        )
        sb_im, stats_im = calibrated_bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_neff=std_min_effective,
            collect_stats=True
        )

        cnt = calibrated_bin_data(
            uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="count", verbose=0
        )

        mean_re[i] = vb_re
        mean_im[i] = vb_im
        std_re[i]  = sb_re
        std_im[i]  = sb_im
        counts[i]  = cnt

        # NEW: pixels that used calibrated low-information sigma
        n_fallback_re = int(stats_re.get("n_fallback", 0))
        n_fallback_im = int(stats_im.get("n_fallback", 0))
        C_hat_re = stats_re.get("C_hat", np.nan)
        C_hat_im = stats_im.get("C_hat", np.nan)

        pbar.set_postfix(
            fallback_std_re=n_fallback_re,
            fallback_std_im=n_fallback_im,
            C_re=float(C_hat_re) if np.isfinite(C_hat_re) else np.nan,
            C_im=float(C_hat_im) if np.isfinite(C_hat_im) else np.nan,
        )

    return (np.flip(np.asarray(mean_re), axis=1),
            np.flip(np.asarray(mean_im), axis=1),
            np.flip(np.asarray(std_re),  axis=1),
            np.flip(np.asarray(std_im),  axis=1),
            np.flip(np.asarray(counts),  axis=1),
            u_edges, v_edges)


def _make_w_edges(
    ww: np.ndarray,
    w_bins: Union[int, np.ndarray],
    *,
    w_range: Optional[Tuple[float, float]] = None,
    w_abs: bool = False,
) -> np.ndarray:
    """
    Create w bin edges.

    Parameters
    ----------
    ww : ndarray
        Full w array used to determine default range.
    w_bins : int or ndarray
        If int, number of uniform bins in w. If ndarray, explicit bin edges.
    w_range : (min, max), optional
        Range for uniform bins. If None, uses data min/max (after abs if w_abs=True).
    w_abs : bool
        If True, bins |w| instead of w.

    Returns
    -------
    w_edges : ndarray, shape (Nw+1,)
    """
    wvals = np.asarray(ww, dtype=float)
    if w_abs:
        wvals = np.abs(wvals)

    if isinstance(w_bins, np.ndarray):
        w_edges = np.asarray(w_bins, dtype=float)
        if w_edges.ndim != 1 or w_edges.size < 2:
            raise ValueError("If w_bins is an array, it must be 1D with length >= 2 (bin edges).")
        if not np.all(np.isfinite(w_edges)):
            raise ValueError("w bin edges contain non-finite values.")
        if np.any(np.diff(w_edges) <= 0):
            raise ValueError("w bin edges must be strictly increasing.")
        return w_edges

    # integer number of bins
    n_w = int(w_bins)
    if n_w <= 0:
        raise ValueError("If w_bins is an int, it must be >= 1.")

    if w_range is None:
        wmin = float(np.nanmin(wvals))
        wmax = float(np.nanmax(wvals))
    else:
        wmin, wmax = map(float, w_range)

    if not np.isfinite(wmin) or not np.isfinite(wmax):
        raise ValueError("w range contains non-finite values.")
    if wmax <= wmin:
        raise ValueError(f"Invalid w range: max ({wmax}) must be > min ({wmin}).")

    return np.linspace(wmin, wmax, n_w + 1, dtype=float)


def grid_cube_all_stats_wbinned(
    *,
    frequencies: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
    vis_re: np.ndarray,
    vis_imag: np.ndarray,
    weight: np.ndarray,
    npix: int = 501,
    pad_uv: float = 0.0,
    w_bins: Union[int, np.ndarray] = 8,
    w_range: Optional[Tuple[float, float]] = None,
    w_abs: bool = False,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable[[ArrayLike, float], np.ndarray]] = None,
    p_metric: int = 1,
    std_min_effective: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if uu.shape != vv.shape or uu.shape != ww.shape:
        raise ValueError(f"uu, vv, ww must have the same shape. Got uu={uu.shape}, vv={vv.shape}, ww={ww.shape}.")
    if vis_re.shape != uu.shape or vis_imag.shape != uu.shape or weight.shape != uu.shape:
        raise ValueError("vis_re, vis_imag, weight must match uu/vv/ww shape.")

    u_edges, v_edges, delta_u, trunc_r = make_uv_grid(uu, vv, npix=npix, pad_uv=pad_uv)
    centers = build_grid_centers(u_edges, v_edges)

    w_edges = _make_w_edges(ww, w_bins, w_range=w_range, w_abs=w_abs)
    Nw = len(w_edges) - 1

    if window_fn is not None:
        window = _bind_window(window_fn, pixel_size=delta_u, window_kwargs=window_kwargs)
    else:
        if window_name is None:
            raise ValueError("Provide either window_name or a ready-made window_fn.")
        window = _window_from_name(window_name, pixel_size=delta_u, window_kwargs=window_kwargs)

    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    mean_re = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)

    pbar = tqdm(range(F), unit="channel", desc="Channels")
    for i in pbar:
        wvals = ww[i].ravel().astype(float)
        if w_abs:
            wvals = np.abs(wvals)

        wbin = np.digitize(wvals, w_edges, right=False) - 1
        valid = (wbin >= 0) & (wbin < Nw)

        # totals for channel postfix
        n_fallback_re_total = 0
        n_fallback_im_total = 0

        u_all   = uu[i].ravel()
        v_all   = vv[i].ravel()
        re_all  = vis_re[i].ravel()
        im_all  = vis_imag[i].ravel()
        wgt_all = weight[i].ravel()

        wbar = tqdm(range(Nw), unit="wbin", desc=f"w-bins (ch {i+1}/{F})", leave=False, ncols=200)
        for b in wbar:
            sel = valid & (wbin == b)
            if not np.any(sel):
                wbar.set_postfix_str("empty")
                continue

            u_b   = u_all[sel]
            v_b   = v_all[sel]
            re_b  = re_all[sel]
            im_b  = im_all[sel]
            wgt_b = wgt_all[sel]

            uv_tree, grid_tree, pairs = precompute_pairs(u_b, v_b, centers, trunc_r, p_metric=p_metric)

            vb_re = calibrated_bin_data(
                u_b, v_b, re_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="mean", verbose=0
            )
            vb_im = calibrated_bin_data(
                u_b, v_b, im_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="mean", verbose=0
            )

            sb_re, stats_re = calibrated_bin_data(
                u_b, v_b, re_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="std", verbose=0,
                std_min_neff=std_min_effective,
                collect_stats=True
            )
            sb_im, stats_im = calibrated_bin_data(
                u_b, v_b, im_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="std", verbose=0,
                std_min_neff=std_min_effective,
                collect_stats=True
            )

            cnt = calibrated_bin_data(
                u_b, v_b, re_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="count", verbose=0
            )

            mean_re[i, b] = vb_re
            mean_im[i, b] = vb_im
            std_re[i, b]  = sb_re
            std_im[i, b]  = sb_im
            counts[i, b]  = cnt

            n_fallback_re = int(stats_re.get("n_fallback", 0))
            n_fallback_im = int(stats_im.get("n_fallback", 0))
            n_fallback_re_total += n_fallback_re
            n_fallback_im_total += n_fallback_im

            # wbar shows number of points + fallback pixels + (optional) C_hat
            wbar.set_postfix(
                n=int(sel.sum()),
                fb_re=n_fallback_re,
                fb_im=n_fallback_im,
                C_re=float(stats_re.get("C_hat", np.nan)),
                C_im=float(stats_im.get("C_hat", np.nan)),
            )

        pbar.set_postfix(
            w_bins=Nw,
            fallback_std_re=int(n_fallback_re_total),
            fallback_std_im=int(n_fallback_im_total),
        )

    return (np.flip(np.asarray(mean_re), axis=2),
            np.flip(np.asarray(mean_im), axis=2),
            np.flip(np.asarray(std_re),  axis=2),
            np.flip(np.asarray(std_im),  axis=2),
            np.flip(np.asarray(counts),  axis=2),
            u_edges, v_edges, w_edges)

