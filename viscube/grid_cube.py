from __future__ import annotations
import warnings
from typing import Callable, Tuple, Sequence, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import inspect
from tqdm import tqdm
from functools import wraps

# Use your existing implementations
from .gridder import bin_data, bin_data_w_efficient
from .windows import (
    kaiser_bessel_window,
    casa_pswf_window,
    pillbox_window,
    sinc_window,
)

# -----------------------
# Low-level utilities
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


_ARCSEC_PER_RAD = 3600 * 180/np.pi

def make_uv_grid(
    uu: np.ndarray,
    vv: np.ndarray,
    npix: int,
    pad_uv: float,
    *,
    fov_arcsec: Optional[float] = None,
    warn_crop: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build symmetric square uv grid; truncation_radius == delta_u.

    Parameters
    ----------
    fov_arcsec : float, optional
        Image-plane field of view in arcseconds.
        If provided, uv cell size is set by delta_u = 1 / fov_rad, where
        fov_rad = fov_arcsec / 206265.

    Notes
    -----
    Assumes u,v are in wavelengths. Then:
      - image-plane angle is radians,
      - Fourier dual spacing satisfies FOV ≈ 1/delta_u.
    """
    if npix <= 0:
        raise ValueError(f"npix must be positive; got {npix}.")

    # Legacy mode: infer grid extent from data (with pad_uv), as before.
    if fov_arcsec is None:
        maxuv = max(np.abs(uu).max(), np.abs(vv).max())
        u_min = -maxuv * (1.0 + pad_uv)
        u_max = +maxuv * (1.0 + pad_uv)
        u_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
        v_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
        delta_u = float(u_edges[1] - u_edges[0])
        truncation_radius = delta_u
        return u_edges, v_edges, delta_u, truncation_radius

    # Explicit-FOV mode (arcsec -> rad)
    fov_arcsec = float(fov_arcsec)
    if not np.isfinite(fov_arcsec) or fov_arcsec <= 0.0:
        raise ValueError(f"fov_arcsec must be a positive finite float; got {fov_arcsec!r}.")

    fov_rad = fov_arcsec / _ARCSEC_PER_RAD  # radians

    # fov_rad and npix fully determine the uv grid in this mode.
    if pad_uv != 0.0:
        warnings.warn(
            "pad_uv is ignored when fov_arcsec is specified (because fov_arcsec and npix "
            "fully determine the uv grid). To change oversampling/resolution, adjust fov_arcsec and/or npix.",
            RuntimeWarning,
            stacklevel=2,
        )

    delta_u = 1.0 / fov_rad
    half_range = 0.5 * npix * delta_u  # uv half-extent

    u_min = -half_range
    u_max = +half_range
    u_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
    v_edges = np.linspace(u_min, u_max, npix + 1, dtype=float)
    truncation_radius = delta_u

    if warn_crop:
        maxuv_data = max(np.abs(uu).max(), np.abs(vv).max())
        if maxuv_data > half_range:
            outside = (np.abs(uu) > half_range) | (np.abs(vv) > half_range)
            frac = float(np.count_nonzero(outside)) / float(outside.size)
            warnings.warn(
                "Requested (fov_arcsec, npix) implies a uv grid smaller than the data extent:\n"
                f"  data max(|u|,|v|) = {maxuv_data:.6g} wavelengths\n"
                f"  grid half-range    = {half_range:.6g} wavelengths\n"
                f"  -> uv-space will be cropped; approx fraction outside grid: {frac:.3%}\n"
                "Consider increasing npix and/or decreasing fov_arcsec.",
                RuntimeWarning,
                stacklevel=2,
            )

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
    fov_arcsec: Optional[float] = None,          # NEW
    pad_uv: float = 0.0,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable[[ArrayLike, float], np.ndarray]] = None,
    p_metric: int = 1,
    std_p: int = 1,
    std_workers: int = 6,
    std_min_effective: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    u_edges, v_edges, delta_u, trunc_r = make_uv_grid(
        uu, vv, npix=npix, pad_uv=pad_uv, fov_arcsec=fov_arcsec, warn_crop=True
    )
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

    pbar = tqdm(range(F), unit="channel")
    for i in pbar:
        uv_tree, grid_tree, pairs = precompute_pairs(uu[i], vv[i], centers, trunc_r, p_metric=p_metric)

        vb_re = bin_data(
            uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="mean", verbose=0
        )
        vb_im = bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="mean", verbose=0
        )

        # std (Re/Im) + stats
        sb_re, stats_re = bin_data(
            uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_effective=std_min_effective, std_workers = std_workers, std_p = std_p,
            collect_stats=True
        )
        sb_im, stats_im = bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_effective=std_min_effective, std_workers = std_workers, std_p = std_p,
            collect_stats=True
        )

        cnt = bin_data(
            uu[i], vv[i], vis_re[i], weight[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="count", verbose=0
        )

        mean_re[i] = vb_re
        mean_im[i] = vb_im
        std_re[i]  = sb_re
        std_im[i]  = sb_im
        counts[i]  = cnt

        pbar.set_postfix(
            expansion_pix_re=stats_re,
            expansion_pix_im=stats_im
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
    fov_arcsec: Optional[float] = None,
    pad_uv: float = 0.0,
    w_bins: Union[int, np.ndarray] = 8,
    w_range: Optional[Tuple[float, float]] = None,
    w_abs: bool = False,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable[[ArrayLike, float], np.ndarray]] = None,
    p_metric: int = 1,
    # Std controls
    std_p: int = 1,
    std_workers: int = 6,
    std_min_effective: int = 5,
    std_expand_step: float = 0.1,
    std_expand_acceleration: float = 1.5,
    std_max_expand_cells: float = 5.0,   # replaces std_max_expand_iter
    tqdm_ncols: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid complex visibilities into UV pixels, also computing per-pixel uncertainty estimates,
    but with a staged fallback strategy for std:

      Pass A (per w-bin): std is computed WITHOUT expansion. Cells with insufficient effective
        samples are marked as NaN and recorded in a failure mask.

      Pass B (collapse W -> UV): compute UV-only std over ALL points in the channel (all w),
        again WITHOUT expansion, and use it to fill failed cells across all w-bins.

      Pass C (expand remaining): for cells still missing after Pass B, compute UV-only std WITH
        expansion enabled (incremental search radius), and fill remaining NaNs.

    """

    # -----------------------
    # Basic validation
    # -----------------------
    if uu.shape != vv.shape or uu.shape != ww.shape:
        raise ValueError(f"uu, vv, ww must have the same shape. Got uu={uu.shape}, vv={vv.shape}, ww={ww.shape}.")
    if vis_re.shape != uu.shape or vis_imag.shape != uu.shape or weight.shape != uu.shape:
        raise ValueError("vis_re, vis_imag, weight must match uu/vv/ww shape.")
    if uu.ndim < 2:
        raise ValueError("Expected uu/vv/ww to be shaped (F, ...). Got ndim < 2.")

    # -----------------------
    # UV grid + window binding
    # -----------------------
    u_edges, v_edges, delta_u, trunc_r = make_uv_grid(
        uu, vv, npix=npix, pad_uv=pad_uv, fov_arcsec=fov_arcsec, warn_crop=True
    )
    centers = build_grid_centers(u_edges, v_edges)

    w_edges = _make_w_edges(ww, w_bins, w_range=w_range, w_abs=w_abs)
    Nw = len(w_edges) - 1

    if window_fn is not None:
        window = _bind_window(window_fn, pixel_size=delta_u, window_kwargs=window_kwargs)
    else:
        if window_name is None:
            raise ValueError("Provide either window_name or a ready-made window_fn.")
        window = _window_from_name(window_name, pixel_size=delta_u, window_kwargs=window_kwargs)

    # Dimensions (kept as in your snippet)
    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    mean_re = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)

    # -----------------------
    # Main loop over channels
    # -----------------------
    pbar = tqdm(range(F), unit="channel", desc="Channels", ncols=tqdm_ncols)

    for i in pbar:
        # Flatten channel data
        u_all   = uu[i].ravel()
        v_all   = vv[i].ravel()
        wvals   = ww[i].ravel().astype(float)
        re_all  = vis_re[i].ravel()
        im_all  = vis_imag[i].ravel()
        wgt_all = weight[i].ravel()

        if w_abs:
            wvals = np.abs(wvals)

        # Assign each datum to a w-bin
        wbin = np.digitize(wvals, w_edges, right=False) - 1
        valid = (wbin >= 0) & (wbin < Nw)

        # Track which cells failed per w-bin (for later filling)
        fail_re_bins = np.zeros((Nw, Nu, Nv), dtype=bool)
        fail_im_bins = np.zeros((Nw, Nu, Nv), dtype=bool)

        wbar = tqdm(
            range(Nw),
            unit="wbin",
            desc=f"w-bins (ch {i+1}/{F})",
            leave=False,
            ncols=tqdm_ncols,
        )

        # -----------------------
        # Pass A: per w-bin gridding (std without expansion)
        # -----------------------
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

            uv_tree, grid_tree, pairs = precompute_pairs(
                u_b, v_b, centers, trunc_r, p_metric=p_metric
            )

            # mean
            vb_re = bin_data_w_efficient(
                u_b, v_b, re_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs, delta_u=delta_u,
                statistics_fn="mean",
                verbose=0
            )
            vb_im = bin_data_w_efficient(
                u_b, v_b, im_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs, delta_u=delta_u,
                statistics_fn="mean",
                verbose=0
            )

            # std: NO expansion, record failures as NaN + mask
            sb_re, fail_re = bin_data_w_efficient(
                u_b, v_b, re_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs, delta_u=delta_u,
                statistics_fn="std",
                verbose=0,
                std_min_effective=std_min_effective,
                std_workers=std_workers,
                std_p=std_p,
                std_expand_step=std_expand_step,     # unused in no-fallback but passed for API
                std_no_fallback=True,
                collect_fail_mask=True,
            )
            sb_im, fail_im = bin_data_w_efficient(
                u_b, v_b, im_b, wgt_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs, delta_u=delta_u,
                statistics_fn="std",
                verbose=0,
                std_min_effective=std_min_effective,
                std_workers=std_workers,
                std_p=std_p,
                std_expand_step=std_expand_step,
                std_no_fallback=True,
                collect_fail_mask=True,
            )

            # count
            cnt = bin_data_w_efficient(
                u_b, v_b, re_b, wgt_b, (u_edges, v_edges), 
                window, trunc_r, uv_tree, grid_tree, pairs, delta_u=delta_u,
                statistics_fn="count",
                verbose=0
            )

            mean_re[i, b] = vb_re
            mean_im[i, b] = vb_im
            std_re[i, b]  = sb_re
            std_im[i, b]  = sb_im
            counts[i, b]  = cnt

            fail_re_bins[b] = fail_re
            fail_im_bins[b] = fail_im

            wbar.set_postfix(
                fail_std_re=int(fail_re.sum()),
                fail_std_im=int(fail_im.sum()),
            )

        # -----------------------
        # Pass B/C: UV-only fallback (collapse W -> UV)
        # -----------------------
        any_fail_re = bool(fail_re_bins.any())
        any_fail_im = bool(fail_im_bins.any())

        if any_fail_re or any_fail_im:
            # Summarise what Pass A left behind before starting fallback passes
            total_fail_re = int(fail_re_bins.sum())
            total_fail_im = int(fail_im_bins.sum())
            tqdm.write(
                f"\n[Ch {i+1}/{F}] Pass A done. "
                f"Failures needing fallback — re: {total_fail_re} cells, "
                f"im: {total_fail_im} cells across {Nw} w-bins."
            )

            # Precompute UV mapping once for the full channel (all w)
            tqdm.write(f"[Ch {i+1}/{F}] Precomputing UV pairs for full channel (all w)...")
            uv_tree_all, grid_tree_all, pairs_all = precompute_pairs(
                u_all, v_all, centers, trunc_r, p_metric=p_metric
            )
            tqdm.write(f"[Ch {i+1}/{F}] UV pairs ready.")

            # ---- Pass B: UV-only std WITHOUT expansion ----
            std_uv_re_noexp = None
            std_uv_im_noexp = None

            tqdm.write(f"[Ch {i+1}/{F}] >>> Pass B: UV-only std (no expansion) <<<")

            if any_fail_re:
                tqdm.write(f"[Ch {i+1}/{F}]   Pass B — computing UV std (re) ...")
                std_uv_re_noexp, fail_uv_re = bin_data_w_efficient(
                    u_all, v_all, re_all, wgt_all, (u_edges, v_edges),
                    window, trunc_r, uv_tree_all, grid_tree_all, pairs_all, delta_u=delta_u,
                    statistics_fn="std",
                    verbose=0,
                    std_min_effective=std_min_effective,
                    std_workers=std_workers,
                    std_p=std_p,
                    std_expand_step=std_expand_step,
                    std_no_fallback=True,
                    collect_fail_mask=True,
                )
                # Fill all per-wbin failures using UV-only (collapsed-W) std
                filled_re_B = 0
                for b in range(Nw):
                    m = fail_re_bins[b]
                    if m.any():
                        # Only count cells where UV std actually provided a value
                        fillable = m & ~np.isnan(std_uv_re_noexp)
                        std_re[i, b][m] = std_uv_re_noexp[m]
                        filled_re_B += int(fillable.sum())
                still_nan_re = int(np.isnan(std_re[i]).sum())
                tqdm.write(
                    f"[Ch {i+1}/{F}]   Pass B (re) done. "
                    f"Filled {filled_re_B} cells. "
                    f"Still NaN after B: {still_nan_re}. "
                    f"UV-level failures in B: {int(fail_uv_re.sum())}."
                )

            if any_fail_im:
                tqdm.write(f"[Ch {i+1}/{F}]   Pass B — computing UV std (im) ...")
                std_uv_im_noexp, fail_uv_im = bin_data_w_efficient(
                    u_all, v_all, im_all, wgt_all, (u_edges, v_edges),
                    window, trunc_r, uv_tree_all, grid_tree_all, pairs_all, delta_u=delta_u,
                    statistics_fn="std",
                    verbose=0,
                    std_min_effective=std_min_effective,
                    std_workers=std_workers,
                    std_p=std_p,
                    std_expand_step=std_expand_step,
                    std_no_fallback=True,
                    collect_fail_mask=True,
                )
                filled_im_B = 0
                for b in range(Nw):
                    m = fail_im_bins[b]
                    if m.any():
                        fillable = m & ~np.isnan(std_uv_im_noexp)
                        std_im[i, b][m] = std_uv_im_noexp[m]
                        filled_im_B += int(fillable.sum())
                still_nan_im = int(np.isnan(std_im[i]).sum())
                tqdm.write(
                    f"[Ch {i+1}/{F}]   Pass B (im) done. "
                    f"Filled {filled_im_B} cells. "
                    f"Still NaN after B: {still_nan_im}. "
                    f"UV-level failures in B: {int(fail_uv_im.sum())}."
                )

            # ---- Pass C: expand ONLY remaining NaNs ----
            remain_re = np.isnan(std_re[i]).any(axis=0) if any_fail_re else None
            remain_im = np.isnan(std_im[i]).any(axis=0) if any_fail_im else None

            need_C_re = any_fail_re and remain_re is not None and remain_re.any()
            need_C_im = any_fail_im and remain_im is not None and remain_im.any()

            if need_C_re or need_C_im:
                tqdm.write(
                    f"[Ch {i+1}/{F}] >>> Pass C: UV-only std WITH expansion "
                    f"(re needs C: {need_C_re}, im needs C: {need_C_im}) <<<"
                )
            else:
                tqdm.write(f"[Ch {i+1}/{F}] Pass C not needed — no remaining NaNs after Pass B.")

            if need_C_re:
                tqdm.write(
                    f"[Ch {i+1}/{F}]   Pass C — computing expanded UV std (re) "
                    f"for {int(remain_re.sum())} UV cells still missing ..."
                )
                std_uv_re_exp, _ncoarse_re = bin_data_w_efficient(
                    u_all, v_all, re_all, wgt_all, (u_edges, v_edges),
                    window, trunc_r, uv_tree_all, grid_tree_all, pairs_all,
                    delta_u=delta_u,                               # NEW
                    statistics_fn="std",
                    verbose=0,
                    std_min_effective=std_min_effective,
                    std_workers=std_workers,
                    std_p=std_p,
                    std_expand_step=std_expand_step,
                    std_expand_acceleration=std_expand_acceleration,
                    std_max_expand_cells=std_max_expand_cells,     # NEW
                    std_no_fallback=False,
                    collect_stats=True,
                )
                filled_re_C = 0
                for b in range(Nw):
                    m = np.isnan(std_re[i, b])
                    if m.any():
                        fillable = m & ~np.isnan(std_uv_re_exp)
                        std_re[i, b][m] = std_uv_re_exp[m]
                        filled_re_C += int(fillable.sum())
                still_nan_re_C = int(np.isnan(std_re[i]).sum())
                tqdm.write(
                    f"[Ch {i+1}/{F}]   Pass C (re) done. "
                    f"Expanded cells (n_coarse): {_ncoarse_re}. "
                    f"Filled {filled_re_C} cells. "
                    f"Remaining NaN after C: {still_nan_re_C}."
                )

            if need_C_im:
                tqdm.write(
                    f"[Ch {i+1}/{F}]   Pass C — computing expanded UV std (im) "
                    f"for {int(remain_im.sum())} UV cells still missing ..."
                )
                std_uv_im_exp, _ncoarse_im = bin_data_w_efficient(
                    u_all, v_all, im_all, wgt_all, (u_edges, v_edges),
                    window, trunc_r, uv_tree_all, grid_tree_all, pairs_all,
                    delta_u=delta_u,                               # NEW
                    statistics_fn="std",
                    verbose=0,
                    std_min_effective=std_min_effective,
                    std_workers=std_workers,
                    std_p=std_p,
                    std_expand_step=std_expand_step,
                    std_expand_acceleration=std_expand_acceleration,
                    std_max_expand_cells=std_max_expand_cells,     # NEW
                    std_no_fallback=False,
                    collect_stats=True,
                )
                filled_im_C = 0
                for b in range(Nw):
                    m = np.isnan(std_im[i, b])
                    if m.any():
                        fillable = m & ~np.isnan(std_uv_im_exp)
                        std_im[i, b][m] = std_uv_im_exp[m]
                        filled_im_C += int(fillable.sum())
                still_nan_im_C = int(np.isnan(std_im[i]).sum())
                tqdm.write(
                    f"[Ch {i+1}/{F}]   Pass C (im) done. "
                    f"Expanded cells (n_coarse): {_ncoarse_im}. "
                    f"Filled {filled_im_C} cells. "
                    f"Remaining NaN after C: {still_nan_im_C}."
                )

            # Final per-channel summary
            final_nan_re = int(np.isnan(std_re[i]).sum())
            final_nan_im = int(np.isnan(std_im[i]).sum())
            tqdm.write(
                f"[Ch {i+1}/{F}] <<< Fallback complete. "
                f"Final NaN count — re: {final_nan_re}, im: {final_nan_im} >>>\n"
            )

        # Channel-level postfix
        pbar.set_postfix(w_bins=Nw)

    # Keep your final axis flip behavior unchanged
    return (
        np.flip(np.asarray(mean_re), axis=2),
        np.flip(np.asarray(mean_im), axis=2),
        np.flip(np.asarray(std_re),  axis=2),
        np.flip(np.asarray(std_im),  axis=2),
        np.flip(np.asarray(counts),  axis=2),
        u_edges, v_edges, w_edges
    )







