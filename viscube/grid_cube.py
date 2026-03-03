import warnings
from typing import Callable, Tuple, Sequence, Optional, Union
import numpy as np
from scipy.spatial import cKDTree
import inspect
from tqdm import tqdm
from functools import wraps

# Use your existing implementations
from .gridder import bin_data
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
    sigma_re: np.ndarray,
    sigma_im: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply per-channel mask and compact arrays.
    Returns frequencies, u0, v0, vis0, w0, sigma_re0, sigma_im0.

    Assumes the number of valid visibilities is the same for every channel
    (as in your current implementation). If not, this should be changed to ragged lists.
    """
    F = len(frequencies)
    Nmasked = int(mask[0].sum())

    u0 = np.zeros((F, Nmasked), dtype=np.float64)
    v0 = np.zeros((F, Nmasked), dtype=np.float64)
    vis0 = np.zeros((F, Nmasked), dtype=np.complex128)
    w0 = np.zeros((F, Nmasked), dtype=np.float64)
    s_re0 = np.zeros((F, Nmasked), dtype=np.float64)
    s_im0 = np.zeros((F, Nmasked), dtype=np.float64)

    for i in range(F):
        mi = mask[i]
        # Optional safety check:
        if int(mi.sum()) != Nmasked:
            raise ValueError(
                f"mask has variable valid count across channels; channel {i} has {int(mi.sum())}, "
                f"expected {Nmasked}. Use a ragged representation instead."
            )
        u0[i] = uu[i][mi]
        v0[i] = vv[i][mi]
        vis0[i] = vis[i][mi]
        w0[i] = weight[i][mi]
        s_re0[i] = sigma_re[i][mi]
        s_im0[i] = sigma_im[i][mi]

    return frequencies, u0, v0, vis0, w0, s_re0, s_im0


def hermitian_augment(
    u0: np.ndarray,
    v0: np.ndarray,
    vis0: np.ndarray,
    w0: np.ndarray,
    sigma_re0: np.ndarray,
    sigma_im0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hermitian augment:
      (u, v, Re, Im, w, sigma_re, sigma_im)
      -> concat with
      (-u, -v, +Re, -Im, w, sigma_re, sigma_im)

    Returns
    -------
    uu, vv, vis_re, vis_imag, w, sigma_re_aug, sigma_im_aug
    """
    uu = np.concatenate([u0, -u0], axis=1)
    vv = np.concatenate([v0, -v0], axis=1)

    vis_re = np.concatenate([vis0.real,  vis0.real], axis=1)
    vis_imag = np.concatenate([vis0.imag, -vis0.imag], axis=1)

    w = np.concatenate([w0, w0], axis=1)

    # Variance does not change under sign flip/conjugation
    sigma_re_aug = np.concatenate([sigma_re0, sigma_re0], axis=1)
    sigma_im_aug = np.concatenate([sigma_im0, sigma_im0], axis=1)

    return uu, vv, vis_re, vis_imag, w, sigma_re_aug, sigma_im_aug


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
    window_fn,
    truncation_radius,
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

def uv_grid_to_fft_image_convention(arr_uv: np.ndarray) -> np.ndarray:
    """
    Convert UV grid from [u, v] axis order to image/FFT-friendly [v, u] row/col order.
    Works for 2D or cubes with last two axes = (Nu, Nv).
    """
    # swap last two axes: (..., u, v) -> (..., v, u)
    #return np.swapaxes(arr_uv, -2, -1)
    return np.flip(np.swapaxes(arr_uv, -2, -1), axis=-2)


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
                      ):
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
    invvar_group_re: np.ndarray,   # NEW: same shape as vis_re
    invvar_group_im: np.ndarray,   # NEW: same shape as vis_imag
    npix: int = 501,
    fov_arcsec: Optional[float] = None,
    pad_uv: float = 0.0,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn = None,
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
        uv_tree, grid_tree, pairs = precompute_pairs(
            uu[i], vv[i], centers, trunc_r, p_metric=p_metric
        )

        # Means
        vb_re = bin_data(
            uu[i], vv[i], vis_re[i], weight[i], None, (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="mean", verbose=0
        )
        vb_im = bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], None, (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="mean", verbose=0
        )

        # Hybrid std (empirical normal pixels, propagated fallback on low-info)
        sb_re, stats_re = bin_data(
            uu[i], vv[i], vis_re[i], weight[i], invvar_group_re[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_effective=std_min_effective,
            std_workers=std_workers, std_p=std_p,
            collect_stats=True
        )
        sb_im, stats_im = bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], invvar_group_im[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_effective=std_min_effective,
            std_workers=std_workers, std_p=std_p,
            collect_stats=True
        )

        # Counts
        cnt = bin_data(
            uu[i], vv[i], vis_re[i], weight[i], None, (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="count", verbose=0
        )

        mean_re[i] = vb_re
        mean_im[i] = vb_im
        std_re[i]  = sb_re
        std_im[i]  = sb_im
        counts[i]  = cnt

        pbar.set_postfix(
            fallback_pix_re=stats_re,
            fallback_pix_im=stats_im,
        )

    return (
        uv_grid_to_fft_image_convention(np.asarray(mean_re)),
        uv_grid_to_fft_image_convention(np.asarray(mean_im)),
        uv_grid_to_fft_image_convention(np.asarray(std_re)),
        uv_grid_to_fft_image_convention(np.asarray(std_im)),
        uv_grid_to_fft_image_convention(np.asarray(counts)),
        u_edges, v_edges
    )


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
    invvar_group_re: np.ndarray,   # NEW: same shape as vis_re
    invvar_group_im: np.ndarray,   # NEW: same shape as vis_imag
    npix: int = 501,
    fov_arcsec: Optional[float] = None,
    pad_uv: float = 0.0,
    w_bins: Union[int, np.ndarray] = 8,
    w_range: Optional[Tuple[float, float]] = None,
    w_abs: bool = False,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable] = None,
    p_metric: int = 1,
    # Std controls (kept aligned with bin_data)
    std_p: int = 1,
    std_workers: int = 6,
    std_min_effective: int = 5,
    tqdm_ncols: int = 200,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Grid complex visibilities into UVW-binned UV pixels using `bin_data`.
    """

    # -----------------------
    # Basic validation
    # -----------------------
    if uu.shape != vv.shape or uu.shape != ww.shape:
        raise ValueError(
            f"uu, vv, ww must have the same shape. "
            f"Got uu={uu.shape}, vv={vv.shape}, ww={ww.shape}."
        )
    if vis_re.shape != uu.shape or vis_imag.shape != uu.shape or weight.shape != uu.shape:
        raise ValueError("vis_re, vis_imag, weight must match uu/vv/ww shape.")
    if invvar_group_re.shape != uu.shape or invvar_group_im.shape != uu.shape:
        raise ValueError("invvar_group_re and invvar_group_im must match uu/vv/ww shape.")
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

    # Dimensions
    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    mean_re = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    std_re  = np.full((F, Nw, Nu, Nv), np.nan, dtype=np.float64)
    mean_im = np.zeros((F, Nw, Nu, Nv), dtype=np.float64)
    std_im  = np.full((F, Nw, Nu, Nv), np.nan, dtype=np.float64)
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
        invv_re_all = invvar_group_re[i].ravel()
        invv_im_all = invvar_group_im[i].ravel()

        if w_abs:
            wvals = np.abs(wvals)

        # Assign each datum to a w-bin
        # right=False means bins are [edge_k, edge_{k+1})
        wbin = np.digitize(wvals, w_edges, right=False) - 1
        valid = (wbin >= 0) & (wbin < Nw)

        # Channel diagnostics
        ch_fallback_re = 0   # number of UV pixels that triggered low-info fallback (Re)
        ch_fallback_im = 0   # number of UV pixels that triggered low-info fallback (Im)
        ch_nan_re = 0        # NaN pixels remaining (Re)
        ch_nan_im = 0        # NaN pixels remaining (Im)

        wbar = tqdm(
            range(Nw),
            unit="wbin",
            desc=f"w-bins (ch {i+1}/{F})",
            leave=False,
            ncols=tqdm_ncols,
        )

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
            invv_re_b = invv_re_all[sel]
            invv_im_b = invv_im_all[sel]

            # Precompute geometry for this UVW subset
            uv_tree, grid_tree, pairs = precompute_pairs(
                u_b, v_b, centers, trunc_r, p_metric=p_metric
            )

            # -----------------------
            # Pass A: regular UVW gridding (means/counts unchanged)
            # -----------------------
            vb_re = bin_data(
                u_b, v_b, re_b, wgt_b, None, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="mean", verbose=0
            )
            vb_im = bin_data(
                u_b, v_b, im_b, wgt_b, None, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="mean", verbose=0
            )
            cnt = bin_data(
                u_b, v_b, re_b, wgt_b, None, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="count", verbose=0
            )

            sb_re, stats_re = bin_data(
                u_b, v_b, re_b, wgt_b, invv_re_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="std", verbose=0,
                std_min_effective=std_min_effective,
                std_workers=std_workers,
                std_p=std_p,
                collect_stats=True,
            )
            sb_im, stats_im = bin_data(
                u_b, v_b, im_b, wgt_b, invv_im_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="std", verbose=0,
                std_min_effective=std_min_effective,
                std_workers=std_workers,
                std_p=std_p,
                collect_stats=True,
            )

            # Store
            mean_re[i, b] = vb_re
            mean_im[i, b] = vb_im
            std_re[i, b]  = sb_re
            std_im[i, b]  = sb_im
            counts[i, b]  = cnt

            # Diagnostics
            fallback_re_bin = int(stats_re)  # n_fallback returned by bin_data
            fallback_im_bin = int(stats_im)
            
            nan_re_bin = int(np.isnan(sb_re).sum())
            nan_im_bin = int(np.isnan(sb_im).sum())
            
            ch_fallback_re += fallback_re_bin
            ch_fallback_im += fallback_im_bin
            ch_nan_re += nan_re_bin
            ch_nan_im += nan_im_bin
            
            wbar.set_postfix(
                fallback_re=fallback_re_bin,
                fallback_im=fallback_im_bin,
                nan_re=nan_re_bin,
                nan_im=nan_im_bin,
            )

        pbar.set_postfix(
            w_bins=Nw,
            fallback_re=ch_fallback_re,
            fallback_im=ch_fallback_im,
            nan_re=ch_nan_re,
            nan_im=ch_nan_im,
        )

    # Keep final axis-flip behavior unchanged
    return (
        uv_grid_to_fft_image_convention(np.asarray(mean_re)),
        uv_grid_to_fft_image_convention(np.asarray(mean_im)),
        uv_grid_to_fft_image_convention(np.asarray(std_re)),
        uv_grid_to_fft_image_convention(np.asarray(std_im)),
        uv_grid_to_fft_image_convention(np.asarray(counts)),
        u_edges, v_edges, w_edges,
    )







