"""
High-level spectral-cube gridding API and uv-grid utilities.

This is VisCube's main user-facing module. A typical pipeline is:

1. :func:`load_and_mask` — apply the per-channel validity mask and compact
   the visibility arrays.
2. :func:`hermitian_augment` — append the conjugate copies
   ``V(-u, -v) = conj(V(u, v))`` so the gridded plane is Hermitian.
3. :func:`grid_cube_all_stats` (kernel-overlap) or
   :func:`grid_cube_all_stats_nonoverlap` (one-cell-per-visibility) — grid
   every channel and estimate per-cell means, standard errors, and counts.
   :func:`grid_cube_all_stats_wbinned` additionally bins in w.

The remaining functions are building blocks (uv-grid construction, window
binding, axis-convention conversion) and Hermitian half-plane utilities
for likelihoods that keep only the non-redundant half of the uv plane.
"""
import warnings
from typing import Callable, Tuple, Sequence, Optional, Union
import numpy as np
from scipy.spatial import cKDTree
import inspect
from tqdm import tqdm
from functools import wraps

from .gridder import bin_data, bin_channel_nonoverlap
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
    Apply the per-channel validity mask and compact the visibility arrays.

    Keeps only the visibilities where ``mask`` is True in each channel and
    packs them into dense ``(F, Nmasked)`` arrays.

    Parameters
    ----------
    frequencies : ndarray, shape (F,)
        Channel frequencies in Hz. Passed through unchanged.
    uu, vv : ndarray, shape (F, Nvis)
        uv coordinates per channel, in wavelengths.
    vis : ndarray, shape (F, Nvis), complex
        Complex visibilities.
    weight : ndarray, shape (F, Nvis)
        Per-visibility measurement weights.
    sigma_re, sigma_im : ndarray, shape (F, Nvis)
        Per-visibility noise sigma of the real and imaginary parts (e.g.
        from :func:`viscube.sigma_per_baseline.sigma_by_baseline_scan_time_diff`).
    mask : ndarray, shape (F, Nvis), bool
        Validity mask; True marks visibilities to keep.

    Returns
    -------
    frequencies : ndarray, shape (F,)
        Unchanged channel frequencies.
    u0, v0 : ndarray, shape (F, Nmasked)
        Masked, compacted uv coordinates.
    vis0 : ndarray, shape (F, Nmasked), complex
        Masked, compacted visibilities.
    w0 : ndarray, shape (F, Nmasked)
        Masked, compacted weights.
    sigma_re0, sigma_im0 : ndarray, shape (F, Nmasked)
        Masked, compacted noise sigmas.

    Raises
    ------
    ValueError
        If the number of valid visibilities differs between channels
        (the compacted representation requires a constant count; use a
        ragged representation otherwise).
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
    Hermitian-augment the visibilities.

    For a real sky image the visibility function obeys
    ``V(-u, -v) = conj(V(u, v))``, so each measurement can be mirrored to
    the opposite uv point. This function concatenates, along the
    visibility axis::

        (u, v, Re, Im, w, sigma_re, sigma_im)
        -> concat with
        (-u, -v, +Re, -Im, w, sigma_re, sigma_im)

    doubling the number of samples per channel. Note the augmented copies
    are appended AFTER the originals; when gridding with
    :func:`grid_cube_all_stats_nonoverlap`, pass the pre-augmentation
    count as ``n_aug`` so DC-cell duplicates can be dropped.

    Parameters
    ----------
    u0, v0 : ndarray, shape (F, N)
        uv coordinates per channel, in wavelengths.
    vis0 : ndarray, shape (F, N), complex
        Complex visibilities.
    w0 : ndarray, shape (F, N)
        Per-visibility measurement weights.
    sigma_re0, sigma_im0 : ndarray, shape (F, N)
        Per-visibility noise sigma of the real and imaginary parts
        (unchanged under conjugation).

    Returns
    -------
    uu, vv : ndarray, shape (F, 2N)
        Augmented uv coordinates.
    vis_re, vis_imag : ndarray, shape (F, 2N)
        Augmented real and imaginary parts (imaginary part sign-flipped in
        the copies).
    w : ndarray, shape (F, 2N)
        Augmented weights.
    sigma_re_aug, sigma_im_aug : ndarray, shape (F, 2N)
        Augmented noise sigmas.
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
    Build a symmetric square uv grid; ``truncation_radius == delta_u``.

    Two modes:

    - **Legacy (``fov_arcsec=None``)**: the grid extent is inferred from
      the data, ``u_max = max(|u|, |v|) * (1 + pad_uv)``, and split into
      ``npix`` cells.
    - **Explicit FOV**: the uv cell size is fixed by the requested
      image-plane field of view, ``delta_u = 1 / fov_rad``, and the grid
      spans ``npix * delta_u``. ``pad_uv`` is ignored (with a warning),
      and a RuntimeWarning is raised if data fall outside the grid.

    Parameters
    ----------
    uu, vv : ndarray
        uv coordinates in wavelengths (any shape); used to infer the grid
        extent (legacy mode) or to check for cropping (explicit-FOV mode).
    npix : int
        Number of uv cells per axis.
    pad_uv : float
        Fractional padding of the data extent in legacy mode. Ignored when
        ``fov_arcsec`` is given.
    fov_arcsec : float, optional
        Image-plane field of view in arcseconds.
        If provided, uv cell size is set by delta_u = 1 / fov_rad, where
        fov_rad = fov_arcsec / 206265.
    warn_crop : bool
        In explicit-FOV mode, warn (RuntimeWarning) if any ``|u|`` or
        ``|v|`` exceeds the grid half-range, reporting the affected
        fraction.

    Returns
    -------
    u_edges, v_edges : ndarray, shape (npix+1,)
        Bin edges of the grid (identical for u and v).
    delta_u : float
        uv cell size in wavelengths.
    truncation_radius : float
        Kernel truncation radius for the overlap gridder (== ``delta_u``).

    Raises
    ------
    ValueError
        If ``npix <= 0`` or ``fov_arcsec`` is not a positive finite float.

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
    Compute the (u, v) center coordinates of every grid cell.

    Cells are enumerated with u as the outer (slowest) index and v as the
    inner index, matching the flattened cell order used by the gridding
    engines (``cell = iu * Nv + jv``).

    Parameters
    ----------
    u_edges, v_edges : ndarray, shapes (Nu+1,), (Nv+1,)
        Bin edges of the uv grid.

    Returns
    -------
    centers : ndarray, shape (Nu * Nv, 2)
        Cell-center coordinates ``(u, v)``, one row per cell.
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
    Build KD-trees and query kernel-support neighbor pairs for one channel.

    Parameters
    ----------
    uu_i, vv_i : ndarray
        uv coordinates of this channel's visibilities, in wavelengths.
    centers : ndarray, shape (Nu * Nv, 2)
        Grid-cell centers from :func:`build_grid_centers`.
    truncation_radius : float
        Neighbor-search radius (typically ``delta_u``).
    p_metric : int
        Minkowski p-norm for the ball query (1 = Manhattan, giving a
        square/diamond support; 2 = Euclidean, giving a circular support).

    Returns
    -------
    uv_tree : scipy.spatial.cKDTree
        Tree over the visibility uv points.
    grid_tree : scipy.spatial.cKDTree
        Tree over the grid-cell centers.
    pairs : list of list of int
        For each flattened grid cell, the indices of the visibilities
        within ``truncation_radius`` of its center.
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
    Convenience wrapper: grid one frequency channel with :func:`bin_data`.

    Computes the five standard per-cell statistics (mean/std of the real
    and imaginary parts, plus counts) for a single channel. No propagated
    low-info std fallback is available here (``invvar_group=None``), so
    low-info cells get ``std = NaN``; the full-cube drivers
    (:func:`grid_cube_all_stats` and friends) pass per-visibility inverse
    variances instead and should be preferred.

    Parameters
    ----------
    uu_i, vv_i : ndarray
        uv coordinates of this channel's visibilities, in wavelengths.
    vis_re_i, vis_imag_i : ndarray
        Real and imaginary parts of the visibilities.
    w_i : ndarray
        Per-visibility measurement weights.
    u_edges, v_edges : ndarray
        Bin edges of the uv grid.
    window_fn : callable
        Bound window ``window(u, center)`` (see :mod:`viscube.windows`).
    truncation_radius : float
        Kernel truncation radius used to build ``pairs``.
    uv_tree, grid_tree, pairs
        Precomputed neighbor geometry from :func:`precompute_pairs`.
    verbose_mean, verbose_std : int
        Passed to :func:`bin_data` as ``verbose`` (currently unused there).

    Returns
    -------
    vis_bin_re, std_bin_re, vis_bin_imag, std_bin_imag, counts : ndarray
        Per-cell statistics, each of shape (Nu, Nv).
    """
    bins = (u_edges, v_edges)
    params = (w_i, None, bins, window_fn, truncation_radius, uv_tree, grid_tree, pairs)

    vis_bin_re   = bin_data(uu_i, vv_i, vis_re_i, *params, statistics_fn="mean",  verbose=verbose_mean)
    std_bin_re   = bin_data(uu_i, vv_i, vis_re_i, *params, statistics_fn="std",   verbose=verbose_std)
    vis_bin_imag = bin_data(uu_i, vv_i, vis_imag_i, *params, statistics_fn="mean", verbose=verbose_mean)
    std_bin_imag = bin_data(uu_i, vv_i, vis_imag_i, *params, statistics_fn="std",  verbose=verbose_std)
    counts       = bin_data(uu_i, vv_i, vis_re_i,  *params, statistics_fn="count", verbose=verbose_mean)

    return vis_bin_re, std_bin_re, vis_bin_imag, std_bin_imag, counts

def uv_grid_to_fft_image_convention(arr_uv: np.ndarray) -> np.ndarray:
    """
    Convert a uv grid from [u, v] axis order to FFT/image [v, u] row/col order.

    The gridding engines index cells as ``(u, v)``; FFT-based imaging code
    expects row/col = ``(v, u)`` with the v axis flipped so that row 0 is
    at the top (north up). This swaps the last two axes and flips the new
    row axis. The result is elementwise aligned with a model's
    ``fftshift(fft2(ifftshift(image)))`` plane.

    Parameters
    ----------
    arr_uv : ndarray
        Array whose last two axes are ``(Nu, Nv)`` — a single 2D plane or
        a cube ``(F, Nu, Nv)``.

    Returns
    -------
    ndarray
        Array with last two axes ``(Nv, Nu)`` in image/FFT convention.

    Notes
    -----
    The v-axis flip is deliberate and load-bearing for the output
    orientation; do not remove it.
    """
    # swap last two axes: (..., u, v) -> (..., v, u)
    return np.flip(np.swapaxes(arr_uv, -2, -1), axis=-2)


# -----------------------
# User-facing helpers
# -----------------------

def _bind_window(fn, pixel_size, window_kwargs):
    """
    Return a callable ``window(u, center)`` with kwargs safely bound.

    Only passes arguments that ``fn`` actually accepts, and always passes
    ``pixel_size`` if ``fn`` accepts it and it is not already provided in
    ``window_kwargs``.

    Parameters
    ----------
    fn : callable
        Base window function ``fn(u, center, **kwargs)`` from
        :mod:`viscube.windows` (or user-supplied with the same signature).
    pixel_size : float
        uv cell size to bind as the window's ``pixel_size``.
    window_kwargs : dict or None
        Extra keyword arguments to bind (e.g. ``{"m": 6}``).

    Returns
    -------
    callable
        Two-argument window ``bound(u, center)``; the base function and
        bound kwargs are exposed as ``bound._window_base`` and
        ``bound._window_kwargs``.
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
    Build a bound ``window(u, center)`` callable from a window name.

    Parameters
    ----------
    name : str
        One of ``"kb"/"kaiser"/"kaiser_bessel"/"kaiser-bessel"``,
        ``"pswf"/"casa"/"spheroidal"``, ``"pillbox"/"boxcar"``, ``"sinc"``
        (case-insensitive).
    pixel_size : float
        uv cell size to bind as the window's ``pixel_size``.
    window_kwargs : dict, optional
        Extra keyword arguments; only those the chosen window accepts are
        forwarded.

    Returns
    -------
    callable
        Bound two-argument window (see :func:`_bind_window`).

    Raises
    ------
    ValueError
        If ``name`` is not a recognized window name.
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
    n_eff_mode: str = "both"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid a spectral cube of visibilities and estimate per-cell statistics.

    This is VisCube's main kernel-overlap gridder. For every channel it
    builds the uv grid, gathers each cell's neighboring visibilities
    (within ``truncation_radius = delta_u`` of the cell center), and
    computes weighted means, hybrid standard errors (empirical scatter
    with a propagated-variance fallback for low-information cells), and
    counts. Inputs are expected to be hermitian-augmented already (see
    :func:`hermitian_augment`).

    See :func:`grid_cube_all_stats_nonoverlap` for a much faster
    schema-identical alternative in which each visibility lands in exactly
    one cell.

    Parameters
    ----------
    frequencies : ndarray, shape (F,)
        Channel frequencies in Hz (not used in the gridding itself; kept
        for API symmetry with the extraction pipeline).
    uu, vv : ndarray, shape (F, N)
        uv coordinates per channel, in wavelengths.
    vis_re, vis_imag : ndarray, shape (F, N)
        Real and imaginary parts of the visibilities.
    weight : ndarray, shape (F, N)
        Per-visibility measurement weights, used for the means and the
        empirical std branch.
    invvar_group_re, invvar_group_im : ndarray, shape (F, N)
        Per-visibility inverse variances of the real/imaginary parts (e.g.
        ``1 / sigma**2`` with sigma from
        :func:`viscube.sigma_per_baseline.sigma_by_baseline_scan_time_diff`).
        Used ONLY in the low-info std fallback.
    npix : int
        Number of uv cells per axis.
    fov_arcsec : float, optional
        Image-plane field of view in arcseconds; if given, fixes
        ``delta_u = 1 / fov_rad`` (see :func:`make_uv_grid`).
    pad_uv : float
        Fractional padding of the data extent in legacy grid mode; ignored
        when ``fov_arcsec`` is given.
    window_name : str, optional
        Name of the gridding window (see :mod:`viscube.windows`); default
        ``"kaiser_bessel"``. Ignored if ``window_fn`` is given.
    window_kwargs : dict, optional
        Extra keyword arguments for the window (e.g. ``{"m": 6}``).
    window_fn : callable, optional
        Ready-made window ``fn(u, center, **kwargs)`` overriding
        ``window_name``.
    p_metric : int
        Minkowski p-norm for the neighbor search (1 = square support,
        2 = circular).
    std_p, std_workers : int
        Passed through to :func:`viscube.gridder.bin_data` (currently
        unused there; retained for backward compatibility).
    std_min_effective : int
        Cells with effective sample size below this threshold use the
        propagated invvar fallback for the std.
    n_eff_mode : {"geometric", "both"}
        Effective-sample-size definition (see
        :func:`viscube.gridder.bin_data`).

    Returns
    -------
    mean_re, mean_im : ndarray, shape (F, npix, npix)
        Gridded visibility means (real/imaginary), in FFT/image [v, u]
        convention (see :func:`uv_grid_to_fft_image_convention`).
    std_re, std_im : ndarray, shape (F, npix, npix)
        Per-cell standard errors of the means (NaN where not estimable).
    counts : ndarray, shape (F, npix, npix)
        Number of contributing visibilities per cell (useful for masking).
    u_edges, v_edges : ndarray, shape (npix+1,)
        Bin edges of the uv grid (in the original [u, v] convention).
    """
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
            collect_stats=True, n_eff_mode = n_eff_mode,
        )
        sb_im, stats_im = bin_data(
            uu[i], vv[i], vis_imag[i], weight[i], invvar_group_im[i], (u_edges, v_edges),
            window, trunc_r, uv_tree, grid_tree, pairs,
            statistics_fn="std", verbose=0,
            std_min_effective=std_min_effective,
            std_workers=std_workers, std_p=std_p,
            collect_stats=True, n_eff_mode = n_eff_mode,
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


def grid_cube_all_stats_nonoverlap(
    *,
    frequencies: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    vis_re: np.ndarray,
    vis_imag: np.ndarray,
    weight: np.ndarray,
    invvar_group_re: np.ndarray,
    invvar_group_im: np.ndarray,
    npix: int = 501,
    fov_arcsec: Optional[float] = None,
    pad_uv: float = 0.0,
    m: int = 1,
    beta: float = 2.0,
    std_min_effective: int = 5,
    n_eff_mode: str = "both",
    drop_dc_duplicates: bool = True,
    n_aug: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-overlapping drop-in alternative to :func:`grid_cube_all_stats`.

    Each visibility lands in exactly ONE uv cell (searchsorted binning)
    with within-bin weighting ``weight * KB(du)*KB(dv)`` (``m=1`` makes
    the KB support coincide exactly with the bin; ``beta=0`` is a
    pillbox). Output tuple and per-cell statistics are schema-identical to
    :func:`grid_cube_all_stats`, but the vectorized binning is orders of
    magnitude faster than the KDTree overlap path.

    A forward model comparing against data gridded this way should
    multiply its image by the matching analytic taper from
    :func:`viscube.deapodization.make_kb_taper_map` (same ``m``/``beta``)
    before the FFT.

    Parameters
    ----------
    frequencies : ndarray, shape (F,)
        Channel frequencies in Hz (not used in the gridding itself; kept
        for API symmetry).
    uu, vv : ndarray, shape (F, N)
        uv coordinates per channel, in wavelengths (hermitian-augmented;
        see :func:`hermitian_augment`).
    vis_re, vis_imag : ndarray, shape (F, N)
        Real and imaginary parts of the visibilities.
    weight : ndarray, shape (F, N)
        Per-visibility measurement weights.
    invvar_group_re, invvar_group_im : ndarray, shape (F, N)
        Per-visibility inverse variances of the real/imaginary parts, used
        ONLY in the low-info std fallback.
    npix : int
        Number of uv cells per axis.
    fov_arcsec : float, optional
        Image-plane field of view in arcseconds; if given, fixes
        ``delta_u = 1 / fov_rad`` (see :func:`make_uv_grid`).
    pad_uv : float
        Fractional padding of the data extent in legacy grid mode; ignored
        when ``fov_arcsec`` is given.
    m : int
        Kaiser–Bessel within-bin kernel support in cells (``m=1``
        recommended: kernel support == bin).
    beta : float
        Kaiser–Bessel shape parameter; ``beta = 0`` gives a pillbox.
    std_min_effective : int
        Cells with effective sample size below this threshold use the
        propagated invvar fallback for the std.
    n_eff_mode : {"geometric", "both"}
        Effective-sample-size definition (see
        :func:`viscube.gridder.bin_data`).
    drop_dc_duplicates : bool
        Drop hermitian-augmented copies that land in the DC cell (default
        True; requires ``n_aug``). Set False only if the input is NOT
        hermitian-augmented.
    n_aug : int, optional
        Number of ORIGINAL (pre-augmentation) visibilities per channel,
        i.e. the first ``n_aug`` entries are originals and the rest are
        conjugate copies. Without DC dedup, a short-baseline visibility
        and its conjugate both land in the DC cell: imag forced to ~0 and
        SE understated by sqrt(2).

    Returns
    -------
    mean_re, mean_im : ndarray, shape (F, npix, npix)
        Gridded visibility means (real/imaginary), in FFT/image [v, u]
        convention.
    std_re, std_im : ndarray, shape (F, npix, npix)
        Per-cell standard errors of the means (NaN where not estimable).
    counts : ndarray, shape (F, npix, npix)
        Number of contributing visibilities per cell.
    u_edges, v_edges : ndarray, shape (npix+1,)
        Bin edges of the uv grid (in the original [u, v] convention).

    Warns
    -----
    RuntimeWarning
        If ANY visibility falls outside the grid (silently dropped
        baselines were a long-standing silent failure mode); the dropped
        fraction and worst channel are reported.

    Raises
    ------
    ValueError
        If ``drop_dc_duplicates=True`` and ``n_aug`` is not given.
    """
    if drop_dc_duplicates and n_aug is None:
        raise ValueError(
            "drop_dc_duplicates=True requires n_aug (number of pre-augmentation "
            "visibilities per channel). Pass n_aug, or drop_dc_duplicates=False "
            "if the input is NOT hermitian-augmented."
        )

    u_edges, v_edges, delta_u, _ = make_uv_grid(
        uu, vv, npix=npix, pad_uv=pad_uv, fov_arcsec=fov_arcsec, warn_crop=True
    )

    F = uu.shape[0]
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1

    mean_re = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_re  = np.zeros((F, Nu, Nv), dtype=np.float64)
    mean_im = np.zeros((F, Nu, Nv), dtype=np.float64)
    std_im  = np.zeros((F, Nu, Nv), dtype=np.float64)
    counts  = np.zeros((F, Nu, Nv), dtype=np.float64)

    dc_from = int(n_aug) if drop_dc_duplicates else None
    tot_dropped = 0
    tot_dc_dropped = 0
    tot_fallback = 0
    n_vis_total = 0
    per_channel_dropped = []

    pbar = tqdm(range(F), unit="channel")
    for i in pbar:
        kw = dict(m=m, beta=beta, std_min_effective=std_min_effective,
                  n_eff_mode=n_eff_mode, dc_dedup_from=dc_from)
        mr, sr, cnt, st_re = bin_channel_nonoverlap(
            uu[i], vv[i], vis_re[i], weight[i], invvar_group_re[i],
            u_edges, v_edges, **kw)
        mi, si, _, st_im = bin_channel_nonoverlap(
            uu[i], vv[i], vis_imag[i], weight[i], invvar_group_im[i],
            u_edges, v_edges, **kw)

        mean_re[i], std_re[i], counts[i] = mr, sr, cnt
        mean_im[i], std_im[i] = mi, si

        n_vis_total += uu[i].size
        tot_dropped += st_re["n_dropped"]
        tot_dc_dropped += st_re["n_dc_dropped"]
        tot_fallback += st_re["n_fallback"] + st_im["n_fallback"]
        per_channel_dropped.append(st_re["n_dropped"])
        pbar.set_postfix(dropped=st_re["n_dropped"],
                         dc_dedup=st_re["n_dc_dropped"],
                         fallback_re=st_re["n_fallback"],
                         fallback_im=st_im["n_fallback"])

    if tot_dropped > 0:
        worst = int(np.argmax(per_channel_dropped))
        warnings.warn(
            f"grid_cube_all_stats_nonoverlap: {tot_dropped} of {n_vis_total} "
            f"visibilities ({tot_dropped / n_vis_total:.2%}) fall OUTSIDE the uv grid "
            f"and were DROPPED (worst channel {worst}: {per_channel_dropped[worst]}). "
            "The grid half-range is smaller than the longest baseline — increase npix "
            "and/or decrease fov_arcsec.",
            RuntimeWarning,
            stacklevel=2,
        )
    print(f"[grid_nonoverlap] dropped outside grid: {tot_dropped}/{n_vis_total} "
          f"({(tot_dropped / n_vis_total if n_vis_total else 0):.2%}); "
          f"DC-duplicate copies removed: {tot_dc_dropped}; "
          f"low-info std fallbacks (re+im): {tot_fallback}")

    return (
        uv_grid_to_fft_image_convention(mean_re),
        uv_grid_to_fft_image_convention(mean_im),
        uv_grid_to_fft_image_convention(std_re),
        uv_grid_to_fft_image_convention(std_im),
        uv_grid_to_fft_image_convention(counts),
        u_edges, v_edges
    )


# -----------------------
# Hermitian half-plane utilities
# -----------------------
#
# For a REAL image, the fftshifted uv plane of odd side npix (center
# c = npix//2) is Hermitian-symmetric under (row, col) -> (2c-row, 2c-col):
# every cell with row > c has its conjugate duplicate at row < c, and the
# DC row (row == c) maps onto itself with columns mirrored about col == c.
# The arrays returned by grid_cube_all_stats* (after
# uv_grid_to_fft_image_convention) are elementwise aligned with the model's
# fftshift(fft2(ifftshift(image))) plane, so the non-redundant half is the
# slab rows [c:], with the left half of its first row (the DC row) masked.


def half_plane_slab(arr_conv: np.ndarray, npix: int) -> np.ndarray:
    """
    Slice the non-redundant Hermitian half-plane slab out of a full plane.

    For a real image, the fftshifted uv plane of odd side ``npix`` is
    Hermitian-symmetric under ``(row, col) -> (2c-row, 2c-col)`` with
    ``c = npix // 2``: every cell with ``row > c`` has its conjugate
    duplicate at ``row < c``. The non-redundant half is therefore the slab
    of rows ``[c:]`` — with the caveat that the DC row (slab row 0) still
    mirrors onto itself; apply :func:`half_plane_mask_fix` to the
    corresponding mask slab to remove those duplicates.

    Parameters
    ----------
    arr_conv : ndarray
        Full-plane array in FFT/image convention, last two axes
        ``(npix, npix)`` (e.g. an output of :func:`grid_cube_all_stats`).
    npix : int
        Grid side length. Must be ODD (even npix has a Nyquist row that
        breaks the slab symmetry).

    Returns
    -------
    ndarray
        View of the slab rows ``[npix//2:]``, shape
        ``(..., (npix+1)//2, npix)``.

    Raises
    ------
    ValueError
        If ``npix`` is even or the last two axes are not (npix, npix).
    """
    npix = int(npix)
    if npix % 2 != 1:
        raise ValueError(f"half_plane_slab requires odd npix, got {npix}.")
    if arr_conv.shape[-2:] != (npix, npix):
        raise ValueError(
            f"expected last two axes ({npix}, {npix}), got {arr_conv.shape[-2:]}."
        )
    return arr_conv[..., npix // 2:, :]


def half_plane_mask_fix(mask_slab: np.ndarray, npix: int) -> np.ndarray:
    """
    Zero the conjugate-duplicate DC-row columns of a half-plane mask slab.

    Slab row 0 (see :func:`half_plane_slab`) is the self-mirroring DC row,
    whose columns ``[0:npix//2]`` duplicate columns ``[npix//2+1:]``. Using
    both halves of that row would double-count those cells in a
    half-plane likelihood.

    Parameters
    ----------
    mask_slab : ndarray
        Half-plane mask slab, last two axes ``((npix+1)//2, npix)``
        (boolean or numeric).
    npix : int
        Full-plane side length (odd).

    Returns
    -------
    ndarray
        Copy of ``mask_slab`` with columns ``[0:npix//2]`` of slab row 0
        zeroed (set to False for boolean masks). The DC cell itself
        (``slab[..., 0, npix//2]``) is KEPT.

    Raises
    ------
    ValueError
        If the last two axes are not ``((npix+1)//2, npix)``.
    """
    npix = int(npix)
    c = npix // 2
    if mask_slab.shape[-2:] != ((npix + 1) // 2, npix):
        raise ValueError(
            f"expected half-plane slab last axes ({(npix + 1) // 2}, {npix}), "
            f"got {mask_slab.shape[-2:]}."
        )
    out = mask_slab.copy()
    out[..., 0, :c] = False if out.dtype == bool else 0
    return out


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
    n_eff_mode: str = "both"
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Grid complex visibilities into w-binned uv pixels (UVW gridding).

    Like :func:`grid_cube_all_stats`, but each channel's visibilities are
    first partitioned into bins of the w coordinate and gridded per w-bin,
    yielding a ``(F, Nw, npix, npix)`` hypercube. Useful when the array is
    non-coplanar and the w term cannot be ignored.

    Parameters
    ----------
    frequencies : ndarray, shape (F,)
        Channel frequencies in Hz (not used in the gridding itself; kept
        for API symmetry).
    uu, vv, ww : ndarray, shape (F, N)
        uvw coordinates per channel, in wavelengths (hermitian-augmented;
        note w must flip sign along with u, v).
    vis_re, vis_imag : ndarray, shape (F, N)
        Real and imaginary parts of the visibilities.
    weight : ndarray, shape (F, N)
        Per-visibility measurement weights.
    invvar_group_re, invvar_group_im : ndarray, shape (F, N)
        Per-visibility inverse variances of the real/imaginary parts, used
        ONLY in the low-info std fallback.
    npix : int
        Number of uv cells per axis.
    fov_arcsec : float, optional
        Image-plane field of view in arcseconds; if given, fixes
        ``delta_u = 1 / fov_rad`` (see :func:`make_uv_grid`).
    pad_uv : float
        Fractional padding of the data extent in legacy grid mode; ignored
        when ``fov_arcsec`` is given.
    w_bins : int or ndarray
        Number of uniform w bins, or explicit bin edges (see
        :func:`_make_w_edges`).
    w_range : (float, float), optional
        Range for uniform w bins; defaults to the data min/max.
    w_abs : bool
        If True, bin ``|w|`` instead of w (often increases per-bin counts).
    window_name : str, optional
        Name of the gridding window (see :mod:`viscube.windows`). Ignored
        if ``window_fn`` is given.
    window_kwargs : dict, optional
        Extra keyword arguments for the window (e.g. ``{"m": 6}``).
    window_fn : callable, optional
        Ready-made window ``fn(u, center, **kwargs)`` overriding
        ``window_name``.
    p_metric : int
        Minkowski p-norm for the neighbor search.
    std_p, std_workers : int
        Passed through to :func:`viscube.gridder.bin_data` (currently
        unused there; retained for backward compatibility).
    std_min_effective : int
        Cells with effective sample size below this threshold use the
        propagated invvar fallback for the std.
    tqdm_ncols : int
        Width of the progress bars.
    n_eff_mode : {"geometric", "both"}
        Effective-sample-size definition (see
        :func:`viscube.gridder.bin_data`).

    Returns
    -------
    mean_re, mean_im : ndarray, shape (F, Nw, npix, npix)
        Gridded visibility means per w-bin, in FFT/image [v, u] convention.
    std_re, std_im : ndarray, shape (F, Nw, npix, npix)
        Per-cell standard errors (NaN where not estimable or w-bin empty).
    counts : ndarray, shape (F, Nw, npix, npix)
        Number of contributing visibilities per cell.
    u_edges, v_edges : ndarray, shape (npix+1,)
        Bin edges of the uv grid (in the original [u, v] convention).
    w_edges : ndarray, shape (Nw+1,)
        Bin edges of the w axis.

    Raises
    ------
    ValueError
        If the input array shapes are inconsistent.
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
                collect_stats=True, n_eff_mode = n_eff_mode,
            )
            sb_im, stats_im = bin_data(
                u_b, v_b, im_b, wgt_b, invv_im_b, (u_edges, v_edges),
                window, trunc_r, uv_tree, grid_tree, pairs,
                statistics_fn="std", verbose=0,
                std_min_effective=std_min_effective,
                std_workers=std_workers,
                std_p=std_p,
                collect_stats=True, n_eff_mode = n_eff_mode,
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







