"""
Image-plane apodization (taper) maps for gridding kernels.

Gridding convolves the visibilities with a kernel, which multiplies the
image plane by the kernel's Fourier transform (the taper). A forward model
that FFTs an image and reads it at uv cell centers must MULTIPLY its image
by the matching taper map before the FFT so that it is consistent with
data gridded as per-cell kernel-weighted means.

Two ways to build the taper are provided:

- :func:`make_apodization_1d` / :func:`make_apodization_map` — numeric:
  sample the kernel on integer uv-grid offsets and FFT it. Appropriate for
  wide kernels (support of several cells).
- :func:`make_kb_taper_1d` / :func:`make_kb_taper_map` — analytic: the
  exact Fourier transform of the Kaiser–Bessel kernel
  (:func:`viscube.windows.kb_kernel_1d`). Required for small kernels
  (e.g. ``m=1``, as used by
  :func:`viscube.grid_cube.grid_cube_all_stats_nonoverlap`), where integer
  sampling only hits offset 0 and the numeric route degenerates.

:func:`stabilized_inverse_map` reproduces the legacy divide-by-apodization
behavior only; multiplying by the taper is the corrected convention.
"""
import inspect
from functools import wraps
from typing import Callable, Optional

import numpy as np

from .windows import (
    kaiser_bessel_window,
    casa_pswf_window,
    pillbox_window,
    sinc_window,
)


def _bind_window(fn, pixel_size, window_kwargs):
    """
    Return a callable ``window(u, center)`` with kwargs safely bound.

    Only passes arguments that ``fn`` actually accepts, and always passes
    ``pixel_size`` if ``fn`` accepts it and it is not already provided.
    Local copy of :func:`viscube.grid_cube._bind_window` so this module
    stays import-independent.

    Parameters
    ----------
    fn : callable
        Base window function ``fn(u, center, **kwargs)``.
    pixel_size : float
        uv cell size to bind as the window's ``pixel_size``.
    window_kwargs : dict or None
        Extra keyword arguments to bind.

    Returns
    -------
    callable
        Two-argument window ``bound(u, center)``.
    """
    params = inspect.signature(fn).parameters
    kw = dict(window_kwargs or {})
    if "pixel_size" in params and "pixel_size" not in kw:
        kw["pixel_size"] = pixel_size

    @wraps(fn)
    def bound(u, c):
        return fn(u, c, **kw)

    return bound


def _window_from_name(
    name: str,
    *,
    pixel_size: float,
    window_kwargs: Optional[dict] = None,
):
    """
    Build a bound ``window(u, center)`` callable from a window name.

    Accepts the same names as the gridding API: ``"kb"/"kaiser"/
    "kaiser_bessel"/"kaiser-bessel"``, ``"pswf"/"casa"/"spheroidal"``,
    ``"pillbox"/"boxcar"``, ``"sinc"`` (case-insensitive).

    Parameters
    ----------
    name : str
        Window name.
    pixel_size : float
        uv cell size to bind as the window's ``pixel_size``.
    window_kwargs : dict, optional
        Extra keyword arguments; only those the chosen window accepts are
        forwarded.

    Returns
    -------
    callable
        Bound two-argument window.

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


def make_apodization_1d(
    *,
    npix: int,
    delta_u: float,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable] = None,
    normalize: str = "peak",
) -> np.ndarray:
    """
    Build the 1D image-plane apodization profile implied by a separable
    UV gridding kernel sampled on the FFT grid.

    The kernel is sampled at integer uv-grid offsets about the central
    cell and inverse-FFTed to the image plane. NOTE: for kernels whose
    support is a single cell (e.g. ``m=1``) this sampling only hits offset
    0 and returns a constant profile — use the analytic
    :func:`make_kb_taper_1d` instead.

    Parameters
    ----------
    npix : int
        FFT grid size.
    delta_u : float
        UV-cell size of the final gridded plane, in wavelengths.
    window_name : str, optional
        Name of the gridding window (same conventions as VisCube
        gridding); ignored if ``window_fn`` is given.
    window_kwargs : dict, optional
        Extra keyword arguments for the window (e.g. ``{"m": 6}``).
    window_fn : callable, optional
        Ready-made window ``fn(u, center, **kwargs)`` overriding
        ``window_name``.
    normalize : {"peak", "center", None}
        Normalization applied to the 1D profile: unit maximum ("peak"),
        unit central pixel ("center"), or none.

    Returns
    -------
    apo_1d : ndarray, shape (npix,)
        Real-valued, fftshifted image-plane apodization profile.

    Raises
    ------
    ValueError
        If ``npix`` or ``delta_u`` is non-positive, neither ``window_name``
        nor ``window_fn`` is provided, or ``normalize`` is invalid.
    """
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}.")
    if delta_u <= 0:
        raise ValueError(f"delta_u must be positive, got {delta_u}.")

    if window_fn is not None:
        window = _bind_window(window_fn, pixel_size=delta_u, window_kwargs=window_kwargs)
    else:
        if window_name is None:
            raise ValueError("Provide either window_name or window_fn.")
        window = _window_from_name(
            window_name,
            pixel_size=delta_u,
            window_kwargs=window_kwargs,
        )

    # Sample the 1D kernel on integer uv-grid offsets about the central cell.
    offsets = (np.arange(npix, dtype=float) - (npix // 2)) * float(delta_u)
    kern_1d = np.asarray(window(offsets, 0.0), dtype=float)

    # Enforce exact symmetry up to numerical precision.
    kern_1d = 0.5 * (kern_1d + kern_1d[::-1])

    apo_1d = np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(kern_1d),
            norm="backward",
        )
    )
    apo_1d = np.real_if_close(apo_1d, tol=1000).real

    if normalize == "peak":
        s = np.max(np.abs(apo_1d))
        if s > 0:
            apo_1d = apo_1d / s
    elif normalize == "center":
        s = apo_1d[npix // 2]
        if s != 0:
            apo_1d = apo_1d / s
    elif normalize is None:
        pass
    else:
        raise ValueError("normalize must be 'peak', 'center', or None.")

    return apo_1d


def make_apodization_map(
    *,
    npix: int,
    delta_u: float,
    window_name: Optional[str] = "kaiser_bessel",
    window_kwargs: Optional[dict] = None,
    window_fn: Optional[Callable] = None,
    normalize: str = "peak",
) -> np.ndarray:
    """
    Build the 2D separable image-plane apodization map.

    Outer product of the 1D profile from :func:`make_apodization_1d` with
    itself, re-normalized according to ``normalize``. See that function's
    caveat about single-cell kernels (use :func:`make_kb_taper_map` for
    ``m=1`` Kaiser–Bessel gridding).

    Parameters
    ----------
    npix : int
        FFT grid size.
    delta_u : float
        UV-cell size of the final gridded plane, in wavelengths.
    window_name : str, optional
        Name of the gridding window; ignored if ``window_fn`` is given.
    window_kwargs : dict, optional
        Extra keyword arguments for the window (e.g. ``{"m": 6}``).
    window_fn : callable, optional
        Ready-made window overriding ``window_name``.
    normalize : {"peak", "center", None}
        Normalization applied to the profile and to the final map.

    Returns
    -------
    apo_2d : ndarray, shape (npix, npix)
        Real-valued, fftshifted image-plane apodization map.
    """
    apo_1d = make_apodization_1d(
        npix=npix,
        delta_u=delta_u,
        window_name=window_name,
        window_kwargs=window_kwargs,
        window_fn=window_fn,
        normalize=normalize,
    )
    apo_2d = np.outer(apo_1d, apo_1d)

    if normalize == "peak":
        s = np.max(np.abs(apo_2d))
        if s > 0:
            apo_2d = apo_2d / s
    elif normalize == "center":
        s = apo_2d[npix // 2, npix // 2]
        if s != 0:
            apo_2d = apo_2d / s

    return apo_2d


def _kb_sinhc(z2: np.ndarray) -> np.ndarray:
    """
    Branch-safe evaluation of ``sinh(sqrt(z2)) / sqrt(z2)``.

    Analytically continued to ``sin(sqrt(-z2)) / sqrt(-z2)`` for
    ``z2 < 0``, with the ``z2 -> 0`` limit equal to 1. Branches are
    evaluated under masks (a naive ``np.where`` would evaluate sinh on the
    sin-branch arguments and overflow).

    Parameters
    ----------
    z2 : array-like
        Squared argument; may be negative (sin branch).

    Returns
    -------
    ndarray
        Function values, same shape as ``z2``.
    """
    z2 = np.asarray(z2, dtype=float)
    out = np.ones_like(z2)
    pos = z2 > 1e-24
    neg = z2 < -1e-24
    sp = np.sqrt(z2[pos])
    out[pos] = np.sinh(sp) / sp
    sn = np.sqrt(-z2[neg])
    out[neg] = np.sin(sn) / sn
    return out


def make_kb_taper_1d(
    *,
    npix: int,
    delta_u: float,
    m: int = 1,
    beta: float = 2.0,
    normalize: Optional[str] = "peak",
) -> np.ndarray:
    """
    ANALYTIC image-plane taper of the Kaiser–Bessel gridding kernel
    (:func:`viscube.windows.kb_kernel_1d` with the same m, beta), i.e. the
    exact Fourier transform of::

        k(u) = I0(beta * sqrt(1 - (2u/W)^2)) / I0(beta),  |u| <= W/2,
        W = m * delta_u:

        c(x) ~ sinh(sqrt(beta^2 - (pi W x)^2)) / sqrt(beta^2 - (pi W x)^2),

    continued to the sin branch when ``|pi W x| > beta``; ``beta = 0``
    gives exactly ``sinc(W x)`` (pillbox taper). Evaluated on the
    fftshifted image grid ``x_i = (i - npix//2) / (npix * delta_u)`` [rad].

    This is the map a forward model must MULTIPLY its image by (before the
    FFT) so that reading the FFT at cell centers matches data gridded as
    per-cell kernel-weighted means. It replaces `make_apodization_1d` for
    small kernels: that function samples the kernel at INTEGER delta_u
    offsets, which for m=1 hits only offset 0 and returns a constant map.

    Parameters
    ----------
    npix : int
        FFT grid size.
    delta_u : float
        UV-cell size of the gridded plane, in wavelengths.
    m : int
        Kaiser–Bessel kernel support in cells; must match the value used
        for gridding (``m=1`` for the non-overlap gridder default).
    beta : float
        Kaiser–Bessel shape parameter; must match the gridding value.
        ``beta = 0`` gives exactly a sinc taper (pillbox kernel).
    normalize : {"peak", "center", None}
        Normalization applied to the profile.

    Returns
    -------
    taper : ndarray, shape (npix,)
        Real-valued, fftshifted analytic image-plane taper profile.

    Raises
    ------
    ValueError
        If ``npix``/``delta_u`` is non-positive, ``beta < 0``, or
        ``normalize`` is invalid.
    """
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}.")
    if delta_u <= 0:
        raise ValueError(f"delta_u must be positive, got {delta_u}.")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}.")

    W = m * float(delta_u)
    x = (np.arange(npix, dtype=float) - (npix // 2)) / (npix * float(delta_u))
    z2 = beta ** 2 - (np.pi * W * x) ** 2
    taper = _kb_sinhc(z2)

    if normalize == "peak":
        s = np.max(np.abs(taper))
        if s > 0:
            taper = taper / s
    elif normalize == "center":
        s = taper[npix // 2]
        if s != 0:
            taper = taper / s
    elif normalize is None:
        pass
    else:
        raise ValueError("normalize must be 'peak', 'center', or None.")
    return taper


def make_kb_taper_map(
    *,
    npix: int,
    delta_u: float,
    m: int = 1,
    beta: float = 2.0,
    normalize: Optional[str] = "peak",
) -> np.ndarray:
    """
    2D separable analytic Kaiser–Bessel taper map.

    Outer product of :func:`make_kb_taper_1d` with itself, re-normalized
    according to ``normalize``. This is the map a forward model multiplies
    its image by (before the FFT) to match data gridded with
    :func:`viscube.grid_cube.grid_cube_all_stats_nonoverlap` using the
    same ``m`` and ``beta``.

    Parameters
    ----------
    npix : int
        FFT grid size.
    delta_u : float
        UV-cell size of the gridded plane, in wavelengths.
    m : int
        Kaiser–Bessel kernel support in cells (must match the gridding).
    beta : float
        Kaiser–Bessel shape parameter (must match the gridding).
    normalize : {"peak", "center", None}
        Normalization applied to the profile and to the final map.

    Returns
    -------
    taper_2d : ndarray, shape (npix, npix)
        Real-valued, fftshifted analytic image-plane taper map.
    """
    t1 = make_kb_taper_1d(npix=npix, delta_u=delta_u, m=m, beta=beta,
                          normalize=normalize)
    t2 = np.outer(t1, t1)
    if normalize == "peak":
        s = np.max(np.abs(t2))
        if s > 0:
            t2 = t2 / s
    elif normalize == "center":
        s = t2[npix // 2, npix // 2]
        if s != 0:
            t2 = t2 / s
    return t2


def stabilized_inverse_map(
    apo: np.ndarray,
    *,
    eps_fraction: float = 1e-3,
    clamp_max: Optional[float] = 1e3,
) -> np.ndarray:
    """
    Stabilized elementwise inverse of an apodization/taper map.

    Verbatim numpy port of the legacy logic that used to live inside
    ``VisibilityCubePadded.__init__`` (threshold at eps_fraction * max|apo|,
    zero below threshold, clamp to ±clamp_max). Kept ONLY so the legacy
    divide-by-apodization behavior remains reproducible (pass the result as
    ``image_taper_map``); the corrected convention is to MULTIPLY by the
    taper, not divide.

    Parameters
    ----------
    apo : ndarray
        Apodization/taper map to invert.
    eps_fraction : float
        Threshold fraction: cells with ``|apo| < eps_fraction * max|apo|``
        get an inverse of 0 instead of blowing up.
    clamp_max : float, optional
        Clip the inverse to ``[-clamp_max, clamp_max]``; None disables
        clamping.

    Returns
    -------
    ndarray
        Stabilized elementwise inverse, same shape as ``apo``.
    """
    apo = np.asarray(apo, dtype=float)
    thresh = float(eps_fraction) * np.max(np.abs(apo))
    safe = np.abs(apo) >= thresh
    inv = np.where(safe, 1.0 / np.where(safe, apo, 1.0), 0.0)
    if clamp_max is not None:
        inv = np.clip(inv, -float(clamp_max), float(clamp_max))
    return inv


def save_apodization_map(
    path: str,
    apodization_map: np.ndarray,
    *,
    npix: Optional[int] = None,
    delta_u: Optional[float] = None,
    window_name: Optional[str] = None,
    window_kwargs: Optional[dict] = None,
    normalize: Optional[str] = None,
):
    """
    Save an apodization map plus minimal metadata to a .npz file.

    Parameters
    ----------
    path : str
        Output .npz path.
    apodization_map : ndarray
        Map to save (any shape).
    npix : int, optional
        FFT grid size used to build the map (stored as -1 if None).
    delta_u : float, optional
        UV-cell size used to build the map (stored as NaN if None).
    window_name : str, optional
        Window name used to build the map (stored as "" if None).
    window_kwargs : dict, optional
        Window kwargs used to build the map (stored as ``repr`` string).
    normalize : str, optional
        Normalization mode used to build the map (stored as "" if None).
    """
    np.savez(
        path,
        apodization_map=np.asarray(apodization_map),
        npix=(-1 if npix is None else int(npix)),
        delta_u=(np.nan if delta_u is None else float(delta_u)),
        window_name=("" if window_name is None else str(window_name)),
        window_kwargs=("" if window_kwargs is None else repr(window_kwargs)),
        normalize=("" if normalize is None else str(normalize)),
    )


def load_apodization_map(path: str) -> np.ndarray:
    """
    Load only the apodization map from a file written by
    :func:`save_apodization_map`.

    Parameters
    ----------
    path : str
        Path to the .npz file.

    Returns
    -------
    ndarray
        The stored ``apodization_map``, as float64 (metadata is ignored).
    """
    with np.load(path, allow_pickle=False) as f:
        return np.asarray(f["apodization_map"], dtype=float)