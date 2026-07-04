"""
Gridding kernel (window) functions.

All windows here are 1D and separable: a 2D gridding kernel is built as
``w(u, v) = w_u(u) * w_v(v)``. Each window is evaluated at uv coordinates
``u`` relative to a grid-cell ``center`` and returns weights in ``[0, 1]``
that vanish outside the kernel support (except for the soft-truncated
sinc window).

The callables in this module follow the signature ``window(u, center,
**kwargs)`` expected by the high-level gridding functions in
:mod:`viscube.grid_cube`, which bind ``pixel_size`` and any extra keyword
arguments automatically (see ``window_name`` / ``window_kwargs`` there).
"""
import numpy as np
from typing import Optional
from scipy.special import i0  # Modified Bessel I0

# ---- CASA/Schwab 1984 spheroidal rational approximation coefficients ----
_SPH_A = np.array([0.01624782, -0.05350728, 0.1464354, -0.2347118,
                   0.2180684, -0.09858686, 0.01466325], dtype=float)  # a6..a0
_SPH_B1 = 0.2177793  # denominator linear coeff


def _spheroid_vec(eta: np.ndarray) -> np.ndarray:
    """
    Vectorized prolate-spheroidal rational approximation.

    Evaluates the Schwab (1984) rational-polynomial approximation of the
    zero-order prolate spheroidal wave function used by CASA's gridder.

    Parameters
    ----------
    eta : ndarray
        Normalized offset from the kernel center, ``eta = du / half_width``.
        The approximation is valid for ``|eta| <= 1``; entries outside that
        range evaluate to 0.

    Returns
    -------
    ndarray
        Spheroidal function values, same shape as ``eta``.
    """
    out = np.zeros_like(eta, dtype=float)
    mask = np.abs(eta) <= 1.0
    if np.any(mask):
        n = eta[mask]**2 - 1.0
        a6, a5, a4, a3, a2, a1, a0 = _SPH_A
        num = ((((((a6*n + a5)*n + a4)*n + a3)*n + a2)*n + a1)*n + a0)
        den = _SPH_B1 * n + 1.0
        out[mask] = num / den
    return out


def kaiser_bessel_window(u,
                         center: float,
                         *,
                         pixel_size: float = 0.015,
                         m: int = 6,
                         beta: Optional[float] = None,
                         normalize: bool = True) -> np.ndarray:
    """
    1D Kaiser–Bessel interpolation window (separable in u, v).

    Parameters
    ----------
    u : array-like
        Coordinates (same units as center).
    center : float
        Grid-cell center coordinate.
    pixel_size : float
        Grid pixel size in uv units.
    m : int
        *Total* support width in pixel units (covers m * pixel_size).
        Effective half-width = 0.5 * m * pixel_size.
    beta : float, optional
        Shape parameter. If None, use a reasonable default for oversamp≈2,
        ``beta = pi * sqrt((m/2)^2 - 0.8)``. ``beta = 0`` reduces to a
        pillbox.
    normalize : bool
        If True, normalize so that the maximum evaluated weight is 1.
        Note this normalizes by the max of the *evaluated batch*; for
        vectorized accumulation over many cells at once use
        :func:`kb_kernel_1d` instead, which needs no batch normalization.

    Returns
    -------
    w : ndarray
        Kernel weights, same shape as ``u``; zero outside the support
        ``|u - center| > 0.5 * m * pixel_size``.

    Raises
    ------
    ValueError
        If ``beta`` is None with ``m < ~1.8`` (default-beta formula invalid),
        or if the evaluation produces negative weights.
    """
    u = np.asarray(u, dtype=float)
    half_width = 0.5 * m * pixel_size
    dist = np.abs(u - center)

    w = np.zeros_like(u, dtype=float)
    mask = dist <= half_width
    if not np.any(mask):
        return w

    if beta is None:
        if (m / 2.0) ** 2 <= 0.8:
            raise ValueError(
                f"kaiser_bessel_window: the default beta formula "
                f"pi*sqrt((m/2)^2 - 0.8) is invalid for m={m} (m < ~1.8). "
                "Pass beta explicitly (beta=0 is a pillbox)."
            )
        beta = np.pi * np.sqrt((m / 2.0)**2 - 0.8)

    t = dist[mask] / half_width  # in [0,1]
    arg = beta * np.sqrt(1.0 - t**2)
    denom = i0(beta)
    w_vals = i0(arg) / denom
    if normalize and w_vals.size:
        w0 = w_vals.max()
        if w0 > 0:
            w_vals = w_vals / w0
    w[mask] = w_vals
    if np.any(w < 0):
        raise ValueError(
            "kaiser_bessel_window produced negative weights "
            f"(beta={beta}, center={center}, denom={denom})."
        )
    return w


def kb_kernel_1d(du,
                 *,
                 delta_u: float,
                 m: int = 1,
                 beta: float = 2.0) -> np.ndarray:
    """
    Analytic 1D Kaiser–Bessel gridding kernel evaluated at uv offsets ``du``
    from a cell center::

        k(du) = I0(beta * sqrt(1 - (du/half_width)^2)) / I0(beta),
        half_width = 0.5 * m * delta_u,  k = 0 outside |du| <= half_width.

    Peak value is exactly 1 at du = 0; ``beta = 0`` reduces exactly to a
    pillbox (boxcar) of total width ``m * delta_u``.

    Unlike ``kaiser_bessel_window(normalize=True)``, this applies NO
    batch normalization (that normalization divides by the max of the
    evaluated batch, which is wrong when accumulating many cells at once),
    so it is safe for vectorized binning. It is the kernel used by
    :func:`viscube.gridder.bin_channel_nonoverlap`, and its exact analytic
    image-plane taper is :func:`viscube.deapodization.make_kb_taper_1d`.

    Parameters
    ----------
    du : array-like
        uv-coordinate offsets from the cell center, in the same units as
        ``delta_u`` (wavelengths).
    delta_u : float
        uv cell size in wavelengths. Must be positive.
    m : int
        Total kernel support in cells; the kernel vanishes for
        ``|du| > 0.5 * m * delta_u``. ``m = 1`` confines the kernel to a
        single cell.
    beta : float
        Kaiser–Bessel shape parameter (>= 0). ``beta = 0`` gives a pillbox.

    Returns
    -------
    w : ndarray
        Kernel weights in ``[0, 1]``, same shape as ``du``.

    Raises
    ------
    ValueError
        If ``delta_u <= 0`` or ``beta < 0``.
    """
    if delta_u <= 0:
        raise ValueError(f"delta_u must be positive, got {delta_u}.")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}.")
    du = np.asarray(du, dtype=float)
    half_width = 0.5 * m * float(delta_u)
    t2 = (du / half_width) ** 2
    w = np.zeros_like(du, dtype=float)
    inside = t2 <= 1.0
    if beta == 0.0:
        w[inside] = 1.0
    else:
        w[inside] = i0(beta * np.sqrt(1.0 - t2[inside])) / i0(beta)
    return w


def casa_pswf_window(u,
                     center: float,
                     *,
                     pixel_size: float = 0.015,
                     m: int = 5,
                     normalize: bool = True) -> np.ndarray:
    """
    CASA/Schwab prolate-spheroidal gridding kernel (1-D separable form).

    Uses the Schwab (1984) rational approximation of the zero-order prolate
    spheroidal wave function, multiplied by the standard ``(1 - eta^2)``
    factor, matching CASA's default gridding kernel.

    Parameters
    ----------
    u : array-like
        Coordinates (same units as ``center``).
    center : float
        Grid-cell center coordinate.
    pixel_size : float
        Grid pixel size in uv units.
    m : int
        Total *integer* support (number of pixels). Typical choice: m=5.
    normalize : bool
        Normalize so that peak value at center is 1.

    Returns
    -------
    w : ndarray
        Kernel weights, same shape as ``u``; zero outside the support
        ``|u - center| > 0.5 * m * pixel_size``.
    """
    u = np.asarray(u, dtype=float)
    dpix = (u - center) / pixel_size        # distance in pixels
    half_width = m / 2.0                    # in pixels
    eta = dpix / half_width                 # maps support to |eta|<=1
    w = (1.0 - eta**2) * _spheroid_vec(eta)
    w[np.abs(eta) > 1.0] = 0.0

    if normalize:
        w0 = _spheroid_vec(np.array([0.0]))[0]
        if w0 != 0:
            w /= w0
    return w


def pillbox_window(u, center: float, pixel_size: float = 0.015, m: int = 1):
    """
    Boxcar (pillbox) window with total width ``m * pixel_size``.

    Parameters
    ----------
    u : array-like
        Coordinates (same units as ``center``).
    center : float
        Grid-cell center coordinate.
    pixel_size : float
        Grid pixel size in uv units.
    m : int
        Total support width in pixel units.

    Returns
    -------
    w : ndarray
        1.0 inside ``|u - center| <= 0.5 * m * pixel_size``, 0.0 outside.
    """
    u = np.asarray(u, dtype=float)
    half = 0.5 * m * pixel_size
    return (np.abs(u - center) <= half).astype(float)


def sinc_window(u, center: float, pixel_size: float = 0.015, m: int = 1):
    """
    Normalized sinc window, ``sinc((u - center) / (m * pixel_size))``.

    Uses NumPy's normalized sinc, ``sinc(x) = sin(pi x) / (pi x)``. Note the
    window is NOT hard-truncated here: weights beyond the first null are
    small but nonzero (soft truncation); the effective cutoff is imposed by
    the gridder's ``truncation_radius``.

    Parameters
    ----------
    u : array-like
        Coordinates (same units as ``center``).
    center : float
        Grid-cell center coordinate.
    pixel_size : float
        Grid pixel size in uv units.
    m : int
        Scale factor: the first null falls at ``|u - center| = m * pixel_size``.

    Returns
    -------
    w : ndarray
        Window values, same shape as ``u``.
    """
    u = np.asarray(u, dtype=float)
    # Use normalized sinc: np.sinc(x) = sin(pi x)/(pi x)
    return np.sinc((u - center) / (m * pixel_size))
