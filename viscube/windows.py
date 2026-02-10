import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
from scipy.special import i0  # Modified Bessel I0
import sys

# ---- CASA/Schwab 1984 spheroidal rational approximation coefficients ----
_SPH_A = np.array([0.01624782, -0.05350728, 0.1464354, -0.2347118,
                   0.2180684, -0.09858686, 0.01466325], dtype=float)  # a6..a0
_SPH_B1 = 0.2177793  # denominator linear coeff


def _spheroid_vec(eta: np.ndarray) -> np.ndarray:
    """Vectorized prolate-spheroidal rational approximation (valid for |eta|<=1)."""
    out = np.zeros_like(eta, dtype=float)
    mask = np.abs(eta) <= 1.0
    if np.any(mask):
        n = eta[mask]**2 - 1.0
        a6, a5, a4, a3, a2, a1, a0 = _SPH_A
        num = ((((((a6*n + a5)*n + a4)*n + a3)*n + a2)*n + a1)*n + a0)
        den = _SPH_B1 * n + 1.0
        out[mask] = num / den
    return out


def kaiser_bessel_window(u: ArrayLike,
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
        Shape parameter. If None, use a reasonable default for oversamp≈2:
            beta ≈ π * sqrt( (m/2)^2 - 0.8 )
    normalize : bool
        If True, normalize so that window(center) == 1.

    Returns
    -------
    w : ndarray (same shape as u)
    """
    u = np.asarray(u, dtype=float)
    half_width = 0.5 * m * pixel_size
    dist = np.abs(u - center)

    w = np.zeros_like(u, dtype=float)
    mask = dist <= half_width
    if not np.any(mask):
        return w

    if beta is None:
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
        print("Error: negative weights")
        print(f"U values: {u}")
        print(f"Center values: {center}")
        print(f"Beta values: {beta}")
        print(f"W values: {w}")
        print(f"Denom values: {denom}")
        sys.exit()
    return w


def casa_pswf_window(u: ArrayLike,
                     center: float,
                     *,
                     pixel_size: float = 0.015,
                     m: int = 5,
                     normalize: bool = True) -> np.ndarray:
    """
    CASA/Schwab prolate-spheroidal gridding kernel (1-D separable form).

    Parameters
    ----------
    u : array-like
    center : float
    pixel_size : float
    m : int
        Total *integer* support (number of pixels). Typical choice: m=5.
    normalize : bool
        Normalize so that peak value at center is 1.

    Returns
    -------
    w : ndarray
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


def pillbox_window(u: ArrayLike, center: float, pixel_size: float = 0.015, m: int = 1):
    """Boxcar window with total width m * pixel_size."""
    u = np.asarray(u, dtype=float)
    half = 0.5 * m * pixel_size
    return (np.abs(u - center) <= half).astype(float)


def sinc_window(u: ArrayLike, center: float, pixel_size: float = 0.015, m: int = 1):
    """Sinc window truncated by m * pixel_size (soft truncation)."""
    u = np.asarray(u, dtype=float)
    # Use normalized sinc: np.sinc(x) = sin(pi x)/(pi x)
    return np.sinc((u - center) / (m * pixel_size))
