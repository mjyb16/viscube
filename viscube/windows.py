import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, Callable, Sequence, Union
from numpy.typing import ArrayLike
from scipy.special import i0  # Modified Bessel I0
try:
    from scipy.special import pro_ang1
    _HAVE_PRO_ANG1 = True
except ImportError:
    _HAVE_PRO_ANG1 = False


# Defining some window functions. We could add more in the future but their effect needs to be taken into account in the forward model. 


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
    u : coordinates (array)
    center : float
        Grid-cell center coordinate (same units as u).
    pixel_size : float
        Grid pixel size in uv units (arcsec, wavelengths, etc).
    m : int
        *Total* support width in pixel units (i.e. covers m * pixel_size).
        Effective half-width = 0.5 * m * pixel_size.
    beta : float, optional
        Shape parameter. If None, a heuristic is used (oversampling≈2 assumed):
            beta ≈ π * sqrt( (m/2)^2 - 0.8 )
        (Beatty / Fessler style). Increase beta -> steeper taper.
    normalize : bool
        Normalize so that window(center) == 1.

    Returns
    -------
    w : ndarray (same shape as u)
    """
    half_width = 0.5 * m * pixel_size
    dist = np.abs(u - center)

    w = np.zeros_like(u, dtype=float)
    mask = dist <= half_width
    if not np.any(mask):
        return w

    if beta is None:
        # Simple default good for oversamp ~2
        beta = np.pi * np.sqrt((m / 2.0)**2 - 0.8)

    # Normalized distance in [-1,1]
    t = dist[mask] / half_width   # ∈ [0,1]
    # KB functional form
    arg = beta * np.sqrt(1 - t**2)
    denom = i0(beta)
    w_vals = i0(arg) / denom
    if normalize:
        # Already normalized to 1 at t=0; still, guard numerically.
        w_vals /= w_vals.max()
    w[mask] = w_vals
    return w


# ---- Schwab 1984 (CASA-style) spheroidal rational approximation coefficients ----
_SPH_A = np.array([0.01624782, -0.05350728, 0.1464354, -0.2347118,
                   0.2180684, -0.09858686, 0.01466325], dtype=float)  # a6..a0
_SPH_B1 = 0.2177793  # denominator linear coeff

def _spheroid_scalar(eta: float) -> float:
    """Scalar version (mirrors your original), returns large number if |eta|>1."""
    if abs(eta) > 1.0:
        return 0.0  # you had 1e30 then multiplied by (1-eta^2)-> negative huge; better just 0
    n = eta*eta - 1.0
    # Horner evaluation for numerator in descending powers of n (a6 n^6 + ... + a0)
    num = (((((( _SPH_A[0]*n + _SPH_A[1])*n + _SPH_A[2])*n + _SPH_A[3])*n + _SPH_A[4])*n + _SPH_A[5])*n + _SPH_A[6])
    den = _SPH_B1 * n + 1.0
    return num / den

def spheroid_vec(eta):
    """Vectorized spheroidal rational approximation (|eta|<=1)."""
    eta = np.asarray(eta, dtype=float)
    out = np.zeros_like(eta)
    mask = np.abs(eta) <= 1.0
    if np.any(mask):
        n = eta[mask]**2 - 1.0
        # Evaluate numerator via Horner
        a6,a5,a4,a3,a2,a1,a0 = _SPH_A
        num = ((((((a6*n + a5)*n + a4)*n + a3)*n + a2)*n + a1)*n + a0)
        den = _SPH_B1 * n + 1.0
        out[mask] = num / den
    return out

def casa_pswf_window(u,
                     center,
                     *,
                     pixel_size: float = 0.015,
                     m: int = 5,
                     normalize: bool = True):
    """
    CASA/Schwab prolate-spheroidal gridding kernel (separable 1-D form).
    Credit: Ryan Loomis via beams_and_weighting

    Parameters
    ----------
    u : array-like
        Coordinates (same units as `center`).
    center : float
        Grid point center coordinate.
    pixel_size : float
        Pixel spacing in `u` units.
    m : int
        Total *integer* support (number of pixels). Schwab kernel typically uses m=5.
    normalize : bool
        Normalize so that peak value (at center) is 1.

    Returns
    -------
    w : ndarray
        Window weights; zero outside half-width m/2 * pixel_size.
    """
    u = np.asarray(u, dtype=float)
    # Distance in pixels
    dpix = (u - center) / pixel_size
    half_width = m / 2.0  # in pixels
    # Dimensionless eta (|eta|<=1 inside support)
    eta = dpix / half_width
    w = (1.0 - eta**2) * spheroid_vec(eta)
    # Zero outside support
    w[np.abs(eta) > 1.0] = 0.0

    if normalize:
        # Peak at eta=0
        w0 = spheroid_vec(0.0)  # (1 - 0)*spheroid(0)
        if w0 != 0:
            w /= w0
    return w

def pillbox_window(u, center, pixel_size=0.015, m=1):
    """
    u: coordinate of the data points to be aggregated (u or v)
    center: coordinate of the center of the pixel considered. 
    pixel_size: Size of a pixel in the (u,v)-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    return np.where(np.abs(u - center) <= m * pixel_size / 2, 1, 0)


def sinc_window(u, center, pixel_size=0.015, m=1):
    """
    u: coordinate of the data points to be aggregated (u or v)
    center: coordinate of the center of the pixel considered. 
    pixel_size: Size of a pixel in the uv-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    return np.sinc(np.abs(u - center) / m / pixel_size)
