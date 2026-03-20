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

    return bound


def _window_from_name(
    name: str,
    *,
    pixel_size: float,
    window_kwargs: Optional[dict] = None,
):
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

    Parameters
    ----------
    npix : int
        FFT grid size.
    delta_u : float
        UV-cell size of the final gridded plane, in wavelengths.
    window_name / window_kwargs / window_fn
        Same conventions as VisCube gridding.
    normalize : {"peak", "center", None}
        Normalization applied to the 1D profile.

    Returns
    -------
    apo_1d : ndarray, shape (npix,)
        Real-valued, fftshifted image-plane apodization profile.
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

    Returns
    -------
    apo_2d : ndarray, shape (npix, npix)
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
    Save apodization map plus minimal metadata to a .npz file.
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
    Load only the apodization map from a .npz file created by save_apodization_map.
    """
    with np.load(path, allow_pickle=False) as f:
        return np.asarray(f["apodization_map"], dtype=float)