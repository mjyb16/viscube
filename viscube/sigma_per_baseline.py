import numpy as np
from typing import Tuple

def mad_std(x: np.ndarray, axis=None) -> np.ndarray:
    """
    Robust std estimate via MAD. For Gaussian: std ≈ 1.4826 * MAD.

    Parameters
    ----------
    x : ndarray
        Input array (can contain NaNs).
    axis : int or tuple of ints, optional
        Axis along which to compute the robust std.

    Returns
    -------
    std : ndarray
        Robust standard deviation estimate.
    """
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def sigma_by_baseline_scan_time_diff(
    data: np.ndarray,        # (nchan, nvis) complex
    mask: np.ndarray,        # (nchan, nvis) bool
    time_row: np.ndarray,    # (nvis,)
    scan_row: np.ndarray,    # (nvis,)
    ant1_row: np.ndarray,    # (nvis,)
    ant2_row: np.ndarray,    # (nvis,)
    *,
    min_pairs: int = 8,
    sigma_floor: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-visibility sigma separately for real and imaginary parts using
    time-differenced visibilities within groups defined by (scan, baseline).

    Sigma is computed per-channel per-group, then assigned to all visibilities
    in that group.

    Uses consecutive time differences:
        diff = x[t+1] - x[t]
    and converts diff-std to per-sample std via:
        sigma = std(diff) / sqrt(2)

    Parameters
    ----------
    data : ndarray, complex, shape (nchan, nvis)
        Complex visibilities.
    mask : ndarray, bool, shape (nchan, nvis)
        Valid-data mask.
    time_row : ndarray, shape (nvis,)
        Time stamps per visibility row.
    scan_row : ndarray, shape (nvis,)
        Scan number per visibility row.
    ant1_row, ant2_row : ndarray, shape (nvis,)
        Antenna IDs defining baselines.
    min_pairs : int, optional
        Minimum number of valid consecutive pairs required per channel/group.
    sigma_floor : float, optional
        Lower floor for sigma values.

    Returns
    -------
    sigma_re : ndarray, shape (nchan, nvis)
        Estimated per-visibility sigma for real part.
    sigma_im : ndarray, shape (nchan, nvis)
        Estimated per-visibility sigma for imaginary part.
    """
    nchan, nvis = data.shape
    sigma_re = np.full((nchan, nvis), np.nan, dtype=np.float64)
    sigma_im = np.full((nchan, nvis), np.nan, dtype=np.float64)

    # Baseline id (order-invariant)
    a1 = ant1_row.astype(np.int64)
    a2 = ant2_row.astype(np.int64)
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)

    # Pack baseline into one int (safe if antenna ids < 65536)
    baseline_id = (lo << 16) + hi

    # Group by (scan, baseline)
    key = (scan_row.astype(np.int64) << 32) + baseline_id
    uniq_keys, inv = np.unique(key, return_inverse=True)

    for g in range(uniq_keys.size):
        idx = np.where(inv == g)[0]
        if idx.size < (min_pairs + 1):
            continue

        # Sort by time within group
        t = time_row[idx]
        order = np.argsort(t)
        idx = idx[order]

        # Consecutive pairs
        i0 = idx[:-1]
        i1 = idx[1:]

        # Valid pairs for each channel
        valid_pairs = mask[:, i0] & mask[:, i1]
        if not np.any(valid_pairs):
            continue

        # Real and imag diffs, masked with NaNs
        diffs_re = np.where(valid_pairs, data[:, i1].real - data[:, i0].real, np.nan)
        diffs_im = np.where(valid_pairs, data[:, i1].imag - data[:, i0].imag, np.nan)

        # Robust std of diffs (per channel)
        std_diff_re = mad_std(diffs_re, axis=1)
        std_diff_im = mad_std(diffs_im, axis=1)

        # Convert diff-std -> per-sample sigma
        sigma_g_re = np.maximum(std_diff_re / np.sqrt(2.0), sigma_floor)
        sigma_g_im = np.maximum(std_diff_im / np.sqrt(2.0), sigma_floor)

        # Require enough valid pairs per channel
        n_pairs = np.sum(np.isfinite(diffs_re), axis=1)  # same as diffs_im validity
        good_chan = n_pairs >= min_pairs

        if np.any(good_chan):
            gc = np.where(good_chan)[0]
            sigma_re[gc[:, None], idx[None, :]] = sigma_g_re[gc][:, None]
            sigma_im[gc[:, None], idx[None, :]] = sigma_g_im[gc][:, None]

    # Fallback for remaining NaNs: per-channel median sigma
    for ch in range(nchan):
        # Real
        s_re = sigma_re[ch]
        med_re = np.nanmedian(s_re)
        if np.isfinite(med_re):
            sigma_re[ch, ~np.isfinite(s_re)] = med_re
        else:
            sigma_re[ch, ~np.isfinite(s_re)] = sigma_floor

        # Imag
        s_im = sigma_im[ch]
        med_im = np.nanmedian(s_im)
        if np.isfinite(med_im):
            sigma_im[ch, ~np.isfinite(s_im)] = med_im
        else:
            sigma_im[ch, ~np.isfinite(s_im)] = sigma_floor

    return sigma_re, sigma_im