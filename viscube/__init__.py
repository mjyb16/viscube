"""
VisCube: a visibility-space gridder for radio interferometry data.

Converts measurement-set visibilities (as numpy arrays) into regularly
gridded uv-plane spectral cubes with statistically robust per-cell
uncertainty estimates. Pure NumPy/SciPy.

The most commonly used entry points, all re-exported here:

- :func:`viscube.grid_cube.load_and_mask`,
  :func:`viscube.grid_cube.hermitian_augment` — prepare visibilities.
- :func:`viscube.sigma_per_baseline.sigma_by_baseline_scan_time_diff` —
  per-visibility noise estimation.
- :func:`viscube.grid_cube.grid_cube_all_stats`,
  :func:`viscube.grid_cube.grid_cube_all_stats_nonoverlap`,
  :func:`viscube.grid_cube.grid_cube_all_stats_wbinned` — gridding with
  uncertainty estimation.
- :mod:`viscube.deapodization` — image-plane taper maps for forward models.
"""
from . import __meta__

__version__ = __meta__.version

from .grid_cube import (
    grid_cube_all_stats,
    grid_cube_all_stats_wbinned,
    grid_cube_all_stats_nonoverlap,
    load_and_mask,
    hermitian_augment,
    make_uv_grid,
    build_grid_centers,
    half_plane_slab,
    half_plane_mask_fix,
)

from .gridder import bin_channel_nonoverlap

from .windows import kb_kernel_1d

from .sigma_per_baseline import sigma_by_baseline_scan_time_diff

from .deapodization import (
    make_apodization_1d,
    make_apodization_map,
    make_kb_taper_1d,
    make_kb_taper_map,
    stabilized_inverse_map,
    save_apodization_map,
    load_apodization_map,
)