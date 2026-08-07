# viscube — package notes for Claude

CPU-only (NumPy/SciPy) ALMA visibility gridding + robust noise estimation.
Editable install in the `latest_supermage` env; synced to DRAC cluster
virtualenvs alongside supermage.

## Which gridder to use
- **CURRENT:** `grid_cube_all_stats_nonoverlap` (m=1, β=2.0 Kaiser-Bessel;
  one cell per visibility — kernel support coincides exactly with the bin).
  Orders of magnitude faster than the KDTree overlap gridder, and what the
  2026-07 publication pipelines use. Forward models must **MULTIPLY** their
  image by `make_kb_taper_map` with the SAME m/β (analytic KB taper — numeric
  taper sampling degenerates at m=1).
- `grid_cube_all_stats` (m=6 KDTree overlap) and `grid_cube_all_stats_wbinned`
  are legacy — keep for reproducibility, don't use for new datasets.
- Half-plane workflow: `half_plane_slab` / `half_plane_mask_fix` store the
  non-redundant Hermitian slab; `hermitian_full_from_slab` /
  `real_full_from_slab` (added 2026-07-15, promoted from the sknkwx viz
  pipelines — which still carry their own private copies) rebuild the full
  plane from a saved slab for dirty imaging, machine-precision lossless.
  Legacy full-plane grids Hermitian-double-counted (σ understated up
  to ~√2–2×).
- The nonoverlap gridder raises a RuntimeWarning + prints a summary when
  visibilities fall outside the grid; pipeline notebooks assert **0% dropped**.
  A dc_dedup step removes duplicated DC-cell entries (short baselines).

## Conventions / gotchas
- The v-flip in `uv_grid_to_fft_image_convention` is LOAD-BEARING (north at
  row 0). Removing it broke the dirty image once already — do not touch.
- Output arrays are `(F, Nu, Nv)` — frequency axis FIRST.
- `delta_u` is set by the FOV alone (`delta_u = 1/fov_rad`); resolution is set
  by npix. Decision (NGC 4697, 2026-06-25): grid at ~3 px/beam
  (`less_oversample`); cropping below that under-resolves the BH sphere of
  influence and biases cuspy models. Model-side oversampling CANNOT compensate.
- `stabilized_inverse_map` (divide convention) is legacy-only — kept for
  reproducing old runs, never for new forward models.
- Noise: `sigma_by_baseline_scan_time_diff` supplies the `invvar_group_*`
  inputs required by the hybrid std estimator.
- viscube is CPU-only: move tensors off GPU before calling into it.

## Docs / tests
- Sphinx/jupyter-book docs; `docs/source/api.rst` documents the viscube
  modules (a caskade-template leftover was fixed 2026-07-04).
- **No test suite.** De-facto validation lives in the sknkwx gridding
  notebooks (`publication_run_nautilus/gridding/binning_scheme_tests.ipynb`
  etc.) and their in-notebook assertions.
