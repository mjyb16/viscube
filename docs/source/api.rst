VisCube API
===========

The complete VisCube application programming interface, organized by
module. Everything documented here (except the low-level gridding engines
and window functions) is also re-exported at the package top level, so
``from viscube import grid_cube_all_stats_nonoverlap`` and
``viscube.grid_cube.grid_cube_all_stats_nonoverlap`` are equivalent.

High-level gridding: ``viscube.grid_cube``
------------------------------------------

.. automodule:: viscube.grid_cube
    :members:
    :member-order: bysource

Low-level gridding engines: ``viscube.gridder``
-----------------------------------------------

.. automodule:: viscube.gridder
    :members:
    :member-order: bysource

Gridding kernels: ``viscube.windows``
-------------------------------------

.. automodule:: viscube.windows
    :members:
    :member-order: bysource

Noise estimation: ``viscube.sigma_per_baseline``
------------------------------------------------

.. automodule:: viscube.sigma_per_baseline
    :members:
    :member-order: bysource

Image-plane tapers: ``viscube.deapodization``
---------------------------------------------

.. automodule:: viscube.deapodization
    :members:
    :member-order: bysource
