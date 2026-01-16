# VisCube

## Welcome to the documentation pages for VisCube!

VisCube is a visibility-space gridder for radio interferometry datasets. It has several powerful features:

1. Can handle both single-channel/continuum and multi-frequency/spectral cube datasets
2. Statistically robust uncertainty estimation in the UV plane (for details, see [here](https://viscube.readthedocs.io/en/latest/statistics.html)).
3. Built natively in Python using numpy

The name VisCube is a nod to [VisRead](https://mpol-dev.github.io/visread/), a package documenting how to load data from Measurement Sets into memory. The CASA-based I/O procedures described in this documentation (see [here](https://viscube.readthedocs.io/en/latest/extracting_ms_to_py.html)) build on the VisRead tutorials, to which I owe a large chunk of my CASA knowledge. 