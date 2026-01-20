# VisCube

## Welcome to the documentation pages for VisCube!

VisCube is a visibility-space gridder for radio interferometry datasets. It has several powerful features:

1. Can handle both single-channel/continuum and multi-frequency/spectral cube datasets
2. Statistically robust uncertainty estimation in the UV plane (for details, see [here](https://viscube.readthedocs.io/en/latest/statistics.html)).
3. Built natively in Python using numpy

To learn how to use VisCube, I suggest following the documentation in the following order:

1. Run the ["Beginner's Guide"](https://viscube.readthedocs.io/en/latest/notebooks/example_basic.html) on your local machine to get acquainted with VisCube's functionality and make sure your installation is functional
2. Read through the ["How-to Guide: Extracting data from Measurement Sets"](https://viscube.readthedocs.io/en/latest/extracting_ms_to_py.html) to learn how to prepare your own measurement sets for VisCube
3. Once you have learnt one of the methods taught in the "extraction How-to Guides", you can run the code from the Beginner's guide on your own data (with modifications as needed). 

This project is a spinoff of a continuum-only gridder I worked on with Noé Dia and Alexandre Adam; that gridder has been released as part of the [IRIS interferometric imaging pipeline.](https://github.com/enceladecandy/iris)

The name VisCube is a nod to [VisRead](https://mpol-dev.github.io/visread/), a package documenting how to load data from Measurement Sets into memory. The CASA-based I/O procedures described in this documentation (see [here](https://viscube.readthedocs.io/en/latest/extracting_ms_to_py.html)) build on the VisRead tutorials, to which I owe a large chunk of my CASA knowledge. 

WARNING: DOCUMENTATION AND SOURCE CODE UNDER CONSTRUCTION. Docs are incomplete and breaking changes to the API may still occur (albeit rarely). 