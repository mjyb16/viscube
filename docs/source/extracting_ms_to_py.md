# How-to Guide: Extracting data from Measurement Sets

One of the critical preprocessing steps one must take in order to use VisCube is extracting data into memory and saving it in numpy format. This is because radio interferometry data is stored in measurement sets, while VisCube expects numpy arrays as input. There are several ways to go about this process (in order, from oldest to most recent):

1. CASA routines run within the Python environment of Monolithic CASA
2. Modular CASA routines run in a Notebook/python script
3. XRADIO routines run in a Notebook/python script

Of these, option 1 is the least flexible, so I do not include instructions on how to do it. 

For option 2, see the [intro tutorial on extracting data with CASA](https://viscube.readthedocs.io/en/latest/notebooks/casa_io_basic.html) as well as the [tutorial on combining multiple observations with CASA](https://viscube.readthedocs.io/en/latest/notebooks/combining_low_high_res_casa.html). 

For an example of option 3, see the [intro tutorial on XRADIO with VisCube](https://viscube.readthedocs.io/en/latest/notebooks/xradio_uvw_continuum.html). 

**For all of these tutorials, I assume you have a calibrated measurement set lying around...if you do not, you can download a measurement set from the ALMA archive and follow the instructions on calibration/preprocessing in CASA using the ALMA pipeline and various CASA tasks (for spectral line data, you will probably want to learn about uvcontsub and mstransform).**