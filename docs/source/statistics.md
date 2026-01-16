# Statistical uncertainty estimation in VisCube

(To be completed!)

VisCube not only grids visibilities, but also can robustly estimate statistics (e.g., standard deviation) within each grid in the visibility plane. The way VisCube calculates visibility plane standard deviations is described below. To calculate the standard deviation, VisCube uses the following formula:

$$\sigma = \frac{\sigma_{wt}}{\sqrt{n_{eff}}}$$

References: 

[James Kirchner's pages](https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf)

Bevington, P. R., Data Reduction and Error Analysis for the Physical Sciences, 336 pp.,
McGraw-Hill, 1969.