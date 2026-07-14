# Statistical uncertainty estimation in VisCube

VisCube not only grids visibilities, but also can robustly estimate statistics (e.g., standard error) within each grid cell in the visibility plane. The way VisCube calculates visibility plane standard deviations is described below. To calculate the standard error, VisCube uses the following formula, which corrects for the degrees of freedom ($n_{eff}$):

$$\sigma = \frac{\sigma_{wt}}{\sqrt{n_{eff}}}$$

where $\sigma_{wt}$ is the weighted standard deviation:

$$\sigma_{wt} = \frac{\sum_{i=1}^N w_i (x_i - \overline{x})^2}{\sum_{i=1}^N w_i}$$

and $n_{eff}$ is the effective number of visibilities within the gridded cell:

$$n_{eff} = \frac{\left( \sum_{i=1}^N w_i \right)^2}{ \sum_{i=1}^N (w_i)^2}$$

The weights $w_i$ are defined by the gridding convolutional function's value at that (u,v) point multiplied by that visibility's weight as extracted from the measurement set. 

If a cell lacks sufficient numbers of visibilities to calculate an uncertainty, VisCube uses the neighboring visibilities from same [scan](https://casaguides.nrao.edu/index.php/Glossary_for_ALMA_Data_Processing) to estimate the uncertainty.

Note that due to Hermitian augmentation, each visibility shows up twice in the gridded cells. If you are using the gridded data as input to a Gaussian likelihood, the best way to account for this is to remove the duplicates by using the half-plane tools of VisCube.

References: 

[James Kirchner's pages](https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf)

Bevington, P. R., Data Reduction and Error Analysis for the Physical Sciences, 336 pp.,
McGraw-Hill, 1969.