# Statistical uncertainty estimation in VisCube

(To be completed!)

VisCube not only grids visibilities, but also can robustly estimate statistics (e.g., standard error) within each grid cell in the visibility plane. The way VisCube calculates visibility plane standard deviations is described below. To calculate the standard error, VisCube uses the following formula, which corrects for the degrees of freedom ($$n_{eff}$$):

$$\sigma = \frac{\sigma_{wt}}{\sqrt{n_{eff}}}$$

where $$\sigma_{wt}$$ is the weighted standard deviation:

$$\sigma_{wt} = \frac{\sum_{i=1}^N w_i (x_i - \overline{x})^2}{\sum_{i=1}^N w_i}$$

and $$n_{eff}$$ is the effective number of visibilities within the gridded cell:

$$n_{eff} = \frac{\left( \sum_{i=1}^N w_i \right)^2}{ \sum_{i=1}^N (w_i)^2}$$

The weights $$w_i$$ are defined by the gridding convolutional function's value at that (u,v) point multiplied by that visibility's weight as extracted from the measurement set. 

Since $$\sigma$$ is technically a standard error calculated from the sample variance, the effective probability distribution distribution is not Gaussian, but rather [T-distributed](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Occurrence_and_applications). This means that if you use the uncertainties calculated by VisCube in a Gaussian likelihood, you must compensate for the increased density in the tails of the T distribution. A simple rule of thumb is to just multiply the VisCube standard errors by a factor of 2.


References: 

[James Kirchner's pages](https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf)

Bevington, P. R., Data Reduction and Error Analysis for the Physical Sciences, 336 pp.,
McGraw-Hill, 1969.