# oblate_lc: oblate transit lightcurves (for fun)

This package produces transit lightcurves for oblate exoplanets, specifically in the non-limb-darkened case. It does so by calculating the area of two overlapping ellipses using the procedure detailed in [Hughes & Chraibi 2014](https://link.springer.com/article/10.1007/s00791-013-0214-3).

This work was the precursor to [**squishyplanet**](https://github.com/ben-cassese/squishyplanet), which is a fully-flexible triaxial lightcurve model that can accomodate oblateness, tidal-locking, and arbitrary-order limb-darkening, among other features.

Many thanks to Avishi Poddar for coding up the original working version of this code.