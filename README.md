# thecov

Code for estimating theoretical covariance matrices of power spectrum multipoles based on [CovaPT](https://github.com/JayWadekar/CovaPT/) by [Wadekar & Scoccimaro 2019](http://arxiv.org/abs/1910.02914). Implements Gaussian + Trispectrum (from tree-level perturbation theory) + Super-sample (beat coupling + local averaging) terms in a cut sky.

Under active development.

To calculate a Gaussian covariance for cut-sky geometry:

```python

import thecov.covariance
import thecov.geometry
import thecov.base
import thecov.utils

# Load randoms and input power spectra (P0,P2,P4) for the Gaussian covariance

geometry = thecov.geometry.SurveyGeometry(random_catalog=randoms, Nmesh=31, BoxSize=3750, alpha=1/10)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(kmin=0, kmax=0.25, dk=0.005)

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.compute_covariance()
```
