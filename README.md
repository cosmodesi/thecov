# thecov

Code for estimating theoretical covariance matrices of power spectrum multipoles in arbitrary geometries based on [CovaPT](https://github.com/JayWadekar/CovaPT/) by [Wadekar & Scoccimaro 2019](http://arxiv.org/abs/1910.02914). Uses tree-level perturbation theory to estimate the connected term and includes super-sample covariance (beat coupling + local averaging).

Under active development.

Status:

- Gaussian (box): ready.
- Gaussian (cut-sky): ready.
- Regular trispectrum: interface being reworked.
- Super-sample: to be validated.

## Installation

```sh
pip install git+ssh://git@github.com/cosmodesi/thecov.git
```
## Usage examples

### Gaussian covariance in cubic box geometry

```python
from thecov import BoxGeometry, GaussianCovariance

geometry = BoxGeometry(volume=2000**3, nbar=1e-3)

covariance = GaussianCovariance(geometry)
covariance.set_kbins(kmin=0, kmax=0.25, dk=0.005)

# Load input power spectra (P0, P2, P4) for the Gaussian covariance

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.compute_covariance()

# Covariance object has properties covariance.cov, covariance.cor, and
# functions like covariance.get_ell_cov(ell1, ell2) to output what you need.
```

### Gaussian covariance in cut sky geometry

```python

from cosmoprimo.fiducial import DESI
from mockfactory import Catalog, utils

from thecov import SurveyGeometry, GaussianCovariance

# Define cosmology used in coordinate transformations
cosmo = DESI()

# Load random catalog
randoms = Catalog.read(f'your_catalog.fits')

# Any catalog filtering/manipulations should go here

# Should define FKP weights column with this name
randoms['WEIGHT_FKP'] = 1./(1. + 1e4*randoms['NZ'])  # FKP weights are optional

# Convert sky coordinates to cartesian
randoms['POSITION'] = utils.sky_to_cartesian(cosmo.comoving_radial_distance(randoms['Z']), randoms['RA'], randoms['DEC'], degree=Truee)

# Create geometry object to be used in covariance calculation
geometry = SurveyGeometry(randoms, nmesh=64, boxpad=1.2, alpha=1. / 10., kmodes_sampled=2000)

covariance = GaussianCovariance(geometry)
covariance.set_kbins(kmin=0, kmax=0.4, dk=0.005)

covariance.set_shotnoise(shotnoise) # optional but recommended

# Load input power spectra (P0, P2, P4) for the Gaussian covariance

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.compute_covariance()

# Covariance object has functions covariance.cov, covariance.cor,
# covariance.get_ell_cov(ell1, ell2), etc. to output what you need.
```
