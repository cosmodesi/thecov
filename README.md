# thecov

Code for estimating theoretical covariance matrices of power spectrum multipoles in arbitrary geometries based on [CovaPT](https://github.com/JayWadekar/CovaPT/) by [Wadekar & Scoccimaro 2019](http://arxiv.org/abs/1910.02914). Uses tree-level perturbation theory to estimate the connected term and includes super-sample covariance (beat coupling + local averaging).

Under active development.

Status:

- Gaussian (box): ready.
- Gaussian (cut-sky): works, but being further validated.
- Regular trispectrum: interface being reworked.
- Super-sample: to be validated.

## Installation

```sh
pip install git+ssh://git@github.com/cosmodesi/thecov.git
```
## Usage examples

### Gaussian covariance in cubic box geometry

```python
import thecov.covariance
import thecov.geometry

geometry = thecov.geometry.BoxGeometry(volume=2000**3, nbar=1e-3)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(kmin=0, kmax=0.25, dk=0.005)

# Load input power spectra (P0,P2,P4) for the Gaussian covariance

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.compute_covariance()

# Covariance object has properties covariance.cov, covariance.cor, and
# functions like covariance.get_ell_cov(ell1, ell2) to output what you need.
```

### Gaussian covariance in cut sky geometry

```python

import thecov.covariance
import thecov.geometry

# nbodykit is required to handle random catalogs
from nbodykit.source.catalog import CSVCatalog
from nbodykit.lab import cosmology, transform

# Define cosmology used in nbodykit coordinate transformations
cosmo = cosmology.Cosmology(h=0.7).match(Omega0_m=0.31)

# Load random catalog using nbodykit
randoms = CSVCatalog(f'your_catalog.dat')

# Any catalog filtering/manipulations should go here

# Should define FKP weights column with this name
randoms['WEIGHT_FKP'] = 1./(1. + 1e4*randoms['NZ'])

# Convert sky coordinates to cartesian
randoms['Position'] = transform.SkyToCartesian(
    randoms['RA'], randoms['DEC'], randoms['Z'], degrees=True, cosmo=cosmo)

# Create geometry object to be used in covariance calculation
geometry = thecov.geometry.SurveyGeometry(random_catalog=randoms, Nmesh=31, BoxSize=3750, alpha=1/10)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(kmin=0, kmax=0.25, dk=0.005)

# Load input power spectra (P0,P2,P4) for the Gaussian covariance

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)


covariance.compute_covariance()

# Covariance object has functions covariance.cov, covariance.cor,
# covariance.get_ell_cov(ell1, ell2), etc. to output what you need.
```
