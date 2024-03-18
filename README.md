# thecov

Module with general tools to calculate theoretical covariance matrices of power spectrum multipoles in arbitrary geometries based on [CovaPT](https://github.com/JayWadekar/CovaPT/) ([Wadekar & Scoccimarro 2019](http://arxiv.org/abs/1910.02914)) and [PowerSpecCovFFT](https://github.com/archaeo-pteryx/PowerSpecCovFFT) ([Kobayashi 2023](https://arxiv.org/abs/2308.08593)). Tree-level perturbation theory is used to estimate the connected term, including super-sample covariance (beat coupling + local averaging terms).

Under active development, testing and validation. Version 1.0 will be released with the DESI 2024 power spectrum analytical covariance paper.

## Installation

```sh
pip install git+https://github.com/cosmodesi/thecov
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

### Gaussian covariance in sky geometry

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

## Citations

If you use this code in a scientific publication, don't forget to cite:

```
@unpublished{Alves2024prep,
  author = "Alves, Otavio and {DESI Collaboration}",
  title  = "Analytical covariance matrices of DESI galaxy power spectrum multipoles",
  note   = "(in prep.)",
  year   = "2024"
}

@article{Wadekar:2019rdu,
    author = "Wadekar, Digvijay and Scoccimarro, Roman",
    title = "{Galaxy power spectrum multipoles covariance in perturbation theory}",
    eprint = "1910.02914",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevD.102.123517",
    journal = "Phys. Rev. D",
    volume = "102",
    number = "12",
    pages = "123517",
    year = "2020"
}

@article{Kobayashi:2023vpu,
    author = "Kobayashi, Yosuke",
    title = "{Fast computation of the non-Gaussian covariance of redshift-space galaxy power spectrum multipoles}",
    eprint = "2308.08593",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevD.108.103512",
    journal = "Phys. Rev. D",
    volume = "108",
    number = "10",
    pages = "103512",
    year = "2023"
}
```
