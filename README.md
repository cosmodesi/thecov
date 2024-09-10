# thecov

Module with general tools to calculate theoretical covariance matrices of power spectrum multipoles in arbitrary geometries based on [CovaPT](https://github.com/JayWadekar/CovaPT/) ([Wadekar & Scoccimarro 2019](http://arxiv.org/abs/1910.02914)) and [PowerSpecCovFFT](https://github.com/archaeo-pteryx/PowerSpecCovFFT) ([Kobayashi 2023](https://arxiv.org/abs/2308.08593)). Tree-level perturbation theory is used to estimate the connected term, including super-sample covariance (beat coupling + local averaging terms).

Under active development, testing and validation. Version 1.0 will be released with the DESI 2024 power spectrum analytical covariance paper.

## Installation

```sh
pip install git+https://github.com/cosmodesi/thecov
```
## Usage

```python
import thecov.geometry
import thecov.covariance

# Libraries to handle basic cosmology and catalog manipulations
import mockfactory.Catalog
import mockfactory.utils
import cosmoprimo

# Define fiducial cosmology used in calculations
cosmo = cosmoprimo.fiducial.DESI()

# Load random catalog
randoms = mockfactory.Catalog.read(f'your_catalog.fits')

# Any catalog filtering/manipulations should go here

# Should define FKP weights column with this name
randoms['WEIGHT_FKP'] = 1./(1. + 1e4*randoms['NZ'])  # FKP weights are optional

# Convert sky coordinates to cartesian using fiducial cosmology
randoms['POSITION'] = mockfactory.utils.sky_to_cartesian(
                          cosmo.comoving_radial_distance(randoms['Z']),
                          randoms['RA'],
                          randoms['DEC'],
                          degree=True)

# Create geometry object to be used in covariance calculation
geometry = thecov.geometry.SurveyGeometry(
                            randoms,
                            kmax_window=0.02, # Nyquist wavelength of window FFTs
                            boxpad=2., # multiplies the box size inferred from catalog
                            alpha=0.1, # N_galaxies / N_randoms
                            kmodes_sampled=5000, # max N samples used in integ
                           )

kmin, kmax, dk = 0.0, 0.5, 0.005

gaussian = thecov.covariance.GaussianCovariance(geometry)
gaussian.set_kbins(kmin, kmax, dk)

# Load input power spectra (P0, P2, P4) for the Gaussian covariance

gaussian.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
gaussian.set_galaxy_pk_multipole(P2, 2)
gaussian.set_galaxy_pk_multipole(P4, 4)

gaussian.compute_covariance()

# Galaxy bias b1 and effective redshift zeff
b1, zeff = 2.0, 0.5

t0 = thecov.covariance.RegularTrispectrumCovariance(geometry)
t0.set_kbins(kmin, kmax, dk)

plin = cosmo.get_fourier()

t0.set_linear_matter_pk(np.vectorize(lambda k: plin.pk_kz(k, zeff)))

# Other bias parameters will be automatically determined
# assuming local lagrangian approximation if not given
t0.set_params(fgrowth=cosmo.growth_rate(zeff), b1=b1)
t0.compute_covariance()

# Creating a new geometry object with finer grid for SSC calcs.
# Larger boxpad yields smaller k-fundamental.
geometry_ssc = thecov.geometry.SurveyGeometry(randoms,
                                              kmax_window=0.1,
                                              boxpad=2.0,
                                              alpha=0.1)
ssc = thecov.covariance.SuperSampleCovariance(geometry_ssc)
ssc.set_kbins(kmin, kmax, dk)
ssc.set_linear_matter_pk(np.vectorize(lambda k: plin.pk_kz(k, zeff)))
ssc.set_params(fgrowth=cosmo.growth_rate(zeff), b1=b1);
ssc.compute_covariance()

covariance = gaussian + t0 + ssc

# Covariance object has functions covariance.cov, covariance.cor,
# covariance.get_ell_cov(ell1, ell2), etc. to output what you need.
```

### Cubic box geometry

For cubic box geometry, just use the `BoxGeometry` object instead:

```python
geometry = thecov.geometry.BoxGeometry(volume=2000**3, nbar=1e-3)
```

You'll be able to compute Gaussian + $T_0$ contributions with that object. Super-sample covariance is zero in such a geometry.

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
