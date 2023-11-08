import thecov.covariance
import thecov.geometry
import thecov.utils
import thecov.base

from nbodykit.source.catalog import CSVCatalog
from nbodykit.lab import cosmology, transform

import numpy as np
import dask.array as da

import matplotlib
from matplotlib import pyplot as plot


cosmo = cosmology.Cosmology(h=0.7).match(Omega0_m=0.31)

# Let's use a 10x random catalog to make things faster for now

# From: https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x10.tar.gz
# Columns described in: http://skiesanduniverses.org/Products/MockCatalogues/SDSS/BOSSLRGDR12MDP/
randoms = CSVCatalog('/home/oalves/desi/analytical_covariance/survey_geometry/data/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x10.dat',
                     names=['RA', 'DEC', 'Z', 'NZ', 'BIAS', 'VETO FLAG', 'Weight'],
                     delim_whitespace=True)

randoms = randoms[randoms['VETO FLAG'] > 0]
randoms = randoms[(randoms['Z'] > 0.5) * (randoms['Z'] < 0.75)]

randoms['WEIGHT_FKP'] = 1./(1. + 1e4*randoms['NZ'])

randoms['Position'] = transform.SkyToCartesian(
    randoms['RA'], randoms['DEC'], randoms['Z'], degrees=True, cosmo=cosmo)

# gcov.compute_cartesian_ffts(tqdm=tqdm)

geometry = thecov.geometry.SurveyGeometry(random_catalog=randoms, Nmesh=31, BoxSize=3750, alpha=0.1, delta_k_max=3, kmodes_sampled=500)
# geometry.WinKernel = np.load('/home/oalves/desi/analytical_covariance/survey_geometry/data/Wij_k50_HighZ_NGC.npy')

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(0., 0.25, 0.005)

k  = np.loadtxt('/home/oalves/desi/analytical_covariance/survey_geometry/data/k_Patchy.dat')
P0 = np.loadtxt('/home/oalves/desi/analytical_covariance/survey_geometry/data/P0_fit_Patchy.dat')
P2 = np.loadtxt('/home/oalves/desi/analytical_covariance/survey_geometry/data/P2_fit_Patchy.dat')
P4 = np.loadtxt('/home/oalves/desi/analytical_covariance/survey_geometry/data/P4_fit_Patchy.dat')

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.alphabar = 0.01

covariance.compute_covariance()

covariance_mock = thecov.base.MultipoleCovariance.loadtxt('/home/oalves/desi/analytical_covariance/drive/Patchy_CovarianceMatrix.dat')

fig, _, _ = thecov.utils.plot_cov_diag([covariance_mock, covariance], label=['Mock', 'Analytic'])

thecov.utils.plot_cov(covariance, covariance_mock, label_a='Analytic', label_b='Mock')

thecov.utils.ridgeplot_cov([covariance.get_ell_cov(0,0), covariance_mock.get_ell_cov(0,0)], logplot=False, k=covariance.kmid, hspace=-0.4, nrange=5);