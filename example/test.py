
import numpy as np

from nbodykit.source.catalog import CSVCatalog
from nbodykit.lab import cosmology, transform

import sys, os

try:
    basedir = os.environ['BASEDIR']
except KeyError:
    basedir = '.'

sys.path.insert(0, f'{basedir}/src')

import thecov.covariance
import thecov.geometry
import thecov.base
import thecov.utils

cosmo = cosmology.Cosmology(h=0.7).match(Omega0_m=0.31)

randoms = CSVCatalog(f'{basedir}/example/data/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x10.dat',
                     names=['RA', 'DEC', 'Z', 'NZ', 'BIAS', 'VETO FLAG', 'Weight'],
                     delim_whitespace=True)

randoms = randoms[randoms['VETO FLAG'] > 0]
randoms = randoms[(randoms['Z'] > 0.5) * (randoms['Z'] < 0.75)]

randoms['WEIGHT_FKP'] = 1./(1. + 1e4*randoms['NZ'])

randoms['Position'] = transform.SkyToCartesian(
    randoms['RA'], randoms['DEC'], randoms['Z'], degrees=True, cosmo=cosmo)

geometry = thecov.geometry.SurveyGeometry(random_catalog=randoms, Nmesh=31, BoxSize=3750, alpha=1/10)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(0, 0.25, 0.005)

P0 = np.loadtxt(f'{basedir}/example/data/P0_fit_Patchy.dat')
P2 = np.loadtxt(f'{basedir}/example/data/P2_fit_Patchy.dat')
P4 = np.loadtxt(f'{basedir}/example/data/P4_fit_Patchy.dat')

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.compute_covariance()

geometry.save_window_kernels(f'{basedir}/example/data/window_kernels.npz')