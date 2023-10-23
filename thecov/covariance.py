"""Module for computing the covariance matrix of power spectrum multipoles.

This module currently contains only the GaussianCovariance class, which is used
to compute the Gaussian covariance matrix of power spectrum multipoles in a given geometry.

Example
-------
>>> from thecov import BoxGeometry, GaussianCovariance
>>> geometry = SurveyGeometry(random_catalog=randoms, alpha=1. / 10)
>>> covariance = GaussianCovariance(geometry)
>>> covariance.set_kbins(kmin=0, kmax=0.25, dk=0.005)
>>> covariance.set_pk(P0, 0, has_shotnoise=False)
>>> covariance.set_pk(P2, 2)
>>> covariance.set_pk(P4, 4)
>>> covariance.compute_covariance()
"""

import logging
import warnings
import itertools as itt

import numpy as np

from . import base, geometry

__all__ = ['GaussianCovariance']

class GaussianCovariance(base.MultipoleFourierCovariance):
    '''Gaussian covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    '''
    logger = logging.getLogger('GaussianCovariance')

    def __init__(self, geometry):
        base.MultipoleFourierCovariance.__init__(self)

        self.geometry = geometry

        self._pk = {}
        self.alphabar = None
        # alphabar is used to scale the shotnoise contributions to the covariance with (1 + alphabar) factors
        if hasattr(geometry, 'alpha'):
            self.alphabar = geometry.alpha

    @property
    def shotnoise(self):
        '''Shotnoise of the sample.

        Returns
        -------
        float
            Shotnoise value.'''
        
        if isinstance(geometry, geometry.SurveyGeometry):
            return (1 + self.alphabar) * self.geometry.I('12')/self.geometry.I('22')
        else:
            return self.geometry.shotnoise

    def set_shotnoise(self, shotnoise):
        '''Set shotnoise to specified value. Also scales alphabar so that (1 + alphabar)*I12/I22
        matches the specified shotnoise.

        Parameters
        ----------
        shotnoise : float
            shotnoise = (1 + alpha)*I12/I22.
        '''

        self.logger.info(f'Estimated shotnoise was {self.geometry.shotnoise}')
        self.logger.info(f'Setting shotnoise to {shotnoise}.')
        # self.geometry.shotnoise = shotnoise
        self.alphabar = shotnoise * self.geometry.I('22') / self.geometry.I('12') - 1
        self.logger.info(f'Setting alphabar to {self.alphabar} based on given shotnoise value.')

    def set_pk(self, pk, ell, has_shotnoise=False):
        '''Set the input power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        pk : array_like
            Power spectrum.
        ell : int
            Multipole of the power spectrum.
        has_shotnoise : bool, optional
            Whether the power spectrum has shotnoise included or not.
        '''

        if ell == 0 and has_shotnoise:
            self.logger.info(f'Removing shotnoise = {self.shotnoise} from ell = 0.')
            self._pk[ell] = pk - self.shotnoise
        else:
            self._pk[ell] = pk

    def get_pk(self, ell, force_return=False, remove_shotnoise=True):
        '''Get the input power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        ell : int
            Multipole of the power spectrum.
        force_return : bool, optional
            Whether to return a zero array if the power spectrum is not set.
        remove_shotnoise : bool, optional
            Whether to remove the shotnoise from the power spectrum monopole.
        '''

        if ell in self._pk.keys():
            pk = self._pk[ell]
            if (not remove_shotnoise) and ell == 0:
                self.logger.info(f'Adding shotnoise = {self.shotnoise} to ell = 0.')
                return pk + self.shotnoise
            return pk
        elif force_return:
            return np.zeros(self.kbins)

    def compute_covariance(self, ells=(0, 2, 4)):
        '''Compute the covariance matrix for the given geometry and power spectra.

        Parameters
        ----------
        ells : tuple, optional
            Multipoles of the power spectra to have the covariance calculated for.
        '''

        self._ells = ells
        self._mshape = (self.kbins, self.kbins)

        if isinstance(self.geometry, geometry.BoxGeometry):
            return self._compute_covariance_box()

        if isinstance(self.geometry, geometry.SurveyGeometry):
            return self._compute_covariance_survey()

    def _compute_covariance_box(self):
        '''Compute the covariance matrix for a box geometry.

        Returns
        -------
        self : GaussianCovariance
            Covariance matrix.
        '''

        # If the power spectrum for a given ell is not set, use a zero array instead
        P0 = self.get_pk(0, force_return=True, remove_shotnoise=False)
        P2 = self.get_pk(2, force_return=True)
        P4 = self.get_pk(4, force_return=True)

        cov = {}

        cov[0, 0] = P0**2 + 1/5*P2**2 + 1/9*P4**2
        cov[0, 2] = 2*P0*P2 + 2/7*P2 ** 2 + 4/7*P2*P4 + 100/693*P4**2
        cov[0, 4] = 2*P0*P4 + 18/35*P2**2 + 40/77*P2*P4 + 162/1001*P4**2
        cov[2, 2] = 5*P0**2 + 20/7*P0*P2 + 20/7*P0*P4 + \
            15/7*P2**2 + 120/77*P2*P4 + 8945/9009*P4**2
        cov[2, 4] = 36/7*P0*P2 + 200/77*P0*P4 + 108 / \
            77*P2**2 + 3578/1001*P2*P4 + 900/1001*P4**2
        cov[4, 4] = 9*P0**2 + 360/77*P0*P2 + 2916/1001*P0*P4 + \
            16101/5005*P2**2 + 3240/1001*P2*P4 + 42849/17017*P4**2

        for l1, l2 in itt.combinations_with_replacement(self.ells, r=2):
            self.set_ell_cov(l1, l2, 2/self.nmodes * np.diag(cov[l1, l2]))

        if (self.eigvals < 0).any():
            warnings.warn('Covariance matrix is not positive definite.')

        return self

    def _compute_covariance_survey(self):
        '''Compute the covariance matrix for a survey geometry.

        Returns
        -------
        self : GaussianCovariance
            Covariance matrix.
        '''

        # If kbins are set for the covariance matrix but not for the geometry,
        # set them for the geometry as well
        if self.is_kbins_set and not self.geometry.is_kbins_set:
            self.geometry.set_kbins(self.kmin, self.kmax, self.dk)

        WinKernel = self.geometry.get_window_kernels()

        # delta_k_max off-diagonal elements of the covariance
        # matrix will be computed each side of the diagonal
        delta_k_max = WinKernel.shape[1]//2

        # Number of k bins
        kbins = self.kbins

        # If the power spectrum for a given ell is not set, use a zero array instead
        P0 = self.get_pk(0, force_return=True, remove_shotnoise=True)
        P2 = self.get_pk(2, force_return=True)
        P4 = self.get_pk(4, force_return=True)

        cov = np.zeros((kbins, kbins, 6))

        for ki in range(kbins):
            # Iterate delta_k_max bins either side of the diagonal
            for kj in range(max(ki - delta_k_max, 0), min(ki + delta_k_max + 1, kbins)):
                # Relative index of k2 for WinKernel elements
                delta_k = kj - ki + delta_k_max

                cov[ki][kj] = \
                    WinKernel[ki, delta_k, 0]*P0[ki]*P0[kj] + \
                    WinKernel[ki, delta_k, 1]*P0[ki]*P2[kj] + \
                    WinKernel[ki, delta_k, 2]*P0[ki]*P4[kj] + \
                    WinKernel[ki, delta_k, 3]*P2[ki]*P0[kj] + \
                    WinKernel[ki, delta_k, 4]*P2[ki]*P2[kj] + \
                    WinKernel[ki, delta_k, 5]*P2[ki]*P4[kj] + \
                    WinKernel[ki, delta_k, 6]*P4[ki]*P0[kj] + \
                    WinKernel[ki, delta_k, 7]*P4[ki]*P2[kj] + \
                    WinKernel[ki, delta_k, 8]*P4[ki]*P4[kj] + \
                    (1 + self.alphabar)*(
                        WinKernel[ki, delta_k, 9]*(P0[ki] + P0[kj])/2. +
                        WinKernel[ki, delta_k, 10]*P2[ki] + WinKernel[ki, delta_k, 11]*P4[ki] +
                        WinKernel[ki, delta_k, 12]*P2[kj] +
                    WinKernel[ki, delta_k, 13]*P4[kj]
                ) + \
                    (1 + self.alphabar)**2 * WinKernel[ki, delta_k, 14]

        self.set_ell_cov(0, 0, cov[:, :, 0])
        self.set_ell_cov(2, 2, cov[:, :, 1])
        self.set_ell_cov(4, 4, cov[:, :, 2])
        self.set_ell_cov(0, 2, cov[:, :, 3])
        self.set_ell_cov(0, 4, cov[:, :, 4])
        self.set_ell_cov(2, 4, cov[:, :, 5])

        if (self.eigvals < 0).any():
            warnings.warn('Covariance matrix is not positive definite.')

        if not np.allclose(self.cov, self.cov.T):
            warnings.warn('Covariance matrix is not symmetric.')

        return self
