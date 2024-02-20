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
import itertools as itt

import numpy as np

from . import base, geometry

__all__ = ['GaussianCovariance']

class GaussianCovariance(base.PowerSpectrumMultipolesCovariance):
    '''Gaussian covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    '''

    def __init__(self, geometry=None):
        base.PowerSpectrumMultipolesCovariance.__init__(self, geometry=geometry)
        self.logger = logging.getLogger('GaussianCovariance')

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
            self.logger.warning('Covariance matrix is not positive definite.')

        return self

    def _compute_covariance_survey(self):
        '''Compute the covariance matrix for a survey geometry.

        Returns
        -------
        self : GaussianCovariance
            Covariance matrix.
        '''

        func = lambda ik,jk:                 self._get_pk_pk_term(ik, jk) + \
                    (1 + self.alphabar)    * self._get_pk_shotnoise_term(ik, jk) + \
                    (1 + self.alphabar)**2 * self._get_shotnoise_shotnoise_term(ik, jk)

        self._set_survey_covariance(self._build_covariance_survey(func), self)

        if (self.eigvals < 0).any():
            self.logger.warning('Covariance matrix is not positive definite.')

        if not np.allclose(self.cov, self.cov.T):
            self.logger.warning('Covariance matrix is not symmetric.')

        return self
    
    def _build_covariance_survey(self, func):

        # If kbins are set for the covariance matrix but not for the geometry,
        # set them for the geometry as well
        if self.is_kbins_set and not self.geometry.is_kbins_set:
            self.geometry.set_kbins(self.kmin, self.kmax, self.dk)

        cov = np.zeros((self.kbins, self.kbins, 6))

        for ki in range(self.kbins):
            # Iterate delta_k_max bins either side of the diagonal
            for kj in range(max(ki - self.geometry.delta_k_max, 0), min(ki + self.geometry.delta_k_max + 1, self.kbins)):
                cov[ki][kj] = func(ki, kj)

        return cov
    
    @staticmethod
    def _set_survey_covariance(cov_array, covariance=None):
        if covariance is None:
            covariance = base.MultipoleFourierCovariance()

        covariance.set_ell_cov(0, 0, cov_array[:, :, 0])
        covariance.set_ell_cov(2, 2, cov_array[:, :, 1])
        covariance.set_ell_cov(4, 4, cov_array[:, :, 2])
        covariance.set_ell_cov(0, 2, cov_array[:, :, 3])
        covariance.set_ell_cov(0, 4, cov_array[:, :, 4])
        covariance.set_ell_cov(2, 4, cov_array[:, :, 5])

        return covariance

    def _get_pk_pk_term(self, ik, jk):
        
        WinKernel = self.geometry.get_window_kernels()

        # delta_k_max off-diagonal elements of the covariance
        # matrix will be computed each side of the diagonal
        delta_k = jk - ik + self.geometry.delta_k_max

        P0 = self.get_pk(0, force_return=True, remove_shotnoise=True)
        P2 = self.get_pk(2, force_return=True)
        P4 = self.get_pk(4, force_return=True)
        
        return \
            WinKernel[ik, delta_k, 0]*P0[ik]*P0[jk] + \
            WinKernel[ik, delta_k, 1]*P0[ik]*P2[jk] + \
            WinKernel[ik, delta_k, 2]*P0[ik]*P4[jk] + \
            WinKernel[ik, delta_k, 3]*P2[ik]*P0[jk] + \
            WinKernel[ik, delta_k, 4]*P2[ik]*P2[jk] + \
            WinKernel[ik, delta_k, 5]*P2[ik]*P4[jk] + \
            WinKernel[ik, delta_k, 6]*P4[ik]*P0[jk] + \
            WinKernel[ik, delta_k, 7]*P4[ik]*P2[jk] + \
            WinKernel[ik, delta_k, 8]*P4[ik]*P4[jk]

    def _get_pk_shotnoise_term(self, ik, jk):
        
        WinKernel = self.geometry.get_window_kernels()

        # delta_k_max off-diagonal elements of the covariance
        # matrix will be computed each side of the diagonal
        delta_k = jk - ik + self.geometry.delta_k_max

        P0 = self.get_pk(0, force_return=True, remove_shotnoise=True)
        P2 = self.get_pk(2, force_return=True)
        P4 = self.get_pk(4, force_return=True)
        
        return  WinKernel[ik, delta_k, 9]*(P0[ik] + P0[jk])/2. + \
                WinKernel[ik, delta_k, 10]*P2[ik] + WinKernel[ik, delta_k, 11]*P4[ik] + \
                WinKernel[ik, delta_k, 12]*P2[jk] + \
                WinKernel[ik, delta_k, 13]*P4[jk]
    
    def _get_shotnoise_shotnoise_term(self, ik, jk):
        
        WinKernel = self.geometry.get_window_kernels()

        # delta_k_max off-diagonal elements of the covariance
        # matrix will be computed each side of the diagonal
        delta_k = jk - ik + self.geometry.delta_k_max
        
        return WinKernel[ik, delta_k, 14]
    
    def rescale_shotnoise(self, ref_cov, set=True):
        original_alphabar = self.geometry.alpha
        set_alphabar = self.alphabar

        original_shotnoise = (1 + original_alphabar) * self.geometry.I('12')/self.geometry.I('22')
        set_shotnoise =      (1 +      set_alphabar) * self.geometry.I('12')/self.geometry.I('22')

        from scipy.optimize import root_scalar
        result = root_scalar(self._get_shotnoise_rescaling_func(ref_cov), x0=0., x1=0.001)

        new_alphabar = result.root
        new_shotnoise = (1 + new_alphabar) * self.geometry.I('12')/self.geometry.I('22')

        self.logger.info(f'alphabar rescaling: {original_alphabar} -> {set_alphabar} -> {new_alphabar}')
        self.logger.info(f'shotnoise rescaling: {original_shotnoise} -> {set_shotnoise} -> {new_shotnoise}')
        self.logger.info(f'ratio: {new_shotnoise/set_shotnoise}')

        if set:
            self.alphabar = new_alphabar

        return new_alphabar, new_shotnoise

    def _get_shotnoise_rescaling_func(self, ref_cov):

        def get_covariance(alphabar):
            cov_func = lambda ik,jk:        self._get_pk_pk_term(ik, jk) + \
                        (1 + alphabar)    * self._get_pk_shotnoise_term(ik, jk) + \
                        (1 + alphabar)**2 * self._get_shotnoise_shotnoise_term(ik, jk)

            return self._build_covariance_survey(cov_func)

        get_dcov_dalphabar = lambda alphabar: \
                               self._build_covariance_survey(self._get_pk_shotnoise_term) + \
            2*(1 + alphabar) * self._build_covariance_survey(self._get_shotnoise_shotnoise_term)
        
        @np.vectorize
        def dlikelihood(alphabar):
            covariance = self._set_survey_covariance(get_covariance(alphabar)).cov
            precision_matrix = np.linalg.inv(covariance)
            dcov_dalphabar = self._set_survey_covariance(get_dcov_dalphabar(alphabar)).cov

            return np.trace((ref_cov.cov - covariance) @ precision_matrix @ dcov_dalphabar @ precision_matrix)
        
        return dlikelihood
        
        
    def load_pk(self, filename, remove_shotnoise=None, set_shotnoise=True):
        '''Load power spectrum from pypower file and set it to be used for the covariance calculation.

        Parameters
        ----------
        filename : str
            Name of the pypower file containing the power spectrum.
        remove_shotnoise : bool, optional
            Whether pypower should be used to remove the shotnoise from the power spectrum monopole.
            If None, will be determined based on the geometry used.
        set_shotnoise : bool, optional
            Whether to rescale shotnoise matching the value in the power spectrum file.
        '''

        from pypower import PowerSpectrumMultipoles
        
        self.logger.info(f'Loading power spectrum from {filename}.')
        pypower = PowerSpectrumMultipoles.load(filename)

        kmin_file, kmax_file = pypower.kedges[[0, -1]]
        dk_file = np.diff(pypower.kedges).mean()

        if not self.is_kbins_set:
            self.set_kbins(kmin_file, kmax_file, dk_file)

        if self.kmin < kmin_file or self.kmax > kmax_file:
            raise ValueError('kmin and kmax of the covariance matrix must be within the range of the power spectrum file.')

        imin = np.round((self.kmin - kmin_file)/dk_file).astype(int)
        imax = np.round((self.kmax - kmin_file)/dk_file).astype(int)
        di   = self.dk/dk_file

        self.logger.info(f'Cutting power spectrum from {kmin_file} < k < {kmax_file} to {self.kmin} < k < {self.kmax}.')
        
        if np.allclose(np.round(di), di):
            di = np.round(di).astype(int)
        else:
            raise ValueError(f'dk = {self.dk} must be a multiple of dk_file = {dk_file}.')
        if di != 1:
            self.logger.info(f'Rebinning power spectrum by a factor of {di}. From dk = {dk_file} to dk = {self.dk}.')

        if remove_shotnoise is None:
            if self.geometry is None:
                remove_shotnoise = True
            elif isinstance(self.geometry, geometry.BoxGeometry):
                remove_shotnoise = False
            elif isinstance(self.geometry, geometry.SurveyGeometry):
                remove_shotnoise = True
            else:
                remove_shotnoise = True

        if remove_shotnoise:
            self.logger.info('pypower is removing shotnoise from the power spectrum.')
        else:
            self.logger.info('pypower is NOT removing shotnoise from the power spectrum.')

        P0, P2, P4 = pypower[imin:imax:di].get_power(remove_shotnoise=remove_shotnoise, complex=False)

        if set_shotnoise and self.geometry is not None:
            self.set_shotnoise(shotnoise = pypower.shotnoise)

        self.set_pk(P0, 0, has_shotnoise=not remove_shotnoise)
        self.set_pk(P2, 2)
        self.set_pk(P4, 4)