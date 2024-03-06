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

        assert len(pk) == self.kbins, 'Power spectrum must have the same number of bins as the covariance matrix.'

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

        force_return : bool, float, optional
            If the power spectrum for the given ell is not set, return a zero array if True or the specified value if a float.
            
        remove_shotnoise : bool, optional
            Whether to remove the shotnoise from the power spectrum monopole.
        '''

        if ell in self._pk.keys():
            pk = self._pk[ell]
            if (not remove_shotnoise) and ell == 0:
                self.logger.info(f'Adding shotnoise = {self.shotnoise} to ell = 0.')
                return pk + self.shotnoise
            return pk
        elif type(force_return) != bool:
            return force_return*np.ones(self.kbins)
        elif force_return:
            return np.zeros(self.kbins)

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

        # terms without the power spectrum have to be multiplied by its relative normalization pk_renorm
        func = lambda ik,jk:                                   self._get_pk_pk_term(ik, jk) + \
                    (1 + self.alpha)    * self.pk_renorm    * self._get_pk_shotnoise_term(ik, jk) + \
                    (1 + self.alpha)**2 * self.pk_renorm**2 * self._get_shotnoise_shotnoise_term(ik, jk)

        self._set_survey_covariance(self._build_covariance_survey(func), self)
        eigvals = self.eigvals
        if (eigvals < 0).any():
            self.logger.warning(f'Covariance matrix is not positive definite. Worst of {sum(eigvals < 0)} negative eigenvalues is {eigvals.min():.2e}.')
            # extra_modes = int(0.2*self.geometry.kmodes_sampled)
            # self.geometry.kmodes_sampled += extra_modes
            # self.logger.warning(f'Sampling {extra_modes} more kmodes. Total = {self.geometry.kmodes_sampled}.')
            # self.geometry.compute_window_kernels()
            # self._compute_covariance_survey()
        self.logger.info(f'Condition number is {eigvals.max()/eigvals[eigvals > 0].min():.2e}.')

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
    
    def rescale_shotnoise(self, ref_cov, set=True, preproc=None):
        original_alpha = self.geometry.alpha
        set_alpha = self.alpha

        original_shotnoise = (1 + original_alpha) * self.pk_renorm * self.geometry.I('12')/self.geometry.I('22')
        set_shotnoise =      (1 +      set_alpha) * self.pk_renorm * self.geometry.I('12')/self.geometry.I('22')

        from scipy.optimize import root_scalar
        result = root_scalar(self._get_shotnoise_rescaling_func(ref_cov, preproc=preproc), x0=0, x1=set_alpha)

        new_alpha = result.root
        new_shotnoise = (1 + new_alpha) * self.pk_renorm * self.geometry.I('12')/self.geometry.I('22')

        self.logger.info(f'alpha rescaling: {original_alpha} -> {set_alpha} -> {new_alpha}')
        self.logger.info(f'shotnoise rescaling: {original_shotnoise} -> {set_shotnoise} -> {new_shotnoise}')
        self.logger.info(f'ratio: {new_shotnoise/set_shotnoise}')

        if set:
            self.alpha = new_alpha

        return new_alpha, new_shotnoise

    def _get_shotnoise_rescaling_func(self, reference, preproc=None):
        if preproc is None:
            preproc = lambda x: x

        @np.vectorize
        def dlikelihood(alpha):

            cov_func = lambda ik,jk:                         self._get_pk_pk_term(ik, jk) + \
                        (1 + alpha)    * self.pk_renorm    * self._get_pk_shotnoise_term(ik, jk) + \
                        (1 + alpha)**2 * self.pk_renorm**2 * self._get_shotnoise_shotnoise_term(ik, jk)
            
            get_dcov_dalpha = self._build_covariance_survey(self._get_pk_shotnoise_term) + \
              2*(1 + alpha) * self._build_covariance_survey(self._get_shotnoise_shotnoise_term)
            
            covariance = preproc(self._set_survey_covariance(self._build_covariance_survey(cov_func))).cov
            precision_matrix = np.linalg.inv(covariance)
            dcov_dalpha = preproc(self._set_survey_covariance(get_dcov_dalpha)).cov

            return np.trace((reference.cov - covariance) @ precision_matrix @ dcov_dalpha @ precision_matrix)
        
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

class TrispectrumCovariance(base.PowerSpectrumMultipolesCovariance):
    '''Regular trispectrum covariance matrix of power spectrum multipoles in a given geometry.

    Attributes
    ----------
    geometry : geometry.Geometry
        Geometry of the survey. Can be a BoxGeometry or a SurveyGeometry object.
    '''

    def __init__(self, geometry=None):

        base.PowerSpectrumMultipolesCovariance.__init__(self, geometry=geometry)
        self.logger = logging.getLogger('TrispectrumCovariance')

        from powercovfft import PowerSpecCovFFT
        self.calculator = PowerSpecCovFFT()

    def set_kbins(self, kmin, kmax, dk):
        '''Set the k-binning for the covariance matrix.

        Parameters
        ----------
        kmin : float
            Minimum k value.
        kmax : float
            Maximum k value.
        dk : float
            Width of the k bins.
        '''

        base.PowerSpectrumMultipolesCovariance.set_kbins(self, kmin, kmax, dk)

        ## Set the FFTLog
        config_fft = {'nu':-0.3, 'kmin':1e-5, 'kmax':1e+1, 'nmax':512}
        self.calculator.set_power_law_decomp(config_fft)

        k1, k2 = self.kmin_matrices

        ## Precompute the master integrals
        self.calculator.calc_master_integral(k1, k2)

        return self
    
    def set_pk(self, pk_linear, k=None):
        '''Set the input linear power spectrum to be used for the covariance calculation.

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        pk : array_like
            Power spectrum.
        '''

        if callable(pk_linear):
            self.pk_linear = pk_linear
        else:
            if len(k) != len(pk_linear):
                self.logger.error('k and pk must have the same length.')

            from scipy.interpolate import InterpolatedUnivariateSpline
            pk_spline = InterpolatedUnivariateSpline(np.log(k), np.log(pk_linear))

            self.pk_linear = lambda k: np.exp(pk_spline(np.log(k)))

        self.calculator.set_pk_lin(self.pk_linear)
    
    def set_params(self, fgrowth, b1, b2=None, g2=None, b3=None, g3=None, g2x=None, g21=None):
        '''Set the bias parameters to be used for the covariance calculation. If the optional
           parameters are not set, will use expressions for non-local bias (g_i) from local
           lagrangian approximation and non-linear bias (b_i) from peak-background split fit
           of arXiv:1511.01096 rescaled using Appendix C.2 of arXiv:1812.03208, (useful if
           those parameters aren't constrained).

        Parameters
        ----------
        fgrowth : float
            Growth rate of the linear power spectrum.
        b1 : float
            Linear bias.
        b2 : float, optional
            Quadratic bias.
        g2 : float, optional
            gamma_2 non-local bias.
        b3 : float, optional
            Cubic bias.
        g3 : float, optional
            gamma_3 non-local bias.
        g2x : float, optional
            gamma_2^x non-local bias.
        g21 : float, optional
            gamma_{21} non-local  bias.
        '''

        if g2 is None:
            g2 = -2/7*(b1 - 1)
        if g3 is None:
            g3 = 11/63*(b1 - 1)
        if b2 is None:
            b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4/3*g2
        if g2x is None:
            g2x = -2/7*b2
        if g21 is None:
            g21 = -22/147*(b1 - 1)
        if b3 is None:
            b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3 + 4*g2x - 4/3*g3 - 8/3*g21 - 32/21*g2

        #    equation 78 of arXiv:1812.03208 (Wadekar   basis)
        # vs equation A1 of arXiv:2308.08593 (Kobayashi basis)
        self.calculator.bias = {
            'b1': b1,
            'b2': b2,
            'bG2': g2,
            'b3': b3,
            'bG3': g3,
            'bdG2': g2x,
            'bGamma3': -4/7*g21, # equation 44 of arXiv:1812.03208
        }

        self.calculator.fgrowth = fgrowth

        return self

    def _compute_covariance_box(self):
        '''Compute the covariance matrix for a box geometry.

        Returns
        -------
        self : TrispectrumCovariance
            Covariance matrix.
        '''

        self.calculator.vol = self.geometry.volume
        self.calculator.ndens = self.geometry.nbar
                                  
        self._build_covariance()

        return self

    def _compute_covariance_survey(self):
        '''Compute the covariance matrix for a survey geometry.

        Returns
        -------
        self : TrispectrumCovariance
            Covariance matrix.
        '''

        self.calculator.vol = self.geometry.I('22')**2 / self.geometry.I('44')
        self.calculator.ndens = self.geometry.I('44') / self.geometry.I('34')
        self.calculator.ndens2 = self.geometry.I('44') / self.geometry.I('24')
        
        self._build_covariance()

        return self
    
    def _build_covariance(self):

        ## Compute the elementary integrals Eq. (13)
        self.calculator.calc_base_integral()

        k1, k2 = self.kmin_matrices

        ## Compute the non-Gaussian covariance for each combination of multipoles
        self.set_ell_cov(0, 0, self.calculator.get_cov_T0(0, 0, k1, k2))
        self.set_ell_cov(2, 2, self.calculator.get_cov_T0(2, 2, k1, k2))
        self.set_ell_cov(4, 4, self.calculator.get_cov_T0(4, 4, k1, k2))
        self.set_ell_cov(0, 2, self.calculator.get_cov_T0(0, 2, k1, k2))
        self.set_ell_cov(0, 4, self.calculator.get_cov_T0(0, 4, k1, k2))
        self.set_ell_cov(2, 4, self.calculator.get_cov_T0(2, 4, k1, k2))

        return self
