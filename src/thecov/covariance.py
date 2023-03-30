
import itertools as itt

import numpy as np
from scipy import fft, interpolate, integrate

from tqdm import tqdm

from . import trispectrum, base, utils, geometry


class GaussianCovariance(base.MultipoleCovariance, base.FourierBinned):

    def __init__(self, geometry):
        base.MultipoleCovariance.__init__(self)
        base.FourierBinned.__init__(self)

        self.geometry = geometry

        self._pk = {}

    @property
    def shotnoise(self):
        return self.geometry.shotnoise

    def set_pk(self, pk, ell, has_shotnoise=False):
        self._pk[ell] = pk
        if ell == 0 and not has_shotnoise:
            print(f'Adding shotnoise = {self.shotnoise} to ell = 0.')
            self._pk[ell] += self.shotnoise

    def get_pk(self, ell, force_return=False):
        if ell in self._pk.keys():
            return self._pk[ell]
        elif force_return:
            return np.zeros(self.kbins)

    def compute_covariance(self, ells=(0,2,4)):

        self._ells = ells
        self._mshape = (self.kbins, self.kbins)

        if isinstance(self.geometry, geometry.BoxGeometry):
            return self._compute_covariance_box()

        if isinstance(self.geometry, geometry.SurveyGeometry):
            return self._compute_covariance_survey()

    def _compute_covariance_box(self):

        P0 = self.get_pk(0, force_return=True)
        P2 = self.get_pk(2, force_return=True)
        P4 = self.get_pk(4, force_return=True)

        c = {}

        c[0,0] = P0**2 + 1/5*P2**2 + 1/9*P4**2
        c[0,2] = 2*P0*P2 + 2/7*P2** 2 + 4/7*P2*P4 + 100/693*P4**2
        c[0,4] = 2*P0*P4 + 18/35*P2**2 + 40/77*P2*P4 + 162/1001*P4**2
        c[2,2] = 5*P0**2 + 20/7*P0*P2 + 20/7*P0*P4 + 15/7*P2**2 + 120/77*P2*P4 + 8945/9009*P4**2
        c[2,4] = 36/7*P0*P2 + 200/77*P0*P4 + 108/77*P2**2 + 3578/1001*P2*P4 + 900/1001*P4**2
        c[4,4] = 9*P0**2 + 360/77*P0*P2 + 2916/1001*P0*P4 + 16101/5005*P2**2 + 3240/1001*P2*P4 + 42849/17017*P4**2

        for l1,l2 in itt.combinations_with_replacement(ells, r=2):
            self.set_ell_cov(l1, l2, 2/self.nmodes * np.diag(c[l1,l2]))

        return self.cov

    def _compute_covariance_survey(self):

        if self.is_kbins_set and not self.geometry.is_kbins_set:
            self.geometry.set_kbins(self.kmin, self.kmax, self.dk)
        
        WinKernel = self.geometry.get_window_kernels()

        kbins = self.kbins

        P0 = self.get_pk(0, force_return=True)
        P4 = self.get_pk(2, force_return=True)
        P2 = self.get_pk(4, force_return=True)
        
        C = np.zeros((kbins, kbins, 6))

        for ki in range(self.kbins):
            for kj in range(max(ki-3, 0), min(ki+4, kbins)):
                j = kj - ki + 3

                C[ki][kj] = \
                    WinKernel[ki,j,0]*P0[ki]*P0[kj] + \
                    WinKernel[ki,j,1]*P0[ki]*P2[kj] + \
                    WinKernel[ki,j,2]*P0[ki]*P4[kj] + \
                    WinKernel[ki,j,3]*P2[ki]*P0[kj] + \
                    WinKernel[ki,j,4]*P2[ki]*P2[kj] + \
                    WinKernel[ki,j,5]*P2[ki]*P4[kj] + \
                    WinKernel[ki,j,6]*P4[ki]*P0[kj] + \
                    WinKernel[ki,j,7]*P4[ki]*P2[kj] + \
                    WinKernel[ki,j,8]*P4[ki]*P4[kj] + \
                    1.01*(
                        WinKernel[ki,j,9]*(P0[ki] + P0[kj])/2. +
                        WinKernel[ki,j,10]*P2[ki] + WinKernel[ki,j,11]*P4[ki] +
                        WinKernel[ki,j,12]*P2[kj] + WinKernel[ki,j,13]*P4[kj]
                ) + 1.01**2 * WinKernel[ki,j,14]

        self.set_ell_cov(0, 0, C[:, :, 0])
        self.set_ell_cov(2, 2, C[:, :, 1])
        self.set_ell_cov(4, 4, C[:, :, 2])
        self.set_ell_cov(0, 2, C[:, :, 3])
        self.set_ell_cov(0, 4, C[:, :, 4])
        self.set_ell_cov(2, 4, C[:, :, 5])

        return self.cov


# -------------- TO BE REFACTORED AND VALIDATED ----------------

class TrispectrumSurveyWindowCovariance(base.MultipoleCovariance, base.FourierBinned):

    def __init__(self, gaussian_cov=None):
        super().__init__()
        self._covariance = None
        self.T0 = None
        self.Plin = None

        self._gaussian_cov = gaussian_cov

    def set_bias_parameters(self, b1, be, g2, b2, g3, g2x, g21, b3):
        self.T0 = trispectrum.T0(b1, be, g2, b2, g3, g2x, g21, b3)

    def load_Plin(self, k, P):
        self.Plin = interpolate.InterpolatedUnivariateSpline(k, P)

    def _trispectrum_element(self, l1, l2, k1, k2):

        # We'll still implement hexadecupole in trispectrum.
        # For now, return 0
        # if 4 in (l1,l2):
        # return 0

        # Returns the tree-level trispectrum as a function of multipoles and k1, k2
        T0 = self.T0
        Plin = self.Plin

        T0.l1 = l1
        T0.l2 = l2

        trisp_integrand = np.vectorize(self._trispectrum_integrand)

        expr = self.I['44']*(Plin(k1)**2*Plin(k2)*T0.ez3(k1, k2) + Plin(k2)**2*Plin(k1)*T0.ez3(k2, k1))\
            + 8*self.I['34'] * Plin(k1)*Plin(k2)*T0.e34o44_1(k1, k2)

        return (integrate.quad(trisp_integrand, -1, 1, args=(k1, k2))[0]/2. + expr)/self.I['22']**2

    def _trispectrum_integrand(self, u12, k1, k2):

        Plin = self.Plin
        T0 = self.T0

        return (8*self.I['44']*(Plin(k1)**2*T0.e44o44_1(u12, k1, k2) + Plin(k2)**2*T0.e44o44_1(u12, k2, k1))
                + 16*self.I['44']*Plin(k1)*Plin(k2) *
                T0.e44o44_2(u12, k1, k2)
                + 8*self.I['34']*(Plin(k1)*T0.e34o44_2(u12,
                                  k1, k2) + Plin(k2)*T0.e34o44_2(u12, k2, k1))
                + 2*self.I['24']*T0.e24o44(u12, k1, k2)
                ) * Plin(np.sqrt(k1**2+k2**2+2*k1*k2*u12))

    def compute_covariance(self, tqdm=tqdm):

        kbins = self._gaussian_cov.kbins
        k = self._gaussian_cov.k

        trisp = np.vectorize(self._trispectrum_element)

        covariance = np.zeros(2*[3*kbins])

        for i in tqdm(range(kbins), total=kbins, desc="Computing trispectrum contribution"):
            # covariance[i,        :  kbins] = trisp(0, 0, k[i], k)
            # covariance[i,   kbins:2*kbins] = trisp(0, 2, k[i], k)
            covariance[i, 2*kbins:3*kbins] = trisp(0, 4, k[i], k)

            # covariance[kbins+i,   kbins:2*kbins] = trisp(2, 2, k[i], k)
            covariance[kbins+i, 2*kbins:3*kbins] = trisp(2, 4, k[i], k)

            covariance[2*kbins+i, 2*kbins:3*kbins] = trisp(4, 4, k[i], k)

        covariance[kbins:, :kbins] = np.transpose(covariance[:kbins, kbins:])

        self._covariance = covariance

    @property
    def I(self):
        return self._gaussian_cov.I


class SuperSampleCovariance(base.MultipoleCovariance, base.FourierBinned):

    def __init__(self, survey_geometry, gaussian_cov):
        super().__init__()
        self._covariance = None
        self.T0 = None
        self.Plin = None

        self._survey_geometry = survey_geometry

        self._gaussian_cov = gaussian_cov

    def set_bias_parameters(self, b1, be, g2, b2, g3, g2x, g21, b3):
        self.T0 = trispectrum.T0(b1, be, g2, b2, g3, g2x, g21, b3)

    def load_Plin(self, k, P):
        self.Plin = interpolate.InterpolatedUnivariateSpline(k, P)

    @property
    def I(self):
        return self._gaussian_cov.I

    def compute_sigma_factors(self):
        Plin = self.Plin

        # Calculating the RMS fluctuations of supersurvey modes
        # (e.g., sigma22Sq which was defined in Eq. (33) and later calculated in Eq.(65)

        # kmin = 0.
        # kmax = 0.25
        # dk = 0.005
        # k = np.arange(kmin + dk/2, kmax + dk/2, dk)
        kmax = (self._survey_geometry.Nmesh + 1) * \
            np.pi/self._survey_geometry.BoxSize

        sigma22Sq = np.zeros((3, 3))
        sigma10Sq = np.zeros((3, 3))
        sigma22x10 = np.zeros((3, 3))

        kmean, powerW10, powerW22, powerW22xW10 = self._survey_geometry.compute_power_multipoles()

        for (i, l1), (j, l2) in itt.product(enumerate((0, 2, 4)), repeat=2):
            Pwin = interpolate.InterpolatedUnivariateSpline(
                kmean, powerW22xW10[l1, l2])
            sigma22x10[i, j] = integrate.quad(
                lambda q: q**2*Plin(q)*Pwin(q)/2/np.pi**2, 0, kmax)[0]

        for (i, l1), (j, l2) in itt.combinations_with_replacement(enumerate((0, 2, 4)), 2):
            Pwin = interpolate.InterpolatedUnivariateSpline(
                kmean, powerW22[l1, l2])
            sigma22Sq[i, j] = integrate.quad(
                lambda q: q**2*Plin(q)*Pwin(q)/2/np.pi**2, 0, kmax)[0]
            sigma22Sq[j, i] = sigma22Sq[i, j]

            Pwin = interpolate.InterpolatedUnivariateSpline(
                kmean,  powerW10[l1, l2])
            sigma10Sq[i, j] = integrate.quad(
                lambda q: q**2*Plin(q)*Pwin(q)/2/np.pi**2, 0, kmax)[0]
            sigma10Sq[j, i] = sigma10Sq[i, j]

        return sigma10Sq, sigma22Sq, sigma22x10

    def compute_covariance(self):
        Plin = self.Plin
        kbins = self._gaussian_cov.kbins
        k = self._gaussian_cov.k

        sigma10Sq, sigma22Sq, sigma22x10 = self.compute_sigma_factors()

        # Derivative of the linear power spectrum
        dlnPk = Plin.derivative()(k)*k/Plin(k)

        # Kaiser terms
        rsd = {
            0: 1 + (2*self.T0.be)/3 + self.T0.be**2/5,
            2: (4*self.T0.be)/3 + (4*self.T0.be**2)/7,
            4: (8*self.T0.be**2)/35
        }

        # Legendre polynomials
        def lp(l, mu):
            if l == 0:
                return 1
            if l == 2:
                return (3*mu**2 - 1)/2.
            if l == 4:
                return (35*mu**4 - 30*mu**2 + 3)/8.

        # Calculating multipoles of the Z12 kernel
        def Z12Multipoles(i, l, dlnpk):
            return integrate.quad(lambda mu: lp(i, mu)*self.T0.Z12Kernel(l, mu, dlnpk), -1, 1)[0]

        Z12Multipoles = np.vectorize(Z12Multipoles)

        b1 = self.T0.b1
        b2 = self.T0.b2
        be = self.T0.be

        I = self._survey_geometry.I

        # Terms used in the LA calculation
        covaLAterm = np.zeros((3, len(k)))
        for l, i, j in itt.product(range(3), repeat=3):
            covaLAterm[l] += 1/4.*sigma22x10[i, j]*Z12Multipoles(2*i, 2*l, dlnPk) \
                * integrate.quad(lambda mu: lp(2*j, mu)*(1 + be*mu**2), -1, 1)[0]
        covaBC = {}
        covaLA = {}
        for l1, l2 in itt.combinations_with_replacement([0, 2, 4], 2):
            covaBC[l1, l2] = np.zeros((len(k), len(k)))
            for i, j in itt.product(range(3), repeat=2):
                covaBC[l1, l2] += 1/4.*sigma22Sq[i, j]*np.outer(Plin(k)*Z12Multipoles(2*i, l1, dlnPk),
                                                                Plin(k)*Z12Multipoles(2*j, l2, dlnPk))

            # covaLA[l1,l2] = -rsd[l2]*np.outer(Plin(k)*(covaLAterm[int(l1/2)] + I['32']/I['22']/I['10']*rsd[l1]*Plin(k)*b2/b1**2+2/I['10']*rsd[l1]), Plin(k)) \
            #         - rsd[l1]*np.outer(Plin(k), Plin(k)*(covaLAterm[int(l2/2)] + I['32']/I['22']/I['10']*rsd[l2]*Plin(k)*b2/b1**2+2/I['10']*rsd[l2])) \
            #         + sigma10Sq[0,0]*rsd[l1]*rsd[l2]*np.outer(Plin(k),Plin(k))

            covaLA[l1, l2] = -rsd[l2] * np.outer(Plin(k)*(covaLAterm[int(l1/2)] + I['32']/I['22']/I['10']*rsd[l1]*Plin(k)*b2/b1**2+2/I['10']*rsd[l1]), Plin(k)) \
                - rsd[l1]*np.outer(Plin(k), Plin(k)*(covaLAterm[int(l2/2)] + I['32']/I['22']/I['10']*rsd[l2]*Plin(k)*b2/b1**2+2/I['10']*rsd[l2])) \
                + (np.sum(1/4.*sigma10Sq[i, j]*integrate.quad(lambda mu: lp(2*i, mu)*(1 + be*mu**2), -1, 1)[0] * integrate.quad(lambda mu: lp(2*j, mu)*(1 + be*mu**2), -1, 1)[0])
                   + 1/I['10']) * rsd[l1]*rsd[l2] * np.outer(Plin(k), Plin(k))

        self.covaBC = {key: Covariance(covaBC[key]) for key in covaBC}

        self.covaLA = {key: Covariance(covaLA[key]) for key in covaBC}

        self._multipole_covariance = {key: Covariance(
            covaLA[key] + covaBC[key]) for key in covaBC}

        covariance = np.zeros(2*[3*kbins])

        C = self._multipole_covariance

        self._covariance = np.block([
            [C[0, 0].cov, C[0, 2].cov, C[0, 4].cov],
            [C[0, 2].cov, C[2, 2].cov, C[2, 4].cov],
            [C[0, 4].cov, C[2, 4].cov, C[4, 4].cov],
        ])
        self._covariance_BC = np.block([
            [covaBC[0, 0], covaBC[0, 2], covaBC[0, 4]],
            [covaBC[0, 2], covaBC[2, 2], covaBC[2, 4]],
            [covaBC[0, 4], covaBC[2, 4], covaBC[4, 4]],
        ])
        self._covariance_LA = np.block([
            [covaLA[0, 0], covaLA[0, 2], covaLA[0, 4]],
            [covaLA[0, 2], covaLA[2, 2], covaLA[2, 4]],
            [covaLA[0, 4], covaLA[2, 4], covaLA[4, 4]],
        ])
