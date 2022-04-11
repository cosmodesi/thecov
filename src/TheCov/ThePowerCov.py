import copy
from abc import ABC, abstractmethod
import itertools as itt

import numpy as np
import dask.array as da
from scipy import fft
from scipy import interpolate
from scipy import integrate

from tqdm import tqdm

from . import Trispectrum, utils


class Covariance():

    def __init__(self, covariance=None):
        self._covariance = covariance

    @property
    def cov(self):
        return self._covariance

    @property
    def cor(self):
        v = np.sqrt(np.diag(self._covariance))
        outer_v = np.outer(v, v)
        outer_v[outer_v == 0] = np.inf
        correlation = self._covariance / outer_v
        correlation[self._covariance == 0] = 0
        return correlation

    def __add__(self, y):
        return Covariance(self.cov + (y.cov if isinstance(y, Covariance) else y))

    def __sub__(self, y):
        return Covariance(self.cov - (y.cov if isinstance(y, Covariance) else y))

    def save(self, filename):
        np.savez(filename if filename.strip()[-4:] == '.npz' else filename + '.npz',
                 covariance=self._covariance)

    @classmethod
    def load(cls, filename):
        covariance = cls()
        with np.load(filename, mmap_mode='r') as data:
            covariance._covariance = data['covariance']
        return covariance

    @classmethod
    def loadtxt(cls, *args, **kwargs):
        covariance = cls()
        covariance._covariance = np.loadtxt(*args, **kwargs)
        return covariance

    @classmethod
    def from_array(cls, a):
        covariance = cls()
        covariance._covariance = a
        return a

class PowerCovariance(Covariance):

    def __init__(self):
        super().__init__()

        self._kmin = None
        self._kmax = None
        self._dk = None
        self._k = None

    def set_kbins(self, kmin, kmax, dk):
        self._dk = dk
        self._kmax = kmax
        self._kmin = kmin
        self._k = np.arange(kmin+dk/2, kmax+dk/2, dk)

    @property
    def kbins(self):
        return len(self._k)

    @property
    def k(self):
        return self._k

class PowerMultipoleCovariance(PowerCovariance):

    def __init__(self):
        super().__init__()

        self._multipole_covariance = None
        self._P = {}
        
    def load_multipole(self, k, P_ell, ell):
        self._P[ell] = interpolate.InterpolatedUnivariateSpline(k, P_ell)

    def set_multipole_interpolator(self, P_ell_k, ell):
        self._P[ell] = P_ell_k

    def get_multipole_interpolator(self, ell):
        return self._P[ell]

    def get_multipole_covariance(self, l1=None, l2=None):
        if  self._multipole_covariance == None:
            kbins = self.cov.shape[0]//3 # TODO: improve this
            self._multipole_covariance = {}
            self._multipole_covariance[0,0] = Covariance(self.cov[:kbins, :kbins])
            self._multipole_covariance[0,2] = Covariance(self.cov[:kbins, kbins:2*kbins])
            self._multipole_covariance[0,4] = Covariance(self.cov[:kbins, 2*kbins:3*kbins])
            self._multipole_covariance[2,2] = Covariance(self.cov[kbins:2*kbins,   kbins:2*kbins])
            self._multipole_covariance[2,4] = Covariance(self.cov[kbins:2*kbins, 2*kbins:3*kbins])
            self._multipole_covariance[4,4] = Covariance(self.cov[2*kbins:3*kbins, 2*kbins:3*kbins])

        if None in (l1, l2):
            assert l1 == None and l2 == None
            return self._multipole_covariance

        return self._multipole_covariance[l1,l2]

    @property
    def mcov(self):
        return self.get_multipole_covariance()


class GaussianBoxCovariance(PowerMultipoleCovariance):

    def __init__(self, Lbox=None, nbar=None):
        super().__init__()
        if Lbox:
            self.Lbox = Lbox
        if nbar:
            self.nbar = nbar

    @property
    def Pshot(self):
        return 1/self.nbar

    @Pshot.setter
    def Pshot(self, Pshot):
        self.nbar = 1/Pshot

    def compute_covariance(self):
        Cov = self._compute_gaussian_covariance_diagonal()
        self._multipole_covariance = {key: np.diag(Cov[key]) for key in Cov.keys()}
        self._covariance = np.block([[np.diag(Cov[l1,l2]) if l1 < l2 else np.diag(Cov[l2,l1]).T for l1 in [0,2,4]] for l2 in [0,2,4]])

    def _compute_gaussian_covariance_diagonal(self):

        k  = self._k
        dk = self._dk

        P0_SN = lambda k: self._P[0](k) + self.Pshot

        Nmodes = self.Lbox**3/3/(2*np.pi**2) * ((k+dk/2)**3 - (k-dk/2)**3)

        P = self._P

        Cov = {}

        Cov[0,0] = 2/Nmodes*(P0_SN(k)**2 + P[2](k)**2/5 + P[4](k)**2/9)
        Cov[0,2] = 2/Nmodes*(2*P0_SN(k)*P[2](k) + 2*P[2](k)**2/7 + 4*P[2](k)*P[4](k)/7 + 100*P[4](k)**2/693)
        Cov[0,4] = 2/Nmodes*(2*P0_SN(k)*P[4](k) + 18*P[2](k)**2/35 + 40*P[2](k)*P[4](k)/77 + 162*P[4](k)**2/1001)
        Cov[2,2] = 2/Nmodes*(5*P0_SN(k)**2 + 20*P0_SN(k)*P[2](k)/7 + 20*P0_SN(k)*P[4](k)/7 + 15*P[2](k)**2/7 + 120*P[2](k)*P[4](k)/77 + 8945*P[4](k)**2/9009)
        Cov[2,4] = 2/Nmodes*(36*P0_SN(k)*P[2](k)/7 + 200*P0_SN(k)*P[4](k)/77 + 108*P[2](k)**2/77 + 3578*P[2](k)*P[4](k)/1001 + 900*P[4](k)**2/1001)
        Cov[4,4] = 2/Nmodes*(9*P0_SN(k)**2 + 360*P0_SN(k)*P[2](k)/77 + 2916*P0_SN(k)*P[4](k)/1001 + 16101*P[2](k)**2/5005 + 3240*P[2](k)*P[4](k)/1001 + 42849*P[4](k)**2/17017)

        return Cov


class GaussianSurveyWindowCovariance(PowerMultipoleCovariance):

    @classmethod
    def from_randoms(cls):
        pass

    def set_randoms(self, random_catalog, Nmesh=None, BoxSize=None, alpha=None, mesh_kwargs=None):
        if mesh_kwargs is None:
            mesh_kwargs = {}
        if Nmesh:   self.Nmesh   = Nmesh
        if BoxSize: self.BoxSize = BoxSize
        if alpha:   self.alpha   = alpha

        self._mesh_kwargs = {
            'Nmesh':       self.Nmesh,
            'BoxSize':     self.BoxSize,
            'interlaced':  True,
            'compensated': True,
            'resampler':   'tsc',
        }

        self._mesh_kwargs.update(mesh_kwargs)

        self._randoms = random_catalog

        self.randoms['RelativePosition'] = self.randoms['Position']
        self.randoms['Position'] += da.array(3*[self.BoxSize/2])

        self.ngals = self.randoms.size * self.alpha

        self._I = {}
        for i,j in ['22', '11', '12', '10', '24', '14', '34', '44', '32']:
            self.randoms[f'W{i}{j}'] = self.randoms['NZ']**(int(i)-1) * self.randoms['WEIGHT_FKP']**int(j)
            # Computing I_ij integrals
            self._I[f'{i}{j}'] = self.randoms[f'W{i}{j}'].sum() * self.alpha

    def compute_cartesian_ffts(self, Wij=('W12', 'W22'), tqdm=tqdm):
        num_ffts = lambda l: (l+1)*(l+2)/2

        mesh_kwargs = self._mesh_kwargs

        Fij = {}
        self._I = {k: self.I[k].compute() for k in self.I.keys()}

        x = self.randoms['RelativePosition'].T

        for w in Wij:

            print(f'Computing moments of {w}')

            print('Computing 0th order moment...', end='')
            Fij[w] = self.randoms.to_mesh(value=w, **mesh_kwargs).paint(mode='complex')
            print(' Done!')

            print('Computing 2nd order moments')
            for (i,i_label),(j,j_label) in tqdm(itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2), total=num_ffts(2)):
                label = w + i_label + j_label
                self.randoms[label] = self.randoms[w] * x[i]*x[j] /(x[0]**2 + x[1]**2 + x[2]**2)
                Fij[label] = self.randoms.to_mesh(value=label, **mesh_kwargs).paint(mode='complex')

            print('Computing 4th order moments')
            for (i,i_label),(j,j_label),(k,k_label),(l,l_label) in tqdm(itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4), total=num_ffts(4)):
                label = w + i_label + j_label + k_label + l_label
                self.randoms[label] = self.randoms[w] * x[i]*x[j]*x[k]*x[l] /(x[0]**2 + x[1]**2 + x[2]**2)**2
                Fij[label] = self.randoms.to_mesh(value=label, **mesh_kwargs).paint(mode='complex')

        self._W = {k[1:]: self._format_fft(Fij[k], k[:3]) for k in Fij.keys()}

    def save_cartesian_ffts(self, filename):
        np.savez(filename if filename.strip()[-4:] == '.npz' else filename + '.npz',
                 **{f'W{k}': self.W[k] for k in self.W.keys()},
                 **{f'I{k}': self.I[k] for k in self.I.keys()},
                 BoxSize=self.BoxSize,
                 Nmesh=self.Nmesh,
                 alpha=self.alpha)

    def load_cartesian_ffts(self, filename):
        with np.load(filename, mmap_mode='r') as data:
            self._W = {f[1:]: data[f] for f in data.files if f[0] == 'W' }
            self._I = {f[1:]: data[f] for f in data.files if f[0] == 'I' }

            self.BoxSize = data['BoxSize']
            self.Nmesh   = data['Nmesh']
            self.alpha   = data['alpha']

    def compute_window_kernels(self, kmodes_sampled, icut, tqdm=tqdm):
        # As the window falls steeply with k, only low-k regions are needed for the calculation.
        # Therefore, cutting out the high-k modes in the FFTs using the icut parameter

        cutW = lambda W: W[W.shape[0]//2-icut-1 : W.shape[0]//2+icut,
                           W.shape[1]//2-icut-1 : W.shape[1]//2+icut,
                           W.shape[2]//2-icut-1 : W.shape[2]//2+icut]

        self._Wcut = {key: cutW(self.W[key]) for key in self.W.keys()}

        kBinWidth = self._dk
        nBins = self.kbins
        kfun = 2*np.pi/self.BoxSize

        # Recording the k-modes in different shells
        # Bin_kmodes contains [kx,ky,kz,radius] values of all the modes in the bin

        nk = int(kBinWidth*nBins/kfun)+1

        ix, iy, iz = np.mgrid[-nk:nk+1,
                              -nk:nk+1,
                              -nk:nk+1]

        rk = np.sqrt(ix**2 + iy**2 + iz**2)

        Bin_ModeNum = np.zeros(nBins,dtype=int)

        Bin_kmodes = []
        for i in range(nBins):
            Bin_kmodes.append([])

        sort = (rk*kfun/kBinWidth).astype(int)

        for i in tqdm(range(nBins), desc="Sorting k-modes in shells"):
            ind = (sort == i)
            Bin_ModeNum[i] = len(ix[ind])
            Bin_kmodes[i] = np.hstack((ix[ind].reshape(-1,1),
                                       iy[ind].reshape(-1,1),
                                       iz[ind].reshape(-1,1),
                                       rk[ind].reshape(-1,1)))

        self._Bin_ModeNum = Bin_ModeNum
        self._Bin_kmodes = Bin_kmodes

        self.WinKernel = np.array([
            self._compute_window_kernel_row(Nbin, kmodes_sampled=kmodes_sampled, icut=icut, tqdm=tqdm)
                          for Nbin, _ in tqdm(enumerate(self._k), total=self.kbins, desc="Computing window kernels")])

    def save_window_kernels(self, filename):
        np.savez(filename if filename.strip()[-4:] == '.npz' else filename + '.npz',
                 WinKernel=self.WinKernel)

    def load_window_kernels(self, filename):
        with np.load(filename, mmap_mode='r') as data:
            self.WinKernel = data['WinKernel']

    def _compute_window_kernel_row(self, Nbin, kmodes_sampled, icut, tqdm=tqdm):
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [7,15,6] dimensions. 
        #    The first dim corresponds to the k-bin of k2 
        #    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

        I22 = self.I['22']

        W = self._Wcut

        Bin_ModeNum = self._Bin_ModeNum
        Bin_kmodes = self._Bin_kmodes
        kBinWidth = self._dk
        nBins = self.kbins
        kfun = 2*np.pi/self.BoxSize

        avgW00 = np.zeros((2*3+1, 15), dtype='<c8')
        avgW22 = avgW00.copy()
        avgW44 = avgW00.copy()
        avgW20 = avgW00.copy()
        avgW40 = avgW00.copy()
        avgW42 = avgW00.copy()

        iix, iiy, iiz = np.mgrid[-icut:icut+1,
                                 -icut:icut+1,
                                 -icut:icut+1]

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)

        if (kmodes_sampled < Bin_ModeNum[Nbin]):
            norm = kmodes_sampled
            sampled = (np.random.rand(kmodes_sampled)*Bin_ModeNum[Nbin]).astype(int)
        else:
            norm = Bin_ModeNum[Nbin]
            sampled = np.arange(Bin_ModeNum[Nbin], dtype=int)

        # Randomly select a mode in the k1 bin
        # for n in tqdm(sampled, leave=False, desc=f"Row {Nbin}"):
        for n in sampled:
            ik1x, ik1y, ik1z, rk1 = Bin_kmodes[Nbin][n]

            if rk1 == 0.:
                k1xh = 0
                k1yh = 0
                k1zh = 0
            else:
                k1xh = ik1x/rk1
                k1yh = ik1y/rk1
                k1zh = ik1z/rk1

            # Build a 3D array of modes around the selected mode   
            k2xh = ik1x-iix
            k2yh = ik1y-iiy
            k2zh = ik1z-iiz

            rk2 = np.sqrt(k2xh**2 + k2yh**2 + k2zh**2)

            sort = (rk2*kfun/kBinWidth).astype(int) - Nbin # to decide later which shell the k2 mode belongs to
            ind = (rk2 == 0)
            if ind.any() > 0:
                rk2[ind] = 1e10

            k2xh /= rk2
            k2yh /= rk2
            k2zh /= rk2
            #k2 hat arrays built

            # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
            # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
            # L(i) refers to multipoles

            W_L0 = W['22']
            Wc_L0 = np.conj(W['22'])

            xx = W['22xx']*k1xh**2 + W['22yy']*k1yh**2 + W['22zz']*k1zh**2 + 2.*W['22xy']*k1xh*k1yh + 2.*W['22yz']*k1yh*k1zh + 2.*W['22xz']*k1zh*k1xh
            W_k1L2 = 1.5*xx - 0.5*W['22']
            W_k2L2 = 1.5*(W['22xx']*k2xh**2 + W['22yy']*k2yh**2 + W['22zz']*k2zh**2
                          + 2.*W['22xy']*k2xh*k2yh + 2.*W['22yz']*k2yh*k2zh + 2.*W['22xz']*k2zh*k2xh) - 0.5*W['22']
            Wc_k1L2 = np.conj(W_k1L2)
            Wc_k2L2 = np.conj(W_k2L2)

            W_k1L4 = 35./8.*(W['22xxxx']*k1xh**4           +     W['22yyyy']*k1yh**4           +     W['22zzzz']*k1zh**4
                        + 4.*W['22xxxy']*k1xh**3*k1yh      +  4.*W['22xxxz']*k1xh**3*k1zh      +  4.*W['22xyyy']*k1yh**3*k1xh
                        + 4.*W['22yyyz']*k1yh**3*k1zh      +  4.*W['22xzzz']*k1zh**3*k1xh      +  4.*W['22yzzz']*k1zh**3*k1yh
                        + 6.*W['22xxyy']*k1xh**2*k1yh**2   +  6.*W['22xxzz']*k1xh**2*k1zh**2   +  6.*W['22yyzz']*k1yh**2*k1zh**2
                       + 12.*W['22xxyz']*k1xh**2*k1yh*k1zh + 12.*W['22xyyz']*k1yh**2*k1xh*k1zh + 12.*W['22xyzz']*k1zh**2*k1xh*k1yh) \
                - 5./2.*W_k1L2 - 7./8.*W_L0

            Wc_k1L4 = np.conj(W_k1L4)

            k1k2 = W['22xxxx']*(k1xh*k2xh)**2 + W['22yyyy']*(k1yh*k2yh)**2+W['22zzzz']*(k1zh*k2zh)**2 \
                 + W['22xxxy']*(k1xh*k1yh*k2xh**2 + k1xh**2*k2xh*k2yh)*2 \
                 + W['22xxxz']*(k1xh*k1zh*k2xh**2 + k1xh**2*k2xh*k2zh)*2 \
                 + W['22yyyz']*(k1yh*k1zh*k2yh**2 + k1yh**2*k2yh*k2zh)*2 \
                 + W['22yzzz']*(k1zh*k1yh*k2zh**2 + k1zh**2*k2zh*k2yh)*2 \
                 + W['22xyyy']*(k1yh*k1xh*k2yh**2 + k1yh**2*k2yh*k2xh)*2 \
                 + W['22xzzz']*(k1zh*k1xh*k2zh**2 + k1zh**2*k2zh*k2xh)*2 \
                 + W['22xxyy']*(k1xh**2*k2yh**2   + k1yh**2*k2xh**2   + 4.*k1xh*k1yh*k2xh*k2yh) \
                 + W['22xxzz']*(k1xh**2*k2zh**2   + k1zh**2*k2xh**2   + 4.*k1xh*k1zh*k2xh*k2zh) \
                 + W['22yyzz']*(k1yh**2*k2zh**2   + k1zh**2*k2yh**2   + 4.*k1yh*k1zh*k2yh*k2zh) \
                 + W['22xyyz']*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                 + W['22xxyz']*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                 + W['22xyzz']*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh*k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            W_k2L4 = 35./8.*(W['22xxxx']*k2xh**4           +     W['22yyyy']*k2yh**4           +     W['22zzzz']*k2zh**4
                             +  4.*W['22xxxy']*k2xh**3*k2yh      +  4.*W['22xxxz']*k2xh**3*k2zh      +  4.*W['22xyyy']*k2yh**3*k2xh
                             +  4.*W['22yyyz']*k2yh**3*k2zh      +  4.*W['22xzzz']*k2zh**3*k2xh      +  4.*W['22yzzz']*k2zh**3*k2yh
                             +  6.*W['22xxyy']*k2xh**2*k2yh**2   +  6.*W['22xxzz']*k2xh**2*k2zh**2   +  6.*W['22yyzz']*k2yh**2*k2zh**2
                             + 12.*W['22xxyz']*k2xh**2*k2yh*k2zh + 12.*W['22xyyz']*k2yh**2*k2xh*k2zh + 12.*W['22xyzz']*k2zh**2*k2xh*k2yh) \
                - 5./2.*W_k2L2 -7./8.*W_L0

            Wc_k2L4 = np.conj(W_k2L4)

            W_k1L2_k2L2 = 9./4.*k1k2 - 3./4.*xx - 1./2.*W_k2L2
            W_k1L2_k2L4 = 2/7.*W_k1L2 + 20/77.*W_k1L4 # approximate as 6th order FFTs not simulated
            W_k1L4_k2L2 = W_k1L2_k2L4 # approximate
            W_k1L4_k2L4 = 1/9.*W_L0 + 100/693.*W_k1L2 + 162/1001.*W_k1L4

            Wc_k1L2_k2L2 = np.conj(W_k1L2_k2L2)
            Wc_k1L2_k2L4 = np.conj(W_k1L2_k2L4)
            Wc_k1L4_k2L2 = Wc_k1L2_k2L4
            Wc_k1L4_k2L4 = np.conj(W_k1L4_k2L4)

            k1k2W12 = W['12xxxx']*(k1xh*k2xh)**2 + W['12yyyy']*(k1yh*k2yh)**2 + W['12zzzz']*(k1zh*k2zh)**2 \
                    + W['12xxxy']*(k1xh*k1yh*k2xh**2 + k1xh**2*k2xh*k2yh)*2 \
                    + W['12xxxz']*(k1xh*k1zh*k2xh**2 + k1xh**2*k2xh*k2zh)*2 \
                    + W['12yyyz']*(k1yh*k1zh*k2yh**2 + k1yh**2*k2yh*k2zh)*2 \
                    + W['12yzzz']*(k1zh*k1yh*k2zh**2 + k1zh**2*k2zh*k2yh)*2 \
                    + W['12xyyy']*(k1yh*k1xh*k2yh**2 + k1yh**2*k2yh*k2xh)*2 \
                    + W['12xzzz']*(k1zh*k1xh*k2zh**2 + k1zh**2*k2zh*k2xh)*2 \
                    + W['12xxyy']*(k1xh**2*k2yh**2   + k1yh**2*k2xh**2   + 4.*k1xh*k1yh*k2xh*k2yh) \
                    + W['12xxzz']*(k1xh**2*k2zh**2   + k1zh**2*k2xh**2   + 4.*k1xh*k1zh*k2xh*k2zh) \
                    + W['12yyzz']*(k1yh**2*k2zh**2   + k1zh**2*k2yh**2   + 4.*k1yh*k1zh*k2yh*k2zh) \
                    + W['12xyyz']*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                    + W['12xxyz']*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                    + W['12xyzz']*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh*k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            xxW12 =     W['12xx']*k1xh**2   +    W['12yy']*k1yh**2   +    W['12zz']*k1zh**2 \
                   + 2.*W['12xy']*k1xh*k1yh + 2.*W['12yz']*k1yh*k1zh + 2.*W['12xz']*k1zh*k1xh

            W12_L0 = W['12']
            W12_k1L2 = 1.5*xxW12 - 0.5*W['12']
            W12_k1L4 = 35./8.*(W['12xxxx']*k1xh**4           +     W['12yyyy']*k1yh**4            +    W['12zzzz']*k1zh**4
                          + 4.*W['12xxxy']*k1xh**3*k1yh      +  4.*W['12xxxz']*k1xh**3*k1zh       + 4.*W['12xyyy']*k1yh**3*k1xh
                          + 6.*W['12xxyy']*k1xh**2*k1yh**2   +  6.*W['12xxzz']*k1xh**2*k1zh**2    + 6.*W['12yyzz']*k1yh**2*k1zh**2
                         + 12.*W['12xxyz']*k1xh**2*k1yh*k1zh + 12.*W['12xyyz']*k1yh**2*k1xh*k1zh + 12.*W['12xyzz']*k1zh**2*k1xh*k1yh) \
                    - 5./2.*W12_k1L2 - 7./8.*W12_L0

            W12_k1L4_k2L2 = 2/7.*W12_k1L2 + 20/77.*W12_k1L4
            W12_k1L4_k2L4 = 1/9.*W12_L0 + 100/693.*W12_k1L2 + 162/1001.*W12_k1L4

            W12_k2L2 = 1.5*( W['12xx']*k2xh**2   +    W['12yy']*k2yh**2   +    W['12zz']*k2zh**2
                        + 2.*W['12xy']*k2xh*k2yh + 2.*W['12yz']*k2yh*k2zh + 2.*W['12xz']*k2zh*k2xh) - 0.5*W['12']

            W12_k2L4 = 35./8.*(W['12xxxx']*k2xh**4           +     W['12yyyy']*k2yh**4           +     W['12zzzz']*k2zh**4
                          + 4.*W['12xxxy']*k2xh**3*k2yh      +  4.*W['12xxxz']*k2xh**3*k2zh      +  4.*W['12xyyy']*k2yh**3*k2xh
                          + 4.*W['12yyyz']*k2yh**3*k2zh      +  4.*W['12xzzz']*k2zh**3*k2xh      +  4.*W['12yzzz']*k2zh**3*k2yh
                          + 6.*W['12xxyy']*k2xh**2*k2yh**2   +  6.*W['12xxzz']*k2xh**2*k2zh**2   +  6.*W['12yyzz']*k2yh**2*k2zh**2
                         + 12.*W['12xxyz']*k2xh**2*k2yh*k2zh + 12.*W['12xyyz']*k2yh**2*k2xh*k2zh + 12.*W['12xyzz']*k2zh**2*k2xh*k2yh) \
                    - 5./2.*W12_k2L2 - 7./8.*W12_L0

            W12_k1L2_k2L2 = 9./4.*k1k2W12 -3./4.*xxW12 - 1./2.*W12_k2L2

            W_k1L2_Sumk2L22 = 1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24 = 2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22 = 1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24 = 2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44 = 1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4

            C00exp = [Wc_L0  *W_L0, Wc_L0  *W_k2L2, Wc_L0  *W_k2L4,
                      Wc_k1L2*W_L0, Wc_k1L2*W_k2L2, Wc_k1L2*W_k2L4,
                      Wc_k1L4*W_L0, Wc_k1L4*W_k2L2, Wc_k1L4*W_k2L4]

            C00exp += [2.*W_L0  *W12_L0, W_k1L2*W12_L0,         W_k1L4 *W12_L0,
                          W_k2L2*W12_L0, W_k2L4*W12_L0, np.conj(W12_L0)*W12_L0]

            C22exp = [Wc_k2L2*W_k1L2           + Wc_L0*W_k1L2_k2L2,
                      Wc_k2L2*W_k1L2_k2L2      + Wc_L0*W_k1L2_Sumk2L22,
                      Wc_k2L2*W_k1L2_k2L4      + Wc_L0*W_k1L2_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L2      + Wc_k1L2*W_k1L2_k2L2,
                      Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L2      + Wc_k1L4*W_k1L2_k2L2,
                      Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]

            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2 + W_k1L2_k2L2*W12_L0+W_L0*W12_k1L2_k2L2,

                       0.5*((1/5.*W_L0+2/7.*W_k1L2 + 18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2
                          + (1/5.*W_k2L2+2/7.*W_k1L2_k2L2 + 18/35.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L2_k2L2),

                       0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2
                          + (2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L2*W12_k2L2 + (1/5.*W_L0 + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L2
                          + (1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4)*W12_L0 + W_k2L2*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L4*W12_k2L2 + (2/7.*W_k2L2      + 20/77.*W_k2L4     )*W12_k1L2
                          + W_k2L4*W12_k1L2_k2L2 + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4)*W12_L0),

                       np.conj(W12_k1L2_k2L2)*W12_L0 + np.conj(W12_k1L2)*W12_k2L2]

            C44exp = [Wc_k2L4     *W_k1L4      + Wc_L0  *W_k1L4_k2L4,
                      Wc_k2L4     *W_k1L4_k2L2 + Wc_L0  *W_k1L4_Sumk2L24,
                      Wc_k2L4     *W_k1L4_k2L4 + Wc_L0  *W_k1L4_Sumk2L44,
                      Wc_k1L2_k2L4*W_k1L4      + Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,
                      Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,
                      Wc_k1L4_k2L4*W_k1L4      + Wc_k1L4*W_k1L4_k2L4,
                      Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,
                      Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]

            C44exp += [  W_k1L4     *W12_k2L4 + W_k2L4*W12_k1L4 \
                       + W_k1L4_k2L4*W12_L0   + W_L0  *W12_k1L4_k2L4,

                       0.5*((2/7.*W_k1L2      + 20/77.*W_k1L4     )*W12_k2L4 + W_k1L2_k2L4*W12_k1L4
                            + (2/7.*W_k1L2_k2L4 + 20/77.*W_k1L4_k2L4)*W12_L0   + W_k1L2     *W12_k1L4_k2L4),

                       0.5*((1/9.*W_L0   + 100/693.*W_k1L2      + 162/1001.*W_k1L4     )*W12_k2L4 + W_k1L4_k2L4*W12_k1L4
                            + (1/9.*W_k2L4 + 100/693.*W_k1L2_k2L4 + 162/1001.*W_k1L4_k2L4)*W12_L0   + W_k1L4     *W12_k1L4_k2L4),

                       0.5*(W_k1L4_k2L2*W12_k2L4 + (2/7.*W_k2L2      + 20/77.*W_k2L4     )*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L4 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L4 + (1/9.*W_L0   + 100/693.*W_k2L2      + 162/1001.*W_k2L4     )*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L4 + (1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4)*W12_L0 ),

                       np.conj(W12_k1L4_k2L4)*W12_L0 + np.conj(W12_k1L4)*W12_k2L4] #1/(nbar)^2

            C20exp = [Wc_L0  *W_k1L2,   Wc_L0*W_k1L2_k2L2, Wc_L0  *W_k1L2_k2L4,
                      Wc_k1L2*W_k1L2, Wc_k1L2*W_k1L2_k2L2, Wc_k1L2*W_k1L2_k2L4,
                      Wc_k1L4*W_k1L2, Wc_k1L4*W_k1L2_k2L2, Wc_k1L4*W_k1L2_k2L4]

            C20exp += [W_k1L2*W12_L0 + W['22']*W12_k1L2,
                       0.5*((1/5.*W['22'] + 2/7.*W_k1L2 + 18/35.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L2),
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L2),
                       0.5*(W_k1L2_k2L2*W12_L0 + W_k2L2*W12_k1L2),
                       0.5*(W_k1L2_k2L4*W12_L0 + W_k2L4*W12_k1L2),
                       np.conj(W12_k1L2)*W12_L0]

            C40exp = [Wc_L0*W_k1L4,   Wc_L0  *W_k1L4_k2L2, Wc_L0  *W_k1L4_k2L4,
                      Wc_k1L2*W_k1L4, Wc_k1L2*W_k1L4_k2L2, Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L4*W_k1L4, Wc_k1L4*W_k1L4_k2L2, Wc_k1L4*W_k1L4_k2L4]

            C40exp += [W_k1L4*W12_L0 + W['22']*W12_k1L4,
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L4),
                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2+162/1001.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L4),
                       0.5*(W_k1L4_k2L2*W12_L0 + W_k2L2*W12_k1L4),
                       0.5*(W_k1L4_k2L4*W12_L0 + W_k2L4*W12_k1L4),
                       np.conj(W12_k1L4)*W12_L0]

            C42exp = [Wc_k2L2*W_k1L4           + Wc_L0  *W_k1L4_k2L2,
                      Wc_k2L2*W_k1L4_k2L2      + Wc_L0  *W_k1L4_Sumk2L22,
                      Wc_k2L2*W_k1L4_k2L4      + Wc_L0  *W_k1L4_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L4      + Wc_k1L2*W_k1L4_k2L2,
                      Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L4      + Wc_k1L4*W_k1L4_k2L2,
                      Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]

            C42exp += [ W_k1L4*W12_k2L2    + W_k2L2*W12_k1L4 \
                      + W_k1L4_k2L2*W12_L0 + W['22']*W12_k1L4_k2L2,

                    0.5*((2/7.*W_k1L2      + 20/77.*W_k1L4     )*W12_k2L2 + W_k1L2_k2L2*W12_k1L4
                        +(2/7.*W_k1L2_k2L2 + 20/77.*W_k1L4_k2L2)*W12_L0   + W_k1L2     *W12_k1L4_k2L2),

                    0.5*((1/9.*W['22'] + 100/693.*W_k1L2      + 162/1001.*W_k1L4     )*W12_k2L2 + W_k1L4_k2L2*W12_k1L4
                        +(1/9.*W_k2L2  + 100/693.*W_k1L2_k2L2 + 162/1001.*W_k1L4_k2L2)*W12_L0   + W_k1L4*W12_k1L4_k2L2),

                    0.5*(W_k1L4_k2L2*W12_k2L2 + (1/5.*W['22'] + 2/7.*W_k2L2      + 18/35.*W_k2L4     )*W12_k1L4
                       + W_k2L2*W12_k1L4_k2L2 + (1/5.*W_k1L4  + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4)*W12_L0),

                    0.5*(W_k1L4_k2L4*W12_k2L2 + (2/7.*W_k2L2      + 20/77.*W_k2L4     )*W12_k1L4
                       + W_k2L4*W12_k1L4_k2L2 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12_L0),

                    np.conj(W12_k1L4_k2L2)*W12_L0+np.conj(W12_k1L4)*W12_k2L2] #1/(nbar)^2

            for i in range(-3,4):
                ind = (sort == i)
                for j in range(15):
                    avgW00[i+3,j] += np.sum(C00exp[j][ind])
                    avgW22[i+3,j] += np.sum(C22exp[j][ind])
                    avgW44[i+3,j] += np.sum(C44exp[j][ind])
                    avgW20[i+3,j] += np.sum(C20exp[j][ind])
                    avgW40[i+3,j] += np.sum(C40exp[j][ind])
                    avgW42[i+3,j] += np.sum(C42exp[j][ind])

        for i in range(0, 2*3+1):
            if(i+Nbin-3 >= nBins or i+Nbin-3 < 0):
                avgW00[i]*=0
                avgW22[i]*=0
                avgW44[i]*=0
                avgW20[i]*=0
                avgW40[i]*=0
                avgW42[i]*=0
            else:
                avgW00[i] = avgW00[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
                avgW22[i] = avgW22[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
                avgW44[i] = avgW44[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
                avgW20[i] = avgW20[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
                avgW40[i] = avgW40[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
                avgW42[i] = avgW42[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)

        l_factor = lambda l1,l2: (2*l1+1) * (2*l2+1) * (2 if 0 in (l1,l2) else 1)

        avgWij = np.zeros((7, 15, 6))

        avgWij[:,:,0] = l_factor(0,0)*np.real(avgW00)
        avgWij[:,:,1] = l_factor(2,2)*np.real(avgW22)
        avgWij[:,:,2] = l_factor(4,4)*np.real(avgW44)
        avgWij[:,:,3] = l_factor(2,0)*np.real(avgW20)
        avgWij[:,:,4] = l_factor(4,0)*np.real(avgW40)
        avgWij[:,:,5] = l_factor(4,2)*np.real(avgW42)

        return avgWij

    def _format_fft(self, fourier, window):

        # PFFT omits modes that are determined by the Hermitian condition. Adding them:
        fourier_full = utils.r2c_to_c2c_3d(fourier)

        if window == 'W12':
            fourier_full = np.conj(fourier_full)

        return fft.fftshift(fourier_full)[::-1,::-1,::-1] * self.ngals

    @property
    def I(self):
        return self._I

    @property
    def W(self):
        return self._W

    @property
    def randoms(self):
        return self._randoms

    def compute_covariance(self):
        self._Pfit = {key: self._P[key](self._k) for key in self._P.keys()}

        Cov = np.array([self._compute_gaussian_covariance_row(ki) for ki, k in enumerate(self._k)])

        self._multipole_covariance = {
            (0,0): Covariance(Cov[:,:,0]),
            (2,2): Covariance(Cov[:,:,1]),
            (4,4): Covariance(Cov[:,:,2]),
            (0,2): Covariance(Cov[:,:,3]),
            (0,4): Covariance(Cov[:,:,4]),
            (2,4): Covariance(Cov[:,:,5]),
        }

        C = {k: self._multipole_covariance[k].cov for k in self._multipole_covariance.keys()}

        self._covariance = np.block([
            [C[0,0], C[0,2], C[0,4]],
            [C[0,2], C[2,2], C[2,4]],
            [C[0,4], C[2,4], C[4,4]],
        ])

    def _compute_gaussian_covariance_row(self, ki):

        kbins = self.kbins
        assert ki < kbins
        Win = self.WinKernel[ki]

        Pfit = self._Pfit

        Crow = np.zeros((kbins, 6))

        for kj in range(max(ki-3, 0), min(ki+4, kbins)):
            j = kj - ki + 3

            Crow[kj] = \
                Win[j,0]*Pfit[0][ki]*Pfit[0][kj] + \
                Win[j,1]*Pfit[0][ki]*Pfit[2][kj] + \
                Win[j,2]*Pfit[0][ki]*Pfit[4][kj] + \
                Win[j,3]*Pfit[2][ki]*Pfit[0][kj] + \
                Win[j,4]*Pfit[2][ki]*Pfit[2][kj] + \
                Win[j,5]*Pfit[2][ki]*Pfit[4][kj] + \
                Win[j,6]*Pfit[4][ki]*Pfit[0][kj] + \
                Win[j,7]*Pfit[4][ki]*Pfit[2][kj] + \
                Win[j,8]*Pfit[4][ki]*Pfit[4][kj] + \
                1.01*(
                    Win[j,9]*(Pfit[0][ki] +           Pfit[0][kj])/2. +
                    Win[j,10]*Pfit[2][ki] + Win[j,11]*Pfit[4][ki] +
                    Win[j,12]*Pfit[2][kj] + Win[j,13]*Pfit[4][kj]
                ) + \
                1.01**2*Win[j,14]

        return Crow

class TrispectrumSurveyWindowCovariance(PowerMultipoleCovariance):

    def __init__(self, gaussian_cov=None):
        super().__init__()
        self._covariance = None
        self.T0 = None
        self.Plin = None

        self._gaussian_cov = gaussian_cov

    def set_bias_parameters(self, b1, be, g2, b2, g3, g2x, g21, b3):
        self.T0 = Trispectrum.T0(b1, be, g2, b2, g3, g2x, g21, b3)

    def load_Plin(self, k, P):
        self.Plin = interpolate.InterpolatedUnivariateSpline(k, P)

    def _trispectrum_element(self, l1, l2, k1, k2):

        # We'll still implement hexadecupole in trispectrum.
        # For now, return 0 
        if 4 in (l1,l2): 
            return 0

        # Returns the tree-level trispectrum as a function of multipoles and k1, k2
        T0 = self.T0
        Plin = self.Plin

        T0.l1 = l1
        T0.l2 = l2

        trisp_integrand = np.vectorize(self._trispectrum_integrand)

        expr =    self.I['44']*(Plin(k1)**2*Plin(k2)*T0.ez3(k1,k2) + Plin(k2)**2*Plin(k1)*T0.ez3(k2,k1))\
              + 8*self.I['34']* Plin(k1)*Plin(k2)*T0.e34o44_1(k1,k2)

        return (integrate.quad(trisp_integrand, -1, 1, args=(k1,k2))[0]/2. + expr)/self.I['22']**2

    def _trispectrum_integrand(self, u12, k1, k2):

        Plin = self.Plin
        T0   = self.T0

        return  (    8*self.I['44']*(Plin(k1)**2*T0.e44o44_1(u12,k1,k2) + Plin(k2)**2*T0.e44o44_1(u12,k2,k1))
                  + 16*self.I['44']*Plin(k1)*Plin(k2)*T0.e44o44_2(u12,k1,k2)
                  +  8*self.I['34']*(Plin(k1)*T0.e34o44_2(u12,k1,k2) + Plin(k2)*T0.e34o44_2(u12,k2,k1))
                  +  2*self.I['24']*T0.e24o44(u12,k1,k2)
                ) * Plin(np.sqrt(k1**2+k2**2+2*k1*k2*u12))

    def compute_covariance(self, tqdm=tqdm):

        kbins = self._gaussian_cov.kbins
        k = self._gaussian_cov.k

        trisp = np.vectorize(self._trispectrum_element)

        covaT0mult = np.zeros(2*[3*kbins])

        for i in tqdm(range(kbins), total=kbins, desc="Computing trispectrum contribution"):
            covaT0mult[i,        :  kbins] = trisp(0, 0, k[i], k)
            covaT0mult[i,   kbins:2*kbins] = trisp(0, 2, k[i], k)
            covaT0mult[i, 2*kbins:3*kbins] = trisp(0, 4, k[i], k)
            
            covaT0mult[kbins+i,   kbins:2*kbins] = trisp(2, 2, k[i], k)
            covaT0mult[kbins+i, 2*kbins:3*kbins] = trisp(2, 4, k[i], k)

            covaT0mult[2*kbins+i, 2*kbins:3*kbins] = trisp(4, 4, k[i], k)

        covaT0mult[kbins:, :kbins] = np.transpose(covaT0mult[:kbins, kbins:])

        self._covariance = covaT0mult

    @property
    def I(self):
        return self._gaussian_cov.I