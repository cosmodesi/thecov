
from abc import ABC, abstractmethod
import itertools as itt

import multiprocessing as mp

import numpy as np
import dask.array as da
from scipy import fft
from nbodykit.algorithms import zhist

from tqdm import tqdm

from . import base, utils

# Window functions needed for Gaussian covariance calculation
W_LABELS = ['12', '12xx', '12xy', '12xz', '12yy', '12yz', '12zz', '12xxxx', '12xxxy', '12xxxz', '12xxyy', '12xxyz', '12xxzz', '12xyyy', '12xyyz', '12xyzz', '12xzzz', '12yyyy', '12yyyz', '12yyzz', '12yzzz', '12zzzz', '22', '22xx', '22xy', '22xz', '22yy', '22yz', '22zz', '22xxxx', '22xxxy', '22xxxz', '22xxyy', '22xxyz', '22xxzz', '22xyyy', '22xyyz', '22xyzz', '22xzzz', '22yyyy', '22yyyz', '22yyzz', '22yzzz', '22zzzz']

class Geometry(ABC):
    pass

class BoxGeometry(Geometry):
    def __init__(self, volume=None, nbar=None):
        self._volume = volume
        self._nbar = nbar
        self._zmin = None
        self._zmax = None
        self.fsky = 1.0

    @property
    def volume(self):
        return self._volume
    
    @volume.setter
    def volume(self, volume):
        self._volume = volume

    @property
    def nbar(self):
        return self._nbar

    @nbar.setter
    def nbar(self, nbar):
        self._nbar = nbar

    @property
    def shotnoise(self):
        return 1/self.nbar

    @shotnoise.setter
    def shotnoise(self, shotnoise):
        self.nbar = 1./shotnoise

    @property
    def zedges(self):
        return self._zedges

    @property
    def zmid(self):
        return (self.zedges[1:] + self.zedges[:-1])/2
    
    @property
    def zmin(self):
        return self._zmin if self._zmin is not None else self.zedges[0]

    @property
    def zmax(self):
        return self._zmax if self._zmax is not None else self.zedges[-1]

    @property
    def zavg(self):
        bin_volume = np.diff(self.cosmo.comoving_distance(self.zedges)**3)
        return np.average(self.zmid, weights=self.nz * bin_volume)

    def set_effective_volume(self, zmin, zmax, fsky=None):

        if fsky is not None:
            self.fsky = fsky

        self._zmin = zmin
        self._zmax = zmax
            
        self.volume = self.fsky * 4/3 * np.pi * (self.cosmo.comoving_distance(zmax)**3 - self.cosmo.comoving_distance(zmin)**3)

        return self.volume

    def set_nz(self, zedges, nz, *args, **kwargs):
        assert len(zedges) == len(nz) + 1, "Length of zedges should equal length of nz + 1."
        
        self._zedges = np.array(zedges)
        self._nz = np.array(nz)[np.argsort(self.zmid)]
        self._zedges.sort()

        self.set_effective_volume(zmin=self.zmin, zmax=self.zmax, *args, **kwargs)
        print(f'Effective volume: {self.volume:.3e} (Mpc/h)^3')

        bin_volume = self.fsky * np.diff(self.cosmo.comoving_distance(self.zedges)**3)
        self.nbar = np.average(self.nz, weights=bin_volume)
        print(f'Estimated nbar: {self.nbar:.3e} (Mpc/h)^-3')

    def set_randoms(self, randoms, alpha=1.0):
        import healpy as hp

        nside = 512
        hpixel = hp.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)
        unique_hpixels  = np.unique(hpixel)
        self.fsky = len(unique_hpixels.compute())/hp.nside2npix(nside)

        print(f'fsky estimated from randoms: {self.fsky:.3f}')

        nz_hist = zhist.RedshiftHistogram(randoms, self.fsky, self.cosmo, redshift='Z')
        self.set_nz(zedges=nz_hist.bin_edges, nz=nz_hist.nbar*alpha)

    @property
    def area(self):
        return self.fsky * 360**2 / np.pi
    
    @area.setter
    def area(self, area):
        self.fsky = area / 360**2 * np.pi

    @property
    def nz(self):
        return self._nz

    @property
    def ngals(self):
        return self.nbar * self.volume
    
    @property
    def cosmo(self):
        if not hasattr(self, '_cosmo'):
            print('Cosmology object not set. Using default cosmology.')
            from nbodykit.lab import cosmology
            self._cosmo = cosmology.Cosmology(h=0.7).match(Omega0_m=0.31)
        return self._cosmo
    
    @cosmo.setter
    def cosmo(self, cosmo):
        self._cosmo = cosmo
    


class SurveyGeometry(Geometry, base.FourierBinned):

    def __init__(self, random_catalog=None, Nmesh=None, BoxSize=None, k2_range=3, kmodes_sampled=250, alpha=1.0, mesh_kwargs=None, tqdm=tqdm):

        base.FourierBinned.__init__(self)

        self.Nmesh = Nmesh
        self.alpha = alpha
        self.k2_range = k2_range
        self.kmodes_sampled = kmodes_sampled

        if np.ndim(BoxSize) == 0:
            self.BoxSize = 3*[BoxSize]
        elif np.ndim(BoxSize) == 1:
            self.BoxSize = BoxSize

        if tqdm is not None:
            self.tqdm = tqdm
        else:
            def tqdm(x): return x

        self._mesh_kwargs = {
            'Nmesh':       self.Nmesh,
            'BoxSize':     self.BoxSize,
            'interlaced':  True,
            'compensated': True,
            'resampler':   'tsc',
        }

        if mesh_kwargs is not None:
            self._mesh_kwargs.update(mesh_kwargs)

        if random_catalog is not None:
            self._randoms = random_catalog

            self._randoms['RelativePosition'] = self._randoms['Position']
            self._randoms['Position'] += da.array(self.BoxSize)/2

            self._ngals = self.randoms.size * self.alpha
        else:
            self._ngals = None

        self._W = {}
        self._I = {}

        # for i,j in ['22', '11', '12', '10', '24', '14', '34', '44', '32']:
        #     self._randoms[f'W{i}{j}'] = self._randoms['NZ']**(int(i)-1) * self._randoms['WEIGHT_FKP']**int(j)
        #     # Computing I_ij integrals
        #     self._I[f'{i}{j}'] = (self._randoms[f'W{i}{j}'].sum() * self.alpha).compute()


    def W_cat(self, W):
        w = W.lower().replace("w", "")

        if f'W{w}' not in self._randoms.columns:
            self._randoms[f'W{w}'] = self._randoms['NZ']**(int(w[0])-1) * self._randoms['WEIGHT_FKP']**int(w[1])
        return self._randoms[f'W{w}']

    def I(self, W):
        w = W.lower().replace("i", "").replace("w", "")
        if w not in self._I:
            self.W_cat(w)
            self._I[w] = (self._randoms[f'W{w}'].sum() * self.alpha).compute()
        return self._I[w]

    def W(self, W):
        w = W.lower().replace("w", "")
        if w not in self._W:
            self.W_cat(w)
            self.compute_cartesian_fft(w.replace("x", "").replace("y", "").replace("z", ""))
        return self._W[w]

    def compute_cartesian_ffts(self, Wij=('W12', 'W22')):
        [self.W(w) for w in Wij]

    def compute_cartesian_fft(self, W, tqdm=tqdm):

        w = W.lower().replace('w', '')

        mesh_kwargs = self._mesh_kwargs

        x = self.randoms['RelativePosition'].T
        
        with tqdm(total=22, desc=f'Computing moments of W{w}') as pbar:
            self.set_cartesian_fft(f'W{w}', self._format_fft(self.randoms.to_mesh(
                value=f'W{w}', **mesh_kwargs).paint(mode='complex'), w))

            pbar.update(1)

            for (i, i_label), (j, j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
                label = f'W{w}{i_label}{j_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * \
                    x[i]*x[j] / (x[0]**2 + x[1]**2 + x[2]**2)
                self.set_cartesian_fft(label, self._format_fft(self.randoms.to_mesh(
                    value=label, **mesh_kwargs).paint(mode='complex'), w))

                pbar.update(1)

            for (i, i_label), (j, j_label), (k, k_label), (l, l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
                label = f'W{w}{i_label}{j_label}{k_label}{l_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * x[i] * \
                    x[j]*x[k]*x[l] / (x[0]**2 + x[1]**2 + x[2]**2)**2
                self.set_cartesian_fft(label, self._format_fft(self.randoms.to_mesh(
                    value=label, **mesh_kwargs).paint(mode='complex'), w))

                pbar.update(1)

    def compute_cartesian_fft_single_mesh(self, W):

        w = W.replace('W', '').replace('w', '')
        self.W_cat(w)

        mesh_kwargs = self._mesh_kwargs

        real_field = self.randoms.to_mesh(value='W12', **mesh_kwargs).paint(mode='real')

        self._W[w] = self._format_fft(real_field.r2c(), w)

        for (i, i_label), (j, j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
            self._W[f'{w}{i_label}{j_label}'] = self._format_fft(real_field.apply(lambda x,v: np.nan_to_num(v*x[i]*x[j]/(x[0]**2 + x[1]**2 + x[2]**2)), kind='relative').r2c(), w)

        for (i, i_label), (j, j_label), (k, k_label), (l, l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
            self._W[f'{w}{i_label}{j_label}{k_label}{l_label}'] = self._format_fft(real_field.apply(lambda x,v: np.nan_to_num(v*x[i]*x[j]*x[k]*x[l]/(x[0]**2 + x[1]**2 + x[2]**2)**2), kind='relative').r2c(), w)


    def _format_fft(self, fourier, window):

        # PFFT omits modes that are determined by the Hermitian condition. Adding them:
        fourier_full = utils.r2c_to_c2c_3d(fourier)

        if window == '12':
            fourier_full = np.conj(fourier_full)

        return fft.fftshift(fourier_full)[::-1, ::-1, ::-1] * self.ngals

    def _unformat_fft(self, fourier, window):

        fourier_cut = fft.ifftshift(
            fourier[::-1, ::-1, ::-1])[:, :, :fourier.shape[2]//2+1] / self.ngals

        return np.conj(fourier_cut) if window == '12' else fourier_cut
    
    def set_cartesian_fft(self, label, W):
        w = label.lower().replace('w', '')
        if w not in self._W:
            # Create object proper for multiprocessing
            self._W[w] = np.frombuffer(mp.RawArray('d', 2*self.Nmesh**3))\
                           .view(np.complex128)\
                           .reshape(*3*[self.Nmesh])
        self._W[w][:,:,:] = W

    def save_cartesian_ffts(self, filename):
        np.savez(filename if filename.strip()[-4:] == '.npz' else f'{filename}.npz', **{f'W{k.replace("W","")}': self._W[k] for k in self._W}, **{
                 f'I{k.replace("I","")}': self._I[k] for k in self._I}, BoxSize=self.BoxSize, Nmesh=self.Nmesh, alpha=self.alpha)

    def load_cartesian_ffts(self, filename):
        with np.load(filename, mmap_mode='r') as data:

            self.BoxSize = data['BoxSize']
            self.Nmesh = data['Nmesh']
            self.alpha = data['alpha']
            
            for f in data.files:
                if f[0] == 'W':
                    self.set_cartesian_fft(f, data[f])
                elif f[0] == 'I':  
                    self._I[f[1:]] = data[f]

            self._mesh_kwargs = {
                'Nmesh':       self.Nmesh,
                'BoxSize':     self.BoxSize,
                'interlaced':  True,
                'compensated': True,
                'resampler':   'tsc',
            }

    @property
    def randoms(self):
        return self._randoms

    @property
    def ngals(self):
        return self._ngals
    @property
    def kbins(self):
        return len(self.kmid)

    def get_window_kernels(self):
        if not (hasattr(self, 'WinKernel') and self.WinKernel is not None):
            self.compute_window_kernels()
        return self.WinKernel

    # As the window falls steeply with k, only low-k regions are needed for the calculation.
    # Therefore, high-k modes in the FFTs can be discarted
    def compute_window_kernels(self, tqdm=tqdm):

        # Make sure these FFTs are computed
        # self.W('12')
        # self.W('22')

        # Recording the k-modes in different shells
        # Bin_kmodes contains [kx,ky,kz,radius] values of all the modes in the bin

        box_volume = np.prod(np.array(self.BoxSize, dtype=float))
        kfun = 2*np.pi/box_volume**(1/3)

        kmodes = np.array([[utils.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(self.kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])

        for w in W_LABELS:
            self._W[w] = np.frombuffer(mp.RawArray('d', 2*self.Nmesh**3))\
                           .view(np.complex128)\
                           .reshape(*3*[self.Nmesh])

        self.compute_cartesian_fft('W12')
        self.compute_cartesian_fft('W22')

        init_params = {
            'I22': self.I('22'),
            'Volume': np.prod(np.array(self.BoxSize, dtype=float)),
            'kbins': self.kbins,
            'dk': self.dk,
            'Nmesh': self.Nmesh,
            'k2_range': self.k2_range,
            'kedges': self.kedges
        }

        def init_worker(*args):
            global shared_w
            global shared_params
            shared_w = {}
            for w,l in zip(args, W_LABELS):
                shared_w[l] = np.frombuffer(w).view(np.complex128).reshape(*3*[self.Nmesh])
            shared_params = args[-1]

        with mp.Pool(initializer=init_worker, initargs=[*[self._W[w] for w in W_LABELS], init_params]) as pool:
            self.WinKernel = np.array(list(tqdm(pool.imap(self._compute_window_kernel_row, enumerate(kmodes)), total=self.kbins, desc="Computing window kernels")))

        # self.WinKernel = np.array([
        #     self._compute_window_kernel_row(Nbin)
        #     for Nbin in tqdm(range(self.kbins), total=self.kbins, desc="Computing window kernels")])

    def save_window_kernels(self, filename):
        np.savez(filename if filename.strip()
                 [-4:] == '.npz' else f'{filename}.npz', WinKernel=self.WinKernel, kmin=self.kmin, kmax=self.kmax, dk=self.dk)

    def load_window_kernels(self, filename):
        with np.load(filename, mmap_mode='r') as data:
            self.WinKernel = data['WinKernel']
            self.set_kbins(data['kmin'], data['kmax'], data['dk'])

    @staticmethod
    def _compute_window_kernel_row(args):
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [2*k2_range+1,15,6] dimensions.
        #    The first dim corresponds to the k-bin of k2
        #    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]
        
        Nbin, Bin_kmodes = args

        I22        = shared_params['I22']
        k2_range   = shared_params['k2_range']
        box_volume = shared_params['Volume']
        nBins      = shared_params['kbins']
        kBinWidth  = shared_params['dk']
        Nmesh      = shared_params['Nmesh']
        kedges     = shared_params['kedges']

        W = shared_w

        # I22 = self.I('22')

        # W = self._W

        # box_volume = np.prod(np.array(self.BoxSize, dtype=float))

        Bin_ModeNum = utils.nmodes(box_volume, kedges[:-1], kedges[1:])
        # Bin_kmodes = self._kmodes

        # kBinWidth = self.dk
        # nBins = self.kbins
        kfun = 2*np.pi/box_volume**(1/3)

        # The Gaussian covariance drops quickly away from diagonal.
        # Only k2_range points to each side of the diagonal are calculated.
        # k2_range = self.k2_range

        # Nmesh = self.Nmesh

        # Worker stuff

        avgW00 = np.zeros((2*k2_range+1, 15), dtype='<c8')
        avgW22 = avgW00.copy()
        avgW44 = avgW00.copy()
        avgW20 = avgW00.copy()
        avgW40 = avgW00.copy()
        avgW42 = avgW00.copy()

        iix, iiy, iiz = np.mgrid[-Nmesh//2+1:Nmesh//2+1,
                                 -Nmesh//2+1:Nmesh//2+1,
                                 -Nmesh//2+1:Nmesh//2+1]

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)

        kmodes_sampled = len(Bin_kmodes)

        for ik1x, ik1y, ik1z, rk1 in Bin_kmodes:

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

            # to decide later which shell the k2 mode belongs to
            sort = (rk2*kfun/kBinWidth).astype(int) - Nbin
            ind = (rk2 == 0)
            if ind.any() > 0:
                rk2[ind] = 1e10

            k2xh /= rk2
            k2yh /= rk2
            k2zh /= rk2
            # k2 hat arrays built

            # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
            # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
            # L(i) refers to multipoles

            W_L0 = W['22']
            Wc_L0 = np.conj(W['22'])

            xx = W['22xx']*k1xh**2 + W['22yy']*k1yh**2 + W['22zz']*k1zh**2 + 2. * \
                W['22xy']*k1xh*k1yh + 2.*W['22yz'] * \
                k1yh*k1zh + 2.*W['22xz']*k1zh*k1xh
            W_k1L2 = 1.5*xx - 0.5*W['22']
            W_k2L2 = 1.5*(W['22xx']*k2xh**2 + W['22yy']*k2yh**2 + W['22zz']*k2zh**2
                          + 2.*W['22xy']*k2xh*k2yh + 2.*W['22yz']*k2yh*k2zh + 2.*W['22xz']*k2zh*k2xh) - 0.5*W['22']
            Wc_k1L2 = np.conj(W_k1L2)
            Wc_k2L2 = np.conj(W_k2L2)

            W_k1L4 = 35./8.*(W['22xxxx']*k1xh**4 + W['22yyyy']*k1yh**4 + W['22zzzz']*k1zh**4
                             + 4.*W['22xxxy']*k1xh**3*k1yh + 4.*W['22xxxz'] *
                             k1xh**3*k1zh + 4.*W['22xyyy']*k1yh**3*k1xh
                             + 4.*W['22yyyz']*k1yh**3*k1zh + 4.*W['22xzzz'] *
                             k1zh**3*k1xh + 4.*W['22yzzz']*k1zh**3*k1yh
                             + 6.*W['22xxyy']*k1xh**2*k1yh**2 + 6.*W['22xxzz'] *
                             k1xh**2*k1zh**2 + 6.*W['22yyzz']*k1yh**2*k1zh**2
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
                + W['22xxyy']*(k1xh**2*k2yh**2 + k1yh**2*k2xh**2 + 4.*k1xh*k1yh*k2xh*k2yh) \
                + W['22xxzz']*(k1xh**2*k2zh**2 + k1zh**2*k2xh**2 + 4.*k1xh*k1zh*k2xh*k2zh) \
                + W['22yyzz']*(k1yh**2*k2zh**2 + k1zh**2*k2yh**2 + 4.*k1yh*k1zh*k2yh*k2zh) \
                + W['22xyyz']*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                + W['22xxyz']*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                + W['22xyzz']*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh *
                               k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            W_k2L4 = 35./8.*(W['22xxxx']*k2xh**4 + W['22yyyy']*k2yh**4 + W['22zzzz']*k2zh**4
                             + 4.*W['22xxxy']*k2xh**3*k2yh + 4.*W['22xxxz'] *
                             k2xh**3*k2zh + 4.*W['22xyyy']*k2yh**3*k2xh
                             + 4.*W['22yyyz']*k2yh**3*k2zh + 4.*W['22xzzz'] *
                             k2zh**3*k2xh + 4.*W['22yzzz']*k2zh**3*k2yh
                             + 6.*W['22xxyy']*k2xh**2*k2yh**2 + 6.*W['22xxzz'] *
                             k2xh**2*k2zh**2 + 6.*W['22yyzz']*k2yh**2*k2zh**2
                             + 12.*W['22xxyz']*k2xh**2*k2yh*k2zh + 12.*W['22xyyz']*k2yh**2*k2xh*k2zh + 12.*W['22xyzz']*k2zh**2*k2xh*k2yh) \
                - 5./2.*W_k2L2 - 7./8.*W_L0

            Wc_k2L4 = np.conj(W_k2L4)

            W_k1L2_k2L2 = 9./4.*k1k2 - 3./4.*xx - 1./2.*W_k2L2
            # approximate as 6th order FFTs not simulated
            W_k1L2_k2L4 = 2/7.*W_k1L2 + 20/77.*W_k1L4
            W_k1L4_k2L2 = W_k1L2_k2L4  # approximate
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
                + W['12xxyy']*(k1xh**2*k2yh**2 + k1yh**2*k2xh**2 + 4.*k1xh*k1yh*k2xh*k2yh) \
                + W['12xxzz']*(k1xh**2*k2zh**2 + k1zh**2*k2xh**2 + 4.*k1xh*k1zh*k2xh*k2zh) \
                + W['12yyzz']*(k1yh**2*k2zh**2 + k1zh**2*k2yh**2 + 4.*k1yh*k1zh*k2yh*k2zh) \
                + W['12xyyz']*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                + W['12xxyz']*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                + W['12xyzz']*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh *
                               k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            xxW12 = W['12xx']*k1xh**2 + W['12yy']*k1yh**2 + W['12zz']*k1zh**2 \
                + 2.*W['12xy']*k1xh*k1yh + 2.*W['12yz'] * \
                k1yh*k1zh + 2.*W['12xz']*k1zh*k1xh

            W12_L0 = W['12']
            W12_k1L2 = 1.5*xxW12 - 0.5*W['12']
            W12_k1L4 = 35./8.*(W['12xxxx']*k1xh**4 + W['12yyyy']*k1yh**4 + W['12zzzz']*k1zh**4
                               + 4.*W['12xxxy']*k1xh**3*k1yh + 4.*W['12xxxz'] *
                               k1xh**3*k1zh + 4.*W['12xyyy']*k1yh**3*k1xh
                               + 6.*W['12xxyy']*k1xh**2*k1yh**2 + 6.*W['12xxzz'] *
                               k1xh**2*k1zh**2 + 6.*W['12yyzz']*k1yh**2*k1zh**2
                               + 12.*W['12xxyz']*k1xh**2*k1yh*k1zh + 12.*W['12xyyz']*k1yh**2*k1xh*k1zh + 12.*W['12xyzz']*k1zh**2*k1xh*k1yh) \
                - 5./2.*W12_k1L2 - 7./8.*W12_L0

            W12_k1L4_k2L2 = 2/7.*W12_k1L2 + 20/77.*W12_k1L4
            W12_k1L4_k2L4 = 1/9.*W12_L0 + 100/693.*W12_k1L2 + 162/1001.*W12_k1L4

            W12_k2L2 = 1.5*(W['12xx']*k2xh**2 + W['12yy']*k2yh**2 + W['12zz']*k2zh**2
                            + 2.*W['12xy']*k2xh*k2yh + 2.*W['12yz']*k2yh*k2zh + 2.*W['12xz']*k2zh*k2xh) - 0.5*W['12']

            W12_k2L4 = 35./8.*(W['12xxxx']*k2xh**4 + W['12yyyy']*k2yh**4 + W['12zzzz']*k2zh**4
                               + 4.*W['12xxxy']*k2xh**3*k2yh + 4.*W['12xxxz'] *
                               k2xh**3*k2zh + 4.*W['12xyyy']*k2yh**3*k2xh
                               + 4.*W['12yyyz']*k2yh**3*k2zh + 4.*W['12xzzz'] *
                               k2zh**3*k2xh + 4.*W['12yzzz']*k2zh**3*k2yh
                               + 6.*W['12xxyy']*k2xh**2*k2yh**2 + 6.*W['12xxzz'] *
                               k2xh**2*k2zh**2 + 6.*W['12yyzz']*k2yh**2*k2zh**2
                               + 12.*W['12xxyz']*k2xh**2*k2yh*k2zh + 12.*W['12xyyz']*k2yh**2*k2xh*k2zh + 12.*W['12xyzz']*k2zh**2*k2xh*k2yh) \
                - 5./2.*W12_k2L2 - 7./8.*W12_L0

            W12_k1L2_k2L2 = 9./4.*k1k2W12 - 3./4.*xxW12 - 1./2.*W12_k2L2

            W_k1L2_Sumk2L22 = 1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24 = 2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22 = 1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24 = 2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44 = 1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4

            C00exp = [Wc_L0 * W_L0, Wc_L0 * W_k2L2, Wc_L0 * W_k2L4,
                      Wc_k1L2*W_L0, Wc_k1L2*W_k2L2, Wc_k1L2*W_k2L4,
                      Wc_k1L4*W_L0, Wc_k1L4*W_k2L2, Wc_k1L4*W_k2L4]

            C00exp += [2.*W_L0 * W12_L0, W_k1L2*W12_L0,         W_k1L4 * W12_L0,
                       W_k2L2*W12_L0, W_k2L4*W12_L0, np.conj(W12_L0)*W12_L0]

            C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,
                      Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,
                      Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,
                      Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,
                      Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]

            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2 + W_k1L2_k2L2*W12_L0+W_L0*W12_k1L2_k2L2,

                       0.5*((1/5.*W_L0+2/7.*W_k1L2 + 18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2
                            + (1/5.*W_k2L2+2/7.*W_k1L2_k2L2 + 18/35.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L2_k2L2),

                       0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2
                            + (2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L2*W12_k2L2 + (1/5.*W_L0 + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L2
                            + (1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4)*W12_L0 + W_k2L2*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L4*W12_k2L2 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L2
                            + W_k2L4*W12_k1L2_k2L2 + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4)*W12_L0),

                       np.conj(W12_k1L2_k2L2)*W12_L0 + np.conj(W12_k1L2)*W12_k2L2]

            C44exp = [Wc_k2L4 * W_k1L4 + Wc_L0 * W_k1L4_k2L4,
                      Wc_k2L4 * W_k1L4_k2L2 + Wc_L0 * W_k1L4_Sumk2L24,
                      Wc_k2L4 * W_k1L4_k2L4 + Wc_L0 * W_k1L4_Sumk2L44,
                      Wc_k1L2_k2L4*W_k1L4 + Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,
                      Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,
                      Wc_k1L4_k2L4*W_k1L4 + Wc_k1L4*W_k1L4_k2L4,
                      Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,
                      Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]

            C44exp += [W_k1L4 * W12_k2L4 + W_k2L4*W12_k1L4
                       + W_k1L4_k2L4*W12_L0 + W_L0 * W12_k1L4_k2L4,

                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4
                            + (2/7.*W_k1L2_k2L4 + 20/77.*W_k1L4_k2L4)*W12_L0 + W_k1L2 * W12_k1L4_k2L4),

                       0.5*((1/9.*W_L0 + 100/693.*W_k1L2 + 162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4
                            + (1/9.*W_k2L4 + 100/693.*W_k1L2_k2L4 + 162/1001.*W_k1L4_k2L4)*W12_L0 + W_k1L4 * W12_k1L4_k2L4),

                       0.5*(W_k1L4_k2L2*W12_k2L4 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L4 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L4 + (1/9.*W_L0 + 100/693.*W_k2L2 + 162/1001.*W_k2L4)*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L4 + (1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4)*W12_L0),

                       np.conj(W12_k1L4_k2L4)*W12_L0 + np.conj(W12_k1L4)*W12_k2L4]  # 1/(nbar)^2

            C20exp = [Wc_L0 * W_k1L2,   Wc_L0*W_k1L2_k2L2, Wc_L0 * W_k1L2_k2L4,
                      Wc_k1L2*W_k1L2, Wc_k1L2*W_k1L2_k2L2, Wc_k1L2*W_k1L2_k2L4,
                      Wc_k1L4*W_k1L2, Wc_k1L4*W_k1L2_k2L2, Wc_k1L4*W_k1L2_k2L4]

            C20exp += [W_k1L2*W12_L0 + W['22']*W12_k1L2,
                       0.5*((1/5.*W['22'] + 2/7.*W_k1L2 + 18 /
                            35.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L2),
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)
                            * W12_L0 + W_k1L4*W12_k1L2),
                       0.5*(W_k1L2_k2L2*W12_L0 + W_k2L2*W12_k1L2),
                       0.5*(W_k1L2_k2L4*W12_L0 + W_k2L4*W12_k1L2),
                       np.conj(W12_k1L2)*W12_L0]

            C40exp = [Wc_L0*W_k1L4,   Wc_L0 * W_k1L4_k2L2, Wc_L0 * W_k1L4_k2L4,
                      Wc_k1L2*W_k1L4, Wc_k1L2*W_k1L4_k2L2, Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L4*W_k1L4, Wc_k1L4*W_k1L4_k2L2, Wc_k1L4*W_k1L4_k2L4]

            C40exp += [W_k1L4*W12_L0 + W['22']*W12_k1L4,
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)
                            * W12_L0 + W_k1L2*W12_k1L4),
                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2+162 /
                            1001.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L4),
                       0.5*(W_k1L4_k2L2*W12_L0 + W_k2L2*W12_k1L4),
                       0.5*(W_k1L4_k2L4*W12_L0 + W_k2L4*W12_k1L4),
                       np.conj(W12_k1L4)*W12_L0]

            C42exp = [Wc_k2L2*W_k1L4 + Wc_L0 * W_k1L4_k2L2,
                      Wc_k2L2*W_k1L4_k2L2 + Wc_L0 * W_k1L4_Sumk2L22,
                      Wc_k2L2*W_k1L4_k2L4 + Wc_L0 * W_k1L4_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L4 + Wc_k1L2*W_k1L4_k2L2,
                      Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L4 + Wc_k1L4*W_k1L4_k2L2,
                      Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]

            C42exp += [W_k1L4*W12_k2L2 + W_k2L2*W12_k1L4
                       + W_k1L4_k2L2*W12_L0 + W['22']*W12_k1L4_k2L2,

                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4
                            + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L2 * W12_k1L4_k2L2),

                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2 + 162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4
                            + (1/9.*W_k2L2 + 100/693.*W_k1L2_k2L2 + 162/1001.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L4_k2L2),

                       0.5*(W_k1L4_k2L2*W12_k2L2 + (1/5.*W['22'] + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L2 + (1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4)*W12_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L2 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L2 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12_L0),

                       np.conj(W12_k1L4_k2L2)*W12_L0+np.conj(W12_k1L4)*W12_k2L2]  # 1/(nbar)^2

            for i in range(-k2_range, k2_range+1):
                ind = (sort == i)
                for j in range(15):
                    avgW00[i+3, j] += np.sum(C00exp[j][ind])
                    avgW22[i+3, j] += np.sum(C22exp[j][ind])
                    avgW44[i+3, j] += np.sum(C44exp[j][ind])
                    avgW20[i+3, j] += np.sum(C20exp[j][ind])
                    avgW40[i+3, j] += np.sum(C40exp[j][ind])
                    avgW42[i+3, j] += np.sum(C42exp[j][ind])

        for i in range(0, 2*k2_range+1):
            if (i+Nbin-k2_range >= nBins or i+Nbin-k2_range < 0):
                avgW00[i] *= 0
                avgW22[i] *= 0
                avgW44[i] *= 0
                avgW20[i] *= 0
                avgW40[i] *= 0
                avgW42[i] *= 0
            else:
                avgW00[i] = avgW00[i]/(kmodes_sampled*Bin_ModeNum[Nbin+i-k2_range]*I22**2)
                avgW22[i] = avgW22[i]/(kmodes_sampled*Bin_ModeNum[Nbin+i-k2_range]*I22**2)
                avgW44[i] = avgW44[i]/(kmodes_sampled*Bin_ModeNum[Nbin+i-k2_range]*I22**2)
                avgW20[i] = avgW20[i]/(kmodes_sampled*Bin_ModeNum[Nbin+i-k2_range]*I22**2)
                avgW40[i] = avgW40[i]/(kmodes_sampled*Bin_ModeNum[Nbin+i-k2_range]*I22**2)
                avgW42[i] = avgW42[i]/(kmodes_sampled*Bin_ModeNum[Nbin+i-k2_range]*I22**2)

        def l_factor(l1, l2): return (2*l1+1) * \
            (2*l2+1) * (2 if 0 in (l1, l2) else 1)

        avgWij = np.zeros((2*k2_range+1, 15, 6))

        avgWij[:, :, 0] = l_factor(0, 0)*np.real(avgW00)
        avgWij[:, :, 1] = l_factor(2, 2)*np.real(avgW22)
        avgWij[:, :, 2] = l_factor(4, 4)*np.real(avgW44)
        avgWij[:, :, 3] = l_factor(2, 0)*np.real(avgW20)
        avgWij[:, :, 4] = l_factor(4, 0)*np.real(avgW40)
        avgWij[:, :, 5] = l_factor(4, 2)*np.real(avgW42)

        return avgWij

    @property
    def shotnoise(self):
        return self.I('12')/self.I('22')

    # -------------- TO BE VALIDATED --------------

    def get_window_multipoles(self, window):
        dk = 2*np.pi/np.sum(np.array(self.BoxSize)**3)**(1/3)
        Nmesh = self.Nmesh

        kx, ky, kz = np.mgrid[-Nmesh//2:Nmesh//2, -
                              Nmesh//2:Nmesh//2, -Nmesh//2:Nmesh//2] * dk
        kmod = np.sqrt(kx**2 + ky**2 + kz**2)

        kmod[Nmesh//2, Nmesh//2, Nmesh//2] = np.inf  # 0th mode
        khx, khy, khz = kx/kmod, ky/kmod, kz/kmod
        kmod[Nmesh//2, Nmesh//2, Nmesh//2] = 0.  # 0th mode

        W_L0 = self.W(window).copy()

        W_L2 = \
            self.W(f'{window}xx') * 3*khx**2/2 + \
            self.W(f'{window}xy') * 3*khx*khy + \
            self.W(f'{window}xz') * 3*khx*khz + \
            self.W(f'{window}yy') * 3*khy**2/2 + \
            self.W(f'{window}yz') * 3*khy*khz + \
            self.W(f'{window}zz') * 3*khz**2/2 + \
            self.W(window) * -1/2

        W_L4 = \
            self.W(f'{window}xxxx') * 35*khx**4/8 + \
            self.W(f'{window}xxxy') * 35*khx**3*khy/2 + \
            self.W(f'{window}xxxz') * 35*khx**3*khz/2 + \
            self.W(f'{window}xxyy') * 105*khx**2*khy**2/4 + \
            self.W(f'{window}xxyz') * 105*khx**2*khy*khz/2 + \
            self.W(f'{window}xxzz') * 105*khx**2*khz**2/4 + \
            self.W(f'{window}xyyy') * 35*khx*khy**3/2 + \
            self.W(f'{window}xyyz') * 105*khx*khy**2*khz/2 + \
            self.W(f'{window}xyzz') * 105*khx*khy*khz**2/2 + \
            self.W(f'{window}xzzz') * 35*khx*khz**3/2 + \
            self.W(f'{window}yyyy') * 35*khy**4/8 + \
            self.W(f'{window}yyyz') * 35*khy**3*khz/2 + \
            self.W(f'{window}yyzz') * 105*khy**2*khz**2/4 + \
            self.W(f'{window}yzzz') * 35*khy*khz**3/2 + \
            self.W(f'{window}zzzz') * 35*khz**4/8 + \
            self.W(f'{window}xx') * -15*khx**2/4 + \
            self.W(f'{window}xy') * -15*khx*khy/2 + \
            self.W(f'{window}xz') * -15*khx*khz/2 + \
            self.W(f'{window}yy') * -15*khy**2/4 + \
            self.W(f'{window}yz') * -15*khy*khz/2 + \
            self.W(f'{window}zz') * -15*khz**2/4 + \
            self.W(window) * 3/8

        return W_L0, W_L2, W_L4

    def compute_power_multipoles(self):
        dk = 2*np.pi/np.sum(np.array(self.BoxSize)**3)**(1/3)
        Nmesh = self.Nmesh

        kx, ky, kz = np.mgrid[-Nmesh//2:Nmesh//2, -
                              Nmesh//2:Nmesh//2, -Nmesh//2:Nmesh//2] * dk
        kmod = np.sqrt(kx**2 + ky**2 + kz**2)

        kmin = 0.
        kmax = (self.Nmesh + 1) * dk/2

        # central k
        k = np.arange(kmin + dk/2, kmax + dk/2, dk)

        kedges = np.arange(kmin, kmax + dk, dk)

        kbin = np.digitize(kmod, kedges) - 1

        kmean = np.array([np.average(kmod[kbin == i])
                         for i, _ in enumerate(k)])
        # kmean = 3/4*((k+dk/2)**4 - (k-dk/2)**4)/((k+dk/2)**3 - (k-dk/2)**3)

        W10_L0, W10_L2, W10_L4 = self.get_window_multipoles('W10')
        W22_L0, W22_L2, W22_L4 = self.get_window_multipoles('W22')

        def compute_power(A, B=None):
            if B is None:
                B = A

            integrand = np.real(A * np.conj(B))

            return np.array([np.average(integrand[kbin == i]) for i, _ in enumerate(k)])

        powerW22 = {(l1, l2): (2*l1+1)*(2*l2+1)*compute_power(W1, W2)/self.I['22']**2
                    for (l1, W1), (l2, W2) in itt.combinations_with_replacement(utils.enum2([W22_L0, W22_L2, W22_L4]), 2)}

        powerW10 = {(l1, l2): (2*l1+1)*(2*l2+1)*compute_power(W1, W2)/self.I['10']**2
                    for (l1, W1), (l2, W2) in itt.combinations_with_replacement(utils.enum2([W10_L0, W10_L2, W10_L4]), 2)}

        powerW22xW10 = {(l1, l2): (2*l1+1)*(2*l2+1)*compute_power(W1, W2)/self.I['22']/self.I['10']
                        for (l1, W1), (l2, W2) in itt.product(utils.enum2([W22_L0, W22_L2, W22_L4]),
                                                              utils.enum2([W10_L0, W10_L2, W10_L4]))}

        # subtracting shot noise
        powerW22[0, 0] -= self.alpha*self.I['34']/self.I['22']**2
        powerW10[0, 0] -= self.alpha/self.I['10']
        powerW22xW10[0, 0] -= self.alpha/self.I['10']

        if kmin == 0:
            # manually setting k = 0 values
            for l1, l2 in powerW10.items():
                powerW10[l1, l2][0] = 1. if l1 == l2 else 0.

            for l1, l2 in powerW22:
                powerW22[l1, l2][0] = 1. if l1 == l2 else 0.

            for l1, l2 in powerW22xW10:
                powerW22xW10[l1, l2][0] = 1. if l1 == l2 else 0.

        return kmean, powerW10, powerW22, powerW22xW10
