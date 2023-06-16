"""Module containing classes that represent the geometry to be used in the covariance calculation.

Classes
-------
Geometry
    Abstract class that defines the interface for the geometry classes.
BoxGeometry
    Class that represents the geometry of a periodic cubic box.
SurveyGeometry
    Class that represents the geometry of a survey in cut-sky.
"""

from abc import ABC
import itertools as itt

import multiprocessing as mp
import os

import numpy as np
from scipy import fft

from tqdm import tqdm as shell_tqdm

from . import base, utils

__all__ = ['BoxGeometry', 'SurveyGeometry']

# Window functions needed for Gaussian covariance calculation
W_LABELS = ['12', '12xx', '12xy', '12xz', '12yy', '12yz', '12zz', '12xxxx', '12xxxy', '12xxxz', '12xxyy', '12xxyz', '12xxzz', '12xyyy', '12xyyz', '12xyzz', '12xzzz', '12yyyy', '12yyyz', '12yyzz', '12yzzz',
            '12zzzz', '22', '22xx', '22xy', '22xz', '22yy', '22yz', '22zz', '22xxxx', '22xxxy', '22xxxz', '22xxyy', '22xxyz', '22xxzz', '22xyyy', '22xyyz', '22xyzz', '22xzzz', '22yyyy', '22yyyz', '22yyzz', '22yzzz', '22zzzz']


class Geometry(ABC):
    pass


class BoxGeometry(Geometry):
    '''Class that represents the geometry of a periodic cubic box.
    
    Attributes
    ----------
    BoxSize : float
        Size of the box.
    Nmesh : int
        Number of mesh points in each dimension.
    alpha : float
        Factor to multiply the number of galaxies in the box.
    
    Methods
    -------
    set_boxsize
        Set the size of the box.
    set_nmesh
        Set the number of mesh points in each dimension.
    set_alpha
        Set the factor to multiply the number of galaxies in the box.
    '''
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
        '''Estimates the Poissonian shotnoise of the sample as 1/nbar.
        
        Returns
        -------
        float
            Poissonian shotnoise of the sample = 1/nbar.'''
        return 1/self.nbar

    @shotnoise.setter
    def shotnoise(self, shotnoise):
        '''Sets the Poissonian shotnoise of the sample and nbar = 1/shotnoise.
        
        Parameters
        ----------
        shotnoise : float
            Shotnoise of the sample.'''
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
        '''Set the effective volume of the box.
        
        Parameters
        ----------
        zmin : float
            Minimum redshift of the sample.
        zmax : float
            Maximum redshift of the sample.
        fsky : float, optional
            Fraction of the sky covered by the sample. If not given, the current value of fsky is used.
            
        Returns
        -------
        float
            Effective volume of the box.'''

        if fsky is not None:
            self.fsky = fsky

        self._zmin = zmin
        self._zmax = zmax

        self.volume = self.fsky * 4/3 * np.pi * \
            (self.cosmo.comoving_distance(zmax)**3 -
             self.cosmo.comoving_distance(zmin)**3)

        return self.volume

    def set_nz(self, zedges, nz, *args, **kwargs):
        '''Set the effective volume and number density of the box based on the
        redshift distribution of the sample.

        Parameters
        ----------
        zedges : array_like
            Array of redshift bin edges.
        nz : array_like
            Array of redshift distribution of the sample.
        *args, **kwargs
            Arguments and keyword arguments to be passed to set_effective_volume.
        '''
        assert len(zedges) == len(nz) + \
            1, "Length of zedges should equal length of nz + 1."

        self._zedges = np.array(zedges)
        self._nz = np.array(nz)[np.argsort(self.zmid)]
        self._zedges.sort()

        self.set_effective_volume(
            zmin=self.zmin, zmax=self.zmax, *args, **kwargs)
        print(f'Effective volume: {self.volume:.3e} (Mpc/h)^3')

        bin_volume = self.fsky * \
            np.diff(self.cosmo.comoving_distance(self.zedges)**3)
        self.nbar = np.average(self.nz, weights=bin_volume)
        print(f'Estimated nbar: {self.nbar:.3e} (Mpc/h)^-3')

    def set_randoms(self, randoms, alpha=1.0):
        '''Estimates the effective volume and number density of the box based on a
        provided catalog of randoms.
        
        Parameters
        ----------
        randoms : array_like
            Catalog of randoms.
        alpha : float, optional
            Factor to multiply the number density of the randoms. Default is 1.0.
        '''
        import healpy as hp
        from nbodykit.algorithms import zhist

        nside = 512
        hpixel = hp.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)
        unique_hpixels = np.unique(hpixel)
        self.fsky = len(unique_hpixels.compute())/hp.nside2npix(nside)

        print(f'fsky estimated from randoms: {self.fsky:.3f}')

        nz_hist = zhist.RedshiftHistogram(
            randoms, self.fsky, self.cosmo, redshift='Z')
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
    '''Class that represents the geometry of a survey in cut-sky.
    
    Attributes
    ----------
    random_catalog : nbodykit.catalog.CatalogSource
        Catalog of randoms.
    Nmesh : int
        Number of mesh points in each dimension to be used in the calculation of the FFTs.
    BoxSize : float
        Size of the box.
    alpha : float
        Factor to multiply the number density of the randoms. Default is 1.0.
    delta_k_max : int
        Number of k bins to be calculated each side of the diagonal.
    kmodes_sampled : int
        Number of k modes to be sampled in the calculation of the window kernels.
    mesh_kwargs : dict
        Arguments to be passed to nbodykit.mesh.MeshSource.to_mesh.
    tqdm : callable
        Function to be used for progress bar. Default is tqdm from tqdm package.
    
    Methods
    -------
    set_kbins(kmin, kmax, dk)
        Set the k bins to be used in the calculation of the covariance.
    compute_window_kernels
        Compute the window kernels to be used in the calculation of the covariance.
    save_window_kernels(filename)
        Save the window kernels to a file.
    load_window_kernels(filename)
        Load the window kernels from a file.
    
    Notes
    -----
    The window kernels are computed using the method described in [1]_.
    
    References
    ----------
    .. [1] https://arxiv.org/abs/1910.02914
    '''

    def __init__(self, random_catalog=None, Nmesh=None, BoxSize=None, kmax_mask=0.05, delta_k_max=3, kmodes_sampled=250, alpha=1.0, mesh_kwargs=None, tqdm=shell_tqdm):

        assert Nmesh is None or Nmesh % 2 == 1, 'Please, use an odd integer for Nmesh.'
        
        base.FourierBinned.__init__(self)

        self.alpha = alpha
        self.delta_k_max = delta_k_max
        self.kmodes_sampled = kmodes_sampled

        self.tqdm = tqdm

        self._W = {}
        self._I = {}

        if random_catalog is not None:
            self._randoms = random_catalog

            self._ngals = self.randoms.size * self.alpha

            pos_max = np.max(random_catalog['Position'], axis=0).compute()
            pos_min = np.min(random_catalog['Position'], axis=0).compute()

            survey_center = (pos_max + pos_min)/2
            print(f'Survey center is at xyz = {survey_center}. Centering coordinates.')

            self._randoms['OriginalPosition'] = self._randoms['Position']
            # self._randoms['Position'] -= np.array(survey_center)
            self._randoms['Position'] += np.array(3*[BoxSize/2])
        else:
            self._ngals = None

        if BoxSize is None and random_catalog is not None:
            # Estimate BoxSize from random catalog
            self.BoxSize = max(pos_max - pos_min)
            print(f'Box size estimated from the randoms is {pos_max - pos_min}. Using BoxSize = {self.BoxSize}.')
        elif np.ndim(BoxSize) == 0:
            self.BoxSize = BoxSize
        elif np.ndim(BoxSize) == 1:
            self.BoxSize = max(BoxSize)

        if Nmesh is None:
            # Pick odd value that will give at least k_mask = kmax_mask in the FFTs
            self.Nmesh = int(kmax_mask*self.BoxSize/np.pi)//2 * 2 + 1
            print(f'Using Nmesh = {self.Nmesh} for the window FFTs.')
        else:
            self.Nmesh = Nmesh

        k_nyquist = np.pi*self.Nmesh/self.BoxSize
        print(f'Nyquist wavelength of window FFTs = {k_nyquist}.')

        if k_nyquist < kmax_mask:
            print(f'WARNING: Nyquist frequency smaller than required kmax_mask = {kmax_mask}.')

        self._mesh_kwargs = {
            'Nmesh':       self.Nmesh,
            'BoxSize':     self.BoxSize,
            'interlaced':  True,
            'compensated': True,
            'resampler':   'tsc',
        }

        if mesh_kwargs is not None:
            self._mesh_kwargs.update(mesh_kwargs)

        print(f'Average of {self.nbar/self.alpha*(self.BoxSize/self.Nmesh)**3} objects per voxel.')

    def W_cat(self, W):
        '''Adds a column to the random catalog with the window function Wij.
        
        Parameters
        ----------
        W : str
            Window function label.
        
        Returns
        -------
        array_like
            Window function Wij.
        '''
        w = W.lower().replace("w", "")

        if f'W{w}' not in self._randoms.columns:
            self._randoms[f'W{w}'] = self._randoms['NZ']**(
                int(w[0])-1) * self._randoms['WEIGHT_FKP']**int(w[1])
        return self._randoms[f'W{w}']

    def I(self, W):
        '''Computes the integral Iij of the window function Wij.
        
        Parameters
        ----------
        W : str
            Window function label.
            
        Returns
        -------
        float
            Integral Iij of the window function Wij.
        '''
        w = W.lower().replace("i", "").replace("w", "")
        if w not in self._I:
            self.W_cat(w)
            self._I[w] = (self._randoms[f'W{w}'].sum() * self.alpha).compute()
        return self._I[w]

    def W(self, W):
        '''Returns FFT of the window function Wij. If it has not been computed yet, it is computed.
        
        Parameters
        ----------
        W : str
            Window function label.
        
        Returns
        -------
        array_like
            FFT of the window function Wij.
        '''
        w = W.lower().replace("w", "")
        if w not in self._W:
            self.W_cat(w)
            self.compute_cartesian_fft(
                w.replace("x", "").replace("y", "").replace("z", ""))
        return self._W[w]

    def compute_cartesian_ffts(self, Wij=('W12', 'W22')):
        '''Computes the FFTs of the window functions Wij.
        
        Parameters
        ----------
        Wij : array_like, optional
            List of window function labels. Default is ["W12", "W22"].
        '''
        [self.W(w) for w in Wij]

    def compute_cartesian_fft(self, W):
        '''Computes the FFT of the window function Wij.
        
        Parameters
        ----------
        W : str
            Window function label.
        '''
        w = W.lower().replace('w', '')

        self.W_cat(w)

        mesh_kwargs = self._mesh_kwargs

        x = self.randoms['OriginalPosition'].T

        with self.tqdm(total=22, desc=f'Computing moments of W{w}') as pbar:
            self.set_cartesian_fft(f'W{w}', self._format_fft(self.randoms.to_mesh(
                value=f'W{w}', **mesh_kwargs).paint(mode='complex')))
            
            # Check if zero mode of FFT Wij is consistent with integral Iij
            center = self.Nmesh//2
            if self.Nmesh % 2 == 0:
                center -= 1
            Ww = self.W(w)[center, center, center]
            Iw = self.I(w)

            if not np.isclose(Ww.real, Iw, rtol=1e-2) and np.isclose(Ww.imag, 0, rtol=1e-2):
                print(f'WARNING: Inconsistency between W{w}_0 = {Ww} and I{w} = {Iw}.')
            

            pbar.update(1)

            for (i, i_label), (j, j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
                label = f'W{w}{i_label}{j_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * \
                    x[i]*x[j] / (x[0]**2 + x[1]**2 + x[2]**2)
                self.set_cartesian_fft(label, self._format_fft(self.randoms.to_mesh(
                    value=label, **mesh_kwargs).paint(mode='complex')))

                pbar.update(1)

            for (i, i_label), (j, j_label), (k, k_label), (l, l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
                label = f'W{w}{i_label}{j_label}{k_label}{l_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * x[i] * \
                    x[j]*x[k]*x[l] / (x[0]**2 + x[1]**2 + x[2]**2)**2
                self.set_cartesian_fft(label, self._format_fft(self.randoms.to_mesh(
                    value=label, **mesh_kwargs).paint(mode='complex')))

                pbar.update(1)

    # def compute_cartesian_fft_single_mesh(self, W):

    #     w = W.replace('W', '').replace('w', '')
    #     self.W_cat(w)

    #     mesh_kwargs = self._mesh_kwargs

    #     real_field = self.randoms.to_mesh(
    #         value='W12', **mesh_kwargs).paint(mode='real')

    #     self._W[w] = self._format_fft(real_field.r2c(), w)

    #     for (i, i_label), (j, j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
    #         self._W[f'{w}{i_label}{j_label}'] = self._format_fft(real_field.apply(
    #             lambda x, v: np.nan_to_num(v*x[i]*x[j]/(x[0]**2 + x[1]**2 + x[2]**2)), kind='relative').r2c(), w)

    #     for (i, i_label), (j, j_label), (k, k_label), (l, l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
    #         self._W[f'{w}{i_label}{j_label}{k_label}{l_label}'] = self._format_fft(real_field.apply(
    #             lambda x, v: np.nan_to_num(v*x[i]*x[j]*x[k]*x[l]/(x[0]**2 + x[1]**2 + x[2]**2)**2), kind='relative').r2c(), w)

    def _format_fft(self, fourier):

        # PFFT omits modes that are determined by the Hermitian condition. Adding them:
        fourier_full = utils.r2c_to_c2c_3d(fourier)

        return fft.fftshift(fourier_full)[::-1, ::-1, ::-1] * self.ngals

    def _unformat_fft(self, fourier, window):

        fourier_cut = fft.ifftshift(
            fourier[::-1, ::-1, ::-1])[:, :, :fourier.shape[2]//2+1] / self.ngals

        return np.conj(fourier_cut) if window == '12' else fourier_cut

    def set_cartesian_fft(self, label, W):
        '''Set the FFT of the window function Wij.
        
        Parameters
        ----------
        label : str
            Window function label.
        W : array_like
            FFT of the window function Wij.
        '''
        w = label.lower().replace('w', '')
        if w not in self._W:
            # Create object proper for multiprocessing
            self._W[w] = np.frombuffer(mp.RawArray('d', 2*self.Nmesh**3))\
                           .view(np.complex128)\
                           .reshape(*3*[self.Nmesh])
        self._W[w][:, :, :] = W

    def save_cartesian_ffts(self, filename):
        '''Save the FFTs of the window functions Wij to a file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the FFTs of the window functions Wij.
        '''
        np.savez(filename if filename.strip()[-4:] == '.npz' else f'{filename}.npz', **{f'W{k.replace("W","")}': self._W[k] for k in self._W}, **{
                 f'I{k.replace("I","")}': self._I[k] for k in self._I}, BoxSize=self.BoxSize, Nmesh=self.Nmesh, alpha=self.alpha)

    def load_cartesian_ffts(self, filename):
        '''Load the FFTs of the window functions Wij from a file.
        
        Parameters
        ----------
        filename : str
            Name of the file to load the FFTs of the window functions Wij from.
        '''
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
        '''Returns the window kernels to be used in the calculation of the covariance.
        
        Returns
        -------
        array_like
            Window kernels to be used in the calculation of the covariance.
        '''
        if not (hasattr(self, 'WinKernel') and self.WinKernel is not None):
            self.compute_window_kernels()
        return self.WinKernel

    def compute_window_kernels(self):
        '''Computes the window kernels to be used in the calculation of the covariance.
        
        Notes
        -----
        The window kernels are computed using the method described in [1]_.
        
        References
        ----------
        .. [1] https://arxiv.org/abs/1910.02914
        '''

        kfun = 2*np.pi/self.BoxSize

        # sample kmodes from each k1 bin
        # kmodes = np.array([[utils.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(
        #     self.kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])
        
        kmodes, Nmodes = utils.sample_from_cube(self.kmax/kfun, self.dk/kfun, self.kmodes_sampled)

        # Note: as the window falls steeply with k, only low-k regions are needed for the calculation.
        # Therefore, high-k modes in the FFTs can be discarded

        self.compute_cartesian_fft('W12')
        self.compute_cartesian_fft('W22')

        init_params = {
            'I22': self.I('22'),
            'BoxSize': self.BoxSize,
            'kbins': self.kbins,
            'dk': self.dk,
            'Nmesh': self.Nmesh,
            'delta_k_max': self.delta_k_max,
            'Nmodes': Nmodes,
        }

        def init_worker(*args):
            global shared_w
            global shared_params
            shared_w = {}
            for w, l in zip(args, W_LABELS):
                shared_w[l] = np.frombuffer(w).view(
                    np.complex128).reshape(*3*[self.Nmesh])
            shared_params = args[-1]

        if hasattr(self, 'num_threads'):
            num_threads = self.num_threads
        elif 'OMP_NUM_THREADS' in os.environ:
            num_threads = os.environ['OMP_NUM_THREADS']
        else:
            num_threads = os.cpu_count()

        # Calls the function _compute_window_kernel_row in parallel for each k1 bin
        with mp.Pool(processes=num_threads, initializer=init_worker, initargs=[*[self._W[w] for w in W_LABELS], init_params]) as pool:
            self.WinKernel = np.array(list(self.tqdm(pool.imap(self._compute_window_kernel_row, enumerate(
                kmodes)), total=self.kbins, desc="Computing window kernels")))

        # self.WinKernel = np.array([
        #     self._compute_window_kernel_row(Nbin)
        #     for Nbin in self.tqdm(range(self.kbins), total=self.kbins, desc="Computing window kernels")])

    def save_window_kernels(self, filename):
        '''Save the window kernels to a file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the window kernels.'''
        np.savez(filename if filename.strip()
                 [-4:] == '.npz' else f'{filename}.npz', WinKernel=self.WinKernel, kmin=self.kmin, kmax=self.kmax, dk=self.dk)

    def load_window_kernels(self, filename):
        '''Load the window kernels from a file.
        
        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''
        with np.load(filename, mmap_mode='r') as data:
            self.WinKernel = data['WinKernel']
            self.set_kbins(data['kmin'], data['kmax'], data['dk'])

    @staticmethod
    def _compute_window_kernel_row(args):

        k1_bin_index, Bin_kmodes = args

        I22 = shared_params['I22']

        BoxSize = shared_params['BoxSize']
        kfun = 2*np.pi/BoxSize,

        nBins = shared_params['kbins']
        kBinWidth = shared_params['dk']

        Nmesh = shared_params['Nmesh']

        W = shared_w

        # Bin_ModeNum = utils.nmodes(BoxSize**3, kedges[:-1], kedges[1:])
        Bin_ModeNum = shared_params['Nmodes']

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        icut = Nmesh//2
        Nbin = k1_bin_index

        avgWij = np.zeros((2*3+1,15,6))

        avgW00 = np.zeros((2*3+1,15), dtype='<c8')
        avgW22 = avgW00.copy()
        avgW44 = avgW00.copy()
        avgW20 = avgW00.copy()
        avgW40 = avgW00.copy()
        avgW42 = avgW00.copy()

        ix,iy,iz,k2xh,k2yh,k2zh = np.zeros((6,2*icut+1, 2*icut+1, 2*icut+1))
        
        for i in range(2*icut+1):
            ix[i,:,:] += i-icut
            iy[:,i,:] += i-icut
            iz[:,:,i] += i-icut


        norm = len(Bin_kmodes)
            
        # if (kmodes_sampled < Bin_ModeNum[Nbin]):
        #     norm = kmodes_sampled
        #     sampled=(np.random.rand(kmodes_sampled)*Bin_ModeNum[Nbin]).astype(int)
        # else:
        #     norm = Bin_ModeNum[Nbin]
        #     sampled=np.arange(Bin_ModeNum[Nbin],dtype=int)
        
        # Randomly select a mode in the k1 bin
        for ik1x,ik1y,ik1z,rk1 in Bin_kmodes:
            if (rk1 == 0.):
                k1xh = 0
                k1yh = 0
                k1zh = 0
            else:
                k1xh = ik1x/rk1
                k1yh = ik1y/rk1
                k1zh = ik1z/rk1
                
            # Build a 3D array of modes around the selected mode   
            k2xh = ik1x-ix
            k2yh = ik1y-iy
            k2zh = ik1z-iz

            rk2 = np.sqrt(k2xh**2+k2yh**2+k2zh**2)
            sort = (rk2*kfun/kBinWidth).astype(int)-Nbin # to decide later which shell the k2 mode belongs to
            ind = (rk2 == 0)

            if (ind.any() > 0):
                rk2[ind] = 1e10

            k2xh/=rk2
            k2yh/=rk2
            k2zh/=rk2
            #k2 hat arrays built
            
            # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
            # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
            # L(i) refers to multipoles
                
            # k1xh is a scalar
            # k2xh.shape = (31,31,31)
            
            W_L0 = W['22']
            Wc_L0 = np.conj(W['22'])

            xx=W['22xx']*k1xh**2+W['22yy']*k1yh**2+W['22zz']*k1zh**2+2.*W['22xy']*k1xh*k1yh+2.*W['22yz']*k1yh*k1zh+2.*W['22xz']*k1zh*k1xh
            
            W_k1L2=1.5*xx-0.5*W['22']
            W_k2L2=1.5*(W['22xx']*k2xh**2+W['22yy']*k2yh**2+W['22zz']*k2zh**2 \
            +2.*W['22xy']*k2xh*k2yh+2.*W['22yz']*k2yh*k2zh+2.*W['22xz']*k2zh*k2xh)-0.5*W['22']
            Wc_k1L2=np.conj(W_k1L2)
            Wc_k2L2=np.conj(W_k2L2)
            
            W_k1L4=35./8.*(W['22xxxx']*k1xh**4 +W['22yyyy']*k1yh**4+W['22zzzz']*k1zh**4 \
        +4.*W['22xxxy']*k1xh**3*k1yh +4.*W['22xxxz']*k1xh**3*k1zh +4.*W['22xyyy']*k1yh**3*k1xh \
        +4.*W['22yyyz']*k1yh**3*k1zh +4.*W['22xzzz']*k1zh**3*k1xh +4.*W['22yzzz']*k1zh**3*k1yh \
        +6.*W['22xxyy']*k1xh**2*k1yh**2+6.*W['22xxzz']*k1xh**2*k1zh**2+6.*W['22yyzz']*k1yh**2*k1zh**2 \
        +12.*W['22xxyz']*k1xh**2*k1yh*k1zh+12.*W['22xyyz']*k1yh**2*k1xh*k1zh +12.*W['22xyzz']*k1zh**2*k1xh*k1yh) \
        -5./2.*W_k1L2 -7./8.*W_L0
            Wc_k1L4=np.conj(W_k1L4)
            
            k1k2=W['22xxxx']*(k1xh*k2xh)**2+W['22yyyy']*(k1yh*k2yh)**2+W['22zzzz']*(k1zh*k2zh)**2 \
                +W['22xxxy']*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
                +W['22xxxz']*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
                +W['22yyyz']*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
                +W['22yzzz']*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
                +W['22xyyy']*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
                +W['22xzzz']*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
                +W['22xxyy']*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
                +W['22xxzz']*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
                +W['22yyzz']*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
                +W['22xyyz']*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
                +W['22xxyz']*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
                +W['22xyzz']*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
            
            W_k2L4=35./8.*(W['22xxxx']*k2xh**4 +W['22yyyy']*k2yh**4+W['22zzzz']*k2zh**4 \
        +4.*W['22xxxy']*k2xh**3*k2yh +4.*W['22xxxz']*k2xh**3*k2zh +4.*W['22xyyy']*k2yh**3*k2xh \
        +4.*W['22yyyz']*k2yh**3*k2zh +4.*W['22xzzz']*k2zh**3*k2xh +4.*W['22yzzz']*k2zh**3*k2yh \
        +6.*W['22xxyy']*k2xh**2*k2yh**2+6.*W['22xxzz']*k2xh**2*k2zh**2+6.*W['22yyzz']*k2yh**2*k2zh**2 \
        +12.*W['22xxyz']*k2xh**2*k2yh*k2zh+12.*W['22xyyz']*k2yh**2*k2xh*k2zh +12.*W['22xyzz']*k2zh**2*k2xh*k2yh) \
        -5./2.*W_k2L2 -7./8.*W_L0
            Wc_k2L4=np.conj(W_k2L4)
            
            W_k1L2_k2L2= 9./4.*k1k2 -3./4.*xx -1./2.*W_k2L2
            W_k1L2_k2L4=2/7.*W_k1L2+20/77.*W_k1L4 #approximate as 6th order FFTs not simulated
            W_k1L4_k2L2=W_k1L2_k2L4 #approximate
            W_k1L4_k2L4=1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4
            Wc_k1L2_k2L2= np.conj(W_k1L2_k2L2)
            Wc_k1L2_k2L4=np.conj(W_k1L2_k2L4)
            Wc_k1L4_k2L2=Wc_k1L2_k2L4
            Wc_k1L4_k2L4=np.conj(W_k1L4_k2L4)
            
            k1k2W12=np.conj(W['12xxxx'])*(k1xh*k2xh)**2+np.conj(W['12yyyy'])*(k1yh*k2yh)**2+np.conj(W['12zzzz'])*(k1zh*k2zh)**2 \
                +np.conj(W['12xxxy'])*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
                +np.conj(W['12xxxz'])*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
                +np.conj(W['12yyyz'])*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
                +np.conj(W['12yzzz'])*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
                +np.conj(W['12xyyy'])*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
                +np.conj(W['12xzzz'])*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
                +np.conj(W['12xxyy'])*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
                +np.conj(W['12xxzz'])*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
                +np.conj(W['12yyzz'])*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
                +np.conj(W['12xyyz'])*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
                +np.conj(W['12xxyz'])*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
                +np.conj(W['12xyzz'])*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
            
            xxW12=np.conj(W['12xx'])*k1xh**2+np.conj(W['12yy'])*k1yh**2+np.conj(W['12zz'])*k1zh**2+2.*np.conj(W['12xy'])*k1xh*k1yh+2.*np.conj(W['12yz'])*k1yh*k1zh+2.*np.conj(W['12xz'])*k1zh*k1xh
        
            W12_L0 = np.conj(W['12'])
            W12_k1L2=1.5*xxW12-0.5*np.conj(W['12'])
            W12_k1L4=35./8.*(np.conj(W['12xxxx'])*k1xh**4 +np.conj(W['12yyyy'])*k1yh**4+np.conj(W['12zzzz'])*k1zh**4 \
        +4.*np.conj(W['12xxxy'])*k1xh**3*k1yh +4.*np.conj(W['12xxxz'])*k1xh**3*k1zh +4.*np.conj(W['12xyyy'])*k1yh**3*k1xh \
        +6.*np.conj(W['12xxyy'])*k1xh**2*k1yh**2+6.*np.conj(W['12xxzz'])*k1xh**2*k1zh**2+6.*np.conj(W['12yyzz'])*k1yh**2*k1zh**2 \
        +12.*np.conj(W['12xxyz'])*k1xh**2*k1yh*k1zh+12.*np.conj(W['12xyyz'])*k1yh**2*k1xh*k1zh +12.*np.conj(W['12xyzz'])*k1zh**2*k1xh*k1yh) \
        -5./2.*W12_k1L2 -7./8.*W12_L0
            W12_k1L4_k2L2=2/7.*W12_k1L2+20/77.*W12_k1L4
            W12_k1L4_k2L4=1/9.*W12_L0+100/693.*W12_k1L2+162/1001.*W12_k1L4
            W12_k2L2=1.5*(np.conj(W['12xx'])*k2xh**2+np.conj(W['12yy'])*k2yh**2+np.conj(W['12zz'])*k2zh**2\
            +2.*np.conj(W['12xy'])*k2xh*k2yh+2.*np.conj(W['12yz'])*k2yh*k2zh+2.*np.conj(W['12xz'])*k2zh*k2xh)-0.5*np.conj(W['12'])
            W12_k2L4=35./8.*(np.conj(W['12xxxx'])*k2xh**4 +np.conj(W['12yyyy'])*k2yh**4+np.conj(W['12zzzz'])*k2zh**4 \
        +4.*np.conj(W['12xxxy'])*k2xh**3*k2yh +4.*np.conj(W['12xxxz'])*k2xh**3*k2zh +4.*np.conj(W['12xyyy'])*k2yh**3*k2xh \
        +4.*np.conj(W['12yyyz'])*k2yh**3*k2zh +4.*np.conj(W['12xzzz'])*k2zh**3*k2xh +4.*np.conj(W['12yzzz'])*k2zh**3*k2yh \
        +6.*np.conj(W['12xxyy'])*k2xh**2*k2yh**2+6.*np.conj(W['12xxzz'])*k2xh**2*k2zh**2+6.*np.conj(W['12yyzz'])*k2yh**2*k2zh**2 \
        +12.*np.conj(W['12xxyz'])*k2xh**2*k2yh*k2zh+12.*np.conj(W['12xyyz'])*k2yh**2*k2xh*k2zh +12.*np.conj(W['12xyzz'])*k2zh**2*k2xh*k2yh) \
        -5./2.*W12_k2L2 -7./8.*W12_L0
            
            W12_k1L2_k2L2= 9./4.*k1k2W12 -3./4.*xxW12 -1./2.*W12_k2L2
            
            W_k1L2_Sumk2L22=1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24=2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22=1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24=2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44=1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4
            
            C00exp = [Wc_L0*W_L0,Wc_L0*W_k2L2,Wc_L0*W_k2L4,\
                    Wc_k1L2*W_L0,Wc_k1L2*W_k2L2,Wc_k1L2*W_k2L4,\
                    Wc_k1L4*W_L0,Wc_k1L4*W_k2L2,Wc_k1L4*W_k2L4]
            
            C00exp += [2.*W_L0*W12_L0,W_k1L2*W12_L0,W_k1L4*W12_L0,\
                    W_k2L2*W12_L0,W_k2L4*W12_L0,np.conj(W12_L0)*W12_L0]
            
            C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,\
                    Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,\
                    Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,\
                    Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,\
                    Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,\
                    Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,\
                    Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,\
                    Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,\
                    Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]
            
            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2\
                    +W_k1L2_k2L2*W12_L0+W_L0*W12_k1L2_k2L2,\
                    0.5*((1/5.*W_L0+2/7.*W_k1L2+18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2\
    +(1/5.*W_k2L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L2_k2L2),\
        0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2\
    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L2_k2L2),\
    0.5*(W_k1L2_k2L2*W12_k2L2+(1/5.*W_L0+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L2\
    +(1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4)*W12_L0 + W_k2L2*W12_k1L2_k2L2),\
    0.5*(W_k1L2_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L2\
    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4)*W12_L0 + W_k2L4*W12_k1L2_k2L2),\
                    np.conj(W12_k1L2_k2L2)*W12_L0+np.conj(W12_k1L2)*W12_k2L2]
            
            C44exp = [Wc_k2L4*W_k1L4 + Wc_L0*W_k1L4_k2L4,\
                    Wc_k2L4*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L24,\
                    Wc_k2L4*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L44,\
                    Wc_k1L2_k2L4*W_k1L4 + Wc_k1L2*W_k1L4_k2L4,\
                    Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,\
                    Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,\
                    Wc_k1L4_k2L4*W_k1L4 + Wc_k1L4*W_k1L4_k2L4,\
                    Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,\
                    Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]
            
            C44exp += [W_k1L4*W12_k2L4 + W_k2L4*W12_k1L4\
                    +W_k1L4_k2L4*W12_L0+W_L0*W12_k1L4_k2L4,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4\
    +(2/7.*W_k1L2_k2L4+20/77.*W_k1L4_k2L4)*W12_L0 + W_k1L2*W12_k1L4_k2L4),\
    0.5*((1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4\
    +(1/9.*W_k2L4+100/693.*W_k1L2_k2L4+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k1L4*W12_k1L4_k2L4),\
    0.5*(W_k1L4_k2L2*W12_k2L4+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
    +(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L4),\
    0.5*(W_k1L4_k2L4*W12_k2L4+(1/9.*W_L0+100/693.*W_k2L2+162/1001.*W_k2L4)*W12_k1L4\
    +(1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L4),\
                    np.conj(W12_k1L4_k2L4)*W12_L0+np.conj(W12_k1L4)*W12_k2L4] #1/(nbar)^2
            
            C20exp = [Wc_L0*W_k1L2,Wc_L0*W_k1L2_k2L2,Wc_L0*W_k1L2_k2L4,\
                    Wc_k1L2*W_k1L2,Wc_k1L2*W_k1L2_k2L2,Wc_k1L2*W_k1L2_k2L4,\
                    Wc_k1L4*W_k1L2,Wc_k1L4*W_k1L2_k2L2,Wc_k1L4*W_k1L2_k2L4]
            
            C20exp += [W_k1L2*W12_L0 + W_L0*W12_k1L2,\
                    0.5*((1/5.*W_L0+2/7.*W_k1L2+18/35.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L2),\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L2),\
                    0.5*(W_k1L2_k2L2*W12_L0 + W_k2L2*W12_k1L2),\
                    0.5*(W_k1L2_k2L4*W12_L0 + W_k2L4*W12_k1L2),\
                    np.conj(W12_k1L2)*W12_L0]
            
            C40exp = [Wc_L0*W_k1L4,Wc_L0*W_k1L4_k2L2,Wc_L0*W_k1L4_k2L4,\
                    Wc_k1L2*W_k1L4,Wc_k1L2*W_k1L4_k2L2,Wc_k1L2*W_k1L4_k2L4,\
                    Wc_k1L4*W_k1L4,Wc_k1L4*W_k1L4_k2L2,Wc_k1L4*W_k1L4_k2L4]
            
            C40exp += [W_k1L4*W12_L0 + W_L0*W12_k1L4,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L4),\
                    0.5*((1/9.*W['22']+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L4),\
                    0.5*(W_k1L4_k2L2*W12_L0 + W_k2L2*W12_k1L4),\
                    0.5*(W_k1L4_k2L4*W12_L0 + W_k2L4*W12_k1L4),\
                    np.conj(W12_k1L4)*W12_L0]
            
            C42exp = [Wc_k2L2*W_k1L4 + Wc_L0*W_k1L4_k2L2,\
                    Wc_k2L2*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L22,\
                    Wc_k2L2*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L24,\
                    Wc_k1L2_k2L2*W_k1L4 + Wc_k1L2*W_k1L4_k2L2,\
                    Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,\
                    Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,\
                    Wc_k1L4_k2L2*W_k1L4 + Wc_k1L4*W_k1L4_k2L2,\
                    Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,\
                    Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]
            
            C42exp += [W_k1L4*W12_k2L2 + W_k2L2*W12_k1L4+\
                    W_k1L4_k2L2*W12_L0+W['22']*W12_k1L4_k2L2,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4\
        +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L4_k2L2),\
        0.5*((1/9.*W['22']+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4\
    +(1/9.*W_k2L2+100/693.*W_k1L2_k2L2+162/1001.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L4_k2L2),\
    0.5*(W_k1L4_k2L2*W12_k2L2+(1/5.*W['22']+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L4\
    +(1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L2),\
    0.5*(W_k1L4_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
    +(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L2),\
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
                
        for i in range(0,2*3+1):
            if(i+Nbin-3>=nBins or i+Nbin-3<0): 
                avgW00[i] *= 0
                avgW22[i] *= 0
                avgW44[i] *= 0
                avgW20[i] *= 0
                avgW40[i] *= 0
                avgW42[i] *= 0
                continue

            avgW00[i] = avgW00[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
            avgW22[i] = avgW22[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
            avgW44[i] = avgW44[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
            avgW20[i] = avgW20[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
            avgW40[i] = avgW40[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
            avgW42[i] = avgW42[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
            
        avgWij[:,:,0] =    2.*np.real(avgW00)
        avgWij[:,:,1] =   25.*np.real(avgW22)
        avgWij[:,:,2] =   81.*np.real(avgW44)
        avgWij[:,:,3] = 5.*2.*np.real(avgW20)
        avgWij[:,:,4] = 9.*2.*np.real(avgW40)
        avgWij[:,:,5] =   45.*np.real(avgW42)
        
        return avgWij

    @staticmethod
    def _compute_window_kernel_row_original(args):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.'''
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [2*delta_k_max+1,15,6] dimensions.
        #    The first dim corresponds to the k-bin of k2
        #    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

        k1_bin_index, Bin_kmodes = args

        I22 = shared_params['I22']

        BoxSize = shared_params['BoxSize']
        kfun = 2*np.pi/BoxSize,

        nBins = shared_params['kbins']
        kBinWidth = shared_params['dk']
        kedges = shared_params['kedges']

        Nmesh = shared_params['Nmesh']

        W = shared_w

        # Bin_ModeNum = utils.nmodes(BoxSize**3, kedges[:-1], kedges[1:])
        Bin_ModeNum = shared_params['Nmodes']

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        avgW00 = np.zeros((2*delta_k_max+1, 15), dtype='<c8')
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

        for ik1x, ik1y, ik1z, ik1r in Bin_kmodes:

            if ik1r <= 1e-10:
                k1xh = 0
                k1yh = 0
                k1zh = 0
            else:
                k1xh = ik1x/ik1r
                k1yh = ik1y/ik1r
                k1zh = ik1z/ik1r

            # Build a 3D array of modes around the selected mode
            k2xh = ik1x-iix
            k2yh = ik1y-iiy
            k2zh = ik1z-iiz

            k2r = np.sqrt(k2xh**2 + k2yh**2 + k2zh**2)

            # to decide later which shell the k2 mode belongs to
            k2_bin_index = (k2r*kfun/kBinWidth).astype(int)

            k2r[k2r <= 1e-10] = np.inf

            k2xh /= k2r
            k2yh /= k2r
            k2zh /= k2r
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

            k1k2W12 = np.conj(W['12xxxx'])*(k1xh*k2xh)**2 + np.conj(W['12yyyy'])*(k1yh*k2yh)**2 + np.conj(W['12zzzz'])*(k1zh*k2zh)**2 \
                + np.conj(W['12xxxy'])*(k1xh*k1yh*k2xh**2 + k1xh**2*k2xh*k2yh)*2 \
                + np.conj(W['12xxxz'])*(k1xh*k1zh*k2xh**2 + k1xh**2*k2xh*k2zh)*2 \
                + np.conj(W['12yyyz'])*(k1yh*k1zh*k2yh**2 + k1yh**2*k2yh*k2zh)*2 \
                + np.conj(W['12yzzz'])*(k1zh*k1yh*k2zh**2 + k1zh**2*k2zh*k2yh)*2 \
                + np.conj(W['12xyyy'])*(k1yh*k1xh*k2yh**2 + k1yh**2*k2yh*k2xh)*2 \
                + np.conj(W['12xzzz'])*(k1zh*k1xh*k2zh**2 + k1zh**2*k2zh*k2xh)*2 \
                + np.conj(W['12xxyy'])*(k1xh**2*k2yh**2 + k1yh**2*k2xh**2 + 4.*k1xh*k1yh*k2xh*k2yh) \
                + np.conj(W['12xxzz'])*(k1xh**2*k2zh**2 + k1zh**2*k2xh**2 + 4.*k1xh*k1zh*k2xh*k2zh) \
                + np.conj(W['12yyzz'])*(k1yh**2*k2zh**2 + k1zh**2*k2yh**2 + 4.*k1yh*k1zh*k2yh*k2zh) \
                + np.conj(W['12xyyz'])*(k1xh*k1zh*k2yh**2 + k1yh**2*k2xh*k2zh + 2.*k1yh*k2yh*(k1zh*k2xh + k1xh*k2zh))*2 \
                + np.conj(W['12xxyz'])*(k1yh*k1zh*k2xh**2 + k1xh**2*k2yh*k2zh + 2.*k1xh*k2xh*(k1zh*k2yh + k1yh*k2zh))*2 \
                + np.conj(W['12xyzz'])*(k1yh*k1xh*k2zh**2 + k1zh**2*k2yh *
                               k2xh + 2.*k1zh*k2zh*(k1xh*k2yh + k1yh*k2xh))*2

            xxW12 = np.conj(W['12xx'])*k1xh**2 + np.conj(W['12yy'])*k1yh**2 + np.conj(W['12zz'])*k1zh**2 \
                + 2.*np.conj(W['12xy'])*k1xh*k1yh + 2.*np.conj(W['12yz']) * \
                k1yh*k1zh + 2.*np.conj(W['12xz'])*k1zh*k1xh

            W12c_L0 = np.conj(W['12'])
            W12_k1L2 = 1.5*xxW12 - 0.5*np.conj(W['12'])
            W12_k1L4 = 35./8.*(np.conj(W['12xxxx'])*k1xh**4 + np.conj(W['12yyyy'])*k1yh**4 + np.conj(W['12zzzz'])*k1zh**4
                               + 4.*np.conj(W['12xxxy'])*k1xh**3*k1yh + 4.*np.conj(W['12xxxz']) *
                               k1xh**3*k1zh + 4.*np.conj(W['12xyyy'])*k1yh**3*k1xh
                               + 6.*np.conj(W['12xxyy'])*k1xh**2*k1yh**2 + 6.*np.conj(W['12xxzz']) *
                               k1xh**2*k1zh**2 + 6.*np.conj(W['12yyzz'])*k1yh**2*k1zh**2
                               + 12.*np.conj(W['12xxyz'])*k1xh**2*k1yh*k1zh + 12.*np.conj(W['12xyyz'])*k1yh**2*k1xh*k1zh + 12.*np.conj(W['12xyzz'])*k1zh**2*k1xh*k1yh) \
                - 5./2.*W12_k1L2 - 7./8.*W12c_L0

            W12_k1L4_k2L2 = 2/7.*W12_k1L2 + 20/77.*W12_k1L4
            W12_k1L4_k2L4 = 1/9.*W12c_L0 + 100/693.*W12_k1L2 + 162/1001.*W12_k1L4

            W12_k2L2 = 1.5*(np.conj(W['12xx'])*k2xh**2 + np.conj(W['12yy'])*k2yh**2 + np.conj(W['12zz'])*k2zh**2
                            + 2.*np.conj(W['12xy'])*k2xh*k2yh + 2.*np.conj(W['12yz'])*k2yh*k2zh + 2.*np.conj(W['12xz'])*k2zh*k2xh) - 0.5*np.conj(W['12'])

            W12_k2L4 = 35./8.*(np.conj(W['12xxxx'])*k2xh**4 + np.conj(W['12yyyy'])*k2yh**4 + np.conj(W['12zzzz'])*k2zh**4
                               + 4.*np.conj(W['12xxxy'])*k2xh**3*k2yh + 4.*np.conj(W['12xxxz']) *
                               k2xh**3*k2zh + 4.*np.conj(W['12xyyy'])*k2yh**3*k2xh
                               + 4.*np.conj(W['12yyyz'])*k2yh**3*k2zh + 4.*np.conj(W['12xzzz']) *
                               k2zh**3*k2xh + 4.*np.conj(W['12yzzz'])*k2zh**3*k2yh
                               + 6.*np.conj(W['12xxyy'])*k2xh**2*k2yh**2 + 6.*np.conj(W['12xxzz']) *
                               k2xh**2*k2zh**2 + 6.*np.conj(W['12yyzz'])*k2yh**2*k2zh**2
                               + 12.*np.conj(W['12xxyz'])*k2xh**2*k2yh*k2zh + 12.*np.conj(W['12xyyz'])*k2yh**2*k2xh*k2zh + 12.*np.conj(W['12xyzz'])*k2zh**2*k2xh*k2yh) \
                - 5./2.*W12_k2L2 - 7./8.*W12c_L0

            W12_k1L2_k2L2 = 9./4.*k1k2W12 - 3./4.*xxW12 - 1./2.*W12_k2L2

            W_k1L2_Sumk2L22 = 1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24 = 2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22 = 1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24 = 2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44 = 1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4

            C00exp = [Wc_L0 * W_L0, Wc_L0 * W_k2L2, Wc_L0 * W_k2L4,
                      Wc_k1L2*W_L0, Wc_k1L2*W_k2L2, Wc_k1L2*W_k2L4,
                      Wc_k1L4*W_L0, Wc_k1L4*W_k2L2, Wc_k1L4*W_k2L4]

            C00exp += [2.*W_L0 * W12c_L0, W_k1L2*W12c_L0,         W_k1L4 * W12c_L0,
                       W_k2L2*W12c_L0, W_k2L4*W12c_L0, np.conj(W12c_L0)*W12c_L0]

            C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,
                      Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,
                      Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,
                      Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,
                      Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,
                      Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,
                      Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,
                      Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,
                      Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]

            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2 + W_k1L2_k2L2*W12c_L0+W_L0*W12_k1L2_k2L2,

                       0.5*((1/5.*W_L0+2/7.*W_k1L2 + 18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2
                            + (1/5.*W_k2L2+2/7.*W_k1L2_k2L2 + 18/35.*W_k1L4_k2L2)*W12c_L0 + W_k1L2*W12_k1L2_k2L2),

                       0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2
                            + (2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12c_L0 + W_k1L4*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L2*W12_k2L2 + (1/5.*W_L0 + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L2
                            + (1/5.*W_k1L2 + 2/7.*W_k1L2_k2L2 + 18/35.*W_k1L2_k2L4)*W12c_L0 + W_k2L2*W12_k1L2_k2L2),

                       0.5*(W_k1L2_k2L4*W12_k2L2 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L2
                            + W_k2L4*W12_k1L2_k2L2 + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L2_k2L4)*W12c_L0),

                       np.conj(W12_k1L2_k2L2)*W12c_L0 + np.conj(W12_k1L2)*W12_k2L2]

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
                       + W_k1L4_k2L4*W12c_L0 + W_L0 * W12_k1L4_k2L4,

                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4
                            + (2/7.*W_k1L2_k2L4 + 20/77.*W_k1L4_k2L4)*W12c_L0 + W_k1L2 * W12_k1L4_k2L4),

                       0.5*((1/9.*W_L0 + 100/693.*W_k1L2 + 162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4
                            + (1/9.*W_k2L4 + 100/693.*W_k1L2_k2L4 + 162/1001.*W_k1L4_k2L4)*W12c_L0 + W_k1L4 * W12_k1L4_k2L4),

                       0.5*(W_k1L4_k2L2*W12_k2L4 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L4 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12c_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L4 + (1/9.*W_L0 + 100/693.*W_k2L2 + 162/1001.*W_k2L4)*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L4 + (1/9.*W_k1L4 + 100/693.*W_k1L4_k2L2 + 162/1001.*W_k1L4_k2L4)*W12c_L0),

                       np.conj(W12_k1L4_k2L4)*W12c_L0 + np.conj(W12_k1L4)*W12_k2L4]  # 1/(nbar)^2

            C20exp = [Wc_L0 * W_k1L2,   Wc_L0*W_k1L2_k2L2, Wc_L0 * W_k1L2_k2L4,
                      Wc_k1L2*W_k1L2, Wc_k1L2*W_k1L2_k2L2, Wc_k1L2*W_k1L2_k2L4,
                      Wc_k1L4*W_k1L2, Wc_k1L4*W_k1L2_k2L2, Wc_k1L4*W_k1L2_k2L4]

            C20exp += [W_k1L2*W12c_L0 + W['22']*W12_k1L2,
                       0.5*((1/5.*W['22'] + 2/7.*W_k1L2 + 18 /
                            35.*W_k1L4)*W12c_L0 + W_k1L2*W12_k1L2),
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)
                            * W12c_L0 + W_k1L4*W12_k1L2),
                       0.5*(W_k1L2_k2L2*W12c_L0 + W_k2L2*W12_k1L2),
                       0.5*(W_k1L2_k2L4*W12c_L0 + W_k2L4*W12_k1L2),
                       np.conj(W12_k1L2)*W12c_L0]

            C40exp = [Wc_L0*W_k1L4,   Wc_L0 * W_k1L4_k2L2, Wc_L0 * W_k1L4_k2L4,
                      Wc_k1L2*W_k1L4, Wc_k1L2*W_k1L4_k2L2, Wc_k1L2*W_k1L4_k2L4,
                      Wc_k1L4*W_k1L4, Wc_k1L4*W_k1L4_k2L2, Wc_k1L4*W_k1L4_k2L4]

            C40exp += [W_k1L4*W12c_L0 + W['22']*W12_k1L4,
                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)
                            * W12c_L0 + W_k1L2*W12_k1L4),
                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2+162 /
                            1001.*W_k1L4)*W12c_L0 + W_k1L4*W12_k1L4),
                       0.5*(W_k1L4_k2L2*W12c_L0 + W_k2L2*W12_k1L4),
                       0.5*(W_k1L4_k2L4*W12c_L0 + W_k2L4*W12_k1L4),
                       np.conj(W12_k1L4)*W12c_L0]

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
                       + W_k1L4_k2L2*W12c_L0 + W['22']*W12_k1L4_k2L2,

                       0.5*((2/7.*W_k1L2 + 20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4
                            + (2/7.*W_k1L2_k2L2 + 20/77.*W_k1L4_k2L2)*W12c_L0 + W_k1L2 * W12_k1L4_k2L2),

                       0.5*((1/9.*W['22'] + 100/693.*W_k1L2 + 162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4
                            + (1/9.*W_k2L2 + 100/693.*W_k1L2_k2L2 + 162/1001.*W_k1L4_k2L2)*W12c_L0 + W_k1L4*W12_k1L4_k2L2),

                       0.5*(W_k1L4_k2L2*W12_k2L2 + (1/5.*W['22'] + 2/7.*W_k2L2 + 18/35.*W_k2L4)*W12_k1L4
                            + W_k2L2*W12_k1L4_k2L2 + (1/5.*W_k1L4 + 2/7.*W_k1L4_k2L2 + 18/35.*W_k1L4_k2L4)*W12c_L0),

                       0.5*(W_k1L4_k2L4*W12_k2L2 + (2/7.*W_k2L2 + 20/77.*W_k2L4)*W12_k1L4
                            + W_k2L4*W12_k1L4_k2L2 + (2/7.*W_k1L4_k2L2 + 20/77.*W_k1L4_k2L4)*W12c_L0),

                       np.conj(W12_k1L4_k2L2)*W12c_L0+np.conj(W12_k1L4)*W12_k2L2]  # 1/(nbar)^2
            
            for delta_k in range(-delta_k_max, delta_k_max + 1):
                # k2_bin_index has shape (Nmesh, Nmesh, Nmesh)
                # k1_bin_index is a scalar
                modes = (k2_bin_index - k1_bin_index == delta_k)

                # Iterating over terms (m,m') that will multiply P_m(k1)*P_m'(k2) in the sum
                for term in range(15):
                    avgW00[delta_k + delta_k_max, term] += np.sum(C00exp[term][modes])
                    avgW22[delta_k + delta_k_max, term] += np.sum(C22exp[term][modes])
                    avgW44[delta_k + delta_k_max, term] += np.sum(C44exp[term][modes])
                    avgW20[delta_k + delta_k_max, term] += np.sum(C20exp[term][modes])
                    avgW40[delta_k + delta_k_max, term] += np.sum(C40exp[term][modes])
                    avgW42[delta_k + delta_k_max, term] += np.sum(C42exp[term][modes])

        for i in range(0, 2*delta_k_max + 1):
            if (i + k1_bin_index - delta_k_max >= nBins or i + k1_bin_index - delta_k_max < 0):
                avgW00[i] = 0
                avgW22[i] = 0
                avgW44[i] = 0
                avgW20[i] = 0
                avgW40[i] = 0
                avgW42[i] = 0
            else:
                avgW00[i] /= kmodes_sampled*Bin_ModeNum[k1_bin_index + i - delta_k_max]*I22**2
                avgW22[i] /= kmodes_sampled*Bin_ModeNum[k1_bin_index + i - delta_k_max]*I22**2
                avgW44[i] /= kmodes_sampled*Bin_ModeNum[k1_bin_index + i - delta_k_max]*I22**2
                avgW20[i] /= kmodes_sampled*Bin_ModeNum[k1_bin_index + i - delta_k_max]*I22**2
                avgW40[i] /= kmodes_sampled*Bin_ModeNum[k1_bin_index + i - delta_k_max]*I22**2
                avgW42[i] /= kmodes_sampled*Bin_ModeNum[k1_bin_index + i - delta_k_max]*I22**2

        def l_factor(l1, l2): return (2*l1 + 1) * \
            (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)

        avgWij = np.zeros((2*delta_k_max+1, 15, 6))

        avgWij[:, :, 0] = l_factor(0,0)*np.real(avgW00)
        avgWij[:, :, 1] = l_factor(2,2)*np.real(avgW22)
        avgWij[:, :, 2] = l_factor(4,4)*np.real(avgW44)
        avgWij[:, :, 3] = l_factor(2,0)*np.real(avgW20)
        avgWij[:, :, 4] = l_factor(4,0)*np.real(avgW40)
        avgWij[:, :, 5] = l_factor(4,2)*np.real(avgW42)

        return avgWij

    @property
    def shotnoise(self):
        return self.I('12')/self.I('22')

    @property
    def nbar(self):
        return 1/self.shotnoise
