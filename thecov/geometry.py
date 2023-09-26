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

import os
from abc import ABC
import itertools as itt
import multiprocessing as mp
import logging
import warnings

import numpy as np
from scipy import fft

from tqdm import tqdm as shell_tqdm

from . import base, utils

__all__ = ['BoxGeometry', 'SurveyGeometry']

# Window functions needed for Gaussian covariance calculation
W_LABELS = ['12', '12xx', '12xy', '12xz', '12yy', '12yz', '12zz', '12xxxx', '12xxxy', '12xxxz', '12xxyy', '12xxyz', '12xxzz', '12xyyy', '12xyyz', '12xyzz', '12xzzz', '12yyyy', '12yyyz', '12yyzz', '12yzzz',
            '12zzzz', '22', '22xx', '22xy', '22xz', '22yy', '22yz', '22zz', '22xxxx', '22xxxy', '22xxxz', '22xxyy', '22xxyz', '22xxzz', '22xyyy', '22xyyz', '22xyzz', '22xzzz', '22yyyy', '22yyyz', '22yyzz', '22yzzz', '22zzzz']


class Geometry(ABC):

    def save_attributes(self, filename, attrs):
        np.savez(filename if filename.strip()
                 [-4:] == '.npz' else f'{filename}.npz', **{a: getattr(self, a) for a in attrs})

    def load_attributes(self, filename, attrs=None):
        with np.load(filename, mmap_mode='r') as data:
            if attrs is None:
                attrs = data.files
            for a in attrs:
                setattr(self, a, data[a])



class BoxGeometry(Geometry):
    '''Class that represents the geometry of a periodic cubic box.

    Attributes
    ----------
    boxsize : float
        Size of the box.
    nmesh : int
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
    logger = logging.getLogger('BoxGeometry')

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
        return 1. / self.nbar

    @shotnoise.setter
    def shotnoise(self, shotnoise):
        '''Sets the Poissonian shotnoise of the sample and nbar = 1/shotnoise.

        Parameters
        ----------
        shotnoise : float
            Shotnoise of the sample.'''
        self.nbar = 1. / shotnoise

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
        bin_volume = np.diff(self.cosmo.comoving_radial_distance(self.zedges)**3)
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

        self.volume = self.fsky * 4. / 3. * np.pi * \
            (self.cosmo.comoving_radial_distance(zmax)**3 -
             self.cosmo.comoving_radial_distance(zmin)**3)

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
        self.logger.info(f'Effective volume: {self.volume:.3e} (Mpc/h)^3')

        bin_volume = self.fsky * \
            np.diff(self.cosmo.comoving_radial_distance(self.zedges)**3)
        self.nbar = np.average(self.nz, weights=bin_volume)
        self.logger.info(f'Estimated nbar: {self.nbar:.3e} (Mpc/h)^-3')

    def set_randoms(self, randoms, alpha=1.0, bins=None, fsky=None):
        '''Estimates the effective volume and number density of the box based on a
        provided catalog of randoms.

        Parameters
        ----------
        randoms : array_like
            Catalog of randoms.
        alpha : float, optional
            Factor to multiply the number density of the randoms. Default is 1.0.
        '''
        from mockfactory import RedshiftDensityInterpolator

        if fsky is None:
            import healpy as hp

            nside = 512
            hpixel = hp.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)
            unique_hpixels = np.unique(hpixel)
            self.fsky = len(unique_hpixels) / hp.nside2npix(nside)

            self.logger.info(f'fsky estimated from randoms: {self.fsky:.3f}')
        else:
            self.fsky = fsky

        nz_hist = RedshiftDensityInterpolator(z=randoms['Z'], bins=bins, fsky=self.fsky, distance=self.cosmo.comoving_radial_distance)
        self.set_nz(zedges=nz_hist.edges, nz=nz_hist.nbar * alpha)

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
            self.logger.info('Cosmology object not set. Using fiducial cosmology DESI.')
            from cosmoprimo.fiducial import DESI
            self._cosmo = DESI()
        return self._cosmo

    @cosmo.setter
    def cosmo(self, cosmo):
        self._cosmo = cosmo


class SurveyGeometry(Geometry, base.FourierBinned):
    '''Class that represents the geometry of a survey in cut-sky.

    Warning
    -------
    Does not run with MPI.

    Attributes
    ----------
    randoms : Catalog
        Catalog of randoms.
    nmesh : int
        Number of mesh points in each dimension to be used in the calculation of the FFTs.
    boxsize : float
        Size of the box.
    boxpad : float
        Box padding.
    delta_k_max : int
        Number of k bins to be calculated each side of the diagonal.
    kmodes_sampled : int
        Number of k modes to be sampled in the calculation of the window kernels.
    alpha : float
        Factor to multiply the number density of the randoms. Default is 1.0.
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
    logger = logging.getLogger('SurveyGeometry')

    def __init__(self, randoms, nmesh=None, boxsize=None, boxpad=2.0, kmax_mask=0.05, delta_k_max=3, kmodes_sampled=400, alpha=1.0, tqdm=shell_tqdm, **kwargs):

        base.FourierBinned.__init__(self)

        self.alpha = alpha
        self.delta_k_max = delta_k_max
        self.kmodes_sampled = kmodes_sampled

        self._shotnoise = None

        self.tqdm = tqdm

        self._W = {}
        self._I = {}

        from mockfactory import Catalog
        from pypower import CatalogMesh
        self._randoms = Catalog(randoms)
        self._ngals = self.randoms.size * self.alpha  # not used
        for name in ['WEIGHT', 'WEIGHT_FKP']:
            if name not in self._randoms: self._randoms[name] = np.ones(self._randoms.size, dtype='f8')
        if 'NZ' not in self._randoms:
            from mockfactory import RedshiftDensityInterpolator
            import healpy as hp
            nside = 512
            distance = np.sqrt(np.sum(randoms['POSITION']**2))
            xyz = randoms['POSITION'] / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels = np.unique(hpixel)
            fsky = len(unique_hpixels) / hp.nside2npix(nside)
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = RedshiftDensityInterpolator(z=distance, fsky=fsky)
            randoms['NZ'] = self.alpha * nbar(distance)

        self._mesh = CatalogMesh(data_positions=self._randoms['POSITION'], data_weights=self._randoms['WEIGHT'],
                                 position_type='pos', nmesh=nmesh, boxsize=boxsize, boxpad=boxpad, dtype='c16',
                                 **{'interlacing': 3, 'resampler': 'tsc', **kwargs})
        self.logger.info(f'Using box size {self._mesh.boxsize}, box center {self._mesh.boxcenter} and nmesh {self._mesh.nmesh}.')
        self.boxsize = self._mesh.boxsize[0]
        self.nmesh = self._mesh.nmesh[0]
        assert np.allclose(self._mesh.boxsize, self.boxsize) and np.all(self._mesh.nmesh, self.nmesh)

        k_nyquist = np.pi * self.mesh.nmesh / self._mesh.boxsize
        self.logger.info(f'Nyquist wavelength of window FFTs = {k_nyquist}.')

        if np.any(k_nyquist < kmax_mask):
            warnings.warn('Nyquist frequency smaller than required kmax_mask = {kmax_mask}.')

        self.logger.info(f'Average of {self._mesh.data_size1 * (self.boxsize / self.nmesh)**3} objects per voxel.')

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
            self._randoms[f'W{w}'] = self._randoms['NZ']**(int(w[0])-1) * (self._randoms['WEIGHT'] * self._randoms['WEIGHT_FKP'])**int(w[1])
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
            self._I[w] = self._randoms[f'W{w}'].sum() * self.alpha
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
            self.compute_cartesian_fft(w.replace("x", "").replace("y", "").replace("z", ""))
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

        self._ikgrid = []
        for i in range(3):
            iik = np.arange(self.nmesh)
            iik[iik >= self.nmesh // 2] -= self.nmesh
            self._ikgrid.append(iik)

        def get_mesh(weights):
            toret = self._mesh.clone(data_positions=self.randoms['position'], data_weights=weights, position_type='pos').to_mesh(compensate=True).r2c()
            toret *= self.nmesh**3 * self.alpha
            return toret.value

        x = self.randoms['position'].T

        with self.tqdm(total=22, desc=f'Computing moments of W{w}') as pbar:
            self.set_cartesian_fft(f'W{w}', get_mesh(f'W{w}'))
            self.W(w)
            self.I(w)
            pbar.update(1)

            for (i, i_label), (j, j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
                label = f'W{w}{i_label}{j_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * \
                    x[i] * x[j] / (x[0]**2 + x[1]**2 + x[2]**2)
                self.set_cartesian_fft(label, get_mesh(label))

                pbar.update(1)

            for (i, i_label), (j, j_label), (k, k_label), (l, l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
                label = f'W{w}{i_label}{j_label}{k_label}{l_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * x[i] * \
                    x[j] * x[k] * x[l] / (x[0]**2 + x[1]**2 + x[2]**2)**2
                self.set_cartesian_fft(label, get_mesh(label))

                pbar.update(1)

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
            self._W[w] = np.frombuffer(mp.RawArray('d', 2 * self.nmesh.prod()))\
                           .view(np.complex128)\
                           .reshape(*self.nmesh)
        self._W[w][:, :, :] = W

    def save_cartesian_ffts(self, filename):
        '''Save the FFTs of the window functions Wij to a file.

        Parameters
        ----------
        filename : str
            Name of the file to save the FFTs of the window functions Wij.
        '''
        np.savez(filename if filename.endswith('.npz') else f'{filename}.npz', **{f'W{k.replace("W", "")}': self._W[k] for k in self._W}, **{
                 f'I{k.replace("I", "")}': self._I[k] for k in self._I}, **{name: getattr(self, name) for name in ['boxsize', 'nmesh', 'alpha']})

    def load_cartesian_ffts(self, filename):
        '''Load the FFTs of the window functions Wij from a file.

        Parameters
        ----------
        filename : str
            Name of the file to load the FFTs of the window functions Wij from.
        '''
        with np.load(filename, mmap_mode='r') as data:

            for name in ['boxsize', 'nmesh', 'alpha']:
                setattr(self, name, data[name])

            for f in data.files:
                if f[0] == 'W':
                    self.set_cartesian_fft(f, data[f])
                elif f[0] == 'I':
                    self._I[f[1:]] = data[f]

    @property
    def randoms(self):
        return self._randoms

    @property
    def ngals(self):
        return self._ngals

    @ngals.setter
    def ngals(self, ngals):
        self._ngals = ngals

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

        kfun = 2 * np.pi / self.boxsize

        kmodes = np.array([[utils.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(
                           self.kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])

        Nmodes = utils.nmodes(self.boxsize**3, self.kedges[:-1], self.kedges[1:])

        # sample kmodes from each k1 bin
        # kmodes, Nmodes = utils.sample_kmodes(self.kmax, self.dk, self.boxsize, self.kmodes_sampled)

        # kmodes, Nmodes = utils.sample_from_cube(self.kmax/kfun, self.dk/kfun, self.kmodes_sampled)

        # Note: as the window falls steeply with k, only low-k regions are needed for the calculation.
        # Therefore, high-k modes in the FFTs can be discarded

        self.compute_cartesian_fft('W12')
        self.compute_cartesian_fft('W22')

        init_params = {
            'I22': self.I('22'),
            'boxsize': self.boxsize,
            'kbins': self.kbins,
            'dk': self.dk,
            'nmesh': self.nmesh,
            'ikgrid': self._ikgrid,
            'delta_k_max': self.delta_k_max,
            'nmodes': Nmodes,
        }

        def init_worker(*args):
            global shared_w
            global shared_params
            shared_w = {}
            for w, l in zip(args, W_LABELS):
                shared_w[l] = np.frombuffer(w).view(
                    np.complex128).reshape(*self.nmesh)
            shared_params = args[-1]

        num_threads = getattr(self, 'num_threads', os.environ.get('OMP_NUM_THREADS', os.cpu_count()))

        # Calls the function _compute_window_kernel_row in parallel for each k1 bin
        with mp.Pool(processes=num_threads, initializer=init_worker, initargs=[*[self._W[w] for w in W_LABELS], init_params]) as pool:
            self.WinKernel = np.array(list(self.tqdm(pool.imap(self._compute_window_kernel_row, enumerate(
                kmodes)), total=len(kmodes), desc="Computing window kernels")))

        # self.WinKernel = np.array([
        #     self._compute_window_kernel_row(Nbin)
        #     for Nbin in self.tqdm(range(self.kbins), total=self.kbins, desc="Computing window kernels")])

    def save_window_kernels(self, filename):
        '''Save the window kernels to a file.

        Parameters
        ----------
        filename : str
            Name of the file to save the window kernels.'''
        self.save_attributes(filename, ['alpha',
                                       'delta_k_max',
                                       'kmodes_sampled',
                                       'shotnoise',
                                       'ngals',
                                       'boxsize',
                                       'nmesh',
                                       'dk',
                                       'kmax',
                                       'kmin',
                                       'WinKernel'])

    def load_window_kernels(self, filename):
        '''Load the window kernels from a file.

        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''
        self.load_attributes(filename)

    @classmethod
    def from_window_kernels_file(cls, filename):
        '''Create geometry object from window kernels file.

        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''
        geometry = cls.__new__(cls)
        geometry.load_window_kernels(filename)
        return geometry

    @staticmethod
    def _compute_window_kernel_row(args):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.'''
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [2*delta_k_max+1,15,6] dimensions.
        #    The first dim corresponds to the k-bin of k2
        #    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

        k1_bin_index, bin_kmodes = args
        I22 = shared_params['I22']
        boxsize = shared_params['boxsize']
        kfun = 2 * np.pi / boxsize
        nbins = shared_params['kbins']
        dk = shared_params['dk']

        W = shared_w

        # nmodes = utils.nmodes(boxsize**3, kedges[:-1], kedges[1:])
        nmodes = shared_params['nmodes']

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        avgW00 = np.zeros((2*delta_k_max+1, 15), dtype='<c8')
        avgW22 = avgW00.copy()
        avgW44 = avgW00.copy()
        avgW20 = avgW00.copy()
        avgW40 = avgW00.copy()
        avgW42 = avgW00.copy()

        iix, iiy, iiz = np.meshgrid(*shared_params['ikgrid'], indexing='ij')

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)

        kmodes_sampled = len(bin_kmodes)

        for ik1x, ik1y, ik1z, ik1r in bin_kmodes:

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
            k2_bin_index = (k2r * kfun / dk).astype(int)

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
                # k2_bin_index has shape (nmesh, nmesh, nmesh)
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
            if (i + k1_bin_index - delta_k_max >= nbins or i + k1_bin_index - delta_k_max < 0):
                avgW00[i] = 0
                avgW22[i] = 0
                avgW44[i] = 0
                avgW20[i] = 0
                avgW40[i] = 0
                avgW42[i] = 0
            else:
                avgW00[i] /= kmodes_sampled*nmodes[k1_bin_index + i - delta_k_max]*I22**2
                avgW22[i] /= kmodes_sampled*nmodes[k1_bin_index + i - delta_k_max]*I22**2
                avgW44[i] /= kmodes_sampled*nmodes[k1_bin_index + i - delta_k_max]*I22**2
                avgW20[i] /= kmodes_sampled*nmodes[k1_bin_index + i - delta_k_max]*I22**2
                avgW40[i] /= kmodes_sampled*nmodes[k1_bin_index + i - delta_k_max]*I22**2
                avgW42[i] /= kmodes_sampled*nmodes[k1_bin_index + i - delta_k_max]*I22**2

        def l_factor(l1, l2):
            return (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)

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
        if self._shotnoise is None:
            return (1 + self.alpha)*self.I('12')/self.I('22')
        return self._shotnoise

    @shotnoise.setter
    def shotnoise(self, shotnoise):
        self._shotnoise = shotnoise

    @property
    def nbar(self):
        return 1 / self.shotnoise

    @nbar.setter
    def nbar(self, nbar):
        self.shotnoise = 1 / nbar
