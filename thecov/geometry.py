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

import os, time, pickle
from abc import ABC
import itertools as itt
import multiprocessing as mp
import logging

import numpy as np

from tqdm import tqdm as shell_tqdm

from . import base, utils, math

__all__ = ['BoxGeometry',
           'SurveyGeometry']

# Window functions needed for Gaussian covariance calculation
W_LABELS = ['12', '12xx', '12xy', '12xz', '12yy', '12yz', '12zz', '12xxxx', '12xxxy', '12xxxz', '12xxyy', '12xxyz', '12xxzz', '12xyyy', '12xyyz', '12xyzz', '12xzzz', '12yyyy', '12yyyz', '12yyzz', '12yzzz',
            '12zzzz', '22', '22xx', '22xy', '22xz', '22yy', '22yz', '22zz', '22xxxx', '22xxxy', '22xxxz', '22xxyy', '22xxyz', '22xxzz', '22xyyy', '22xyyz', '22xyzz', '22xzzz', '22yyyy', '22yyyz', '22yyzz', '22yzzz', '22zzzz']

class Geometry(ABC):

    def save(self, filename):
        utils.mkdir(os.path.dirname(filename))
        
        ext = os.path.splitext(filename)[1]

        start = time.time()
        if ext == '.gz':
            import gzip
            with gzip.open(filename, 'wb') as f:
                pickle.dump(self, f)
        elif ext == '.xz':
            import lzma
            with lzma.open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)

        self.logger.info(f'Saved to {filename} in {time.time() - start:.3f} s.')

    def load(self, filename):
        start = time.time()

        ext = os.path.splitext(filename)[1]
        if ext == '.gz':
            import gzip
            with gzip.open(filename, 'rb') as f:
                obj = pickle.load(f)
        elif ext == '.xz':
            import lzma
            with lzma.open(filename, 'rb') as f:
                obj = pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        
        self.__dict__.update(obj.__dict__)
        self.logger.info(f'Loaded from {filename} in {time.time() - start:.3f} s.')

class BoxGeometry(Geometry):
    '''Class that represents the geometry of a periodic cubic box.

    Attributes
    ----------
    boxsize : float
        Size of the box.
    nmesh : int
        Number of mesh points in each dimension.
    alpha : float
        <number of galaxies>/<number of randoms> in the box.

    Methods
    -------
    set_boxsize
        Set the size of the box.
    set_nmesh
        Set the number of mesh points in each dimension.
    set_alpha
        Set the alpha parameter.
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
        '''Set the effective volume of the box based on the redshift limits of the sample and the fraction of the sky covered.

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
        Catalog of randoms, with (at least) column 'POSITION'. (Optionally: WEIGHT, WEIGHT_FKP, NZ).
    nmesh : int
        Number of mesh points in each dimension to be used in the calculation of the FFTs.
    boxsize : float
        Size of the box.
    boxpad : float
        Box padding.
    kmodes_sampled : int
        Number of k modes to be sampled in the calculation of the window kernels.
    alpha : float
        Factor to multiply the number density of the randoms. Default is 1.0.
    nthreads : int
        Number of threads to be used in the calculation of the window kernels.
    resume_file : str
        Name of the file to save the window kernels as they are calculated. If the file exists, the
        calculation is resumed from it.
    tqdm : callable
        Function to be used for progress bar. Default is tqdm from tqdm package.

    Methods
    -------
    set_kbins(kmin, kmax, dk)
        Set the k bins to be used in the calculation of the covariance.
    compute_window_kernels
        Compute the window kernels to be used in the calculation of the covariance.
    load_resume_file(filename)
        Load the window kernels from a file.

    Notes
    -----
    The window kernels are computed using the method described in [1]_.

    References
    ----------
    .. [1] https://arxiv.org/abs/1910.02914
    '''

    def __init__(self, randoms, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax_window=0.02, kmodes_sampled=5000, alpha=1.0, nthreads=None, resume_file=None, tqdm=shell_tqdm, **kwargs):

        base.FourierBinned.__init__(self)

        self.logger = logging.getLogger('SurveyGeometry')

        self.alpha = alpha
        self.kmodes_sampled = kmodes_sampled

        # self._shotnoise = None
        self.nthreads = nthreads
        if self.nthreads is None:
            self.nthreads = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))

        self.logger.debug(f'{self.nthreads} threads available.')

        self.tqdm = tqdm

        from mockfactory import Catalog
        from pypower import CatalogMesh
        self._randoms = Catalog(randoms)
        for name in ['WEIGHT', 'WEIGHT_FKP']:
            if name not in self._randoms: self._randoms[name] = np.ones(self._randoms.size, dtype='f8')
        if 'NZ' not in self._randoms:
            self.logger.warning('NZ column not found in randoms. Estimating it with RedshiftDensityInterpolator.')
            from mockfactory import RedshiftDensityInterpolator
            import healpy as hp
            nside = 512
            distance = np.sqrt(np.sum(randoms['POSITION']**2, axis=-1))
            xyz = randoms['POSITION'] / distance[:, None]
            hpixel = hp.vec2pix(nside, *xyz.T)
            unique_hpixels = np.unique(hpixel)
            fsky = len(unique_hpixels) / hp.nside2npix(nside)
            self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
            nbar = RedshiftDensityInterpolator(z=distance, fsky=fsky)
            self._randoms['NZ'] = self.alpha * nbar(distance)
        if nmesh is None and cellsize is None:
            # Pick value that will give at least k_mask = kmax_window in the FFTs
            cellsize = np.pi / kmax_window / (1. + 1e-9)

        self._mesh = CatalogMesh(data_positions=self._randoms['POSITION'], data_weights=self._randoms['WEIGHT'],
                                 position_type='pos', nmesh=nmesh, cellsize=cellsize, boxsize=boxsize, boxpad=boxpad, dtype='c16',
                                 **{'interlacing': 3, 'resampler': 'tsc', **kwargs})
        self.logger.info(f'Using box size {self._mesh.boxsize}, box center {self._mesh.boxcenter} and nmesh {self._mesh.nmesh}.')
        self.boxsize = self._mesh.boxsize[0]
        self.nmesh = self._mesh.nmesh[0]
        assert np.allclose(self._mesh.boxsize, self.boxsize) and np.all(self._mesh.nmesh == self.nmesh)

        self.logger.info(f'Fundamental wavenumber of window FFTs = {self.kfun}.')
        self.logger.info(f'Nyquist wavenumber of window FFTs = {self.knyquist}.')

        if self.knyquist < kmax_window:
            self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required kmax_window = {kmax_window}.')

        self.logger.info(f'Average of {self._mesh.data_size / self.nmesh**3} objects per voxel.')

        self.WinKernel = None
        self.WinKernel_error = None

        if resume_file is not None:
            self.set_resume_file(resume_file)
        else:
            self._resume_file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['logger', 'nthreads', 'tqdm', '_randoms', '_mesh', '_resume_file']:
            del state[key]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    @property
    def knyquist(self):
        return np.pi * self.nmesh / self.boxsize
    @property
    def kfun(self):
        return 2 * np.pi / self.boxsize
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self.clean()

    def W_cat(self, W):
        '''Returns the window function Wij for the objects in the catalog.
           If it has not been computed yet, it is computed.

        Parameters
        ----------
        W : str
            Window function label (e.g. '12', '22', '12xx', '22xy', etc.)

        Returns
        -------
        array_like
            Window function Wij.
        '''
        w = W.lower().replace("w", "")

        if f'W{w}' not in self._randoms.columns():
            self._randoms[f'W{w}'] = self._randoms['NZ']**(
                int(w[0])-1) * self._randoms['WEIGHT_FKP']**int(w[1]) * self._randoms[f'WEIGHT'] * self.alpha

        return self._randoms[f'W{w}']

    @utils.cache_method
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
        self.W_cat(w)
        return self._randoms[f'W{w}'].sum().tolist()

    def W(self, W):
        '''Returns an FFT of the window function Wij. If it has not been computed yet, it is computed.

        Parameters
        ----------
        W : str
            Window function label.

        Returns
        -------
        array_like
            FFT of the window function Wij.
        '''
        if not hasattr(self, '_W'):
            self._W = {}
        w = W.lower().replace("w", "")
        if w not in self._W:
            self.W_cat(w)
            if 'x' in w or 'y' in w or 'z' in w:
                self.compute_cartesian_ffts(w)
            else:
                self.set_cartesian_fft(f'W{w}', self.get_fft(f'W{w}'))
        return self._W[w]

    @property
    def kfun(self):
        """Fundamental wavenumber of the box."""
        return 2 * np.pi / self.boxsize

    @property
    def ikgrid(self):
        """Grid of wavenumber indices."""
        ikgrid = []
        for _ in range(3):
            iik = np.arange(self.nmesh)
            iik[iik >= self.nmesh // 2] -= self.nmesh
            ikgrid.append(iik)
        return ikgrid

    def get_mesh(self,label):
        """Returns a mesh object for the window function Wij.
        
        Parameters
        ----------
        label : str
            Window function label.
            
        Returns
        -------
            Mesh
                Mesh object for the window function Wij."""
        weights = self.W_cat(label) if 'w' in label.lower() else self.randoms[label]
        return self._mesh.clone(data_positions=self.randoms['POSITION'],
                                data_weights=weights,
                                position_type='pos',
                                ).to_mesh(compensate=True)

    def get_fft(self, label):
        """Returns the FFT of the window function Wij.
        
        Parameters
        ----------
        label : str
            Window function label.
            
        Returns
        -------
            array_like
                FFT of the window function Wij."""
        toret = self.get_mesh(label).r2c()
        toret *= self.nmesh**3
        return toret.value
    
    def compute_pypower(self, label1, label2=None, *args, **kwargs):
        if label2 is None:
            label2 = label1

        mesh1 = self.get_mesh(label1)
        mesh2 = mesh1 if label2 == label1 else self.get_mesh(label2)

        from pypower import MeshFFTPower
        return MeshFFTPower(mesh1=mesh1, mesh2=mesh2, *args, **kwargs)
        
    def compute_power(self, label1, label2=None, kedges='auto'):
        if label2 is None:
            label2 = label1

        fourier1 = self.get_fft(label1)

        if label1 == label2:
            fourier2 = fourier1
        else:
            fourier2 = self.get_fft(label2)

        power = fourier1 * fourier2.conj()

        kx, ky, kz = np.meshgrid(*self.ikgrid)
        kpower = np.sqrt(kx**2 + ky**2 + kz**2) * self.kfun

        if kedges == 'self':
            kedges = self.kedges
        elif kedges == 'auto':
            kedges = np.arange(0, self.knyquist + self.kfun/2, self.kfun)

        ibin = np.digitize(kpower, kedges)

        pk = np.array([power[ibin == i].mean() for i in range(1, len(kedges))])
        pk = pk.real if label1 == label2 else pk

        return kedges, pk
    
    def compute_Qmus(self, W1, W2=None, sedges=None, *args, **kwargs):
        if W2 is None:
            W2 = W1
        from pycorr import TwoPointCounter

        muedges = np.linspace(-1, 1., 201)

        edges = (np.geomspace(1, 3400., 441) if sedges is None else sedges, muedges)
        
        pos = self.randoms['POSITION'].T

        counts = TwoPointCounter(mode='smu',
                                 edges=edges,
                                 positions1=pos,
                                 positions2=None if W1 == W2 else pos,
                                 weights1=self.W_cat(W1),
                                 weights2=None if W1 == W2 else self.W_cat(W2),
                                 *args, **kwargs)

        v = 2. / 3. * np.pi * edges[0][:, None]**3 * edges[1]
        dv = np.diff(np.diff(v, axis=0), axis=-1)

        Qmus = counts.wcounts/dv

        savg = counts.sepavg()

        return savg, muedges, Qmus
    
    def compute_cartesian_ffts(self, W):
        '''Computes the FFT of the window function Wij.

        Parameters
        ----------
        W : str
            Window function label.
        '''
        w = W.lower().replace('w', '').replace('x', '').replace('y', '').replace('z', '')
        self.W_cat(w)

        x = self.randoms['POSITION'].T

        with self.tqdm(total=22, desc=f'Computing moments of W{w}') as pbar:
            self.set_cartesian_fft(f'W{w}', self.get_fft(f'W{w}'))
            pbar.update(1)

            for (i, i_label), (j, j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
                label = f'W{w}{i_label}{j_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * \
                    x[i] * x[j] / (x[0]**2 + x[1]**2 + x[2]**2)
                self.set_cartesian_fft(label, self.get_fft(label))

                pbar.update(1)

            for (i, i_label), (j, j_label), (k, k_label), (l, l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
                label = f'W{w}{i_label}{j_label}{k_label}{l_label}'
                self.randoms[label] = self.randoms[f'W{w}'] * x[i] * \
                    x[j] * x[k] * x[l] / (x[0]**2 + x[1]**2 + x[2]**2)**2
                self.set_cartesian_fft(label, self.get_fft(label))

                pbar.update(1)

        if self._resume_file is not None:
            self.save(self._resume_file)

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
        # Create object proper for multiprocessing
        self._W[w] = np.frombuffer(mp.RawArray('d', 2 * int(self.nmesh)**3)).view(np.complex128).reshape(self.nmesh, self.nmesh, self.nmesh)
        self._W[w][...] = W

    @property
    def randoms(self):
        return self._randoms

    @property
    def kbins(self):
        return len(self.kmid)

    def compute_window_power(self):

        if self._window_power is not None:
            return self._window_power
        
        self.W('22xx')
        self.W('10xx')
            
        self.logger.info('Computing window power spectra.')

        ikx, iky, ikz = np.meshgrid(*self.ikgrid)
        ikr = np.sqrt(ikx**2 + iky**2 + ikz**2)
        ikr[ikr == 0] = np.inf

        kx, ky, kz = ikx/ikr, iky/ikr, ikz/ikr
        ikr[ikr == np.inf] = 0.
        kr = ikr * self.kfun
    
        W22_L0 = self.W('22')
    
        W22_L2 = 1.5*(self.W('22xx')*kx**2+self.W('22yy')*ky**2+self.W('22zz')*kz**2+2.*self.W('22xy')*kx*ky+2.*self.W('22yz')*ky*kz+2.*self.W('22xz')*kz*kx) - 0.5*W22_L0
                
        W22_L4 = 35./8.*(self.W('22xxxx')*kx**4 +self.W('22yyyy')*ky**4+self.W('22zzzz')*kz**4 \
                +4.*self.W('22xxxy')*kx**3*ky +4.*self.W('22xxxz')*kx**3*kz +4.*self.W('22xyyy')*ky**3*kx \
                +4.*self.W('22yyyz')*ky**3*kz +4.*self.W('22xzzz')*kz**3*kx +4.*self.W('22yzzz')*kz**3*ky \
                +6.*self.W('22xxyy')*kx**2*ky**2+6.*self.W('22xxzz')*kx**2*kz**2+6.*self.W('22yyzz')*ky**2*kz**2 \
                +12.*self.W('22xxyz')*kx**2*ky*kz+12.*self.W('22xyyz')*ky**2*kx*kz +12.*self.W('22xyzz')*kz**2*kx*ky) \
                -5./2.*W22_L2 -7./8.*W22_L0

        W10_L0 = self.W('10')
    
        W10_L2 = 1.5*(self.W('10xx')*kx**2+self.W('10yy')*ky**2+self.W('10zz')*kz**2+2.*self.W('10xy')*kx*ky+2.*self.W('10yz')*ky*kz+2.*self.W('10xz')*kz*kx) - 0.5*W10_L0
                
        W10_L4 = 35./8.*(self.W('10xxxx')*kx**4 +self.W('10yyyy')*ky**4+self.W('10zzzz')*kz**4 \
                +4.*self.W('10xxxy')*kx**3*ky +4.*self.W('10xxxz')*kx**3*kz +4.*self.W('10xyyy')*ky**3*kx \
                +4.*self.W('10yyyz')*ky**3*kz +4.*self.W('10xzzz')*kz**3*kx +4.*self.W('10yzzz')*kz**3*ky \
                +6.*self.W('10xxyy')*kx**2*ky**2+6.*self.W('10xxzz')*kx**2*kz**2+6.*self.W('10yyzz')*ky**2*kz**2 \
                +12.*self.W('10xxyz')*kx**2*ky*kz+12.*self.W('10xyyz')*ky**2*kx*kz +12.*self.W('10xyzz')*kz**2*kx*ky) \
                -5./2.*W10_L2 -7./8.*W10_L0
        
        self.logger.info('Binning window power spectra.')

        if not self.is_kbins_set:
            self.set_kbins(0, self.knyquist, self.kfun)
        
        ibin = np.digitize(kr, self.kedges)
        
        poles = np.array([W22_L0, W22_L2, W22_L4, W10_L0, W10_L2, W10_L4])

        # Computing power spectra
        # stopping at 15 because we only need monopole for W10xW10
        power = [ A*np.conj(B) for A, B in self.tqdm(utils.limit(itt.combinations_with_replacement(poles, 2), 16), total=16,
                                                     desc='Calculating power.' )]
        power.append(kr) # to compute kavg

        power = self.tqdm(power, total=17,
                         desc='Averaging power spectra over bins.')
        
        self._window_power = np.array([[p[ibin == i].real.mean() for i in range(1, self.kbins+1)] for p in power])

        # removing shotnoise from mono x monopole
        self._window_power[0]  -= self.I('34') * self.alpha # W22xW22
        self._window_power[3]  -= self.I('22') * self.alpha # W22xW10
        # not removing from W10xW10
        # self._window_power[15] -= self.I('10') * self.alpha # W10xW10

        for i, (w1,w2) in enumerate(itt.combinations_with_replacement(['22', '22', '22', '10', '10', '10'], 2)):
            if i > 15:
                break
            self._window_power[i] /= self.I(w1)*self.I(w2) # normalizing

        for i, (l1,l2) in utils.limit(enumerate(itt.combinations_with_replacement(2*[0,2,4], 2)), 16):
            if i > 15:
                break
            self._window_power[i] *= (2*l1 + 1)*(2*l2 + 1)
            # Manually setting k=0 modes
            self._window_power[i,0] = 1 if l1 == l2 else 0
        # kavg of first bin
        if np.isnan(self._window_power[-1,0]):
            self._window_power[-1,0] = 0

        if self._resume_file is not None:
            self.save_resume_file(self._resume_file)
            
        self.logger.info('Window power spectra computed.')

        return self._window_power
    
    def get_window_power_interpolators(self):
        if self._window_power is None:
            self.compute_window_power()

        from scipy.interpolate import InterpolatedUnivariateSpline
        kavg = self._window_power[-1]
        return [InterpolatedUnivariateSpline(kavg, Pwin) for Pwin in self._window_power[:-1]]

    def get_window_kernels(self):
        '''Returns the window kernels to be used in the calculation of the covariance.

        Returns
        -------
        array_like
            Window kernels to be used in the calculation of the covariance.
        '''
        if self.WinKernel is None or np.isnan(self.WinKernel).any():
            self.compute_window_kernels()
        return self.WinKernel
    
    @property
    def delta_k_max(self):
        return self.nmesh // 2 - 1

    def compute_window_kernels(self):
        '''Computes the window kernels to be used in the calculation of the covariance.

        Notes
        -----
        The window kernels are computed using the method described in [1]_.

        References
        ----------
        .. [1] https://arxiv.org/abs/1910.02914
        '''

        # sample kmodes from each k1 bin

        # SAMPLE FROM SHELL
        # kfun = 2 * np.pi / self.boxsize
        # kmodes = np.array([[math.sample_from_shell(kmin/kfun, kmax/kfun) for _ in range(
        #                    self.kmodes_sampled)] for kmin, kmax in zip(self.kedges[:-1], self.kedges[1:])])
        # Nmodes = math.nmodes(self.boxsize**3, self.kedges[:-1], self.kedges[1:])

        # SAMPLE FROM CUBE
        # kmodes, Nmodes = math.sample_from_cube(self.kmax/kfun, self.dk/kfun, self.kmodes_sampled)

        # HYBRID SAMPLING
        kmodes, Nmodes =  math.sample_kmodes(kmin=self.kmin,
                                             kmax=self.kmax,
                                             dk=self.dk,
                                             boxsize=self.boxsize,
                                             max_modes=self.kmodes_sampled,
                                             k_shell_approx=0.1)

        assert len(kmodes) == self.kbins and len(Nmodes) == self.kbins, \
            f'Error in thecov.utils.sample_kmodes: results should have length {self.kbins}, but had {len(kmodes)}. Parameters were kmin={self.kmin},kmax={self.kmax},dk={self.dk},boxsize={self.boxsize},max_modes={self.kmodes_sampled},k_shell_approx={0.1}).'

        # Calculate window FFTs if they haven't been initialized yet
        self.W('W12xx')
        self.W('W22xx')

        init_params = {
            'boxsize': self.boxsize,
            'dk': self.dk,
            'nmesh': self.nmesh,
            'ikgrid': self.ikgrid,
            'delta_k_max': self.delta_k_max,
        }

        def init_worker(*args):
            global shared_w
            global shared_params
            shared_w = {}
            for w, l in zip(args, W_LABELS):
                shared_w[l] = np.frombuffer(w).view(np.complex128).reshape(self.nmesh, self.nmesh, self.nmesh)
            shared_params = args[-1]

        if self.WinKernel is None and self.WinKernel_error is None:
            # Format is [k1_bins, k2_bins, P_i x P_j term, Cov_ij]
            self.WinKernel = np.empty([self.kbins, 2*self.delta_k_max+1, 15, 6])
            self.WinKernel.fill(np.nan)

            self.WinKernel_error = np.empty([self.kbins, 2*self.delta_k_max+1, 15, 6])
            self.WinKernel_error.fill(np.nan)

        ell_factor = lambda l1,l2: (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)

        last_save = time.time()

        for i, km in self.tqdm(enumerate(kmodes), desc='Computing window kernels', total=self.kbins):

            if self._resume_file is not None:
                # Skip rows that were already computed
                if not np.isnan(self.WinKernel[i,0,0,0]):
                    # self.logger.debug(f'Skipping bin {i} of {self.kbins}.')
                    continue

            init_params['k1_bin_index'] = i + self.kmin//self.dk
            kmodes_sampled = len(km)

            # Splitting kmodes in chunks to be sent to each worker
            chunks = np.array_split(km, self.nthreads)

            with mp.Pool(processes=min(self.nthreads, len(chunks)),
                         initializer=init_worker,
                         initargs=[*[self._W[w] for w in W_LABELS], init_params]) as pool:
                
                results = pool.map(self._compute_window_kernel_row, chunks)

                self.WinKernel[i] = np.sum(results, axis=0) / kmodes_sampled

                std_results = np.std(results, axis=0) / np.sqrt(len(results))
                mean_results = np.mean(results, axis=0)
                mean_results[std_results == 0] = 1
                self.WinKernel_error[i] =  std_results / mean_results
        
                for k2_bin_index in range(0, 2*self.delta_k_max + 1):
                    if (k2_bin_index + i - self.delta_k_max >= self.kbins or k2_bin_index + i - self.delta_k_max < 0):
                        self.WinKernel[i, k2_bin_index, :, :] = 0
                    else:
                        self.WinKernel[i, k2_bin_index, :, :] /= Nmodes[i + k2_bin_index - self.delta_k_max]

            self.WinKernel[i, ..., 0] *= ell_factor(0,0)
            self.WinKernel[i, ..., 1] *= ell_factor(2,2)
            self.WinKernel[i, ..., 2] *= ell_factor(4,4)
            self.WinKernel[i, ..., 3] *= ell_factor(2,0)
            self.WinKernel[i, ..., 4] *= ell_factor(4,0)
            self.WinKernel[i, ..., 5] *= ell_factor(4,2)
            
            if self._resume_file is not None and (time.time() - last_save) > 600:
                self.save(self._resume_file)
                last_save = time.time()
                
        self.logger.info('Window kernels computed.')

        if self._resume_file is not None:
            self.save(self._resume_file)

    def clean(self):
        '''Clean window kernels and power spectra.'''
        self.WinKernel = None
        self.WinKernel_error = None
        self._window_power = None
        self._W = {}
        self._I = {}

    @staticmethod
    def _compute_window_kernel_row(bin_kmodes):
        '''Computes a row of the window kernels. This function is called in parallel for each k1 bin.'''
        # Gives window kernels for L=0,2,4 auto and cross covariance (instead of only L=0 above)

        # Returns an array with [2*delta_k_max+1,15,6] dimensions.
        #    The first dim corresponds to the k-bin of k2
        #    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

        #    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
        #    to obtain the final covariance (see function 'Wij' below)

        #    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

        k1_bin_index = shared_params['k1_bin_index']
        boxsize = shared_params['boxsize']
        kfun = 2 * np.pi / boxsize
        dk = shared_params['dk']

        W = shared_w

        # The Gaussian covariance drops quickly away from diagonal.
        # Only delta_k_max points to each side of the diagonal are calculated.
        delta_k_max = shared_params['delta_k_max']

        WinKernel = np.zeros((2*delta_k_max+1, 15, 6))

        iix, iiy, iiz = np.meshgrid(*shared_params['ikgrid'], indexing='ij')

        k2xh = np.zeros_like(iix)
        k2yh = np.zeros_like(iiy)
        k2zh = np.zeros_like(iiz)

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

            # Expressions below come straight from CovaPT (arXiv:1910.02914)

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
                    WinKernel[delta_k + delta_k_max, term, 0] += np.sum(np.real(C00exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 1] += np.sum(np.real(C22exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 2] += np.sum(np.real(C44exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 3] += np.sum(np.real(C20exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 4] += np.sum(np.real(C40exp[term][modes]))
                    WinKernel[delta_k + delta_k_max, term, 5] += np.sum(np.real(C42exp[term][modes]))
        
        return WinKernel

    def load_resume_file(self, filename):
        '''Load the window kernels from a file.

        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''
        self.logger.info(f'Loading window kernels from {filename}.')
        self.load(filename)
        # Cartesian FFTs need to be loaded through the setter
        for key in self._W:
            self.set_cartesian_fft(key, self._W[key])

    def set_resume_file(self, filename):
        '''Set the resume file for the window kernels.

        Parameters
        ----------
        filename : str
            Name of the file to save the window kernels.
        '''
        self._resume_file = filename

        if self._resume_file is not None:
            try:
                self.load_resume_file(self._resume_file)
                self.logger.warning(f'Loaded resume file {self._resume_file}. This might override your settings. See debug messages for more details on the loaded attributes.')
            except FileNotFoundError:
                self.logger.info(f'File {self._resume_file} not found. Creating resume file.')
                utils.mkdir(os.path.dirname(self._resume_file))
                self.save(self._resume_file)

    @classmethod
    def from_window_kernels_file(cls, filename):
        '''Create geometry object from window kernels file.

        Parameters
        ----------
        filename : str
            Name of the file to load the window kernels from.
        '''
        geometry = cls.__new__(cls)
        geometry.load_resume_file(filename)
        return geometry
    
    @property
    def fkp_shotnoise(self):
        return (1 + self.alpha)*self.I('12')/self.I('22')

    @property
    def shotnoise(self):
        self.logger.warning('Using FKP shotnoise.')
        return self.fkp_shotnoise


