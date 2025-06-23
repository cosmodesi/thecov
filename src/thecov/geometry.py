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

import os, time
import itertools as itt
import multiprocessing as mp
import logging

import numpy as np
import scipy as sp

from tqdm import tqdm as shell_tqdm

import mockfactory
from pypower import CatalogMesh

from . import base, utils, math

__all__ = ['BoxGeometry',
           'SurveyGeometry']

class Geometry(base.BaseClass):
    pass
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


class SurveyGeometry(Geometry, base.LinearBinning):

    def __init__(self, randoms1, alpha1, randoms2=None, alpha2=None, nmesh=None, cellsize=None, boxsize=None, boxpad=2., kmax=0.02, ellmax=4, **kwargs):

        base.LinearBinning.__init__(self)

        self.logger = logging.getLogger('Window')
        self.tqdm = shell_tqdm

        self._alpha1 = alpha1
        self._alpha2 = alpha1 if alpha2 is None else alpha2

        self._kmax = kmax
        self._ellmax = ellmax

        self._randoms = [randoms1]
        self._mesh = []

        if randoms2 is not None:
            self._randoms.append(randoms2)

        largest_boxsize = 0
        largest_nmesh = 0

        for randoms, alpha in zip(self._randoms, [self._alpha1, self._alpha2]):
            if not isinstance(randoms, mockfactory.Catalog):
                randoms = mockfactory.Catalog(randoms)

            # Check if the randoms have weights, otherwise set them to 1
            for name in ['WEIGHT', 'WEIGHT_FKP']:
                if name not in randoms:
                    self.logger.warning(f'{name} column not found in randoms. Setting it to 1.')
                    self.randoms[name] = np.ones(self.randoms.size, dtype='f8')

            # Check if the randoms have a number density column, otherwise estimate it using RedshiftDensityInterpolator
            if 'NZ' not in randoms:
                self.logger.warning('NZ column not found in randoms. Estimating it with RedshiftDensityInterpolator.')
                import healpy as hp
                nside = 512
                distance = np.sqrt(np.sum(randoms['POSITION']**2, axis=-1))
                xyz = randoms['POSITION'] / distance[:, None]
                hpixel = hp.vec2pix(nside, *xyz.T)
                unique_hpixels = np.unique(hpixel)
                fsky = len(unique_hpixels) / hp.nside2npix(nside)
                self.logger.info(f'fsky estimated from randoms: {fsky:.3f}')
                nbar = mockfactory.RedshiftDensityInterpolator(z=distance, fsky=fsky)
                self.randoms['NZ'] = alpha * nbar(distance)

            # Check if the randoms have nmesh and cellsize, otherwise set them using the kmax parameter
            if nmesh is None and cellsize is None:
                # Pick value that will give at least k_mask = kmax_window in the FFTs
                cellsize = np.pi / kmax / (1. + 1e-9)

            self._mesh.append(CatalogMesh(
                    data_positions=randoms['POSITION'],
                    data_weights=randoms['WEIGHT'],
                    position_type='pos',
                    nmesh=nmesh,
                    cellsize=cellsize,
                    boxsize=boxsize,
                    boxpad=boxpad,
                    dtype='c16',
                    **{'interlacing': 3, 'resampler': 'tsc', **kwargs}
                ))
            
            if self._mesh[-1].boxsize[0] > largest_boxsize:
                largest_boxsize = self._mesh[-1].boxsize[0]

            if self._mesh[-1].nmesh[0] > largest_nmesh:
                largest_nmesh = self._mesh[-1].nmesh[0]

        # If there are multiple windows, set the box size and nmesh to the largest values
        if len(self._mesh) > 1:
            for mesh in self._mesh:
                mesh._set_box(nmesh=largest_nmesh, boxsize=largest_boxsize, wrap=False)

        # Log the box size and nmesh
        
        self.boxsize = self._mesh[0].boxsize[0]
        self.nmesh = self._mesh[0].nmesh[0]
        self.logger.info(f'Using box size {self.boxsize}, box center {self._mesh[0].boxcenter} and nmesh {self.nmesh}.')

        self.logger.info(f'Fundamental wavenumber of window meshes = {self.kfun}.')
        self.logger.info(f'Nyquist wavenumber of window meshes = {self.knyquist}.')

        if kmax is not None and self.knyquist < kmax:
            self.logger.warning(f'Nyquist wavelength {self.knyquist} smaller than required window kmax = {kmax}.')

        self.logger.info(f'Average of {self._mesh[0].data_size / self.nmesh**3} objects per voxel.')

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['logger', 'tqdm', '_randoms', '_mesh', '_resume_file']:
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
    def alpha1(self):
        return self._alpha1
    
    @property
    def alpha2(self):
        return self._alpha2
    
    @property
    def alpha(self):
        return self.alpha1

    @property
    def ikgrid(self):
        """Grid of wavenumber indices."""
        ikgrid = []
        for _ in range(3):
            iik = np.arange(self.nmesh)
            iik[iik >= self.nmesh // 2] -= self.nmesh
            ikgrid.append(iik)
        return ikgrid
    
    def I(self, nbar_power, fkp_power):
        return (self._randoms[0]['NZ']**(nbar_power-1) * \
                self._randoms[0]['WEIGHT_FKP']**fkp_power * \
                self._randoms[0]['WEIGHT'] * \
                self.alpha1).sum().tolist()
    
    @staticmethod
    def _shotnoise_mesh(mesh, randoms, alpha):
        """Compute the shotnoise mesh S_AB = nbar * fkp^2."""

        return mesh.clone(
            data_positions=randoms['POSITION'],
            data_weights=randoms['WEIGHT_FKP']**2 * randoms[f'WEIGHT'] * alpha,
            position_type='pos',
        ).to_mesh(compensate=True)

    def mesh(self, ell, m, shotnoise=False, fourier=False, threshold=None):
        """Compute the product of meshes and multiply by real Ylm evaluated at the same coordinates.

        Parameters
        ----------
        ell : int
            Degree of the spherical harmonic.

        m : int
            Order of the spherical harmonic.

        shotnoise : bool, optional
            If True, the shotnoise mesh is used instead of the original mesh. Default is False.

        fourier : bool, optional
            If True, the Fourier transform of the mesh is returned. Default is False.

        Returns
        -------
        mesh
            Resulting mesh after computation.
        """

        assert ell >= 0, "ell must be non-negative"
        assert abs(m) <= ell, "m must be less than or equal to ell"

        Ylm = math.get_real_Ylm(ell, m)

        # Initialize the result mesh
        if shotnoise:
            result = SurveyGeometry._shotnoise_mesh(self._mesh[0], self.randoms[0], self.alpha1)
        else:
            result = self._mesh[0].copy().to_mesh(compensate=True)

        
        if len(self._randoms) > 1 and not shotnoise:
            result *= self._mesh[1].to_mesh(compensate=True)

        # Iterate over slabs to save memory
        for slab in range(result.shape[0]):
            # Multiply by other mesh (or by itself) to obtain W_AB
            if len(self._randoms) == 1 or shotnoise:
                result[slab, ...] *= result[slab, ...]

            # Multiply by real Ylm evaluated at the same coordinates
            result[slab, ...] *= Ylm(result.x[0][slab], result.x[1][0], result.x[2][0])

            if fourier:
                # pmesh fft convention is F(k) = 1/N^3 \sum_{r} e^{-ikr} F(r); let us correct it here
                result[slab, ...] *= self.nmesh**3

        result = result.value if not fourier else result.r2c().value

        if threshold is not None:
            # Convert the result to a sparse array to save memory
            result[np.abs(result) < threshold] = 0
            result = base.SparseNDArray.from_dense(result, shape_in=(self.nmesh,self.nmesh), shape_out=self.nmesh)
        
        return result
    

    # -------------- OLD -----------------

    def compute_window_power(self):

        ikx, iky, ikz = np.meshgrid(*self.ikgrid)
        ikr = np.sqrt(ikx**2 + iky**2 + ikz**2)
        ikr[ikr == 0] = np.inf

        kx, ky, kz = ikx/ikr, iky/ikr, ikz/ikr
        ikr[ikr == np.inf] = 0.
        kr = ikr * self.kfun
    
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
            
        self.logger.info('Window power spectra computed.')

        return self._window_power
    
    def get_window_power_interpolators(self):
        if self._window_power is None:
            self.compute_window_power()

        from scipy.interpolate import InterpolatedUnivariateSpline
        kavg = self._window_power[-1]
        return [InterpolatedUnivariateSpline(kavg, Pwin) for Pwin in self._window_power[:-1]]

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
            
            if self._resume_file is not None and not self._resume_file_readonly and (time.time() - last_save) > 600:
                self.save(self._resume_file)
                last_save = time.time()
                
        self.logger.info('Window kernels computed.')

        if self._resume_file is not None and not self._resume_file_readonly:
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
