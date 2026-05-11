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

__all__ = ['SurveyGeometry']

class Geometry(base.BaseClass):
    pass

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
    