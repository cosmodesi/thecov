'''Module containing basic classes to deal with covariance matrices.'''

import os, time, copy

import numpy as np
import scipy

from . import utils, math
import logging

__all__ = ['Covariance',
           'MultipoleCovariance',
           'LinearBinning',
           'FourierCovariance',
           'MultipoleFourierCovariance']


class BaseClass:
    """
    Base class that implements copy, save/load, etc.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, **kwargs):
        new = self.__copy__()
        new.__dict__.update(kwargs)
        return new

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        return getattr(self, 'mpicomm', None) is not None and self.mpicomm.size > 1

    def save(self, filename):
        """Save to ``filename``."""
        start = time.time()
        if not self.with_mpi or self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            np.save(filename, self.__getstate__(), allow_pickle=True)
        # if self.with_mpi:
        #     self.mpicomm.Barrier()

        if hasattr(self, 'logger'):
            self.logger.info(f'Saved to {filename} in {time.time() - start:.3f}s.')

    @classmethod
    def load(cls, filename):
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new

class Covariance(BaseClass):
    '''A class that represents a covariance matrix.
    Implements basic operations such as correlation matrix computation, etc.
    '''

    def __init__(self, covariance=None):
        '''Initializes a Covariance object.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        '''

        self._cov = covariance

    @property
    def cov(self):
        '''The covariance matrix.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        return self._cov

    @cov.setter
    def cov(self, covariance):
        '''Sets the covariance matrix.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        '''

        self._cov = covariance

    @property
    def cor(self):
        '''Returns the correlation matrix.

        The correlation matrix is obtained by dividing each element of the covariance matrix by
        the product of the standard deviations of the corresponding variables.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the correlation matrix.
        '''

        cov = self.cov
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        outer_v[outer_v == 0] = np.inf
        cor = cov / outer_v
        cor[cov == 0] = 0
        return cor

    def symmetrize(self):
        """Symmetrizes the covariance matrix in place."""
        self.cov = (self.cov + self.cov.T)/2

    def symmetrized(self):
        '''Returns a symmetrized copy of the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the symmetrized covariance matrix.
        '''
        new_cov = self.copy()
        new_cov.symmetrize()
        return new_cov
    
    def regularize(self, mode='zero'):
        eigvals, eigvecs = self.eig
        if mode == 'zero':
            eigvals[eigvals < 0] = 0
        elif mode == 'flip':
            eigvals = np.abs(eigvals)
        elif mode == 'minpos':
            eigvals[eigvals < 0] = min(eigvals[eigvals > 0])
        self.cov = np.einsum('ij,jk,kl->il', eigvecs, np.diag(eigvals), eigvecs.T)
    
    def regularized(self):
        new_cov = self.copy()
        new_cov.regularize()
        return new_cov

    def __add__(self, y):
        return Covariance(self.cov + (y.cov if isinstance(y, Covariance) else y))

    def __sub__(self, y):
        return self.__add__(-y)

    def __mul__(self, y):
        return Covariance(self.cov * y)

    def __truediv__(self, y):
        return Covariance(self.cov / y)

    @property
    def T(self):
        '''Returns the transpose of the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the transpose of the covariance matrix.
        '''

        new_cov = self.copy()
        new_cov.cov = new_cov.cov.T
        return new_cov

    @property
    def shape(self):
        '''Returns the shape of the covariance.

        Returns
        -------
        tuple
            A tuple with the shape of the covariance matrix.
        '''

        return self.cov.shape

    @property
    def eig(self):
        '''Compute the eigenvalues and right eigenvectors of the covariance.

        Returns
        -------
        A namedtuple with the following attributes:
            eigenvalues
            (..., M) array
                The eigenvalues, each repeated according to its multiplicity.
                The eigenvalues are not necessarily ordered. The resulting
                array will be of complex type, unless the imaginary part is
                zero in which case it will be cast to a real type. When a is
                real the resulting eigenvalues will be real (0 imaginary
                part) or occur in conjugate pairs

            eigenvectors
            (...), M, M) array
                The normalized (unit “length”) eigenvectors, such that the
                column eigenvectors[:,i] is the eigenvector corresponding to
                the eigenvalue eigenvalues[i].
        '''

        return np.linalg.eig(self.cov)

    @property
    def eigvals(self):
        '''Compute the eigenvalues of the covariance.

        Returns
        -------
        (..., M,) ndarray
            The eigenvalues, each repeated according to its multiplicity.
            They are not necessarily ordered, nor are they necessarily
            real for real matrices.
        '''

        return np.linalg.eigvals(self.cov)

    def savetxt(self, filename):
        '''Saves the covariance as a text file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be saved.
        '''
        utils.mkdir(os.path.dirname(filename))
        np.savetxt(filename, self.cov)

    @classmethod
    def loadtxt(cls, *args, **kwargs):
        '''Loads the covariance from a text file with a specified filename.

        Parameters
        -------
        *args
            Arguments to be passed to numpy.loadtxt.
        **kwargs
            Keyword arguments to be passed to numpy.loadtxt.

        Returns
        -------
        Covariance
            Covariance object.
        '''

        return cls.from_array(np.loadtxt(*args, **kwargs))

    @classmethod
    def from_array(cls, a):
        '''Creates a Covariance object from a numpy array.

        Parameters
        -------
        numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.

        Returns
        -------
        Covariance
            Covariance object.
        '''

        return cls(covariance=a)


class MultipoleCovariance(Covariance):
    '''A class to represent a covariance matrix for a set of multipoles.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix.
    cor : numpy.ndarray
        The correlation matrix.
    '''

    def __init__(self, symmetric=False):
        self._multipole_covariance = {}
        self._symmetric = symmetric

    def set_ell_cov(self, l1, l2, cov):
        '''Sets the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1 : int
            The first multipole.
        l2 : int
            The second multipole.
        cov : Covariance or numpy.ndarray
            The covariance matrix. Can be an instance of Covariance or a numpy array.
        '''

        if self._symmetric and l1 > l2:
            return self.set_ell_cov(l2, l1, cov.T if cov is not None else None)

        self._multipole_covariance[l1, l2] = cov
        

    def get_ell_cov(self, l1, l2, cls=Covariance):
        '''Returns the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1
            the first multipole.
        l2
            the second multipole.

        Returns
        -------
        Covariance
            A Covariance object corresponding to the covariance matrix for the given multipoles.
        '''

        if self._symmetric and l1 > l2:
            return self.get_ell_cov(l2, l1, cls=cls).T

        if (l1, l2) in self._multipole_covariance:
            return self._multipole_covariance[l1, l2]

    def is_ell_set(self, l1, l2):
        return (l1,l2) in self._multipole_covariance.keys()

    @property
    def ells(self):
        '''Returns sorted lists of unique first and second multipoles used in the covariance matrices.

        Returns
        -------
        tuple of two lists
        '''
        ells1, ells2 = set(), set()
        
        for (l1, l2) in self._multipole_covariance.keys():
            ells1.add(l1)
            ells2.add(l2)
            
        return sorted(ells1), sorted(ells2)

    def has_ells(self, l1, l2):
        if self._symmetric and l1 > l2:
            return self.has_ells(l2, l1)
        return (l1, l2) in self._multipole_covariance.keys()

    @property
    def symmetric(self):
        return self._symmetric

    @ells.setter
    def ells(self, ells):
        '''Initializes all entries in the self._multipole_covariance dict based on the input ells tuple.

        Parameters
        ----------
        ells : tuple
            A tuple of two lists: (l1s, l2s), where l1s and l2s are lists of multipoles.
        '''
        
        # Initialize the covariance matrices for each pair
        for l1 in ells[0]:
            for l2 in ells[1]:
                if not self.has_ells(l1,l2):
                    self.set_ell_cov(l1, l2, None)

    @property
    def cov(self):
        '''This function calculates the full covariance matrix by stacking covariances for different multipoles
        in ascending order.

        Returns
        -------
        numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        ells1, ells2 = self.ells

        return np.vstack([np.hstack([self.get_ell_cov(l1, l2).cov for l2 in ells2]) for l1 in ells1])

    @cov.setter
    def cov(self, cov):
        '''Sets the full covariance matrix from covariances for different multipoles stacked
        in ascending order.

        Parameters
        ----------
        cov : numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        ells1, ells2 = self.ells

        assert cov.ndim == 2, "Covariance should be a matrix (ndim == 1)."
        assert cov.shape[0] % len(ells1) == 0, \
            "Can't resolve covariance structure as shape is not a multiple of the number of ells."
        assert cov.shape[1] % len(ells2) == 0, \
            "Can't resolve covariance structure as shape is not a multiple of the number of ells."

        size1 = cov.shape[0]//len(ells1)
        size2 = cov.shape[1]//len(ells2)

        for i1,l1 in enumerate(ells1):
            for i2,l2 in enumerate(ells2):
                self.set_ell_cov(l1,l2,Covariance(cov[i1*size1:(i1+1)*size1,i2*size2:(i2+1)*size2]))

    def __add__(self, y):
        assert isinstance(y, MultipoleCovariance)

        cov = MultipoleCovariance(symmetric=self.symmetric and y.symmetric)
        ells1, ells2 = self.ells
        for l1 in ells1:
            for l2 in ells2:
                cov.set_ell_cov(l1,l2, self.get_ell_cov(l1,l2) + y.get_ell_cov(l1,l2))
        return cov

    def __sub__(self, y):
        return self.__add__(-y)

    def __mul__(self, y):
        cov = self.deepcopy()
        cov.foreach(lambda x: x*y)
        return cov

    def __truediv__(self, y):
        return self * (1/y)
    
    def foreach(self, func):
        '''Applies a function to each covariance matrix.

        Parameters
        ----------
        func : function
            The function to be applied to each covariance matrix.
        '''

        for (l1, l2), cov in self._multipole_covariance.items():
            self.set_ell_cov(l1, l2, func(cov))
        
        return self


    @classmethod
    def from_array(cls, cov):
        '''Creates a MultipoleCovariance object from a numpy array corresponding to the full covariance matrix.

        Parameters
        ----------
        cov
            (n,n) numpy array with elements corresponding to the covariance.
        ells
            the multipoles for which the covariance matrix is defined.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        '''

        cov = cls()
        cov.cov = cov

        return cov


class LinearBinning:
    '''A class to represent an observable linearly binned in wavenumber k.

    Attributes
    ----------
    kmin: float
        The minimum value of the wavenumber k.
    kmax: float
        The maximum value of the wavenumber k.
    dk: float
        The spacing between k-bins.
    '''

    def __init__(self, kmin=None, kmax=None, dk=None) -> None:
        self.kmin, self.kmax, self.dk = kmin, kmax, dk

    def set_kbins(self, kmin, kmax, dk):
        '''This function defines the k-bins.

        Parameters
        ----------
        kmin: float
            The minimum value of the wavenumber k.
        kmax: float
            The maximum value of the wavenumber k.
        dk: float
            The spacing between k-bins.
        nmodes: numpy.ndarray, optional
            The number of modes to be used in the calculation. It is an optional parameter.
            If omitted, it is calculated from the volume of spherical shells.
        '''

        self.dk = dk
        self.kmax = kmax
        self.kmin = kmin

    @property
    def is_kbins_set(self):
        '''Check if k-bins were defined.

        Returns
        -------
            bool, True if k-bins were defined, False otherwise.
        '''
        return None not in (self.dk, self.kmin, self.kmax)

    @property
    def kbins(self):
        '''Returns the total number of k-bins.

        Returns
        -------
        int
            The total number of k-bins.
        '''

        return len(self.kmid)

    @property
    def kmid(self):
        '''
        Returns the midpoints of the k-bins.

        Returns
        -------
        numpy.ndarray
            The midpoints of the k-bins.
        '''

        return np.arange(self.kmin + self.dk/2, self.kmax + self.dk/2, self.dk)

    @property
    def kavg(self):
        '''
        Returns the average k of the k-bins. Assumes spherical approximation to
        integrate k-modes, which fails for small k.

        Returns
        -------
        numpy.ndarray
            The average k of the k-bins.
        '''
        return 3/4*(self.kedges[1:]**4 - self.kedges[:-1]**4)/ \
                   (self.kedges[1:]**3 - self.kedges[:-1]**3)

    @property
    def kedges(self):
        '''
        Returns the edges of the k-bins.

        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''

        return np.arange(self.kmin, self.kmax + self.dk/2, self.dk)

    @property
    def kfun(self):
        '''Fundamental wavenumber of the box 2*pi/Lbox.

        Returns
        -------
        float
            The fundamental wavenumber of the box.
        '''

        return 2*np.pi/self.volume**(1/3)

    @property
    def nmodes(self):
        '''This function calculates the number of modes per k-bin shell. If nmodes was not provided, it is
        extimated from the volume of each shell.

        Returns
        -------
        numpy.ndarray
            The number of modes per k-bin shell.
        '''

        if hasattr(self, '_nmodes'):
            return self._nmodes

        return math.nmodes(self.volume, self.kedges[:-1], self.kedges[1:])

    @nmodes.setter
    def nmodes(self, nmodes):
        '''Manually sets the number of modes per k-bin shell.

        Parameters
        -------
        nmodes : numpy.ndarray
            The number of modes per k-bin shell.
        '''

        self._nmodes = nmodes

class FourierCovariance(Covariance):

    def __init__(self, kbin1=None, kbin2=None):
        if kbin2 is None:
            kbin2 = kbin1

        self.kbin1 = kbin1
        self.kbin2 = kbin2

    def kcut(self, kmin=None, kmax=None):
        if kmin is None:
            kmin = max(self.kbin1.kmin, self.kbin2.kmin)

        if kmax is None:
            kmax = min(self.kbin1.kmax, self.kbin2.kmax)

        imin1 = (self.kbin1.kmid >= kmin).argmax()
        imin2 = (self.kbin2.kmid >= kmin).argmax()

        imax1 = len(self.kbin1.kmid) if (self.kbin1.kmid <= kmax).all() else (self.kbin1.kmid <= kmax).argmin()
        imax2 = len(self.kbin2.kmid) if (self.kbin2.kmid <= kmax).all() else (self.kbin2.kmid <= kmax).argmin()

        self.cov = self.cov[imin1:imax1, imin2:imax2]

        self.kbin1.kmin, self.kbin1.kmax = kmin, kmax
        self.kbin2.kmin, self.kbin2.kmax = kmin, kmax

        return self

    @property
    def kmid_matrices(self):
        k1 = np.einsum('i,j->ij', self.kbin1.kmid, np.ones(self.kbin2.kbins))
        k2 = np.einsum('i,j->ji', self.kbin2.kmid, np.ones(self.kbin1.kbins))

        return k1, k2

    @property
    def kmin_matrices(self):
        k1 = np.einsum('i,j->ij', self.kbin1.kedges[:-1], np.ones(self.kbin2.kbins))
        k2 = np.einsum('i,j->ji', self.kbin2.kedges[:-1], np.ones(self.kbin1.kbins))

        return k1, k2

class MultipoleFourierCovariance(MultipoleCovariance, FourierCovariance):

    def __init__(self):
        MultipoleCovariance.__init__(self)
        FourierCovariance.__init__(self)
        self.logger = logging.getLogger('MultipoleFourierCovariance')

    @property
    def kmid_ell_matrices(self):
        ells1, ells2 = self.ells

        kfull1 = np.concatenate([self.kbin1.kmid for _ in ells1])
        kfull2 = np.concatenate([self.kbin2.kmid for _ in ells2])

        k1 = np.einsum('i,j->ij', kfull1, np.ones_like(kfull2))
        k2 = np.einsum('i,j->ji', kfull2, np.ones_like(kfull1))

        return k1, k2

    @property
    def ell_matrices(self):
        ells1, ells2 = self.ells

        kells1 = np.einsum('i,j->ij', ells1, np.ones(self.kbin1.kbins)).flatten()
        kells2 = np.einsum('i,j->ji', ells2, np.ones(self.kbin2.kbins)).flatten()

        ell1 = np.einsum('i,j->ij', kells1, np.ones_like(kells2))
        ell2 = np.einsum('i,j->ji', kells2, np.ones_like(kells1))

        return ell1, ell2

    def savecsv(self, filename, fmt=['%.d', '%.d', '%.4f', '%.4f', '%.8e']):
        k1, k2 = self.kmid_ell_matrices
        ell1, ell2 = self.ell_matrices

        cov = self.cov

        mask = ell1 <= ell2 if self.symmetric else np.ones_like(ell1, dtype=bool)
        utils.mkdir(os.path.dirname(filename))
        np.savetxt(filename, np.concatenate([ell1[mask].reshape(-1, 1),
                                             ell2[mask].reshape(-1, 1),
                                               k1[mask].reshape(-1, 1),
                                               k2[mask].reshape(-1, 1),
                                              cov[mask].reshape(-1, 1)], axis=1), fmt=fmt, header='ell1 ell2 kmid1 kmid2 cov')
    def loadcsv(self, filename):
        raise NotImplementedError

        # ell1, ell2, k1, k2, value = np.loadtxt(filename).T

        # k1 = np.unique(k1)
        # kbins = len(k)

        # assert np.allclose(k, np.unique(k2)), "k1 and k2 are not consistent"

        # dk = np.mean(np.diff(k))
        # kmin = k.min() - dk/2
        # kmax = k.max() + dk/2

        # ells = np.unique(ell1)
        # assert np.allclose(ells, np.unique(ell2)), "ell1 and ell2 are not consistent"

        # ells_both_ways = len(value) == (len(ells)*kbins)**2
        # ells_one_way   = len(value) == (len(ells)**2 + len(ells))/2 * kbins**2

        # assert ells_one_way or ells_both_ways, 'length of covariance file doesn\'nt match'

        # self.set_kbins(kmin, kmax, dk)

        # assert np.allclose(np.unique(k1), self.kmid), "k bins are not linearly spaced"

        # kmid_matrix = np.einsum('i,j->ij', k, np.ones_like(k))

        # for l1, l2 in itt.combinations_with_replacement(ells, r=2):
        #     block_mask = (ell1 == l1) & (ell2 == l2)
        #     assert np.allclose(k1[block_mask].reshape(kmid_matrix.shape),   kmid_matrix)
        #     assert np.allclose(k2[block_mask].reshape(kmid_matrix.T.shape), kmid_matrix.T)
        #     c = value[block_mask].reshape(kbins, kbins)
        #     self.set_ell_cov(l1, l2, c)

        # return self

    @classmethod
    def fromcsv(cls, filename):
        cov = cls()
        cov.loadcsv(filename)
        return cov
    
    def set_ell_cov(self, l1, l2, cov, cls=FourierCovariance):
        cov = super().set_ell_cov(l1, l2, cov, cls=cls)
        if not cov.is_kbins_set:
            cov.set_kbins(self.kmin, self.kmax, self.dk)
        return cov
    
    def get_ell_cov(self, l1, l2, cls=FourierCovariance):
        return super().get_ell_cov(l1, l2, cls)

    def kcut(self, kmin=None, kmax=None):
        self.foreach(lambda cov: cov.kcut(kmin, kmax))
        self.set_kbins(kmin, kmax, self.dk)
        
        self.logger.info(f'kcut to {self.kmin} < k < {self.kmax}')

        return self
    
    def set_kbins(self, kmin, kmax, dk, nmodes=None):
        size = (kmax - kmin)/dk
        size = (np.round(size) if np.allclose(np.round(size), size) else size).astype(int)
        self._mshape = (size, size)
        self.foreach(lambda cov: cov.set_kbins(kmin, kmax, dk, nmodes))
        return super().set_kbins(kmin, kmax, dk, nmodes)

class SparseNDArray:
    """
    A class to represent a sparse ND array using scipy.sparse.csr_matrix.
    Indices are split between shape_out and shape_in, as if the array is
    a 2D matrix (shape_out x shape_in). Matrix multiplication is done using
    the @ operator and requires the shapes to be compatible, i.e., shape_in
    of the leftmost array must match shape_out of the rightmost array.
    """
    def __init__(self, shape_out, shape_in):
        self.shape_in = np.asarray(shape_in).astype(int)
        self.shape_out = np.asarray(shape_out).astype(int)
        self._matrix = scipy.sparse.csr_matrix((np.prod(shape_out), np.prod(shape_in)))

    def _nd_to_2d_indices(self, *indices):
        indices = np.asarray(indices).astype(int)
        if len(indices) == len(self.shape_out) + len(self.shape_in):
            i = np.ravel_multi_index(indices[:len(self.shape_out)], self.shape_out)
            j = np.ravel_multi_index(indices[len(self.shape_out):], self.shape_in)
            return i,j
        elif len(indices) == len(self.shape_out):
            i = np.ravel_multi_index(indices, self.shape_out)
            # j = np.arange(np.prod(self.shape_in))
            return i
        
    
    def __setitem__(self, indices, value):
        indices = np.asarray(indices).astype(int)
        if len(indices) == len(self.shape_out) + len(self.shape_in):
            try:
                self._matrix[self._nd_to_2d_indices(*indices)] = value
            except IndexError:
                raise IndexError(f"Indices {indices} are out of bounds for array with shape shape_out={self.shape_out}, shape_in={self.shape_in}.")
        elif len(indices) == len(self.shape_out):
            if isinstance(value, SparseNDArray):
                self._matrix[self._nd_to_2d_indices(*indices)] = value._matrix
            else:
                self._matrix[self._nd_to_2d_indices(*indices)] = value.flatten()
        else:
            raise ValueError(f"Invalid number of indices: {len(indices)}. Expected {len(self.shape_out) + len(self.shape_in)} or {len(self.shape_out)}.")

    def __getitem__(self, indices):
        indices = np.asarray(indices).astype(int)
        return self._matrix[self._nd_to_2d_indices(*indices)]

    def __repr__(self):
        return f"SparseNDArray(shape_out={self.shape_out} -> {np.prod(self.shape_out)}, shape_in={self.shape_in} -> {np.prod(self.shape_in)}, nnz={self._matrix.nnz})"
    
    def to_dense(self):
        """
        Convert the sparse matrix back to a dense ND array.
        """
        return self._matrix.toarray().reshape(self.shape_out.tolist() + self.shape_in.tolist())

    @staticmethod
    def from_dense(dense_array, shape_out=None, shape_in=None):
        """
        Create a SparseNDArray from a dense array.
        """
        if shape_out is None:
            shape_out = dense_array.shape[:-len(dense_array.shape)//2]
        if shape_in is None:
            shape_in = dense_array.shape[len(dense_array.shape)//2:]
        sparse_array = SparseNDArray(shape_in, shape_out)
        sparse_array._matrix = scipy.sparse.csr_matrix(dense_array.reshape(np.prod(shape_out), np.prod(shape_in)))
        return sparse_array
    
    def __add__(self, other):
        if isinstance(other, SparseNDArray):
            assert (self.shape_in == other.shape_in) and (self.shape_out == other.shape_out), \
                "Shapes do not match for multiplication."
            
            import copy
            other = copy.deepcopy(other)
            other._matrix += self._matrix
            return other
        else:
            raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")
        
    def __mul__(self, other):
        if isinstance(other, SparseNDArray):
            assert (self.shape_in == other.shape_in) and (self.shape_out == other.shape_out), \
                "Shapes do not match for multiplication."
            
            import copy
            other = copy.deepcopy(other)
            other._matrix *= self._matrix
            return other
        else:
            raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")
        
    def __matmul__(self, other):
        if isinstance(other, SparseNDArray):
            assert (np.all(self.shape_in == other.shape_out)), \
                "Shapes do not match for matrix multiplication."
            other = copy.deepcopy(other)
            other._matrix = self._matrix.dot(other._matrix)
            other.shape_out = self.shape_out
            return other
        elif isinstance(other, np.ndarray):
            result = copy.deepcopy(self)
            result._matrix = scipy.sparse.csr_matrix(self._matrix.dot(other.reshape(np.prod(self.shape_in), -1)))
            result.shape_in = other.shape[len(self.shape_in):]
            return result
        
        else:
            raise ValueError(f"Operation not supported between {self.__class__} and {other.__class__}.")
        
    def __sizeof__(self):
        return self._matrix.data.nbytes + self._matrix.indptr.nbytes + self._matrix.indices.nbytes
    
    def save(self, filename):
        """
        Save the sparse matrix to a file.
        """
        np.savez(filename,
                 data=self._matrix.data,
                 indices=self._matrix.indices,
                 indptr=self._matrix.indptr,
                 shape=self._matrix.shape,
                 shape_out=self.shape_out,
                 shape_in=self.shape_in)

    @classmethod
    def load(cls, filename):
        """
        Load the sparse matrix from a file.
        """
        loader = np.load(filename)
        obj = cls(loader['shape_out'], loader['shape_in'])
        obj._matrix = scipy.sparse.csr_matrix((loader['data'],
                                               loader['indices'],
                                               loader['indptr']),
                                               shape=loader['shape'])
        return obj

    def reshape(self, shape_out=None, shape_in=None):
        """
        Reshape the sparse matrix.
        """
        result = copy.deepcopy(self)
        result._matrix = self._matrix.reshape(np.prod(shape_out), np.prod(shape_in))
        if shape_out is not None:
            result.shape_out = shape_out
        if shape_in is not None:
            result.shape_in = shape_in
        return result

    def transpose(self):
        """
        Transpose the sparse matrix.
        """
        result = copy.deepcopy(self)
        result._matrix = self._matrix.transpose()
        result.shape_out, result.shape_in = self.shape_in, self.shape_out
        return result
    
    @property
    def T(self):
        """
        Transpose the sparse matrix.
        """
        return self.transpose()