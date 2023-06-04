'''Module containing basic classes to deal with covariance matrices.

Classes
-------
Covariance
    A class to represent a covariance matrix. Implements basic operations such as correlation matrix computation, etc.
MultipoleCovariance
    A class to represent a covariance matrix for an observable that's splitted in multipoles (ell1,ell2).
FourierBinned
    A class that implements k-binning of an observable.
'''

import itertools as itt
import numpy as np
import copy

from . import utils


class Covariance:
    '''A class to represent a covariance matrix.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix.
    '''

    def __init__(self, covariance=None):
        '''Initializes a Covariance object.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        '''

        self._covariance = covariance

    @property
    def cov(self):
        '''The covariance matrix.

        Returns
        -------
        numpy.ndarray
            (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        return self._covariance

    @cov.setter
    def cov(self, covariance):
        '''Sets the covariance matrix.

        Parameters
        ----------
        covariance : numpy.ndarray
            (n,n) numpy array with elements corresponding to the covariance.
        '''

        self._covariance = covariance

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
        self._covariance = (self._covariance + self._covariance.T)/2

    def symmetrized(self):
        '''Returns a symmetrized copy of the covariance matrix.
        
        Returns
        -------
        Covariance
            Covariance object corresponding to the symmetrized covariance matrix.
        '''
        new_cov = copy.deepcopy(self)
        new_cov.symmetrize()
        return new_cov

    def __add__(self, y):
        obj = copy.deepcopy(self)
        obj.cov += (y.cov if isinstance(y, Covariance) else y)

        return obj

    def __sub__(self, y):

        obj = copy.deepcopy(self)
        obj.cov -= (y.cov if isinstance(y, Covariance) else y)

        return obj
    
    def __mul__(self, y):

        obj = copy.deepcopy(self)
        obj.cov *= y

        return obj
    
    def __truediv__(self, y):
        obj = copy.deepcopy(self)
        obj.cov /= y

        return obj

    @property
    def T(self):
        '''This function transposes the covariance matrix.

        Returns
        -------
        Covariance
            Covariance object corresponding to the transpose of the covariance matrix.
        '''

        obj = copy.deepcopy(self)
        obj.cov = obj.cov.T

        return obj

    @property
    def shape(self):
        '''Returns the shape of the covariance.

        Returns
        -------
        tuple
            A tuple with the shape of the covariance matrix.
        '''

        return self.cov.shape

    def save(self, filename):
        '''Saves the covariance as a .npz file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be saved.
        '''

        np.savez(filename if filename.strip(
        )[-4:] in ('.npz', '.npy') else f'{filename}.npz', covariance=self.cov)

    def savetxt(self, filename):
        '''Saves the covariance as a text file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be saved.
        '''

        np.savetxt(filename, self.cov)

    @classmethod
    def load(cls, filename):
        '''Loads the covariance from a .npz file with a specified filename.

        Parameters
        -------
        filename : string
            The name of the file where the covariance matrix will be loaded from.
        '''

        with np.load(filename, mmap_mode='r') as data:
            return cls(data['covariance'])

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

    def __init__(self):
        self._multipole_covariance = {}
        self._ells = []
        self._mshape = (0, 0)

    def __add__(self, y):
        obj = copy.deepcopy(self)

        if isinstance(y, MultipoleCovariance):
            assert self.ells == y.ells, "ells are not the same"

        obj.set_full_cov(self.cov + (y.cov if isinstance(y, Covariance) else y), self.ells)

        return obj

    def __sub__(self, y):
        obj = copy.deepcopy(self)
        
        if isinstance(y, MultipoleCovariance):
            assert self.ells == y.ells, "ells are not the same"

        obj.set_full_cov(self.cov - (y.cov if isinstance(y, Covariance) else y), self.ells)

        return obj
    
    def __mul__(self, y):
        obj = copy.deepcopy(self)

        obj.set_full_cov(self.cov * y, self.ells)

        return obj
    
    def __truediv__(self, y):
        obj = copy.deepcopy(self)

        obj.set_full_cov(self.cov / y, self.ells)

        return obj

    @property
    def cov(self):
        '''This function calculates the full covariance matrix by stacking covariances for different multipoles
        in ascending order.

        Returns
        -------
        numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        ells = self.ells
        cov = np.zeros(np.array(self._mshape)*len(ells))
        for (i, l1), (j, l2) in itt.product(enumerate(ells), enumerate(ells)):
            cov[i*self._mshape[0]:(i+1)*self._mshape[0],
                j*self._mshape[1]:(j+1)*self._mshape[1]] = self.get_ell_cov(l1, l2).cov
        return cov
    
    @cov.setter
    def cov(self, cov):
        '''Sets the full covariance matrix from covariances for different multipoles stacked
        in ascending order.

        Parameters
        ----------
        cov : numpy.ndarray
            An (n,n) numpy array corresponding to the elements of the covariance matrix.
        '''

        self.set_full_cov(cov, self.ells)

    @property
    def ells(self):
        '''The multipoles for which the covariance matrix is defined. Sorted in ascending order.

        Returns
        -------
        tuple
            A tuple of multipole values.
        '''

        return sorted(self._ells)

    def get_ell_cov(self, l1, l2, force_return=False):
        '''Returns the covariance matrix for a given pair of multipoles.

        Parameters
        ----------
        l1
            the first multipole.
        l2
            the second multipole.
        force_return, boolean, optional
            `force_return` if True, returns a zero matrix if the covariance matrix is not defined.

        Returns
        -------
        Covariance
            A Covariance object corresponding to the covariance matrix for the given multipoles.
        '''

        if l1 > l2:
            return self.get_ell_cov(l2, l1).T

        if (l1, l2) in self._multipole_covariance:
            return self._multipole_covariance[l1, l2]
        elif force_return:
            return Covariance(np.zeros(self._mshape))

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

        if self._ells == []:
            self._mshape = cov.shape

        assert cov.shape == self._mshape, "ell covariance has shape inconsistent with other ells"

        if l1 not in self.ells:
            self._ells.append(l1)
        if l2 not in self.ells:
            self._ells.append(l2)

        self._multipole_covariance[min((l1, l2)), max((l1, l2))] = cov if isinstance(
            cov, Covariance) else Covariance(cov)

    def set_full_cov(self, cov_array, ells=(0, 2, 4)):
        '''Sets the full covariance matrix from stacked covariances for different multipoles.

        Parameters
        ----------
        cov_array
            (n,n) numpy array with elements corresponding to the covariance.
        ells
            the multipoles for which the covariance matrix is defined.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        '''

        assert cov_array.ndim == 2, "Covariance should be a matrix (ndim == 1)."
        assert cov_array.shape[0] == cov_array.shape[1], "Covariance matrix should be a square matrix."
        assert cov_array.shape[0] % len(ells) == 0, \
            "Covariance matrix shape should be a multiple of the number of ells."

        c = cov_array
        self._ells = ells
        self._mshape = tuple(np.array(cov_array.shape)//len(ells))

        for (i, l1), (j, l2) in itt.combinations_with_replacement(enumerate(ells), r=2):
            self.set_ell_cov(l1, l2, c[i*c.shape[0]//len(ells):(i+1)*c.shape[0]//len(ells),
                                       j*c.shape[1]//len(ells):(j+1)*c.shape[1]//len(ells)])
        return self
    

    @classmethod
    def from_array(cls, *args, **kwargs):
        '''Creates a MultipoleCovariance object from a numpy array corresponding to the full covariance matrix.

        Parameters
        ----------
        cov_array
            (n,n) numpy array with elements corresponding to the covariance.
        ells
            the multipoles for which the covariance matrix is defined.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        '''

        cov = cls()
        cov.set_full_cov(*args, **kwargs)
        
        return cov

    @classmethod
    def loadtxt(cls, *args, **kwargs):
        '''Loads the covariance from a text file with a specified filename.

        Parameters
        ----------
        filename
            The name of the file where the covariance matrix will be loaded from.

        Returns
        -------
        MultipoleCovariance
            A MultipoleCovariance object.
        '''

        return cls.from_array(np.loadtxt(*args, **kwargs))
        
    def symmetrize(self):
        '''Symmetrizes the covariance matrix in place.'''
        for _, cov in self._multipole_covariance.items():
            cov.symmetrize()


class FourierBinned:
    '''A class to represent a power spectrum binned in wavenumber k. Only linear binning is supported.

    Attributes
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

    Methods
    -------
    set_kbins
        This function defines the k-bins. Only linear binning is supported.
    '''

    def set_kbins(self, kmin, kmax, dk, nmodes=None):
        '''This function defines the k-bins. Only linear binning is supported.

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

        if (nmodes is not None):
            self._nmodes = nmodes

    @property
    def is_kbins_set(self):
        '''Check if k-bins were defined.
        
        Returns
        -------
            bool, True if k-bins were defined, False otherwise.
        '''
        if hasattr(self, 'dk') and hasattr(self, 'kmin') and hasattr(self, 'kmax'):
            return None not in (self.dk, self.kmin, self.kmax)
        else:
            return False

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
    def kedges(self):
        '''
        Returns the edges of the k-bins.
        
        Returns
        -------
        numpy.ndarray
            The edges of the k-bins.
        '''

        return np.arange(self.kmin, self.kmax + self.dk, self.dk)

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
    def volume(self):
        '''Returns the volume of the object. If not available, return that of the associated geometry.
        
        Returns
        -------
        float
            The volume of the object.
        '''

        if hasattr(self, '_volume'):
            return self._volume

        if hasattr(self, 'geometry'):
            return self.geometry.volume

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

        return utils.nmodes(self.volume, self.kedges[:-1], self.kedges[1:])

    @nmodes.setter
    def nmodes(self, nmodes):
        '''Manually sets the number of modes per k-bin shell.
        
        Parameters
        -------
        nmodes : numpy.ndarray
            The number of modes per k-bin shell.
        '''

        self._nmodes = nmodes
        return nmodes
