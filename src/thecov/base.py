
import itertools as itt
import numpy as np

from . import utils

class Covariance:

    def __init__(self, covariance=None):
        self._covariance = covariance

    @property
    def cov(self):
        return self._covariance

    @property
    def cor(self):
        cov = self.cov
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        outer_v[outer_v == 0] = np.inf
        cor = cov / outer_v
        cor[cov == 0] = 0
        return cor

    def __add__(self, y):
        return Covariance(self.cov + (y.cov if isinstance(y, Covariance) else y))

    def __sub__(self, y):
        return Covariance(self.cov - (y.cov if isinstance(y, Covariance) else y))

    @property
    def T(self):
        return Covariance(self.cov.T)

    @property
    def shape(self):
        return self.cov.shape

    def save(self, filename):
        np.savez(filename if filename.strip()[-4:] in ('.npz', '.npy') else f'{filename}.npz', covariance=self.cov)

    def savetxt(self, filename):
        np.savetxt(filename, self.cov)

    @classmethod
    def load(cls, filename):
        with np.load(filename, mmap_mode='r') as data:
            return cls(data['covariance'])

    @classmethod
    def loadtxt(cls, *args, **kwargs):
        return cls.from_array(np.loadtxt(*args, **kwargs))

    @classmethod
    def from_array(cls, a):
        return cls(covariance=a)

class MultipoleCovariance(Covariance):

    def __init__(self):
        super().__init__()

        self._multipole_covariance = {}
        self._ells = []
        self._mshape = (0,0)

    @property
    def cov(self):
        ells = self.ells
        cov = np.zeros(np.array(self._mshape)*len(ells))
        for (i, l1),(j, l2) in itt.product(enumerate(ells), enumerate(ells)):
            cov[i*self._mshape[0]:(i+1)*self._mshape[0],
                j*self._mshape[1]:(j+1)*self._mshape[1]] = self.get_ell_cov(l1, l2).cov
        return cov

    @property
    def ells(self):
        return sorted(self._ells)

    def get_ell_cov(self, l1, l2, force_return=False):
        if l1 > l2:
            return self.get_ell_cov(l2, l1).T
        
        if (l1, l2) in self._multipole_covariance:
            return self._multipole_covariance[l1, l2]
        elif force_return:
            return np.zeros(self._mshape)

    def set_ell_cov(self, l1, l2, cov):
        if self._ells == []:
            self._mshape = cov.shape
        
        assert cov.shape == self._mshape, "ell covariance has shape inconsistent with other ells"

        if l1 not in self.ells:
            self._ells.append(l1)
        if l2 not in self.ells:
            self._ells.append(l2)

        self._multipole_covariance[min((l1,l2)), max((l1,l2))] = cov if isinstance(cov, Covariance) else Covariance(cov)

    @classmethod
    def from_array(cls, cov_array, ells=(0,2,4)):
        assert cov_array.ndim == 2, "Covariance should be a matrix (ndim == 1)."
        assert cov_array.shape[0] == cov_array.shape[1], "Covariance matrix should be a square matrix."
        assert cov_array.shape[0] % len(ells) == 0, "Covariance matrix shape should be a multiple of the number of ells."
        c = cov_array
        cov = cls()
        cov._ells = ells
        cov._mshape = tuple(np.array(cov_array.shape)//len(ells))
        for (i, l1),(j, l2) in itt.combinations_with_replacement(enumerate(ells), r=2):
            cov.set_ell_cov(l1, l2, c[i*c.shape[0]//len(ells):(i+1)*c.shape[0]//len(ells),
                                      j*c.shape[1]//len(ells):(j+1)*c.shape[1]//len(ells)])
        return cov

    @classmethod
    def loadtxt(cls, *args, **kwargs):
        return cls.from_array(np.loadtxt(*args, **kwargs))


class FourierBinned:

    def __init__(self, kmin=None, kmax=None, dk=None):
        self.set_kbins(kmin, kmax, dk)

    def set_kbins(self, kmin, kmax, dk, nmodes=None):
        self.dk = dk
        self.kmax = kmax
        self.kmin = kmin

        if(nmodes is not None):
            self._nmodes = nmodes

    @property
    def is_kbins_set(self):
        return None not in (self.dk, self.kmin, self.kmax)

    @property
    def kbins(self):
        return len(self.kmid)

    @property
    def kmid(self):
        return np.arange(self.kmin + self.dk/2, self.kmax + self.dk/2, self.dk)

    @property
    def kedges(self):
        return np.arange(self.kmin, self.kmax + self.dk, self.dk)

    @property
    def kfun(self):
        return 2*np.pi/self.volume**(1/3)

    @property
    def volume(self):
        if hasattr(self, '_volume'):
            return self._volume
        
        if hasattr(self, 'geometry'):
            return self.geometry.volume
    
    @property
    def nmodes(self):
        if hasattr(self, '_nmodes'):
            return self._nmodes
        
        return utils.nmodes(self.volume, self.kedges[:-1], self.kedges[1:])

    @nmodes.setter
    def nmodes(self, nmodes):
        self._nmodes = nmodes
        return nmodes
