import numpy as np
import thecov.base
import os

def test_multipole_covariance_symmetrization():
    cov00, cov22, cov44, cov02, cov04, cov24 = np.random.rand(6, 100, 100)

    cov = thecov.base.MultipoleCovariance()

    cov.set_ell_cov(0, 0, cov00)
    cov.set_ell_cov(2, 2, cov22)
    cov.set_ell_cov(4, 4, cov44)

    cov.set_ell_cov(0, 2, cov02)
    cov.set_ell_cov(0, 4, cov04)
    cov.set_ell_cov(4, 2, cov24.T)

    assert (cov.get_ell_cov(0,2).cov == cov02).all()
    assert (cov.get_ell_cov(2,0).cov == cov02.T).all()

    assert (cov.get_ell_cov(0,4).cov == cov04).all()
    assert (cov.get_ell_cov(4,0).cov == cov04.T).all()

    assert (cov.get_ell_cov(2,4).cov == cov24).all()
    assert (cov.get_ell_cov(4,2).cov == cov24.T).all()

    assert not (cov.get_ell_cov(0,0).cov == cov.get_ell_cov(0,0).cov.T).all()
    assert not (cov.get_ell_cov(2,2).cov == cov.get_ell_cov(2,2).cov.T).all()
    assert not (cov.get_ell_cov(4,4).cov == cov.get_ell_cov(4,4).cov.T).all()

    cov.symmetrize()

    assert (cov.get_ell_cov(0,2).cov == cov02).all()
    assert (cov.get_ell_cov(2,0).cov == cov02.T).all()

    assert (cov.get_ell_cov(0,4).cov == cov04).all()
    assert (cov.get_ell_cov(4,0).cov == cov04.T).all()

    assert (cov.get_ell_cov(2,4).cov == cov24).all()
    assert (cov.get_ell_cov(4,2).cov == cov24.T).all()

    assert (cov.get_ell_cov(0,0).cov == cov.get_ell_cov(0,0).cov.T).all()
    assert (cov.get_ell_cov(2,2).cov == cov.get_ell_cov(2,2).cov.T).all()
    assert (cov.get_ell_cov(4,4).cov == cov.get_ell_cov(4,4).cov.T).all()

    assert not (cov.get_ell_cov(0,2).cov == cov.get_ell_cov(0,2).cov.T).all()
    assert not (cov.get_ell_cov(2,4).cov == cov.get_ell_cov(2,4).cov.T).all()
    assert not (cov.get_ell_cov(4,0).cov == cov.get_ell_cov(4,0).cov.T).all()

    assert (cov.get_ell_cov(0,0).cov == (cov00 + cov00.T)/2).all()
    assert (cov.get_ell_cov(2,2).cov == (cov22 + cov22.T)/2).all()
    assert (cov.get_ell_cov(4,4).cov == (cov44 + cov44.T)/2).all()


def test_multipole_covariance_addition():
    cov1_00, cov1_22, cov1_44, cov1_02, cov1_04, cov1_24 = np.random.rand(6, 100, 100)
    cov2_00, cov2_22, cov2_44, cov2_02, cov2_04, cov2_24 = np.random.rand(6, 100, 100)

    cov1 = thecov.base.MultipoleCovariance()
    cov2 = thecov.base.MultipoleCovariance()

    cov1.set_ell_cov(0,0, cov1_00)
    cov1.set_ell_cov(2,2, cov1_22)
    cov1.set_ell_cov(4,4, cov1_44)

    cov1.set_ell_cov(0,2, cov1_02)
    cov1.set_ell_cov(0,4, cov1_04)
    cov1.set_ell_cov(4,2, cov1_24.T)

    cov2.set_ell_cov(0,0, cov2_00)
    cov2.set_ell_cov(2,2, cov2_22)
    cov2.set_ell_cov(4,4, cov2_44)

    cov2.set_ell_cov(0,2, cov2_02)
    cov2.set_ell_cov(0,4, cov2_04)
    cov2.set_ell_cov(4,2, cov2_24.T)

    addition = cov1 + cov2

    assert (addition.get_ell_cov(0,0).cov == cov1_00 + cov2_00).all()
    assert (addition.get_ell_cov(2,2).cov == cov1_22 + cov2_22).all()
    assert (addition.get_ell_cov(4,4).cov == cov1_44 + cov2_44).all()

    assert (addition.get_ell_cov(0,2).cov == cov1_02 + cov2_02).all()
    assert (addition.get_ell_cov(0,4).cov == cov1_04 + cov2_04).all()
    assert (addition.get_ell_cov(2,4).cov == cov1_24 + cov2_24).all()

def test_multipole_fourier_covariance_save_load_csv():
    cov = thecov.base.MultipoleFourierCovariance()
    cov.set_kbins(0., 0.4, 0.005)

    cov00, cov22, cov44, cov02, cov04, cov24 = np.random.rand(6, cov.kbins, cov.kbins)

    cov.set_ell_cov(0, 0, cov00)
    cov.set_ell_cov(2, 2, cov22)
    cov.set_ell_cov(4, 4, cov44)

    cov.set_ell_cov(0, 2, cov02)
    cov.set_ell_cov(0, 4, cov04)
    cov.set_ell_cov(4, 2, cov24.T)

    cov.savecsv('test1.txt')
    cov.savecsv('test2.txt', ells_both_ways=True)

    cov1 = thecov.base.MultipoleFourierCovariance.fromcsv('test1.txt')
    cov2 = thecov.base.MultipoleFourierCovariance.fromcsv('test2.txt')

    assert np.allclose(cov1.cov, cov.cov)
    assert np.allclose(cov2.cov, cov.cov)

    os.remove('test1.txt')
    os.remove('test2.txt')