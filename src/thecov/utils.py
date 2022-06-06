
import numpy as np

def r2c_to_c2c_3d(fourier):

    fourier_c = np.conj(fourier[:, :, (-2 if fourier.shape[0]%2 == 0 else -1):0:-1])

    fourier_c[1:, :, :] = fourier_c[:0:-1, :, :]
    fourier_c[:, 1:, :] = fourier_c[:, :0:-1, :]

    return np.concatenate((fourier, fourier_c), axis=2)

def triangle_cov(upper, lower, diagonal='upper'):
    assert diagonal in ['upper', 'lower']
    cov = np.triu(upper) + np.tril(lower)
    cov -= np.diag(np.diag(upper if diagonal == 'lower' else lower))
    return cov
    
# python's enumerate but with a custom step = 2
def enum2(xs, start=0, step=2):
    for x in xs:
        yield (start, x)
        start += step