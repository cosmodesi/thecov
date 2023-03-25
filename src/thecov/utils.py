
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

# uniformly samples from a shell
def sample_from_shell(rmin, rmax, discrete=True):
    
    r = rmin + (rmax - rmin) * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(1 - 2 * np.random.rand())
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    if(discrete):
        x,y,z = int(np.round(x)), int(np.round(y)), int(np.round(z))
        r = np.sqrt(x**2 + y**2 + z**2)
        if(r < rmin or r > rmax):
            return sample_from_shell(rmin, rmax, discrete)
    
    return x,y,z,r

def plot_cov(cova, covb=None, k=None, kmax=None, num_multipoles=3, label_a=None, label_b=None, vmin=-1, vmax=1, num_ticks=5, **kwargs):

    def cov2corr(covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation
    
    fig, axes = plot.subplots(1, 1, figsize=(12,10), sharey=True, facecolor='white')
    
    if k is None:
        k = np.arange(cova.shape[0])
    else:
        axes.set_xlabel(r"$k$  [h/Mpc]")
        axes.set_ylabel(r"$k$  [h/Mpc]")
        
    if len(k) == cova.shape[0]//num_multipoles:
        # if k goes from kmin to kmax only once, repeat it num_multipoles for mono/quadru/hexadeca/...pole
        k = np.concatenate(num_multipoles*[k])
    
    if covb is not None:
        cov = triangle_covs(cov2corr(cova), cov2corr(covb))
    else:
        cov = cov2corr(cova)
    
    if kmax is not None:
        # cut the covariance to kmax
        cov = cov[np.einsum('i,j->ij', k <= kmax, k <= kmax)].reshape(2*[sum(k <= kmax)])
        kbins = sum(k <= kmax)//num_multipoles
        min_k, max_k = min(k[k <= kmax]), max(k[k <= kmax])
    else:
        kbins = len(k)//num_multipoles
        min_k, max_k = min(k), max(k)
        
    i2k_fac = ((max_k - min_k)/kbins)
    
    axes.xaxis.set_major_formatter(plot.FuncFormatter(lambda v,i: f'{(v%kbins)*i2k_fac + min_k:.2f}'))
    axes.yaxis.set_major_formatter(plot.FuncFormatter(lambda v,i: f'{(v%kbins)*i2k_fac + min_k:.2f}'))

    axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(num_multipoles*num_ticks))
    axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(num_multipoles*num_ticks))
    
    for i in range(num_multipoles-1):
        axes.axvline((i+1)*kbins,  color='#888', ls='dashed', lw=1)
        axes.axhline((i+1)*kbins,  color='#888', ls='dashed', lw=1)

    for i,j in itt.combinations_with_replacement(range(num_multipoles), 2):
        axes.text((i+1/2)*kbins, (j+1/2)*kbins, f'{2*i}{2*j}', fontsize='18')
        axes.text((j+1/2)*kbins, (i+1/2)*kbins, f'{2*i}{2*j}', fontsize='18')

    if label_a is not None:
        axes.text(kbins*0.04, (num_multipoles-0.12)*kbins,  label_a,  fontsize=20)
    if label_b is not None:
        axes.text((num_multipoles - 0.05*(len(label_b) + 1.5))*kbins, 0.04*kbins, label_b, fontsize=20)
        
    # axes.text(10, 3*kbins+20, r'$P_\ell(k)$ correlation matrix.')

    colorbar = fig.colorbar(axes.imshow(cov, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, **kwargs), pad=0.01)
    fig.tight_layout()
    return fig, axes, colorbar