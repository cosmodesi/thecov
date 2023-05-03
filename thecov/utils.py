
import numpy as np

def r2c_to_c2c_3d(fourier):

    fourier_c = np.conj(fourier[:, :, (-2 if fourier.shape[0]%2 == 0 else -1):0:-1])

    fourier_c[1:, :, :] = fourier_c[:0:-1, :, :]
    fourier_c[:, 1:, :] = fourier_c[:, :0:-1, :]

    return np.concatenate((fourier, fourier_c), axis=2)

def triangle_cov(upper, lower, diagonal='upper'):
    assert diagonal in ['upper', 'lower'], "Argument diagonal should be either 'upper' or 'lower'."
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

def nmodes(volume, kmin, kmax):
    return volume/3/(2*np.pi**2) * (kmax**3 - kmin**3)

def cov2cor(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def plot_cov_array(cova, covb=None, k=None, kmax=None, num_multipoles=3, label_a=None, label_b=None, vmin=-1, vmax=1, num_ticks=5, **kwargs):
    import matplotlib.pyplot as plot
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    import itertools as itt
    from . import base

    matplotlib.rc('font', size=14, family='STIXGeneral')
    matplotlib.rc('axes', labelsize=18) 
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['figure.dpi']= 100
    matplotlib.rcParams['figure.facecolor']= 'white'

    color_a = '#333'
    color_b = '#e13'
    color_c = '#1e3'

    cmap = LinearSegmentedColormap.from_list("cmap_name", ['#04f', '#fff', '#f30'])

    fig, axes = plot.subplots(1, 1, figsize=(12,10), sharey=True, facecolor='white')
    
    if k is None:
        if isinstance(cova, base.FourierBinned):
            k = cova.kmid
            axes.set_xlabel(r"$k$  [h/Mpc]")
            axes.set_ylabel(r"$k$  [h/Mpc]")
        else:
            k = np.arange(cova.shape[0])
    else:
        axes.set_xlabel(r"$k$  [h/Mpc]")
        axes.set_ylabel(r"$k$  [h/Mpc]")
        
    # if k goes from kmin to kmax only once, repeat it num_multipoles for mono/quadru/hexadeca/...pole
    if len(k) == cova.shape[0]//num_multipoles:
        k = np.concatenate(num_multipoles*[k])
    
    if covb is not None:
        cov = triangle_cov(cova.cor if isinstance(cova, base.Covariance) else cov2cor(cova), 
                            covb.cor if isinstance(covb, base.Covariance) else cov2cor(covb))
    else:
        cov = cova.cor if isinstance(cova, base.Covariance) else cov2cor(cova)
    
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

def plot_cov(cova, covb=None, kmax=None, label_a=None, label_b=None, vmin=-1, vmax=1, num_ticks=5, **kwargs):
    return plot_cov_array(cova=cova.cov, covb=covb.cov if covb is not None else None, k=cova.kmid, kmax=kmax, label_a=None, label_b=None, vmin=-1, vmax=1, num_ticks=5, **kwargs)

def plot_cov_diag(cova, covb=None, k=None, label_a=None, label_b=None, klim=None, portrait=False):
    import matplotlib.pyplot as plot
    if covb is None:
        if portrait:
            fig, axes = plot.subplots(3, 2, figsize=(10,8))
            axes1 = axes.T.flatten()
            axes2 = axes1
        else:
            fig, axes = plot.subplots(2, 3, figsize=(10.5,5))
            axes1 = axes.T.flatten()
            axes2 = axes1
    else:
        from matplotlib.gridspec import GridSpec
        if portrait:
            # Portrait mode
            fig = plot.figure(figsize=(10,8))
            gs = GridSpec(12, 2, figure=fig)
            axes1 = [fig.add_subplot(gs[4*i:4*i+2,j]) for i,j in np.mgrid[0:3, 0:2].T.reshape(-1, 2)]
            axes2 = [fig.add_subplot(gs[4*i+2,j]) for i,j in np.mgrid[0:3, 0:2].T.reshape(-1, 2)]
        else:
            # Landscape mode
            fig = plot.figure(figsize=(10.5, 5))
            gs = GridSpec(8, 3, figure=fig)
            axes1 = [fig.add_subplot(gs[4*i:4*i+2,j]) for i,j in np.mgrid[0:2, 0:3].T.reshape(-1, 2, order='F')]
            axes2 = [fig.add_subplot(gs[4*i+2,j]) for i,j in np.mgrid[0:2, 0:3].T.reshape(-1, 2, order='F')]

    if k is None:
        k  = cova.kmid

    p2 = cova.get_pk(0)**2 if hasattr(cova, 'get_pk') else 1.0

    for (l1, l2), ax1, ax2 in zip([(0,0), (2,2), (4,4), (0,2), (0,4), (2,4)], axes1, axes2):

        a = np.diag(cova.get_ell_cov(l1,l2).cov)/p2
        
        ax1.semilogy(k,  a, label=label_a, c='r')
        ax1.semilogy(k, -a, c='r', ls='dashed')
        
        ax1.set_ylabel(r"$C_{l1l2}(k,k)/P_0(k)^2$".replace('l1', str(l1)).replace('l2', str(l2)))
        ax2.set_xlabel('k [h/Mpc]')

        if covb is not None:
            b = np.diag(covb.get_ell_cov(l1,l2).cov)/p2

            ax1.semilogy(k,  b, label=label_b, c='k')
            ax1.semilogy(k, -b, c='k', ls='dashed')
            
            ax2.plot(k,  a/b-1, label='frac. diff', c='r')
            ax2.axhline(0, c='k', ls='dashed')
        
            frac_lim = 3*np.std((a/b-1)[2:])
            ax2.set_ylim(-frac_lim, frac_lim)

            ax1.set_xticks([])

            if klim is not None:
                ax2.set_xlim(*klim)
        
        if klim is not None:
            ax1.set_xlim(*klim)

    fig.get_axes()[0].legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4 if covb is None else 0.005)

    return fig, axes1, axes2
