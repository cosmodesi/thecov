"""This module contains utility functions for the covariance calculation, plotting, etc.
"""

import numpy as np
import matplotlib.pyplot as plot

import collections.abc

__all__ = ['triangle_cov', 'cov2cor', 'plot_cov_array', 'plot_cov', 'plot_cov_diag', 'ridgeplot_cov']

def r2c_to_c2c_3d(fourier):
    """Completes a 3D Fourier array generated using PFFT's r2c method with the elements
    that are omitted due to the Hermitian symmetry of the Fourier transform.
    
    Parameters
    ----------
    fourier : array_like
        3D Fourier array generated using PFFT's r2c method.
        
    Returns
    -------
    array_like
        Completed 3D Fourier array.
    """

    fourier_c = np.conj(fourier[:, :, (-2 if fourier.shape[0]%2 == 0 else -1):0:-1])

    fourier_c[1:, :, :] = fourier_c[:0:-1, :, :]
    fourier_c[:, 1:, :] = fourier_c[:, :0:-1, :]

    return np.concatenate((fourier, fourier_c), axis=2)

def triangle_cov(upper, lower, diagonal='upper'):
    '''Construct a covariance matrix from its upper and lower triangular parts.
    
    Parameters
    ----------
    upper : array_like
        Upper triangular part of the covariance matrix.
    lower : array_like
        Lower triangular part of the covariance matrix.
    diagonal : str, optional
        Whether the diagonal of the covariance matrix is in the upper or lower triangular part.
        Default is 'upper'.
        
    Returns
    -------
    array_like
        Covariance matrix.
    '''
    assert diagonal in ['upper', 'lower'], "Argument diagonal should be either 'upper' or 'lower'."
    cov = np.triu(upper) + np.tril(lower)
    cov -= np.diag(np.diag(upper if diagonal == 'lower' else lower))
    return cov
    
# python's enumerate but with a custom step = 2
def enum2(xs, start=0, step=2):
    """Enumerate a sequence with a custom step.
    
    Parameters
    ----------
    xs : sequence
        Sequence to enumerate.
    start : int, optional
        Starting index. Default is 0.
    step : int, optional
        Step of the enumeration. Default is 2.
    
    Returns
    -------
    generator
        Generator of tuples (index, element).
    """
    for x in xs:
        yield (start, x)
        start += step

def sample_from_shell(rmin, rmax, discrete=True):
    """Sample a point uniformly from a spherical shell.
    
    Parameters
    ----------
    rmin : float
        Minimum radius of the shell.
    rmax : float
        Maximum radius of the shell.
    discrete : bool, optional
        If True, the sampled point will be rounded to the nearest integer.
        Default is True.
        
    Returns
    -------
    x,y,z,r : float
        Coordinates of the sampled point.
    """
    
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

def sample_from_cube(rmax, dr, max_modes:int=200):
    
    iL = np.ceil(rmax).astype(int)

    ix, iy, iz = np.mgrid[-iL:iL+1,-iL:iL+1,-iL:iL+1]
    ir = np.sqrt(ix**2 + iy**2 + iz**2)

    sort = (ir/dr).astype(int)
    
    modes = []
    Nmodes = []
    for i in range(np.ceil(rmax/dr).astype(int)):
        mask = sort == i
        N = np.sum(mask)
        if N > max_modes:
            rand_mask = np.random.randint(N, size=max_modes)
            modes.append(np.array([ix[mask][rand_mask],
                                   iy[mask][rand_mask],
                                   iz[mask][rand_mask],
                                   ir[mask][rand_mask]]).T)
        else:
            modes.append(np.array([ix[mask],
                                   iy[mask],
                                   iz[mask],
                                   ir[mask]]).T)
        Nmodes.append(N)

    return modes, Nmodes

def nmodes(volume, kmin, kmax):
    '''Compute the number of modes in a given shell.
    
    Parameters
    ----------
    volume : float
        Volume of the survey.
    kmin : float
        Minimum k of the shell.
    kmax : float
        Maximum k of the shell.
        
    Returns
    -------
    float
        Number of modes.
    '''
    return volume/3/(2*np.pi**2) * (kmax**3 - kmin**3)

def cov2cor(covariance):
    '''Compute the correlation matrix from the covariance matrix.
    
    Parameters
    ----------
    covariance : array_like
        Covariance matrix.
        
    Returns
    -------
    array_like
        Correlation matrix.'''
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def plot_cov_array(cova, covb=None, k=None, kmax=None, num_multipoles=3, label_a=None, label_b=None, vmin=-1, vmax=1, num_ticks=5, **kwargs):
    '''Plot the correlation matrix of a covariance matrix in array form.
    
    Parameters
    ----------
    cova : array_like
        Covariance matrix.
    covb : array_like, optional
        Second covariance matrix to plot. Default is None.
    k : array_like, optional
        k bins of the covariance matrix. Default is None.
    kmax : float, optional
        Maximum k to plot. Default is None.
    num_multipoles : int, optional
        Number of multipoles to plot. Default is 3.
    label_a : str, optional
        Label for the first covariance matrix. Default is None.
    label_b : str, optional
        Label for the second covariance matrix. Default is None.
    vmin : float, optional
        Minimum value of the colorbar. Default is -1.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    num_ticks : int, optional
        Number of ticks on the axes. Default is 5.
    **kwargs
        Additional arguments to pass to matplotlib's imshow.
    '''
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
    '''Plot the correlation matrix of a Covariance object.
    
    Parameters
    ----------
    cova : Covariance
        Covariance matrix.
    covb : Covariance, optional
        Second covariance matrix to plot. Default is None.
    kmax : float, optional
        Maximum k to plot. Default is None.
    label_a : str, optional
        Label for the first covariance matrix. Default is None.
    label_b : str, optional
        Label for the second covariance matrix. Default is None.
    vmin : float, optional
        Minimum value of the colorbar. Default is -1.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    num_ticks : int, optional
        Number of ticks on the axes. Default is 5.
    **kwargs
        Additional arguments to pass to matplotlib's imshow.

    Returns
    -------
    fig, axes, colorbar : matplotlib figure, axes, and colorbar
    '''
    return plot_cov_array(cova=cova.cov, covb=covb.cov if covb is not None else None, k=cova.kmid, kmax=kmax, label_a=None, label_b=None, vmin=-1, vmax=1, num_ticks=5, **kwargs)

def plot_cov_diag(cov, k=None, label=None, klim=None, colors=['k', 'r', 'g', 'b'], portrait=False, logplot=True):
    '''Plot the diagonal of a MultipoleCovariance object.

    Parameters
    ----------
    cov : MultipoleCovariance or list of MultipoleCovariance
        Covariance matrix.
    k : array_like, optional
        k bins of the covariance matrix. Default is None.
    label : str or list of str, optional
        Label for the covariance matrix. Default is None.
    klim : tuple, optional
        k limits to plot. Default is None.
    colors : list of str, optional
    
    Returns
    -------
    fig, axes1, axes2 : matplotlib figure and axes (for main plot and fractional difference)
    '''
    if not isinstance(cov, collections.abc.Sequence):
        cov = [cov]
        
    if label is None:
        label = len(cov)*['']
        
    if not isinstance(label, collections.abc.Sequence):
        label = [label]
        

    if len(cov) == 1:
        if portrait:
            fig, axes = plot.subplots(3, 2, figsize=(15,15))
            axes1 = axes.T.flatten()
            axes2 = axes1
        else:
            fig, axes = plot.subplots(2, 3, figsize=(20,10))
            axes1 = axes.T.flatten()
            axes2 = axes1
    else:
        from matplotlib.gridspec import GridSpec
        if portrait:
            # Portrait mode
            fig = plot.figure(figsize=(15,15))
            gs = GridSpec(12, 2, figure=fig)
            axes1 = [fig.add_subplot(gs[4*i:4*i+2,j]) for i,j in np.mgrid[0:3, 0:2].T.reshape(-1, 2)]
            axes2 = [fig.add_subplot(gs[4*i+2,j]) for i,j in np.mgrid[0:3, 0:2].T.reshape(-1, 2)]
        else:
            # Landscape mode
            fig = plot.figure(figsize=(16, 8))
            gs = GridSpec(8, 3, figure=fig)
            axes1 = [fig.add_subplot(gs[4*i:4*i+2,j]) for i,j in np.mgrid[0:2, 0:3].T.reshape(-1, 2, order='F')]
            axes2 = [fig.add_subplot(gs[4*i+2,j]) for i,j in np.mgrid[0:2, 0:3].T.reshape(-1, 2, order='F')]

    if k is None:
        for c in cov:
            if hasattr(c, 'kmid'):
                k  = c.kmid
                break

    p2 = 1
    div_by_p2 = False
    for c in cov:
        if hasattr(c, 'get_pk'):
            p2 = c.get_pk(0)**2
            div_by_p2 = True
            break
    
    for (l1, l2), ax1, ax2 in zip([(0,0), (2,2), (4,4), (0,2), (0,4), (2,4)], axes1, axes2):
        
        for c,l,color in zip(cov,label,colors):

            a = np.diag(c.get_ell_cov(l1,l2).cov)/p2
            if logplot:
                ax1.semilogy(k,  a, label=l, c=color)
                ax1.semilogy(k, -a, c=color, ls='dashed')
            else:
                ax1.plot(k,  a, label=l, c=color)

        for c,l,color in zip(cov[1:], label[1:], colors[1:]):
            a = np.diag(c.get_ell_cov(l1,l2).cov)/p2
            b = np.diag(cov[0].get_ell_cov(l1,l2).cov)/p2
            
            ax2.plot(k,  a/b-1, label='frac. diff', c=color)
            
        ax1.set_ylabel(f"$C_{{{l1}{l2}}}(k,k){r'/P_0(k)^2' if div_by_p2 else ''}$")
        ax2.set_xlabel('k [h/Mpc]')

        if len(cov) > 1:
        
            ax2.axhline(0, c=colors[0], ls='dashed')

            frac_lim = 3*np.std((a/b-1)[2:])
            if(np.isnan(frac_lim) or np.isinf(frac_lim)):
                frac_lim = 1.0
            ax2.set_ylim(-frac_lim, frac_lim)

        ax1.set_xticks([])

        if klim is not None:
            ax2.set_xlim(*klim)
        
        if klim is not None:
            ax1.set_xlim(*klim)

    if label != len(label)*['']:
        fig.get_axes()[0].legend()
        
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4 if len(cov) > 1 else 0.005)

    return fig, axes1, axes2


def _get_ridgeplot_line(cov, center, nrange):
    assert cov.ndim == 2
    y = cov[center, max(0, center - nrange):min(cov.shape[0], center + nrange + 1)]

    start = nrange - center if nrange > center else 0
    end = start + len(y)

    x = np.arange(start, end)

    return x,y


def ridgeplot_cov(cov, k=None, step=1, nrange=5, figsize=(5,25), logplot=False, hspace=-0.4):
    '''Plot rows of a covariance matrix as ridgeplots.
    
    Parameters
    ----------
    cov : array_like
        Covariance matrix.
    k : array_like, optional
        k bins of the covariance matrix. Default is None.
    step : int, optional
        Step between the rows of the covariance matrix. Default is 1.
    nrange : int, optional
        Number of bins to plot on each side of the diagonal. Default is 5.
    figsize : tuple, optional
        Figure size. Default is (5,25).
    logplot : bool, optional
        If True, plot the y axis in log scale. Default is False.
    hspace : float, optional
        Spacing between the rows. Default is -0.4.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    '''
    if not isinstance(cov, collections.abc.Sequence):
        cov = [cov]
    
    fig, axes = plot.subplots(cov[0].shape[0]//step, 1, figsize=figsize)
    
    for i,ax in enumerate(axes):
        for c in cov:
            x, y = _get_ridgeplot_line(c.cov, i*step, nrange)

            # plotting the distribution
            ax.semilogy(x,y) if logplot else ax.plot(x,y)
            ax.scatter(x,y,s=5)

        ax.axvline(nrange, ls='dotted', c='k', alpha=0.5)

        if k is not None:
            ax.annotate(f'{k[i]:.4f}', xy = (2*nrange + 1, min(y)))
        
        ax.set_xlim((0,2*nrange + 1))
        
        # remove borders, axis ticks, and labels
        ax.axis('off')

    plot.subplots_adjust(hspace=hspace)

    return fig, axes