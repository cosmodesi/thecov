"""This module contains plotting functions for covariances
"""

import numpy as np
import matplotlib.pyplot as plot

import collections.abc

from . import math

def plot_cov(cov, label=None, kmax=None, num_ticks=5, plot_sizes={}, **kwargs):
    '''Plot the correlation matrix of a covariance matrix in array form.

    Parameters
    ----------
    cov : covariance or list of covariances
        Covariance matrix.
    label : str or list of str
        Label(s) for the covariance matrix (or matrices). Default is None.
    kmax : float, optional
        Maximum k to plot. Default is None.
    num_ticks : int, optional
        Number of ticks on the axes. Default is 5.
    **kwargs
        Additional arguments to pass to matplotlib's imshow.
    '''
    
    cova, covb = cov if hasattr(cov, '__len__') else (cov, None)
    label_a, label_b = label if hasattr(label, '__len__') else (label, None)

    import matplotlib.pyplot as plot
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    import itertools as itt
    from . import base

    # matplotlib.rc('font', family='STIXGeneral')
    # matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['figure.dpi']= 150
    matplotlib.rcParams['figure.facecolor']= 'white'

    _plot_sizes = {
        'figsize': (9,8),
        'labelsize': 26,
        'ticksize': 20,
        'legendsize': 20,
    }

    _plot_sizes.update(plot_sizes)

    cmap = LinearSegmentedColormap.from_list("cmap_name", ['#04f', '#fff', '#f30'])

    fig, axes = plot.subplots(1, 1, figsize=_plot_sizes['figsize'], sharey=True, facecolor='white')

    if isinstance(cova, base.FourierBinned) or isinstance(covb, base.FourierBinned):
        k = cova.kmid if isinstance(cova, base.FourierBinned) else covb.kmid
        axes.set_xlabel(r"k  [h/Mpc]", fontsize=_plot_sizes['labelsize'])
        axes.set_ylabel(r"k  [h/Mpc]", fontsize=_plot_sizes['labelsize'])
    else:
        k = np.arange(cova.shape[0])

    num_multipoles = len(cova.ells)
        
    # if k goes from kmin to kmax only once, repeat it num_multipoles for mono/quadru/hexadeca/...pole
    if len(k) == cova.shape[0]//num_multipoles:
        k = np.concatenate(num_multipoles*[k])

    if covb is not None:
        cov = math.triangle_cov(cova.cor if isinstance(cova, base.Covariance) else math.cov2cor(cova),
                           covb.cor if isinstance(covb, base.Covariance) else math.cov2cor(covb))
    else:
        cov = cova.cor if isinstance(cova, base.Covariance) else math.cov2cor(cova)

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
        axes.text((i+1/2)*kbins - 4, (j+1/2)*kbins - 4, f'{2*i}{2*j}', fontsize=_plot_sizes['ticksize'])
        axes.text((j+1/2)*kbins - 4, (i+1/2)*kbins - 4, f'{2*i}{2*j}', fontsize=_plot_sizes['ticksize'])

    if label_a is not None:
        axes.text(kbins*0.04, (num_multipoles-0.18)*kbins,  label_a,  fontsize=_plot_sizes['legendsize'])
    if label_b is not None:
        axes.text((num_multipoles - 0.07*(len(label_b) + 1.5))*kbins, 0.04*kbins, label_b, fontsize=_plot_sizes['legendsize'])

    # axes.text(10, 3*kbins+20, r'$P_\ell(k)$ correlation matrix.')
    plot.yticks(fontsize=_plot_sizes['ticksize'])
    plot.xticks(fontsize=_plot_sizes['ticksize'], rotation=45)

    colorbar = fig.colorbar(axes.imshow(cov, origin='lower', vmin=-1, vmax=1, cmap=cmap, **kwargs), pad=0.01)
    fig.tight_layout()
    return fig, axes, colorbar

def plot_cov_diag(cov, k=None, label=None, klim=None, colors=['k', 'r', 'g', 'b'], portrait=False, logplot=True, fracdif_range=None, div_by_pk=True, plot_sizes={}):
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

    _plot_sizes = {
        'figsize': (15,15) if portrait else (20,10),
        'labelsize': 26,
        'ticksize': 20,
        'legendsize': 20,
    }

    _plot_sizes.update(plot_sizes)

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
    if div_by_pk:
        for c in cov:
            if hasattr(c, 'get_pk'):
                p2 = c.get_pk(0)**2
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

        ax1.set_ylabel(f"$C_{{{l1}{l2}}}(k,k){r'/P_0(k)^2' if div_by_pk else ''}$", fontsize=_plot_sizes['labelsize'])
        ax2.set_xlabel('k [h/Mpc]', fontsize=_plot_sizes['labelsize'])

        if len(cov) > 1:

            ax2.axhline(0, c=colors[0], ls='dashed')

            if fracdif_range == None:
                fracdif_range = 3*np.std((a/b-1)[4:])
            if(np.isnan(fracdif_range) or np.isinf(fracdif_range)):
                fracdif_range = 1.0
            ax2.set_ylim(-fracdif_range, fracdif_range)

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
