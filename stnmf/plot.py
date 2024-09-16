"""
Visualization (:mod:`stnmf.plot`)
=================================

Collection of functions for visualization of STNMF results. The function
`all` combines all plotting sub-routines for obtaining a figure with all
relevant information at one glance.

.. autosummary::
    :toctree: generated/

    all
    modules
    outlines
    sta
    weights
    autocorrelation
    scalebar
    getcolors

"""
from cycler import Cycler
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import Divider as mpl_divider
from mpl_toolkits.axes_grid1 import Size as mpl_size
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import warnings

from .spatial import moransi

__all__ = [
    'all',
    'modules',
    'outlines',
    'sta',
]

# Default colors
default_color = '#1b9e77'
_cmap1 = LinearSegmentedColormap.from_list('rwb', ['red', 'white', 'black'])
_cmap1.set_bad(color='lightseagreen', alpha=0.25)
_cmap2 = plt.get_cmap('binary_r')
_cmap2.name = 'filter'  # Accessing private attribute?
_cmap2.set_bad(color='magenta', alpha=0.25)

# Internal constants
_FIGURE_WIDTH = 6.25  # Fits well on A4 and US letter paper


class ColormapWarning(UserWarning):
    """Turn the breaking error into a friendly warning and continue"""
    pass


try:
    plt.colormaps.register(cmap=_cmap1)
except ValueError as e:
    warnings.warn(str(e), ColormapWarning)
try:
    plt.colormaps.register(cmap=_cmap2, name='filter')
except ValueError as e:
    warnings.warn(str(e), ColormapWarning)


def modules(m, colors=default_color, axs=None, **kwargs):
    """
    Plot the modules in a grid figure

    Parameters
    ----------
    m : (r,x,y) array_like
        Spatial modules, where `r` is the number of modules and `x` and
        `y` are the spatial dimensions

    colors : color, iterable, matplotlib.colors.Colormap, optional
        Single color or color sequence for axis frame to differentiate
        localized modules. If single color, the same color is used on
        the plots of all localized subunits. If iterable or colormap,
        colors are iterated over modules. If None, then no coloring is
        applied. Default is '#1b9e77'

    axs : array_like of matplotlib.axes.Axes, optional
        Existing figure axes to draw the modules into. If None, a new
        figure is created. Default is None

    Keyword arguments
    -----------------
    cmap : str or matplotlib.colors.Colormap, optional
        Color map to use. Recommended are 'rwb' for convenient view
        (red-white-black with masked values lightseagreen) and 'filter'
        for conventional spatial contrast filter colors
        (black-gray-white with masked values magenta). Default is 'rwb'

    localized : (l,) array_like, optional
        Indices of localized modules, where `l` is the number of
        localized modules. Not required, but if already available, may
        speed up figure creation. If None, localized modules are
        determined based on the Moran's I threshold in
        `moransi_threshold`

    moransi_threshold : float, optional
        Moran's I threshold to differentiate localized modules. Default
        is 0.25

    Raises
    ------
    ValueError
        If `m` has incorrect shape.

    IndexError
        If number of `axs` and length `m` do not match.
    """
    m = np.asarray(m)
    if m.ndim != 3:
        raise ValueError('m is expected to be three dimensional')
    r, x, y = m.shape

    if axs is None:
        # Estimate a good axes layout
        nx = int(np.ceil(np.sqrt(r)))
        ny = int(np.ceil(r / nx))
        fig_ratio = (nx * x) / (ny * y)
        fig_width = _FIGURE_WIDTH
        fig_height = fig_width / fig_ratio
        fig, axs = plt.subplots(ny, nx, figsize=(fig_width, fig_height),
                                sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.01, right=0.99,
                            bottom=0.01, top=0.99)
        # Remove excess axes
        axs = axs.reshape(-1)
        for i in range(nx*ny - r):
            axs[r + i].remove()
    else:
        axs = np.atleast_1d(axs).reshape(-1)
        if axs.size != r:
            raise IndexError('Number of axes and modules inconsistent')

    cmap = plt.get_cmap(kwargs.get('cmap', 'rwb') or 'rwb')
    im_prop = dict(origin='lower', interpolation='nearest', cmap=cmap)
    plt.setp(axs, xticks=[], yticks=[], aspect='equal')

    # Flip sign of OFF cells for convenient view (black instead of red pixels)
    if cmap.name == 'rwb' and not np.any(m > 0):
        m = -m

    for i, (module, ax) in enumerate(zip(m, axs)):
        vm = np.nanmax(np.abs(module), initial=0) or 1
        ax.imshow(module.T, vmin=-vm, vmax=vm, **im_prop)
        plt.setp(ax.spines.values(), lw=1, color='#404040')

    # Color the localized modules
    if colors is not None:
        localized = kwargs.get('localized', None)
        if localized is None:
            thr = kwargs.get('moransi_threshold', 0.25)
            localized = np.nonzero(moransi(m) > thr)[0]
        colors = getcolors(colors, len(localized))
        for ax, sty in zip(axs[localized], colors()):
            plt.setp(ax.spines.values(), lw=2, **sty)


def outlines(subunits, sta=None, colors=default_color, numbered=True,
             filled=True, ax=None, **kwargs):
    """
    Plot the outlines of the modules

    Parameters
    ----------
    subunits : (r,) array_like
        Outlines of `r` subunits. Each element is an `(2, n)` array_like
        containing `n` coordinate pairs. `n` may be different for each
        outline. Outlines may be arbitrary, i.e. either contours or
        ellipses

    sta : (n,2) array_like, optional
        Outline of spike-triggered average with `n` coordinate pairs.
        Omit to only draw the subunit outlines. Default is None

    colors : color, iterable, matplotlib.colors.Colormap, optional
        Single color or color sequence for the subunit outlines. If
        single color, the same color is used for all outlines. If
        iterable or colormap, colors are iterated over outlines. Default
        is '#1b9e77'

    numbered : bool, optional
        Number the subunit outlines. Numbering is done in the order of
        the subunits provided. Default is True

    filled : bool, opional
        Color fill the subunits outlines. Default is True

    ax : matplotlib.axes.Axes, optional
        Existing figure axis to draw the outlines into. If None, a new
        figure is created. Default is None

    Notes
    -----
    There is no error handling in favor of performance
    """
    if ax is None:
        fig_width = fig_height = 2
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        ax.set(facecolor='none', frame_on=False)

    ax.set(xticks=[], yticks=[], aspect='equal')
    plt.setp(ax.spines.values(), lw=1)
    fprop = dict(ha='center', va='center', clip_on=False, size=12, zorder=4)
    lw = 2

    # Subunit outlines
    colors = getcolors(colors, len(subunits))
    fcolors = getcolors(colors, len(subunits), key='facecolor')
    multi_colored = len(colors) > 1
    sty = dict(clip_on=False)
    sty2 = dict(solid_joinstyle='round', solid_capstyle='round', clip_on=False)
    for i, (outline, col, fc) in enumerate(zip(subunits, colors(), fcolors())):
        outline_t = np.transpose(outline)
        if filled and multi_colored:
            # All outlines different color: Colored outlines
            ax.plot(*outline_t, zorder=3, lw=lw, **col, **sty2)
            ax.fill(*outline_t, zorder=1, ec='none', alpha=0.2, **fc, **sty)
        elif filled:
            # All outlines one color: Black outline, more opaque fill
            ax.plot(*outline_t, zorder=3, lw=lw, color='black', **sty2)
            ax.fill(*outline_t, zorder=1, ec='none', alpha=0.3, **fc, **sty)
        else:
            ax.plot(*outline_t, zorder=3, lw=lw, **col, **sty2)
        if numbered:
            txy = np.mean(outline, 0)
            col = mcolors.to_rgb(col['color'])
            col = mcolors.rgb_to_hsv(col)
            col[2] = min(0.8, col[2])  # Darken color for readability
            col = mcolors.hsv_to_rgb(col)
            ax.text(*txy, f'{i+1:d}', **fprop, color=col)

    # STA outline
    if sta is not None:
        ax.plot(*np.transpose(sta), zorder=2, lw=lw, color='#4c4c4c',
                clip_on=False)


def sta(s, ax=None, **kwargs):
    """
    Plot the spike-triggered average (STA)

    Parameters
    ----------
    s : (x,y) array_like
        Spatial profile of the spike-triggered average

    ax : matplotlib.axes.Axes, optional
        Existing figure axes to draw the STA into. If None, a new figure
        is created. Default is None

    Keyword arguments
    -----------------
    cmap : str or matplotlib.colors.Colormap, optional
        Color map to use. Recommended are 'filter' for conventional
        spatial contrast filter colors (black-gray-white with masked
        values magenta) and 'rwb' for convenient view (red-white-black
        with masked values lightseagreen). Default is 'filter'

    Raises
    ------
    ValueError
        If `s` has incorrect shape.
    """
    s = np.asarray(s)
    if s.ndim != 2:
        raise ValueError('s is expected to be two dimensional')
    x, y = s.shape

    if ax is None:
        fig_ratio = x / y
        fig_width = 2
        fig_height = fig_width / fig_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        ax.set_facecolor('none')

    cmap = plt.get_cmap(kwargs.get('cmap', 'filter') or 'filter')
    im_prop = dict(origin='lower', interpolation='nearest', cmap=cmap)
    ax.set(xticks=[], yticks=[], aspect='equal')
    plt.setp(ax.spines.values(), lw=1, color='#404040')

    vm = np.nanmax(np.abs(s), initial=0) or 1
    ax.imshow(s.T, vmin=-vm, vmax=vm, **im_prop)


def weights(w, m, colors=default_color, ax=None, yticks=True, **kwargs):
    """
    Plot the average weights of the modules

    Parameters
    ----------
    w : (r,) or (r,sp) array_like
        Weight or weights of each module. Either already averaged `(r,)`
        or all weights `(r, sp)`, where `r` is the number of modules and
        `sp` is the number of spikes

    m : (l,) or (r,x,y) array_like
        Indices of localized modules or the spatial modules themselves,
        where `l` is the number of localized modules or `r` is the
        number of all modules and `x` and `y` are the spatial
        dimensions. If modules, localized modules are determined based
        on the Moran's I threshold in `moransi_threshold`

    colors : color, iterable, matplotlib.colors.Colormap, optional
        Single color or color sequence for the data points to
        differentiate localized modules. If single color, the same color
        is used on the plots of all localized subunits.  If iterable or
        colormap, colors are iterated over modules. Default is '#1b9e77'

    ax : matplotlib.axes.Axes, optional
        Existing figure axes to draw the weights into. If None, a new
        figure is created. Default is None

    yticks : bool, optional
        Display weights on y-axis. Default True

    Keyword arguments
    -----------------
    moransi_threshold : float, optional
        Moran's I threshold to differentiate localized modules. Default
        is 0.25

    Raises
    ------
    ValueError
        If `m` or `w` has incorrect shape.
    """
    w = np.squeeze(w)
    if w.ndim == 2:
        w = w.mean(axis=-1)
    if w.ndim != 1:
        raise ValueError('w is expected to be one or two dimensional')
    r = w.size

    ndim = np.ndim(m)
    if ndim == 1:
        thr = np.isin(range(r), m)
    elif ndim == 3:
        thr = moransi(m) > kwargs.get('moransi_threshold', 0.25)
        m = np.nonzero(thr)[0]
    else:
        raise ValueError('m is expected to be one or three dimensional')
    colors = getcolors(colors, len(m), key='mfc')

    if ax is None:
        fig_ratio = 4
        fig_width = 3
        fig_height = fig_width / fig_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        ax.set_facecolor('none')

    ax.set_frame_on(False)
    mprop = dict(mew=1, ms=5, clip_on=False)
    ind = np.arange(r, dtype=int)
    ax.plot(ind[~thr], w[~thr], 'ko', mfc='#a0a0a0', **mprop)
    for x, y, sty in zip(ind[thr], w[thr], colors()):
        ax.plot(x, y, 'ko', **mprop, **sty)

    ax.set_xticks([])
    if not yticks:
        ax.set_yticks([0])
        ax.tick_params(axis='y', pad=-5)
    ax.tick_params(length=0)
    plt.setp(ax.get_yticklabels(), 'fontsize', 8, 'family',
             plt.rcParams['font.family'])
    ax.grid(axis='y', color='lightgray')
    ax.get_ygridlines()[0].set_xdata((0.05, 1))
    margin = np.ptp(w) * 0.1
    ax.set_ylim(min(w.min(), 0) - margin, max(w.max(), 0) + margin)


def autocorrelation(m, colors=default_color, ax=None, **kwargs):
    """
    Plot the spatial autocorrelation of the modules

    Parameters
    ----------
    m : (r,) or (r,x,y) array_like
        Moran's I of the spatial modules, or the modules themselves,
        where `r` is the number of modules `x` and `y` are the spatial
        dimensions. If modules, the autocorrelation is calculated using
        Moran's I

    colors : color, iterable, matplotlib.colors.Colormap, optional
        Single color or color sequence for the data points to
        differentiate localized modules. If single color, the same color
        is used on the plots of all localized subunits. If iterable or
        colormap, colors are iterated over modules. Default is '#1b9e77'

    ax : matplotlib.axes.Axes, optional
        Existing figure axes to draw the autocorrelation into. If None,
        a new figure is created. Default is None

    Keyword arguments
    -----------------
    moransi_threshold : float, optional
        Moran's I threshold to differentiate localized modules. Default
        is 0.25
    """
    ndim = np.ndim(m)
    if ndim == 3:
        m = moransi(m)
    elif ndim != 1:
        raise ValueError('m is expected to be one or three dimensional')
    m = np.clip(m, 0, 1)
    r = m.size
    thr = m > kwargs.get('moransi_threshold', 0.25)
    num_loc = np.count_nonzero(thr)
    colors = getcolors(colors, num_loc, key='mfc')

    if ax is None:
        fig_ratio = 4
        fig_width = 3
        fig_height = fig_width / fig_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        ax.set_facecolor('none')

    mprop = dict(mew=1, ms=5, clip_on=False)
    ind = np.arange(r, dtype=int)
    ax.plot(ind[~thr], m[~thr], 'ko', mfc='#a0a0a0', **mprop)
    for x, y, sty in zip(ind[thr], m[thr], colors()):
        ax.plot(x, y, 'ko', **mprop, **sty)

    ax.set(xticks=[], yticks=[0, 1], ylim=[-0.075, 1.05])
    ax.set_yticks([0.5], minor='True')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=3.5, direction='in', pad=-12, which='both')
    ax.tick_params(axis='y', length=1.75, which='minor')
    plt.setp(ax.spines.values(), 'visible', False)
    plt.setp(ax.spines['left'], 'visible', True, 'bounds', [0, 1])
    plt.setp(ax.get_yticklabels(), 'fontsize', 8, 'family',
             plt.rcParams['font.family'])


def all(s, m, s_outl, m_outl, w=None, autocorr=None,  # noqa: C901
        pixel_size=None, colors='hsv', **kwargs):
    """
    Create figure combining the modules, their outlines, the
    spike-triggered average (STA), and - optionally - the weights and
    autocorrelation values

    Parameters
    ----------
    s : (x,y) array_like
        Spatial spike triggered average (STA), where `x` and `y` are the
        spatial dimensions

    m : (r,x,y) array_like
        Spatial modules, where `r` is the number of modules and `x` and
        `y` are the spatial dimensions

    s_outl : (n,2) array_like
        Outline of STA with `n` coordinate pairs. Default is None

    m_outl : (l,) array_like
        Outlines of the `l` localized subunits. Each element is an
        (2, n) array_like containing `n` coordinate pairs. `n` may be
        different for each outline. Outlines may be arbitrary, i.e.
        either contours or ellipses.

    w : (r,) or (r,sp) array_like, optional
        Weight or weights of each module. Either already averaged `(r,)`
        or all weights `(r, sp)`, where `r` is the number of modules and
        `sp` is the number of spikes. If None, weights and
        autocorrelation plots are omitted (reduced figure). Default is
        None

    autocorr : (r,) array_like, optional
        Moran's I of each spatial modules. Not required, but if already
        available, may speed up figure creation. Default is None

    pixel_size : float, optional
        Size of a stimulus pixel (stixel) in micrometers (μm) for
        displaying the scale bar. If not specified, no scale bar is
        shown. Default is None

    colors : color, iterable, matplotlib.colors.Colormap, optional
        Single color or color sequence to differentiate localized
        modules. If single color, the same color is used on the plots of
        all localized subunits. If iterable or colormap, colors are
        iterated over plots of localized subunits. Default is 'hsv'

    Keyword arguments
    -----------------
    cmap : str or matplotlib.colors.Colormap, optional
        Override colormaps for both STA and modules. By default the
        modules and STA plots have two different colormaps. If
        specified, the colormap is used for all subplots. Recommended
        are 'rwb' for convenient view (red-white-black with masked
        values lightseagreen) and 'filter' for conventional spatial
        contrast filter colors (black-gray-white with masked values
        magenta). If None, modules are visualized with 'rwb' and STA
        with 'filter'. Default is None

    moransi_threshold : float, optional
        Moran's I threshold to differentiate localized modules. Default
        is 0.25

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure

    Raises
    ------
    ValueError
        If `s`, `m`, `outl`, or `w` have incorrect shape.

    IndexError
        If the spatial dimensions of `s` and `m` do not match.
    """
    s = np.asarray(s)
    if s.ndim != 2:
        raise ValueError('s is expected to be two dimensional')

    m = np.asarray(m)
    if m.ndim != 3:
        raise ValueError('m is expected to be three dimensional')
    r, x, y = m.shape

    if (x, y) != s.shape:
        raise IndexError('Spatial dimensions of STA and modules do not match')

    if w is not None:
        w = np.asarray(w)
        if w.ndim == 2:
            w = w.mean(axis=1)
        elif w.ndim != 1:
            raise ValueError('w is expected to be one or two dimensional')
        if w.size != r:
            raise IndexError('Number of modules and weights do not match')
        full = True
    else:
        full = False

    if autocorr is not None:
        autocorr = np.asarray(autocorr)
        if autocorr.ndim != 1:
            raise ValueError('autocorr is expected to be one dimensional')
        if autocorr.size != r:
            raise IndexError('autocorr has wrong length')
    else:
        autocorr = moransi(m)

    # Get localized modules and corresponding colors
    thr = kwargs.setdefault('moransi_threshold', 0.25)
    kwargs['localized'] = loc = np.nonzero(autocorr > thr)[0]
    kwargs['colors'] = getcolors(colors, len(loc))
    cmap = kwargs.get('cmap', None)

    # Create figure layout with exact and identical axes padding/gaps
    fig_width = _FIGURE_WIDTH
    pad_edge_inch = 0.05
    pad_gap_inch = 0.05
    pad_gaph_inch = 0.025
    pad_gap2_inch = 0.1
    pad_gap3_inch = 0.35
    nx = int(np.ceil(np.sqrt(r)))
    ny = int(np.ceil(r / nx))
    px_aspect = y/x

    # Calculate ratio between module and sta plot sizes
    vertl_size = mpl_size.Add(mpl_size.Fixed(pad_gap_inch * max(ny-2, 0)),
                              mpl_size.Scaled(y * ny))

    # Fixed sizes in inches
    sz_edge = mpl_size.Fixed(pad_edge_inch)
    sz_gap = mpl_size.Fixed(pad_gap_inch)
    sz_gaph = mpl_size.Fixed(pad_gaph_inch)
    sz_gap2 = mpl_size.Fixed(pad_gap2_inch)
    sz_gap3 = mpl_size.Fixed(pad_gap3_inch)
    sz_nop = mpl_size.Fixed(0)

    # Subplot sizes
    sz_mod_x = mpl_size.Scaled(x)
    sz_mod_y = mpl_size.Scaled(y)
    sz_sta_y = mpl_size.Fraction(0.5, vertl_size)
    sz_sta_x = mpl_size.Fraction(1/px_aspect, sz_sta_y)
    sz_cbar_y = sz_gap
    sz_cbar_x = mpl_size.Fraction(0.5, sz_sta_x)
    sz_cbars_y_neg = mpl_size.Fixed(-pad_gap_inch*2-pad_gaph_inch)
    sz_sta_y2 = mpl_size.Add(sz_sta_y, sz_cbars_y_neg)
    sz_ext_x = mpl_size.Fixed((fig_width - (pad_edge_inch*2+pad_gap3_inch))/2)
    sz_ext_y = mpl_size.Fraction(0.3, sz_ext_x)

    # Axes segregation
    horzt = [sz_edge] + [sz_mod_x, sz_gap]*nx + [sz_sta_x, sz_edge]
    horzc = [sz_edge] + [sz_mod_x, sz_gap]*nx + [sz_cbar_x, sz_cbar_x, sz_edge]
    horzb = [sz_edge, sz_ext_x, sz_gap3, sz_ext_x, sz_edge]
    vertb = [sz_edge, sz_ext_y, sz_gap2] if full else [sz_edge, sz_nop, sz_nop]
    vertl = vertb + list(np.tile([sz_mod_y, sz_gap], ny)[:-1]) + [sz_edge]
    vertr = vertb + [sz_sta_y2, sz_cbar_y, sz_gaph, sz_cbar_y, sz_gap,
                     sz_sta_y, sz_edge]

    # Create figure and axes
    ax_prop = dict(xticks=[], yticks=[], facecolor='none')
    fig, axs = plt.subplots(r+6 if full else r+4, subplot_kw=ax_prop)

    rect = (0, 0, 1, 1)
    dividerl = mpl_divider(fig, rect, horzt, vertl, aspect=True)
    dividerr = mpl_divider(fig, rect, horzt, vertr, aspect=True)
    dividerc = mpl_divider(fig, rect, horzc, vertr, aspect=True)
    dividerb = mpl_divider(fig, rect, horzb, vertr, aspect=False)

    # Adjust figure size
    sw = np.sum(dividerl.get_horizontal_sizes(fig.canvas), axis=0)
    sh = np.sum(dividerl.get_vertical_sizes(fig.canvas), axis=0)
    w_r = (fig_width - sw[1]) / sw[0]  # Ratio of scaled and fixed sizes
    fig_h = w_r * sh[0] + sh[1]
    fig.set_size_inches(fig_width, fig_h)

    # Arrange subplots
    for i in range(r):
        py, px = divmod(i, nx)
        py = ny-1 - py
        axs[i].set_axes_locator(dividerl.new_locator(px*2+1, 2+py*2+1))
    last_col = nx*2 + 1
    c_end = last_col+2 if pixel_size is None else None
    axs[r+0].set_axes_locator(dividerr.new_locator(last_col, 8))
    axs[r+1].set_axes_locator(dividerr.new_locator(last_col, 3))
    axs[r+2].set_axes_locator(dividerc.new_locator(last_col, 6, c_end))
    axs[r+3].set_axes_locator(dividerc.new_locator(last_col, 4, c_end))
    if full:
        axs[r+4].set_axes_locator(dividerb.new_locator(1, 1))
        axs[r+5].set_axes_locator(dividerb.new_locator(3, 1))

    with plt.rc_context({'font.family': 'Helvetica'}):  # Adjust font

        # Plot subunits
        modules(m, axs=axs[:r], **kwargs)

        # Plot ST
        sta(s, ax=axs[r], **kwargs)

        # Plot outlines
        ax = axs[r+1]
        outlines(m_outl, s_outl, ax=ax, **kwargs)
        ax.set_frame_on(False)
        # ax.set(xlim=[-0.5, x-0.5])  # Scale it to the STA
        ax.margins(y=0.15)  # Auto-scale with some margins

        # Add scale bar
        if pixel_size is not None:
            bar_lengths = np.array([25] + list(range(50, 450, 50)) + [np.inf])
            reasonable_length = x*pixel_size // 2.5
            idx = np.searchsorted(bar_lengths, reasonable_length, side='left')
            scalebar(axs[r], pixel_size, extent=bar_lengths[max(idx-1, 0)],
                     loc='upper right', bbox_to_anchor=[1, 0], label_top=False,
                     pad=pad_gap_inch * 7.2, borderpad=0, sep=2)

        # Add color bar(s)
        cmap_prop = dict(orientation='horizontal', ticks=[])
        mapl = ScalarMappable(norm=None, cmap=cmap or 'filter')
        fig.colorbar(mapl, cax=axs[r+2], **cmap_prop)
        if cmap is not None:
            fig.delaxes(axs[r+3])
        else:
            mapl = ScalarMappable(norm=None, cmap='rwb')
            fig.colorbar(mapl, cax=axs[r+3], **cmap_prop)

        # Extended figure
        if full:
            # Plot weights
            ax = axs[r+4]
            weights(w, loc, ax=ax, yticks=False, **kwargs)
            ax.text(0.99, 1, 'Average weight', ha='right', va='top',
                    fontsize=10, transform=ax.transAxes)

            # Plot autocorrelations
            ax = axs[r+5]
            autocorrelation(autocorr, ax=ax, **kwargs)
            ax.text(0.99, 1, 'Autocorrelation', ha='right', va='top',
                    fontsize=10, transform=ax.transAxes)

    return fig


def scalebar(ax, factor, extent=100, unit='μm', **kwargs):
    """
    Wrapper function for adding a scale bar to an axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Figure axes

    factor : float
        Scaling factor from data coordinates to units

    extent : float, optional
        Length of the scale bar in units. Default is 100

    unit : str, optional
        Unit to be displayed. Default is 'μm'

    Keyword arguments
    -----------------
    kwargs
        Additional arguments to be passed to
        `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`
    """
    if factor == 0:
        return
    size = extent / factor
    text = f'{extent:1.4g} {unit}'

    dd = dict(color='black', label_top=True, sep=1.5, pad=0.15, frameon=False,
              size_vertical=0.1, loc='lower left', bbox_transform=ax.transAxes)
    dd.update(kwargs)

    asb = AnchoredSizeBar(ax.transData, size, text, **dd)
    asb.set(clip_on=False)

    # Change cap style for accurate depiction
    rect = asb.size_bar.get_children()[0]
    rect.set(capstyle='butt', joinstyle='miter')
    ax.add_artist(asb)
    return asb


def getcolors(colors, num, key='color'):
    """
    Turn an arbitrary description of colors into a cycler

    Parameters
    ----------
    colors : color, str, iterable, or matplotlib.colors.Colormap
        Single color or iterable (list, tuple, array_like) of colors in
        any representation, or color map. If str, registered colormaps
        will be searched in matplotlib, otherwise it is assumed a named
        color or if 'auto' the default color cycle is used

    num : int
        Number of colors requested. Colors will repeat after that number

    key : str
        Specify the key of the prop cycle. Default is 'color'

    Returns
    -------
    cycler : cycler.Cycler
        Color cycler with the key word specified with `key`
    """
    num = max(1, num)
    if isinstance(colors, Cycler):
        colors = plt.cycler(colors)
        colors.change_key(colors.keys.pop(), key)
        return colors
    if isinstance(colors, str):
        if colors == 'auto':
            colors = plt.cycler(plt.rcParams['axes.prop_cycle'])
            colors.change_key('color', key)
            return colors
        elif colors in plt.colormaps():
            colors = plt.get_cmap(colors)
    if isinstance(colors, Colormap):
        colors = colors(np.linspace(0, 1, min(colors.N+1, num+1))[:-1])
    else:
        colors = mcolors.to_rgba_array(colors)[:num]
    return plt.cycler(key, colors)
