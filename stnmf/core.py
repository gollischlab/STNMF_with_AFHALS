"""
Spike-triggered non-negative matrix factorization
=================================================
"""
import numpy as np
import shapely
import warnings

from . import mf
from . import spatial
from .plot import all as plot_all

__all__ = [
    'STNMF',
]


class STNMF(mf.MF):
    """
    Spike-triggered non-negative matrix factorization (STNMF)

    An object of the class exposes the following attributes and methods.
    For details, see the documentation of the individual items below.

    Spatial modules and weights
    ---------------------------
    .. autosummary::
      modules       -- Recovered modules
      weights       -- Recovered weights
      subunits      -- Localized subunits (norm. subset of modules)
      ratios        -- Scalar subunit weights (norm. weight)
      num_subunits  -- Number of localized subunits

    Subunit attributes
    ------------------
    .. autosummary::
      outlines      -- Contour outlines
      diameters     -- Diameters in micrometers (μm)
      distances     -- Distances from STA center in micrometers (μm)
      polarities    -- Polarities (positive or negative)
      moransi       -- Moran's I autocorrelation
      localized     -- Indices of localized modules

    Object methods
    --------------
    .. autosummary::
      plot          -- Visualize the STNMF results in a figure
      factorize     -- Inherited function from `stnmf.mf.MF`
      flipsign      -- Flip sign of modules to make weights positive
      sort          -- Sort modules by autocorrelation and weight

    Parameter attributes
    --------------------
    .. autosummary::
      pixel_size    -- Pixel size in micrometers (μm)
      moransi_threshold -- Threshold to distinguish localized modules

    Further attributes
    ------------------
    .. autosummary::
      sta           -- Spike-triggered average (STA), mean of STE
      sta_outline   -- Contour outline of the STA
      sta_diameter  -- Diameter of the STA in micrometers (μm)
      coverage      -- Spatial coverage factor of subunits on STA
      modules_info  -- Collection of the properties for all modules
      res           -- Reconstruction residual

    See Also
    --------
    stnmf.mf : Matrix factorization classes

    Examples
    --------
    >>> from stnmf import STNMF
    >>> stnmf = STNMF(ste)

    The created STNMF object exposes further attributes and functions;
    for example, to visualize the resulting decomposition.

    >>> stnmf.plot()

    Notes
    -----
    Find more information in the documentation of the `__init__` method
    below.
    """
    method_default = 'SparseSemiNMF'

    def __new__(cls, *args, **kwargs):
        """Dynamically create appropriate class from provided method"""
        method = kwargs.get('method', STNMF.method_default)
        if isinstance(method, str):
            mfclass = getattr(mf, method)
        else:
            mfclass = method
        kwargs['method'] = mfclass

        clss = type('STNMF', (mfclass,), dict(STNMF.__dict__))
        clss.factorize.__doc__ = super().factorize.__doc__

        obj = mfclass.__new__(clss)
        obj.__init__(*args, **kwargs)
        return obj

    def __init__(self, ste, r=20, init='nnsvdlrc', flipsign=True,
                 sort=True, factorize=True, *args, **kwargs):
        """
        Spike-triggered non-negative matrix factorization using AF-HALS

        Parameters
        ----------
        ste : (x,y,m) array_like
            Spike-triggered stimulus ensemble with `x*y` pixels and `m`
            spikes

        r : int, optional
            Number of modules to recover. Default is 20

        init : (r,x,y) array_like or {'random', 'nnsvdlrc'}, optional
            Initial modules. If array_like, serves as the initial
            modules directly. If 'random' or 'nnsvdlrc', calls the
            functions from `stnmf.init`. If None, no automatic
            initialization will be performed, and has to be done
            manually by calling `init`. Default is 'nnsvdlrc'

        flipsign : bool, optional
            Flip the sign of the modules to make their mean weights
            positive. Default is True

        sort : bool, optional
            Sort modules by mean weights. Default is True

        factorize : bool, optional
            Factorize on initialization. Default is True

        Keyword Arguments
        -----------------
        method : str or class, optional
            Matrix factorization class. Default is 'SparseSemiNMF'

        sparsity : float, optional
            Sparsity regularization weight if `method` is
            'SparseSemiNMF'; otherwise invalid parameter. Default is 1.7

        seed : int, optional
            Random seed for reproducibility. Default is 0

        dtype : str, type or numpy.dtype, optional
            Number format with which to calculate. Default is 'float32'

        pixel_size : float, optional
            Size of a stixel (stimulus pixel) in micrometers (μm) for
            calculation of spatial diameter and distance. If not set,
            these calculations yield zero. Default is 0

        moransi_threshold : float, optional
            Moran's I (autocorrelation) by which to distinguish
            localized from non-localized modules. Default is 0.25

        Factorization Args
        ------------------
        kwargs
            See :func:`STNMF.factorize`

        Raises
        ------
        ValueError
            If `ste` has incorrect dimensions.

        Notes
        -----
        If `factorize` is True (default), the decomposition into
        subunits is performed immediately with the provided
        `Factorization Args`. If set to False, the factors are
        initialized. The factorization is then started manually by
        calling `factorize` with applicable arguments. In that case, any
        `Factorization Args` meant for the factorization that are passed
        to the initialization are invalid and will raise an error.

        The `ste` may contain masked or NaN values to speed up the
        decomposition.

        .. warning::
           To obtain appropriate `diameters` and `distances`, it is
           essential to provide a value for `pixel_size` or to assign
           the attribute later.

        See Also
        --------
        stnmf.mf : Matrix factorization classes with additional keyword
            arguments
        factorize : For more keyword arguments
        """
        if np.ndim(ste) != 3:
            raise ValueError('ste is expected to have three dimensions')

        # Format input
        self.x, self.y = np.shape(ste)[:2]

        # Handle mask and NaN
        if np.ma.is_masked(ste):
            mask = ~ste.mask.sum(axis=-1, dtype='bool')
        else:
            mask = np.isfinite(np.sum(ste, axis=-1))
        if not np.all(mask):
            v = np.asarray(ste)[mask]
            self.mask = (Ellipsis,) + np.nonzero(mask)
            self.masked = True
        else:
            v = np.reshape(ste, (self.x*self.y, -1))
            self.mask = slice(None)
            self.masked = False

        self._sta = np.mean(ste, axis=-1)
        self._sta.setflags(write=False)
        self.pixel_size = kwargs.pop('pixel_size', 0)  # Default none
        self.moransi_threshold = kwargs.pop('moransi_threshold', 0.25)
        self.autoflip = flipsign
        self.autosort = sort

        # Format initialization
        if init is None or isinstance(init, str):
            kwargs['w0'] = init
        elif self.masked:
            kwargs['w0'] = np.asarray(init)[self.mask].T
        else:
            kwargs['w0'] = np.reshape(init, (r, self.x*self.y)).T

        # Pick apart kwargs for initialization and factorization
        if factorize:
            # If supplying these but factorize is False, throw an error!
            ky = ['num_iter', 'callback', 'callback_data', 'callback_kwargs',
                  'disp', 'tqdm_args']
            kwargs_fac = {k: kwargs.pop(k) for k in ky if k in kwargs}

        # Initialize matrix factorization
        self.method = kwargs.pop('method', STNMF.method_default)
        super(self.__class__, self).__init__(v=v, r=r, *args, **kwargs)

        # Create modules and weights
        self._weights = self.h
        if self.masked:
            # Modules are a non-modifiable copy
            dim = (self.r, self.x, self.y)
            self._modules = np.full_like(self.w, np.nan, shape=dim)
            self._modules[self.mask] = self.w.T
            self._modules.setflags(write=False)
        else:
            # Modules are a modifiable view
            self._modules = self.w.T.reshape(self.r, self.x, self.y)

        # Factorize immediately
        if factorize:
            self.factorize(**kwargs_fac)

    def _reset_localized(self):
        """Reset cached attributes based on localized subunits"""
        self._moransi = None
        self._outlines = None
        self._diameters = None
        self._distances = None
        self._polarities = None

    def _reset_sized(self):
        """Reset cached attributes based on pixel size"""
        self._outlines = None
        self._diameters = None
        self._distances = None
        self._sta_outline = None
        self._sta_diameter = None

    def _reset_reconstrution(self):
        """Reset cached attributes based on reconstruction"""
        self._reset_localized()

    def factorize(self, *args, **kwargs):
        super(self.__class__, self).factorize(*args, **kwargs)
        if self.autoflip:
            self.flipsign()
        if self.autosort:
            self.sort()

    def plot(self, full=True, colors='hsv', cmap=None, **kwargs):
        """
        Visualize STNMF results in a figure

        Parameters
        ----------
        full : bool, optional
            Display the full figure. If False, omit weights and
            autocorrelation plots (bottom row). Default is True

        colors : color, iterable, matplotlib.colors.Colormap, optional
            Single color or color sequence to differentiate localized
            modules. If single color, the same color is used on the
            plots of all localized subunits. If iterable or colormap,
            colors are iterated over plots of localized subunits.
            Default is 'hsv'

        cmap : str or matplotlib.colors.Colormap, optional
            Override colormaps for both STA and modules. By default the
            modules and STA plots have two different colormaps. If
            specified, the colormap is used for all subplots.
            Recommended are 'rwb' for convenient view (red-white-black
            with masked values lightseagreen) and 'filter' for
            conventional spatial contrast filter colors
            (black-gray-white with masked values magenta). If None,
            modules are visualized with 'rwb' and STA with 'filter'.
            Default is None

        Keyword Arguments
        -----------------
        kwargs
            Further arguments may be passed to the plotting routines

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure

        Warns
        -----
        PixelSizeWarning
            If no scale bar will be drawn due to missing pixel size

        See Also
        --------
        stnmf.plot.all : Create figure combining all STNMF results
        """
        self._pixel_size_warning()
        kwargs.update(dict(
            w=self.weights if full else None,
            autocorr=self.moransi,
            pixel_size=self.pixel_size or None,
            colors=colors,
            cmap=cmap,
            moransi_threshold=self.moransi_threshold,
        ))
        fig = plot_all(self.sta, self.modules, self.sta_outline, self.outlines,
                       **kwargs)
        return fig

    def flipsign(self):
        """
        Flip sign of modules for their average weights to be positive
        """
        idx = self.weights.mean(axis=1) < 0
        self.weights[idx] = -self.weights[idx]
        self.w[:, idx] = -self.w[:, idx]  # Use W for views and copies

    def sort(self):
        """
        Sort modules by their absolute average weight, localized first
        """
        idx = np.argsort(np.abs(self.weights.mean(axis=1)))[::-1]
        idx_loc = np.isin(idx, self.localized)
        idx0 = idx[idx_loc]
        idx1 = idx[~idx_loc]
        idx = np.append(idx0, idx1)

        self.modules = self.modules[idx]
        self.weights = self.weights[idx]
        self._reset_localized()

    """
    Spatial modules and weights
    ---------------------------
    """

    @property
    def modules(self):
        """
        Recovered spatial modules

        :type: :class:`numpy.ndarray`

        Notes
        -----
        This array is a reshaped view of the matrix factorization
        features `stnmf.mf.MF.w`.

        However, if the STE contains NaNs or is otherwise masked (i.e.
        `masked` is True), this array is an immutable copy that
        can only be modified by direct assignment.

        See Also
        --------
        subunits : Localized subunits only
        stnmf.mf.MF.w : Matrix factorization modules (features)
        """
        if self.masked:
            self._modules.setflags(write=True)
            self._modules.fill(np.nan)
            self._modules[self.mask] = self.w.T
            self._modules.setflags(write=False)
        return self._modules

    @modules.setter
    def modules(self, value):
        if self.masked:
            self._modules.setflags(write=True)
            self._modules[self.mask] = value[self.mask]
            self._modules.setflags(write=False)
            np.copyto(self.w, self._modules[self.mask].T)
        else:
            np.copyto(self._modules, value)
        del self.res

    @property
    def weights(self):
        """
        Recovered weights for all modules and all spikes

        :type: :class:`numpy.ndarray`

        Notes
        -----
        This array is a view of the matrix factorization encodings
        `stnmf.mf.MF.h`.

        See Also
        --------
        ratios : Scalar weight per subunit
        stnmf.mf.MF.h : Matrix factorization weights (encodings)
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        np.copyto(self._weights, value)
        del self.res

    @property
    def subunits(self):
        """
        Localized subunits, i.e. subset of STNMF modules (normalized)

        :type: :class:`numpy.ndarray`
        """
        su = self.modules[self.localized]
        su /= np.linalg.norm(su, axis=(1, 2), keepdims=True)
        return su

    @property
    def ratios(self):
        """
        Scalar weights/contribution ratios of each localized subunit,
        i.e. subset of STNMF weights (averaged and normalized)

        :type: :class:`numpy.ndarray`
        """
        wgt = self.weights[self.localized].mean(axis=1)
        wgt /= np.linalg.norm(wgt)
        return wgt

    @property
    def num_subunits(self):
        """
        Number of localized subunits as determined by their Moran's I
        autocorrelation

        :type: :class:`int`
        """
        return self.localized.size

    """
    Subunit attributes
    ------------------
    """

    @property
    def outlines(self):
        """
        Contour outlines of the localized subunits at contour level
        equivalent to 1.22 standard deviations

        :type: :class:`numpy.ndarray`

        See Also
        --------
        stnmf.spatial.contour : Fit a contour to an image array
        """
        if self._outlines is None:
            self._outlines = spatial.contour(self.subunits, 1.22, smooth=0.5)
            for o in self._outlines:
                o.setflags(write=False)
            self._outlines.setflags(write=False)
        return self._outlines

    @property
    def diameters(self):
        """
        Diameters of the localized subunits as defined by the effective
        diameter of a two-sigma elliptical Gaussian fit in micrometers
        (μm). Only if `pixel_size` is set

        :type: :class:`numpy.ndarray`

        Warns
        -----
        PixelSizeWarning
            If `pixel_size` is not set

        See Also
        --------
        stnmf.spatial.ellipse : Fit a Gaussian ellipse
        pixel_size : Conversion factor to micrometers (μm)
        """
        self._pixel_size_warning()
        if self._diameters is None:
            el = spatial.ellipse(self.subunits)
            a, b = el.T[2:4]
            self._diameters = 2 * np.sqrt(a*b) * self.pixel_size
            self._diameters.setflags(write=False)
        return self._diameters

    @property
    def distances(self):
        """
        Euclidean distances of localized subunits from STA center in
        micrometers (μm). Only if `pixel_size` is set

        Distances are measured from the outline centroids

        :type: :class:`numpy.ndarray`

        Warns
        -----
        PixelSizeWarning
            If `pixel_size` is not set

        See Also
        --------
        outlines : Contour outlines of the localized subunits
        pixel_size : Conversion factor to micrometers (μm)
        """
        self._pixel_size_warning()
        if self._distances is None:
            c_sub = [shapely.LinearRing(o).centroid for o in self.outlines]
            c_sta = shapely.LinearRing(self.sta_outline).centroid
            self._distances = shapely.distance(c_sub, c_sta) * self.pixel_size
            self._distances.setflags(write=False)
        return self._distances

    @property
    def polarities(self):
        """
        Signs of the localized subunits at maximum intensity (positive
        +1 or negative -1)

        :type: :class:`numpy.ndarray`

        See Also
        --------
        flipsign : Flip sign of modules to make weights positive
        """
        if self._polarities is None:
            filters = self.subunits.reshape(self.num_subunits, self.x * self.y)

            # Find maximum intensity pixel
            peak_idx = np.nanargmax(np.abs(filters), axis=1)
            peak = filters[range(self.num_subunits), peak_idx]
            self._polarities = np.copysign(1, peak, dtype=int)

            # Consider non-flipped subunits
            self._polarities *= np.copysign(1, self.ratios, dtype=int)

            self._polarities.setflags(write=False)
        return self._polarities

    @property
    def moransi(self):
        """
        Moran's I autocorrelation of all modules. To obtain the Moran's
        I of the localized subunits only, index the attribute with
        `localized`

        :type: :class:`numpy.ndarray`
        """
        if self._moransi is None:
            self._moransi = spatial.moransi(self.modules)
            self._moransi.setflags(write=False)
        return self._moransi

    @moransi.deleter
    def moransi(self):
        # This attribute is read-only. Delete to force recalculation
        self._reset_localized()

    @property
    def localized(self):
        """
        Indices of localized subunits among all modules as determined by
        their Moran's I autocorrelation and `moransi_threshold`

        :type: :class:`numpy.ndarray`
        """
        return np.nonzero(self.moransi > self.moransi_threshold)[0]

    """
    General parameters
    ------------------
    """

    @property
    def pixel_size(self):
        """
        Pixel size in micrometers (μm) for calculation of spatial
        diameter and distance

        :type: :class:`float`
        """
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, um):
        if um < 0:
            warnings.warn('Pixel size cannot be negative. Set to 0 instead.')
            um = 0
        # Reset any attribute based on pixel size
        if getattr(self, '_pixel_size', -1) != um:
            self._reset_sized()
        self._pixel_size = um

    def _pixel_size_warning(self):
        if not self.pixel_size:
            warnings.warn(
                'No `pixel_size` set for micrometer conversion!',
                PixelSizeWarning
            )

    @property
    def moransi_threshold(self):
        """
        Threshold of Morans'I to distinguish localized subunits

        :type: :class:`float`

        See Also
        --------
        stnmf.spatial.moransi : Calculate the autocorrelation
        """
        return self._moransi_threshold

    @moransi_threshold.setter
    def moransi_threshold(self, value):
        if not -1 < value < 1:
            warnings.warn('Threshold must be in interval [-1, 1]. Ignored.')
            return
        # Reset any attribute based on localized subunits
        if getattr(self, '_moransi_threshold', -1) != value:
            self._reset_localized()
        self._moransi_threshold = value

    """
    Further attributes
    ------------------
    """

    @property
    def sta(self):
        """
        Spatial profile of the spike-triggered average (STA)

        :type: :class:`numpy.ndarray`
        """
        return self._sta

    @property
    def sta_outline(self):
        """
        Contour outline of the STA at contour level equivalent to 1.22
        standard deviations.

        :type: :class:`numpy.ndarray`

        See Also
        --------
        stnmf.spatial.contour : Fit a contour to an image array
        """
        if self._sta_outline is None:
            self._sta_outline = spatial.contour(self.sta, 1.22, smooth=1.2)
            self._sta_outline.setflags(write=False)
        return self._sta_outline

    @property
    def sta_diameter(self):
        """
        Diameter of the STA as defined by the effective diameter of a
        two-sigma elliptical Gaussian fit in micrometers (μm). Only if
        `pixel_size` is set

        :type: :class:`float`

        Warns
        -----
        PixelSizeWarning
            If `pixel_size` is not set

        See Also
        --------
        stnmf.spatial.ellipse : Fit a Gaussian ellipse
        pixel_size : Conversion factor to micrometers (μm)
        """
        self._pixel_size_warning()
        if self._sta_diameter is None:
            el = spatial.ellipse(self.sta)
            a, b = el[2:4]
            self._sta_diameter = 2 * np.sqrt(a*b) * self.pixel_size
        return self._sta_diameter

    @property
    def coverage(self):
        """
        Fraction of spatial coverage of localized subunits on the STA

        Coverage is determined from the subunit and STA outlines

        To avoid effects of poorly reconstructed subunits that each
        cover the entire area of the STA, only the non-overlapping
        portions across the subunits are considered. The fraction of
        this non-overlapping subunit area covering the STA area is the
        coverage factor. This measures the intersection while
        encouraging non-overlapping tiling of the subunits.

        :type: :class:`float`
        """
        sta = shapely.Polygon(self.sta_outline)
        sub_all = [shapely.Polygon(o) for o in self.outlines]

        # Only consider the area where subunits do not overlap
        sub_diff = shapely.Polygon(None)
        for i in range(self.num_subunits):
            for j in range(i+1, self.num_subunits):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    intersect = sub_all[i].intersection(sub_all[j])
                sub_diff = sub_diff.union(intersect)
        sub_mask = shapely.union_all(sub_all).difference(sub_diff)

        # Fraction of the STA covered by non-overlapping subunit space
        covered = sta.intersection(sub_mask)
        coverage_factor = covered.area / sta.area
        return coverage_factor

    @property
    def modules_info(self):
        """
        Collection of all module properties for all modules for easy
        dot- and multi-indexing

        The attribute is a `numpy.recarray` of size `r` and of the
        following fields. It collects all properties in one combined
        array. This allows to multi-index properties of all or several
        modules at a time, structured in a numpy array.

        :type: :class:`numpy.recarray`

        ================= ==============================================
        :attr:`index`     Module index
        :attr:`module`    Spatial module (not normalized)
        :attr:`weight`    Average contribution weight (not normalized)
        :attr:`moransi`   Moran's I autocorrelation
        :attr:`localized` Whether the modules is considered a subunit
        :attr:`polarity`  Sign of the modules
        :attr:`contour`   Countour outline (for localized only)
        :attr:`diameter`  Diameter (for localized only)
        :attr:`distance`  Distance from STA center (for localized only)
        ================= ==============================================
        """
        dtp = [
            ('index', 'uint16'),
            ('module', self.dtype, (self.x, self.y)),
            ('weight', self.dtype),
            ('moransi', self.dtype),
            ('localized', 'bool'),
            ('polarity', 'int8'),
            ('contour', 'object'),
            ('diameter', 'float32'),
            ('distance', 'float32'),
        ]
        m = np.empty(self.r, dtype=dtp).view(np.recarray)

        loc = self.localized
        m.index = range(self.r)
        m.module = self.modules  # Non-normalized
        m.weight = self.weights.mean(axis=1)  # Non-normalized
        m.moransi = self.moransi
        m.localized[loc] = True
        m.polarity[loc] = self.polarities
        m.contour[loc] = self.outlines
        m.diameter[loc] = self.diameters
        m.distance[loc] = self.distances
        return m


class PixelSizeWarning(UserWarning):
    pass
