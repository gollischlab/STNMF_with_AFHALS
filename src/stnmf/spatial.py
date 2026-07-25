"""
Spatial operations (:mod:`stnmf.spatial`)
=========================================

Collection of functions to perform spatial operations on
(two-dimensional) image arrays

.. autosummary::
    :toctree: generated/

    moransi
    ellipse
    contour

"""
import numpy as np
from scipy import ndimage
from scipy.optimize import leastsq
from scipy.signal import convolve
from scipy.stats import norm as scinorm
import shapely
from skimage import measure
import warnings

__all__ = [
    'moransi',
    'ellipse',
    'contour',
]


def moransi(image, p_value=False):
    """
    Compute Moran's I (spatial autocorrelation)

    Based on Moran's I [1]_ [2]_.

    Parameters
    ----------
    image : (sx,sy) array_like, or (r,sx,sy) array_like
        Two-dimensional image, or stack of `r` two-dimensional images

    p_value : bool, optional
        Return the p-value (significance) of the autocorrelation.
        Default is False

    Returns
    -------
    i : float, or (r,) numpy.ndarray
        Moran's I of the `image` array(s)

    p_value : float, or (r,) numpy.ndarray
        Associated p-value indicating the significance of
        autocorrelation. If `p_value` (function argument) is True

    Raises
    ------
    ValueError
        If `image` is neither two- nor three-dimensional.

    References
    ----------
    .. [1] Moran, P. A. P. (1950). Notes on Continuous Stochastic
           Phenomena. Biometrika. 37 (1): 17–23.
           https://doi.org/10.1093/BIOMET/37.1-2.17

    .. [2] Li, Hongfei; Calder, Catherine A.; Cressie, Noel (2007).
           Beyond Moran's I: Testing for Spatial Dependence Based on
           the Spatial Autoregressive Model. Geographical Analysis. 39
           (4): 357–375.
           https://doi.org/10.1111/j.1538-4632.2007.00708.x
    """
    ndim = np.ndim(image)
    if ndim not in [2, 3]:
        raise ValueError('image is expected to be two- or three-dimensional')

    # Mean of empty slice
    if np.size(image) == 0:
        if ndim == 3:
            i = np.zeros_like(image, shape=np.shape(image)[0])
            return (i, np.full_like(i, np.nan)) if p_value else i
        else:
            return (0, 0) if p_value else 0

    # Copy and fill NaNs
    s = np.nan_to_num(image, copy=True)

    sx, sy = s.shape[-2:]
    n = sx * sy
    n_edge = np.max(2 * (sx + sy) - 4, 0)

    # Subtract total mean from each element
    s -= s.mean(axis=(-2, -1), keepdims=True)
    sum_squared = (s**2).sum(axis=(-2, -1))
    sum_weights = 4 * n  # Each element has four weights
    sum_weights -= n_edge + 4  # Edge elements have no neighbors

    # Binary filter defining the neighbors
    filt = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=s.dtype)  # Type for performance
    if ndim == 3:
        filt = filt[None, ...]

    # Multiply each element with the sum of its neighbors
    neighbors_prod = s * convolve(s, filt, mode='same')
    neighbors_prod_sum = neighbors_prod.sum(axis=(-2, -1))

    with np.errstate(invalid='ignore'):
        i = n * neighbors_prod_sum / sum_weights / sum_squared

    if not p_value:
        return i

    # Z-score
    n_squared = n**2
    sum_weights_squared = sum_weights ** 2
    s1 = sum_weights * 2
    s2 = 4*16 + (n_edge-4)*36 + (n-n_edge)*64
    with np.errstate(invalid='ignore'):
        s3 = (s**4).sum(axis=(-2, -1)) / (1/n * sum_squared**2)
    s4 = (n_squared - 3*n + 3) * s1 - n*s2 + 3*sum_weights_squared
    s5 = (n_squared - n) * s1 - 2*n*s2 + 6*sum_weights_squared

    mu = -1 / (n - 1)
    var = (n*s4 - s3*s5) / ((n-1)*(n-2)*(n-3) * sum_weights_squared) - mu**2
    z_score = (i - mu) / np.sqrt(var)

    # P-value
    p_value = scinorm.sf(np.abs(z_score))*2

    return i, p_value


def ellipse(image, sigma=2.0):
    """
    Fit a two-dimensional Gaussian to an image and return a
    parameterized ellipse

    Parameters
    ----------
    image : (x,y) or (r,x,y) array_like
        Two-dimensional image or stack of `r` independent
        two-dimensional images, where `x` and `y` are the spatial
        dimensions

    sigma : float, optional
        Standard deviations (s.d.) of the mean (Mahalanobis distance)
        for the ellipse boundary. 2 s.d. are best suited for analysis
        of receptive fields of retinal ganglion cells [1]_. Default is
        2.0

    Returns
    -------
    (mux, muy, r_major, r_minor, rad) : tuple of float, (r,5) np.ndarray
        Parameterized ellipse, where `mux` and `muy` are the centroid,
        `r_major` and `r_minor` are the semi-major and semi-minor axes,
        that is, major and minor radii, and `rad` is the rotation in
        radians. Or stack of `r` parameterized ellipses in an `(r, 5)`
        float array

    Raises
    ------
    IndexError
        If `image` is not two-dimensional.

    ValueError
        If `sigma` is smaller or equal zero.

    Examples
    --------
    The parameterized ellipse offers a compressed representation of the
    ellipse. For example, it is useful for calculating the effective
    diameter of a receptive field.

    >>> p = ellipse(spatial_filter)
    >>> pixel_size = 30  # Conversion to micrometers
    >>> a, b = p[2:4]    # Major and minor axes radii
    >>> rf_diameter = 2 * np.sqrt(a*b) * pixel_size

    To obtain an outline of the ellipse in data coordinates:

    >>> mux, muy, r_major, r_minor, rad = ellipse(spatial_filter)
    >>> num_points = 200
    >>> ls = np.linspace(0, 2*np.pi, num_points)

    >>> rot = [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]
    >>> pos = [r_major * np.cos(ls), r_minor * np.sin(ls)]
    >>> points = np.dot(rot, pos) + [[mux], [muy]]

    References
    ----------
    .. [1] Baden, T., Berens, P., Franke, K., Román Rosón, M., Bethge,
           M., and Euler, T. (2016). The functional diversity of retinal
           ganglion cells in the mouse. Nature, 529(7586), 345–350.
           https://doi.org/10.1038/nature16468
    """
    if sigma <= 0:
        raise ValueError('sigma has to be greater than zero')
    if np.ndim(image) in (1, 3):
        return np.float32([ellipse(i, sigma) for i in image])
    if np.ndim(image) != 2:
        raise IndexError('image is expected to be two-dimensional')

    # Copy and fill NaNs
    s = np.nan_to_num(image, copy=True)

    def gaussian2d(mux, muy, stdx, stdy, rho, amp):
        def fnc(x, y):
            expo = (-(2*(1-rho**2))**-1*((x-mux)**2/stdx**2
                    + (y-muy)**2/stdy**2 - 2*rho*(x-mux)*(y-muy)/(stdx*stdy)))
            with np.errstate(invalid='ignore'):
                mul = (2 * np.pi * stdx * stdy * np.sqrt(1 - rho**2)) ** -1
            return amp * mul * np.exp(expo)
        return fnc

    # Initialize Gaussian parameters around maximum value of s
    max_idx = np.abs(s).argmax()
    max_x, max_y = np.unravel_index(max_idx, s.shape)
    peak = s[max_x, max_y]
    thr = np.exp(-0.5)  # Threshold equivalent to one sigma
    aprx_area = np.count_nonzero(s / peak > thr)
    aprx_r = np.sqrt(aprx_area / np.pi)
    params = np.float32([max_x, max_y, aprx_r, aprx_r, 0, 0])

    # Fit Gaussian
    def lsfun(p):
        return np.reshape(s - gaussian2d(*p)(*np.indices(s.shape)), -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mux, muy, stdx, stdy, rho, amp = leastsq(lsfun, params)[0]

    # Create ellipse from Gaussian
    cov = np.array([[stdx**2, rho*stdx*stdy], [rho*stdx*stdy, stdy**2]])
    eigenval, eigenvec = np.linalg.eigh(cov)
    r_major = np.sqrt(eigenval[1]) * sigma  # Largest eigenvalue
    r_minor = np.sqrt(eigenval[0]) * sigma
    with np.errstate(divide='ignore'):
        rad = np.arctan(eigenvec[1, 1] / eigenvec[0, 1])

    # Return parameterized ellipse
    return mux, muy, r_major, r_minor, rad


def contour(image, sigma=1.22, smooth=1.2):
    """
    Create contour outline of an image and return coordinate data points

    Parameters
    ----------
    image : (x,y) or (r,x,y) array_like
        Two-dimensional image or stack of `r` independent
        two-dimensional images, where `x` and `y` are the spatial
        dimensions

    sigma : float, optional
        Standard deviation (s.d.) equivalent to a linearly interpolated
        contour level. A contour level at 1.22 s.d. offers very nice
        visual tiling (for mosaics). Default is 1.22

    smooth : {0.5, 1.2} or float, optional
        Sigma for Gaussian smoothing. This value determines the detail
        of the contour line. The larger the number of pixels, the more
        smoothing is required. As a rule of thumb, 0.5 is suitable for
        subunits and 1.2 for receptive fields. See notes. Default is 1.2

    Returns
    -------
    points : (n,2) or (r,) numpy.ndarray
        Contour outline of `n` data points as xy-coordinate pairs in
        image-pixel coordinates or stack of contours in an array of
        object-type (due to varying number of data points)

    Raises
    ------
    IndexError
        If `image` is not two-dimensional.

    ValueError
        If `sigma` is smaller or equal zero.

    ValueError
        If `smooth` is smaller than zero.

    Notes
    -----
    The image is first nearest-neighbor upsampled and smoothed using a
    `smooth`-sigma Gaussian filter to obtain smooth contour lines. The
    returned contour line is the largest-area island among the positive
    closed contours. The contour is then scaled back down to image-pixel
    resolution.

    When obtaining the contour of a receptive field, a value of 1.2 for
    `smooth` is generally suitable, whereas a value of 0.5 works well
    for the smaller-sized subunits.

    .. note::
        However, suitable values for `smooth` depend on the size of the
        receptive field/subunit in terms of stixels (stimulus/image
        pixels) and may need to be increased for coarser stimuli.

    Examples
    --------
    >>> cntr = contour(spatial_filter)

    Visualize the contour on top of the spatial filter

    >>> from matplotlib import pyplot as plt
    >>> vm = np.nanmax(np.abs(spatial_filter), initial=0) or 1
    >>> pretty = dict(cmap='binary_r', origin='lower', aspect='equal',
    >>>               interpolation='nearest', vmin=-vm, vmax=vm)

    >>> plt.imshow(spatial_filter.T, **pretty)  # Draw image
    >>> plt.plot(*cntr.T, 'r', lw=3)            # Draw contour on top

    Use shapely to perform geometrical analyses

    >>> import shapely
    >>> poly = shapely.Polygon(cntr)
    >>> rf_area = poly.area * pixel_size
    """
    if sigma <= 0:
        raise ValueError('sigma has to be greater than zero')
    if smooth < 0:
        raise ValueError('smooth has to be greater than or equal to zero')
    if np.ndim(image) in (1, 3):
        outl = np.empty(len(image), dtype='object')
        outl[:] = [contour(i, sigma, smooth) for i in image]
        return outl
    if np.ndim(image) != 2:
        raise IndexError('image is expected to be two-dimensional')

    # Constants
    scale = 8 if smooth > 0 else 1
    pad = 15

    # Copy and fill NaNs
    image = np.nan_to_num(image, copy=True)
    image_0 = image.copy()

    # Up-sample with nearest neighbor interpolation
    image = ndimage.zoom(image, scale, order=0, mode='grid-constant',
                         grid_mode=True)

    # Add padding to crop window to avoid open contour at image edges
    image = np.pad(image, pad)

    # Smooth the image
    image = ndimage.gaussian_filter(image, sigma=scale*smooth, mode='nearest')

    # Correct overestimated sigma resulting from smoothing
    if smooth > 0:
        # Ellipse diameter of non-smoothed image
        el = ellipse(image_0, sigma=1)
        diam_nonsmoothed = 2 * np.sqrt(np.prod(el[2:4]))
        # Ellipse diameter of smoothed image
        el = ellipse(image, sigma=1)
        diam_smoothed = 2 * np.sqrt(np.prod(el[2:4])) * 1/scale
        # Rescale sigma with the discrepancy
        diam_min = min(diam_smoothed, 1.75)  # Minimum size
        sigma *= max(diam_nonsmoothed, diam_min) / diam_smoothed

    # Max-scale image to positive unity
    max_idx = np.unravel_index(np.abs(image).argmax(), image.shape)
    peak = image[max_idx]
    image /= peak

    # Contour level is the interpolated s.d.
    contour_level = np.exp(-sigma**2 / 2)
    contours = measure.find_contours(image, contour_level)

    # Select largest-area positive contour (exclude holes and islands)
    contours = [c for c in contours if len(c) >= 4]  # Needed next line
    positive = [c for c in contours if not shapely.LinearRing(c).is_ccw]
    area = lambda c: shapely.Polygon(c).area  # noqa: E731
    points = np.float32(max(positive, key=area, default=np.empty((0, 2))))

    # Interpolate low-resolution contour line (2nd degree B-splines)
    while 0 < len(points) < 100:
        points = measure.subdivide_polygon(points, degree=2)

    # Re-crop and scale back down
    points -= pad  # Remove padding
    points *= 1/scale  # Scale down
    points -= 0.5 * (1 - 1/scale)  # Remove grid offset from scaling
    return points
