"""
Spike-triggered analyses (:mod:`stnmf.preprocessing.spiketriggered`)
====================================================================

Collection of functions performing spike-triggered analyses for
multiple cells

.. autosummary::
    :toctree: generated/

    average
    stimulusensemble
    window
    profiles

Data preparation
----------------
:class:`stnmf.STNMF` expects an effective spike-triggered stimulus
ensemble (STE) as input. To format the stimulus and spike response into
an STE the following pipeline can be utilized.

Data requirements
^^^^^^^^^^^^^^^^^

* **Spatiotemporal stimulus** as iterator or iterable with each element
  yielding a `(x,y,trial_length)` :term:`array_like`. For instance,
  that may be a `(num_trials,x,y,trial_length)` :class:`numpy.ndarray`.
  The spatial dimensions should enclose the entire projector screen. It
  will be cropped for each cell. Each array element is a pixel contrast
  value deviating from zero either negatively (darker) or positively
  (brighter).

* **Binned spike counts** as `(num_trials,trial_length,num_cells)`
  :term:`array_like`. Each array element is a non-negative integer
  describing the number of spikes within the (equally sized) time bin.

Pipeline
^^^^^^^^

   >>> from stnmf.preprocessing import spiketriggered

1. Calculate the spike-triggered averages (STAs) of all cells.

   >>> stas = spiketriggered.average(stimulus, spikes, tau)

2. Find the relevant spatial windows around the STAs for all cells.

   >>> crops = spiketriggered.window(stas)

3. Create the temporal profile of the STAs.

   >>> sta_space, sta_temp = spiketriggered.profiles(stas, crops)

4. For each cell `idx` individually, select its respective data

   >>> stim = stimulus[:, crops[idx], :]
   >>> spk = spikes[..., idx]
   >>> temporal = sta_temp[idx]

   and create the STE, the input for STNMF.

   >>> ste = spiketriggered.stimulusensemble(stim, spk, temporal)

.. note::
    Review the individual function signatures to specify additional
    parameters.

"""
import numpy as np

from .. import spatial
from ..util import tqdm

__all__ = [
    'average',
    'stimulusensemble',
    'window',
    'profiles',
]


def average(stimulus, spikes, tau, continuous=False, tqdm_args=None):
    """
    Compute the spike-triggered average (STA) for multiple cells

    Parameters
    ----------
    stimulus : iterator or array_like
        This iterator should yield a `(x,y,trial_length)` array_like
        for each trial, where `trial_length` matches the number
        of elements of `spikes`

    spikes : (num_trials,trial_length,num_cells) array_like
        Binned spike counts of multiple cells. If two dimensional,
        assume `num_cells == 1`

    tau : int
        Temporal filter length

    continuous : bool, optional
        Specify if trials should be treated as connected chunks of a
        continuous recording or if they are separated. Default is False

    tqdm_args : dict, optional
        Arguments to pass to tqdm progress bar. If empty, tqdm is not
        used. Default is None

    Return
    ------
    sta : (num_cells,x,y,tau) numpy.ndarray
        STAs of all cells
    """
    spikes = np.asarray(spikes, dtype='float32')
    if spikes.ndim == 2:
        spikes = spikes[..., None]
    num_trials, num_frames, num_cells = spikes.shape

    # Excluding first tau spikes
    if not continuous:
        spikes = spikes[:, tau-1:, :]
        sum_spikes = spikes.sum(axis=(0, 1))
    else:
        sum_spikes = spikes.reshape(-1, num_cells)[tau-1:].sum(axis=0)

    sta, sts = None, None
    it_spikes = tqdm(spikes, **tqdm_args) if tqdm_args else spikes
    for sp, stim in zip(it_spikes, stimulus):
        # Allocate arrays once on first iteration
        if sta is None:
            x, y = stim.shape[:-1]
            shape = (x*y, tau, num_cells)
            sta = np.zeros(shape, dtype='float32')
            sts = np.zeros(shape, dtype='float32')
            if continuous:
                sp = sp[tau-1:]  # Only on first chunk
                stim_prev = stim[..., -(tau-1):]
        elif continuous:
            stim = np.concatenate((stim_prev, stim), axis=-1)
            stim_prev = stim[..., -(tau-1):]
            num_frames = stim.shape[-1]

        stim = stim.reshape(-1, num_frames).astype('float32')
        for it in range(tau):
            np.matmul(stim[:, it:num_frames-tau+it+1], sp, out=sts[:, it])
        sta += sts

    with np.errstate(invalid='ignore', divide='ignore'):
        sta = np.divide(sta, sum_spikes, dtype='float64')
    sta = np.moveaxis(sta, -1, 0)
    sta = sta.reshape(num_cells, x, y, tau)

    return sta


def window(sta, sd=3):
    """
    Obtain the spatial crop dimensions containing the receptive field
    for multiple cells

    Parameters
    ----------
    sta : (num_cells,x,y,tau) array_like
        STAs of multiple cells. If three dimensional, assume
        `num_cells == 1`

    sd : float, optional
        Standard deviations of a two-dimensional Gaussian fit for the
        spatial windows around the center. Default is 3

    Returns
    -------
    crop : (num_cells,) numpy.ndarray containing tuple of slice
        Object array containing the spatial crop dimensions for all
        cells. Each element is a tuple of slices with
        `(slice(x_min,x_max,None), slice(y_min,y_max,None))`
    """
    sta = np.asarray(sta)
    if sta.ndim == 3:
        sta = sta[None, ...]
    num_cells, x, y, tau = sta.shape

    crop = np.empty(num_cells, dtype='object')
    for i, s in enumerate(sta):
        # Determine preliminary spatial crop
        cropsize = 20
        ind_max = np.abs(s).argmax()
        x_max, y_max, t_max = np.unravel_index(ind_max, s.shape)
        window = np.tile([x_max, y_max], [2, 1]).T + [-cropsize, cropsize+1]
        x_low, y_low, x_high, y_high = np.clip(window.T, 0, (x, y)).reshape(-1)

        # Correct sign, flip spatial STA to positive sign for ellipse fitting
        s = s[x_low:x_high, y_low:y_high, :]
        sta_space = s[..., t_max]
        if -sta_space.min() > sta_space.max():
            sta_space = -sta_space

        # Fit two-dimensional Gaussian and determine crop window
        mux, muy, r_major, r_minor, rad = spatial.ellipse(sta_space, sigma=sd)
        uxy = [r_major, r_minor] * np.cos([rad, rad+np.pi/2])
        vxy = [r_major, r_minor] * np.sin([rad, rad+np.pi/2])
        halfsize = np.linalg.norm([uxy, vxy], axis=1)
        extent = np.int32([[mux, muy] - halfsize, [mux, muy] + halfsize+1])
        extent += [x_low, y_low]  # Add trimmed region for whole screen
        crop[i] = (slice(*extent[:, 0]), slice(*extent[:, 1]))  # Form slices

    return crop


def profiles(sta, crop=None):
    """
    Obtain temporal and spatial profiles of the STA for multiple cells

    Parameters
    ----------
    sta : (num_cells,x,y,tau) array_like
        STAs of multiple cells. If three dimensional, assume
        `num_cells == 1`

    crop : list of tuple of slice, optional
        Windows for the spatial profiles. Output of the `window`
        function. Default is None

    Returns
    -------
    sta_space : (num_cells,) numpy.ndarray
        Object array containing the spatial filter for all cells. Each
        element is a `(crop_x[cell],crop_y[cell],tau)` numpy.ndarray,
        if `crop` is provided, otherwise a `(x,y,tau)` numpy.ndarray

    sta_temp : (num_cells,tau) numpy.ndarray
        Array containing the normalized temporal filters for all cells
    """
    sta = np.asarray(sta)
    if sta.ndim == 3:
        sta = sta[None, ...]
    num_cells, x, y, tau = sta.shape

    if crop is None:
        crop = np.full(num_cells, 0, dtype='O')
        crop[:] = [tuple([Ellipsis])]

    sta_space = np.empty(num_cells, 'O')
    sta_temp = np.empty((num_cells, tau), 'float32')
    for i, s in enumerate(sta):
        ind_max = np.abs(s).argmax()
        x_max, y_max, t_max = np.unravel_index(ind_max, s.shape)
        sta_temp[i] = s[x_max, y_max]
        sta_temp[i] /= np.linalg.norm(sta_temp[i])
        sta_space[i] = s[crop[i] + (t_max,)]

    return sta_space, sta_temp


def stimulusensemble(stimulus, spikes, sta_temp, continuous=False,
                     min_spikes_per_bin=1, dtype='float32'):
    """
    Collect the effective spike-triggered stimulus ensemble (STE)

    Parameters
    ----------
    stimulus : iterator or array_like
        This iterator should yield a `(x,y,trial_length)` array_like
        for each trial, where `trial_length` matches the number
        of elements of `spikes`. The stimulus should be cropped around
        the cell's receptive field. See function `window`

    spikes : (num_trials,trial_length,num_cells) array_like
        Binned spike counts of one cell, i.e. `num_cells` has to be 1
        (see notes). If two dimensional, assume `num_cells == 1`

    sta_temp : (tau,) array_like
        Temporal profile of the spike-triggered average. See function
        `profiles`

    continuous : bool, optional
        Specify if trials should be treated as connected chunks of a
        continuous recording or if they are separated. Default is False

    min_spikes_per_bin : int, optional
        Specify the minimum number of spikes per bin to be considered.
        This is useful for cells with high baseline firing rate. Default
        is 1

    dtype : str or numpy.dtype, optional
        Data type in which to perform and return the computations.
        Default is 'float32'

    Returns
    -------
    ste : (x,y,s) numpy.ndarray
        Effective Spike-triggered stimulus ensemble, where `s` is the
        number of spikes

    Raises
    ------
    NotImplementedError :
        If the last dimension of the spikes array is greater than one

    Notes
    -----
    The current implementation only supports spikes from one cell at a
    time.

    Each spike yields one spike-triggered stimulus, that is, multiple
    spikes in one bin result in identical successive spike-triggered
    stimulus frames in the STE.
    """
    sta_temp = np.asarray(sta_temp, dtype=dtype)
    tau = sta_temp.size

    spikes = np.asarray(spikes, dtype='int8')
    if spikes.ndim == 2:
        spikes = spikes[..., None]
    if spikes.shape[-1] != 1:
        raise NotImplementedError('The current implementation only '
                                  'supports spikes from one cell at a '
                                  'time.')

    # Minimum number of spikes to consider
    spikes = np.maximum(spikes - (min_spikes_per_bin-1), 0)

    # Excluding first tau spikes
    if not continuous:
        spikes = spikes[:, tau-1:, :]
        sum_spikes = spikes.sum()
    else:
        sum_spikes = spikes.reshape(-1)[tau-1:].sum()

    ste, sts = None, None
    s_ind = 0
    for sp, stim in zip(spikes, stimulus):
        # Allocate arrays once on first iteration
        if ste is None:
            shape = stim.shape[:2]
            sts = np.empty(shape + (1,), dtype=dtype)
            ste = np.empty(shape + (sum_spikes,), dtype=dtype)
            if continuous:
                sp = sp[tau-1:]  # Only on first chunk
                stim_prev = stim[..., -(tau-1):]
        elif continuous:
            stim = np.concatenate((stim_prev, stim), axis=-1)
            stim_prev = stim[..., -(tau-1):]

        ind = np.nonzero(sp)[0]
        for i in ind:
            np.dot(stim[..., i:i+tau], sta_temp, out=sts[..., 0])
            num_sp = int(sp[i, 0])
            np.copyto(ste[..., s_ind:s_ind+num_sp], sts)
            s_ind += num_sp

    # Flip sign to match polarity
    sign = np.argmax([sta_temp.argmin(), -np.inf, sta_temp.argmax()]) - 1
    ste *= sign

    return ste
