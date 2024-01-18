Quickstart
==========

To use :mod:`stnmf`, it first has to be imported. Whether in a Python script or in an IPython environment like jupyter notebooks, the procedure is identical.

Importing
---------

All essential functionality of :mod:`stnmf` is encapsulated within the class :class:`STNMF <stnmf.STNMF>` class.
For advanced usage, individual modules can be imported additionally.

.. code-block:: python

    >>> from stnmf import STNMF

Basic usage
-----------

An :class:`STNMF <stnmf.STNMF>` object performs the matrix factorization and offers an interface for results and visualization for a given ganglion cell (exemplary for a neuron of the sensory pathways).

The :class:`STNMF <stnmf.STNMF>` object is constructed from the spiking response of the ganglion cell to spatiotemporal white-noise stimulation, specifically, from the effective spike-triggered stimulus ensemble (STE).
The STE is expected of a Python :term:`array_like` data type, e.g. a nested :class:`list` or :class:`numpy.ndarray`, and has three dimensions:

- **x** : Spatial width of the stimulus
- **y** : Spatial height of the stimulus
- **s** : Number of spikes

The STE should tightly enclose (`x` and `y`) the receptive field to ensure a low-dimensionality input.

.. note::
    For aid in how to construct the STE, visit the :mod:`stnmf.preprocessing` module.

Additional arguments for the construction of the :class:`STNMF <stnmf.STNMF>` object are optional and may affect the factorization.
These include the upper bound of expected subunits (referred to as `r` *modules*) and - crucially - a suitable sparsity regularization parameter.
If not specified, these optional arguments default to general values.

.. note::
    For optimal subunit recovery, the sparsity value should be carefully chosen. That is possible using the methods of :doc:`../advanced/consensus`.

Factorization
-------------

By default, the STNMF algorithm runs immediately when creating the :class:`STNMF <stnmf.STNMF>` object.
Thus, given the STE array `(x, y, s)`, the localized subunits can be recovered with the default parameters with just one line

.. code-block:: python

    >>> stnmf = STNMF(ste)

.. note::
    Different arguments can be specified when running the STNMF algorithm, such as different sparsity-regularization. For more details, view the class documentation of :class:`STNMF <stnmf.STNMF>` or consult the :doc:`examples`.

Results
-------

The object is now stored in the variable `stnmf`. This :class:`STNMF <stnmf.STNMF>` object exposes many properties.
Among the most accessible are the following, where `r` is the number of modules (the defined upper bound) and `l` is the number of the localized subunits as defined by the sufficient spatial autocorrelation.

:attr:`STNMF.subunits <stnmf.STNMF.subunits>` : *(l, x, y)* :class:`numpy.ndarray`
    Recovered spatially localized subunits (normalized), where `l` is the number of subunits (modules considered to be localized subunits), and `x` and `y` are the spatial dimensions from the STE

:attr:`STNMF.ratios <stnmf.STNMF.ratios>` : *(l,)* :class:`numpy.ndarray`
    Scalar weights/contribution ratios of the subunits (:math:`\ell 2`-normalized), i.e. the subset of STNMF weights averaged over the spikes, where `l` is the number of subunits

:attr:`STNMF.outlines <stnmf.STNMF.outlines>` : *(l,)* :class:`numpy.ndarray`
    Contour outlines of the localized subunits

:attr:`STNMF.diameters <stnmf.STNMF.diameters>` : *(l,)* :class:`numpy.ndarray`
    Diameters in micrometers (μm). Only if :attr:`STNMF.pixel_size <stnmf.STNMF.pixel_size>` as been defined

The pixel size mentioned in the size measurement attributes can be set with

.. code-block:: python

    >>> stnmf.pixel_size = 30  # One stixel (stimulus pixel) is 30 μm wide and tall

.. seealso::

    :class:`STNMF <stnmf.STNMF>`
        For many more subunit and decomposition properties

Visualization
-------------

The results can be visualized conveniently in one figure that compiles the output of all plotting routines automatically (see :doc:`../advanced/plotting` and :mod:`stnmf.plot` for details) with

.. code-block:: python

    >>> stnmf.plot()

.. seealso::

    :meth:`STNMF.plot() <stnmf.STNMF.plot>`
        For styling parameters

    :mod:`stnmf.plot`
        For the visualization module

More details
------------

For more details on usage and concrete usage examples, see :doc:`examples`.
