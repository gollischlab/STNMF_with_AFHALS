"""
Spike-triggered non-negative matrix factorization
=================================================

A fast and versatile implementation of spike-triggered non-negative
matrix factorization (STNMF) based on accelerated fast hierarchical
alternating least squares (AF-HALS) algorithms.

This Python package allows fast inference of receptive-field subunits
from the spiking responses of retinal ganglion cells including methods
of hyperparameter tuning.

Described in the paper:

    |citation-apa|

..
    See documentation for citation:
        https://stnmf.readthedocs.io
    If you use this software, please cite it as indicated.

The package provides code to recover spatial subunits from a
spike-triggered stimulus ensemble using matrix factorization to solve

    :math:`\\mathbf{V} \\approx \\mathbf{W} \\mathbf{H} \\,,`

with a non-negativity constraint on :math:`\\mathbf{W}`. Here
:math:`\\mathbf{V}` is the effective spike-triggered stimulus ensemble
(STE), an `(n, m)` :term:`array_like` consisting of `n` pixels and `m`
spikes and :math:`\\mathbf{W}` is an `(n, r)` :term:`array_like`
containing the recovered spatial modules with the corresponding weights
:math:`\\mathbf{H}` as `(r, m)` :term:`array_like`.
"""
__version__ = '0.9.0'
__all__ = [
    'STNMF',
    'preprocessing',
    'spatial',
    'plot',
    'init',
    'nnls',
    'callbacks',
]

from . import spatial
from . import preprocessing
from . import plot
from . import init
from . import nnls
from . import callbacks
from .core import STNMF
