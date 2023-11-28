"""
Example callbacks (:mod:`stnmf.callbacks`)
==========================================

Exemplary callbacks illustrating their use with :class:`stnmf.STNMF`.

.. autosummary::
    :toctree: generated/

    consensus
    perturbation
    residual
    stopping_criterion

Detail
------
Callbacks allow to safely modify the algorithm of subunit decomposition
in many aspects without having to dive deep into the NMF code. The user
can specify a function that will be called at each loop iteration in the
iterative process of NMF. Code execution jumps out of the internal code
into the user function and returns afterwards to continue until the next
iteration. The function defined by the user will be supplied with and
has access to the current (mutable) state of the decomposition at each
iteration.

Among many other things, this allows...

* to record the course of certain properties throughout the iterative
  process,

* to have the process be interrupted prematurely based on termination
  criteria,

* to change properties of the current decomposition to modify the course
  of the decomposition,

* and to leverage parts of the algorithm to completely redefine how
  subunits are decomposed.

A callback function can be understood as a plug-in or add-on extension
of the internal code.

"""
__all__ = [
    'consensus',
    'perturbation',
    'residual',
    'stopping_criterion',
]

from .consensus import callback as consensus
from .perturbation import callback as perturbation
from .residual import callback as residual
from .stopping_criterion import callback as stopping_criterion
