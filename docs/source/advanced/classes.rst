Custom classes
==============

The class :class:`STNMF <stnmf.STNMF>` serves as an interface exposing relevant subunit properties and easing the use of the underlying factorization.
By default, it encapsulates the matrix factorization class :class:`stnmf.mf.SparseSemiNMF` which inherits from the abstract base class :class:`stnmf.mf.MF`.

The decoupling of classes allows to use custom or modified algorithms to implement the matrix factorization, including different factorization updates and constraints.

When creating a new class it should inherit any of the existing classes :mod:`stnmf.mf` but at least the base class :class:`stnmf.mf.MF`.
At minimum, the class should provide an implementation of the function :func:`stnmf.mf.MF.update_w` but may also override any other attributes and methods.

A custom class `my_class` can be specified when using :class:`STNMF <stnmf.STNMF>` as the argument `method` like so:

.. code-block:: python

    >>> from stnmf import STNMF
    >>> STNMF(ste, method=my_class)

Note, however, that this interface of custom implementations only serves for advanced usage.
Instead, we encourage active discussion on Github or in direct correspondence should limitations or bugs arise.
