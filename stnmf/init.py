"""
Initialization of matrix factorization (:mod:`stnmf.init`)
==========================================================

Collection of functions that offer strategies for initialization of
matrix factorization.

Initialization largely contributes to convergence time and the solution
that is found [1]_ [2]_.

.. autosummary::
    :toctree: generated/

    nnsvdlrc
    random

Notes
-----
The functions provided can be called directly by the matrix
factorization classes by passing their name as the argument `w0`.

Examples
--------
The code below is equivalent.

>>> w = stnmf.init.nnsvdlrc(v, r=20)
>>> stnmf.mf.SparseSemiNMF(v, r=20, w0=w)

>>> stnmf.mf.SparseSemiNMF(v, r=20, w0='nnsvdlrc')

References
----------
.. [1] Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S. (2009).
       Nonnegative Matrix and Tensor Factorizations (1st ed.).
       Chichester, UK: John Wiley & Sons, Ltd.
       https://doi.org/10.1002/9780470747278

.. [2] Atif, S. M., Qazi, S., & Gillis, N. (2019). Improved SVD-based
       initialization for nonnegative matrix factorization using
       low-rank correction. Pattern Recognition Letters, 122, 53–59.
       https://doi.org/10.1016/J.PATREC.2019.02.018
"""
import numpy as np
from scipy.sparse.linalg import svds

from .util import create_rng

__all__ = [
    'nnsvdlrc',
    'random',
]


def random(v, r, rng=None, dtype='float32'):
    """
    Initialize feature matrix by random sampling

    Parameters
    ----------
    v : (n,m) array_like
        Input matrix

    r : int
        Number of components

    rng : numpy.random.Generator or int or None, optional
        Random number generator (RNG) instance. If integer, it will be
        the seed for initializing a new RNG. If None, the RNG will be
        initialized without a seed based on OS context. Default is None

    dtype : str or type, optional
        Number format with which to calculate. Default is 'float32'

    Returns
    -------
    w : (n,r) numpy.ndarray
        Initialized `w`

    See Also
    --------
    stnmf.util.create_rng : Create an RNG instance
    """
    if rng is None or isinstance(rng, int):
        rng = create_rng(seed=rng)

    n, m = np.shape(v)
    w = rng.random((r, n), dtype).T  # Transpose: MATLAB compatibility
    return w


def nnsvdlrc(v, r, lrc=True, dtype='float32'):
    """
    Initialize feature matrix with a variation of non-negative singular
    value decomposition with low-rank correction (NNSVD-LRC)

    For details on NNSVD-LRC [1]_ and its modifications to the algorithm
    see notes below.

    Parameters
    ----------
    v : (n,m) array_like
        Input matrix

    r : int
        Number of components

    lrc : dict or bool, optional
        Perform subsequent low-rank correction as matrix factorization
        on the low-rank approximation `v` to improve the initial
        features `w`. This is considered computationally cheaper than
        factorization on the full `v`. If `lrc` is True, use default
        parameters for creating a `MF` object, with one iteration and
        method 'SparseSemiNMF'. If dict, it serves as keyword arguments
        for initialization of the `MF` object, incl. `num_iter`. See
        `stnmf.MF` for details.

    dtype : str or type, optional
        Number format with which to calculate. Default is 'float32'

    Returns
    -------
    w : (n,r) numpy.ndarray
        Initialized `w`

    Notes
    -----
    Since the input matrix `v` is not non-negative in a semi-NMF/MF
    problem, the first rank-one factor of the SVD is not exclusively
    positive, but may contain positive and negative values. Therefore,
    step 2 in the algorithm of [1]_ is skipped here and step 3 is
    performed on all `r` rank-one factors.
    Additionally, unlike in [1]_, only `w` is initialized, as `h` is to
    be inferred subsequently.

    References
    ----------
    .. [1] Atif, S. M., Qazi, S., & Gillis, N. (2019). Improved
           SVD-based initialization for nonnegative matrix factorization
           using low-rank correction. Pattern Recognition Letters, 122,
           53–59. https://doi.org/10.1016/J.PATREC.2019.02.018
    """
    v = np.asarray(v, dtype=dtype)
    n, m = v.shape

    # Initialize the truncated SVD deterministically by all ones
    init = np.ones(np.min(v.shape), dtype=dtype)

    # Compute truncated SVD with p components (ceil r/2)
    p = int((r+1)/2)
    u, s, vt = svds(v, k=p, v0=init, maxiter=np.iinfo(np.int32).max,
                    random_state=0)

    # svds does not guarantee any ordering
    order = np.argsort(s)[::-1]
    u = u[:, order]
    s = s[order]

    # Integrate the singular values into the components
    s = np.sqrt(np.diag(s))
    w_p = u @ s

    # Flip sign for reproducible component order under sign ambiguity
    max_i = np.abs(w_p).argmax(axis=0)
    sign = np.copysign(1, w_p[max_i, range(p)])
    w_p *= sign[None, ...]

    # Create empty array for W
    w = np.zeros((n, r), dtype=dtype)

    # Add each factor twice: Once positive, once negative inverted
    np.maximum(w_p, 0, out=w[:, ::2])
    np.maximum(-w_p[:, :int(r/2)], 0, out=w[:, 1::2])

    # Ensure nonzero columns/rows
    np.nan_to_num(w, nan=1e-16, copy=False)
    w[:, w.sum(axis=0) == 0] = 1e-16

    # Perform low-rank correction on the low-rank approximation of V
    if lrc is not False:
        h_p = s @ vt[order] * sign[..., None]
        v_p = w_p @ h_p  # Low-rank approximation of V

        # Obtain arguments for factorization
        mf_args = lrc if isinstance(lrc, dict) else dict()
        method = mf_args.pop('method', 'SparseSemiNMF')
        lowrank_iter = mf_args.pop('num_iter', 1)

        # Obtain matrix factorization class
        if isinstance(method, str):
            from . import mf
            mfclass = getattr(mf, method)
        else:
            mfclass = method

        # Run factorization iterations
        fact = mfclass(v_p, r, w0=w, dtype=dtype, **mf_args)
        fact.factorize(num_iter=lowrank_iter, disp=False)
        w = fact.w

    return w
