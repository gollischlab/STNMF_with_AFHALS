"""
Nonnegative least squares solvers (:mod:`stnmf.nnls`)
=====================================================

Functions that solve or approximate the non-negative least squares
problem which sits at the core of non-negative matrix factorization.

.. autosummary::
    :toctree: generated/

    afhals

"""
import numpy as np

__all__ = [
    'afhals',
]


def afhals(vht, hht, w0, rho=0.0, alpha=0.5, eps=0.1, lmbda=0, inplace=False):
    """
    Accelerated fast hierarchical alternating least squares (AF-HALS)
    update

    AF-HALS is based on fast hierarchical alternating least squares
    (Fast HALS) [1]_ combined with accelerated hierarchical alternating
    least squares (A-HALS) [2]_, as inner update for `W` in:

        :math:`\\mathbf{V} \\approx \\mathbf{W}\\mathbf{H}^\\top`

    Parameters
    ----------
    vht : (n,r) array_like
        Product of `v` and `h.T`, where `v` is an `(n, m)` array_like
        and `h` is an `(r, m)` array_like

    hht : (r,r) array_like
        Product of `h` and `h.T`, where `h` is `(r, m)`

    w0 : (m,r) array_like
        Initialized matrix `w`

    rho : float, optional
        Rho for `w` as computed in [2]_. If zero, no inner update loops
        will be done, which reverts back to fast HALS. Default is 0.0

    alpha : float, optional
        This is the scaling parameter for rho as described in [2]_.
        Default is 0.5

    eps : float, optional
        Epsilon is the scaling parameter for the relative error as
        described in [2]_. Default is 0.1

    lmbda : float, optional
        Weight of sparsity constraint. Default is 0

    inplace : bool, optional
        If True, compute `w` in-place by modifying `w0`. This only
        succeeds if `w` is of type `numpy.ndarray`. Default is False

    Returns
    -------
    w : (m,r) numpy.ndarray
        Updated matrix `w`

    Notes
    -----
    Since the update is symmetric, if updating `h` instead, the first
    three parameters change to the following. Additionally, a suitable
    `rho` should be supplied for `h` instead.

    Other Parameters
    ----------------
    vtw : (m,r) array_like
        Product of `v.T` and `w`, where `v` is an `(n, m)` array_like
        and `w` is an `(n, r)` array_like

    wtw : (r,r) array_like
        Product of `w.T` and `w`, where `w` is an `(n, r)` array_like

    ht0 : (m,r) array_like
        Initialized matrix `h.T`, where `h` is an `(r, m)` array_like

    Notes
    -----
    All of the matrix operations are performed in-place to reduce memory
    cost and computational time on repeated allocations. The update of
    `w` is performed with

    .. math::
        \\mathbf{W}_j\\leftarrow\\left [\\mathbf{W}_j+
        (\\mathbf{V}\\mathbf{H}^\\top)_j-\\mathbf{W}
        (\\mathbf{H}\\mathbf{H}^\\top)_j-\\lambda\\right]_+\\,,

    where :math:`\\mathbf{A}_j` is the :math:`j` th column of
    :math:`\\mathbf{A}` and :math:`[\\mathbf{A}]_+` is the element wise
    :math:`\\max(\\mathbf{A}, 0)`.

    The expensive matrix calculations
    :math:`\\mathbf{V}\\mathbf{H}^\\top` and
    :math:`\\mathbf{H}\\mathbf{H}^\\top` are done separately from the
    outer update loop as suggested by [2]_. Additionally,
    :math:`\\mathbf{W}+(\\mathbf{V}\\mathbf{H}^\\top)` is
    calculated only once for each inner update loop.

    References
    ----------
    .. [1] Cichocki, A., & Phan, A. H. (2009). Fast local algorithms
           for large scale nonnegative matrix and tensor
           factorizations. IEICE Transactions on Fundamentals of
           Electronics, Communications and Computer Sciences, E92-A(3),
           708–721. https://doi.org/10.1587/transfun.E92.A.708

    .. [2] Gillis, N., & Glineur, F. (2012). Accelerated Multiplicative
           Updates and Hierarchical ALS Algorithms for Nonnegative
           Matrix Factorization. Neural Computation, 24(4), 1085–1105.
           https://doi.org/10.1162/NECO_a_00256
    """
    vht = np.asarray(vht)
    hht = np.asarray(hht)
    w = np.array(w0, copy=not inplace)
    n, r = w.shape

    max_iter = int(1 + alpha * rho)
    w_prev = w.copy()
    err_0 = None

    # Pre-allocate memory for in-place operations
    mem_n = np.empty((n,), dtype=w.dtype)
    mem_nr = np.empty_like(w)

    for i in range(max_iter):
        # Inner update loop
        np.add(w, vht, out=mem_nr)
        for j in range(r):
            # Opt: np.maximum(w[:, j] + vht[:, j] - w @ hht[:, j] - lmda, 0)
            np.matmul(w, hht[:, j], out=mem_n)
            np.subtract(mem_nr[:, j], mem_n, out=w[:, j])
            if lmbda > 0:
                np.subtract(w[:, j], lmbda, out=w[:, j])
            np.maximum(w[:, j], 0, out=w[:, j])

        # Additional termination criterion by improvement
        np.subtract(w, w_prev, out=mem_nr)
        norm = np.linalg.norm(mem_nr, ord='fro')
        if not i:
            err_0 = eps * norm
        elif norm <= err_0:
            break
        np.copyto(w_prev, w)

    # Ensure nonzero columns
    np.nan_to_num(w, nan=1e-16, copy=False)
    w[:, w.sum(axis=0) == 0] = 1e-16

    return w
