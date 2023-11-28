"""
Matrix factorization classes (:mod:`stnmf.mf`)
==============================================

These classes are decoupled from the concepts of STNMF. The matrix
factorizations follow the nomenclature and descriptors of the
non-negative matrix factorization (NMF) literature. The objective is to
solve

    :math:`\\mathbf{V} \\approx \\mathbf{W}\\mathbf{H} \\,,`

where :math:`V \\in \\mathbb{R}^{n\\times m}` is the input matrix
consisting of :math:`n` variables and :math:`m` observations,
:math:`W \\in \\mathbb{R}^{n\\times r}` are the :math:`r`
recovered features and :math:`H \\in \\mathbb{R}^{r\\times m}` are
the corresponding encodings.

The classes inherit from the class `MF` that decomposes
:math:`V` through iterative updates of :math:`W` and :math:`H` to be
implemented by the sub classes.

.. autosummary::
    :toctree: generated/

    MF
    SemiNMF
    SparseSemiNMF

"""
import numpy as np

from .nnls import afhals
from . import init
from .util import create_rng, trange

__all__ = [
    'SparseSemiNMF',
]


class MF(object):
    """Abstract matrix factorization class"""
    desc = 'Matrix factorization'  #: :class:`str`:Readable analysis name

    def __init__(self, v, r, w0='nnsvdlrc', seed=0, rng=None, dtype='float32'):
        """
        Matrix factorization

        Parameters
        ----------
        v : (n,m) array_like
            Input matrix to decompose

        r : int
            Number of features to recover

        w0 : (n,r) array_like, or {'random', 'nnsvdlrc'}, optional
            If provided, serves as the argument `w0` for the `init`
            function to initialize the features `w`. If None, no
            automatic initialization will be performed, and will have to
            be done manually with `init`. Default is 'nnsvdlrc'

        seed : int, optional
            Random number generator (RNG) seed for reproducibility. If
            None, the RNG is initialized without a seed based on OS
            context. The RNG is only used if initialization is random or
            if using custom callbacks. Default is 0

        rng : numpy.random.Generator, optional
            Random number generator (RNG) instance. If provided, this
            overrides `seed`. Otherwise, the default RNG (twister) will
            be initialized with `seed`. The RNG is only used if
            initialization is random or if using custom callbacks.
            Default is None

        dtype : str or type, optional
            Number format with which to calculate. Default is 'float32'

        Raises
        ------
        ValueError
            If matrix sizes mismatch.

        ValueError
            If `r` is smaller than two.

        See Also
        --------
        stnmf.init : Initialization procedures
        stnmf.util.create_rng : Default RNG instance
        """
        if np.ndim(v) != 2 or min(np.shape(v)) < 2:
            raise ValueError('v is expected to be two-dimensional')
        if r < 2:
            raise ValueError('r has to be at least 2 or larger')

        self.dtype = dtype
        self.r = r

        # Copy V if necessary
        self.v = np.asarray(v, dtype=dtype)
        self.n, self.m = v.shape

        # Allocate memory
        self.h = np.zeros((r, self.m), dtype=dtype)
        """:class:`numpy.ndarray`:Encodings"""
        self.w = np.zeros((self.n, r), dtype=dtype)
        """:class:`numpy.ndarray`:Features"""
        self._res = None

        # Set up random number generator
        if rng is not None:
            self.seed = None
            self.rng = rng
        else:
            self.reseed(seed)

        # Initialize factors
        self.is_initialized = False
        if w0 is not None:
            self.init(w0)

    def init(self, w0='nnsvdlrc', lrc_kwargs=None):
        """
        Initialize features `w`

        Parameters
        ----------
        w0 : (n,r) array_like or {'random', 'nnsvdlrc'}, optional
            Initial features `w`. If array_like, serves as the initial
            `w` directly. If 'random' or 'nnsvdlrd', calls the functions
            from `stnmf.init`. Default is 'nnsvdlrc'

        lrc_kwargs : dict, optional
            Keyword arguments for low-rank correction. Ignored if `w0`
            is not 'nnsvdlrc'. Default is None

        Raises
        ------
        ValueError
            If array sizes mismatch.
        """
        if isinstance(w0, str):
            f = getattr(init, w0)
            if w0 == 'nnsvdlrc':
                method = getattr(self, 'method', self.__class__)
                lrc = dict(method=method, num_iter=1)
                lrc.update(lrc_kwargs or dict())
                arg = lrc  # Parameter lrc for 'nnsvdlrc'
            elif w0 == 'random':
                rng = self.rng
                arg = rng  # Parameter rng for 'random'
            self.w[:] = f(self.v, self.r, arg, self.dtype)
        else:
            if np.shape(w0) != (self.n, self.r):
                raise ValueError('w0 has invalid dimensions')
            np.copyto(self.w, w0)

            # Ensure nonzero columns
            np.nan_to_num(self.w, nan=1e-16, copy=False)
            self.w[:, self.w.sum(axis=0) == 0] = 1e-16

        self.update_h()
        del self.res
        self.is_initialized = True

    def update_w(self):
        """Update features `w`"""
        raise NotImplementedError

    def update_h(self):
        """Update encodings `h` using Moore–Penrose pseudoinverse"""
        np.matmul(np.linalg.pinv(self.w), self.v, out=self.h)

    def residual(self):
        """
        Re-compute reconstruction error

        See Also
        --------
        res : Reconstruction error (residual)
        """
        self._res = np.linalg.norm(self.v - self.w @ self.h, ord='fro')
        return self._res

    @property
    def res(self):
        """
        Reconstruction error (residual) based on Frobenius norm

        :type: :class:`float`
        """
        return self._res or self.residual()

    @res.deleter
    def res(self):
        self._reset_reconstrution()
        self._res = None

    def _reset_reconstrution(self):
        pass

    @np.errstate(divide='ignore', invalid='ignore')
    def factorize(self, num_iter=1000, callback=None, callback_data=None,
                  callback_kwargs=None, disp=True, tqdm_args=None):
        """
        Factorize by iterative updates

        Parameters
        ----------
        num_iter : int, optional
            Number of update iterations. Default is 1000

        callback : function (**kwargs) -> bool, optional
            Callback function to be called after each iteration. See
            notes below for details on the provided function arguments.
            A return value of False will terminate the iteration
            prematurely. Default is None

        callback_data : dict, optional
            Dictionary to store callback data. Depending on the callback
            this dictionary will contain the callback results. Ignored,
            if callback is None. Default is None

        callback_kwargs : dict, optional
            Dictionary serving as additional keyword arguments passed to
            the callback if specified. Ignored, if callback is None.
            Default is None

        disp : bool, optional
            Show progress bar. Equivalent to passing `disabled=True` in
            `tqdm_args`. Default is True

        tqdm_args : dict, optional
            Keyword arguments for `tqdm` progress bar. Ignored if `disp`
            if False. Default is None

        Notes
        -----
        The `callback` function will be supplied with the following
        keyword arguments. Additional arguments can be passed through
        `callback_kwargs`.

        Callback Args
        -------------
        self : stnmf.mf.MF
            Current MF object. Attributes are mutable

        i : int
            Index of the current (completed) iteration. Initialization
            is zero

        itor : tqdm.tqdm
            Iterator used during the factorization. Attributes are
            mutable

        callback_data : dict
            Dictionary to store and preserve callback data. Mutable
        """
        self.num_iter = num_iter  # For later reference
        del self.res

        # Ensure initialization (use defaults)
        if not self.is_initialized:
            self.init()

        # Inlining methods as locals (speed optimization)
        update_w = self.update_w
        update_h = self.update_h

        # Create iterator
        kwargs_t = dict(disable=not disp, desc=self.desc, mininterval=0.5)
        kwargs_t.update(tqdm_args or dict())
        itor = trange(1, num_iter+1, **kwargs_t)
        if itor.pos != 0 and 'leave' not in kwargs_t:
            itor.leave = False

        # Duplicated: If-block outside to avoid 'if' at every iteration
        if callback is None:
            with itor:
                for i in itor:
                    update_w()
                    update_h()
        else:
            # Callback arguments
            if not isinstance(callback_data, dict):
                callback_data = dict()
            kwargs_c = dict(callback_kwargs or {})
            kwargs_c.update(dict(itor=itor, callback_data=callback_data))

            # First callback at zero iterations (after initialization)
            if callback(self, 0, **kwargs_c) is False:
                del self.res
                return

            with itor:
                for i in itor:
                    update_w()
                    update_h()

                    del self.res
                    if callback(self, i, **kwargs_c) is False:
                        break

        del self.res

    def reseed(self, seed=None):
        """
        Re-seed the random number generator (RNG)

        Parameters
        ----------
        seed : int or None, optional
            Random number generator (RNG) seed. If None, the RNG is
            initialized without a seed based on OS context. Default is
            None

        Notes
        -----
        The RNG in this class is only used if initialization is random.
        Calling this method will overwrite the attribute `rng`. It is
        not compatible with using a custom `rng`.

        See Also
        --------
        stnmf.util.create_rng : Create an RNG instance
        """
        self.seed = seed
        self.rng = create_rng(seed)


class SemiNMF(MF):
    """Abstract semi-non-negative matrix factorization (NMF) class"""
    desc = 'Semi-NMF'

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        # Non-negative features
        np.maximum(self.w, 0, out=self.w)


class SparseSemiNMF(SemiNMF):
    """
    Sparse semi-non-negative matrix factorization (NMF)

    Sparse semi-NMF solving

        :math:`\\mathbf{V} \\approx \\mathbf{W}_+\\mathbf{H} \\,,`

    where `w` is non-negative and `v` and `h` are not, using accelerated
    fast HALS algorithms and sparsity constraints.

    See Also
    --------
    stnmf.mf.MF : Abstract matrix factorization class
    """
    desc = 'Sparse semi-NMF'

    def __init__(self, *args, sparsity=1.7, **kwargs):
        """
        Sparse semi-non-negative matrix factorization

        Parameters
        ----------
        sparsity : float, optional
            Regularization weight for sparsity. Default is 1.7

        Keyword Arguments
        -----------------
        kwargs
            See `MF` class

        See Also
        --------
        stnmf.mf.MF : Abstract matrix factorization class
        """
        self.sparsity = sparsity
        super().__init__(*args, **kwargs)

        # Inlining attributes as locals (speed and readability)
        r = self.r
        n = self.n
        m = self.m

        # Compute rho for AF-HALS max iterations assuming non-sparsity
        self.rho_w = 1 + (m*n + n*r) / (m*r + m)
        self.alpha = kwargs.get('alpha', 0.5)
        self.eps = kwargs.get('eps', 0.1)

        # Pre-allocate memory for in-place operations
        self._vht = np.empty((n, r), dtype=self.dtype)
        self._hht = np.empty((r, r), dtype=self.dtype)

    def init(self, *args, lrc_kwargs=None, **kwargs):
        lrc_kwargs = lrc_kwargs or dict()
        lrc_kwargs.setdefault('sparsity', self.sparsity)
        super().init(*args, lrc_kwargs=lrc_kwargs, **kwargs)

    def update_h(self):
        """Update encodings `h` using Moore–Penrose pseudoinverse"""
        super().update_h()

        # Normalize rows
        h = self.h
        h /= np.linalg.norm(h, axis=1, keepdims=True)

    def update_w(self):
        """Update features `w` using AF-HALS"""

        # Inlining attributes as locals (speed optimization)
        h = self.h
        ht = h.T
        vht = self._vht
        hht = self._hht

        np.matmul(self.v, ht, out=vht)
        np.matmul(h, ht, out=hht)
        afhals(vht, hht, self.w, rho=self.rho_w, alpha=self.alpha,
               eps=self.eps, lmbda=self.sparsity, inplace=True)
