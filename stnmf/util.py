"""
Utilities (:mod:`stnmf.util`)
=============================

Utility functions for internal use

.. autosummary::
    :toctree: generated/

    create_rng
    tqdm

"""
import numpy as np
from tqdm import tqdm as std_tqdm

__all__ = [
    'create_rng',
]


def create_rng(seed=None):
    """
    Obtain a new seeded instance of a pseudorandom number generator
    compatible across Python, MATLAB, and R

    Mersenne Twister (MT19937) [1]_.

    Parameters
    ----------
    seed : int, optional
        Initial seed. If None, the RNG will be initialized without a
        seed based on OS context. Default is None

    Returns
    -------
    rng : numpy.random.Generator
        The seeded random number generator

    Notes
    -----
    For exact compatibility with MATLAB, sequences should be sampled in
    column-order, e.g. two-dimensional arrays should be transposed after
    sampling.

    .. warning:: Only basic sampling methods like `random` yield
                 identical sequences of values as in MATLAB. More
                 complex functions like `choice` are not compatible with
                 the twister RNG of MATLAB!

    References
    ----------
    .. [1] Matsumoto, & Nishimura. (1998). "Mersenne twister". ACM
           Transactions on Modeling and Computer Simulation, 8(1), 3–30.
    """
    mt = np.random.MT19937()
    mt.state = np.random.RandomState(seed).get_state()
    rng = np.random.Generator(mt)
    return rng


def tqdm(*args, **kwargs):
    """
    Wrapper function for tqdm exposing the attribute `auto` to switch
    between `tqdm.auto.tqdm` and `tqdm.std.tqdm`.

    Attributes
    ----------
    auto : bool
        If True, use `tqdm.auto.tqdm`. If False resort back to
        `tqdm.std.tqdm`. Default is True

    Notes
    -----
    While `tqdm.auto.tqdm` can distinguish Python from IPython for
    displaying respective progress bars appropriately, jupyter notebooks
    are indistinguishable from the jupyter console. To avoid issues, the
    use of `tqdm` within this package can be adjusted. This here
    function wraps `tqdm` and allows to override its use and to manually
    disable auto detection if desired by setting `stnmf.util.tqdm.auto`
    to False.

    Examples
    --------
    Override the setting and revert to the standard `tqdm` and `trange`.

    >>> import stnmf.util
    >>> stnmf.util.tqdm.auto = False
    """
    if getattr(tqdm, 'auto', True):
        from tqdm.auto import tqdm as auto_tqdm
        return auto_tqdm(*args, **kwargs)
    else:
        return std_tqdm(*args, **kwargs)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)
