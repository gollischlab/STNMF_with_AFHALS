"""
Perturbations
=============
"""
import numpy as np

from ..util import create_rng

__all__ = [
    'apply',
    'callback',
]


def callback(self, i, itor, callback_data, interval=100):
    """
    Callback function to apply different perturbations to the modules at
    certain intervals

    Notes
    -----
    Implemented for subunit recovery with :class:`stnmf.STNMF` only. No
    other :mod:`stnmf.mf` class is supported.

    Different perturbations are chosen based on random sampling. This
    should be largely compatible with the random choice implementation
    of MATLAB.

    Examples
    --------
    The code below applies a random perturbation of the modules every
    `interval` iterations.

    >>> from stnmf import STNMF
    >>> from stnmf.callbacks import perturbation
    >>> stnmf = STNMF(ste, callback=perturbation,
    ...               callback_kwargs=dict(interval=100))
    """
    if i == 0:
        # Allocate arrays for callback
        assert self.__class__.__name__ == 'STNMF', 'Supports STNMF only'
        callback_data['res_prev'] = np.inf
        callback_data['modules_prev'] = np.zeros_like(self.modules)
        callback_data['weights_prev'] = np.zeros_like(self.weights)
    elif i % interval == 0:
        res_prev = callback_data['res_prev']
        modules_prev = callback_data['modules_prev']
        weights_prev = callback_data['weights_prev']
        disp = not itor.disable

        printmsg = f'{i:7}it: {self.res:0.4f}'

        if self.res > res_prev:
            printmsg += f' > {res_prev:0.2f} -> Revert.'
            self.modules = modules_prev
            self.weights = weights_prev
        else:
            printmsg += ''.ljust(21)
            np.copyto(modules_prev, self.modules)
            np.copyto(weights_prev, self.weights)
            callback_data['res_prev'] = self.res

        if i < self.num_iter:
            ret = apply(self.modules, self.localized, disp=disp, rng=self.rng,
                        inplace=not self.masked)
            if disp:
                msg = ret[2]
                printmsg += f' {msg}'
                ret = ret[0]
            if self.masked:
                self.modules = ret

        if disp:
            itor.write(printmsg, file=itor.fp)


def apply(modules, localized=[], disp=True, inplace=False, rng=None):
    """
    Apply randomly chosen perturbation to spatial modules

    Parameters
    ----------
    modules : (r,x,y) array_like
        Modules to perturb with spatial dimensions `x` and `y`

    localized : (l,) array_like, optional
        Indices of localized modules, where `l` is the number of
        localized modules. If empty, all modules are considered
        non-localized. Default is []

    disp : bool, optional
        Return additional information about the perturbation performed.
        Default is True

    inplace : bool, optional
        If True, operate on `modules` in-place. Default is False

    rng : numpy.random.Generator or int or None, optional
        Random number generator (RNG) instance. If integer, it will be
        the seed for initializing a new RNG. If None, the RNG will be
        initialized without a seed based on OS context. Default is None

    Returns
    -------
    modules : (r,x,y) numpy.ndarray
        Perturbed modules

    p : int
        Perturbation type that was applied. -1 if none was applicable.
        This value is only returned if `disp` is True.

    msg : str or None
        Perturbation description. None if no perturbation was performed.
        This value is only returned if `disp` is True.
    """
    modules = np.array(modules, copy=not inplace)
    r = modules.shape[0]

    # Initialize RNG
    if rng is None or isinstance(rng, int):
        rng = create_rng(seed=rng)

    # Random choice implementation to match MATLAB's behavior
    def choice(arr):
        """
        Equivalent to MATLAB's
          `arr(randperm(numel(arr), 1))`
        and
          `datasample(arr, 1)` (stats toolbox)
        """
        return rng.choice(arr, p=np.repeat(1/len(arr), len(arr)))

    # Determine localized and non-localized modules
    nonlocalized = np.nonzero(np.isin(range(r), localized))[0]

    # Determine which perturbations are applicable
    pert_candidates = []
    if localized.size:
        pert_candidates.append(reseed)
        module_local = modules[choice(localized)]
    if nonlocalized.size:
        pert_candidates.append(0)
        module_nonlocal = modules[choice(nonlocalized)]
    if localized.size and nonlocalized.size:
        pert_candidates += [duplicate, splitvert, splithorz]

    msg, p = None, -1
    if pert_candidates:
        p = choice(pert_candidates)

        if p:
            msg = p(module_local, module_nonlocal, rng=rng)
        else:
            # Advanced indexing does not allow in-place modification
            m = modules[nonlocalized]
            msg = reseed(m, rng=rng)
            msg += 's (all non-localized)'
            modules[nonlocalized], m = m, None

    return modules, p, msg if disp else modules


def reseed(modules, *args, rng):
    """
    Re-seed modules
    """
    msg = 'Re-seed module'
    modules[:] = rng.random(modules.shape[::-1], dtype=modules.dtype).T
    return msg


def duplicate(module_src, module_dst, rng):
    """
    Duplicate a module
    """
    msg = 'Duplicate a module and add noise to both copies'
    i_max = np.unravel_index(np.nanargmax(np.abs(module_src)),
                             module_src.shape)
    m_max = module_src[i_max]
    module_dst[:] = module_src
    module_src[:] += rng.random(module_src.shape[::-1]).T * m_max
    module_dst[:] += rng.random(module_src.shape[::-1]).T * m_max
    return msg


def splitvert(module_src, module_dst, rng):
    """
    Split a module horizontally
    """
    msg = 'Split one localized module into halves vertically'
    i_max = np.unravel_index(np.nanargmax(np.abs(module_src)),
                             module_src.shape)[0]
    module_dst[i_max:] = module_src[i_max:]
    module_dst[:i_max] = 0
    module_src[i_max:] = 0
    return msg


def splithorz(module_src, module_dst, rng):
    """
    Split a module horizontally
    """
    msg = 'Split one localized module into halves horizontally'
    i_max = np.unravel_index(np.nanargmax(np.abs(module_src)),
                             module_src.shape)[1]
    module_dst[:, i_max:] = module_src[:, i_max:]
    module_dst[:, :i_max] = 0
    module_src[:, i_max:] = 0
    return msg
