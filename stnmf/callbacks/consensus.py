"""
Consensus analysis
==================
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet

from ..util import tqdm

__all__ = [
    'callback',
    'cpcc',
    'affiliation',
]


def callback(self, i, itor, callback_data, sparsities, num_rep=30,
             max_elem=25000):
    """
    Callback function to run consensus analysis

    The results are provided in a :mod:`numpy.ndarray` as cophenetic
    correlation coefficients stored in the key 'cpcc' in the dictionary
    `callback_data`.

    Parameters
    ----------
    sparsities : (n,) array_like
        List of `n` regularization parameter values to probe

    num_rep : int, optional
        Number of repetitions for each parameter to check consensus
        over. Default is 30

    max_elem : int, optional
        Number of maximum elements to consider in consensus check,
        Higher number may require a lot of memory. Default is 25000

    Notes
    -----
    Selecting `max_elem` elements is done by sampling without
    replacement. The indices are selected at initialization and are
    retained throughout the analysis to always compare the same
    elements.

    .. warning:: Even with identical random number generator and seed,
                 the implementation of choice-without-replacement
                 differs from MATLAB. The consensus analysis with
                 `max_elem < m` is thus not compatible with MATLAB!

    .. note:: Implemented for finding the suitable sparsity
              regularization parameter for subunit recovery with
              :class:`stnmf.STNMF` only. No other parameter type or
              :mod:`stnmf.mf` class is supported.

    For more details on the implementation, visit the source code.

    Examples
    --------
    The code below runs the consensus analysis on a given parameter
    range.

    >>> from stnmf import STNMF
    >>> from stnmf.callbacks import consensus
    >>> results = dict()
    >>> STNMF(ste, callback=consensus, callback_data=results,
    ...       callback_kwargs=dict(sparsities=[0, 0.5, 1.0, 1.5, 2]))
    >>> print(results['cpcc'])
    """
    if i == 0:
        # Initialize before the first iteration
        callback_data['_obj'] = Analysis(self, itor, num_rep, max_elem,
                                         sparsities, callback_data)
    elif i % self.num_iter == 0:
        # Complete repetitions of current parameter
        callback_data['_obj'].advance(i)
    else:
        # Pass through and update progress bar
        callback_data['_obj'].pass_through()


def cpcc(affil):
    """Calculate the cophenetic correlation of an affiliation matrix"""
    affil = np.asarray(affil)
    num_rep, m = affil.shape

    # Create consensus matrix
    cons = np.ones(m * (m-1) // 2, dtype='float32')  # Precision vs. memory
    idx = 0
    for k in range(1, m):
        part = slice(idx, idx + m-k)
        np.equal(affil[:, k:], affil[:, k-1, None]).sum(axis=0, out=cons[part])
        idx += m-k
    cons /= num_rep

    # Get linkage
    cons = np.subtract(1, cons, out=cons)
    z = linkage(cons, method='average')

    # Calculate cophenetic correlation
    with np.errstate(invalid='ignore'):
        coph = cophenet(z, cons)[0]

    return coph


def affiliation(h, elem=slice(None), localized=[], out=None):
    """Create affiliation matrix from encodings"""
    h = np.asarray(h)
    if out is None:
        m = h.shape[1] if isinstance(elem, slice) else len(elem)
        out = np.empty(m, dtype=h.dtype)
    out[:] = np.abs(h[:, elem]).argmax(axis=0)  # Casting does not allow 'out'
    non_loc = np.isin(out, localized, invert=True)
    out[non_loc] = np.nan
    return out


class Analysis:
    """Class for consensus analysis callback"""
    def __init__(self, mf, itor, num_rep, max_elem, sparsities, data):
        assert mf.__class__.__name__ == 'STNMF', 'Supports STNMF only'

        self.mf = mf
        self.num_rep = num_rep
        self.sparsities = sparsities
        self.data = data

        # Allocate arrays
        max_elem = np.clip(max_elem, 0, mf.m) or mf.m
        self.coph = np.full(len(sparsities), np.nan, mf.dtype)
        self.affil = np.full((num_rep, max_elem), np.nan, mf.dtype)
        if mf.m > max_elem:
            # This is not compatible with MATLAB's RNG implementation!
            self.elem = mf.rng.choice(mf.m, max_elem, replace=False)
        else:
            self.elem = slice(None)

        # Adjust iterator and replace iterable
        itor.total = len(sparsities) * num_rep * mf.num_iter
        itor.iterable = range(1, itor.total+1)

        # Create and adjust progress bars
        disabled = itor.disable
        leave = itor.pos == 0
        mininterval = itor.mininterval if not disabled else 0.5
        desc = itor.desc if not disabled else mf.desc
        lpad = max(map(len, [desc, 'Consensus analysis']))
        itor.leave = False
        itor.close()
        self.it_p = tqdm(sparsities, desc='Consensus analysis'.rjust(lpad),
                         unit='parameter', leave=leave, disable=disabled,
                         miniters=1, postfix=dict(sparsity=sparsities[0]))
        self.it_r = tqdm(total=num_rep, desc='Repetition'.rjust(lpad),
                         unit='rep', leave=False, disable=disabled,
                         miniters=1, postfix=dict(seed=0))
        self.it_i = tqdm(total=mf.num_iter, desc=desc.rjust(lpad), leave=False,
                         mininterval=mininterval, disable=disabled)

        # Update callback data for user output and self-reference
        data['_obj'] = self
        data['cpcc'] = self.coph

        # Saved for user if requrested
        if 'affiliation_matrix' in data:
            data['affiliation_matrix'] = self.affil
        if 'selected_indices' in data:
            data['selected_indices'] = self.elem

        # Initialize first run of STNMF
        mf.sparsity = sparsities[0]
        mf.reseed(0)
        mf.init('random')

    def advance(self, i):
        """Advance the consensus analysis after completed repetition"""
        # Inlining attributes as locals (speed and readability)
        mf = self.mf
        it_p = self.it_p
        it_r = self.it_r
        it_i = self.it_i

        # Update progress bars
        it_r.update()
        it_i.reset()
        it_i.clear()
        it_p.n = min(it_p.total, round(it_p.n + 1/it_r.total, 2))
        if it_p.n == round(it_p.n):
            it_p.n = int(it_p.n)
        it_p.update(0)

        # Find current iteration
        total_rep = int((i-1) // mf.num_iter)
        cur_param, cur_rep = divmod(total_rep, self.num_rep)

        # Store affiliations of this repetition
        if mf.num_subunits:  # Only consider localized subunits
            affiliation(mf.h, self.elem, mf.localized, out=self.affil[cur_rep])

        # Calculate consensus among repetitions for this parameter
        if cur_rep == self.num_rep-1:
            if not it_r.disable:
                num_spaces = getattr(it_r, 'ncol', 0)
                it_r.display('Computing consensus...'.ljust(num_spaces))
            self.coph[cur_param] = cpcc(self.affil)

            # Go to next parameter
            it_p.n = min(it_p.total, cur_param+1)
            it_p.update(0)

            # Terminate
            if it_p.n >= it_p.total:
                self.terminate()
                return

            # Reset to next parameter
            mf.seed = 0
            mf.sparsity = self.sparsities[it_p.n]
            it_p.set_postfix(sparsity=mf.sparsity)
            it_r.reset()
            self.affil.fill(np.nan)
        else:
            # Advance to next repetition
            mf.seed += 1

        # Initialize for next run of STNMF
        it_r.set_postfix(seed=mf.seed)
        mf.reseed(mf.seed)
        mf.init('random')

    def pass_through(self):
        """Update progress bar only"""
        self.it_i.update()

    def terminate(self):
        """Make sure the progress bars close on termination"""
        self.it_p.set_postfix(None)
        self.it_p.clear()

        # Close all progress bars
        self.it_i.close()
        self.it_r.close()
        self.it_p.close()

        # Clear temporary values
        try:
            del self.data['_obj']
        except KeyError:
            pass

    def __del__(self):
        self.terminate()
