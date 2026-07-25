"""
Example callback to monitor the reconstruction residual
=======================================================
"""
import numpy as np

__all__ = [
    'callback',
]


def callback(self, i, itor, callback_data, interval=100):
    """
    Callback function to view and record the course of the
    reconstruction residual

    The residual is displayed continuously and is afterwards provided as
    a :mod:`numpy.ndarray` stored in the key 'res' in the dictionary
    `callback_data`.

    Parameters
    ----------
    interval : int, optional
        Compute the residual every `interval` iterations. Default is 100

    Notes
    -----
    This callback function is not specific to :class:`stnmf.STNMF` and
    can be used in conjunction with other :mod:`snmt.mf` classes.

    Examples
    --------
    >>> from stnmf import STNMF
    >>> from stnmf.callbacks import residual
    >>> results = dict()
    >>> stnmf = STNMF(ste, callback=residual, callback_data=results,
    ...               callback_kwargs=dict(interval=100))
    >>> print(results['res'])
    """
    if i == 0:
        # Allocate the array to store the values
        callback_data['res'] = np.zeros(self.num_iter//interval+1, self.dtype)

    if i % interval == 0:
        # Add the current residual
        callback_data['res'][i // interval] = self.res

        # View the residual
        if itor.disable:
            print(f'{i:7}it: {self.res:0.4f}')
        else:
            itor.set_postfix(dict(res=f'{self.res:0.4f}'))
