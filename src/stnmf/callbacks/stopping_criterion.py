"""
Example callback to implement a custom stopping criterion
=========================================================
"""
import numpy as np

__all__ = [
    'callback',
]


def callback(self, i, itor, callback_data, interval=100, delta=1e-6):
    """
    Callback function to implement a custom stopping criterion based on
    relative reconstruction improvement

    Parameters
    ----------
    interval : int, optional
        Evaluate stopping criterion every `interval` iterations

    delta : float, optional
        Fraction of first-iteration improvement. When the difference to
        the previous reconstruction error falls below, terminate the
        factorization. Default is 1e-6

    Notes
    -----
    This callback function is not specific to :class:`stnmf.STNMF` and
    can be used in conjunction with other :mod:`snmt.mf` classes.

    .. warning:: The reconstruction residual is not a suitable metric
                 reconstruction quality in the scope of
                 :class:`stnmf.STNMF`, due to the sparsity involved!

    Examples
    --------
    >>> from stnmf import STNMF
    >>> from stnmf.callbacks import stoppping_criterion
    >>> stnmf = STNMF(ste, callback=stoppping_criterion,
    ...               callback_kwargs=dict(interval=100))
    """
    if i == 0:
        # Store the initial reconstruction residual for comparison
        callback_data['err_0'] = self.res
        callback_data['err_prev'] = np.inf

    elif i == 1:
        # Define threshold of improvement based on first iteration
        callback_data['thr'] = (callback_data['err_0'] - self.res) * delta

    elif i % interval == 0:
        # Termination criterion
        if callback_data['err_prev'] - self.res < callback_data['thr']:
            itor.close()
            return False

        # Otherwise continue and update the previous error
        callback_data['err_prev'] = self.res
