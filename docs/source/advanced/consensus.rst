Hyperparameter selection
========================

An essential part of STNMF is to chose a suitable parameter for sparsity regularization, because sparsity is an integral part of semi-NMF to produce a parts-based decomposition. [1]_
The sparsity is applied to the module matrix of STNMF, such that low-magnitude pixel values are penalized and set to zero.
Along with the non-negativity constraint, this encourages sparse representations.

Careful selection of the sparsity weight results in localized subunits, and depends on the dimensionality of the data and its contamination with noise.
Because of similarities in their response properties, hyperparameter selection of sparsity may be similar for cells of the same functional type. [2]_

Consensus analyses
------------------

Consensus analyses are employed to aid in finding suitable sparsity weights.
It is integrated into STNMF using :doc:`callbacks`.

We start with the effective spike-triggered stimulus ensemble (STE) stored in the variable `ste` from the :doc:`../guides/examples`.

.. code-block:: python

    >>> import numpy as np
    >>> ds = np.DataSource(None)
    >>> ste = np.load(ds.open('https://github.com/gollischlab/STNMF_with_AFHALS/files/13480212/ste.zip', 'rb'))['ste']

We will use the STNMF algorithm and the consensus callback.

.. code-block:: python

    >>> from stnmf import STNMF
    >>> from stnmf.callbacks import consensus

The callback will execute the STNMF algorithm for different sparsity parameter values and return the results as cophenetic correlation coefficients in a Python dictionary.
To access the results afterwards, this dictionary needs to be created beforehand.

.. code-block:: python

    >>> results = dict()

Then, the normal use of STNMF is extended with supplying both the callback function and the results directory.
Furthermore, we specify the desired sparsity values to probe in the `callback_kwargs`.
Note that we do not need to store the created :class:`STNMF <stnmf.STNMF>` object, as we are only interested in the consensus method results.

.. code-block:: python
    :force:

    >>> STNMF(ste, callback=consensus, callback_data=results,
    ...       callback_kwargs=dict(sparsities=[0, 0.5, 1, 1.5, 2]))
    Consensus analysis: 100%|████████████████████| 5/5 [17:36<00:00, 207.65s/parameter, sparsity=2]
            Repetition: 100%|████████████████████| 30/30 [02:42<00:00, 5.57s/rep, seed=29]
       Sparse semi-NMF: 100%|████████████████████| 1000/1000 [00:05<00:00, 172.94it/s]

.. warning::
    Leaving the other callback arguments at their defaults will run the STNMF algorithm 30 times for each sparsity value, that is here a total of 30 x 5 sparsity values = 150 runs of STNMF.

More parameters are described in :func:`stnmf.callbacks.consensus`.

The stability of the decomposition for each sparsity is quantified as the cophenetic correlation coefficient and is found in the directory `results`.

.. code-block:: python

    >>> results['cpcc']
    array([0.22871876, 0.6855058 , 0.9002583 , 0.96028996, 0.95667326],
      dtype=float32)

More information on the hyperparameter selection using consensus analyses is described in detail elsewhere. [2]_

.. [1] Ding, C. H. Q., Tao Li, & Jordan, M. I. (2010). Convex and Semi-Nonnegative Matrix Factorizations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(1), 45–55

.. [2] |citation-apa|
