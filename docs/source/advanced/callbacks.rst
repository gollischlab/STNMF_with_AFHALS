Callbacks
=========

Callbacks allow to safely modify the algorithm of subunit decomposition in many aspects without having to dive deep into the NMF code. The user can specify a function that will be called at each loop iteration in the iterative process of NMF. Code execution jumps out of the internal code into the user function and returns afterwards to continue until the next iteration. The function defined by the user will be supplied with and has access to the current (mutable) state of the decomposition at each iteration.

Among many other things, this allows...

* to record the course of certain properties throughout the iterative process,

* to have the process be interrupted prematurely based on termination criteria,

* to change properties of the current decomposition to modify the course of the factorization,

* and to leverage parts of the algorithm to completely redefine how subunits are decomposed.

A callback function can be understood as a plug-in or add-on extension to the internal code.

**Contents**

.. contents:: :local:

Creating a callback function
----------------------------

To illustrate how to create a custom callback, we will be implementing the function :func:`stnmf.callbacks.residual` step-by-step.

A callback is a function with the following signature.

.. code-block:: python

    function (self, i, itor, callback_data, **kwargs) -> bool

That means the function takes a set of keyword arguments as parameters and returns a boolean.
The return value tells the matrix factorization whether to continue to the next iteration (`True`) or to terminate (`False`).
It can be omitted (the function returns nothing), to signal the continuation as normal.
The arguments for the function are supplied by the matrix factorization and consist of the following

**self** : :class:`stnmf.mf.MF`
    Current MF object. Attributes are mutable

**i** : :class:`int`
    Index of the current (completed) iteration. Initialization is zero

**itor** : :class:`tqdm.tqdm`
    Iterator used during the factorization. Attributes are mutable

**callback_data** : :class:`dict`
    Dictionary to store and preserve callback data. Mutable

The function may take additional parameters that will be populated with the contents of the dictionary `callback_kwargs`.
For now, we will safely ignore additional parameters.

To start off, we create Python file 'my_callback.py' and create the empty function.

.. code-block:: python
    :caption: my_callback.py
    :linenos:

    def callback(self, i, itor, callback_data):
        """This is my first callback"""
        pass

The command `pass` serves as a placeholder here.

To use the callback, we simply import that Python file to pass the callback function to STNMF.

.. code-block:: python

    >>> from stnmf import STNMF
    >>> import my_callback

We will re-use the STE from the :doc:`../guides/examples`.

.. code-block:: python

    >>> import numpy as np
    >>> ds = np.DataSource(None)
    >>> ste = np.load(ds.open('https://github.com/gollischlab/STNMF_with_AFHALS/files/13480212/ste.zip', 'rb'))['ste']

The callback function in ``my_callback.callback`` is passed to the :class:`STNMF <stnmf.STNMF>` constructor.

.. code-block:: python
    :force:

    >>> stnmf = STNMF(ste, sparsity=1.74, callback=my_callback.callback)
    Sparse semi-NMF: 100%|████████████████████| 1000/1000 [00:05<00:00, 169.76it/s]

Because the callback function is empty, there is no noticeable difference in the output and the resulting decomposition.

Accessing the decomposition
---------------------------

From within the callback function, we can access the current state of the decomposition - for example, to display useful information.

To mimic the callback :func:`stnmf.callbacks.residual`, we will need to access the reconstruction error, or residual, of the decomposition.
It is stored in the attribute :attr:`STNMF.res <stnmf.STNMF.res>` of the object `self`.

.. code-block:: python
    :caption: my_callback.py
    :linenos:
    :emphasize-lines: 3

    def callback(self, i, itor, callback_data):
        """This is my first callback"""
        print(self.res)

Callback data
-------------

Printing the residual like so might not be a great idea, because the matrix factorization runs usually for around 1000 iterations.
Such a print would both clutter the output and also slow down the factorization  considerably.
It will be updated so often and fast that it will be of no use for the user.

Instead, we will store the residual on every function call into an array.
A callback is called the first time right after the initialization of the modules - before the first NMF iteration.
We will use this 0-th iteration to create an array to store all values into.
The argument `i` specifies the current iteration and the :class:`STNMF <stnmf.STNMF>` object `self` tells us the maximum number of iterations in the attribute `num_iter`.

.. code-block:: python
    :caption: my_callback.py
    :linenos:
    :emphasize-lines: 4-9

    def callback(self, i, itor, callback_data):
        """This is my first callback"""

        if i == 0:
            # Allocate the array to store the values
            array = np.zeros(self.num_iter + 1, self.dtype)

        # Add the current residual
        array[i] = self.res

Here, `array` is a local variable such that it will not outlive each function call.
Any iteration after the 0-th one will raise an error, because the variable does not exist.

Instead the array should be created in a persistent way.
For that, the dictionary `callback_data` is useful.
It allows to both keep track of information in between iterations and will also serve as interface to provide an output in the end.

.. code-block:: python
    :caption: my_callback.py
    :linenos:
    :emphasize-lines: 6,9

    def callback(self, i, itor, callback_data):
        """This is my first callback"""

        if i == 0:
            # Allocate the array to store the values
            callback_data['res'] = np.zeros(self.num_iter + 1, self.dtype)

        # Add the current residual
        callback_data['res'][i] = self.res

Callback results
----------------

Using this callback as described above will store a series of residuals in `callback_data`.
However, the dictionary `callback_data` only outlives the factorization if it had been defined prior to and outside of the factorization.
We will create a dictionary called 'results' to be able to access the residuals afterwards.

.. code-block:: python
    :force:

    >>> results = dict()
    >>> stnmf = STNMF(ste, sparsity=1.74, callback=my_callback.callback,
    ...               callback_data=results)
    Sparse semi-NMF: 100%|████████████████████| 1000/1000 [00:28<00:00, 34.86it/s]

    >>> results['res']
    array([2308.3042, 2289.2097, 2288.6548, ..., 2286.4116, 2286.4114,
       2286.4116], dtype=float32)

Callback arguments
------------------

Computing and recording the residual at every iteration may slow down the factorization and doing so only at certain intervals might improve performance.
Nevertheless, allowing to specify the desired interval will offer more flexibility.

To do so, we add a custom parameter to our callback function.
We will call this function parameter `interval`, but its name can be freely chosen.
To only record the residual in the specified interval, we wrap the assignment into an if-block and use modulo to filter the iterations.
Do not forget to adjust the total length of the array as well.

.. code-block:: python
    :caption: my_callback.py
    :linenos:
    :emphasize-lines: 1,6,8,10

    def callback(self, i, itor, callback_data, interval=100):
        """This is my first callback"""

        if i == 0:
            # Allocate the array to store the values
            callback_data['res'] = np.zeros(self.num_iter//interval+1, self.dtype)

        if i % interval == 0:
            # Add the current residual
            callback_data['res'][i // interval] = self.res

This is it.

We add any custom arguments to the callback using the dictionary `callback_kwargs`.

.. code-block:: python

    >>> stnmf = STNMF(ste, sparsity=1.74, callback=my_callback.callback,
    ...               callback_data=results, callback_kwargs=dict(interval=50))
    Sparse semi-NMF: 100%|████████████████████| 1000/1000 [00:06<00:00, 154.59it/s]

Update progress
---------------

To update the progress bar or display information during the factorization, please have a look into the full source code of the :func:`stnmf.callbacks.residual` callback function.
