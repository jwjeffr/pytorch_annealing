.. role:: raw-math(raw)
    :format: latex html

.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

.. _simulated annealing: https://en.wikipedia.org/wiki/Simulated_annealing

PyTorch QUBO Simulated Annealing
--------------------------------

About
#####

This repository contains a small library ``anneal.py`` which performs `simulated annealing`_ using PyTorch for the quadratic unconstrained binary optimization (`QUBO`_) problem [1]_:

:raw-math:`$$\mathbf{x}^* = \underset{\mathbf{x}\in \{0, 1\}^n}{\arg\min}\;\mathbf{x}^\intercal \mathcal{Q}\mathbf{x}$$`

``anneal.py`` includes the ``metropolis`` function for performing an annealing run and the ``AnnealingRun`` object containing information about the run. ``anneal_lib_example.py`` is an example script using this library, showing how to call ``metropolis`` and plot its results, as well as write its results to a file mid-run.

.. [1] Note that the QUBO problem is strictly defined for an upper-triangular matrix :raw-math:`$\mathcal{Q}$`. This library, however, does not check for this condition.