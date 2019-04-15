#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals


class ExperimentResults(object):

    """Results of an experiment run.

    Attributes:
        param_grid (dict):
        results_grid (dict):

    """

    pass

class Experiment(object):

    """Experiment class, which handles iteration over parameter grids,
    uniform organization of results, overview via csv. Additionally it provides
    an interface for plotting the corresponding results with specified plot
    functions.

    Attributes:
        experiment_function (callable): The function which starts a single run
            of the experiment with fixed parameters.
        n_repetitions (int): Number of times every parameter setting is run.
        param_grid (dict): A dictionary, mapping each parameter to a list of
            values. Every combination of these values defines a parameter setting.


    """

    def __init__(self):
        pass

    def run_experiment(self, **params):
        """Run the experiment with the specified params, n_repetitions times."""
        raise NotImplementedError

    def run(self, param_grid):
        """Run experiments for all parameter settings specified in ´param_grid´."""
        pass