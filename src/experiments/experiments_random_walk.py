#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import os
import sys
import collections
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from joblib import Parallel, delayed

from src.config import _COLORS
from src.experiments.experiment import Experiment
from src.simulation.simulation import run_simulation
from src.simulation.vector_simulation import VectorState, VectorWorld
from src.beast_interface import (run_beast, run_treeannotator, load_trees)
from src.evaluation import (eval_bias, eval_rmse, eval_stdev, eval_mean_offset)
from src.plotting import plot_mean_and_std
from src.util import (total_drift_2_step_drift, total_diffusion_2_step_var,
                      normalize, mkpath, dump, load_from)


LOGGER = logging.getLogger('experiment')


def run_experiment(n_steps, n_expected_leafs, total_drift,
                   total_diffusion, drift_density, p_settle, drift_direction,
                   chain_length, burnin, hpd_values, working_dir,
                   turnover=0.2, clock_rate=1.0, movement_model='rrw',
                   max_fossil_age=0, **kwargs):
    """Run an experiment ´n_runs´ times with the specified parameters.

    Args:
        n_runs (int): Number of times the experiment should be repeated.
        n_steps (int): Number of steps to simulate.
        n_expected_leafs (int): Number data points to be expected in the end
            (only expected, not exact value, due to stochasticity)
        total_drift (float): The total distance that every society will travel
            due to drift over the simulated time.
        total_diffusion (float): The expected total distance that every society
            will move away from the root, due to diffusion.
        drift_density (float): Frequency of drift occurring (does not effect
            the total drift).
        p_settle (float): Probability of stopping drift and 'settling' at the
            current location (only diffusion from this point).
        drift_direction (np.array): The direction of drift.
        chain_length (int): MCMC chain length in BEAST analysis.
        burnin (int): MCMC burnin steps in BEAST analysis.
        hpd_values (list): The values for the HPD coverage statistics.

    Keyword Args:
        movement_model (str): The movement to be used in BEAST analysis
            ('rrw' or 'brownian').
        working_dir (str): The working directory in which intermediate files
            will be dumped.
        drop_fossils (bool): Remove extinct taxa from the sampled phylogeny.

    Returns:
        dict: Statistics of the experiments (different error values).
    """
    # Ensure arrays to be np.array
    root = np.zeros(2)
    drift_direction = np.asarray(drift_direction)
    min_leaves, max_leaves = 0.4 * n_expected_leafs, 2. * n_expected_leafs

    # Paths
    xml_path = working_dir + 'nowhere.xml'

    # Inferred parameters
    drift_direction = normalize(drift_direction)
    step_var = total_diffusion_2_step_var(total_diffusion, n_steps)
    _step_drift = total_drift_2_step_drift(total_drift, n_steps, drift_density=drift_density)
    step_mean = _step_drift * drift_direction

    # Compute birth-/death-rate from n_expected_leaves, n_steps and turnover
    eff_div_rate = np.log(n_expected_leafs) / n_steps
    birth_rate = eff_div_rate / (1 - turnover)
    death_rate = birth_rate * turnover

    # Check parameter validity
    if True:
        assert 0 < drift_density <= 1
        assert 0 <= turnover < 1
        assert 0 <= death_rate < birth_rate <= 1
        for hpd in hpd_values:
            assert 0 < hpd < 100
        assert burnin < chain_length

    valid_tree = False
    while not valid_tree:
        # Run Simulation
        p0 = np.zeros(2)
        world = VectorWorld()
        tree_simu = VectorState(world, p0, step_mean, step_var, clock_rate, birth_rate,
                                drift_frequency=drift_density, death_rate=death_rate)
        run_simulation(n_steps, tree_simu, world)

        # Check whether tree satisfies criteria...
        #    Criteria: not too small/big & root has two extant subtrees
        n_leafs = len([n for n in tree_simu.iter_leafs() if n.height == n_steps])
        valid_tree = (min_leaves < n_leafs < max_leaves)
        for c in tree_simu.children:
            if not any(n.height == n_steps for n in c.iter_leafs()):
                valid_tree = False

    tree_simu.drop_fossils(max_fossil_age)

    # Create an XML file as input for the BEAST analysis
    tree_simu.write_beast_xml(xml_path, chain_length, movement_model=movement_model,
                              drift_prior_std=1.)

    # Run phylogeographic reconstruction in BEAST
    run_beast(working_dir=working_dir)

    results = evaluate(working_dir, burnin, hpd_values, root)

    # Add statistics about simulated tree (to compare between simulation modes)
    results['observed_stdev'] = np.hypot(*np.std(tree_simu.get_leaf_locations(), axis=0))
    leafs_mean = np.mean(tree_simu.get_leaf_locations(), axis=0)
    leafs_mean_offset = leafs_mean - root
    results['observed_drift_x'] = leafs_mean_offset[0]
    results['observed_drift_y'] = leafs_mean_offset[1]
    results['observed_drift_norm'] = np.hypot(*leafs_mean_offset)

    return results


def evaluate(working_dir, burnin, hpd_values, true_root):
    results = {}
    for hpd in hpd_values:
        # Summarize tree using tree-annotator
        tree = run_treeannotator(hpd, burnin, working_dir=working_dir)

        # Compute HPD coverage
        hit = tree.root_in_hpd(true_root, hpd)
        results['hpd_%i' % hpd] = hit
        LOGGER.info('\t\tRoot in %i%% HPD: %s' % (hpd, hit))

    # Load posterior trees for other metrics
    trees = load_trees(working_dir + 'nowhere.trees')

    # Compute and log RMSE
    rmse = eval_rmse(true_root, trees)
    results['rmse'] = rmse
    LOGGER.info('\t\tRMSE: %.2f' % rmse)

    # Compute and log mean offset
    offset = eval_mean_offset(true_root, trees)
    results['bias_x'] = offset[0]
    results['bias_y'] = offset[1]
    LOGGER.info('\t\tMean offset: (%.2f, %.2f)' % tuple(offset))

    # Compute and log bias
    bias = eval_bias(true_root, trees)
    results['bias_norm'] = bias
    LOGGER.info('\t\tBias: %.2f' % bias)

    # Compute and log standard deviation
    stdev = eval_stdev(true_root, trees)
    results['stdev'] = stdev
    LOGGER.info('\t\tStdev: %.2f' % stdev)

    return results


if __name__ == '__main__':
    # MODES
    RANDOM_WALK = 'random_walk'
    CONSTRAINED_EXPANSION = 'constrained_expansion'
    MODES = [RANDOM_WALK, CONSTRAINED_EXPANSION]
    mode = MODES[0]
    N_REPEAT = 20
    HPD_VALUES = [95]

    # MOVEMENT MODEL
    if len(sys.argv) > 1:
        MOVEMENT_MODEL = sys.argv[1]
    else:
        MOVEMENT_MODEL = 'brownian'
    if len(sys.argv) > 2:
        MAX_FOSSIL_AGE = float(sys.argv[2])
    else:
        MAX_FOSSIL_AGE = 0

    # Set working directory
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    WORKING_DIR = 'experiments/{mode}/{mm}_fossils={max_age}/'
    WORKING_DIR = WORKING_DIR.format(mode=mode, mm=MOVEMENT_MODEL, max_age=MAX_FOSSIL_AGE)
    mkpath(WORKING_DIR)

    # Set cwd for logger
    LOGGER_PATH = os.path.join(WORKING_DIR, 'experiment.log')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.addHandler(logging.FileHandler(LOGGER_PATH))
    LOGGER.info('=' * 100)

    # Default experiment parameters
    if mode == RANDOM_WALK:
        simulation_settings = {
            'n_steps': 5000,
            'n_expected_leafs': 100,
            'total_diffusion': 2000.,
            'total_drift': 0.,
            'drift_density': 1.,
            'drift_direction': [0., 1.],
            'p_settle': 0.,
            'max_fossil_age': MAX_FOSSIL_AGE,
        }
    else:
        raise NotImplementedError

    default_settings = {
        # Analysis Parameters
        'movement_model': MOVEMENT_MODEL,
        'chain_length': 350000,
        'burnin': 50000,
        # Experiment Settings
        'hpd_values': HPD_VALUES
    }
    default_settings.update(simulation_settings)

    EVAL_METRICS = ['rmse', 'bias_x', 'bias_y', 'bias_norm', 'stdev'] + \
                   ['hpd_%i' % p for p in HPD_VALUES] + \
                   ['observed_stdev', 'observed_drift_x',  'observed_drift_y', 'observed_drift_norm']

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    if 1:
        if mode == RANDOM_WALK:
            total_drift_values = np.linspace(0., 3., 13) * default_settings['total_diffusion']
            # total_drift_values = np.linspace(0., 3., 4) * default_settings['total_diffusion']
            variable_parameters = {'total_drift': total_drift_values}
        else:
            raise NotImplementedError

    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, N_REPEAT, WORKING_DIR)
    experiment.run(resume=0)

    # # Plot the results
    # # WORKING_DIR = WORKING_DIR.format(mode=mode, time='2018-11-28')
    # results = load_from(WORKING_DIR + 'results.pkl')
    # plot_experiment_results(results, x_name=mode, xscale='linear')
