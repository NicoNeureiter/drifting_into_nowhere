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
import itertools
import shelve

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ParameterGrid
# from joblib import Parallel, delayed

from src.config import _COLORS
from src.simulation.simulation import run_simulation
from src.simulation.vector_simulation import VectorState, VectorWorld
from src.simulation.grid_simulation import GridState
from src.beast_interface import (run_beast, run_treeannotator, load_trees)
from src.evaluation import (eval_bias, eval_rmse, eval_stdev, eval_hpd_hit)
from src.plotting import plot_mean_and_std
from src.util import (total_drift_2_step_drift, total_diffusion_2_step_var,
                      normalize, mkpath, dump, load_from,
                      experiment_preperations)

logger = logging.getLogger('experiment')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# def iter_param_grid(param_dict):
#     param_names = param_dict.keys()
#     for values in itertools.product(*param_dict.values()):
#         yield dict(zip(param_names, values))

CHECKLIST_FILE_NAME = 'checklist.txt'

class Experiment(object):

    def __init__(self, fixed_params, variable_param_ranges, pipeline, working_directory):
        self.fixed_params = fixed_params
        self.variable_param_ranges = variable_param_ranges
        self.pipeline = pipeline
        self.working_directory = working_directory
        mkpath(working_directory)

    def run(self, n_repeat, resume=False):
        # Copy parameter grid and add the run index
        var_param_names = list(self.variable_param_ranges.keys())
        grid = dict(self.variable_param_ranges)
        # grid['i_repeat'] = range(n_repeat)

        checklist_path = os.path.join(self.working_directory, CHECKLIST_FILE_NAME)
        if resume:
            with open(checklist_path, 'r') as checklist_file:
                checklist = checklist_file.readlines()
        else:
            checklist = []

        results_by_params = pd.DataFrame(columns=var_param_names + ['results'])

        # Iterate over the grid
        for var_params in ParameterGrid(grid):
            run_id = format_params(var_params)
            results_path = os.path.join(self.working_directory,
                                        'results_%s.pkl' % run_id)
            row = dict(var_params)

            if run_id in checklist:
                results = load_from(results_path)
                row['results'] = results
                results_by_params.append(row, ignore_index=True)
                continue

            params = dict(var_params, working_dir=self.working_directory, **self.fixed_params)

            # experiment_directory = os.path.join(self.working_directory, run_id)
            # os.mkdir(experiment_directory)

            results = self.pipeline(**params)
            # outputs = {}
            # for operator in self.pipeline:
            #     inputs = dict(params)
            #     inputs.update(outputs)
            #     outputs = operator(**inputs)
            row['results'] = results
            results_by_params.append(row, ignore_index=True)
            print(results_path)
            dump(results, results_path)

            with open(checklist_path, 'a') as checklist_file:
                checklist_file.write(run_id)


def format_params(params):
    return ','.join(['%s=%s' % (k,v) for k, v in params.items()])


def run_experiment(n_runs, n_steps, n_expected_leafs, total_drift,
                   total_diffusion, drift_density, p_settle, drift_direction,
                   chain_length, burnin, hpd_values, working_dir,
                   diversification_mode='birth-death', turnover=0.2,
                   clock_rate=1.0, movement_model='rrw',
                   max_fossil_age=0):
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
    experiment_preperations(working_dir)
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

    metrics = ['rmse', 'bias', 'stdev'] + ['hpd_%.2f' % p for p in hpd_values]
    results = pd.DataFrame(columns=['i_run', 'rmse'] + metrics)
    for i_run in range(n_runs):
        logger.info('\tRun %i...' % i_run)
        run_results = pd.Series(index=results.columns)
        run_results['i_run'] = i_run

        valid_tree = False
        while not valid_tree:
            # Run Simulation
            p0 = np.zeros(2)
            world = VectorWorld()
            tree_simu = VectorState(world, p0, step_mean, step_var, clock_rate, birth_rate,
                                    drift_frequency=drift_density, death_rate=death_rate)
            run_simulation(n_steps, tree_simu, world)

            # Check whether tree satisfies criteria
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

        for hpd in hpd_values:
            # Summarize tree using tree-annotator
            tree = run_treeannotator(hpd, burnin, working_dir=working_dir)

            # Compute HPD coverage
            hit = tree.root_in_hpd(root, hpd)
            run_results['hpd_%.2f' % hpd] = hit
            logger.info('\t\tRoot in %i%% HPD: %s' % (hpd, hit))


        # Load posterior trees for other metrics
        trees = load_trees(working_dir + 'nowhere.trees')

        # Compute and log RMSE
        rmse = eval_rmse(root, trees)
        run_results['rmse'] = rmse
        logger.info('\t\tRMSE: %.2f' % rmse)

        # Compute and log bias
        bias = eval_bias(root, trees)
        run_results['bias'] = bias
        logger.info('\t\tBias: %.2f' % bias)

        # Compute and log standard deviation
        stdev = eval_stdev(root, trees)
        run_results['stdev'] = stdev
        logger.info('\t\tStdev: %.2f' % stdev)

        results.append(run_results, ignore_index=True)

    return results


def run_experiments_varying_drift(default_settings, working_dir):
    logger.info('=' * 100)

    # Drift values to be evaluated
    s = default_settings['total_diffusion']
    total_drift_values = s * np.linspace(0., 3., 13)

    experiment = Experiment(default_settings, {'total_drift': total_drift_values},
                            run_experiment, WORKING_DIR)
    experiment.run(default_settings['n_runs'])

    # # def run(total_drift, i, kwargs):
    # #     logger.info('Experiment with [total_drift = %.2f]' % total_drift)
    # #     kwargs['total_drift'] = total_drift
    # #     default_settings['working_dir'] = working_dir + '_%i' % i
    # #     return total_drift, run_experiment(**kwargs)
    # #
    # # pool = Parallel(n_jobs=2, backend='multiprocessing')
    # # results = pool(delayed(run)(drift, i, default_settings)
    # #                for i, drift in enumerate(total_drift_values))
    # # results = dict(results)
    #
    # results_by_param = {}
    # for total_drift in total_drift_values:
    #     logger.info('Experiment with [total_drift = %.2f]' % total_drift)
    #
    #     default_settings['total_drift'] = total_drift
    #     stats = run_experiment(**default_settings, working_dir=working_dir)
    #     results_by_param[total_drift] = stats
    #
    # # Dump the results in a pickle file
    # dump(results_by_param, working_dir + 'results.pkl')
    # return results_by_param


def run_experiments_varying_p_settle(default_settings, working_dir):
    logger.info('=' * 100)

    # Settling probability values to be evaluated
    p_settle_values = np.linspace(0., 1., 11)

    results = {}
    for p_settle in p_settle_values:
        logger.info('Experiment with [p_settle = %.2f]' % p_settle)

        default_settings['p_settle'] = p_settle
        stats = run_experiment(**default_settings, working_dir=working_dir)
        results[p_settle] = stats

    # Dump the results in a pickle file
    dump(results, working_dir + 'results.pkl')
    return results


def run_experiments_varying_drift_density(default_settings, working_dir):
    logger.info('=' * 100)

    # Settling probability values to be evaluated
    # drift_density_values = np.linspace(0.5, 1., 20)
    drift_density_values = 2**(-np.linspace(0., 10., 11))

    results = {}
    for drift_density in drift_density_values:
        logger.info('Experiment with [drift_density = %.2f]' % drift_density)

        default_settings['drift_density'] = drift_density
        stats = run_experiment(**default_settings, working_dir=working_dir)
        results[drift_density] = stats

    # Dump the results in a pickle file
    dump(results, working_dir + 'results.pkl')
    return results


def plot_experiment_results(results, x_name='Total Drift', xscale='linear'):
    x_values = list(results.keys())

    l2_error_scatter = []
    l2_errors_mean = []
    l2_errors_std = []
    coverage_stats = collections.defaultdict(list)

    for x, stats in results.items():
        l2_errors = stats['l2_error']

        # Compute L2-error statistics
        l2_errors_mean.append(np.mean(l2_errors))
        l2_errors_std.append(np.std(l2_errors))

        # Store L2-error values for scatter plot
        for y in l2_errors:
            l2_error_scatter.append([x, y])

        # Compute coverage statistics
        hpd_coverages = stats['hpd_coverage']
        for p, hits in hpd_coverages.items():
            coverage_stats[p].append(np.mean(hits))

    # Transform to numpy
    l2_error_scatter = np.asarray(l2_error_scatter)
    l2_errors_mean = np.asarray(l2_errors_mean)
    l2_errors_std = np.asarray(l2_errors_std)

    # Plot the mean L2-Errors +/- standard deviation
    plt.scatter(*l2_error_scatter.T, c='lightgrey')
    plt.plot(x_values, x_values, c='k', lw=0.4)
    plot_mean_and_std(x_values, l2_errors_mean, l2_errors_std, color=_COLORS[0],
                      label=r'Average L2-error of reconstructed root')

    plt.xscale(xscale)
    plt.legend()
    plt.xlim(min(x_values), max(x_values))
    plt.ylim(0, None)
    plt.xlabel(x_name)
    plt.show()

    # Plot a curve and
    for i, (p, coverage) in enumerate(coverage_stats.items()):
        plt.plot(x_values, coverage,
                 c=_COLORS[i], label='%i%% HPD coverage' % p)
        plt.axhline(p / 100, c=_COLORS[i], ls='--', alpha=0.3)

    plt.xscale(xscale)
    plt.legend()
    plt.xlim(min(x_values), max(x_values))
    plt.ylim(-0.02, 1.03)
    plt.xlabel(x_name)
    plt.show()


if __name__ == '__main__':
    # MODES
    RANDOM_WALK = 'random_walk'
    CONSTRAINED_EXPANSION = 'constrained_expansion'
    MODES = [RANDOM_WALK, CONSTRAINED_EXPANSION]
    mode = MODES[0]

    # MOVEMENT MODEL
    MOVEMENT_MODEL = 'cdrw'
    MAX_FOSSIL_AGE = 0

    # Set working directory
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    WORKING_DIR = 'experiments/{mode}/{mm}_fossils={max_age}/'
    WORKING_DIR = WORKING_DIR.format(mode=mode, mm=MOVEMENT_MODEL, max_age=MAX_FOSSIL_AGE)
    mkpath(WORKING_DIR)

    # Set cwd for logger
    LOGGER_PATH = os.path.join(WORKING_DIR, 'experiment.log')
    logger.addHandler(logging.FileHandler(LOGGER_PATH))

    # Default experiment parameters
    if mode == RANDOM_WALK:
        simulation_settings = {
            'n_steps': 5000,
            'n_expected_leafs': 100,
            'total_diffusion': 2000.,
            'total_drift': 0.,
            'drift_density': 1.,
            'drift_direction': [0., .1],
            'p_settle': 0.,
            'movement_model': MOVEMENT_MODEL,
            'max_fossil_age': MAX_FOSSIL_AGE,
        }
    else:
        simulation_settings = {
            'n_steps': 5000,
            'grid_size': 100,

        }

    default_settings = {
        # Analysis Parameters
        'chain_length': 500000,
        'burnin': 100000,
        # Experiment Settings
        'n_runs': 30,
        'hpd_values': [80, 95]
    }
    default_settings.update(simulation_settings)

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    if 1:
        if mode == RANDOM_WALK:
            run_experiments_varying_drift(default_settings, working_dir=WORKING_DIR)
        elif mode == CONSTRAINED_EXPANSION:
            pass
        else:
            raise NotImplementedError

    # Plot the results
    # WORKING_DIR = WORKING_DIR.format(mode=mode, time='2018-11-28')
    results = load_from(WORKING_DIR + 'results.pkl')
    plot_experiment_results(results, x_name=mode, xscale='linear')
