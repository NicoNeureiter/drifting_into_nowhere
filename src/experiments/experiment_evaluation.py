#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import sys
import collections
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed

from src.config import COLORS
from src.simulation import Simulation
from src.beast_interface import (run_beast, run_treeannotator)
from src.plotting import plot_mean_and_std
from src.util import (total_drift_2_step_drift, total_diffusion_2_step_var,
                      normalize, dist, mkpath, dump, load_from)

LOGGER_PATH = 'logs/experiment.log'
logger = logging.getLogger('experiment')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(LOGGER_PATH))
logger.addHandler(logging.StreamHandler(sys.stdout))


def run_experiment(n_runs, n_steps, n_expected_societies, total_drift,
                   total_diffusion, drift_density, p_settle, drift_direction,
                   chain_length, burnin, hpd_values, working_dir,
                   movement_model='rrw', root=(0, 0)):
    """Run an experiment ´n_runs´ times with the specified parameters.

    Args:
        n_runs (int): Number of times the experiment should be repeated.
        n_steps (int): Number of steps to simulate.
        n_expected_societies (int): Number data points to be expected in the end
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
        root (np.array): The root location, at which the simulation will start
            (can usually be left at [0,0]).

    Returns:
        dict: Statistics of the experiments (different error values).
    """
    # Ensure arrays to be np.array
    root = np.asarray(root)
    drift_direction = np.asarray(drift_direction)

    # Paths
    mkpath(working_dir)
    xml_path = working_dir + 'nowhere.xml'

    # Inferred parameters
    drift_direction = normalize(drift_direction)
    step_var = total_diffusion_2_step_var(total_diffusion, n_steps)
    _step_drift = total_drift_2_step_drift(total_drift, n_steps, drift_density=drift_density)
    step_mean = _step_drift * drift_direction
    p_split = n_expected_societies / n_steps

    # Check parameter validity
    if True:
        assert 0 < drift_density <= 1
        assert 0 <= p_split <= 1
        for hpd in hpd_values:
            assert 0 < hpd < 100
        assert burnin < chain_length

    l2_errors = []
    hpd_coverages = {p: [] for p in hpd_values}
    for i_run in range(n_runs):
        logger.info('\tRun %i...' % i_run)

        # Run Simulation
        simulation = Simulation(1, 0., step_mean, step_var, p_split,
                                p_settle=p_settle, drift_frequency=drift_density)
        simulation.run(n_steps)

        # Create an XML file as input for the BEAST analysis
        simulation.root.write_beast_xml(xml_path, chain_length,
                                        movement_model=movement_model)

        # Run BEAST
        run_beast(working_dir=working_dir)

        # Compute HPD coverages
        for hpd in hpd_values:
            tree = run_treeannotator(hpd, burnin, working_dir=working_dir)

            hit = tree.root_in_hpd(root, hpd)
            hpd_coverages[hpd].append(hit)
            logger.info('\t\tRoot in %i%% HPD: %s' % (hpd, hit))

        # Compute L2 Error
        l2_err = dist(root, tree.location)
        l2_errors.append(l2_err)
        logger.info('\t\tL2 Error: %.2f' % l2_err)

    # Combine stats in dict and return
    stats = {'l2_error': l2_errors,
             'hpd_coverage': hpd_coverages,
             'n_runs': n_runs}
    return stats


def run_experiments_varying_drift(default_settings, working_dir):
    logger.info('=' * 100)

    # Drift values to be evaluated
    total_drift_values = np.linspace(0., 2., 11)

    # def run(total_drift, i, kwargs):
    #     logger.info('Experiment with [total_drift = %.2f]' % total_drift)
    #     kwargs['total_drift'] = total_drift
    #     default_settings['working_dir'] = working_dir + '_%i' % i
    #     return total_drift, run_experiment(**kwargs)
    #
    # pool = Parallel(n_jobs=2, backend='multiprocessing')
    # results = pool(delayed(run)(drift, i, default_settings)
    #                for i, drift in enumerate(total_drift_values))
    # results = dict(results)

    results = {}
    for total_drift in total_drift_values:
        logger.info('Experiment with [total_drift = %.2f]' % total_drift)

        default_settings['total_drift'] = total_drift
        stats = run_experiment(**default_settings, working_dir=working_dir)
        results[total_drift] = stats

    # Dump the results in a pickle file
    dump(results, working_dir + 'results.pkl')
    return results


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
    plot_mean_and_std(x_values, l2_errors_mean, l2_errors_std, color=COLORS[0],
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
                 c=COLORS[i], label='%i%% HPD coverage' % p)
        plt.axhline(p/100, c=COLORS[i], ls='--', alpha=0.3)

    plt.xscale(xscale)
    plt.legend()
    plt.xlim(min(x_values), max(x_values))
    plt.ylim(-0.02, 1.03)
    plt.xlabel(x_name)
    plt.show()


if __name__ == '__main__':
    P_SETTLE = 'settling_probability'
    DENSITY = 'drift_density'
    DRIFT = 'total_drift'
    MODES = [P_SETTLE, DENSITY, DRIFT]
    mode = MODES[2]

    WORKING_DIR = 'data/experiments/{mode}/{time}/'
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    WORKING_DIR = WORKING_DIR.format(mode=mode, time=today)
    mkpath(WORKING_DIR)

    # Default experiment parameters
    default_settings = {
        # Simulation parameters
        'n_steps': 500,
        'n_expected_societies': 500,
        'total_diffusion': 1.,
        'total_drift': 1.,
        'drift_density': 1.,
        'drift_direction': list(normalize([1., .2])),
        'p_settle': 0.,

        # Analysis Parameters
        'chain_length': 500000,
        'burnin': 20000,

        # Experiment Settings
        'n_runs': 100,
        'hpd_values': [60, 80, 95]
    }

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    if 1:
        if mode == P_SETTLE:
            run_experiments_varying_p_settle(default_settings, working_dir=WORKING_DIR)
        elif mode == DENSITY:
            run_experiments_varying_drift_density(default_settings, working_dir=WORKING_DIR)
        elif mode == DRIFT:
            run_experiments_varying_drift(default_settings, working_dir=WORKING_DIR)

    # Plot the results
    # WORKING_DIR = WORKING_DIR.format(mode=mode, time='2018-11-28')
    results = load_from(WORKING_DIR + 'results.pkl')
    plot_experiment_results(results, x_name=mode, xscale='linear')
