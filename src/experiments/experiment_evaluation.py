#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

from src.config import COLORS
from src.simulation import Simulation
from src.beast_interface import (run_beast, run_treeannotator)
from src.util import (total_drift_2_step_drift, total_diffusion_2_step_var,
                      normalize, dist, mkpath, dump, load_from)

LOGGER_PATH = 'logs/experiment.log'
logger = logging.getLogger('experiment')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(LOGGER_PATH))
logger.addHandler(logging.StreamHandler(sys.stdout))


def run_experiment(n_runs, n_steps, n_expected_societies, total_drift,
                   total_diffusion, drift_density, drift_direction,
                   chain_length, burnin, hpd_values,
                   movement_model='rrw', working_dir='data/beast/', root=(0, 0)):
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
    mkpath('data/beast/')
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
        simulation = Simulation(1, 0., step_mean, step_var,
                                p_split, drift_frequency=drift_density)
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


def run_experiments_varying_drift():
    logger.info('=' * 100)
    logger.info('New Run')
    logger.info('=' * 100)

    working_dir = 'data/beast/'

    # Simulation Parameters
    n_steps = 200
    n_expected_societies = 30
    total_diffusion = 1.
    drift_density = 1.
    drift_direction = normalize([1., .2])

    # Analysis Parameters
    chain_length = 300000
    burnin = 10000

    # Experiment Settings
    n_runs = 20
    hpd_values = [60, 80, 95]
    total_drift_values = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.]
    # total_drift_values = [0.0, 0.5, 1.0, 1.5]

    # Results
    results = {}

    for total_drift in total_drift_values:
        logger.info('Experiment with [total_drift = %.2f]' % total_drift)

        stats = run_experiment(
            n_runs, n_steps, n_expected_societies, total_drift, total_diffusion,
            drift_density, drift_direction, chain_length, burnin, hpd_values,
            working_dir=working_dir)

        results[total_drift] = stats

    # Dump the results in a pickle file
    dump(results, working_dir + 'results.pkl')
    return results


def plot_experiment_results(results):
    # TODO replace fixed 60, 80, 95 coverage lists by dict

    total_drift_values = list(results.keys())

    x = []
    l2_errors_all = []
    l2_errors_mean = []
    l2_errors_std = []
    coverage_60 = []
    coverage_80 = []
    coverage_95 = []

    for total_drift, stats in results.items():
        l2_errors = stats['l2_error']
        l2_errors_mean.append(np.mean(l2_errors))
        l2_errors_std.append(np.std(l2_errors))
        l2_errors_all += l2_errors
        x += [total_drift] * len(l2_errors)

        hpd_coverages = stats['hpd_coverage']
        coverage_60.append(np.mean(hpd_coverages[60]))
        coverage_80.append(np.mean(hpd_coverages[80]))
        coverage_95.append(np.mean(hpd_coverages[95]))

    # Transform to numpy
    l2_errors_mean = np.asarray(l2_errors_mean)
    l2_errors_std = np.asarray(l2_errors_std)

    # Plot the results in scatter plots
    plt.scatter(x, l2_errors_all, c='lightgrey')
    plt.plot(total_drift_values, l2_errors_mean, c=COLORS[0],
             label=r'L2-Error of reconstructed root')
    plt.plot(total_drift_values, l2_errors_mean + l2_errors_std, c=COLORS[0], ls='--')
    plt.plot(total_drift_values, l2_errors_mean - l2_errors_std, c=COLORS[0], ls='--')
    plt.fill_between(total_drift_values,
                     l2_errors_mean - l2_errors_std,
                     l2_errors_mean + l2_errors_std,
                     color=COLORS[0], alpha=0.05, zorder=0)

    plt.legend()
    plt.xlim(total_drift_values[0], total_drift_values[-1])
    plt.show()

    plt.plot(total_drift_values, coverage_60, c=COLORS[0],
             label=r'60% HPD coverage')
    plt.axhline(0.6, c=COLORS[0], ls='--', alpha=0.3)
    plt.plot(total_drift_values, coverage_80, c=COLORS[1],
             label=r'80% HPD coverage')
    plt.axhline(0.8, c=COLORS[1], ls='--', alpha=0.3)
    plt.plot(total_drift_values, coverage_95, c=COLORS[2],
             label=r'90% HPD coverage')
    plt.axhline(0.95, c=COLORS[2], ls='--', alpha=0.3)

    plt.legend()
    plt.xlim(total_drift_values[0], total_drift_values[-1])
    plt.show()


if __name__ == '__main__':
    WORKING_DIR = 'data/beast/'
    MODE = 'load'

    if MODE == 'rerun':
        results = run_experiments_varying_drift()
    else:
        results = load_from(WORKING_DIR + 'results.pkl')

    plot_experiment_results(results)