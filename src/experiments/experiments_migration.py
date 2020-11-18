#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import os
import sys
import json
import datetime

import numpy as np

from src.experiments.experiment import Experiment
from src.simulation.simulation import run_simulation
from src.simulation.migration_simulation import VectorState, VectorWorld
from src.beast_interface import (run_beast)
from src.evaluation import (evaluate, tree_statistics)
from src.util import (total_drift_2_step_drift, total_diffusion_2_step_var,
                      normalize, mkpath, parse_arg)


def run_experiment(n_steps, n_expected_leafs, total_drift,
                   total_diffusion, drift_density, p_settle, drift_direction,
                   chain_length, burnin, hpd_values, working_dir,
                   turnover=0.2, clock_rate=1.0, movement_model='rrw',
                   max_fossil_age=0, min_n_fossils=10, **kwargs):
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

    Kwargs:
        movement_model (str): The movement to be used in BEAST analysis
            ('rrw' or 'brownian').
        working_dir (str): The working directory in which intermediate files
            will be dumped.
        drop_fossils (bool): Remove extinct taxa from the sampled phylogeny.
        max_fossil_age (float): Remove all fossils older than this.
        min_n_fossils (int): If `max_fossil_age` is set: Ensure sampled trees
            have at least this many fossils.

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

    # b = e / (4/5) = e*5/4
    # d = b * 1/5 = e*5/4/5 = e/4

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
        tree_simu, world = run_simulation(n_steps, tree_simu, world, condition_on_root=True)
        tree_simu.drop_fossils(max_fossil_age)

        # Check whether tree satisfies criteria...
        #    Criteria: not too small/big & root has two extant subtrees
        n_leafs = len([n for n in tree_simu.iter_leafs() if n.depth == n_steps])
        valid_tree = (min_leaves < n_leafs < max_leaves)

        if n_leafs < min_leaves:
            print('Invalid: Not enough leafs: %i' % n_leafs)
            continue
        elif n_leafs > max_leaves:
            print('Invalid: Too many leafs: %i' % n_leafs)
            continue
        for c in tree_simu.children:
            if not any(n.depth == n_steps for n in c.iter_leafs()):
                valid_tree = False
                print('Invalid: One side of the tree died!')
                break

        if valid_tree and (max_fossil_age > 0):
            if tree_simu.height() < n_steps:
                # This might happen if all languages on one side of the first split go extinct.
                valid_tree = False
                print('Invalid: Tree lost in height!')
            elif tree_simu.n_fossils() < min_n_fossils:
                valid_tree = False
                print('Invalid: Not enough fossils (only %i)' % tree_simu.n_fossils())

    print('Valid tree with %i leaves and %i fossils' % (tree_simu.n_leafs(), tree_simu.n_fossils()))
    if movement_model == 'tree_statistics':
        results = {}

    else:

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

    # Always include tree stats
    tree_stats = tree_statistics(tree_simu)
    results.update(tree_stats)

    return results


if __name__ == '__main__':
    HPD_VALUES = [80, 95]

    # Tree size settings
    SMALL = 50
    NORMAL = 100
    BIG = 300

    # Experiment CLI arguments
    MOVEMENT_MODEL = parse_arg(1, 'rrw')
    MAX_FOSSIL_AGE = parse_arg(2, 0, float)
    N_REPEAT = parse_arg(3, 100, int)
    TREE_SIZE = parse_arg(4, NORMAL, int)

    # Set working directory
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    max_age_str = str(MAX_FOSSIL_AGE if (MAX_FOSSIL_AGE % 1) else int(MAX_FOSSIL_AGE))
    WORKING_DIR = 'experiments/random_walk/{mm}_treesize={treesize}_fossils={max_age}/'
    WORKING_DIR = WORKING_DIR.format(mm=MOVEMENT_MODEL, treesize=TREE_SIZE, max_age=max_age_str)
    mkpath(WORKING_DIR)

    # Set cwd for logger
    LOGGER_PATH = os.path.join(WORKING_DIR, 'experiment.log')
    LOGGER = logging.getLogger('experiment')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.addHandler(logging.FileHandler(LOGGER_PATH))
    LOGGER.info('=' * 100)

    # Default experiment parameters
    default_settings = {
        # Simulation parameters
        'n_steps': 5000,
        'n_expected_leafs': TREE_SIZE,
        'total_diffusion': 2000.,
        'total_drift': 0.,
        'drift_density': 1.,
        'drift_direction': [0., 1.],
        'p_settle': 0.,
        'max_fossil_age': MAX_FOSSIL_AGE,

        # Analysis parameters
        'movement_model': MOVEMENT_MODEL,
        'chain_length': 1000000,
        'burnin': 100000,

        # Evaluation parameters
        'hpd_values': HPD_VALUES
    }

    EVAL_METRICS = ['size', 'imbalance', 'deep_imbalance',
                    'space_div_dependence', 'clade_overlap']

    if MOVEMENT_MODEL != 'tree_statistics':
        EVAL_METRICS += ['rmse', 'bias_x', 'bias_y', 'bias_norm', 'stdev'] + \
                        ['hpd_%i' % p for p in HPD_VALUES] + \
                        ['observed_stdev', 'observed_drift_x',  'observed_drift_y', 'observed_drift_norm']

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    total_drift_values = np.linspace(0., 3., 7) * default_settings['total_diffusion']
    variable_parameters = {'total_drift': total_drift_values}

    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, N_REPEAT, WORKING_DIR)
    experiment.run(resume=0)
