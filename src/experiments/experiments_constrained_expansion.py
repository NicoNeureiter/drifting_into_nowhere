#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import os
import sys
import json

import scipy
import numpy as np

from src.experiments.experiment import Experiment
from src.experiments.experiments_random_walk import evaluate
from src.simulation.simulation import run_simulation
from src.simulation.grid_simulation import init_cone_simulation
from src.beast_interface import run_beast
from src.util import mkpath


LOGGER = logging.getLogger('experiment')


def run_experiment(n_steps, grid_size, cone_angle, split_size_range,
                   chain_length, burnin, hpd_values, working_dir,
                   movement_model='rrw', **kwargs):
    """Run an experiment ´n_runs´ times with the specified parameters.

    Args:
        n_steps (int): Number of steps to simulate.
        grid_size (int): Size of the simulation grid (the exact grid_size is
            adapted to the cone_angle to achieve consistent area/tree size.
        cone_angle (float): Angle of the free cone for the expansion.
        split_size_range (tuple[int,int]): Minimum and maximum area of a taxon.
        chain_length (int): MCMC chain length in BEAST analysis.
        burnin (int): MCMC burnin steps in BEAST analysis.
        hpd_values (list): The values for the HPD coverage statistics.
        working_dir (str): The working directory in which intermediate files
            will be dumped.

    Keyword Args:
        movement_model (str): The movement to be used in BEAST analysis
            Options: ['brownian', 'rrw', 'cdrw', 'rdrw']

    Returns:
        dict: Statistics of the experiments (different error values).
    """
    # Paths
    xml_path = working_dir + 'nowhere.xml'

    # Inferred parameters
    grid_size = int(grid_size / (cone_angle**0.5))

    # Run Simulation
    p_grow_distr = scipy.stats.beta(1., 1.).rvs
    world, tree_simu, _ = init_cone_simulation(grid_size=(grid_size, grid_size),
                                               p_grow_distr=p_grow_distr,
                                               cone_angle=cone_angle,
                                               split_size_range=split_size_range)
    run_simulation(n_steps, tree_simu, world)
    root = tree_simu.location

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


if __name__ == '__main__':
    N_REPEAT = 60
    HPD_VALUES = [80, 95]

    # MOVEMENT MODEL
    if len(sys.argv) > 1:
        MOVEMENT_MODEL = sys.argv[1]
    else:
        MOVEMENT_MODEL = 'rdrw'

    # Set working directory
    WORKING_DIR = 'experiments/constrained_expansion/%s/' % MOVEMENT_MODEL
    mkpath(WORKING_DIR)

    # Set cwd for logger
    LOGGER_PATH = os.path.join(WORKING_DIR, 'experiment.log')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.addHandler(logging.FileHandler(LOGGER_PATH))
    LOGGER.info('=' * 100)

    # Default experiment parameters
    simulation_settings = {
        'n_steps': 5000,
        'grid_size': 200,
        'split_size_range': (70,100),
    }

    default_settings = {
        # Analysis Parameters
        'movement_model': MOVEMENT_MODEL,
        'chain_length': 1000000,
        'burnin': 100000,
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
    variable_parameters = {'cone_angle': np.linspace(0.2,2,10) * np.pi}
    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, N_REPEAT, WORKING_DIR)
    experiment.run(resume=1)
