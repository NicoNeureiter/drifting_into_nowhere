#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import sys
import json

import scipy
import numpy as np

from src.experiments.experiment import Experiment
from src.evaluation import evaluate, tree_statistics
from src.simulation.simulation import run_simulation
from src.simulation.expansion_simulation_overlap import init_cone_simulation
from src.beast_interface import run_beast
from src.tree import tree_imbalance
from src.util import mkpath, parse_arg, sample_random_subtree


def run_experiment(n_steps, grid_size, cone_angle, split_size_range,
                   chain_length, burnin, hpd_values, working_dir,
                   p_conflict, p_grow_mu=0.5, p_grow_nu = 2, death_rate=0.,
                   movement_model='rrw', fixed_tree_size=None, **kwargs):
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
    # print(grid_size)
    #
    a, b = split_size_range
    split_size_range = (int(a * cone_angle**0.25),
                        int(b * cone_angle**0.25))

    # Run Simulation
    p_grow_distr = scipy.stats.beta(p_grow_nu * p_grow_mu,
                                    p_grow_nu * (1 - p_grow_mu)
                                    ).rvs

    valid_tree = False
    while not valid_tree:
        world, tree_simu, _ = init_cone_simulation(grid_size=(grid_size, grid_size),
                                                   p_grow_distr=p_grow_distr,
                                                   cone_angle=cone_angle,
                                                   split_size_range=split_size_range,
                                                   p_conflict=p_conflict, death_rate=death_rate)
        run_simulation(n_steps, tree_simu, world)

        tree_simu.drop_fossils()
        if fixed_tree_size is None:
            valid_tree = True
        else:
            if tree_simu.n_leafs() >= fixed_tree_size:
                sample_random_subtree(tree_simu, fixed_tree_size)
                valid_tree = True
            else:
                print('!!!', tree_simu.n_leafs())


    root = tree_simu.location

    print('Simulation... done')
    print('n_leafs:', tree_simu.n_leafs())

    if movement_model == 'tree_statistics':
        results = tree_statistics(tree_simu)
    else:
        # Create an XML file as input for the BEAST analysis
        tree_simu.write_beast_xml(xml_path, chain_length, movement_model=movement_model,
                                  drift_prior_std=1.)

        # Run phylogeographic reconstruction in BEAST
        run_beast(working_dir=working_dir)

        results = evaluate(working_dir, burnin, hpd_values, root)
        results['size'] = tree_simu.n_leafs()
        results['root_x'], results['root_y'] = root
        results['root_norm'] = np.hypot(*root)

    # Add statistics about simulated tree (to compare between simulation modes)
    results['observed_stdev'] = np.hypot(*np.std(tree_simu.get_leaf_locations(), axis=0))
    leafs_mean = np.mean(tree_simu.get_leaf_locations(), axis=0)
    leafs_mean_offset = leafs_mean - root
    results['observed_drift_x'] = leafs_mean_offset[0]
    results['observed_drift_y'] = leafs_mean_offset[1]
    results['observed_drift_norm'] = np.hypot(*leafs_mean_offset)

    return results


if __name__ == '__main__':
    HPD_VALUES = [80, 95]

    # MOVEMENT MODEL
    MOVEMENT_MODEL = 'rrw'
    # MOVEMENT_MODEL = 'tree_statistics'
    MOVEMENT_MODEL = parse_arg(1, MOVEMENT_MODEL, str)
    N_REPEAT = parse_arg(2, 60, int)

    # P_CONFLICT = parse_arg(3, 0., float)
    # P_CONFLICT = parse_arg(3, .333, float)
    # P_CONFLICT = parse_arg(3, .666, float)
    P_CONFLICT = parse_arg(3, 1., float)

    # P_CONFLICT = parse_arg(3, .5, float)

    # Set working directory
    WORKING_DIR = 'experiments/constrained_expansion_overlap_%s/%s/' % (P_CONFLICT, MOVEMENT_MODEL)
    mkpath(WORKING_DIR)

    # Set cwd for logger
    LOGGER_PATH = os.path.join(WORKING_DIR, 'experiment.log')
    LOGGER = logging.getLogger('experiment')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.addHandler(logging.FileHandler(LOGGER_PATH))
    LOGGER.info('=' * 100)

    SPLIT = {.0: 32, .333: 60, .666: 75, 1.: 89}
    N_STEPS = {.0: 118, .333: 185, .666: 191, 1.: 200}

    # Default experiment parameters
    simulation_settings = {
        'n_steps': N_STEPS[P_CONFLICT],
        'grid_size': 250,
        'split_size_range': (SPLIT[P_CONFLICT],
                             SPLIT[P_CONFLICT] + 5),
        'p_conflict': P_CONFLICT,

        'p_grow_mu': 0.9,
        'p_grow_nu': 100,
        'death_rate': 0.0,
        'fixed_tree_size': None,
    }

    default_settings = {
        # Analysis Parameters
        'movement_model': MOVEMENT_MODEL,
        # 'chain_length': 1000000,
        'chain_length': 150000,
        'burnin': 50000,
        # Experiment Settings
        'hpd_values': HPD_VALUES,
    }
    default_settings.update(simulation_settings)

    if MOVEMENT_MODEL == 'tree_statistics':
        EVAL_METRICS = [
            'size', 'imbalance',
            'space_div_dependence', 'clade_overlap', 'deep_imbalance']

    else:
        EVAL_METRICS = ['rmse', 'bias_x', 'bias_y', 'bias_norm', 'stdev'] + \
                       ['hpd_%i' % p for p in HPD_VALUES] + \
                       ['size', 'root_x', 'root_y', 'root_norm']

    EVAL_METRICS += ['observed_stdev', 'observed_drift_x',  'observed_drift_y', 'observed_drift_norm']

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    # variable_parameters = {'cone_angle': np.linspace(0.2, 2, 4) * np.pi}
    variable_parameters = {'cone_angle': np.linspace(0.2, 2, 10) * np.pi}
    # variable_parameters = {'cone_angle': np.linspace(0.25, 2, 8) * np.pi}
    # variable_parameters = {'cone_angle': np.array([0.8, ]) * np.pi}

    print(default_settings)
    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, N_REPEAT, WORKING_DIR)
    experiment.run(resume=1)