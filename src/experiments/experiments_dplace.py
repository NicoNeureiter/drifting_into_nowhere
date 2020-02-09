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
from src.evaluation import evaluate, tree_statistics
from src.simulation.simulation import run_simulation
from src.simulation.grid_simulation import init_cone_simulation
from src.beast_interface import run_beast, load_trees
from src.tree import tree_imbalance
from src.util import mkpath, parse_arg

LFAM = 'indo-european'
BANTU_POSTERIOR_PATH = 'data/bantu/posterior.trees'
LOCATIONS_PATH = 'data/bantu/bantu_locations.csv'

bantu_trees = load_trees(BANTU_POSTERIOR_PATH, read_name_mapping=True)
for tree in bantu_trees:
    tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)
    leafs_without_locations = [node.name for node in tree.iter_leafs() if node.location is None]
    tree.remove_nodes_by_name(leafs_without_locations)


for tree in bantu_trees[:3]:
    print(tree.to_newick())
# exit()


def run_experiment(i_tree, **kwargs):
    """Run experiment on samples of the Bantu tree posterior distribution.

    Args:
        i_tree (int): Index of the tree in the Bantu posterior samples.

    Returns:
        dict: Statistics of the experiments (different error values).
    """
    tree = bantu_trees[i_tree]
    results = tree_statistics(tree)
    return results


if __name__ == '__main__':

    # Set working directory
    WORKING_DIR = 'experiments/bantu/tree_statistics/'
    mkpath(WORKING_DIR)

    # Set cwd for logger
    LOGGER_PATH = os.path.join(WORKING_DIR, 'experiment.log')
    LOGGER = logging.getLogger('experiment')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.addHandler(logging.FileHandler(LOGGER_PATH))
    LOGGER.info('=' * 100)

    # Default experiment parameters
    default_settings = {}

    EVAL_METRICS = [
        'size', 'imbalance',
        # 'size_0_small', 'size_0_big', 'size_1_small', 'size_1_big', 'size_2_small', 'size_2_big',
        # 'imbalance_0', 'imbalance_1', 'imbalance_2', 'imbalance_3',
        # 'migr_rate_0', 'migr_rate_0_small', 'migr_rate_0_big', 'migr_rate_1_small', 'migr_rate_1_big', 'migr_rate_2_small', 'migr_rate_2_big',
        # 'drift_rate_0', 'drift_rate_0_small', 'drift_rate_0_big', 'drift_rate_1_small', 'drift_rate_1_big', 'drift_rate_2_small', 'drift_rate_2_big',
        # 'log_div_rate_0', 'log_div_rate_0_small', 'log_div_rate_0_big', 'log_div_rate_1_small', 'log_div_rate_1_big', 'log_div_rate_2_small', 'log_div_rate_2_big',
        'space_div_dependence', 'clade_overlap', 'deep_imbalance']

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    # variable_parameters = {'cone_angle': np.linspace(0.2,2,12) * np.pi}
    variable_parameters = {'i_tree': list(range(len(bantu_trees)))}
    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, 1, WORKING_DIR)
    experiment.run(resume=0)
