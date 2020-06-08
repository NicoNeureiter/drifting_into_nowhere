#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import os
import sys
import json

from src.experiments.experiment import Experiment
from src.evaluation import tree_statistics
from src.beast_interface import load_trees
from src.util import mkpath

BANTU_POSTERIOR_PATH = 'data/bantu/posterior.trees'
LOCATIONS_PATH = 'data/bantu/bantu_locations.csv'
OUTGROUP_NAMES = ['Fefe_Grassfields', 'Mungaka_Grassfields', 'Bamun_Grassfields',
                  'Kom_Grassfields', 'Oku_Grassfields', 'Aghem_Grassfields',
                  'Njen_Grassfields', 'Moghamo_Grassfields', 'Tiv_Tivoid',]

bantu_trees = load_trees(BANTU_POSTERIOR_PATH, read_name_mapping=True)
for tree in bantu_trees:
    tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)
    leafs_without_locations = [node.name for node in tree.iter_leafs() if node.location is None]
    tree.remove_nodes_by_name(leafs_without_locations)
    tree.remove_nodes_by_name(OUTGROUP_NAMES)


for tree in bantu_trees[:3]:
    print(tree.to_newick())


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

    EVAL_METRICS = ['size', 'imbalance', 'deep_imbalance',
                    'space_div_dependence', 'clade_overlap']

    # Safe the default settings
    with open(WORKING_DIR+'settings.json', 'w') as json_file:
        json.dump(default_settings, json_file)

    # Run the experiment
    variable_parameters = {'i_tree': list(range(len(bantu_trees)))}
    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, 1, WORKING_DIR)
    experiment.run(resume=0)
