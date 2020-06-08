#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
import os
import sys
import json

import numpy as np
import pandas as pd
import geopandas as gpd

from src.experiments.experiment import Experiment
from src.evaluation import evaluate, tree_statistics
from src.beast_interface import load_trees
from src.tree import tree_imbalance, rename_nodes, naive_location_reconstruction, assign_lcations
from src.util import mkpath, parse_arg, grey
from src.plotting import plot_tree

LFAM = 'pama-nyungan'
# LFAM = 'sino-tibetan'
# LFAM = 'uto-aztecan'

POSTERIOR_PATH = f'data/dplace/{LFAM}.trees'
CODES_PATH = f'data/dplace/{LFAM}.csv'
LOCATIONS_PATH = 'data/codes_to_locs.csv'


codes = pd.read_csv(CODES_PATH, sep=',', index_col='taxon')
codes.index = codes.index.map(str.lower)
codes = codes[['glottocode']].dropna()

locations = pd.read_csv(LOCATIONS_PATH, sep='\t')
locations = locations[['Glottolog', 'Latitude', 'Longitude']]
locations = locations.dropna()
locations = locations.drop_duplicates('Glottolog')
locations = locations.set_index('Glottolog')

locations = codes.join(locations, on='glottocode', how='inner')
locations = locations[['Latitude', 'Longitude']]

ntrees = None
trees = load_trees(POSTERIOR_PATH, read_name_mapping=True, max_trees=ntrees)

remove_list = []
first = True
for tree in trees[:ntrees]:
    if first:
        for node in tree.iter_leafs():
            if node.name not in locations.index:
                remove_list.append(node.name)
        print('Remove %i of %i leaves' % (len(remove_list), tree.n_leafs()), end='  \t:  \t')
        print(remove_list)

        first = False


    tree.remove_nodes_by_name(remove_list)

    # tree.binarize()

    # rename_nodes(tree, glottocodes)

    locs = []
    for node in tree.iter_leafs():
        y, x = locations.loc[node.name]
        loc = np.array([x, y])
        while tuple(loc) in locs:
            loc = np.array([x, y]) + np.random.normal(0., 0.001, size=(2,))
        locs.append(tuple(loc))
        node._location = loc


def run_experiment(i_tree, **kwargs):
    """Run experiment on samples of the Bantu tree posterior distribution.

    Args:
        i_tree (int): Index of the tree in the Bantu posterior samples.

    Returns:
        dict: Statistics of the experiments (different error values).
    """
    tree = trees[i_tree]
    results = tree_statistics(tree)
    return results


if __name__ == '__main__':
    # Set working directory
    WORKING_DIR = f'experiments/{LFAM}/tree_statistics/'
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
    variable_parameters = {'i_tree': list(range(len(trees)))}
    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, 1, WORKING_DIR)
    experiment.run(resume=0)
