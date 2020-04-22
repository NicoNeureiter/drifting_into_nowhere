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

# LFAM = 'pama-nyungan'
# LFAM = 'sino-tibetan'
LFAM = 'uto-aztecan'
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

# world = gpd.read_file('data/naturalearth_50m_wgs84.geojson')
# ax = world.plot(color=grey(.95), edgecolor=grey(0.7), lw=.33, )

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
            # print('.', end='')
            # print(x, y)
            loc = np.array([x, y]) + np.random.normal(0., 0.001, size=(2,))
        locs.append(tuple(loc))
        node._location = loc

        # leafs_without_locations = [node.name for node in tree.iter_leafs() if node.location is None]
    # tree.remove_nodes_by_name(leafs_without_locations)

#     naive_location_reconstruction(tree)
#     # plot_tree(tree, alpha=0.4, lw=.2)
#     plot_tree(tree, lw=.3)
#
# import matplotlib.pyplot as plt
# plt.xlim(-20, 100)
# plt.ylim(0, 75)
# plt.show()
# exit()


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
    variable_parameters = {'i_tree': list(range(len(trees)))}
    experiment = Experiment(run_experiment, default_settings, variable_parameters,
                            EVAL_METRICS, 1, WORKING_DIR)
    experiment.run(resume=0)
