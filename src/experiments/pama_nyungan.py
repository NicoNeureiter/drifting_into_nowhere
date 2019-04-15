#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import logging

import pandas as pd
import numpy as np
import newick

from src.tree import Tree, get_edge_heights
from src.util import extract_newick_from_nexus


def load_pn_tree(nexus_path):
    with open(nexus_path, 'r') as nexus_file:
        nwk_str = extract_newick_from_nexus(nexus_file.read())
    return Tree.from_newick(nwk_str)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
    from src.plotting import (plot_hpd, plot_root, plot_tree, plot_height,
                              plot_edge, plot_backbone_splits, plot_tree_topology,
                              plot_clades)
    from src.tree import tree_imbalance
    from src.util import mkpath, grey
    from src.analyze_tree import flatten
    from src.experiments.dediu_forest import swap_xy
    from workbench.map_projections import WorlMap

    CHAIN_LENGTH = 100000
    BURNIN = 30000
    HPD = 80

    # Paths to experiment files
    NEXUS_PATH = 'data/pama_nyungan.nex'
    WORKING_DIR = 'experiments/pama-nyungan/'
    mkpath(WORKING_DIR)
    XML_PATH = os.path.join(WORKING_DIR, 'nowhere.xml')
    GEO_TREE_PATH = WORKING_DIR + 'nowhere.tree'

    # Plotting cfg
    cmap = plt.get_cmap('viridis')

    if 0:
        tree = load_pn_tree(NEXUS_PATH)

        # Prepare tree
        tree.binarize()
        swap_xy(tree)
        for node in tree.iter_descendants():
            node.length /= 1000.
        print(tree_imbalance(tree))

        # Write to xml
        tree.write_beast_xml(XML_PATH, CHAIN_LENGTH, diffusion_on_a_sphere=True,
                             movement_model='rrw', adapt_tree=False,
                             # movement_model='brownian', adapt_tree=False,
                             adapt_height=False, jitter=0.02)

        # Run the BEAST analysis
        run_beast(WORKING_DIR)
        run_treeannotator(HPD, BURNIN, WORKING_DIR)

    # Load the tree
    tree = load_tree_from_nexus(GEO_TREE_PATH)

    if 0:  # Plot time
        from src.util import transform_tree_coordinates, time_drift_trafo
        transform_tree_coordinates(tree, time_drift_trafo)
    else:
        # Prep Map
        locations = tree.get_descendant_locations()
        world_map = WorlMap()
        world_map.align_to_data(locations)
        ax = world_map.plot()
        for node in tree.iter_descendants():
            node.location = world_map.project(node.location)

    ax = plt.gca()
    plot_clades(tree, max_clade_size=50)
    plot_root(tree.location, color='#ff0022', s=1600, edgecolor='w')
    # plot_hpd(tree, HPD, projection=world_map.project)
    center = np.mean(tree.get_leaf_locations(), axis=0)
    plt.scatter(*center, color='k', s=1600, marker='o', zorder=3)
    center = np.median(tree.get_leaf_locations(), axis=0)
    plt.scatter(*center, color='k', s=1600, marker='s', zorder=3)

    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()
