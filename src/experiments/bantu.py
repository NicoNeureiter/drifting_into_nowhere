#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import random

import numpy as np

from src.tree import Tree, angle_to_vector
from src.util import mkpath


NEWICK_TREE_PATH = 'data/bantu/bantu.nwk'
POSTERIOR_PATH = 'data/bantu/posterior.trees'
LOCATIONS_PATH = 'data/bantu/bantu_locations.csv'


OUTGROUP_NAMES = [
    'Fefe_Grassfields',
    'Mungaka_Grassfields',
    'Bamun_Grassfields',
    'Kom_Grassfields',
    'Oku_Grassfields',
    'Aghem_Grassfields',
    'Njen_Grassfields',
    'Moghamo_Grassfields',
    # 'Zaambo_Jarawan',
    # 'Bwazza_Jarawan',
    # 'Mbula_Jarawan',
    # 'Bile_Jarawan',
    # 'Kulung_Jarawan',
    # 'Duguri_Jarawan',
    'Tiv_Tivoid',
]


def write_bantu_xml(xml_path, chain_length, root=None, exclude_outgroup=False,
                    movement_model='rrw', adapt_tree=False, adapt_height=False):
    with open(NEWICK_TREE_PATH, 'r') as tree_file:
        tree_str = tree_file.read().lower().strip()
    tree = Tree.from_newick(tree_str.strip())
    tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)


    if exclude_outgroup:
        tree.remove_nodes_by_name(OUTGROUP_NAMES)

    tree = tree.big_child().big_child().big_child()

    tree.write_beast_xml(xml_path, chain_length, root=root,
                         diffusion_on_a_sphere=True, movement_model=movement_model,
                         adapt_tree=adapt_tree, adapt_height=adapt_height)


def write_bantu_sample_xml(xml_path, chain_length, root=None, exclude_outgroup=False,
                    movement_model='rrw', adapt_tree=False, adapt_height=False):
    with open(POSTERIOR_PATH, 'r') as tree_file:
        nexus_str = tree_file.read().lower().strip()

    name_map = read_nexus_name_mapping(nexus_str)
    tree_lines = [line.split(' = ')[-1] for line in nexus_str.split('\n') if line.startswith('\t\ttree')]
    tree_str = random.choice(tree_lines)
    tree = Tree.from_newick(tree_str.strip())

    for node in tree.iter_descendants():
        if node.name in name_map:
            node.name = name_map[node.name]

    print(tree.to_newick())

    tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)
    leafs_without_locations = [node.name for node in tree.iter_leafs() if node.location is None]
    tree.remove_nodes_by_name(leafs_without_locations)

    if exclude_outgroup:
        tree.remove_nodes_by_name(OUTGROUP_NAMES)

    tree.write_beast_xml(xml_path, chain_length, root=root,
                         diffusion_on_a_sphere=True, movement_model=movement_model,
                         adapt_tree=adapt_tree, adapt_height=adapt_height)

DRIFT_ANGLE = -0.5
DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)
def get_space_time_position(node):
    t = -node.height
    x = np.dot(node.location, DRIFT_DIRECTION)
    return x, t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import geopandas as gpd

    from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus, \
        read_nexus_name_mapping, load_trees
    from src.plotting import plot_hpd, plot_tree, plot_posterior_scatter
    from src.util import grey
    from src.config import PINK, TURQUOISE

    world = gpd.read_file('data/naturalearth_50m_wgs84.geojson')
    ax = world.plot(color=grey(.95), edgecolor=grey(0.7), lw=.4, )

    CHAIN_LENGTH = 500000
    BURNIN = 10000
    HPD = 90

    MODEL = 'rrw'
    USE_OUTGROUP = False
    ADAPT_TREE = False
    ADAPT_HEIGHT = False
    FIX_ROOT = False

    SUFFIX = '_missingClades'

    WORKING_DIR = 'data/bantu_{model}_outgroup_{og}_adapttree_{at}_' + \
                  'adaptheight_{ah}_hpd_{hpd}{fixroot}{suffix}/'
    WORKING_DIR = WORKING_DIR.format(
        model=MODEL, og=USE_OUTGROUP, at=ADAPT_TREE, ah=ADAPT_HEIGHT, hpd=HPD,
        fixroot='_fixroot' if FIX_ROOT else '', suffix=SUFFIX
    )
    BANTU_XML_PATH = WORKING_DIR + 'nowhere.xml'
    GEOJSON_PATH = 'africa.geojson'
    GEO_TREE_PATH = WORKING_DIR + 'nowhere.tree'
    GEO_TREES_PATH = WORKING_DIR + 'nowhere.trees'
    mkpath(WORKING_DIR)

    HOMELAND = np.array([10.5, 6.5])
    ax.scatter(*HOMELAND, marker='*', c=TURQUOISE, s=200, zorder=3)

    # with open(NEWICK_TREE_PATH, 'r') as tree_file:
    #     tree_str = tree_file.read().lower().strip()
    # tree = Tree.from_newick(tree_str.strip())
    # tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)
    # locations = tree.get_leaf_locations()
    # from workbench.map_projections import WorldMap
    # locations.

    write_bantu_xml(BANTU_XML_PATH, CHAIN_LENGTH,
                   root=HOMELAND if FIX_ROOT else None,
                   exclude_outgroup=not USE_OUTGROUP,
                   adapt_tree=ADAPT_TREE,
                   adapt_height=ADAPT_HEIGHT,
                   movement_model=MODEL)

    # Run the BEAST analysis
    run_beast(WORKING_DIR)
    run_treeannotator(HPD, BURNIN, WORKING_DIR)

    # Evaluate the results
    tree = load_tree_from_nexus(GEO_TREE_PATH)
    # trees = load_trees(GEO_TREES_PATH)
    okcool = tree.root_in_hpd(HOMELAND, HPD)
    # print('\n\nOk cool: %r' % okcool)

    XLIM = (-20, 60)
    YLIM = (-35, 38)
    swap_xy = False
    LW = 0.7
    cmap = plt.get_cmap('viridis')

    plot_tree(tree, color='k', lw=LW, alpha=0.3)
              # cmap=cmap, color_fun=flatten(invert(get_edge_heights), .7))
    root = tree.location
    plt.scatter(root[0], root[1], marker='*', c=PINK, s=200, zorder=3)
    # plot_hpd(tree, HPD)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()