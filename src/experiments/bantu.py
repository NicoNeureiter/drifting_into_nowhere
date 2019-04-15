#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import numpy as np

from src.tree import Tree, get_edge_heights
from src.util import mkpath


NEWICK_TREE_PATH = 'data/bantu/bantu.nwk'
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
        tree_str = tree_file.read()

    tree = Tree.from_newick(tree_str.strip())
    tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)

    if exclude_outgroup:
        tree.remove_nodes_by_name(OUTGROUP_NAMES)

    # tree = tree.get_subtree([1,1,1,1,1,1])

    tree.write_beast_xml(xml_path, chain_length, root=root,
                         diffusion_on_a_sphere=True, movement_model=movement_model,
                         adapt_tree=adapt_tree, adapt_height=adapt_height)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import geopandas as gpd

    from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
    from src.plotting import plot_hpd, plot_root, plot_tree, plot_height, plot_edge
    from src.util import grey
    from src.config import COLOR_ROOT_TRUE, PINK, TURQUOISE

    CHAIN_LENGTH = 500000
    BURNIN = 50000
    HPD = 80

    MODEL = 'rrw'
    USE_OUTGROUP = 1
    ADAPT_TREE = 0
    ADAPT_HEIGHT = 0
    FIX_ROOT = 1

    WORKING_DIR = 'data/bantu_{model}_outgroup_{og}_adapttree_{at}_' + \
                  'adaptheight_{ah}_hpd_{hpd}{fixroot}/'
    WORKING_DIR = WORKING_DIR.format(
        model=MODEL, og=USE_OUTGROUP, at=ADAPT_TREE, ah=ADAPT_HEIGHT, hpd=HPD,
        fixroot='_fixroot' if FIX_ROOT else ''
    )
    BANTU_XML_PATH = WORKING_DIR + 'nowhere.xml'
    GEOJSON_PATH = 'africa.geojson'
    GEO_TREE_PATH = WORKING_DIR + 'nowhere.tree'
    mkpath(WORKING_DIR)

    # BANTU_ROOT = np.array([6.5, 10.5])
    BANTU_ROOT = np.array([10.5, 6.5])
    #
    write_bantu_xml(BANTU_XML_PATH, CHAIN_LENGTH,
                    root=BANTU_ROOT if FIX_ROOT else None,
                    exclude_outgroup=not USE_OUTGROUP,
                    adapt_tree=ADAPT_TREE,
                    adapt_height=ADAPT_HEIGHT,
                    movement_model=MODEL)
                    # movement_model='brownian')

    # Run the BEAST analysis
    run_beast(WORKING_DIR)
    run_treeannotator(HPD, BURNIN, WORKING_DIR)

    # Evaluate the results
    tree = load_tree_from_nexus(GEO_TREE_PATH)
    okcool = tree.root_in_hpd(BANTU_ROOT, HPD)
    print('\n\nOk cool: %r' % okcool)

    XLIM = (-30, 60)
    YLIM = (-35, 25)
    swap_xy = False
    HOMELAND = np.array([6.5, 10.5])
    LW = 0.1
    cmap = plt.get_cmap('viridis')

    world = gpd.read_file('data/naturalearth_50m_wgs84.geojson')
    ax = world.plot(color=grey(.95), edgecolor=grey(0.7), lw=.4, )

    from src.analyze_tree import flatten, invert

    plot_tree(tree, lw=LW, color='k')
              # cmap=cmap, color_fun=flatten(invert(get_edge_heights), .7))
    root = tree.location
    plt.scatter(root[0], root[1], marker='*', c=PINK, s=500, zorder=3)
    plt.scatter(HOMELAND[1], HOMELAND[0], marker='*', c=TURQUOISE, s=500, zorder=3)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()