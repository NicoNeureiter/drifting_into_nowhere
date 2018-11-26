#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import numpy as np

from src.tree import Node
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
                    movement_model='rrw'):
    with open(NEWICK_TREE_PATH, 'r') as tree_file:
        tree_str = tree_file.read()

    tree = Node.from_newick(tree_str.strip())
    tree.load_locations_from_csv(LOCATIONS_PATH, swap_xy=True)

    if exclude_outgroup:
        tree.remove_nodes_by_name(OUTGROUP_NAMES)

    tree.write_beast_xml(xml_path, chain_length, root=root,
                         diffusion_on_a_sphere=True, movement_model=movement_model)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
    from src.plotting import plot_hpd, plot_root
    from src.config import COLOR_ROOT_TRUE

    CHAIN_LENGTH = 2000000
    BURNIN = 50000
    HPD = 80

    WORKING_DIR = 'data/bantu_withoutgroup_2/'
    SCRIPT_PATH = 'src/beast_pipeline.sh'
    BANTU_XML_PATH = WORKING_DIR + 'nowhere.xml'
    GEOJSON_PATH = 'africa.geojson'
    GEO_TREE_PATH = WORKING_DIR + 'nowhere.tree'
    mkpath(WORKING_DIR)

    # BANTU_ROOT = np.array([6.5, 10.5])
    BANTU_ROOT = np.array([10.5, 6.5])

    write_bantu_xml(BANTU_XML_PATH, CHAIN_LENGTH, root=None, exclude_outgroup=True,
                    movement_model='rrw')
                    # movement_model='brownian')

    # Run the BEAST analysis
    run_beast(WORKING_DIR)
    run_treeannotator(HPD, BURNIN, WORKING_DIR)

    # Evaluate the results
    tree = load_tree_from_nexus(GEO_TREE_PATH)
    okcool = tree.root_in_hpd(BANTU_ROOT, HPD)
    print('\n\nOk cool: %r' % okcool)
