#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import numpy as np

from src.evaluation import check_root_in_hpd
from src.tree import Node


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

    CHAIN_LENGTH = 1000000
    BURNIN = 20000
    HPD = 80

    BASE_DIR = 'data/bantu_brownian/'
    SCRIPT_PATH = 'src/beast_pipeline.sh'
    BANTU_XML_PATH = BASE_DIR + 'nowhere.xml'
    GEOJSON_PATH = 'africa.geojson'
    GEO_TREE_PATH = BASE_DIR + 'nowhere.tree'

    BANTU_ROOT = np.array([6.5, 10.5])

    os.makedirs(os.path.dirname(BASE_DIR), exist_ok=True)

    root = BANTU_ROOT
    write_bantu_xml(BANTU_XML_PATH, CHAIN_LENGTH, root=None, exclude_outgroup=True,
                    movement_model='brownian')

    # Run the BEAST analysis + summary of results (treeannotator)
    os.system('bash {script} {hpd} {burnin} {cwd} {geojson}'.format(
        script=SCRIPT_PATH,
        hpd=HPD,
        burnin=BURNIN,
        cwd=os.path.join(os.getcwd(),BASE_DIR),
        geojson=GEOJSON_PATH
    ))

    # Evaluate the results
    okcool = check_root_in_hpd(GEO_TREE_PATH, HPD, root=BANTU_ROOT[::-1])
    print('\n\nOk cool: %r' % okcool)