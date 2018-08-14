#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

from src.util import read_locations_file
from src.beast_xml_templates import *
from src.tree import Node

NEWICK_TREE_PATH = 'data/bantu/bantu.nwk'
LOCATIONS_PATH = 'data/bantu/bantu_locations.csv'

BANTU_ROOT = np.array([6.5, 10.5])

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


def write_bantu_xml(bantu_xml_path, chain_length, fix_root=False, exclude_outgroup=False):
    with open(XML_TEMPLATE_PATH, 'r') as xml_template_file:
        xml_template = xml_template_file.read()
    with open(NEWICK_TREE_PATH, 'r') as tree_file:
        tree_str = tree_file.read()

    if exclude_outgroup:
        tree = Node.from_newick(tree_str.strip())
        print(tree.tree_size)
        tree.remove_nodes_by_name(OUTGROUP_NAMES)
        tree_str = tree.to_newick()
        print(tree.tree_size)

    locations, _ = read_locations_file(LOCATIONS_PATH)
    locations_xml = ''
    features_xml = ''

    for name, loc in locations.items():
        if exclude_outgroup and name in OUTGROUP_NAMES:
            continue

        locations_xml += LOCATION_TEMPLATE.format(id=name, x=loc[0], y=loc[1])
        features_xml += FEATURES_TEMPLATE.format(id=name, features='0')

    root = BANTU_ROOT
    if fix_root:
        root_precision = 1e8
    else:
        root_precision = 1e-8

    with open(bantu_xml_path, 'w') as beast_xml_file:
        beast_xml_file.write(
            xml_template.format(
                locations=locations_xml,
                features=features_xml,
                tree=tree_str,
                root_x=root[0],
                root_y=root[1],
                root_precision=root_precision,
                chain_length=chain_length,
                ntax=len(locations),
                nchar=1,
                jitter=1.
            )
        )


if __name__ == '__main__':
    import os

    import matplotlib.pyplot as plt
    import geopandas as gpd

    from src.evaluation import check_root_in_hpd
    # from src.plotting import plot_walk

    CHAIN_LENGTH = 100000
    BURNIN = 5000
    HPD = 80

    SCRIPT_PATH = 'src/beast_pipeline.sh'
    BANTU_XML_PATH = 'data/bantu/nowhere.xml'
    GEOJSON_PATH = 'africa.geojson'
    GEO_TREE_PATH = 'data/bantu/nowhere.tree'

    write_bantu_xml(BANTU_XML_PATH, CHAIN_LENGTH, fix_root=False, exclude_outgroup=True)

    # Run the BEAST analysis + summary of results (treeannotator)
    os.system('bash {script} {hpd} {burnin} {cwd} {geojson}'.format(
        script=SCRIPT_PATH,
        hpd=HPD,
        burnin=BURNIN,
        cwd=os.getcwd()+'/data/bantu/',
        geojson=GEOJSON_PATH
    ))

    ax = plt.gca()

    # Evaluate the results
    okcool = check_root_in_hpd(GEO_TREE_PATH, HPD, root=BANTU_ROOT, ax=ax)
    print('\n\nOk cool: %r' % okcool)

    # # # Plot the true (simulated) evolution
    # # plot_walk(simulation, show_path=False, show_tree=True, ax=ax)
    #
    # plt.axis('off')
    # plt.tight_layout()
    #
    # plt.show()