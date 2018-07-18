#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np


NEXUS_TEMPLATE = '''#NEXUS
BEGIN DATA;
    DIMENSIONS NTAX={n_societies} NCHAR={n_features};
    FORMAT DATATYPE=standard SYMBOLS={symbols} MISSING=? GAP=-;
    MATRIX
{data}\t;
END;

BEGIN TREES;
      TREE original_tree = {tree};
END;
'''


def write_nexus(simulation, path, fossils=None):
    data_str = ''

    for i, state in enumerate(simulation.societies):
        name = state.name
        fs = state.featureState.features.astype(int)

        line = '\t\t' + name + '\t' + ''.join(map(str, fs)) + '\n'
        data_str += line

    for i, state in enumerate(simulation.history):
        name = 'h' + state.name
        fs = '?' * simulation.n_features
        line = '\t\t%s\t%s\n' % (name, fs)
        data_str += line

    tree = simulation.get_newick_tree()

    nexus_str = NEXUS_TEMPLATE.format(
        n_societies = simulation.n_sites + len(simulation.history),
        n_features = simulation.n_features,
        symbols = '01',
        data = data_str,
        tree = tree
    )

    with open(path, 'w') as nexus_file:
        nexus_file.write(nexus_str)


LOCATIONS_TAMPLATE = 'traits\tx\ty\n{data}'

def write_locations(simulation, path):
    rows = []
    for i, state in enumerate(simulation.societies):
        name = state.name
        loc = state.geoState.location

        row = '%s\t%f\t%f' % (name, loc[0], loc[1])
        rows.append(row)

    data_str = '\n'.join(rows)

    locations_str = LOCATIONS_TAMPLATE.format(data=data_str)

    with open(path, 'w') as locations_file:
        locations_file.write(locations_str)


XML_TEMPLATE_PATH = 'data/templates/beast_xml_template.xml'
FEATURES_TEMPLATE_PATH = 'data/templates/features_template.xml'
LOCATION_TEMPLATE_PATH = 'data/templates/location_template.xml'


def write_beast_xml(simulation, path, chain_length=100000, fix_root=False):
    with open(XML_TEMPLATE_PATH, 'r') as xml_template_file:
        xml_template = xml_template_file.read()
    with open(FEATURES_TEMPLATE_PATH, 'r') as features_template_file:
        features_template = features_template_file.read()
    with open(LOCATION_TEMPLATE_PATH, 'r') as locations_template_file:
        location_template = locations_template_file.read()

    locations_xml = ''
    features_xml = ''

    for i, state in enumerate(simulation.societies):
        name = state.name

        loc = state.geoState.location
        locations_xml += location_template.format(id=name, x=loc[0], y=loc[1])

        fs = state.featureState.features.astype(int)
        fs_str = ''.join(map(str, fs))
        features_xml += features_template.format(id=name, features=fs_str)

    for i, state in enumerate(simulation.history):
        name = state.name

        locations_xml += '\t\t<taxon id="{name}"/>\n'.format(name=name)

        fs_str = '?' * simulation.n_features
        features_xml += features_template.format(id=name, features=fs_str)

    tree_xml = simulation.get_newick_tree()

    root = simulation.root.geoState.location
    if fix_root:
        root_precision = 1e8
    else:
        root_precision = 1e-8

    with open(path, 'w') as beast_xml_file:
        beast_xml_file.write(
            xml_template.format(
                locations=locations_xml,
                features=features_xml,
                tree=tree_xml,
                root_x=root[0],
                root_y=root[1],
                root_precision=root_precision,
                chain_length=chain_length,
                ntax=simulation.n_sites,
                nchar=simulation.n_features
            )
        )