#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

from src.beast_xml_templates import *


def write_nexus(simulation, path, fossils=None):
    data_str = ''

    for i, state in enumerate(simulation.societies):
        name = state.name
        fs = state.featureState.features.astype(int)

        line = '\t\t' + name + '\t' + ''.join(map(str, fs)) + '\n'
        data_str += line

    # for i, state in enumerate(simulation.history):
    #     name = 'h' + state.name
    #     fs = '?' * simulation.n_features
    #     line = '\t\t%s\t%s\n' % (name, fs)
    #     data_str += line

    tree = simulation.get_newick_tree()

    nexus_str = NEXUS_TEMPLATE.format(
        n_societies = simulation.n_sites,  # + len(simulation.history),
        n_features = simulation.n_features,
        symbols = '01',
        data = data_str,
        tree = tree
    )

    with open(path, 'w') as nexus_file:
        nexus_file.write(nexus_str)

def write_locations(simulation, path):
    rows = []
    for i, state in enumerate(simulation.societies):
        name = state.name
        loc = state.geoState.location

        row = '%s\t%f\t%f' % (name, loc[0], loc[1])
        rows.append(row)

    data_str = '\n'.join(rows)

    locations_str = LOCATIONS_CSV_TEMPLATE.format(data=data_str)

    with open(path, 'w') as locations_file:
        locations_file.write(locations_str)


def write_beast_xml(simulation, path, chain_length, fix_root=False):
    with open(RRW_XML_TEMPLATE_PATH, 'r') as xml_template_file:
        xml_template = xml_template_file.read()

    locations_xml = ''
    features_xml = ''

    for i, state in enumerate(simulation.societies):
        name = state.name

        loc = state.geoState.location
        locations_xml += LOCATION_TEMPLATE.format(id=name, x=loc[0], y=loc[1])

        fs = state.featureState.features.astype(int)
        fs_str = ''.join(map(str, fs))
        features_xml += FEATURES_TEMPLATE.format(id=name, features=fs_str)

    for i, state in enumerate(simulation.history):
        name = state.name

        locations_xml += '\t\t<taxon id="{name}"/>\n'.format(name=name)

        fs_str = '?' * simulation.n_features
        features_xml += FEATURES_TEMPLATE.format(id=name, features=fs_str)

    tree = simulation.get_newick_tree()

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
                tree=tree,
                root_x=root[0],
                root_y=root[1],
                root_precision=root_precision,
                chain_length=chain_length,
                ntax=simulation.n_sites,
                nchar=simulation.n_features,
                jitter=0.,
                spherical=''
            )
        )


class Beast(object):

    def __init__(self):
        pass

    def set_locations(self, locations):
        """Define the locations of tips (and evlt. taxa) to used in BEAST.

        Args:
            locations (dict[str, np.array]):

        Returns:

        """
