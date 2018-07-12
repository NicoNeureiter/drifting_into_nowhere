#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import random

from src.beast_xml_templates import *

EPS = 1e-6

# Paths to template files
NEXUS_TEMPLATE_PATH = 'data/templates/nexus_template.nex'
LOCATIONS_TEMPLATE_PATH = 'data/templates/locations_template.txt'

XML_TEMPLATE_PATH = 'data/templates/beast_xml_template.xml'
FEATURES_TEMPLATE_PATH = 'data/templates/features_template.xml'
LOCATION_XML_TEMPLATE_PATH = 'data/templates/location_template.xml'
FOSSIL_TEMPLATE_PATH = 'data/templates/fossil_template.xml'
FOSSIL_CHILD_TEMPLATE_PATH = 'data/templates/fossil_child_template.xml'

GEOPRIOR_TEMPLATE = 'data/templates/geoprior_template.kml'


def write_nexus(simulation, path):
    with open(NEXUS_TEMPLATE_PATH, 'r') as nex_template_file:
        nexus_template = nex_template_file.read()

    data_str = ''
    for i, state in enumerate(simulation.societies):
        name = state.name
        fs = state.featureState.features.astype(int)

        line = '\t\t' + name + '\t' + ''.join(map(str, fs)) + '\n'
        data_str += line

    # for i, state in enumerate(simulation.history):
    #     name = state.name
    #     fs = '?' * simulation.n_features
    #     line = '\t\t%s\t%s\n' % (name, fs)
    #     data_str += line

    tree = simulation.get_newick_tree()

    nexus_str = nexus_template.format(
        n_societies=simulation.n_sites,  # + len(simulation.history),
        n_features=simulation.n_features,
        symbols='01',
        data=data_str,
        tree=tree
    )

    with open(path, 'w') as nexus_file:
        nexus_file.write(nexus_str)


def write_locations(simulation, path):
    with open(LOCATIONS_TEMPLATE_PATH, 'r') as locations_template_file:
        locations_template = locations_template_file.read()

    rows = []
    for i, state in enumerate(simulation.societies):
        name = state.name
        loc = state.geoState.location

        row = '%s\t%f\t%f' % (name, loc[0], loc[1])
        rows.append(row)

    data_str = '\n'.join(rows)

    locations_str = locations_template.format(data=data_str)

    with open(path, 'w') as locations_file:
        locations_file.write(locations_str)


def get_descendant_mapping(state):
    if not state.children:
        return {}, [state]

    mapping = {}
    descendants = []

    for c in state.children:
        mapping_c, descendants_c = get_descendant_mapping(c)
        mapping.update(mapping_c)
        descendants += descendants_c

    mapping[state] = descendants
    return mapping, descendants


def write_beast_xml(simulation, path, root_known=True,
                    chain_length=100000):
    """Write the simulated languages to an XML file, to be used for analysis in BEAST.

    Args:
        simulation (Simulation): The simulation object (should already have been run).
        path (str): The path to write the XML file to.
        use_fossils (bool): Whether to add information about all hidden states
            TODO Not all! How to select?
        chain_length (int): Length of BEASTs MCMC chain (number of steps).
    """
    with open(XML_TEMPLATE_PATH, 'r') as xml_template_file:
        xml_template = xml_template_file.read()
    with open(FEATURES_TEMPLATE_PATH, 'r') as features_template_file:
        features_template = features_template_file.read()
    with open(LOCATION_XML_TEMPLATE_PATH, 'r') as locations_template_file:
        location_template = locations_template_file.read()
    with open(FOSSIL_TEMPLATE_PATH, 'r') as fossil_template_file:
        fossil_template = fossil_template_file.read()
    with open(FOSSIL_CHILD_TEMPLATE_PATH, 'r') as fossil_child_template_file:
        fossil_child_template = fossil_child_template_file.read()

    locations_xml = ''
    features_xml = ''
    leaf_traits_xml = ''
    fossils_xml = ''
    geopriors_xml = ''
    monophyly_xml = ''
    geoprior_refs_xml = ''

    for i, state in enumerate(simulation.societies):
        name = state.name

        loc = state.geoState.location
        locations_xml += location_template.format(id=name, x=loc[0], y=loc[1])

        fs = state.featureState.features.astype(int)
        fs_str = ''.join(map(str, fs))
        features_xml += features_template.format(id=name, features=fs_str)

        leaf_traits_xml += LEAF_TRAIT.format(id=name)

        kml_path = 'data/beast/kml/{id}.kml'.format(id=name)
        write_geoprior_kml(state, kml_path)
        geopriors_xml += GEO_PRIOR.format(id=name, kml_path=kml_path)
        geoprior_refs_xml += GEO_PRIOR_REF.format(id=name)

    # Add the monophyly constraints for ancestors to the xml and add the geoprior (if requested)
    descendant_mapping, _ = get_descendant_mapping(simulation.root)
    for state, descendants in descendant_mapping.items():
        name = state.name
        loc = state.geoState.location

        descendants_xml = ''
        for desc in descendants:
            descendants_xml += fossil_child_template.format(id=desc.name)

        fossils_xml += fossil_template.format(id=name, x=loc[0], y=loc[1],
                                              descendants=descendants_xml)

        monophyly_xml += MONOPHYLY_STATISTIC.format(id=name)

        # if name != 'fossil_':
        #     # Declare fossil location (as geo prior)
        #     kml_path = 'data/beast/kml/{id}.kml'.format(id=name)
        #     write_geoprior_kml(state, kml_path)
        #     geopriors_xml += GEO_PRIOR.format(id=name, kml_path=kml_path)
        #     int_traits_xml += INTERNAL_TRAIT.format(id=name)


    tree_xml = simulation.get_newick_tree()

    root_x, root_y = simulation.root.geoState.location
    if root_known:
        root_prec = 1e6
    else:
        root_prec = 1e-6


    with open(path, 'w') as beast_xml_file:
        beast_xml_file.write(
            xml_template.format(
                locations=locations_xml,
                features=features_xml,
                fossils=fossils_xml,
                tree=tree_xml,
                leaf_traits=leaf_traits_xml,
                monophyly=monophyly_xml,
                geopriors=geopriors_xml,
                geoprior_references=geoprior_refs_xml,
                root_x=root_x,
                root_y=root_y,
                root_precision=root_prec,
                chain_length=chain_length,
                ntax=simulation.n_sites,
                nchar=simulation.n_features
            )
        )


def write_geoprior_kml(state, path):
    with open(GEOPRIOR_TEMPLATE, 'r') as kml_template_file:
        kml_template = kml_template_file.read()

    x, y = state.geoState.location

    kml_str = kml_template.format(
        id=state.name,
        left=x - EPS,
        right=x + EPS,
        top=y + EPS,
        bottom=y - EPS
    )

    with open(path, 'w') as kml_file:
        kml_file.write(kml_str)
