#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
import logging

from src.beast_xml_templates import *
from src.tree import Tree
from src.util import str_concat_array, extract_newick_from_nexus, SubprocessException, mkpath

BEAST_LOGGER_PATH = 'logs/beast.log'
mkpath(BEAST_LOGGER_PATH)

beast_logger = logging.getLogger('beast')
beast_logger.setLevel(logging.DEBUG)
beast_logger.addHandler(logging.FileHandler(BEAST_LOGGER_PATH))

beast_logger.info('='*100)
beast_logger.info('New Run')
beast_logger.info('='*100)

def write_nexus(simulation, path, fossils=None):
    data_str = ''

    for i, state in enumerate(simulation.societies):
        name = state.name
        fs = state.featureState.features.astype(int)

        line = '\t\t' + name + '\t' + str_concat_array(fs) + '\n'
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


def write_beast_xml(simulation, path, chain_length, fix_root=False, model='rrw'):
    if model == 'rrw':
        BEAST_XML_TEMPLATE = RRW_XML_TEMPLATE_PATH
    elif model == 'brownian':
        BEAST_XML_TEMPLATE = BROWNIAN_XML_TEMPLATE_PATH
    else:
        raise ValueError

    with open(BEAST_XML_TEMPLATE, 'r') as xml_template_file:
        xml_template = xml_template_file.read()

    locations_xml = ''
    features_xml = ''

    for i, state in enumerate(simulation.societies):
        name = state.name

        loc = state.geoState.location
        locations_xml += LOCATION_TEMPLATE.format(id=name, x=loc[0], y=loc[1])

        fs = state.featureState.alignment.astype(int)
        fs_str = str_concat_array(fs)
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


def load_tree_from_nexus(tree_path, location_key='location'):
    with open(tree_path, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Tree.from_newick(newick_str, location_key=location_key)

    return tree


def run_beast(working_dir):
    script_path = 'src/beast_scripts/beast.sh'
    working_dir = os.path.abspath(working_dir)

    bash_command = 'bash {script} {cwd}'.format(
        script=script_path, cwd=working_dir)

    ret = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    beast_logger.info(ret.stdout.decode())
    beast_logger.info(ret.stderr.decode())
    if ret.returncode != 0:
        raise SubprocessException


def run_treeannotator(hpd, burnin, working_dir):
    """Run treeannotator from the BEAST toolbox via bash script. Return the
    summary tree."""

    script_path = 'src/beast_scripts/treeannotator.sh'
    working_dir = os.path.abspath(working_dir)
    tree_path = os.path.join(working_dir, 'nowhere.tree')

    bash_command = 'bash {script} {hpd} {burnin} {cwd}'.format(
        script=script_path, hpd=hpd, burnin=burnin, cwd=working_dir)

    ret = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    beast_logger.info(ret.stdout.decode())
    beast_logger.info(ret.stderr.decode())
    if ret.returncode != 0:
        raise SubprocessException

    return load_tree_from_nexus(tree_path=tree_path)
