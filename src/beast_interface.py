#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
import logging
import time

from src.beast_xml_templates import *
from src.tree import Tree
from src.util import str_concat_array, extract_newick_from_nexus, SubprocessException, mkpath

BEAST_LOGGER_PATH = 'logs/beast.log'
mkpath(BEAST_LOGGER_PATH)

EXPERIMENT_LOGGER = logging.getLogger('experiment')

BEAST_LOGGER = logging.getLogger('beast')
BEAST_LOGGER.setLevel(logging.DEBUG)
BEAST_LOGGER.addHandler(logging.FileHandler(BEAST_LOGGER_PATH))
BEAST_LOGGER.info('=' * 100)
BEAST_LOGGER.info('New Run')
BEAST_LOGGER.info('=' * 100)

def write_nexus(simulation, path, fossils=None):
    data_str = ''

    for i, state in enumerate(simulation.societies):
        name = state.name
        fs = state.featureState.features.astype(int)

        line = '\t\t' + name + '\t' + str_concat_array(fs) + '\n'
        data_str += line

    tree = simulation.get_newick_tree()

    nexus_str = NEXUS_TEMPLATE.format(
        n_societies = simulation.n_sites,
        n_features = simulation.n_features,
        symbols = '01',
        data = data_str,
        tree = tree)

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

def read_translation_table(nexus_path):

    with open(nexus_path, 'r') as tree_file:
        for line in tree_file:
            if line.lower().strip() == 'translate':
                break
        else:
            return None

        name_mapping = {}
        stop = False
        for line in tree_file:
            line = line.strip()
            if line == ';':
                return name_mapping

            if line.endswith(';'):
                stop = True
            line = line.strip(';').strip(',')
            key, value = line.split()
            name_mapping[key] = value

            if stop:
                return name_mapping


def load_tree_from_nexus(tree_path, location_key='location',
                         use_translation_table=False, name_mapping=None):
    if use_translation_table == True:
        name_mapping = read_translation_table(tree_path)
        print(name_mapping)

    with open(tree_path, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Tree.from_newick(newick_str, location_key=location_key,
                                translate=name_mapping)

    return tree


def run_beast(working_dir):
    script_path = 'src/beast_scripts/beast.sh'
    working_dir = os.path.abspath(working_dir)
    bash_command = 'bash {script} {cwd}'.format(
        script=script_path, cwd=working_dir)

    t0 = time.time()
    ret = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    EXPERIMENT_LOGGER.info('\tBEAST Runtime: %.2f' % (time.time() - t0))
    BEAST_LOGGER.info(ret.stdout.decode())
    BEAST_LOGGER.info(ret.stderr.decode())
    if ret.returncode != 0:
        raise SubprocessException


def run_treeannotator(hpd, burnin, working_dir,
                      trees_fname='nowhere.trees', mcc_fname='nowhere.tree'):
    """Run treeannotator from the BEAST toolbox via bash script. Return the
    summary tree."""

    script_path = 'src/beast_scripts/treeannotator.sh'
    working_dir = os.path.abspath(working_dir)
    tree_path = os.path.join(working_dir, 'nowhere.tree')

    bash_command = 'bash {script} {hpd} {burnin} {cwd} {trees_file} {mcc_file}'.format(
        script=script_path, hpd=hpd, burnin=burnin, cwd=working_dir,
        trees_file=trees_fname, mcc_file=mcc_fname
    )

    ret = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    BEAST_LOGGER.info(ret.stdout.decode())
    BEAST_LOGGER.info(ret.stderr.decode())
    if ret.returncode != 0:
        raise SubprocessException

    return load_tree_from_nexus(tree_path=tree_path)


def read_nexus_name_mapping(nexus_str):
    nexus_lines = nexus_str.split('\n')

    for i, line in enumerate(nexus_lines):
        line = line.strip()
        if line.lower() == 'translate':
            break

    name_map = {}
    for line in nexus_lines[i+1:]:
        line = line.strip().strip(',')
        if line == ';' or line.lower().startswith('tree'):
            return name_map
        if line.endswith(';'):
            line = line[:-1]
            id, name = line.split()
            name_map[id] = name
            return name_map

        id, name = line.split()
        name_map[id] = name
    return name_map
    #
    # next(nexus_lines)
    # next(nexus_lines)
    # name_map = {}
    # for line in nexus_lines:
    #     line = line.strip()[:-1]
    #     if line.startswith('tree'):
    #         return name_map
    #
    #     id, name = line.split()
    #     name_map[id] = name
    #
    # return name_map

def load_trees(tree_path, read_name_mapping=False):
    with open(tree_path, 'r') as tree_file:
        nexus_str = tree_file.read().lower().strip()

    if read_name_mapping:
        name_map = read_nexus_name_mapping(nexus_str)
    else:
        name_map = None

    trees = []
    for line in nexus_str.split('\n'):
        if line.strip().startswith('tree '):
            newick_str = line.split(' = ')[-1]
            if newick_str.startswith(r'[&r] '):
                newick_str = newick_str[len(r'[&r] '):]

            newick_str = newick_str.replace(']:[',',')
            newick_str = newick_str.replace(']',']:')
            newick_str = newick_str.replace(':;', ';')

            tree = Tree.from_newick(newick_str, translate=name_map)
            trees.append(tree)

    # print([(t.location, t.attributes) for t in trees])
    return trees