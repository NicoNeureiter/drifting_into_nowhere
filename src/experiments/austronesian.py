#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides an interface to language family trees extracted by the
lgfam-newick package (https://github.com/ddediu/lgfam-newick). The lgfam-newick
package collects tree topologies from different databases and estimates branch-
lengths using different methods and data sources. The resulting newick trees can
be loaded into our custom Tree class using the functions below.

To prepare the lgfam database checkout the github repository above and copy the
output directory to the desired path. This path needs to be provided to
load_newick_from_lgfam() as a base_path (per default the output directory is expected to
be in the 'data/' folder and renamed to 'dediu_forest/').
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import logging

import pandas as pd
import numpy as np
import newick

from src.tree import Tree, get_edge_heights
from src.util import extract_newick_from_nexus

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_newick_from_lgfam(family_name, tree_source='autotyp', branch_length_method='nnls',
                           branch_length_data='asjp16', base_path='data/dediu_forest/'):
    """Load a tree of the language family `family_name` from the Dediu forest
    data set.

    Args:
        family_name (str): The name (string identifier) of the language family.

    Kwargs:
        tree_source (str): Which database the tree is taken from.
            Options: ['autotyp', 'ethnologue', 'glotolog', 'wals']
        branch_length_method (str): Method used to compute the branch lengths.
            Options: ['nnls', 'ga', 'nj']
        branch_length_method (str): Data used for computing the branch lengths.
            Options: ['asjp16', 'wals(gower)', 'autotyp', 'mg2015(autotyp)',...]
        base_path (str): Path to the lgfam output directory.

    Returns:
        str: The language family tree as a newick string.

    Resources:
        https://github.com/ddediu/lgfam-newick/blob/master/paper/family-trees-with-brlength.pdf
    """
    filename = '{tree_source}-newick-{method}+{data}.csv'.format(
        tree_source=tree_source, method=branch_length_method, data=branch_length_data)
    csv_path = os.path.join(base_path, tree_source, filename)

    # Read the tree from csv
    all_families = pd.read_csv(csv_path, sep='\t', header=0, index_col=0)
    family_row = all_families.loc[family_name, :]
    nwk_str = family_row.Tree

    # Some trees did not converge...
    if not family_row.Success:
        raise ValueError('No Newick tree availabe under this configuration!')

    # Parse Newick tree into Tree object
    nwk_str = prepare_newick(nwk_str)
    tree = Tree.from_newick(nwk_str)

    return tree


def load_newick_from_forestML(family_name, tree_source='ASJP+Glottolog',
                              base_path='data/dediu_forest/'):
    filename = 'forestML.csv'
    csv_path = os.path.join(base_path, filename)

    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            t_name, _, line = line.partition(',')
            t_number, _, line = line.partition(',')
            t_source, _, line = line.partition(',')
            bl_method, _, line = line.partition(',')
            t_set, _, line = line.partition(',')
            tree, _, _ = line.partition(';')

            if t_name == family_name and t_source == tree_source:
                return Tree.from_newick(tree)

    raise ValueError('No tree found for language family_name=`%s` and '
                     'tree_source=`%s`' % (family_name, tree_source))


def load_bayesian(family_name):
    code_mapping_path = 'data/%s_glottocodes.csv' % family_name
    code_mapping = pd.read_csv(code_mapping_path, sep=',', header=0, index_col=False)
    code_mapping.loc[:, 'taxon'] = code_mapping.loc[:, 'taxon'].str.lower()
    code_mapping = code_mapping.set_index('taxon')

    with open('data/%s.trees' % family_name, 'r') as nexus_file:
        nwk_str = extract_newick_from_nexus(nexus_file.read())

    tree = Tree.from_newick(nwk_str)
    for node in tree.iter_leafs():
        if node.name in code_mapping.index:
            node.attributes['Glottolog'] = code_mapping.loc[node.name].glottocode
        else:
            logging.warning('No Glottocode found for node `%s`' % node.name)

    return tree


def load_locations_from_lgfam(code='AUTOTYP', base_path='data/dediu_forest/'):
    """Load the locations for every language in the lgfam database into a pandas
    dataframe, indexed by 'index_lang_code'.

    Kwargs:
        index_lang_code (str): The language code standard to be used as an index.
        base_path (str): Path to the lgfam output directory.

    Returns:
        pd.DataFrame: Dataframe conatining the locations.
    """
    filename = 'code_mappings_iso_wals_autotyp_glottolog.csv'
    csv_path = os.path.join(base_path, filename)

    all_languages = pd.read_csv(csv_path, sep='\t', header=0, index_col=False)
    all_languages = all_languages.set_index(code)

    return all_languages  #.loc[:, ['Latitude', 'Longitude']]


def parse_lgfam_newick(tree, locations=None, code='AUTOTYP'):
    # Load locations
    locations = load_locations_from_lgfam(code=code)

    for node in tree.iter_descendants():
        if code in node.attributes:
            node.name += node.attributes[code]

    # Add the locations to the nodes
    for node in tree.iter_leafs():
        lang_code = node.attributes[code]

        if lang_code in locations.index:
            node._location = locations.loc[lang_code, ['Latitude', 'Longitude']]
            node._location = node._location.values.astype(float)

            if len(node._location.shape) == 2:
                # TODO how should we handle this?
                logging.warning('Multiple locaitons found for node %s (%s: %s)',
                                node.name, code, lang_code)
                node._location = np.mean(node._location, axis=0)

            assert node._location.shape == (2,)
        # else:
            # logging.warning('No location found for lanugage: %s' % lang_code)

    tree.binarize()

    return tree

def multi_replace(text, rules):
    """Replace all substrings of s occuring in rules.keys() by the corresponding
    rules.values().

    Args:
        text (str): Original string.
        rules (dict): Rules dict, mapping original patter to the values they
            should be replaced by.

    Returns:
        str: Resulting string.

    """
    import re

    rules = dict((re.escape(k), v) for k, v in rules.iteritems())
    pattern = re.compile("|".join(rules.keys()))
    return pattern.sub(lambda m: rules[re.escape(m.group(0))], text)


def prepare_newick(nwk_str):
    # Some replacements (no whitespace, attribute style lang codes,...)
    nwk_str = nwk_str.replace('\'', '') \
                     .replace(' ', '') \
                     .replace('[i-', '[ISO=') \
                     .replace('[w-', '[WALS=') \
                     .replace('[a-', '[AUTOTYP=') \
                     .replace('[g-', '[Glottolog=') \
                     .replace('][', ',') \
                     .replace('[', '[&')

    # Cut of outer node attributes and add 0 branch length
    # nwk_str = ''.join(nwk_str.rpartition(')')[:-1]) + ':0'
    # nwk_str = nwk_str[:-1] + ':0'
    nwk_str += ';'

    return nwk_str


def shift_longitudes(tree):
    for node in tree.iter_leafs():
        node.location[0] = node.location[0] % 360 - 180


def filter_invalid_locations(tree):
    invalid = []
    for node in tree.iter_leafs():
        if node.location is None:
            invalid.append(node.name)
        elif not np.all(np.isfinite(node.location)):
            invalid.append(node.name)

    tree.remove_nodes_by_name(invalid)


def swap_xy(tree):
    for node in tree.iter_descendants():
        if node.location is not None:
            node.location = node.location[::-1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import geopandas as gpd

    from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
    from src.plotting import (plot_hpd, plot_root, plot_tree, plot_height,
                              plot_clades, plot_tree_topology)
    from src.tree import tree_imbalance
    from src.util import mkpath, grey
    from src.analyze_tree import flatten
    from workbench.map_projections import WorlMap

    CHAIN_LENGTH = 250000
    BURNIN = 3000
    HPD = 80

    SOURCE = 'glottolog'
    CODE = 'Glottolog'
    FAMILY_NAME = 'austronesian'

    # Paths to experiment files
    WORKING_DIR = 'experiments/%s-%s/' % (FAMILY_NAME, SOURCE)
    mkpath(WORKING_DIR)
    XML_PATH = os.path.join(WORKING_DIR, 'nowhere.xml')
    GEO_TREE_PATH = WORKING_DIR + 'nowhere.tree'

    # Plotting cfg
    LW = 0.1
    cmap = plt.get_cmap('viridis')

    # # Load and plot the background map
    # world = gpd.read_file('data/naturalearth_50m_wgs84.geojson')
    # world = world.translate(xoff=-180)
    # ax = world.plot(color=grey(.95), edgecolor=grey(.5), lw=.4)
    # world = world.translate(xoff=360)
    # ax = world.plot(color=grey(.95), edgecolor=grey(.5), lw=.4, ax=ax)

    # tree = load_newick_from_lgfam(FAMILY_NAME, tree_source=SOURCE)
    tree = load_bayesian(FAMILY_NAME)
    print(tree.tree_size())
    tree.binarize()
    tree = parse_lgfam_newick(tree, code=CODE)

    # Prepare tree
    filter_invalid_locations(tree)
    swap_xy(tree)

    # Prepare map and project locations
    locations = tree.get_leaf_locations()
    # plt.hist(locations[:, 0])
    # plt.show()
    world_map = WorlMap()
    world_map.align_to_data(locations)
    ax = world_map.plot()

    for node in tree.iter_leafs():
        node.location = world_map.project(node.location)

    if True:
        # Write to xml
        tree.write_beast_xml(XML_PATH, CHAIN_LENGTH, diffusion_on_a_sphere=True,
                             movement_model='rrw', adapt_tree=False,
                             adapt_height=False, jitter=0.02)
        # Run the BEAST analysis
        run_beast(WORKING_DIR)
        run_treeannotator(HPD, BURNIN, WORKING_DIR)

    # Load the tree and plot it
    tree = load_tree_from_nexus(GEO_TREE_PATH)
    plot_tree(tree, lw=1., cmap=cmap, color_fun=get_edge_heights)  #, alpha_fun=flatten(get_edge_heights, 0.5))
    # plot_backbone_splits(tree, mode='geo', plot_edges=True, n_clades=12)

    # # Prepare tree
    # swap_xy(tree)
    # plt.hist(tree.get_leaf_locations()[:, 0])
    # plt.show()

    plot_clades(tree, max_clade_size=50)
    plot_root(tree.location)
    plot_hpd(tree, HPD, projection=world_map.project)

    # plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()
    exit()

    plot_tree_topology(tree)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()
