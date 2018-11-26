#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import numpy as np
import matplotlib.pyplot as plt

from src.tree import Node
from src.util import mkpath, extract_newick_from_nexus

NEWICK_TREE_PATH = 'data/ie/IE_2MCCs.tree'
ALIGNMENT_PATH = 'data/ie/ie_alignment.csv'
LOCATION_KEY = 'trait'


def write_ie_xml(xml_path, chain_length):
    with open(NEWICK_TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)

    tree = Node.from_newick(newick_str, location_key=LOCATION_KEY)

    tree = tree.get_subtree([0, 0, 0, 0])

    tree.write_beast_xml(xml_path, chain_length, root=None)


def column_agreement(arr):
    positive_agreement = (1-arr).dot(arr.T)
    negative_agreement = arr.dot((1 - arr).T)
    return positive_agreement + negative_agreement

def norm_2d(array):
    """L2 vector norm for an array of 2d vectors."""
    return (array[:, 0] ** 2 + array[:, 1] ** 2) ** .5

def plot_lingo_geo_distances():
    with open(NEWICK_TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)

    tree = Node.from_newick(newick_str, location_key=LOCATION_KEY)

    tree.load_alignment_from_csv(ALIGNMENT_PATH)

    feat_lingo = []
    feat_geo = []
    for node in tree.iter_leafs():
        a = np.array([0.5 if c == '?' else float(c) for c in node.alignment])
        feat_lingo.append(a)
        feat_geo.append(node.location)

    feat_lingo = np.array(feat_lingo)
    feat_geo = np.array(feat_geo)

    print(len(a))
    print(feat_lingo.shape)

    # dist_lingo = feat_lingo.dot(feat_lingo.T)
    dist_lingo = column_agreement(feat_lingo)
    print('-------------------------------------')
    print(feat_geo.shape)
    print(feat_geo[:, np.newaxis].shape)
    diff_geo = feat_geo - feat_geo[:, np.newaxis]
    dist_geo = np.hypot(*diff_geo.T)
    print(diff_geo.shape)
    print(diff_geo.T.shape)
    print('-------------------------------------')

    plt.scatter(np.log1p(dist_geo.flatten()), dist_lingo.flatten(), s=2., alpha=.8, lw=0)
    plt.show()


if __name__ == '__main__':
    CHAIN_LENGTH = 200000
    BURNIN = 5000
    HPD = 80

    SCRIPT_PATH = 'src/beast_pipeline.sh'
    IE_XML_PATH = 'data/ie/nowhere.xml'
    GEOJSON_PATH = 'world.geojson'
    GEO_TREE_PATH = 'data/ie/nowhere.tree'

    mkpath(IE_XML_PATH)

    # plot_lingo_geo_distances()

    write_ie_xml(IE_XML_PATH, CHAIN_LENGTH)

    # Run the BEAST analysis + summary of results (treeannotator)
    os.system('bash {script} {hpd} {burnin} {cwd} {geojson}'.format(
        script=SCRIPT_PATH,
        hpd=HPD,
        burnin=BURNIN,
        cwd=os.getcwd()+'/data/ie/',
        geojson=GEOJSON_PATH
    ))

    # # Evaluate the results
    # okcool = check_root_in_hpd(GEO_TREE_PATH, HPD, root=IE_ROOT)
    # print('\n\nOk cool: %r' % okcool)
