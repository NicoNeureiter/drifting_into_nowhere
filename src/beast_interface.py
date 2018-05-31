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


def write_nexus(simulation, features, path, fossils=None):
    features = np.asarray(features, dtype=int)
    n_societies, n_features = features.shape

    data_str = ''
    for i_society in range(n_societies):
        society_name = 'society_%i' % i_society
        fs = features[i_society]

        line = '\t\t' + society_name + '\t' + ''.join(map(str, fs)) + '\n'
        data_str += line

    tree = simulation.get_newick_tree()

    nexus_str = NEXUS_TEMPLATE.format(
        n_societies = n_societies,
        n_features = n_features,
        symbols = '01',
        data = data_str,
        tree = tree
    )

    with open(path, 'w') as nexus_file:
        nexus_file.write(nexus_str)


LOCATIONS_TAMPLATE = 'traits\tx\ty\n{data}'

def write_locations(locations, path):
    print(locations.shape)

    rows = []
    for i, loc in enumerate(locations):
        name = 'society_%i' % i
        row = '%s\t%f\t%f' % (name, locations[i, 0], locations[i, 1])
        rows.append(row)

    data_str = '\n'.join(rows)

    locations_str = LOCATIONS_TAMPLATE.format(data=data_str)

    with open(path, 'w') as locations_file:
        locations_file.write(locations_str)
