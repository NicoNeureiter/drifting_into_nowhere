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
END;'''


def write_nexus(features, path, fossils=None):
    features = np.asarray(features, dtype=int)
    n_societies, n_features = features.shape

    data_str = ''
    for i_society in range(n_societies):
        society_name = 'society_%i' % i_society
        fs = features[i_society]

        line = '\t\t' + society_name + '\t' + ''.join(map(str, fs)) + '\n'
        data_str += line

    nexus_str = NEXUS_TEMPLATE.format(
        n_societies = n_societies,
        n_features = n_features,
        symbols = '01',
        data = data_str
    )

    with open(path, 'w') as nexus_file:
        nexus_file.write(nexus_str)
