#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from matplotlib import pyplot as plt


def check_root_in_hpd(tree_file_path):
    with open(tree_file_path, 'r') as tree_file:
        tree_str = tree_file.read()

    root_str = tree_str.rpartition(')[&')[-1] \
                       .partition('];')[0]

    x_hpd_str = root_str.partition(r'location1_80%HPD_1={')[-1] \
                        .partition('}')[0]
    y_hpd_str = root_str.partition(r'location2_80%HPD_1={')[-1]\
                        .partition('}')[0]

    x_hpd = np.array([float(x) for x in x_hpd_str.split(',')])
    y_hpd = np.array([float(y) for y in y_hpd_str.split(',')])

    from shapely.geometry import Point, Polygon
    root = Point(0, 0)
    hpd = Polygon(zip(x_hpd, y_hpd))

    plt.fill(x_hpd, y_hpd, alpha=0.2)

    return hpd.contains(root)