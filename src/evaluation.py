#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon


def coverage(root, tree, p_hpd):
    # Ensure that root is a Point object
    if not isinstance(root, Point):
        assert len(root) == 2
        root = Point(root[0], root[1])

    # Check whether any of the HPD polygons () cover
    for polygon in tree.get_hpd(p_hpd):
        if polygon.contains(root):
            return True

    return False


# def check_root_in_hpd(tree_file_path, hpd, root, ax=None, alpha=0.15, color='teal', **plt_kwargs):
#     hpd_regex = 'location[12]_{hpd}%HPD_[0-9]+={{[0-9,.\\-]+}}'.format(hpd=hpd)
#     if not isinstance(root, Point):
#         assert len(root) == 2
#         root = Point(root[0], root[1])
#
#     with open(tree_file_path, 'r') as tree_file:
#         tree_str = tree_file.read()
#
#     # Find the part of the file describing the root info
#     root_str = tree_str.rpartition(')[&')[-1].rpartition('];')[0]
#     assert '(' not in root_str
#     assert ')' not in root_str
#
#     hpd_strs = re.findall(hpd_regex, root_str)
#
#     # No duplicates:
#     assert len(hpd_strs) == len(set(hpd_strs))
#
#     # Extract values for HPD polygons
#     hpds_x = []
#     hpds_y = []
#     for hpd_str in sorted(hpd_strs):
#         hpd_values = hpd_str.partition('{')[-1].partition('}')[0].split(',')
#         hpd = np.array([float(x) for x in hpd_values])
#
#         if hpd_str.startswith('location1'):
#             hpds_x.append(hpd)
#         else:
#             hpds_y.append(hpd)
#
#     # Create the polygons (and draw)
#     polygons = []
#     for xs, ys in zip(hpds_x, hpds_y):
#         polygons.append(Polygon(zip(xs, ys)))
#         a = Polygon(zip(xs, ys)).area
#
#         if ax:
#             ax.fill(xs, ys, alpha=alpha, color=color, **plt_kwargs)
#
#     # HPD covers root if any of the polygons covers it
#     success = any(poly.contains(root) for poly in polygons)
#
#     # Log some info in case of failure
#     if not success:
#         logging.info('Root not in HDP.')
#         logging.info(tree_str.rpartition(')[&')[-1])
#
#     return success