#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np


def bounding_box(points, margin=None):
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    if margin is not None:
        width = x_max - x_min
        height = y_max - y_min

        x_min -= margin * width
        x_max += margin * width
        y_min -= margin * height
        y_max += margin * height

    return x_min, y_min, x_max, y_max


def newick_tree(state):
    if state.children:
        subtrees = ', '.join(map(newick_tree, state.children))
        # return '(%s)h%s:%.1f' % (subtrees, state.name, state.age)
        # return '(%s)[&label="%s"]:%.1f' % (subtrees, 'h'+state.name, state.age)
        return '(%s):%.1f' % (subtrees, state.age)
    else:
        return '%s:%.1f' % (state.name, state.age)