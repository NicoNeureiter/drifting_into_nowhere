#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import csv
import logging
import random as _random

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
        subtrees = ','.join(map(newick_tree, state.children))
        # return '(%s)h%s:%.1f' % (subtrees, state.name, state.length)
        # return '(%s)[&label="%s"]:%.1f' % (subtrees, 'h'+state.name, state.length)
        return '(%s):%.1f' % (subtrees, state.length)
    else:
        return '%s:%.1f' % (state.name, state.length)


def read_locations_file(locations_path, delimiter='\t'):
    locations = {}
    location_missing = []

    with open(locations_path, 'r') as loc_file:
        loc_reader = csv.reader(loc_file, delimiter=delimiter)
        # skip header row
        next(loc_reader)

        # Read file
        for line in loc_reader:
            name = line[0]
            try:
                lat = float(line[1])
                long = float(line[2])
            except ValueError:
                logging.warning('No location provided for society: %s' % name)
                location_missing.append(name)


            # TODO project here ?

            locations[name] = np.array([lat, long])

    return locations, location_missing


def remove_whitespace(s):
    return ''.join(s.split())


def find(s, pattern):
    idx = s.find(pattern)

    if idx == -1:
        return len(s)
    else:
        return idx


def norm(x):
    x = np.asarray(x)
    return x.dot(x)**0.5


def normalize(x):
    x = np.asarray(x)
    return x / norm(x)


def dist(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return norm(x - y)


def bernoulli(p, size=None):
    if size is None and np.isscalar(p):
        return _random.random() < p
    else:
        return np.random.binomial(1, p=p, size=size).astype(bool)