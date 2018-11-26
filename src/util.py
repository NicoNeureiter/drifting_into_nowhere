#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import csv
import logging
import random as _random
import pickle

import numpy as np


def dump(data, path):
    """Dump the given data to the given path (using pickle)."""
    mkpath(path)
    with open(path, 'wb') as dump_file:
        pickle.dump(data, dump_file)


def load_from(path):
    """Load and return data from the given path (using pickle)."""
    with open(path, 'rb') as dump_file:
        return pickle.load(dump_file)


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


def read_locations_file(locations_path, delimiter='\t', swap_xy=False,
                        skip_first_row=True):
    locations = {}
    location_missing = []

    with open(locations_path, 'r') as loc_file:
        loc_reader = csv.reader(loc_file, delimiter=delimiter)

        if skip_first_row:
            next(loc_reader)

        # Read file
        for line in loc_reader:
            name = line[0]
            try:
                x = float(line[1])
                y = float(line[2])
                if swap_xy:
                    x, y = y, x
            except ValueError:
                logging.warning('No location provided for society: %s' % name)
                location_missing.append(name)

            # TODO project here ?

            locations[name] = np.array([x, y])

    return locations, location_missing


def read_alignment_file(alignment_path, delimiter='\t', skip_first_row=True):
    sequences = {}

    with open(alignment_path, 'r') as loc_file:
        alignment_reader = csv.reader(loc_file, delimiter=delimiter)

        if skip_first_row:
            next(alignment_reader)

        # Read file
        for line in alignment_reader:
            name, seq = line
            sequences[name] = seq

    return sequences


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
    return x.dot(x) ** 0.5


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


def grey(v):
    return (v, v, v)


def total_drift_2_step_drift(total_drift, n_steps, drift_density=1.):
    return total_drift / (n_steps * drift_density)


def total_diffusion_2_step_var(total_diffusion, n_steps):
    return total_diffusion ** 2 / n_steps


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class StringTemplate(object):

    def __init__(self, template_string):
        super(StringTemplate, self).__setattr__('template_string', template_string)
        super(StringTemplate, self).__setattr__('format_dict', {})

    def __setattr__(self, key, value):
        # super(StringTemplate, self).__setattr__(key, value)
        self.set_value(key, value)

    def set_value(self, key, value):
        self.format_dict[key] = value

    def set_values(self, **fill_values):
        self.format_dict.update(fill_values)

    def fill(self):
        return self.template_string.format(**self.format_dict)

    def __str__(self):
        return self.fill()


def str_concat_array(a):
    return ''.join(map(str, a))


def extract_tree_line(nexus):
    for line in nexus.split('\n'):
        line = line.strip()
        if line.startswith('tree '):
            return line


def extract_newick_from_nexus(nexus):
    tree_line = extract_tree_line(nexus)
    _, tree_name, _, _, newick = tree_line.split()

    return newick


class SubprocessException(Exception):
    pass