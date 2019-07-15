#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import sys
import csv
import logging
import random
import random as _random
import pickle
import shutil
import datetime

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

    """A class for StringTemplates, where fields can be filled one after another
     in contrast to str.format() where all fields are filled at once. The class
     also provides a little syntactic sugar to set template fields as attributes.

    Attributes:
        template_string (str): String template to be filled with attributes.
        format_dict (dict): Dictionary containing all set attributes.
    """

    def __init__(self, template_string):

        # Since we override StringTemplate.__setattr__(), we need to explicitly
        # use object.__setattr__() to define attributes.
        super(StringTemplate, self).__setattr__('template_string', template_string)
        super(StringTemplate, self).__setattr__('format_dict', {})

    def __setattr__(self, key, value):
        self.set_value(key, value)

    def set_value(self, key, value):
        self.format_dict[key] = value

    def set_values(self, **fill_values):
        self.format_dict.update(fill_values)

    def fill(self):
        # print()
        # print('\n'.join(self.format_dict.keys()))
        # print()
        return self.template_string.format(**self.format_dict)

    def __str__(self):
        return self.fill()


def str_concat_array(a):
    return ''.join(map(str, a))


def extract_tree_line(nexus):
    for line in nexus.split('\n'):
        line = str.lower(line.strip())
        if line.startswith('tree '):
            return line


def extract_newick_from_nexus(nexus):
    tree_line = extract_tree_line(nexus)
    return tree_line.split()[-1]


def transform_tree_coordinates(tree, trafo):
    for node in tree.iter_descendants():
        node.location = trafo(node)


def time_drift_trafo(node):
    x, y = node.location
    # drft = 0.877 * x - 0.479 * y
    # return x, -node.height
    return node.height, y  # + 0.2*x


def unit_vector(rad):
    return np.array([
        np.cos(rad),
        np.sin(rad)
    ])


def mean_angle(radians):
    vectors = unit_vector(radians)
    vector_sum = np.sum(vectors, axis=1)
    return np.arctan2(vector_sum[1], vector_sum[0])


def deg2rad(deg):
    return np.pi * deg / 180.


def rad2deg(rad):
    return 180. * rad / np.pi


class SubprocessException(Exception):
    pass


def experiment_preperations(work_dir):
    # Ensure working directory exists
    now = datetime.datetime.now()
    exp_dir = os.path.join(work_dir, 'experiment_logs_%s/' % now)
    mkpath(exp_dir)

    # Safe state of current file and config to the experiment folder
    base_file = sys.argv[0]
    shutil.copy(base_file, exp_dir)
    shutil.copy(__file__, exp_dir)
    shutil.copy('src/config.py', exp_dir)

    # Generate random seed
    seed = random.randint(0, 1e9)
    # seed = 773565758
    # seed = 333036130
    # seed = 549156425

    # Set it in built-in random and numpy.random
    random.seed(seed)
    np.random.seed(seed)

    # Print and write the seed to a file.
    print('Random seed:', seed)
    with open(os.path.join(exp_dir, 'seed'), 'w') as seed_file:
        seed_file.write(str(seed))

    return exp_dir


def birth_death_expectation(birth_rate, death_rate, n_steps, vrange=None):

    N_SAMPLES = 1000
    samples = np.ones(N_SAMPLES, dtype=int)
    for _ in range(n_steps):
        new_samples = np.copy(samples)
        new_samples += np.random.binomial(samples, birth_rate)
        new_samples -= np.random.binomial(samples, death_rate)
        samples = new_samples

    print('P total extinction: %.2f' % np.mean(samples == 0))
    print('P too small: %.2f' % np.mean(samples < vrange[0]))
    print('P too big: %.2f' % np.mean(samples > vrange[1]))

    if vrange is not None:
        samples = np.array([s for s in samples if vrange[0] <= s <= vrange[1]])

    print('P out of range: %.2f' % (1 - len(samples) / N_SAMPLES))
    print('Expected leafs: %.2f' % np.mean(samples))
    print('Standard dev. leafs: %.2f' % np.std(samples))
    return np.mean(samples)