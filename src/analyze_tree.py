#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
from math import atan2

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from src.tree import Node
from src.util import norm, normalize, dist
from src.plotting import circular_histogram


def extract_tree_line(nexus):
    for line in nexus.split('\n'):
        line = line.strip()
        if line.startswith('tree '):
            return line

def extract_newick_from_nexus(nexus):
    tree_line = extract_tree_line(nexus)
    _, tree_name, _, _, newick = tree_line.split()

    print('Extracted tree: %s' % tree_name)

    return newick

def plot_tree(node: Node, color_fun):
    x, y = node.location
    # plt.scatter([x], [y], c='k', lw=0, alpha=0.5)

    for c in node.children:
        cx, cy = c.location
        color, alpha = color_fun(node, c)

        # plt.plot([x, cx], [y, cy], c=color, alpha=alpha)
        plt.arrow(x, y, (cx-x), (cy-y), color=color, alpha=alpha, width=0.07,
                  length_includes_head=True)
        plot_tree(c, color_fun)


diffrates = []
def color_diffusion_rate(parent, child):
    pxy = parent.location
    cxy = child.location

    dt = child.length

    # cm = plt.get_cmap('plasma')
    cm = plt.get_cmap('viridis')

    rate = dist(pxy, cxy) / dt
    v = 1.1 + 0.15*np.log(rate)

    diffrates.append(v)
    v = np.clip(v, 0., 1.)
    color = cm(1-v)

    height = (parent['height'] + child['height']) / 2.
    a = np.clip(height / 500., 0, 1)

    return color, a


# def plot_diffusion_rate(edges):
#     diff_rates = []
#     xs = []
#     ys = []
#     for p, c in edges:


def angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def plot_eary_late_drift(tree: Node):
    halftime = tree['height'] / 2

    angles_early = []
    lengths_early = []
    angles_late = []
    lengths_late = []
    for parent, child in tree.get_edges():
        step = child.location - parent.location

        length = norm(step)
        angle = atan2(step[1], step[0])
        height = (parent['height'] + child['height']) / 2.

        if height > halftime:
            angles_early.append(angle)
            lengths_early.append(length)
        else:
            angles_late.append(angle)
            lengths_late.append(length)

    # angles_early += list(np.random.random(64000)*np.pi*2)
    nbins = 16
    ax = circular_histogram(angles_early, bins=nbins, normed=1, color=(0.1, 0.6, 0.75),
                            weights=lengths_early, double_hist=True,
                            label='Step direction - earlier half of the expansion'
                            )
    ax = circular_histogram(angles_late, bins=nbins, normed=1, color=(0.5, 0.1, 0.4),
                            weights=lengths_late, double_hist=True,
                            label='Step direction - later half of the expansion',
                            ax=ax)

    return ax


def plot_global_local_drift(tree: Node):
    angles = []
    lengths = []
    angles_from_root = []
    lengths_from_root = []

    for parent, child in tree.get_edges():
        step = child.location - parent.location
        lengths.append(norm(step))
        angles.append(atan2(step[1], step[0]))

        step_from_root = child.location - tree.location
        lengths_from_root.append(norm(step_from_root))
        angles_from_root.append(atan2(step_from_root[1], step_from_root[0]))

    nbins = 16
    ax = circular_histogram(angles_from_root, bins=nbins, normed=1, color=(0.65, 0.2, 0.6),
                            weights=lengths_from_root,
                            double_hist=True,
                            label='Directions from root (global drift)'
                            )
    ax = circular_histogram(angles, bins=nbins, normed=1, color=(0.25, 0.7, 0.85),
                            weights=lengths,
                            double_hist=True,
                            label='Directions from direct ancestor (local drift)',
                            ax=ax
                            )

    return ax


if __name__ == '__main__':
    FAMILY = 'bantu'

    if FAMILY == 'bantu':
        TREE_PATH = 'data/bantu/nowhere.tree'
        LOCATION_KEY = 'location'
        XLIM = (2, 52)
        YLIM = (-35, 15)
        DRIFT_DIRECTION = angle_to_vector(-0.5)

    elif FAMILY == 'ie':
        TREE_PATH = 'data/indo-european/IE_2MCCs.tre'
        LOCATION_KEY = 'trait'
        XLIM = (-25, 120)
        YLIM = (20, 80)

    else:
        raise ValueError()

    # Plot the world map in the background.
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color=(.95, .95, .95), edgecolor='lightgrey', lw=.4, )

    with open(TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Node.from_newick(newick_str)

    tree.set_location_attribute(LOCATION_KEY)

    def color_direction(parent, child):
        # Compute migration direction
        step = child.location - parent.location
        direction = normalize(step)

        # Color according to direction
        cm = plt.get_cmap('PiYG')
        # cm = plt.get_cmap('RdYlGn')
        c = 0.5 * (1 + direction.dot(DRIFT_DIRECTION))
        color = cm(c)

        # Alpha according to migration rate
        # diff_rate = norm(step) / child.length
        # a = 1. + 0.1*np.log(diff_rate)
        # a = np.clip(a, 0., 1.)

        # # Alpha according to height
        height = (parent['height'] + child['height']) / 2.
        a = height > 260.
        a = np.clip(a, 0., 1.)

        return color, a

    plot_tree(tree, color_diffusion_rate)

    plt.gca().set_xlim(XLIM)
    plt.gca().set_ylim(YLIM)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # # ax = plot_eary_late_drift(tree)
    # ax = plot_global_local_drift(tree)
    #
    # ax.set_rticks([])
    # ax.grid(False)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()