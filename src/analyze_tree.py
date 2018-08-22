#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
from math import atan2

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from src.tree import Node
from src.util import norm, normalize, dist, grey
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

        plot_tree(c, color_fun)

        # plt.plot([x, cx], [y, cy], c=color, alpha=alpha)
        plt.arrow(x, y, (cx-x), (cy-y), color=color, alpha=alpha, width=0.07,
                  length_includes_head=True)


def color_diffusion_rate(parent, child):
    pxy = parent.location
    cxy = child.location

    dt = child.length

    # cm = plt.get_cmap('plasma')
    cm = plt.get_cmap('viridis')

    rate = dist(pxy, cxy) / dt
    v = 1.1 + 0.15*np.log(rate)

    v = np.clip(v, 0., 1.)
    color = cm(1-v)

    height = (parent['height'] + child['height']) / 2.
    a = np.clip(height / 500., 0, 1)

    return color, a


def color_height(parent, child):
    # cm = plt.get_cmap('plasma')
    cm = plt.get_cmap('viridis')

    height = (parent['height'] + child['height']) / 2.
    v = np.clip(height / 500., 0, 1)
    c = cm(1-v)

    a = 0.5 + 0.5*v

    return c, a


# def plot_diffusion_rate(edges):
#     diff_rates = []
#     xs = []
#     ys = []
#     for p, c in edges:


def angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


N_BINS = 28
COLOR_1 = (0.1, 0.6, 0.75)
COLOR_2 = (0.5, 0.1, 0.4)


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
    ax = circular_histogram(angles_early, bins=N_BINS, normed=1, color=COLOR_1,
                            weights=lengths_early, double_hist=True,
                            label='Step direction - earlier half of the expansion'
                            )
    ax = circular_histogram(angles_late, bins=N_BINS, normed=1, color=COLOR_2,
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

    ax = circular_histogram(angles_from_root, bins=N_BINS, normed=1, color=COLOR_1,
                            weights=lengths_from_root,
                            double_hist=True,
                            label='Directions from root (global drift)'
                            )
    ax = circular_histogram(angles, bins=N_BINS, normed=1, color=COLOR_2,
                            weights=lengths,
                            double_hist=True,
                            label='Directions from direct ancestor (local drift)',
                            ax=ax
                            )

    return ax


if __name__ == '__main__':
    FAMILY = 'ie'

    if FAMILY == 'bantu':
        TREE_PATH = 'data/bantu/nowhere.tree'
        LOCATION_KEY = 'location'
        XLIM = (2, 52)
        YLIM = (-35, 15)
        DRIFT_ANGLE = -0.5
        DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)

    elif FAMILY == 'ie':
        TREE_PATH = 'data/indo-european/IE_2MCCs.tre'
        LOCATION_KEY = 'trait'
        XLIM = (-25, 120)
        YLIM = (20, 80)
        DRIFT_ANGLE = 2.8
        DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)

    else:
        raise ValueError()

    # Plot the world map in the background.
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color=grey(.95), edgecolor=grey(0.7), lw=.4, )

    with open(TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Node.from_newick(newick_str, location_key=LOCATION_KEY)
        tree = tree.get_subtree([0, 0, 0, 0])

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

    plot_tree(tree, color_direction)

    root = tree.location
    print(root)
    # plt.scatter(HOMELAND[1], HOMELAND[0], marker='*', c=(0., 0.8, .9), s=500, zorder=2)
    plt.scatter(root[0], root[1], marker='*', c=(.9, 0., .8), s=500, zorder=2)

    cm = plt.get_cmap('PiYG')
    px, py = -3., 73.
    l = 4
    n_arrows = 10
    arrow_angles = np.linspace(0,2 * np.pi, n_arrows, endpoint=False)
    arrow_angles = (arrow_angles + DRIFT_ANGLE) % (2 * np.pi)
    for angle in arrow_angles:
        direction = angle_to_vector(angle)
        c = cm(0.5 + 0.5*direction.dot(DRIFT_DIRECTION))
        plt.arrow(px, py, l * direction[0], l * direction[1], color=c,
                  width = 0.2)
    # plt.arrow(px, py, -l * DRIFT_DIRECTION[0], -l * DRIFT_DIRECTION[1], color=cm(0.),
    #           width = 0.2)

    c0 = plt.Circle((px, py), 0.1*l, facecolor='w', edgecolor='k')
    c1 = plt.Circle((px, py), 1.2*l, facecolor=(0,0,0,0), edgecolor='k')
    plt.gca().add_artist(c0)
    plt.gca().add_artist(c1)

    plt.gca().set_xlim(XLIM)
    plt.gca().set_ylim(YLIM)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()

    # ax = plot_eary_late_drift(tree)
    # # ax = plot_global_local_drift(tree)
    #
    # ax.set_rticks([])
    # ax.grid(False)
    # plt.tight_layout(pad=0)
    # plt.legend()
    # plt.show()