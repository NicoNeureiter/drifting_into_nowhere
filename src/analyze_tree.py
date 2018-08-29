#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
from math import atan2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as ColorNormalize
import geopandas as gpd

from src.tree import Node
from src.util import norm, normalize, dist, grey
from src.plotting import circular_histogram
from src.evaluation import check_root_in_hpd

PINK = (0.95, 0.15, 0.8)
TURQUOISE = (0, 0.8, 0.9)


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


def plot_edge(parent, child, no_arrow=False, **kwargs_arrow):
    x, y = parent.location
    cx, cy = child.location
    if x != cx or y != cy:
        if no_arrow:
            plt.plot([x, cx], [y, cy], c=kwargs_arrow['color'], alpha=kwargs_arrow['alpha'])
        else:
            plt.arrow(x, y, (cx-x), (cy-y), length_includes_head=True, **kwargs_arrow)


def plot_tree(tree: Node, color_fun=None, alpha_fun=None, cmap=None,
              cnorm=None, anorm=None, lw=0.1, color='k', alpha = 1., no_arrow=False):

    if color_fun is not None:
        if cmap is None:
            cmap = plt.get_cmap('PiYG')
            # cmap = plt.get_cmap('plasma')
            # cmap = plt.get_cmap('viridis')

        if color_fun is not None:
            color_values = [color_fun(*e) for e in tree.iter_edges()]
            if cnorm is None:
                cnorm = ColorNormalize(vmin=min(color_values), vmax=max(color_values))

        if alpha_fun is not None:
            alphas = [alpha_fun(*e) for e in tree.iter_edges()]
            if anorm is None:
                anorm = ColorNormalize(vmin=min(alphas), vmax=max(alphas))

    for edge in tree.iter_edges():
        if color_fun is not None:
            c = color_fun(*edge)
            color = cmap(cnorm(c))

        if alpha_fun is not None:
            alpha = anorm(alpha_fun(*edge))

        plot_edge(*edge, color=color, alpha=alpha, width=lw, lw=0, no_arrow=no_arrow)


def get_edge_drift_agreement(parent, child):
    # Compute migration direction
    step = child.location - parent.location
    direction = normalize(step)

    # Agreement (dot-product) with drift-direction
    return direction.dot(DRIFT_DIRECTION)


def get_edge_diff_rate(parent, child):
    step = child.location - parent.location
    diff_rate = norm(step) / child.length
    return diff_rate


def get_edge_heights(parent, child):
    return (parent['height'] + child['height']) / 2.


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
    for parent, child in tree.iter_edges():
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

    for parent, child in tree.iter_edges():
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


def invert(fun):
    def inv_fun(*args, **kwargs):
        return 1 - fun(*args, **kwargs)
    return inv_fun


def flatten(fun, alpha):
    def flat_fun(*args, **kwargs):
        return fun(*args, **kwargs) ** alpha
    return flat_fun


if __name__ == '__main__':
    FAMILY = 'bantu'
    # FAMILY = 'ie'

    if FAMILY == 'bantu':
        TREE_PATH = 'data/bantu_brownian/nowhere.tree'
        LOCATION_KEY = 'location'
        XLIM = (2, 52)
        YLIM = (-35, 15)
        DRIFT_ANGLE = -0.5
        DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)
        swap_xy = False
        HOMELAND = np.array([6.5, 10.5])

    elif FAMILY == 'ie':
        # TREE_PATH = 'data/ie/IE_2MCCs.tree'
        # LOCATION_KEY = 'trait'

        TREE_PATH = 'data/ie/nowhere.tree'
        LOCATION_KEY = 'location'

        swap_xy = True
        XLIM = (-25, 120)
        YLIM = (20, 80)
        DRIFT_ANGLE = 2.8
        DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)

    else:
        raise ValueError()

    # cmap = plt.get_cmap('PiYG')
    # cmap = plt.get_cmap('RdYlGn')
    # cmap = plt.get_cmap('plasma')
    cmap = plt.get_cmap('viridis')

    # Plot the world map in the background.
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color=grey(.95), edgecolor=grey(0.7), lw=.4, )

    # Load Tree
    with open(TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Node.from_newick(newick_str, location_key=LOCATION_KEY, swap_xy=swap_xy)
        # if LOCATION_KEY == 'trait':
        #     tree = tree.get_subtree([0, 0, 0, 0])

    # Plot Tree
    plot_tree(tree, cmap=cmap,
              color_fun=invert(get_edge_heights),
              alpha_fun=None)
    okcool = check_root_in_hpd(TREE_PATH, 80, root=HOMELAND[::-1],
                               ax=ax, color=PINK, zorder=2)
    print(okcool)

    # Plot root location
    root = tree.location
    print(root)
    plt.scatter(HOMELAND[1], HOMELAND[0], marker='*', c=TURQUOISE, s=500, zorder=3)
    plt.scatter(root[0], root[1], marker='*', c=PINK, s=500, zorder=3)
    # plt.scatter(root[0], root[1], marker='*', c=(0.85, 0.7, 0.), s=800, zorder=2,
    #             edgecolors=(0.35, 0.3, 0), lw=1.)

    ############################################################################
    # Plot Legend Compass
    ############################################################################

    px, py = -3., 73.
    # px, py = 8., -30.
    r = 2.5
    n_arrows = 10
    arrow_angles = np.linspace(0,2 * np.pi, n_arrows, endpoint=False)
    arrow_angles = (arrow_angles + DRIFT_ANGLE) % (2 * np.pi)
    for angle in arrow_angles:
        direction = angle_to_vector(angle)
        c = cmap(0.5 + 0.5*direction.dot(DRIFT_DIRECTION))
        plt.arrow(px, py, r * direction[0], r * direction[1], color=c,
                  width = 0.05)

    c0 = plt.Circle((px, py), 0.1 * r, facecolor='w', edgecolor='grey', lw=0.5)
    c1 = plt.Circle((px, py), 1.12 * r, facecolor=(0, 0, 0, 0), edgecolor='grey', lw=0.5)
    ax.add_artist(c0)
    ax.add_artist(c1)

    # locs = np.array([leaf.location for leaf in tree.iter_leafs()])
    # plt.scatter(*locs.T, c='k', s=9.)

    # Plot Settings
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()
