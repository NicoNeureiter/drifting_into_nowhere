#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
from math import atan2

import geopandas as gpd

from src.tree import angle_to_vector, get_edge_diff_rate
from src.util import norm, normalize, extract_newick_from_nexus
from src.plotting import *
from src.config import PINK, TURQUOISE


def get_edge_drift_agreement(parent, child):
    # Compute migration direction
    step = child.location - parent.location
    direction = normalize(step)

    # Agreement (dot-product) with drift-direction
    return direction.dot(DRIFT_DIRECTION)


N_BINS = 28
COLOR_1 = (0.1, 0.6, 0.75)
COLOR_2 = (0.5, 0.1, 0.4)


def plot_early_late_drift(tree: Tree):
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

    ax.set_rticks([])
    ax.grid(False)
    plt.tight_layout(pad=0)
    plt.legend()

    return ax


def plot_global_local_drift(tree: Tree):
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

    # ax = circular_histogram(angles_from_root, bins=N_BINS, normed=1, color=COLOR_1,
    #                         weights=lengths_from_root,
    #                         double_hist=True,
    #                         label='Directions from root'
    #                         )
    ax = circular_histogram(angles_from_root, bins=N_BINS, normed=1, color=COLOR_2,
                            weights=lengths,
                            double_hist=True,
                            # label='Directions from direct ancestor',
                            label='Directions from root',
                            # ax=ax
                            )
    ax.set_rticks([])
    ax.grid(False)
    plt.tight_layout(pad=0)
    plt.legend()

    return ax


def invert(fun):
    def inv_fun(*args, **kwargs):
        return 1 - fun(*args, **kwargs)
    return inv_fun


def flatten(fun, alpha):
    def flat_fun(*args, **kwargs):
        v = fun(*args, **kwargs)
        return np.sign(v) * (abs(v) ** alpha)

    return flat_fun


if __name__ == '__main__':
    FAMILY = 'bantu'
    # FAMILY = 'ie'

    if FAMILY == 'bantu':
        LOCATION_KEY = 'location'
        # XLIM = (2, 52)
        # YLIM = (-35, 15)
        XLIM = (-30, 60)
        YLIM = (-35, 25)
        DRIFT_ANGLE = -0.5
        DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)
        swap_xy = False
        HOMELAND = np.array([6.5, 10.5])
        LW = 0.1
        HPD = 80

        # TREE_PATH = 'data/bantu_withoutgroup_2/nowhere.tree'
        TREE_PATH = 'data/bantu_rrw_outgroup_0_adapttree_0_adaptheight_0_hpd_{hpd}/nowhere.tree'.format(hpd=HPD)
        # TREE_PATH = 'data/bantu_brownian_without_outgroup/nowhere.tree'

        PLOT_DRIFT_LEGEND = 0
        PLOT_TREE = 0
        PLOT_HPD = 0
        PLOT_ROOT = 0
        PLOT_HOMELAND = 0
        PLOT_TIPS = 0

    elif FAMILY == 'ie':
        TREE_PATH = 'data/ie/IE_2MCCs.tree'
        LOCATION_KEY = 'trait'

        # TREE_PATH = 'data/ie/nowhere.tree'
        # LOCATION_KEY = 'location'

        swap_xy = True
        XLIM = (-30, 110)
        YLIM = (0, 75)
        DRIFT_ANGLE = 2.8
        DRIFT_DIRECTION = angle_to_vector(DRIFT_ANGLE)
        LW = .12
        HPD = 60

        PLOT_DRIFT_LEGEND = False
        PLOT_TREE = True
        PLOT_HPD = False
        PLOT_ROOT = False
        PLOT_HOMELAND = False
        PLOT_TIPS = False
    else:
        raise ValueError()

    # Plot the world map in the background.
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = gpd.read_file('data/naturalearth_50m_wgs84.geojson')
    # world = gpd.read_file('data/ne_50m_admin_0_countries.geojson')

    ax = world.plot(color=grey(.9),
                    edgecolor=grey(0.4), lw=.33, )
    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('jet')
    # 'viridis', 'inferno', 'plasma', 'gnuplot', 'PiYG', 'RdYlGn',

    # Load Tree
    with open(TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Tree.from_newick(newick_str, location_key=LOCATION_KEY, swap_xy=swap_xy)
        # tree = tree.get_subtree([0, 0, 0, 0])

        okcool = tree.root_in_hpd(HOMELAND, HPD)
        print('\n\nOk cool: %r' % okcool)

    # Plot Tree
    if PLOT_TREE:
        # plot_tree(tree, cmap=cmap,
        #           color_fun=flatten(invert(get_edge_heights), 1.),
        #           # color='darkblue',
        #           # color_fun=get_edge_drift_agreement,
        #           # alpha_fun=flatten(get_edge_heights, .9),
        #           lw=2., no_arrow=False)

        # plot_backbone_splits(tree, plot_edges=True, lw=LW)
        plot_tree(tree, lw=3., color_fun=get_edge_diff_rate, cmap=cmap)
        # plot_clades(tree, max_clade_size=50)
        # plot_subtree_hulls(tree)



    if PLOT_HPD:
        okcool = plot_hpd(tree, HPD, color=PINK)
    if PLOT_ROOT:
        root = tree.location
        plt.scatter(root[0], root[1], marker='*', c=PINK, s=500, zorder=3)
        # plt.scatter(root[0], root[1], marker='*', c='w', s=5000, zorder=3, edgecolor='k')
    if PLOT_HOMELAND:
        plt.scatter(HOMELAND[1], HOMELAND[0], marker='*', c=TURQUOISE, s=500, zorder=3)


    if PLOT_DRIFT_LEGEND:
        px, py = 8., -30.
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

    if PLOT_TIPS:
        locs = np.array([leaf.location for leaf in tree.iter_leafs()])
        plt.scatter(*locs.T, c='k', s=9.)

    # Plot Settings
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()
    exit()

    # # Plot backbone-splits over time.
    # plot_backbone_splits(tree, plot_edges=False)#, lw=3.)
    def get_space_time_position(node):
        t = -node.height
        x = np.dot(node.location, DRIFT_DIRECTION)
        return x, t

    plot_tree(tree, lw=2.5, alpha=0.6, get_node_position=get_space_time_position,
              cmap=None, color_fun=get_edge_diff_rate)
    # plt.scatter(tree.location[0], tree.location[1], marker='*', c='k', s=500, zorder=3)
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()