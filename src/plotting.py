#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math

import numpy as np
from matplotlib import pyplot as plt, animation as animation
from matplotlib.colors import Normalize as ColorNormalize
from scipy.stats import norm as normal, laplace
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from src.tree import Tree
from src.simulation.simulation import State
from src.util import bounding_box, mkpath, grey
from src.config import *

PI = np.pi
TAU = 2 * np.pi

ANIM_FOLDER = 'results/simulation_visualizations/animation/'
PATH_WIDTH = 0.7



def get_location(node):
    return node.location


def plot_height(node, alpha=.6, color='teal', lw=1.):
    if node.parent:
        x, y = node.parent.location
        cx, cy = node.location

        # x = 0.877*x - 0.479*y
        # cx = 0.877*cx - 0.479*cy
        y = -node.parent.height
        cy = -node.height

        plt.plot([x, cx], [y, cy], c=color, alpha=alpha, lw=lw)
    else:
        print(node.location)

    _plot_tree(node, plot_height=True, alpha=alpha, color=color, lw=lw)


def _plot_tree(node, alpha=1., plot_height=False, color='teal', lw=1.):
    x, y = node.location
    if plot_height:
        # x = 0.877*x - 0.479*y
        y = -node.height
    # plt.scatter([x], [y], c='k', lw=0, alpha=0.5)

    if node.children:
        for c in node.children:
            cx, cy = c.location
            if plot_height:
                # cx = 0.877*cx - 0.479*cy
                cy = -c.height
            plt.plot([x, cx], [y, cy], c=color, lw=lw, alpha=alpha)
            _plot_tree(c, alpha=alpha, plot_height=plot_height, color=color, lw=lw)


def plot_edge(parent, child, no_arrow=False, ax=None,
              get_node_position=get_location, **kwargs_arrow):
    if ax is None:
        ax = plt.gca()
    color = kwargs_arrow.pop('color', 'k')
    alpha = kwargs_arrow.pop('alpha', 1.)

    x, y = get_node_position(parent)
    cx, cy = get_node_position(child)
    if x != cx or y != cy:
        if no_arrow:
            return ax.plot([x, cx], [y, cy], c=color, alpha=alpha, **kwargs_arrow)
        else:
            width = kwargs_arrow.pop('lw', 0.001)
            return ax.arrow(x, y, (cx - x), (cy - y), length_includes_head=True,
                            width=width, lw=0, color=color, alpha=alpha)


def plot_tree(tree, color_fun=None, alpha_fun=None, cmap=None,
              cnorm=None, anorm=None, color='k', alpha=1., no_arrow=True,
              ax=None, lw=None, get_node_position=get_location, show_colorbar=False):
    if cmap is None:
        cmap = plt.get_cmap('viridis')

    if color_fun is not None:
        color_values = [color_fun(*e) for e in tree.iter_edges()]
        if cnorm is None:
            vmin, vmax = np.quantile(color_values, [0.05, .95])
            cnorm = plt.Normalize(vmin=vmin, vmax=vmax)
            # cnorm = ColorNormalize(vmin=min(color_values),
            #                        vmax=max(color_values))

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

        plot_edge(*edge, color=color, alpha=alpha, no_arrow=no_arrow, ax=ax, lw=lw,
                  get_node_position=get_node_position)

    if tree.parent:
        edge = tree.parent, tree
        if color_fun is not None:
            c = color_fun(*edge)
            color = cmap(cnorm(c))

        if alpha_fun is not None:
            alpha = anorm(alpha_fun(*edge))

        plot_edge(*edge, color=color, alpha=alpha, no_arrow=no_arrow, ax=ax, lw=lw,
                  get_node_position=get_node_position)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
        sm._A = []
        plt.colorbar(sm)


def plot_walk(tree, show_tree=False, show_root=True, show_tips=True, show_nodes=False,
              ax=None, savefig=None, color='grey', alpha=0.3, lw=0.3):
    """Plot the random walk simulated along a tree.
    
    Args:
        tree (State): The root of the simulated tree.
        show_tree (bool):
        show_root (bool):
        show_tips (bool):
        ax: 
        savefig:
        color:
        alpha:
        lw:

    Returns:
        plt.Axes: The axis of the plot.
    """

    if ax is None:
        ax = plt.gca()

    if show_root:
        plot_root(tree.location, color=COLOR_ROOT_TRUE, label='Simulated root', ax=ax)

    if show_nodes:
        ax.scatter(*tree.get_descendant_locations().T, color=COLOR_PATH, s=20,
                   lw=0, zorder=2)
    if show_tips:
        ax.scatter(*tree.get_leaf_locations().T, color=COLOR_SCATTER, s=80,
                   # lw=1., edgecolor='w',
                   zorder=4)

    for node in tree.iter_descendants():
        walk = np.array(node.location_history)
        k = 12
        w = np.hanning(k)**2
        walk[1:-1, 0] = np.convolve(
            np.pad(walk[:, 0], (k, k), 'edge'), w, mode='same')[k+1:-k-1] / sum(w)
        walk[1:-1, 1] = np.convolve(
            np.pad(walk[:, 1], (k, k), 'edge'), w, mode='same')[k+1:-k-1] / sum(w)
        ax.plot(*walk.T, color=COLOR_PATH, lw=lw, zorder=0)

    if show_tree:
        plot_tree(tree, alpha=alpha, ax=ax)

    if savefig is not None:
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(savefig, format='pdf')

    return ax

def animate_walk(simulation, alpha=0.3, anim_folder=ANIM_FOLDER):
    mkpath(anim_folder)
    walk = simulation.get_location_history()
    n = walk.shape[0]
    n_steps, n_sites, n_dim = walk.shape
    assert n_sites == simulation.n_sites
    assert n_dim == 2

    x_min, y_min, x_max, y_max = bounding_box(walk.reshape(-1, 2), margin=0.05)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.tight_layout(pad=0)
    ax.set_xlim(x_min, x_max), ax.set_xticks([])
    ax.set_ylim(y_min, y_max), ax.set_yticks([])
    w, h = fig.get_size_inches()
    h *= 15 / w
    w = 15
    fig.set_size_inches(w, h)

    paths = []
    for i in range(simulation.n_sites):
        p, = ax.plot([], [], color=COLOR_PATH, lw=PATH_WIDTH, zorder=0)
        paths.append(p)

    root = ax.scatter(*walk[0, :, :].T, marker='*', c=COLOR_ROOT_TRUE, s=500, zorder=5)
    scat = ax.scatter(*walk[0, :, :].T, color=COLOR_SCATTER, s=30, lw=0, zorder=4)

    def update(frame):
        if frame > n+1:
            return

        scat.set_offsets(walk[frame-2:frame+3, :, :].mean(axis=0))
        for i, p in enumerate(paths):
            p.set_data(walk[:frame, i, 0], walk[:frame, i, 1])
        # plt.savefig(anim_folder + 'step_%i.png' % frame)

    anim = animation.FuncAnimation(fig, update, frames=n_steps-1 + 20,
                                   interval=60)
    anim.save('results/animation.gif', dpi=100, writer='imagemagick')
    plt.show()


def circular_histogram(x, bins=50, range=(0, TAU), weights=None, ax=None, label=None,
                       double_hist=False, normed=False, color='teal', bottom=.4, **kwargs):
    if ax is None:
        ax = plt.subplot(111, polar=True)
        first_hist = True
    else:
        first_hist = False

    x = np.asarray(x)
    x = x % TAU

    hist, bin_edges = np.histogram(x, bins=bins, range=range, weights=weights)
    hist = hist.clip(min=0.001*max(hist))

    angles = (bin_edges[1:] + bin_edges[:-1]) / 2
    width = np.diff(bin_edges)

    if normed:
        hist = hist / width.dot(hist)

    w0 = 0.7 * width[0]

    if first_hist:
        z = bottom * max(hist)
        ylim = z + 1.3*max(hist)
    else:
        ylim = ax.get_ylim()[1]
        z = ylim * bottom / (1. + bottom)
        ylim_new = z + max(hist)
        if ylim_new > ylim:
            ylim = ylim_new

    zc = z * np.cos(w0)

    # Draw vertical lines
    ax.vlines(angles, zc, ylim, linestyles='dashed', lw=0.5, color='lightgrey', zorder=2)

    for a, h in zip(angles, hist):
        w1 = _math.atan(zc / (zc+h) * _math.tan(w0))
        h1 = (zc + h) / np.cos(w1)
        z1 = zc / np.cos(w1)

        if not double_hist:
            bar_angles = [a-w0, a-w1, a+w1, a+w0]
            r_lower = [z, z1, z1, z]
            r_upper = [z, h1, h1, z]
        else:
            if first_hist:
                bar_angles = [a-w0, a-w1, a]
                r_lower = [z, z1, zc]
                r_upper = [z, h1, zc+h]
            else:
                bar_angles = [a, a+w1, a+w0]
                r_lower = [zc, z1, z]
                r_upper = [zc+h, h1, z]

        ax.fill_between(bar_angles, r_lower, r_upper, facecolors=color, zorder=2)

    # Add label
    ax.plot([], [], c=color, label=label)

    # ax.spines['polar'].set_color('grey')
    ax.spines['polar'].set_linewidth(0.5)
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    ax.set_ylim(0, ylim)

    return ax


def plot_rrw_step_pdf(lognormal_mean=1., n_samples=2000):
    xmax = 6.
    x = np.linspace(-xmax, xmax, 1001)
    # plt.plot(x, normal.pdf(x, scale=0.1), label='Normal(0, 0.1)')
    plt.plot(x, normal.pdf(x, scale=1.), label='Normal(0, 1)')
    # plt.plot(x, laplace.pdf(x, scale=1.), label='Laplace(0, 1)')

    lognormal_mean = 0.
    for lognormal_stdev in [0.5, 1., 5.]:
        lognormal_mean = -lognormal_stdev**2 / 2
        normal_stdev = np.random.lognormal(lognormal_mean, lognormal_stdev, size=n_samples)
        print(normal_stdev)
        def step_pdf(x):
            return np.mean(normal.pdf(x[:, None], scale=normal_stdev[None, :]), axis=1)

        plt.plot(x, step_pdf(x), label='RRW (s=%.1f)' % lognormal_stdev)

    ax = plt.gca()
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([0, 2])
    plt.legend()
    plt.tight_layout(pad=0)
    plt.show()

def plot_hpd(tree, p_hpd, location_key='location', projection=None, ax=None, **kwargs_plt):
    if ax is None:
        ax = plt.gca()

    kwargs_plt.setdefault('color', COLOR_ROOT_EST)
    kwargs_plt.setdefault('alpha', 0.1)
    kwargs_plt.setdefault('zorder', 1)
    kwargs_plt.setdefault('lw', 0)

    for polygon in tree.get_hpd(p_hpd, location_key=location_key):
        xy = polygon.exterior.xy
        # print(xy)
        if projection is not None:
            xy = projection(xy.T).T
            # print('Projected:', xy)
        ax.fill(*xy, **kwargs_plt)


def plot_posterior_scatter(posterior_trees, **kwargs_plt):
    kwargs_plt.setdefault('color', COLOR_ROOT_EST)
    kwargs_plt.setdefault('alpha', 0.1)
    kwargs_plt.setdefault('zorder', 1)
    kwargs_plt.setdefault('lw', 0)
    kwargs_plt.setdefault('s', 1)

    root_samples = np.array(
        [tree.location for tree in posterior_trees]
    )
    # print('root_samples:', root_samples)
    plt.scatter(*root_samples.T, **kwargs_plt)


def plot_root(root, color=COLOR_ROOT_EST, s=800, lw=2, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    return ax.scatter(root[0], root[1], marker='*', c=color, s=s, lw=lw,
                       zorder=5, **kwargs)
                    # s=1600, edgecolor='w'


def plot_backbone(simulation):
    assert isinstance(simulation, 'SimulationBackbone')
    assert hasattr(simulation, 'backbone_steps')

    parent = simulation.root
    for _ in range(simulation.backbone_steps):
        child = parent.children[0]
        plot_edge(parent, child, no_arrow=True, color='k', lw=0.5)

        parent = child


def simulation_plots(simulation, tree_rec, save_path=None):
    plt.figure(figsize=(25, 25), dpi=100, facecolor='w', edgecolor='k')

    def show(xlim, ylim, equal=True, figname=''):
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if equal:
            plt.axis('equal')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path+figname+'.pdf', format='pdf')
            plt.savefig(save_path+figname+'.png', format='png')

        # Prepare next figure...
        plt.figure(figsize=(25, 25), dpi=100, facecolor='w', edgecolor='k')

    # Plot walk with HPD
    plot_hpd(tree_rec, tree_rec.p_hpd)
    plot_root(tree_rec.location)
    plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    if isinstance(simulation, SimulationBackbone):
        plot_backbone(simulation)
    xlim = plt.xlim()
    ylim = plt.ylim()
    show(xlim, ylim, figname='walk_hpd')

    # Plot walk without HPD
    plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    if isinstance(simulation, SimulationBackbone):
        plot_backbone(simulation)
    show(xlim, ylim, figname='walk')

    # Plot colored clades
    plot_backbone_splits(simulation.root, n_clades=simulation.backbone_steps, bb_side=0, lw=0.02,
                         mode='geo', plot_hull=True)
    show(xlim, ylim, figname='colored_clades_geo')
    plot_backbone_splits(simulation.root, n_clades=simulation.backbone_steps, bb_side=0, lw=0.02,
                         mode='time')
    show(xlim, None, equal=False, figname='colored_clades_time')

    # Plot tree
    # ax = plot_walk(simulation, show_path=False, show_tree=True, ax=plt.gca(), alpha=0.2)
    walk = simulation.get_location_history()
    heights = [-s.height for s in simulation.sites]
    # plot_tree(simulation.root)
    # plot_tree(tree_rec, color=COLOR_ROOT_EST)
    plot_height(simulation.root)
    plot_height(tree_rec, color=COLOR_ROOT_EST)
    plt.scatter(walk[-1, :, 0], heights, color=COLOR_ROOT_TRUE, s=30, lw=0, zorder=4)
    plt.scatter(0, 0, marker='*', c=COLOR_ROOT_TRUE, s=500, zorder=5)
    plt.scatter(tree_rec.location[0], 0, marker='*', c=COLOR_ROOT_EST, s=500, zorder=5)
    show(xlim, None, equal=False, figname='tree')

    # Plot only final locations + Root
    plot_walk(simulation, show_path=False, show_tree=False, ax=plt.gca())
    show(xlim, ylim, figname='tips_root')

    # Plot only final locations
    plot_walk(simulation, show_path=False, show_tree=False, ax=plt.gca(), plot_root=False)
    show(xlim, ylim, figname='tips')


def plot_mean_and_std(x, y_mean, y_std, color='blue', label=None):
    plt.plot(x, y_mean, c=color, label=label)
    plt.plot(x, y_mean + y_std, c=color, ls='--', label='+/- standard deviation')
    plt.plot(x, y_mean - y_std, c=color, ls='--')
    plt.fill_between(x, y_mean-y_std, y_mean+y_std,
                     color=color, alpha=0.05, zorder=0)


def plot_clades(tree, min_clade_size=5, max_clade_size=50, _i=0):
    if tree.is_leaf():
        plt.scatter(*tree.location, c='k')
        return _i

    size = tree.n_leafs()
    if min_clade_size < size < max_clade_size:
        plot_tree(tree, color=COLORS[_i])  #, alpha=0.2)
        plot_tree_hull(tree, COLORS[_i])
        return _i + 1
    else:
        for c in tree.children:
            plot_edge(tree, c)
            _i = plot_clades(c, min_clade_size=min_clade_size,
                             max_clade_size=max_clade_size,_i=_i)

        return _i


def plot_backbone_clades(tree, _i=0):
    if tree.is_leaf():
        plt.scatter(*tree.location, c='k')
        return

    clade, rest = tree.children
    print(clade.tree_size(), rest.tree_size())

    plot_tree(clade, color=COLORS[_i])  # , alpha=0.2)
    plot_tree_hull(clade, COLORS[_i])

    plot_edge(tree, rest, lw=0.2, no_arrow=True)
    plot_backbone_clades(rest, _i=_i+1)


def next_split(tree):
    return tree.big_child(), tree.small_child()


def plot_backbone_splits(tree, plot_edges=True, lw=1., n_clades=10, bb_side=1,
                         plot_hull=True):
    # SUBGROUP_COLOR_IDXS = [0, 1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # SUBGROUP_COLORS = [COLORS[i] for i in SUBGROUP_COLOR_IDXS]
    SUBGROUP_COLORS = COLORS

    p = tree
    for i in range(n_clades):
        if p.is_leaf():
            break
        c_backbone, c_subgroup = next_split(p)
        while c_subgroup.tree_size() < 10:
            print('Subgroup Size:', c_subgroup.tree_size(), '--> skipped')
            if plot_edges:
                plot_tree(c_subgroup, color=grey(0.5), lw=lw)
            plot_edge(p, c_backbone, no_arrow=True, lw=2*lw, color='k', zorder=3)
            p = c_backbone
            if p.is_leaf():
                break
            c_backbone, c_subgroup = next_split(p)
        print('Subgroup Size:', c_subgroup.tree_size())



        if plot_edges:
            plot_tree(c_subgroup, color=SUBGROUP_COLORS[i], lw=lw)

        plot_edge(p, c_backbone, no_arrow=True, lw=2*lw, color='k', zorder=3)
        if plot_hull:
            plot_tree_hull(c_subgroup, c=SUBGROUP_COLORS[i])

        p = c_backbone

    # # if mode == 'geo':
    if plot_edges:
        plot_tree(p, color=SUBGROUP_COLORS[n_clades], lw=lw)
    if plot_hull:
        plot_tree_hull(p, c=SUBGROUP_COLORS[n_clades])


def plot_subtree_hulls(tree: Tree):
    for i in range(4):
        c = COLORS[i]
        plot_tree_hull(tree, c=c)
        plot_root(tree.location, color=c, edgecolor=grey(1.))

        tree = tree.big_child()
        tree = tree.big_child()
        tree = tree.big_child()


def plot_tree_hull(tree, c=None):
    leafs = np.array([l.location for l in tree.iter_descendants()])
    try:
        hull = ConvexHull(leafs)
    except QhullError:
        return

    plot_hull_area(hull, leafs, c=c, alpha=0.4, lw=0)
    plot_hull_lines(hull, leafs, c='k')

    leafs = np.array([l.location for l in tree.iter_leafs()])
    plt.scatter(*leafs.T, c=c, edgecolor='k', lw=0.5, zorder=2)
    # plot_root(tree.location, color=c, edgecolor=grey(1.))


def plot_hull_lines(hull, locs, c=None):
    v = hull.vertices
    plt.plot(locs[v, 0], locs[v, 1], c=c, lw=0.1)
    plt.plot(locs[v[[-1, 0]], 0], locs[v[[-1, 0]], 1], c=c, lw=0.1)


def plot_hull_area(hull, locs, c=None, alpha=None, lw=None):
    v = hull.vertices
    plt.fill(locs[v, 0], locs[v, 1], c=c, lw=lw, alpha=alpha)


def plot_tree_topology(tree, left=0):
    x = left + tree.n_leafs() / 2.
    y = -tree.height
    children_sorted = sorted(tree.children, key=lambda c: c.tree_size())
    for i, c in enumerate(children_sorted):
        cy = -c.height
        cx = left + c.n_leafs() / 2.
        plot_tree_topology(c, left=left)

        plt.plot([x, cx, cx], [y, y, cy], c='k')

        left += c.n_leafs()


if __name__ == '__main__':
    plot_rrw_step_pdf()
