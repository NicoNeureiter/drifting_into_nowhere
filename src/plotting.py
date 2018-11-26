#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math

import numpy as np
from matplotlib import pyplot as plt, animation as animation
from matplotlib.colors import Normalize as ColorNormalize
from scipy.stats import norm as normal

from src.tree import Node
from src.util import bounding_box, mkpath
from src.config import *

PI = np.pi
TAU = 2 * np.pi

ANIM_FOLDER = 'results/simulation_visualizations/animation/'
PATH_WIDTH = 0.6


def _plot_tree(node, alpha=1., plot_height=False, color='teal'):
    x, y = node.location
    if plot_height:
        y = -node.height
    # plt.scatter([x], [y], c='k', lw=0, alpha=0.5)

    if node.children:
        for c in node.children:
            cx, cy = c.location
            if plot_height:
                cy = -c.height
            plt.plot([x, cx], [y, cy], c=color, lw=1., alpha=alpha)
            _plot_tree(c, alpha=alpha, plot_height=plot_height, color=color)


def plot_edge(parent, child, no_arrow=False, **kwargs_arrow):
    x, y = parent.location
    cx, cy = child.location
    if x != cx or y != cy:
        if no_arrow:
            plt.plot([x, cx], [y, cy], c=kwargs_arrow['color'], alpha=kwargs_arrow['alpha'])
        else:
            plt.arrow(x, y, (cx - x), (cy - y), length_includes_head=True, **kwargs_arrow)


def plot_tree(tree: Node, color_fun=None, alpha_fun=None, cmap=None,
              cnorm=None, anorm=None, lw=0.1, color='k', alpha=1., no_arrow=False):
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


def plot_walk(simulation, show_path=True, show_tree=False, ax=None,
              savefig=None, alpha=0.3):
    no_ax = (ax is None)
    if no_ax:
        ax = plt.gca()

    walk = simulation.get_location_history()
    ax.scatter(*walk[-1, :, :].T, color=COLOR_SCATTER, s=30, lw=0, zorder=4)
    plt.scatter(*walk[0, 0], marker='*', c=COLOR_ROOT_TRUE, s=500, zorder=5)

    if show_path:
        for i in range(simulation.n_sites):
            # k = 10
            # w = signal.hann(k)**8.
            # walk[:-1, i, 0] = signal.convolve(
            #     np.pad(walk[:, i, 0], (k, k), 'edge'), w, mode='same')[k:-k-1] / sum(w)
            # walk[:-1, i, 1] = signal.convolve(
            #     np.pad(walk[:, i, 1], (k, k), 'edge'), w, mode='same')[k:-k-1] / sum(w)
            plt.plot(*walk[:, i].T, color=COLOR_PATH, lw=PATH_WIDTH, zorder=0)

    if show_tree:
        plot_tree(simulation.root, alpha=alpha)

    if savefig is not None:
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(savefig, format='pdf')

    elif no_ax:
        plt.axis('off')
        plt.show()


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


def plot_rrw_step_pdf(lognormal_mean=1., n_samples=30000):
    xmax = 1.
    x = np.linspace(-xmax, xmax, 2000)
    plt.plot(x, normal.pdf(x, scale=0.1), label='Normal(0, 0.1)')
    # plt.plot(x, laplace.pdf(x, scale=0.1), label='Laplace(0, 0.1)')

    # for lognormal_stdev in 3**np.linspace(-2, 1, 4):
    for lognormal_stdev in [3.]:
        normal_stdev = np.random.lognormal(lognormal_mean, lognormal_stdev, size=n_samples)
        def step_pdf(x):
            return np.mean(normal.pdf(x[:, None], scale=normal_stdev[None, :]), axis=1)

        plt.plot(x, step_pdf(x), label='RRW (s=%.1f)' % lognormal_stdev)

    ax = plt.gca()
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([0, None])
    plt.legend()
    plt.tight_layout(pad=0)
    plt.show()

def plot_hpd(tree, p_hpd, location_key='location', **kwargs_plt):
    kwargs_plt.setdefault('color', COLOR_ROOT_EST)
    kwargs_plt.setdefault('alpha', 0.2)
    kwargs_plt.setdefault('zorder', 1)

    for polygon in tree.get_hpd(p_hpd, location_key=location_key):
        x, y = polygon.exterior.xy
        plt.fill(x, y, **kwargs_plt)


def plot_root(root, color=COLOR_ROOT_EST):
    return plt.scatter(root[0], root[1], marker='*', c=color, s=500, zorder=5)


def simulation_plots(simulation, tree_rec, save_path=None):

    def show(xlim=None, ylim=None, equal=True):
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if equal:
            plt.axis('equal')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, format='pdf')

    # Plot walk with HPD
    plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    xlim = plt.xlim()
    ylim = plt.ylim()
    show()

    # Plot walk without HPD
    plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    show(xlim, ylim)

    # Plot tree
    # ax = plot_walk(simulation, show_path=False, show_tree=True, ax=plt.gca(), alpha=0.2)
    walk = simulation.get_location_history()
    heights = [-s.height for s in simulation.societies]
    plot_tree(simulation.root, plot_height=True)
    plot_tree(tree_rec, plot_height=True, color=COLOR_ROOT_EST)
    plt.scatter(walk[-1, :, 0], heights, color=COLOR_ROOT_TRUE, s=30, lw=0, zorder=4)
    plt.scatter(0, 0, marker='*', c=COLOR_ROOT_TRUE, s=500, zorder=5)
    plt.scatter(tree_rec.location[0], 0, marker='*', c=COLOR_ROOT_EST, s=500, zorder=5)
    show(xlim, ylim)

    # Plot only final locations + Root
    plot_walk(simulation, show_path=False, show_tree=False, ax=plt.gca())
    plt.scatter(0, 0, marker='*', c=COLOR_ROOT_TRUE, s=500, zorder=5)
    show(xlim, ylim)

    # Plot oly final locations
    plot_walk(simulation, show_path=False, show_tree=False, ax=plt.gca())
    show(xlim, ylim)

if __name__ == '__main__':
    plot_rrw_step_pdf()
