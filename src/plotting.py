#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math

import numpy as np
from matplotlib import pyplot as plt, animation as animation

from src.util import bounding_box


def plot_tree(node, alpha=1.):
    x, y = node.geoState.location
    # plt.scatter([x], [y], c='k', lw=0, alpha=0.5)

    if node.children:
        for c in node.children:
            cx, cy = c.geoState.location
            plt.plot([x, cx], [y, cy], c='teal', lw=1., alpha=alpha)
            plot_tree(c, alpha=alpha)


def plot_walk(simulation, show_path=True, show_tree=False, ax=None,
              savefig=None, alpha=0.3):
    no_ax = (ax is None)
    if no_ax:
        ax = plt.gca()

    walk = simulation.get_location_history()

    if show_path:
        for i in range(simulation.n_sites):
            # w = signal.hann(10)**8.
            # walk[:-1, i, 0] = signal.convolve(
            #     np.pad(walk[:, i, 0], (4, 4), 'edge'), w, mode='same')[4:-5] / sum(w)
            # walk[:-1, i, 1] = signal.convolve(
            #     np.pad(walk[:, i, 1], (4, 4), 'edge'), w, mode='same')[4:-5] / sum(w)
            plt.plot(*walk[:, i].T, color='grey', lw=0.2)

    ax.scatter(*walk[0, 0], color='teal', lw=0)
    ax.scatter(*walk[-1, :, :].T, color='darkred', s=4, lw=0)

    if show_tree:
        plot_tree(simulation.root, alpha=alpha)

    if savefig is not None:
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(savefig, format='pdf')

    elif no_ax:
        plt.axis('off')
        plt.show()


def animate_walk(simulation):
    walk = simulation.get_location_history()
    n_steps, n_sites, n_dim = walk.shape
    assert n_sites == simulation.n_sites
    assert n_dim == 2

    x_min, y_min, x_max, y_max = bounding_box(walk.reshape(-1, 2), margin=0.1)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_min, x_max), ax.set_xticks([])
    ax.set_ylim(y_min, y_max), ax.set_yticks([])

    paths = []
    for i in range(simulation.n_sites):
        p, = ax.plot([], [], color='grey', lw=0.1)
        paths.append(p)

    scat = ax.scatter(*walk[0, :, :].T, color='darkred', lw=0.)

    def update(i_frame):
        scat.set_offsets(walk[i_frame, :, :])
        for i, p in enumerate(paths):
            p.set_data(walk[:i_frame, i, 0], walk[:i_frame, i, 1])

    _ = animation.FuncAnimation(fig, update, frames=n_steps-1,
                                interval=80, repeat_delay=1000.)

    plt.show()


PI = np.pi
TAU = 2 * np.pi
def circular_histogram(x, bins=50, range=(0, TAU), weights=None, ax=None, label=None,
                       double_hist=False, normed=False, color='teal', bottom=.3, **kwargs):
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

    w0 = 0.5 * width[0]

    if first_hist:
        z = bottom * max(hist)
        ylim = z + max(hist)
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