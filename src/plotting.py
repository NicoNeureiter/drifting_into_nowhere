#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import attr
import numpy as np
from matplotlib import pyplot as plt, animation as animation
from scipy import signal

from src.util import bounding_box


def plot_tree(node):
    x, y = node.geoState.location
    plt.scatter([x], [y], c='k', lw=0, alpha=0.5)

    if node.children:
        for c in node.children:
            cx, cy = c.geoState.location
            plt.plot([x, cx], [y, cy], c='teal', lw=1.)
            plot_tree(c)


def plot_walk(simulation, show_path=True, show_tree=False):
    walk = simulation.get_location_history()

    if show_path:
        for i in range(simulation.n_sites):
            # w = signal.hann(10)**8.
            # walk[:-1, i, 0] = signal.convolve(
            #     np.pad(walk[:, i, 0], (4, 4), 'edge'), w, mode='same')[4:-5] / sum(w)
            # walk[:-1, i, 1] = signal.convolve(
            #     np.pad(walk[:, i, 1], (4, 4), 'edge'), w, mode='same')[4:-5] / sum(w)
            plt.plot(*walk[:, i].T, color='grey', lw=0.4)

    plt.scatter(*walk[0, 0], color='teal')
    plt.scatter(*walk[-1, :, :].T, color='darkred')

    if show_tree:
        plot_tree(simulation.root)

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
