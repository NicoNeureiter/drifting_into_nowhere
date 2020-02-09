#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import matplotlib.pyplot as plt



def plot_topology(height, root=0., dx=.5, drift=0.5, ax=None):
    if ax is None:
        ax = plt.gca()

    x = root

    if height == 0:
        return

    x1 = x - dx + drift
    x2 = x + dx + drift
    plt.plot([x, x1], [height, height - 1], c='k')
    plt.plot([x, x2], [height, height - 1], c='k')
    plot_topology(height - 1, root=x1, dx=0.8*dx, drift=drift)
    plot_topology(height - 1, root=x2, dx=0.8*dx, drift=drift)

def plot_drift(n, ax=None):
    if ax is None:
        ax = plt.gca()



if __name__ == '__main__':
    HEIGHT = 3
    plot_topology(height=HEIGHT)

    plt.tight_layout(pad=0.)
    plt.axis('off')
    plt.show()