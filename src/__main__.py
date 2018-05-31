#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from src.simulation import simulate_evolution


if __name__ == '__main__':

    N_STEPS = 60
    N_FEATURES = 20

    simulation = simulate_evolution(N_STEPS, N_FEATURES)

    vertices, edges = simulation.get_tree()
    locs = np.array([v.geoState.location for v in vertices])
    plt.scatter(*locs.T, c='k', lw=0)
    for v1, v2 in edges:
        x1, y1 = vertices[v1].geoState.location
        x2, y2 = vertices[v2].geoState.location
        plt.plot([x1, x2], [y1, y2], c='green')

    history = simulation.get_history()

    # history.animate_geo_history()
    # history.plot_geo_history()

    h = history.feature_history
    from src.beast_interface import write_nexus
    write_nexus(h[-1], path = 'data/simu.nex')

