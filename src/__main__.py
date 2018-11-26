#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from src.simulation import Simulation, SimulationBackbone
from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
from src.plotting import plot_walk, animate_walk, plot_tree, plot_hpd, plot_root
from src.util import (norm, normalize, total_diffusion_2_step_var,
                      total_drift_2_step_drift, mkpath)
from src.tree import Node
from src.config import *

if __name__ == '__main__':
    mkpath('data/beast/')
    WORKING_DIR = 'data/beast/'
    XML_PATH = WORKING_DIR + 'nowhere.xml'
    TREE_PATH = WORKING_DIR + 'nowhere.tree'

    # Simulation Parameters
    ROOT = np.array([0., 0.])
    N_STEPS = 200
    N_FEATURES = 5

    N_EXPECTED_SOCIETIES = 30
    RATE_OF_CHANGE = 0.1
    TOTAL_DRIFT = 30.
    TOTAL_DIFFUSION = 1.
    DRIFT_DENSITY = 1.
    DRIFT_DIRECTION = normalize([1., .2])

    # Analysis Parameters
    CHAIN_LENGTH = 50000
    BURNIN = 5000
    HPD = 60

    # Inferred parameters
    step_var = total_diffusion_2_step_var(TOTAL_DIFFUSION, N_STEPS)
    _step_drift = total_drift_2_step_drift(TOTAL_DRIFT, N_STEPS, drift_density=DRIFT_DENSITY)
    step_mean = _step_drift * DRIFT_DIRECTION
    p_split = N_EXPECTED_SOCIETIES / N_STEPS

    # Check parameter validity
    if True:
        assert 0 < RATE_OF_CHANGE <= 1
        assert 0 < DRIFT_DENSITY <= 1
        assert 0 <= p_split <= 1
        assert 0 < HPD < 100
        assert BURNIN < CHAIN_LENGTH

    backbone = []
    def simulate_backbone():
        sim = Simulation(N_FEATURES, RATE_OF_CHANGE, 3000.*DRIFT_DIRECTION, step_var, p_split=0.3)
        # sim.history = [sim.root]
        # sim.societies = [sim.root.create_child(),
        #                            sim.root.create_child()]
        sim.societies = [sim.root]
        global backbone
        backbone = [sim.root.geoState.location]

        for _ in range(10):
            sim.split(0)

            for _ in range(4):
                sim.societies[0].step()
                backbone.append(sim.societies[0].geoState.location)

        bb_societies = list(sim.societies)

        for i_child, child in enumerate(bb_societies):
            print(i_child)
            clade = {i_child: child}
            prev_steps = len(child.location_history)
            for _ in range(N_STEPS - prev_steps):
                if random.random() < p_split:
                    j1, s = random.choice(list(clade.items()))
                    sim.split(j1)
                    clade[j1] = sim.societies[j1]
                    j2 = sim.n_sites-1
                    clade[j2] = sim.societies[j2]

                for s in clade.values():
                    s.step(step_mean=(0, 0), step_cov=step_var)

        backbone = np.array(backbone)

        return sim

    # Run Simulation
    simulation = SimulationBackbone(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, p_split,
                                    drift_frequency=DRIFT_DENSITY, repulsive_force=0,
                                    backbone_steps=20)
    simulation.run(N_STEPS)

    # Create an XML file as input for the BEAST analysis
    simulation.root.write_beast_xml(output_path=XML_PATH, chain_length=CHAIN_LENGTH, movement_model='rrw')

    # Run BEAST analysis
    run_beast(working_dir=WORKING_DIR)
    run_treeannotator(HPD, BURNIN, working_dir=WORKING_DIR)

    # Plotting
    def get_node_drift(parent, child):
        return norm(child.geoState.step_mean)

    tree = load_tree_from_nexus(tree_path=TREE_PATH)
    plot_root(tree.location)
    plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    plot_hpd(tree, HPD)
    plot_tree(simulation.root, color_fun=get_node_drift, lw=0.01)
    # plot_tree(tree, color='k', lw=0.01)
    plt.show()

    # Print whether root is covered by HPD region
    print('Covered:', tree.root_in_hpd(ROOT, HPD))
