#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt

from src.simulation.vector_simulation import VectorWorld, VectorState, BackboneState
from src.simulation.simulation import run_simulation
from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
from src.plotting import (plot_tree, plot_root, plot_hpd)
from src.tree import get_edge_heights

from src.util import (norm, normalize, total_diffusion_2_step_var,
                      total_drift_2_step_drift, experiment_preperations)
from src.config import *

if __name__ == '__main__':
    WORKING_DIR = 'data/beast/'
    XML_PATH = WORKING_DIR + 'nowhere.xml'
    TREE_PATH = WORKING_DIR + 'nowhere.tree'

    exp_dir = experiment_preperations(WORKING_DIR)

    # Simulation Parameters
    ROOT = np.array([0., 0.])
    N_STEPS = 200
    N_FEATURES = 5

    N_EXPECTED_SOCIETIES = 3000
    CLOCK_RATE = 1.
    TOTAL_DRIFT = 100.
    TOTAL_DIFFUSION = .1
    DRIFT_DENSITY = 1.0
    DRIFT_DIRECTION = normalize([1., 0.])

    # Analysis Parameters
    CHAIN_LENGTH = 100000
    BURNIN = 50000
    HPD = 80

    BACKBONE = 0
    N_BB_STEPS = 20

    # Inferred parameters
    step_var = total_diffusion_2_step_var(TOTAL_DIFFUSION, N_STEPS)
    if BACKBONE:
        _step_drift = total_drift_2_step_drift(TOTAL_DRIFT, N_BB_STEPS, drift_density=DRIFT_DENSITY)
    else:
        _step_drift = total_drift_2_step_drift(TOTAL_DRIFT, N_STEPS, drift_density=DRIFT_DENSITY)
    step_mean = _step_drift * DRIFT_DIRECTION
    p_split = N_EXPECTED_SOCIETIES / N_STEPS

    # Check parameter validity
    if False:
        assert 0 < RATE_OF_CHANGE <= 1
        assert 0 < DRIFT_DENSITY <= 1
        # assert 0 <= p_split <= 1
        assert 0 < HPD < 100
        assert BURNIN < CHAIN_LENGTH

    print(N_STEPS)
    print(step_mean, step_var)

    # Run Simulation
    p0 = np.zeros(2)
    world = VectorWorld(capacity=1.2 * N_EXPECTED_SOCIETIES)
    root = VectorState(world, p0, step_mean, step_var, CLOCK_RATE, p_split,
                       drift_frequency=DRIFT_DENSITY)
    run_simulation(N_STEPS, root, world, 30)

    # # Create an XML file as input for the BEAST analysis
    # root.write_beast_xml(output_path=XML_PATH, chain_length=CHAIN_LENGTH, movement_model='rrw')
    #
    # # Run BEAST analysis
    # run_beast(working_dir=WORKING_DIR)
    # run_treeannotator(HPD, BURNIN, working_dir=WORKING_DIR)
    #
    # tree = load_tree_from_nexus(tree_path=TREE_PATH)
    # plot_root(tree.location)
    # plot_hpd(tree, HPD)



    tree = root
    print(tree.n_leafs())
    plot_tree(tree, color_fun=get_edge_heights)
    plot_root(tree.location, COLOR_ROOT_TRUE)

    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()

    # Print whether root is covered by HPD region
    print('Covered:', tree.root_in_hpd(ROOT, HPD))
