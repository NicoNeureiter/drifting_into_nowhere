#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import random

import matplotlib.pyplot as plt

from src.simulation import Simulation, SimulationBackbone
from src.plotting import (plot_tree, plot_tree_topology)

from src.util import (norm, normalize, total_diffusion_2_step_var,
                      total_drift_2_step_drift, mkpath)
from src.config import *

def experiment_preperations(work_dir):
    import shutil
    import datetime

    # Ensure working directory exists
    now = datetime.datetime.now()
    exp_dir = os.path.join(work_dir, 'experiment_logs_%s/' % now)
    mkpath(exp_dir)

    # Safe state of current file and config to the experiment folder
    shutil.copy(__file__, exp_dir)
    shutil.copy('src/config.py', exp_dir)

    # Generate random see and log it
    seed = random.randint(0, 1e9)
    seed = 115522826
    print(seed)
    with open(os.path.join(exp_dir, 'seed'), 'w') as seed_file:
        seed_file.write(str(seed))

    return exp_dir


if __name__ == '__main__':

    WORKING_DIR = 'data/beast/'
    XML_PATH = WORKING_DIR + 'nowhere.xml'
    TREE_PATH = WORKING_DIR + 'nowhere.tree'

    exp_dir = experiment_preperations(WORKING_DIR)

    # Simulation Parameters
    ROOT = np.array([0., 0.])
    N_STEPS = 500
    N_FEATURES = 5

    N_EXPECTED_SOCIETIES = 500
    RATE_OF_CHANGE = 0.1
    TOTAL_DRIFT = 0.
    TOTAL_DIFFUSION = 1.
    DRIFT_DENSITY = 1.
    DRIFT_DIRECTION = normalize([1., 0.])

    # Analysis Parameters
    CHAIN_LENGTH = 500000
    BURNIN = 20000
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


    # Run Simulation
    if BACKBONE:
        simulation = SimulationBackbone(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, p_split,
                                        drift_frequency=DRIFT_DENSITY, backbone_steps=N_BB_STEPS)
    else:
        simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, p_split,
                                drift_frequency=DRIFT_DENSITY, repulsive_force=.5,
                                p_settle=0.)

    simulation.run(N_STEPS)

    # # Create an XML file as input for the BEAST analysis
    # simulation.root.write_beast_xml(output_path=XML_PATH, chain_length=CHAIN_LENGTH, movement_model='rrw')
    #
    # # # Run BEAST analysis
    # run_beast(working_dir=WORKING_DIR)
    # run_treeannotator(HPD, BURNIN, working_dir=WORKING_DIR)

    # Plotting
    def get_node_drift(parent, child):
        return norm(child.geoState.step_mean)


    # tree = load_tree_from_nexus(tree_path=TREE_PATH)
    # plot_root(tree.location)
    # plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    # animate_walk(simulation)
    # plot_hpd(tree, HPD)
    # # plot_tree(simulation.root, color_fun=get_node_drift, lw=0.01)
    # plot_height(simulation.root)

    tree = simulation.root
    stree = tree.get_subtree([0,0])
    plot_tree(tree, color='lightgrey')
    plot_tree(stree, color='k')
    plt.show()
    plot_tree_topology(stree)



    # simulation_plots(simulation, tree, save_path='results/simu_backbone/3/')

    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()

    # Print whether root is covered by HPD region
    print('Covered:', tree.root_in_hpd(ROOT, HPD))
