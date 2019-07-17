#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging

import matplotlib.pyplot as plt

from src.simulation.vector_simulation import VectorWorld, VectorState, BackboneState
from src.simulation.simulation import run_simulation
from src.beast_interface import run_beast, run_treeannotator, \
    load_tree_from_nexus, load_trees
from src.plotting import (plot_tree, plot_root, plot_hpd, plot_walk, plot_posterior_scatter,
                           COLOR_ROOT_EST, COLOR_ROOT_TRUE)
from src.tree import get_edge_heights

from src.util import (norm, normalize, total_diffusion_2_step_var,
                      total_drift_2_step_drift, experiment_preperations)
from src.config import *

if __name__ == '__main__':
    WORKING_DIR = 'experiments/visualizations/'
    XML_PATH = WORKING_DIR + 'nowhere.xml'
    TREE_PATH = WORKING_DIR + 'nowhere.tree'

    exp_dir = experiment_preperations(WORKING_DIR)
    logging.basicConfig(filename=exp_dir + 'migration.log',
                        level=logging.DEBUG, filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        #  , figsize=(14, 14), dpi=100)

    # Simulation Parameters
    ROOT = np.array([0., 0.])
    N_STEPS = 100

    N_EXPECTED_SOCIETIES = 40
    CLOCK_RATE = 1.0
    TOTAL_DIFFUSION = 1.0
    DRIFT_DENSITY = 1.0
    DRIFT_DIRECTION = normalize([1., 1.])

    # Analysis Parameters
    CHAIN_LENGTH = 100000
    BURNIN = 10000
    HPD = 80

    BACKBONE = 0
    N_BB_STEPS = 20

    DRIFT_VALUES = [0., 1.0, 2.0, 3.0]
    # DRIFT_VALUES = [3.]

    for i, total_drift in enumerate(DRIFT_VALUES):
        for _ in range(1):

            # Inferred parameters
            step_var = total_diffusion_2_step_var(TOTAL_DIFFUSION, N_STEPS)
            _step_drift = total_drift_2_step_drift(total_drift, N_STEPS, drift_density=DRIFT_DENSITY)
            step_mean = _step_drift * DRIFT_DIRECTION
            p_split = N_EXPECTED_SOCIETIES / N_STEPS

            # Check parameter validity
            assert 0 < DRIFT_DENSITY <= 1
            assert 0 <= p_split
            assert 0 < HPD < 100
            assert BURNIN < CHAIN_LENGTH

            if p_split > 1:
                logging.warning("Not enough steps to reach the specified ´N_EXPECTED_SOCIETIES´!")

            # Run Simulation
            p0 = np.zeros(2)
            world = VectorWorld(capacity=1.2 * N_EXPECTED_SOCIETIES)
            root = VectorState(world, p0, step_mean, step_var, CLOCK_RATE, p_split,
                               drift_frequency=DRIFT_DENSITY)
            run_simulation(N_STEPS, root, world)
            tree_simu = root

            # Create an XML file as input for the BEAST analysis
            tree_simu.write_beast_xml(output_path=XML_PATH, chain_length=CHAIN_LENGTH, movement_model='rrw')

            cmap = plt.get_cmap('cividis')
            # Plot simulated tree
            # plot_walk(tree_simu, lw=2, ax=axes[0, i])
            plot_tree(tree_simu, lw=0.5, color_fun=get_edge_heights, ax=axes[0,i], cmap=cmap)
            axes[0, i].scatter(*tree_simu.get_leaf_locations().T, c=cmap(N_STEPS), lw=0, zorder=5, s=2)
            plot_root(tree_simu.location, color=COLOR_ROOT_TRUE, label='Simulated root', ax=axes[0, i])

            # Run BEAST analysis
            if 1:
                run_beast(working_dir=WORKING_DIR)
                tree_beast = run_treeannotator(HPD, BURNIN, working_dir=WORKING_DIR)
            #     # plot_tree(tree_beast, alpha=0.5, lw=1.5, color=COLOR_ROOT_EST)
            #     plot_hpd(tree_beast, HPD, label='Reconstructed {}%% HPD'.format(HPD))
            #     plot_root(tree_beast.location, label='Reconstructed root')

            # trees = load_trees(WORKING_DIR)
            plot_tree(tree_beast, lw=0.5, color_fun=get_edge_heights, ax=axes[1, i], cmap=cmap)  #, alpha=0.5)
            axes[1, i].scatter(*tree_simu.get_leaf_locations().T, c=cmap(N_STEPS), lw=0, zorder=5, s=2)
            plot_root(tree_beast.location, color=COLOR_ROOT_EST, label='Reconstructed root', ax=axes[1, i])
                      # s=100, alpha=0.4)

            # axes[0, i].xticks([])
            # axes[0, i].yticks([])
            # axes[1, i].xticks([])
            # axes[1, i].yticks([])

            axes[0, i].axhline(0., color='lightgrey', lw=1.)
            axes[0, i].axvline(0., color='lightgrey', lw=1.)
            axes[1, i].axhline(0., color='lightgrey', lw=1.)
            axes[1, i].axvline(0., color='lightgrey', lw=1.)



    # fig = plt.gcf()
    # fig.set_size_inches(14, 14)
    # fig.set_dpi(200)
    # plt.axis('off')
    # plt.axis('equal')
    plt.tight_layout(pad=0., w_pad=0, h_pad=0)
    plt.xticks([])
    plt.yticks([])

    # handles, labels = ax.get_legend_handles_labels()
    # labels, handles = zip(*reversed(list(zip(labels, handles))))
    # ax.legend(handles, labels, markerscale=1., fontsize=28, loc=4)
    # plt.savefig('results/poster/drift_reconstruction.png')
    # plt.savefig('results/poster/drift_simulation.png')
    # plt.legend()
    plt.show()

