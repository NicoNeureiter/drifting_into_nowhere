#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt

from src.simulation.simulation import run_simulation
from src.simulation.vector_simulation import VectorWorld, VectorState, BackboneState
from src.beast_interface import (run_beast, run_treeannotator,
                                 load_tree_from_nexus, load_trees)
from src.plotting import (plot_tree, plot_root, plot_hpd, plot_walk,
                          plot_backbone_splits, plot_backbone_clades,
                          plot_posterior_scatter)
from src.tree import get_edge_heights
from src.evaluation import eval_bias, eval_rmse, eval_stdev
from src.util import (norm, normalize, total_diffusion_2_step_var,
                      total_drift_2_step_drift, experiment_preperations,
                      birth_death_expectation)
from src.config import *

if __name__ == '__main__':
    WORKING_DIR = 'data/beast/'
    XML_PATH = WORKING_DIR + 'nowhere.xml'
    TREE_PATH = WORKING_DIR + 'nowhere.tree'
    TREES_PATH = WORKING_DIR + 'nowhere.trees'

    exp_dir = experiment_preperations(WORKING_DIR)

    # plt.figure(figsize=(14, 14), dpi=100)

    # Simulation Parameters
    ROOT = np.array([0., 0.])
    N_STEPS = 1000  # "years"

    N_EXPECTED_LEAVES = 100
    MIN_LEAVES, MAX_LEAVES = 0.4*N_EXPECTED_LEAVES, 2.*N_EXPECTED_LEAVES
    TURNOVER = 0.2         # 0 == No deaths, 1 == Inf births and deaths
    CLOCK_RATE = 1.0
    TOTAL_DRIFT = 0000.0 # "kilometers"
    TOTAL_DIFFUSION = 2000.0  # "kilometers"
    DRIFT_DENSITY = 1.0
    # DRIFT_DIRECTION = normalize([1., 2.])
    DRIFT_DIRECTION = normalize([1., 0])

    # Analysis Parameters
    CHAIN_LENGTH = 20000000
    BURNIN = 100000
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

    eff_div_rate = np.log(N_EXPECTED_LEAVES) / N_STEPS
    birth_rate = eff_div_rate / (1 - TURNOVER)
    death_rate = birth_rate * TURNOVER

    # birth_death_expectation(birth_rate, death_rate, N_STEPS, vrange=(MIN_LEAVES, MAX_LEAVES))

    # Check parameter validity
    if False:
        assert 0 < RATE_OF_CHANGE <= 1
        assert 0 < DRIFT_DENSITY <= 1
        assert 0 <= turnover < 1
        assert 0 <= death_rate < birth_rate <= 1
        assert 0 < HPD < 100
        assert BURNIN < CHAIN_LENGTH

    print('Number of steps:', N_STEPS)
    print('Step mean:', step_mean)
    print('Step variance:', step_var)
    print('Effective diversification rate:', eff_div_rate)
    print('Turnover:', TURNOVER)
    print('Birth rate:', birth_rate)
    print('Death rate:', death_rate)

    valid_tree = False
    while not valid_tree:
        # Run Simulation
        p0 = np.zeros(2)
        world = VectorWorld(capacity=1.5 * N_EXPECTED_LEAVES)
        # root = BackboneState(world, p0, step_mean, step_var, CLOCK_RATE, p_split,
        #                    drift_frequency=DRIFT_DENSITY, bb_stop=N_STEPS/3)
        root = VectorState(world, p0, step_mean, step_var, CLOCK_RATE, birth_rate,
                           drift_frequency=DRIFT_DENSITY, death_rate=death_rate)  #p_split/10)
        run_simulation(N_STEPS, root, world)

        tree_simu = root

        n_leafs = len([n for n in tree_simu.iter_leafs() if n.height == N_STEPS])
        valid_tree = (MIN_LEAVES < n_leafs < MAX_LEAVES)
        for c in root.children:
            if not any(n.height == N_STEPS for n in c.iter_leafs()):
                valid_tree = False

    # print([n.height for n in tree_simu.iter_leafs()])
    tree_simu.drop_fossils()
    print('Number of leafs:', tree_simu.n_leafs())
    # print([n.height for n in tree_simu.iter_leafs()])

    # print('Number of leafs extant:', len(extant_leafs))
    # print('Number of leafs extinct:', len(extinct_leafs))

    # exit()

    # Run BEAST analysis
    if 1:
        # Create an XML file as input for the BEAST analysis
        root.write_beast_xml(output_path=XML_PATH, chain_length=CHAIN_LENGTH, movement_model='rrw',
                             drift_prior_std=1.)
        run_beast(working_dir=WORKING_DIR)
        tree_beast = run_treeannotator(HPD, BURNIN, working_dir=WORKING_DIR)
        # # plot_tree(tree_beast, alpha=0.5, lw=1.5, color=COLOR_ROOT_EST)
        # plot_hpd(tree_beast, HPD, label='Reconstructed {}%% HPD'.format(HPD))
        plot_root(tree_beast.location, label='Reconstructed root')

    # ax = plot_walk(tree_simu, lw=2)
    plot_tree(tree_simu, color='k')  #color_fun=get_edge_heights, ax=ax)
    plt.scatter(*tree_simu.get_leaf_locations().T, c='orange', lw=0, zorder=5)
    plot_root(tree_simu.location, color=COLOR_ROOT_TRUE, label='Simulated root')

    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout(pad=0.)

    trees = load_trees(TREES_PATH)
    plot_posterior_scatter(trees, s=8., alpha=1.)

    root = tree_simu.location
    mse = eval_rmse(root, trees)
    bias = eval_bias(root, trees)
    std = eval_stdev(root, trees)

    print('MSE', mse)
    print('Bias', bias)
    print('Stdev', std)
    print('SQRT(bias^2 + stdev^2) =', np.sqrt(bias**2 + std**2))

    plt.show()
    exit()

    # handles, labels = ax.get_legend_handles_labels()
    # labels, handles = zip(*reversed(list(zip(labels, handles))))
    # ax.legend(handles, labels, markerscale=1., fontsize=28, loc=4)
    # plt.savefig('results/poster/drift_reconstruction.png')
    # plt.savefig('results/poster/drift_simulation.png')
    plt.show()

    for parent, child in tree_simu.iter_edges():
        plt.plot([parent.location[0], child.location[0]], [-parent.height, -child.height], c='grey')
    for parent, child in tree_beast.iter_edges():
        plt.plot([parent.location[0], child.location[0]], [-parent.height, -child.height], c='darkred')
    plt.show()

