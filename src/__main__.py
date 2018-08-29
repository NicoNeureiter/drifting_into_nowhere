#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import check_root_in_hpd
from src.simulation import Simulation
from src.beast_interface import write_beast_xml, write_nexus, write_locations
from src.plotting import plot_walk, animate_walk
from src.util import (normalize, total_diffusion_2_step_var,
                      total_drift_2_step_drift, grey)
from src.analyze_tree import plot_tree, extract_newick_from_nexus, TURQUOISE, PINK
from src.tree import Node

PINK = 'purple'

if __name__ == '__main__':
    os.makedirs(os.path.dirname('data/beast/'), exist_ok=True)
    ax = plt.gca()

    NEXUS_PATH = 'data/nowhere.nex'
    LOCATIONS_PATH = 'data/nowhere_locations.txt'
    XML_PATH = 'data/beast/nowhere.xml'
    SCRIPT_PATH = 'src/beast_pipeline.sh'
    TREE_PATH = 'data/beast/nowhere.tree'

    # Simulation Parameters
    N_STEPS = 30
    N_FEATURES = 20

    N_EXPECTED_SOCIETIES = 15
    RATE_OF_CHANGE = 0.1
    TOTAL_DRIFT = 3.
    TOTAL_DIFFUSION = .5
    DRIFT_DENSITY = 1.
    DRIFT_DIRECTION = normalize([1., 1.])
    # p_split = 1.
    p_split = min(1., N_EXPECTED_SOCIETIES / N_STEPS)

    # Analysis Parameters
    CHAIN_LENGTH = 50000
    BURNIN = 1000
    HPD = 80

    # Check parameter validity
    assert 0 < RATE_OF_CHANGE <= 1
    assert 0 < DRIFT_DENSITY <= 1
    assert 0 < p_split <= 1
    assert 0 < HPD < 100
    assert BURNIN < CHAIN_LENGTH

    step_var = total_diffusion_2_step_var(TOTAL_DIFFUSION, N_STEPS)
    _step_drift = total_drift_2_step_drift(TOTAL_DRIFT, N_STEPS, drift_density=DRIFT_DENSITY)
    step_mean = _step_drift * DRIFT_DIRECTION

    print(p_split)
    print(step_var)

    backbone = []

    def simulate_backbone():
        sim = Simulation(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, p_split)
        # sim.history = [sim.root]
        # sim.societies = [sim.root.create_child(),
        #                            sim.root.create_child()]
        sim.societies = [sim.root]
        global backbone
        backbone = [sim.root]

        for _ in range(20):
            sim.split(0)

            for _ in range(4):
                sim.societies[0].step()

            backbone.append(sim.societies[0])

        bb_societies = list(sim.societies)

        for i_child, child in enumerate(bb_societies):
            print(i_child)
            clade = {i_child: child}
            prev_steps = len(child.location_history)
            for _ in range(N_STEPS - prev_steps):

                # if i_child == 1:
                #     p_split = P_SPLIT
                #     var = step_var
                # else:
                #     p_split = P_SPLIT
                #     var = step_var

                if random.random() < p_split:
                    j1, s = random.choice(list(clade.items()))
                    sim.split(j1)
                    clade[j1] = sim.societies[j1]
                    j2 = sim.n_sites-1
                    clade[j2] = sim.societies[j2]

                for s in clade.values():
                    s.step(step_mean=(0, 0), step_cov=step_var)

            # print(clade)

        # print(sim.societies)

        return sim

    std_sum = 0
    mean_sum = 0

    # N_RUNS = 1
    # for _ in range(N_RUNS):

    # Run Simulation
    simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, p_split,
                            drift_frequency=DRIFT_DENSITY, repulsive_force=0.2)
    simulation.run(N_STEPS)
    # simulation = simulate_backbone()

    # Create an XML file as input for the BEAST analysis
    write_beast_xml(simulation, path=XML_PATH, chain_length=CHAIN_LENGTH, fix_root=False)

    # Run the BEAST analysis + summary of results (treeannotator)
    os.system('bash {script} {hpd} {burnin} {cwd}'.format(
        script=SCRIPT_PATH,
        hpd=HPD,
        burnin=BURNIN,
        cwd=os.getcwd()+'/data/beast/'
    ))

    # Evaluate the results
    okcool = check_root_in_hpd(TREE_PATH, HPD, root=[0, 0], ax=ax, alpha=0.1, color=PINK, zorder=2)
    print('\n\nOk cool: %r' % okcool)

    with open(TREE_PATH, 'r') as tree_file:
        nexus_str = tree_file.read()
        newick_str = extract_newick_from_nexus(nexus_str)
        tree = Node.from_newick(newick_str, location_key='location')
    plt.scatter(tree.location[0], tree.location[1], marker='*', c=PINK, s=500, zorder=3)

    locs = simulation.get_locations()
    mean = np.mean(locs, axis=0)
    std = np.mean((locs - N_STEPS*step_mean)**2., axis=0)**.5
    std_sum += std
    mean_sum += mean

    # print('std: ', std)
    # print('mean: ', mean)

    # Plot the true (simulated) evolution
    print(simulation.root.children)
    # plot_tree(simulation.root, lw=0.005,  #no_arrow=True,
    #           color=grey(.8))
    #           # color=(0.7, 0.8, 0.85), alpha=1.)


    def show(xlim=None, ylim=None):
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.axis('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()


    ax = plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    plt.scatter(0, 0, marker='*', c='teal', s=500, zorder=3)
    xlim = plt.xlim()
    ylim = plt.ylim()
    show()

    ax = plot_walk(simulation, show_path=True, show_tree=False, ax=plt.gca())
    plt.scatter(0, 0, marker='*', c='teal', s=500, zorder=3)
    show()

    ax = plot_walk(simulation, show_path=False, show_tree=True, ax=plt.gca(), alpha=0.2)
    plt.scatter(0, 0, marker='*', c='teal', s=500, zorder=3)
    show(xlim, ylim)

    ax = plot_walk(simulation, show_path=False, show_tree=False, ax=plt.gca())
    plt.scatter(0, 0, marker='*', c='teal', s=500, zorder=3)
    show(xlim, ylim)

    ax = plot_walk(simulation, show_path=False, show_tree=False, ax=plt.gca())
    show(xlim, ylim)

    # print('Average std: ', std_sum / N_RUNS)
    # print('Average mean: ', mean_sum / N_RUNS)


    # backbone = np.array([s.geoState.location for s in backbone])
    # plt.plot(backbone[:, 0], backbone[:, 1], c='darkred')


    # plt.savefig('results/nowhere.pdf', format='pdf')