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
from src.util import normalize


if __name__ == '__main__':
    if not os.path.exists('data/beast'):
        os.mkdir('data/beast')
    ax = plt.gca()

    NEXUS_PATH = 'data/nowhere.nex'
    LOCATIONS_PATH = 'data/nowhere_locations.txt'
    XML_PATH = 'data/beast/nowhere.xml'
    SCRIPT_PATH = 'src/beast_pipeline.sh'
    TREE_PATH = 'data/beast/nowhere.tree'

    # Simulation Parameters
    N_STEPS = 50
    N_FEATURES = 50

    RATE_OF_CHANGE = 0.1
    TOTAL_DRIFT = 1.
    TOTAL_DIFFUSION = 5.
    DRIFT_DENSITY = 1.
    DRIFT_DIRECTION = normalize([1., 0.])
    P_SPLIT = 1.

    # Analysis Parameters
    CHAIN_LENGTH = 20000
    BURNIN = 1000
    HPD = 80

    _step_drift = TOTAL_DRIFT / (N_STEPS * DRIFT_DENSITY)
    step_mean = _step_drift * DRIFT_DIRECTION
    print(_step_drift)
    print(DRIFT_DIRECTION)
    print(step_mean)
    step_var = TOTAL_DIFFUSION ** 2. / (N_STEPS)
    print(step_var)

    backbone = []
    def simulate_backbone():
        sim = Simulation(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, P_SPLIT)
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

                if random.random() < P_SPLIT:
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

    N_RUNS = 50
    for _ in range(N_RUNS):
        # Run Simulation
        simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, P_SPLIT)
        simulation.run(N_STEPS)
        # simulation = simulate_backbone()

        # # Create an XML file as input for the BEAST analysis
        # write_beast_xml(simulation, path=XML_PATH, chain_length=CHAIN_LENGTH, fix_root=False)
        #
        # # Run the BEAST analysis + summary of results (treeannotator)
        # os.system('bash {script} {hpd} {burnin} {cwd}'.format(
        #     script=SCRIPT_PATH,
        #     hpd=HPD,
        #     burnin=BURNIN,
        #     cwd=os.getcwd()+'/data/beast/'
        # ))
        #
        # # Evaluate the results
        # okcool = check_root_in_hpd(TREE_PATH, HPD, root=[0, 0], ax=ax)
        # print('\n\nOk cool: %r' % okcool)

        locs = simulation.get_locations()
        mean = np.mean(locs, axis=0)
        std = np.mean((locs - N_STEPS*step_mean)**2., axis=0)**.5
        std_sum += std
        mean_sum += mean

        print('std: ', std)
        print('mean: ', mean)

        # Plot the true (simulated) evolution
        # plot_walk(simulation, show_path=False, show_tree=True, ax=ax)

        plt.scatter(*locs.T, alpha=0.2, lw=0, c='darkblue')


    print('Average std: ', std_sum / N_RUNS)
    print('Average mean: ', mean_sum / N_RUNS)

    # plt.axis('off')
    plt.tight_layout()


    # backbone = np.array([s.geoState.location for s in backbone])
    # plt.plot(backbone[:, 0], backbone[:, 1], c='darkred')

    plt.axis('equal')
    plt.show()
    # plt.savefig('results/nowhere.pdf', format='pdf')