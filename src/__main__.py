#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import matplotlib.pyplot as plt

from src.evaluation import check_root_in_hpd
from src.simulation import Simulation
from src.beast_interface import write_beast_xml
from src.plotting import plot_walk, animate_walk


if __name__ == '__main__':
    if not os.path.exists('data/beast'):
        os.mkdir('data/beast')

    NEXUS_PATH = 'data/nowhere.nex'
    LOCATIONS_PATH = 'data/nowhere_locations.txt'
    XML_PATH = 'data/beast/nowhere.xml'
    SCRIPT_PATH = 'src/beast_pipeline.sh'
    TREE_PATH = 'data/beast/nowhere.tree'

    # Simulation Parameters
    N_STEPS = 200
    N_FEATURES = 50

    RATE_OF_CHANGE = 0.1
    STEP_MEAN = [0.1, -0.1]
    STEP_VARIANCE = 1.
    P_SPLIT = .3

    # Analysis Parameters
    CHAIN_LENGTH = 100000
    BURNIN = 10000
    HPD = 95

    ax = plt.gca()

    # Run Simulation
    simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, STEP_MEAN, STEP_VARIANCE, P_SPLIT)
    simulation.run(N_STEPS)

    # Create an XML file as input for the BEAST analysis
    write_beast_xml(simulation, path=XML_PATH, chain_length=CHAIN_LENGTH)

    # Run the BEAST analysis + summary of results (treeannotator)
    os.system('bash {script} {hpd} {burnin}'.format(
        script=SCRIPT_PATH,
        hpd=HPD,
        burnin=BURNIN
    ))

    # Evaluate the results
    okcool = check_root_in_hpd(TREE_PATH, HPD, ax=ax)
    print('\n\nOk cool: %r' % okcool)

    # Plot the true (simulated) evolution
    plot_walk(simulation, show_path=False, show_tree=True, ax=ax)

    plt.axis('off')
    plt.tight_layout()

    plt.show()
    # plt.savefig('results/nowhere.pdf', format='pdf')