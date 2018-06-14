#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import logging

import matplotlib.pyplot as plt

from src.evaluation import check_root_in_hpd
from src.simulation import Simulation
from src.beast_interface import write_beast_xml
from src.plotting import plot_walk


if __name__ == '__main__':
    if not os.path.exists('data/beast'):
        os.mkdir('data/beast')
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/no_drift'):
        os.mkdir('results/no_drift')

    logging.basicConfig(filename='results/no_drift/nowhere.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    NEXUS_PATH = 'data/nowhere.nex'
    LOCATIONS_PATH = 'data/nowhere_locations.txt'
    XML_PATH = 'data/beast/nowhere.xml'
    SCRIPT_PATH = 'src/beast_pipeline.sh'
    TREE_PATH = 'data/beast/nowhere.tree'

    # Experiment settings
    N_RUNS = 50

    # Simulation Parameters
    N_STEPS = 200
    N_FEATURES = 50

    RATE_OF_CHANGE = 0.1
    STEP_MEAN = [0.1, -0.1]
    STEP_VARIANCE = 1.
    P_SPLIT = .3

    # Analysis Parameters
    CHAIN_LENGTH = 2000000
    BURNIN = 20000
    HPD = 90

    okcools = 0

    for i_run in range(N_RUNS):
        fig, ax = plt.subplots()

        # Run Simulation
        simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, STEP_MEAN, STEP_VARIANCE, P_SPLIT)
        simulation.run(N_STEPS)

        # Create an XML file as input for the BEAST analysis
        write_beast_xml(simulation, path=XML_PATH, chain_length=CHAIN_LENGTH)

        # Run the BEAST analysis + summary of results (treeannotator)
        os.system('bash {script} {hpd} {burnin}'.format(script=SCRIPT_PATH,
                                                        hpd=HPD,
                                                        burnin=BURNIN))

        # Evaluate the results
        okcool = check_root_in_hpd(TREE_PATH, HPD, ax=ax)
        okcools += okcool

        logging.info('Run %i: %s' % (i_run, okcool))

        # Plot the true (simulated) evolution
        plot_walk(simulation, show_path=False, show_tree=True, ax=ax)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/no_drift/run_%i.pdf' % i_run, format='pdf')


    print(okcools)
    print(okcools / N_RUNS)

    logging.info('Experiments finished...\n\n################  RESULTS  ################')
    logging.info('Successes: %i', okcools)
    logging.info('Probability coverage (%i hpd): %.4f', HPD, (okcools / N_RUNS))