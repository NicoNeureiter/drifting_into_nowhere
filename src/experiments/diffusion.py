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
    if not os.path.exists('data/diffusion'):
        os.mkdir('data/diffusion')
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/diffusion'):
        os.mkdir('results/diffusion')

    logging.basicConfig(filename='results/diffusion/nowhere.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('\n################################  NEW RUN  ################################')

    SCRIPT_PATH = 'src/beast_pipeline.sh'
    XML_PATH = 'data/diffusion/nowhere.xml'
    TREE_PATH = 'data/diffusion/nowhere.tree'

    # Experiment settings
    N_RUNS = 50

    # Simulation Parameters
    N_STEPS = 200
    N_FEATURES = 50

    RATE_OF_CHANGE = 0.1
    STEP_MEAN = [0., 0.]
    STEP_VARIANCE = 1.
    P_SPLIT = .3

    # Analysis Parameters
    CHAIN_LENGTH = 1000000
    BURNIN = 10000
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
        os.system('bash {script} {hpd} {burnin} {cwd}'.format(
            script=SCRIPT_PATH, hpd=HPD, burnin=BURNIN, cwd='data/diffusion/'))

        # Evaluate the results
        okcool = check_root_in_hpd(TREE_PATH, HPD, ax=ax)
        okcools += okcool

        logging.info('Run %i: %s' % (i_run, okcool))

        # Plot the true (simulated) evolution
        plot_walk(simulation, show_path=False, show_tree=True, ax=ax)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/diffusion/run_%i.pdf' % i_run, format='pdf')


    print(okcools)
    print(okcools / N_RUNS)

    logging.info('Experiments finished...\n\n'
                 '################################  RESULTS  ################################')
    logging.info('Successes: %i', okcools)
    logging.info('Probability coverage (%i hpd): %.4f', HPD, (okcools / N_RUNS))