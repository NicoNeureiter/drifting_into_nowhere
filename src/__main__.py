#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

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
    N_STEPS = 40
    N_FEATURES = 50

    RATE_OF_CHANGE = 0.1
    STEP_MEAN = [.01, 0.]
    STEP_VARIANCE = 1.
    P_SPLIT = .3

    # Run Simulation
    simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, STEP_MEAN, STEP_VARIANCE, P_SPLIT)
    simulation.run(N_STEPS)

    # Create an XML file as input for the BEAST analysis
    write_beast_xml(simulation, path=XML_PATH)

    # Run the BEAST analysis + summary of results (treeannotator)
    os.system('bash %s' % SCRIPT_PATH)

    # Evaluate the results
    okcool = check_root_in_hpd(TREE_PATH)
    print('\n\nOk cool: %r' % okcool)

    # Plot the true (simulated) evolution
    plot_walk(simulation, show_tree=True)
