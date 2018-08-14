#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import logging
import pickle
import datetime
import itertools

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from src.evaluation import check_root_in_hpd
from src.simulation import Simulation
from src.beast_interface import write_beast_xml
from src.util import normalize
from src.plotting import plot_walk


def run_experiments(custom_params):
    today = datetime.date.today()
    params_str = '__'.join([str(v) for tpl in custom_params for v in tpl])
    BASE_DIRECTORY = 'experiments/migration/%s/%s/' % (today, params_str)
    os.makedirs(os.path.dirname(BASE_DIRECTORY), exist_ok=True)

    SCRIPT_PATH = 'src/beast_pipeline.sh'
    XML_PATH = os.path.join(BASE_DIRECTORY, 'nowhere.xml')
    TREE_PATH = os.path.join(BASE_DIRECTORY, 'nowhere.tree')

    logging.basicConfig(filename=BASE_DIRECTORY+'migration.log',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    params = {
        'N_RUNS': 10,
        'N_STEPS': 100,
        'N_FEATURES': 50,
        'RATE_OF_CHANGE': 0.1,
        'TOTAL_DRIFT': 1.,
        'TOTAL_DIFFUSION': 1.,
        'DRIFT_DENSITY': 1.,
        'DRIFT_DIRECTION': normalize([0.1, -0.1]),
        'P_SPLIT': .5,
        'CHAIN_LENGTH': 100000,
        'BURNIN': 50000,
        'HPD': 90,
    }
    params.update(custom_params)

    N_RUNS = params['N_RUNS']
    N_STEPS = params['N_STEPS']
    N_FEATURES = params['N_FEATURES']
    RATE_OF_CHANGE = params['RATE_OF_CHANGE']
    TOTAL_DRIFT = params['TOTAL_DRIFT']
    TOTAL_DIFFUSION = params['TOTAL_DIFFUSION']
    DRIFT_DENSITY = params['DRIFT_DENSITY']
    DRIFT_DIRECTION = params['DRIFT_DIRECTION']
    P_SPLIT = params['P_SPLIT']
    CHAIN_LENGTH = params['CHAIN_LENGTH']
    BURNIN = params['BURNIN']
    HPD = params['HPD']

    okcools = 0
    logging.info('Started exmperiments for parameters: %s', dict(custom_params))
    for i_run in range(N_RUNS):

        fig, ax = plt.subplots()

        step_drift = TOTAL_DRIFT / (N_STEPS * DRIFT_DENSITY)
        step_mean = step_drift * DRIFT_DIRECTION
        step_var = TOTAL_DIFFUSION ** 2. / N_STEPS

        # Run Simulation
        simulation = Simulation(N_FEATURES, RATE_OF_CHANGE, step_mean, step_var, P_SPLIT,
                                drift_frequency=DRIFT_DENSITY)
        simulation.run(N_STEPS)

        # Create an XML file as input for the BEAST analysis
        write_beast_xml(simulation, path=XML_PATH, chain_length=CHAIN_LENGTH)

        # Run the BEAST analysis + summary of results (treeannotator)
        os.system('bash "{script}" {hpd} {burnin} "{cwd}"'.format(
            script=SCRIPT_PATH, hpd=HPD, burnin=BURNIN, cwd=BASE_DIRECTORY))

        # Evaluate the results
        okcool = check_root_in_hpd(TREE_PATH, HPD, root=[0, 0], ax=ax)
        okcools += okcool

        # Plot the true (simulated) evolution
        plot_walk(simulation, show_path=False, show_tree=True, ax=ax,
                  savefig=BASE_DIRECTORY+'nowhere_%i.pdf' % i_run)

        logging.info('Run %i: %s' % (i_run, okcool))

    coverage = okcools / N_RUNS
    res = {
        'hpd_coverage': coverage,
        'hits': okcools,
        'params': params
    }.update(custom_params)

    logging.info('Successes: %i', okcools)
    logging.info('Probability coverage (%i hpd): %.4f', HPD, coverage)

    return res


if __name__ == '__main__':
    os.makedirs(os.path.dirname('results/migration/'), exist_ok=True)

    logging.basicConfig(filename='results/migration/migration.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('\n##############################  NEW EXPERIMENT ##############################')

    param_map = {
        'TOTAL_DRIFT': [0., 0.25, 0.5, 1.0, 1.25, 1.5],
        'DRIFT_DENSITY': [0.1, 0.5, 1.]
    }

    param_names, param_values = zip(*param_map.items())
    param_grid = itertools.product(*param_values)
    named_param_grid = [tuple(zip(param_names, values)) for values in param_grid]

    run = delayed(run_experiments)
    results_list = Parallel(n_jobs=3)(
        run(named_params) for named_params in named_param_grid
    )

    results = {named_params: res
               for named_params, res in zip(named_param_grid, results_list)}

    with open('results/migration/results.pkl', 'wb') as res_file:
        pickle.dump(results, res_file)