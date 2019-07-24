#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import pandas as pd

from src.util import parse_arg

X_NAME_BY_SIM_TYPE = {
    'random_walk': 'total_drift',
    'constrained_expansion': 'cone_angle'
}

if __name__ == '__main__':
    simulation_type = parse_arg(1, 'random_walk', str)
    BASE_DIR = 'experiments/%s/' % simulation_type
    RESULTS_FNAME = 'results.csv'

    x_name = X_NAME_BY_SIM_TYPE[simulation_type]

    for experiment in os.listdir(BASE_DIR):
        experiment_path = os.path.join(BASE_DIR, experiment)
        results_path = os.path.join(experiment_path, RESULTS_FNAME)
        print(results_path)
        results = pd.read_csv(results_path)

        results = results.groupby(x_name)
        results_mean = results.mean()
        results_count = results.count()['i_repeat']
        # print(results_mean.index)
        print(results_count)
        # print(results_mean['bias_norm'].round(decimals=2).values)
        print()