#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

from collections import OrderedDict
from sklearn.model_selection import ParameterGrid

from src.util import mkpath, experiment_preperations, touch


CHECKLIST_FILE_NAME = 'checklist.txt'
RESULTS_FILE_NAME = 'results.csv'


def format_params(params):
    return ','.join(['%s=%s' % (k,v) for k, v in params.items()])


class Experiment(object):

    """Experiment class, which handles iteration over parameter grids,
    uniform organization of results, overview via csv.

    Attributes:
        pipeline (list[callable]): The sequential composition of these functions
            define the experiment pipline. All functions in the pipeline get
             the parameters and the results from the previous function as input
             arguments.
        fixed_params (OrderedDict): A dictionary, mapping each fixed parameter
            to a value.
        variable_param_options (OrderedDict): A dictionary, mapping each variable
            parameter to the range of values which should be evaluated.
        n_repeat (int): Number of times every parameter setting is run.
        eval_metrics (list): Names of the evaluation metrics (for header).
            TODO add evaluation functions as arguments?
        working_directory (str): The path to the working directory in which the
            temporary files and final results are stored.
    """

    def __init__(self, pipeline, fixed_params, variable_param_options,
                 eval_metrics, n_repeat, working_directory):
        self.pipeline = pipeline
        self.fixed_params = OrderedDict(fixed_params)
        self.variable_param_options = OrderedDict(variable_param_options)
        self.eval_metrics = eval_metrics
        self.working_directory = working_directory
        self.n_repeat = n_repeat
        mkpath(working_directory)

        # Handle repetitions as a variable parameter
        self.variable_param_options['i_repeat'] = list(range(n_repeat))

    @property
    def var_param_names(self):
        return list(self.variable_param_options.keys())


    @property
    def columns(self):
        return self.var_param_names + self.eval_metrics

    def run(self, resume=False):
        """Run the experiment ´n_repeat´ times for each parameter combination.
        Kwargs:
            resume (bool): Whether to resume a previous run (if available).
        """
        experiment_preperations(self.working_directory)
        checklist = self.init_or_resume(resume)

        # Iterate over the grid
        grid = ParameterGrid(self.variable_param_options)
        pipeline_args = dict(self.fixed_params, working_dir=self.working_directory)
        for var_params in grid:
            run_id = format_params(var_params)
            print('\n' + run_id)
            # logging.info('\tRun experiment with settings: %s' % run_id)
            if run_id in checklist:
                continue

            pipeline_args.update(var_params)
            run_results = self.pipeline(**pipeline_args)
            # outputs = {}
            # for operator in self.pipeline:
            #     inputs = dict(params)
            #     inputs.update(outputs)
            #     outputs = operator(**inputs)

            self.write_run_results(var_params, run_results)

    def init_or_resume(self, resume):
        results_path_2 = os.path.join(self.working_directory, RESULTS_FILE_NAME)
        checklist_path = os.path.join(self.working_directory, CHECKLIST_FILE_NAME)

        if resume and os.path.exists(checklist_path):
            with open(checklist_path, 'r') as checklist_file:
                checklist = checklist_file.read().splitlines()
        else:
            # Reset and init checklist file and checklist
            open(checklist_path, 'w').close()
            checklist = []
            # Init results_file (csv) with header
            with open(results_path_2, 'w') as results_file:
                results_file.write(','.join(self.columns) + '\n')

        return checklist

    def write_run_results(self, var_params, run_results):
        results_path = os.path.join(self.working_directory, RESULTS_FILE_NAME)
        checklist_path = os.path.join(self.working_directory, CHECKLIST_FILE_NAME)

        with open(results_path, 'a') as results_file:
            row = dict(var_params)
            row.update(run_results)
            row_data = [repr(row[k]) for k in self.columns]
            results_file.write(','.join(row_data) + '\n')

        with open(checklist_path, 'a') as checklist_file:
            run_id = format_params(var_params)
            checklist_file.write(run_id + '\n')
