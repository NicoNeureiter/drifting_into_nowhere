#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hpd_coverages(hpd_values, simulation, movement_model, fossil_age=None,
                       x_name='total_drift', ax=None):
    if ax is None:
        ax = plt.gca()

    working_dir = os.path.join('experiments', simulation, movement_model)
    if fossil_age is not None:
        working_dir += '_fossils=%s' % fossil_age
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)

    # results = results.groupby('cone_angle').mean()
    results = results.groupby(x_name).mean()
    x = results.index
    for hpd in hpd_values:
        metric = 'hpd_%i'% hpd
        y = results[metric]
        ax.plot(x, y, label=metric)

    ax.set_xlable(x_name)


def plot_error_stats(simulation, movement_model, fossil_age=None,
                     x_name='total_drift', ax=None):
    if ax is None:
        ax = plt.gca()

    working_dir = os.path.join('experiments', simulation, movement_model)
    if fossil_age is not None:
        working_dir += '_fossils=%s' % fossil_age
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)

    results = results.groupby(x_name).mean()
    x = results.index
    results['bias'] = np.hypot(results['bias_x'],
                               results['bias_y'])
    results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                          results['observed_drift_y'])
    for metric in ['bias', 'observed_drift']:  # 'rmse'
        ax.plot(x, results[metric], label=metric)

    ax.set_xlabel(x_name)

def plot_error_stats_by_empirical_drift(simulation, movement_model, fossil_age=None,
                     x_name='cone_angle', ax=None):
    if ax is None:
        ax = plt.gca()

    working_dir = os.path.join('experiments', simulation, movement_model)
    if fossil_age is not None:
        working_dir += '_fossils=%s' % fossil_age
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)
    results = results*100.

    x = results['observed_drift_norm']
    for metric in ['rmse', 'bias_norm']:
        ax.scatter(x, results[metric].values, s=6.)

    results = results.groupby(x_name).mean()
    results['bias'] = np.hypot(results['bias_x'],
                                     results['bias_y'])
    results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                          results['observed_drift_y'])
    x = results['observed_drift']
    for metric in ['rmse', 'bias']:
        ax.plot(x, results[metric], label=metric)

    ax.set_xlabel('observed_drift')


def set_row_labels(axes, labels):
    for ax, lbl in zip(axes[:, 0], labels):
        ax.set_ylabel(lbl + '        ', rotation=0, size='large')


def set_column_labels(axes, labels):
    for ax, lbl in zip(axes[0], labels):
        ax.set_title(lbl)


if __name__ == '__main__':
    CE = 'constrained_expansion'
    RW = 'random_walk'
    SIMULATION = RW

    HPD_VALUES = [80, 95]
    # MOVEMENT_MODEL = 'brownian'
    MOVEMENT_MODEL = 'rrw'

    if SIMULATION == RW:
        x_name = 'total_drift'
    elif SIMULATION == CE:
        x_name = 'cone_angle'
    else:
        raise ValueError('Unknown simulation type: %s' % SIMULATION)

    MOVEMENT_MODELS = ['rrw', 'cdrw', 'rdrw']
    FOSSIL_AGES = [0., 200., np.inf]
    fig, axes = plt.subplots(len(MOVEMENT_MODELS), len(FOSSIL_AGES),
                             sharex=True, sharey=True)
    if len(FOSSIL_AGES) == 1:
        axes = axes[:, np.newaxis]

    for i, mm in enumerate(MOVEMENT_MODELS):
        for j, foss_age in enumerate(FOSSIL_AGES):
            plot_error_stats(SIMULATION, mm, fossil_age=foss_age, x_name=x_name,
                             ax=axes[i,j])
            # plot_hpd_coverages(HPD_VALUES, SIMULATION, mm, fossil_age=foss_age,
            #                    x_name=x_name, ax=axes[i,j])
            # plot_error_stats_by_empirical_drift(SIMULATION, mm, x_name=x_name, ax=axes[i,j])

    set_row_labels(axes, MOVEMENT_MODELS)
    set_column_labels(axes, ['Max. fossil age: %s' % age for age in FOSSIL_AGES])


    # plot_error_stats(SIMULATION, MOVEMENT_MODEL, x_name=x_name, fossil_age=None)
    # plot_error_stats_by_empirical_drift(SIMULATION, MOVEMENT_MODEL, x_name=x_name)

    plt.legend()
    plt.ylim(0, None)
    if SIMULATION == RW:
        plt.xlim(0, 6000)
    else:
        # plt.xlim(0, 2*np.pi)
        # plt.xlim(0, 7250)
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()
