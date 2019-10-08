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


def plot_tree_stats_by_empirical_drift(simulation, x_name='cone_angle', ax=None):
    if ax is None:
        ax = plt.gca()

    working_dir = os.path.join('experiments', simulation, 'tree_statistics')
    if  x_name != 'cone_angle':
        working_dir += '_fossils=500'
        print(working_dir)
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)

    # results = results[results[x_name] < 0.26*np.pi]
    # results = results[results[x_name] > 0.24*np.pi]
    # results = results[results[x_name] == 0.25*np.pi]

    # results.plot.scatter('log_div_rate_0', 'diffusion_rate_0', ax=ax, c='k', s=5)
    # ax.scatter(results['log_div_rate_0_small'], results['diffusion_rate_0_small'], s=5)  # , c='k'
    # ax.scatter(results['log_div_rate_0_big'], results['diffusion_rate_0_big'], s=5)  # , c='k'
    # ax.scatter(results['log_div_rate_1_small'], results['diffusion_rate_1_small'], s=5)  # , c='k'
    # ax.scatter(results['log_div_rate_1_big'], results['diffusion_rate_1_big'], s=5)  # , c='k'
    # ax.scatter(log_div_rates, diffusion_rates, s=20)

    # results = results.groupby(x_name).mean()
    x_unique = results[x_name].unique()
    observed_drift = []
    correlation = []
    imbalance = []
    deep_imbalance = []
    for x in x_unique:
        stats = results[results[x_name] == x]
        log_div_rates = np.concatenate([stats['log_div_rate_0_small'],
                                        stats['log_div_rate_0_big'],
                                        stats['log_div_rate_1_small'],
                                        stats['log_div_rate_1_big']])
        diffusion_rates = np.concatenate([stats['diffusion_rate_0_small'],
                                          stats['diffusion_rate_0_big'],
                                          stats['diffusion_rate_1_small'],
                                          stats['diffusion_rate_1_big']])

        idx_not_na = np.isfinite(log_div_rates) & np.isfinite(diffusion_rates)
        log_div_rates = log_div_rates[idx_not_na]
        diffusion_rates = diffusion_rates[idx_not_na]
        r2 = np.corrcoef(log_div_rates, diffusion_rates)[0, 1]
        correlation.append(r2)

        mean_stats = stats.mean(axis=0)
        imbalance.append(mean_stats['imbalance'])
        # deep_imbalance.append(mean_stats['deep_imbalance'])

        # observed_drift = np.hypot(mean_stats['observed_drift_x'],
        #                           mean_stats['observed_drift_y'])

    ax.plot(x_unique, correlation, label='Speciation-migration correlation')
    print(correlation)
    ax.plot(x_unique, imbalance, label='Imbalance')
    # ax.plot(x_unique, deep_imbalance, label='Deep imbalanace')

    ax.set_xlabel('Observed drift')


if __name__ == '__main__':
    CE = 'constrained_expansion'
    RW = 'random_walk'
    SIMULATION = CE

    HPD_VALUES = [80, 95]

    if SIMULATION == RW:
        x_name = 'total_drift'
    elif SIMULATION == CE:
        x_name = 'cone_angle'
    else:
        raise ValueError('Unknown simulation type: %s' % SIMULATION)

    ax = plt.subplot()
    plot_tree_stats_by_empirical_drift(SIMULATION, x_name=x_name, ax=ax)

    plt.legend()
    plt.ylim(0, None)
    # if SIMULATION == RW:
    #     plt.xlim(0, 6000)
    # else:
    #     # plt.xlim(0, 2*np.pi)
    #     # plt.xlim(0, 7250)
    plt.xlim(0, None)
    plt.tight_layout()
    plt.show()
