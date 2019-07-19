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
    results['total_bias'] = np.hypot(results['mean_offset_x'],
                                     results['mean_offset_y'])
    results['empirical_drift'] = np.hypot(results['observed_drift_x'],
                                          results['observed_drift_y'])
    for metric in ['rmse', 'total_bias', 'empirical_drift']:  # 'total_bias'
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

    x = results['observed_drift']
    for metric in ['rmse', 'bias']:
        ax.scatter(x, results[metric], label=metric, s=6.)

    results = results.groupby(x_name).mean()
    results['total_bias'] = np.hypot(results['mean_offset_x'],
                                     results['mean_offset_y'])
    results['empirical_drift'] = np.hypot(results['observed_drift_x'],
                                          results['observed_drift_y'])
    x = results['empirical_drift']
    for metric in ['rmse', 'total_bias']:
        ax.plot(x, results[metric], label=metric)

    ax.set_xlabel('empirical_drift')


def set_row_labels(axes, labels):
    for ax, lbl in zip(axes[:, 0], labels):
        ax.set_ylabel(lbl + '        ', rotation=0, size='large')


def set_column_labels(axes, labels):
    for ax, lbl in zip(axes[0], labels):
        ax.set_title(lbl)


if __name__ == '__main__':
    CE = 'constrained_expansion'
    RW = 'random_walk'
    SIMULATION = CE

    HPD_VALUES = [80, 95]
    # MOVEMENT_MODEL = 'brownian'
    MOVEMENT_MODEL = 'rrw'

    if SIMULATION == RW:
        x_name = 'total_drift'
    elif SIMULATION == CE:
        x_name = 'cone_angle'
    else:
        raise ValueError('Unknown simulation type: %s' % SIMULATION)

    # MOVEMENT_MODELS = ['rrw', 'cdrw']
    # FOSSIL_AGES = [0., 200., np.inf]
    # fig, axes = plt.subplots(len(MOVEMENT_MODELS), len(FOSSIL_AGES),
    #                          sharex=True, sharey=True)
    #
    # for i, mm in enumerate(MOVEMENT_MODELS):
    #     for j, foss_age in enumerate(FOSSIL_AGES):
    #         # plot_error_stats(SIMULATION, mm, fossil_age=foss_age, x_name=x_name,
    #         #                  ax=axes[i,j])
    #         plot_hpd_coverages(HPD_VALUES, SIMULATION, mm, fossil_age=foss_age,
    #                            x_name=x_name, ax=axes[i,j])
    #
    # set_row_labels(axes, MOVEMENT_MODELS)
    # set_column_labels(axes, ['Max. fossil age: %s' % age for age in FOSSIL_AGES])


    # plot_error_stats(SIMULATION, MOVEMENT_MODEL, x_name=x_name, fossil_age=0)
    plot_error_stats_by_empirical_drift(SIMULATION, MOVEMENT_MODEL, x_name=x_name)

    plt.legend()
    plt.ylim(0, None)
    if SIMULATION == RW:
        plt.xlim(0, 6000)
    else:
        # plt.xlim(0, 2*np.pi)
        plt.xlim(0, 7250)
    plt.tight_layout()
    plt.show()
