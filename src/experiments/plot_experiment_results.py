#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELPAD_X = 10
LABELPAD_Y = 15
LABELS = {
    'total_drift': 'Total trend',
    'cone_angle': 'Cone angle',
    'observed_drift': 'Observed trend',
    'rmse': 'RMSE',
    'bias': 'Bias',
}

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

    ax.set_xlabel(LABELS[x_name], labelpad=LABELPAD_X)


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

    ax.set_xlabel(LABELS[x_name], labelpad=LABELPAD_X)

def plot_error_stats_by_empirical_drift(simulation, movement_model, fossil_age=None,
                     x_name='cone_angle', ax=None):
    if ax is None:
        ax = plt.gca()

    working_dir = os.path.join('experiments', simulation, movement_model)
    if fossil_age is not None:
        working_dir += '_fossils=%s' % fossil_age
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)
    results = results

    x = results['observed_drift_norm']
    for metric in ['rmse', 'bias_norm']:
        ax.scatter(x, results[metric].values, s=1., alpha=0.1)
        # plt.show()

    results = results.groupby(x_name).mean()
    results['bias'] = np.hypot(results['bias_x'],
                                     results['bias_y'])
    results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                          results['observed_drift_y'])
    x = results['observed_drift']
    for metric in ['rmse', 'bias']:
        ax.plot(x, results[metric], label=LABELS[metric])

    ax.set_xlabel(LABELS['observed_drift'], labelpad=LABELPAD_X)


def set_row_labels(axes, labels):
    for ax, lbl in zip(axes[:, 0], labels):
        ax.set_ylabel(str.upper(lbl), rotation=90, size='large', labelpad=LABELPAD_Y)


def set_column_labels(axes, labels):
    for ax, lbl in zip(axes[0], labels):
        ax.set_title(lbl)


if __name__ == '__main__':
    import matplotlib
    # matplotlib.use('ps')
    plt.rcParams.update({'font.size': 16})
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rc('text.latex', preamble=r'\usepackage{color}')
    # matplotlib.rc('text.latex', preamble=r'\usepackage{unicode-math}')
    # matplotlib.rc('text.latex', preamble=r'\setmathfont{TeX Gyre DejaVu Math}[version=dejavu]')

    CE = 'constrained_expansion_20190829'
    RW = 'random_walk_20190829'
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

    MOVEMENT_MODELS = ['rrw', 'cdrw'] #, 'rdrw']
    if SIMULATION == RW:
        FOSSIL_AGES = [0., 500., np.inf]
    else:
        FOSSIL_AGES = [None]

    fig, axes = plt.subplots(len(MOVEMENT_MODELS), len(FOSSIL_AGES),
                             sharex=True, sharey=True)

    if len(FOSSIL_AGES) == 1:
        axes = axes[:, np.newaxis]
    else:
        col_labels = []
        for age in FOSSIL_AGES:
            if age == 0:
                col_labels.append('No ancient samples')
            elif np.isfinite(age):
                # col_labels.append('Sample age ⩽ %i' % age)
                col_labels.append('Sample age$\leq %i$' % age)
            else:
                col_labels.append('All ancient samples')
        set_column_labels(axes, col_labels)


    for i, mm in enumerate(MOVEMENT_MODELS):
        for j, foss_age in enumerate(FOSSIL_AGES):
            # plot_error_stats(SIMULATION, mm, fossil_age=foss_age, x_name=x_name,
            #                  ax=axes[i,j])
            # plot_hpd_coverages(HPD_VALUES, SIMULATION, mm, fossil_age=foss_age,
            #                    x_name=x_name, ax=axes[i,j])
            plot_error_stats_by_empirical_drift(SIMULATION, mm, fossil_age=foss_age, x_name=x_name, ax=axes[i,j])

    set_row_labels(axes, MOVEMENT_MODELS)


    # plot_error_stats(SIMULATION, MOVEMENT_MODEL, x_name=x_name, fossil_age=None)
    # plot_error_stats_by_empirical_drift(SIMULATION, MOVEMENT_MODEL, x_name=x_name)


    plt.ylim(0, 9500)
    plt.gca().set_yticks(np.linspace(0, 8000, 5))

    if SIMULATION == RW:
        axes[0, 2].legend(loc=1)
        plt.xlim(0, 5800)
        plt.gca().set_xticks(np.linspace(0, 4000, 3))
        plt.subplots_adjust(left=0.2, right=0.87, top=0.953, bottom=0.095, wspace=0, hspace=0)
    else:
        axes[0, 0].legend(loc=1)
        plt.xlim(0, 7000)
        plt.gca().set_xticks(np.linspace(0, 6000, 4))
        plt.subplots_adjust(left=0.25, right=0.82, top=0.953, bottom=0.095, wspace=0, hspace=0)

    plt.show()
