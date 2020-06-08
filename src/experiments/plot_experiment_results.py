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
    'observed_drift': 'Directional trend',
    'rmse': 'RMSE',
    'bias': 'Bias',
    'hpd_80': '80% HPD',
    'hpd_95': '95% HPD',
}

# def plot_hpd_coverages(hpd_values, simulation, movement_model, fossil_age=None,
#                        x_name='total_drift', ax=None):
#     if ax is None:
#         ax = plt.gca()
#
#     working_dir = os.path.join('experiments', simulation, movement_model)
#     if fossil_age is not None:
#         working_dir += '_fossils=%s' % fossil_age
#     results_csv_path = os.path.join(working_dir, 'results.csv')
#     results = pd.read_csv(results_csv_path)
#
#     # results = results.groupby('cone_angle').mean()
#     results = results.groupby(x_name).mean()
#     x = results.index
#     for hpd in hpd_values:
#         metric = 'hpd_%i'% hpd
#         y = results[metric]
#         ax.plot(x, y, label=metric)
#
#     ax.set_xlabel(LABELS[x_name], labelpad=LABELPAD_X)
#
#
# def plot_error_stats(simulation, movement_model, fossil_age=None,
#                      x_name='total_drift', ax=None):
#     if ax is None:
#         ax = plt.gca()
#
#     working_dir = os.path.join('experiments', simulation, movement_model)
#     if fossil_age is not None:
#         working_dir += '_fossils=%s' % fossil_age
#     results_csv_path = os.path.join(working_dir, 'results.csv')
#     results = pd.read_csv(results_csv_path)
#
#     results = results.groupby(x_name).mean()
#     x = results.index
#     results['bias'] = np.hypot(results['bias_x'],
#                                results['bias_y'])
#     results['observed_drift'] = np.hypot(results['observed_drift_x'],
#                                           results['observed_drift_y'])
#     for metric in ['bias', 'observed_drift']:  # 'rmse'
#         ax.plot(x, results[metric], label=metric)
#
#     ax.set_xlabel(LABELS[x_name], labelpad=LABELPAD_X)


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
    if not PLOT_HPD:
        for metric in ['rmse', 'bias_norm']:
            ax.scatter(x, results[metric].values, s=1., alpha=0.1)

    results = results.groupby(x_name).mean()
    results['bias'] = np.hypot(results['bias_x'],
                               results['bias_y'])
    results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                         results['observed_drift_y'])
    x = results['observed_drift']
    if PLOT_HPD:
        for hpd in HPD_VALUES:
            metric = 'hpd_%i'% hpd
            ax.plot(x, results[metric], label=LABELS[metric])
    else:
        for metric in ['rmse', 'bias']:
            ax.plot(x, results[metric], label=LABELS[metric])

    ax.set_xlabel(LABELS['observed_drift'], labelpad=LABELPAD_X)


def plot_error_stats_by_empirical_drift_varying_tree_size(simulation, movement_model, tree_size,
                                                          x_name='cone_angle', ax=None):
    if ax is None:
        ax = plt.gca()

    working_dir = os.path.join('experiments', simulation, movement_model + '_treesize=%i' % tree_size)
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)
    results = results

    x = results['observed_drift_norm']
    if not PLOT_HPD:
        for metric in ['rmse', 'bias_norm']:
            ax.scatter(x, results[metric].values, s=1., alpha=0.1)

    results = results.groupby(x_name).mean()
    results['bias'] = np.hypot(results['bias_x'],
                               results['bias_y'])
    results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                         results['observed_drift_y'])
    x = results['observed_drift']
    if PLOT_HPD:
        for hpd in HPD_VALUES:
            metric = 'hpd_%i'% hpd
            ax.plot(x, results[metric], label=LABELS[metric])
    else:
        for metric in ['rmse', 'bias']:
            ax.plot(x, results[metric], label=LABELS[metric])

    ax.set_xlabel(LABELS['observed_drift'], labelpad=LABELPAD_X)


def set_row_labels(axes, labels):
    for ax, lbl in zip(axes[:, -1], labels):
        x = ax.get_xlim()[1]
        y = 0.42 * ax.get_ylim()[1]
        ax.text(x, y, str.upper(lbl), rotation=-90, size='large')


def set_column_labels(axes, labels):
    for ax, lbl in zip(axes[0], labels):
        ax.set_title(lbl, pad=8)


def format_fossil_age(age):
    if age == 0:
        return 'No historical data'
    elif np.isfinite(age):
        return 'Sample age$\leq %i$' % age
    else:
        return 'All sample ages'


if __name__ == '__main__':
    import matplotlib
    plt.rcParams.update({'font.size': 16})
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)

    # CE = 'constrained_expansion'
    CE = 'constrained_expansion_20190829'
    # RW = 'random_walk'
    RW = 'random_walk_20190829'

    SIMULATION = RW
    # SIMULATION = CE

    HPD_VALUES = [95, 80]
    # MOVEMENT_MODEL = 'brownian'
    MOVEMENT_MODEL = 'rrw'

    PLOT_HPD = 1
    VARY_TREE_SIZE = 0

    if SIMULATION == RW:
        x_name = 'total_drift'
    elif SIMULATION == CE:
        x_name = 'cone_angle'
    else:
        raise ValueError('Unknown simulation type: %s' % SIMULATION)

    MOVEMENT_MODELS = ['rrw', 'cdrw']#, 'rdrw']
    if SIMULATION == RW:
        FOSSIL_AGES = [0., 500., np.inf]
    else:
        FOSSIL_AGES = [None]
    TREE_SIZES = [50, 100, 300]

    n_rows = len(MOVEMENT_MODELS)
    n_cols = len(TREE_SIZES) if VARY_TREE_SIZE else len(FOSSIL_AGES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 + 3*n_cols, 2 + 3*n_rows),
                             sharex=False, sharey=False)

    if len(MOVEMENT_MODELS) == 1:
        axes = np.array([axes])
    if VARY_TREE_SIZE:
        if len(TREE_SIZES) == 1:
            axes = axes[:, np.newaxis]
        else:
            set_column_labels(axes, ['Tree size = %i' % ts for ts in TREE_SIZES])
    else:
        if len(FOSSIL_AGES) == 1:
            axes = axes[:, np.newaxis]
            set_column_labels(axes, map(format_fossil_age, [0.]))
        else:
            set_column_labels(axes, map(format_fossil_age, FOSSIL_AGES))

    for i, mm in enumerate(MOVEMENT_MODELS):
        if VARY_TREE_SIZE:
            for j, tree_size in enumerate(TREE_SIZES):
                plot_error_stats_by_empirical_drift_varying_tree_size(SIMULATION, mm, tree_size=tree_size, x_name=x_name, ax=axes[i, j])
        else:
            for j, foss_age in enumerate(FOSSIL_AGES):
                plot_error_stats_by_empirical_drift(SIMULATION, mm, fossil_age=foss_age, x_name=x_name, ax=axes[i, j])

    # Define legend location
    axes[0, -1].legend(loc=1)

    # Adjust axis limits and ticks
    if PLOT_HPD:
        for ax in axes.flatten():
            # Add reference lines at the HPD thresholds
            ax.axhline(0.8, color='lightgrey', ls='dotted', zorder=0)
            ax.axhline(0.95, color='lightgrey', ls='dotted', zorder=0)
            ax.set_ylim(0, 1.01)
            ax.set_yticks([])

        for ax in axes[:, 0]:
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
            ax.set_yticklabels([0, 20, 40, 60, 80, 95])

    else:
        for ax in axes.flatten():
            ax.set_ylim(0, 8000)
            ax.set_yticks([])

        for ax in axes[:, 0]:
            ax.set_yticks(np.linspace(0, 6000, 4))

    # Fix x-axis
    for ax in axes.flatten():
        ax.set_xlim(0, 6400)
        ax.set_xticks([])
    for ax in axes[-1]:
        ax.set_xticks(np.linspace(0, 6000, 4))
    for ax in axes[:-1].flatten():
        ax.set_xlabel('')

    # Set row labels, if needed
    if len(MOVEMENT_MODELS) > 1:
        set_row_labels(axes, MOVEMENT_MODELS)

    # Adjust the spacing between and around subplots
    if n_rows == 1:
        plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.12, wspace=0.05, hspace=0.05)
    if n_rows == 2:
        plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.072, wspace=0.05, hspace=0.05)
    if n_cols == 1:
        plt.subplots_adjust(left=0.1, right=0.94)

    plt.show()
