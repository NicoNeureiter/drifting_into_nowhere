#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT_BIAS = 0
HPD = 95
LABELPAD_X = 10
LABELPAD_Y = 15
LABELS = {
    'total_drift': 'Total trend',
    'cone_angle': 'Cone angle',
    'observed_drift': 'Observed trend',
    'rmse': 'RMSE',
    'bias': 'Bias',
}


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

    results = results.groupby(x_name).mean()
    results['bias'] = np.hypot(results['bias_x'],
                               results['bias_y'])
    results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                         results['observed_drift_y'])
    x = results['observed_drift']
    for metric in ['rmse', 'bias']:
        ax.plot(x, results[metric], label=LABELS[metric])

    ax.set_xlabel(LABELS['observed_drift'], labelpad=LABELPAD_X)

def plot_error_stats_by_empirical_drift_varying_overlap(simulation, movement_model, fossil_age=None,
                                                        x_name='cone_angle', ax=None):
    if ax is None:
        ax = plt.gca()

    ls = {
        0.0: 'solid',
        0.333: 'dashed',
        0.5: 'dashed',
        0.666: 'dashdot',
        1.0: 'dotted'
    }
    lw = {
        0.0: 2.,
        0.333: 2.,
        0.5: 2.,
        0.666: 1.8,
        1.0: 1.9,
    }
    color = next(ax._get_lines.prop_cycler)['color']
    if PLOT_BIAS:
        plt.plot([0, 3000], [0, 3000], c='lightgray', lw=0.5)

    for overlap in [0.0, 0.333, 0.666, 1.0]:
        working_dir = os.path.join('experiments',
                                   simulation + '_overlap_%s' % overlap,
                                   movement_model)
        results_csv_path = os.path.join(working_dir, 'results.csv')
        results = pd.read_csv(results_csv_path)

        # from matplotlib.patches import Circle
        # prop = 'bias_'
        # ax.axhline(0, c='k')
        # ax.axvline(0, c='k')
        # for angle in np.unique(results['cone_angle']):
        #     color = next(ax._get_lines.prop_cycler)['color']
        #     xy = results[results['cone_angle'] == angle]
        #     ax.scatter(xy[prop+'x'].values, xy[prop+'y'].values, color=color)
        #     r = np.hypot(xy[prop + 'x'].values.mean(), xy[prop + 'y'].values.mean())
        #     circle = Circle((0, 0), r, color=color, fill=False)
        #     ax.add_artist(circle)
        #
        # return

        # x = results['observed_drift_norm']
        # # for metric in ['rmse', 'bias_norm']:
        # for metric in ['bias_norm']:
        #     ax.scatter(x, results[metric].values, s=1., alpha=0.5)

        results = results.groupby(x_name).mean()
        results['bias'] = np.hypot(results['bias_x'],
                                   results['bias_y'])
        results['observed_drift'] = np.hypot(results['observed_drift_x'],
                                             results['observed_drift_y'])
        x = results['observed_drift']

        if PLOT_BIAS:
            # for metric in ['rmse', 'bias']:
            for metric in ['bias']:
                # ax.plot(x, results[metric], label='Overlap = %s' % (np.round(overlap, 2)),
                ax.plot(x, results[metric], label=r'$\rho_{overlap}$ = %.2f' % overlap,
                        color=color, ls=ls[overlap], lw=lw[overlap])
        else:
            # for hpd in [HPD]:
            for hpd in [HPD]:
                metric = 'hpd_%i' % hpd
                y = results[metric]
                # ax.plot(x, y, label=LABELS[metric])
                ax.plot(x, y, label=r'$\rho_{overlap}$ = %.2f' % overlap,
                        color=color, ls=ls[overlap], lw=lw[overlap])

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
    fig = plt.figure(figsize=(9., 6.6))

    CE = 'constrained_expansion'
    # CE = 'constrained_expansion_20190829'
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

    MOVEMENT_MODEL = 'rrw'

    ax = plt.gca()

    plot_error_stats_by_empirical_drift_varying_overlap(SIMULATION, MOVEMENT_MODEL,
                                                        x_name=x_name, ax=ax)

    # plt.ylim(0, 9500)
    # plt.gca().set_yticks(np.linspace(0, 8000, 5))
    # plt.ylim(0, 2500)
    # plt.ylim(0, 1)
    plt.ylim(0, None)

    # plt.xlim(0, 7000)
    # ax.set_xticks(np.linspace(0, 6000, 4))
    plt.xlim(0, None)
    plt.ylabel('Bias')
    ax.legend(loc=1)
    plt.subplots_adjust(left=0.1, right=0.98, top=0.999, bottom=0.09)
    plt.show()
