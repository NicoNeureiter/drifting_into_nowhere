#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joypy

CE = 'constrained_expansion'
RW = 'random_walk'
BANTU = 'bantu'
SCENARIOS = [RW, CE, BANTU]
# COLORS = ['#991f88', '#cc8800', 'teal']
COLORS = ['#a02590', '#cc8800', 'teal']

# hist_range = [(-1, 1), (0, 1), (0.16, 0.51)]
# hist_range = [(-1, 1), (0.16, 0.51), (0, 1)]
hist_range = [(0.0, 0.68), (-1, 1), (0, 1)]


def load_experiment_results(scenario):
    working_dir = os.path.join('experiments', scenario, 'tree_statistics')
    results_csv_path = os.path.join(working_dir, 'results.csv')
    return pd.read_csv(results_csv_path)


def plot_hist(x, lbl, ax, i):
    # ax.hist(x, label=lbl, range=(0, 1), bins=30, density=True)
    ax.hist(x, label=[RW, CE, BANTU], range=hist_range[i], bins=25, density=True,
            fill=False, histtype='step', lw=2)


def plot_violin(x, lbl, ax, i):
    # for i, y in enumerate(x):
    x = [y[~np.isnan(y)] for y in x]
    parts = ax.violinplot(x, positions=[0,1,2],
                          # showmeans=True,
                          showmedians=True,
                          showextrema=False)

    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS[j])
        pc.set_edgecolor('k')
        pc.set_alpha(1)

    # parts['cmeans'].set_edgecolor('k')
    parts['cmedians'].set_edgecolor('k')

    ax.set_xticks([])
    # ax.set_xticklabels(SCENARIOS)


def plot_joyplot(dm_corr, axes):
    # joypy.joyplot(x, ax=ax)
    pass

def plot_stat(x, lbl=None, ax=None, i=0):
    if ax is None:
        ax = plt.gca()

    # plot_hist(x, lbl, ax, i)
    plot_violin(x, lbl, ax, i)


# def plot_dm_corr(scenario, ax=None):
#     results = load_experiment_results(scenario)
#     dm_corr = results['space_div_dependence']
#     plot_stat(dm_corr, lbl=scenario, ax=ax, i=0)
#
#
# def plot_imbalance(scenario, ax=None):
#     results = load_experiment_results(scenario)
#     imbalance = results['deep_imbalance']
#     plot_stat(imbalance, lbl=scenario, ax=ax, i=1)
#
#
# def plot_clade_overlap(scenario, ax=None):
#     results = load_experiment_results(scenario)
#     clade_overlap = results['clade_overlap']
#     plot_stat(clade_overlap, lbl=scenario, ax=ax, i=2)


def get_dm_corr(scenario, ax=None):
    results = load_experiment_results(scenario)
    return results['space_div_dependence']


def get_imbalance(scenario, ax=None):
    results = load_experiment_results(scenario)
    return results['deep_imbalance']


def get_clade_overlap(scenario, ax=None):
    results = load_experiment_results(scenario)
    return 1 - 2*results['clade_overlap']


def set_row_labels(axes, labels):
    for ax, lbl in zip(axes[:, 0], labels):
        ax.set_ylabel(lbl + '        ', rotation=0, size='large')


def set_column_labels(axes, labels):
    for ax, lbl in zip(axes[0], labels):
        ax.set_title(lbl)


if __name__ == '__main__':
    import matplotlib
    plt.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    fig, axes = plt.subplots(1,3)
    ax = plt.gca()

    dm_corr = []
    imbalance = []
    clade_overlap = []
    for scenario in SCENARIOS:
        # plot_dm_corr(SIMULATION, axes[0])
        # plot_imbalance(SIMULATION, axes[1])
        # plot_clade_overlap(SIMULATION, axes[2])
        dm_corr.append(get_dm_corr(scenario))
        imbalance.append(get_imbalance(scenario))
        clade_overlap.append(get_clade_overlap(scenario))

    # dct = [{"x": x, "name": sim} for xs, sim in zip(dm_corr, [RW, CE, BANTU]) for x in xs]
    # df = pd.DataFrame(dct)
    # plot_stat(df, None, ax=axes, i=0)

    print('Space-diversification dependence')
    for i, scen in enumerate(SCENARIOS):
        score = np.nanmean(dm_corr[i])
        score_std = np.nanstd(dm_corr[i])
        low, high = np.nanquantile(dm_corr[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (scen.ljust(21), score, score_std, low, high))

    print('Clade overlap')
    for i, scen in enumerate(SCENARIOS):
        score = np.nanmean(clade_overlap[i])
        score_std = np.nanstd(clade_overlap[i])
        low, high = np.nanquantile(clade_overlap[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (scen.ljust(21), score, score_std, low, high))

    print('Tree imbalance')
    for i, scen in enumerate(SCENARIOS):
        score = np.nanmean(imbalance[i])
        score_std = np.nanstd(imbalance[i])
        low, high = np.nanquantile(imbalance[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (scen.ljust(21), score, score_std, low, high))

    plot_stat(np.array(clade_overlap), None, ax=axes[0], i=0)
    plot_stat(np.array(dm_corr), None, ax=axes[1], i=1)
    plot_stat(np.array(imbalance), None, ax=axes[2], i=2)

    axes[0].set_xlabel('Clade overlap')
    axes[1].set_xlabel('Diversity-space dependence')
    axes[2].set_xlabel('Tree imbalance')
    for i, ax in enumerate(axes):
        ax.set_ylim(hist_range[i])

    # axes[0].set_yticks(np.linspace(0.2, 0.5, 4))
    axes[0].set_yticks(np.linspace(0., 0.6, 4))
    axes[1].set_yticks(np.linspace(-1, 1, 5))


    labels = ['Directed random walks', 'Constrained expansions', 'Bantu languages']
    legend_elements = [Patch(facecolor=COLORS[i], label=labels[i]) for i in range(3)]
    axes[2].legend(handles=legend_elements, loc=4, prop={'size': 16})

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.028, right=0.999, top=0.988, bottom=0.046, wspace=0.18, hspace=0.0)
    plt.show()
