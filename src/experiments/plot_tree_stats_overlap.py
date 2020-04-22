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
# OVERLAPS = [0.0, 0.125, 0.25,
#             0.375,
#             0.5, 0.625, 0.75, 0.875, 1.0]
OVERLAPS = [.0, .2, .4, .6, .8, 1.]
# COLORS = ['#991f88', '#cc8800', 'teal']
# COLORS = ['#a02590', '#cc8800', 'teal']
# COLORS = [plt.get_cmap('Dark2')(i) for i in range(len(OVERLAPS))]
# COLORS = [plt.get_cmap('gist_earth')(i) for i in np.linspace(0.2, .8, len(OVERLAPS))]
# COLORS = [plt.get_cmap('YlOrBr')(i) for i in np.linspace(0.2, .8, len(OVERLAPS))]
# COLORS = [plt.get_cmap('copper')(i) for i in np.linspace(.75, 0.4, len(OVERLAPS))]
COLORS =  [plt.get_cmap('YlOrBr')(0.9 - 0.6*x) for x in OVERLAPS]
EDGECOLORS =  [plt.get_cmap('YlOrBr')(1. - 0.6*x) for x in OVERLAPS]
# COLORS = [plt.get_cmap('copper')(1. - 0.7*x) for x in OVERLAPS]
# COLORS = ['#cc7711', '#dd5533', '#cc4450', '#cc3366']

# hist_range = [(-1, 1), (0, 1), (0.16, 0.51)]
# hist_range = [(-1, 1), (0.16, 0.51), (0, 1)]
hist_range = [(0.0, 0.68), (-1, 1), (0, 1)]



def load_experiment_results(overlap):
    scenario = '%s_overlap_%s' % (CE, overlap)
    working_dir = os.path.join('experiments', scenario, 'tree_statistics')
    results_csv_path = os.path.join(working_dir, 'results.csv')
    results = pd.read_csv(results_csv_path)
    # results = results[np.isclose(results['cone_angle'], 0.628318530717959)]
    return results


# def plot_hist(x, lbl, ax, i):
#     # ax.hist(x, label=lbl, range=(0, 1), bins=30, density=True)
#     ax.hist(x, label=[RW, CE, BANTU], range=hist_range[i], bins=25, density=True,
#             fill=False, histtype='step', lw=2)


def drop_outliers(data, m=3.):
    # return data[abs(data - np.mean(data)) < m * np.std(data)]
    return data


def plot_violin(x, lbl, ax, i):
    # for i, y in enumerate(x):
    x = [drop_outliers(y[~np.isnan(y)]) for y in x]


    parts = ax.violinplot(x, positions=OVERLAPS,
                          # showmeans=True,
                          showmedians=True,
                          showextrema=False,
                          widths=0.07)

    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS[j])
        # pc.set_edgecolor('k')
        pc.set_edgecolor(EDGECOLORS[j])
        pc.set_alpha(0.9)

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


def get_dm_corr(overlap):
    results = load_experiment_results(overlap)
    return results['space_div_dependence']


def get_imbalance(overlap):
    results = load_experiment_results(overlap)
    return results['deep_imbalance']


def get_clade_overlap(overlap):
    results = load_experiment_results(overlap)
    return results['clade_overlap']


def get_size(overlap):
    results = load_experiment_results(overlap)
    return results['size']


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
    size = []
    for overlap in OVERLAPS:
        # plot_dm_corr(SIMULATION, axes[0])
        # plot_imbalance(SIMULATION, axes[1])
        # plot_clade_overlap(SIMULATION, axes[2])
        dm_corr.append(get_dm_corr(overlap))
        imbalance.append(get_imbalance(overlap))
        clade_overlap.append(get_clade_overlap(overlap))
        size.append(get_size(overlap))

    # dct = [{"x": x, "name": sim} for xs, sim in zip(dm_corr, [RW, CE, BANTU]) for x in xs]
    # df = pd.DataFrame(dct)
    # plot_stat(df, None, ax=axes, i=0)

    print('Space-diversification dependence')
    for i, overlap in enumerate(OVERLAPS):
        score = np.nanmean(dm_corr[i])
        score_std = np.nanstd(dm_corr[i])
        low, high = np.nanquantile(dm_corr[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (str(overlap).ljust(10), score, score_std, low, high))

    print('Clade overlap')
    for i, overlap in enumerate(OVERLAPS):
        score = np.nanmean(clade_overlap[i])
        score_std = np.nanstd(clade_overlap[i])
        low, high = np.nanquantile(clade_overlap[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (str(overlap).ljust(10), score, score_std, low, high))

    print('Tree imbalance')
    for i, overlap in enumerate(OVERLAPS):
        score = np.nanmean(imbalance[i])
        score_std = np.nanstd(imbalance[i])
        low, high = np.nanquantile(imbalance[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (str(overlap).ljust(10), score, score_std, low, high))

    print('Tree size')
    for i, overlap in enumerate(OVERLAPS):
        score = np.nanmean(size[i])
        score_std = np.nanstd(size[i])
        low, high = np.nanquantile(size[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (str(overlap).ljust(10), score, score_std, low, high))

    plot_stat(np.array(clade_overlap), None, ax=axes[0], i=0)
    plot_stat(np.array(dm_corr), None, ax=axes[1], i=1)
    plot_stat(np.array(imbalance), None, ax=axes[2], i=2)

    axes[0].set_xlabel('Clade overlap')
    axes[1].set_xlabel('Diversity-space dependence')
    axes[2].set_xlabel('Tree imbalance')
    for i, ax in enumerate(axes):
        # ax.set_xticks(range(len(OVERLAPS)))
        ax.set_xticks(OVERLAPS)
        # ax.set_xticklabels(OVERLAPS)
        ax.set_ylim(hist_range[i])

    # axes[0].set_yticks(np.linspace(0.2, 0.5, 4))
    axes[0].set_yticks(np.linspace(0., 0.6, 4))
    axes[1].set_yticks(np.linspace(-1, 1, 5))

    # labels = ['Directed random walks', 'Constrained expansions', 'Bantu languages']
    # legend_elements = [Patch(facecolor=COLORS[i], label=labels[i]) for i in range(3)]
    # axes[2].legend(handles=legend_elements, loc=4, prop={'size': 16})

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.028, right=0.999, top=0.988, bottom=0.1, wspace=0.18, hspace=0.0)
    plt.show()
