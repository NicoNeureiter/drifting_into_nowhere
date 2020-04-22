#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joypy
import seaborn as sns


CE = 'constrained_expansion'
RW = 'random_walk'
BANTU = 'bantu'
SCENARIOS = [RW, CE, BANTU]
# COLORS = ['#991f88', '#cc8800', 'teal']
COLORS = ['#a02590', '#cc8800', 'teal']
LABELS = {
    'random_walk': 'Migration',
    'constrained_expansion': 'Expansion',
    'bantu': 'Bantu',
}
# hist_range = [(-1, 1), (0, 1), (0.16, 0.51)]
# hist_range = [(-1, 1), (0.16, 0.51), (0, 1)]
xrange = [(0.0, 0.68), (-1, 1.), (0.2, 1)]
# yrange = [(0., 17), (0., 7), (0., 8)]
xwidth = [maxi - mini for mini, maxi in xrange]
yrange = [(0., 8./w) for w in xwidth]
# print(yrange)
# exit()

xticks = [
    [0., 0.2, 0.4, 0.6],
    [-1, -.5, 0, .5, 1],
    [.2, .4, .6, .8, 1]
]

def load_experiment_results(scenario):
    working_dir = os.path.join('experiments', scenario, 'tree_statistics')
    results_csv_path = os.path.join(working_dir, 'results.csv')
    return pd.read_csv(results_csv_path)


def plot_hist(x, lbl, ax, i):
    # ax.hist(x, label=lbl, range=(0, 1), bins=30, density=True)
    ax.hist(x, label=[RW, CE, BANTU], range=xrange[i], bins=25, density=True,
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


def plot_joyplot(x, axes):
    new_fig, new_axes = joypy.joyplot(x, ax=axes)
    print([ax for ax in new_axes])
    for ax in axes:
        ax.set_xticks([])

def plot_stat(x, lbl=None, ax=None, i=0):
    if ax is None:
        ax = plt.gca()

    plot_joyplot(x, ax)


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
    # plt.rcParams.update({'font.size': 18})
    # matplotlib.rc('xtick', labelsize=12)
    # matplotlib.rc('ytick', labelsize=12)
    # fig, axes = plt.subplots(3, 3)
    # ax = plt.gca()
    # print(axes)

    dm_corr = []
    imbalance = []
    clade_overlap = []
    df_dict = dict(scenario=[], metric=[], value=[])
    METRICS = ['Clade overlap', 'Diversity-space dependence', 'Tree imbalance']
    for scenario in SCENARIOS:
        # plot_dm_corr(SIMULATION, axes[0])
        # plot_imbalance(SIMULATION, axes[1])
        # plot_clade_overlap(SIMULATION, axes[2])

        xs = get_clade_overlap(scenario)
        clade_overlap.append(xs)
        df_dict['scenario'] += [scenario]*len(xs)
        df_dict['metric'] += [METRICS[0]] * len(xs)
        df_dict['value'] += list(xs)

        xs = get_dm_corr(scenario)
        dm_corr.append(xs)
        df_dict['scenario'] += [scenario]*len(xs)
        df_dict['metric'] += [METRICS[1]] * len(xs)
        df_dict['value'] += list(xs)

        xs = get_imbalance(scenario)
        imbalance.append(xs)
        df_dict['scenario'] += [scenario]*len(xs)
        df_dict['metric'] += [METRICS[2]] * len(xs)
        df_dict['value'] += list(xs)

    df = pd.DataFrame(df_dict)
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

    # plot_stat(np.array(clade_overlap), None, ax=axes[0], i=0)
    # plot_stat(np.array(dm_corr), None, ax=axes[1], i=1)
    # plot_stat(np.array(imbalance), None, ax=axes[2], i=2)

    palette = sns.cubehelix_palette(5, rot=-.25, light=.5, hue=0.2)
    grid = sns.FacetGrid(df, row='scenario', col='metric', hue='scenario', aspect=15, height=.5,
                         palette=palette, sharex=False, sharey=False)

    FACECOLOR = 'darkgray'
    # Draw the densities in a few steps
    # grid.map(sns.kdeplot, 'value', clip_on=False, shade=True, alpha=1, lw=.5, bw='silverman', facecolor=FACECOLOR)
    grid.map(sns.kdeplot, 'value', clip_on=False, shade=True, alpha=1, lw=.5, bw='scott')
    grid.map(sns.kdeplot, 'value', clip_on=False, color='w', lw=1, bw='scott')
    # grid.map(plt.axhline, y=0, lw=1, clip_on=False, color=FACECOLOR)
    grid.map(plt.axhline, y=0, lw=1, clip_on=False)

    # # Define and use a simple function to label the plot in axes coordinates
    # def label(x, color, label):
    #     ax = plt.gca()
    #     ax.text(0, .2, label, fontweight='bold', color=color,
    #             ha='left', va='center', transform=ax.transAxes)
    #
    # grid.map(label, 'value')


    # Set the subplots to overlap
    # grid.fig.subplots_adjust(hspace=0.1, wspace=0.16, left=0.03, bottom=0.07, top=0.93)
    grid.fig.subplots_adjust(hspace=0.1, wspace=0.16, left=0.075, bottom=0.07, top=0.93)
    # grid.axes[2, 0].set_xlim(0, 0.7)

    # Remove axes details that don't play well with overlap
    grid.set(yticks=[])
    # grid.set
    grid.set_xlabels('')
    grid.set_titles('')
    for i in range(3):
        LABELPAD_Y = 0.
        grid.axes[0, i].set_title(METRICS[i], size='large', fontweight='bold')
        # grid.axes[i, 0].set_ylabel(LABELS[SCENARIOS[i]], rotation=90, size='large', labelpad=15)
        grid.axes[i, 0].set_ylabel(LABELS[SCENARIOS[i]], rotation=0, size='large', labelpad=20, fontweight='bold')

        grid.axes[0, i].set_xticks([])
        grid.axes[1, i].set_xticks([])
        grid.axes[2, i].set_xticks(xticks[i])

        for j in range(3):
            grid.axes[j, i].set_xlim(*xrange[i])
            grid.axes[j, i].set_ylim(*yrange[i])

    grid.despine(bottom=True, left=True)

    from matplotlib.patches import Rectangle
    h = 0.75
    b = -0.05
    for i in range(3):
        for j in range(3):
            w = .1 * xwidth[j]
            eps = 0.002 * xwidth[j]
            left_rect = Rectangle((xrange[j][0]-w-eps, b), w, h, fill=True, color='w', clip_on=False, lw=0, zorder=100)
            right_rect = Rectangle((xrange[j][1]+eps, b), w, h, fill=True, color='w', clip_on=False, lw=0, zorder=100)
            grid.axes[i, j].add_patch(left_rect)
            grid.axes[i, j].add_patch(right_rect)

    plt.show()



    # plot_stat(clade_overlap, None, ax=axes[:, 0], i=0)
    # plot_stat(dm_corr, None, ax=axes[:, 1], i=1)
    # plot_stat(imbalance, None, ax=axes[:, 2], i=2)

    # axes[2, 0].set_xlabel('Clade overlap')
    # axes[2, 1].set_xlabel('Diversity-space dependence')
    # axes[2, 2].set_xlabel('Tree imbalance')
    # for i, ax in enumerate(axes):
    #     ax.set_ylim(hist_range[i])
    #
    # axes[0].set_yticks(np.linspace(0.2, 0.5, 4))
    # axes[0, 0].set_yticks(np.linspace(0., 0.6, 4))
    # axes[1, 0].set_yticks(np.linspace(-1, 1, 5))
    #
    # labels = ['Directed random walks', 'Constrained expansions', 'Bantu languages']
    # legend_elements = [Patch(facecolor=COLORS[i], label=labels[i]) for i in range(3)]
    # axes[2].legend(handles=legend_elements, loc=4, prop={'size': 16})
    #
    # plt.tight_layout(pad=0.1)
    # plt.subplots_adjust(left=0.028, right=0.999, top=0.988, bottom=0.046, wspace=0.18, hspace=0.0)
    plt.show()
