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
IE = 'indo-european'
AUSTRON = 'austronesian'
PN = 'pama-nyungan'
ST = 'sino-tibetan'
TG = 'tupi-guarani'
ARAWAK = 'arawak'
SEMITIC = 'semitic'
URALIC = 'uralic'
SCENARIOS = [RW, CE, BANTU, AUSTRON, IE, PN, ST, SEMITIC, URALIC, TG, ARAWAK]
SCREEN_NAMES = {
    RW: 'Directed random walk',
    CE: 'Constrained expansion',
    BANTU: 'Bantu',
    AUSTRON: '\nAustronesian',
    IE: 'IE',
    PN: '\nPama-Nyung.',
    ST: 'Sino-Tibetan',
    SEMITIC: '\nSemitic',
    URALIC: 'Uralic',
    TG: '\nTupi-Guarani',
    ARAWAK: 'Arawak',
}

# COLOR_LFAM = 'teal'
# COLOR_LFAM = '#226685'
# COLOR_LFAM = '#0088b0'
COLOR_LFAM = '#0090cc'
COLOR_LFAM_EDGE = 'k'
# COLORS = ['#991f88', '#cc8800', 'teal']
# COLORS = ['#a02590', '#cc8800', COLOR_LFAM, COLOR_LFAM, COLOR_LFAM, COLOR_LFAM]
COLORS = ['#dd3377', '#cc8800', COLOR_LFAM, COLOR_LFAM, COLOR_LFAM, COLOR_LFAM]


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


def plot_violin(x, lbl, ax, i, sizes=None):
    x = [y[~np.isnan(y)] for y in x]

    parts = ax.violinplot(x, positions=list(range(len(SCENARIOS)-2)),
                          # showmeans=True,
                          showmedians=True,
                          showextrema=False)

    if sizes is None:
        c = np.ones_like(x)
    else:
        c = np.asarray(sizes)
        c = c / 100.
        c = np.clip(c, 0., 1.)
        # c = c / np.max(c)
        # print(c)

    for j, pc in enumerate(parts['bodies']):
        # pc.set_facecolor(COLORS[j])
        pc.set_facecolor(COLOR_LFAM)
        pc.set_edgecolor(COLOR_LFAM_EDGE)
        # pc.set_alpha(1)
        pc.set_alpha(c[j])

    # parts['cmeans'].set_edgecolor('k')
    parts['cmedians'].set_edgecolor(COLOR_LFAM_EDGE)

    ax.set_xticks([])
    # ax.set_xticklabels(SCENARIOS)


def plot_joyplot(dm_corr, axes):
    # joypy.joyplot(x, ax=ax)
    pass

def plot_stat(x, lbl=None, ax=None, i=0, sizes=None):
    if ax is None:
        ax = plt.gca()

    # plot_hist(x, lbl, ax, i)
    plot_violin(x, lbl, ax, i, sizes=sizes)


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

def get_sample_size(scenario):
    results = load_experiment_results(scenario)
    return results['size'].values[0]

def get_dm_corr(scenario, ax=None):
    results = load_experiment_results(scenario)
    return results['space_div_dependence'].values


def get_imbalance(scenario, ax=None):
    results = load_experiment_results(scenario)
    return results['deep_imbalance'].values


def get_clade_overlap(scenario, ax=None):
    results = load_experiment_results(scenario)
    return 1 - 2*results['clade_overlap'].values


def set_row_labels(axes, labels):
    for ax, lbl in zip(axes[:, 0], labels):
        ax.set_ylabel(lbl + '        ', rotation=0, size='large')


def set_column_labels(axes, labels):
    for ax, lbl in zip(axes[0], labels):
        ax.set_title(lbl)


if __name__ == '__main__':
    import matplotlib
    plt.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=12)
    fig, axes = plt.subplots(1,3)
    ax = plt.gca()

    dm_corr = []
    imbalance = []
    clade_overlap = []
    sizes = []
    for scenario in SCENARIOS:
        dm_corr.append(get_dm_corr(scenario))
        imbalance.append(get_imbalance(scenario))
        clade_overlap.append(get_clade_overlap(scenario))
        sizes.append(get_sample_size(scenario))
        print('\t%s:   %i' % (scenario, sizes[-1]))

    # dct = [{"x": x, "name": sim} for xs, sim in zip(dm_corr, [RW, CE, BANTU]) for x in xs]
    # df = pd.DataFrame(dct)
    # plot_stat(df, None, ax=axes, i=0)

    print('Clade overlap')
    for i, scen in enumerate(SCENARIOS):
        ax = axes[0]
        score = np.nanmean(clade_overlap[i])
        score_median = np.nanmedian(clade_overlap[i])
        score_std = np.nanstd(clade_overlap[i])
        low, high = np.nanquantile(clade_overlap[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (scen.ljust(21), score, score_std, low, high))
        if i < 2:
            # ax.axhline(score, c=COLORS[i], ls='dashed', lw=1., zorder=0)
            ax.axhline(np.nanmedian(clade_overlap[i]), c=COLORS[i], ls='dotted', lw=1., zorder=0)
            low, high = np.nanquantile(clade_overlap[i], [0.8, 0.2])
            ax.fill_between([-1000,1000], [low, low], [high, high], facecolor=COLORS[i], alpha=0.1)
            low, high = np.nanquantile(clade_overlap[i], [0.3, 0.7])
            ax.fill_between([-1000,1000], [low, low], [high, high], facecolor=COLORS[i], alpha=0.1)

            ax.text(-.3, score_median, SCREEN_NAMES[scen], color=COLORS[i], size=10, zorder=0, verticalalignment='bottom', alpha=0.6)

    plot_stat(np.array(clade_overlap[2:]), None, ax=axes[0], i=0, sizes=sizes[2:])

    print('Space-diversification dependence')
    for i, scen in enumerate(SCENARIOS):
        ax = axes[1]
        score = np.nanmean(dm_corr[i])
        score_median = np.nanmedian(dm_corr[i])
        score_std = np.nanstd(dm_corr[i])
        low, high = np.nanquantile(dm_corr[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (scen.ljust(21), score, score_std, low, high))
        if i < 2:
            # ax.axhline(score, c=COLORS[i], ls='dashed', lw=1., zorder=0)
            ax.axhline(np.nanmedian(dm_corr[i]), c=COLORS[i], ls='dotted', lw=1., zorder=0)
            low, high = np.nanquantile(dm_corr[i], [0.8, 0.2])
            ax.fill_between([-1000,1000], [low, low], [high, high], facecolor=COLORS[i], alpha=0.1)
            low, high = np.nanquantile(dm_corr[i], [0.3, 0.7])
            ax.fill_between([-1000,1000], [low, low], [high, high], facecolor=COLORS[i], alpha=0.1)

            ax.text(-.3, score_median, SCREEN_NAMES[scen], color=COLORS[i], size=10, zorder=0, verticalalignment='bottom', alpha=0.6)
        # else:
    plot_stat(np.array(dm_corr[2:]), None, ax=axes[1], i=1, sizes=sizes[2:])


    print('Tree imbalance')
    for i, scen in enumerate(SCENARIOS):
        ax = axes[2]
        score = np.nanmean(imbalance[i])
        score_median = np.nanmedian(imbalance[i])
        score_std = np.nanstd(imbalance[i])
        low, high = np.nanquantile(imbalance[i], [0.025, 0.975])
        print('\t%s:   %.2f ± %.2f    [%.2f, %.2f]' % (scen.ljust(21), score, score_std, low, high))
        if i < 2:
            # ax.axhline(score, c=COLORS[i], ls='dashed', lw=1., zorder=0)
            ax.axhline(np.nanmedian(imbalance[i]), c=COLORS[i], ls='dotted', lw=1., zorder=0)
            low, high = np.nanquantile(imbalance[i], [0.8, 0.2])
            ax.fill_between([-1000,1000], [low, low], [high, high], facecolor=COLORS[i], alpha=0.1)
            low, high = np.nanquantile(imbalance[i], [0.3, 0.7])
            ax.fill_between([-1000,1000], [low, low], [high, high], facecolor=COLORS[i], alpha=0.1)

            ax.text(-.3, score_median, SCREEN_NAMES[scen], color=COLORS[i], size=10, zorder=0, verticalalignment='bottom', alpha=0.6)
        # else:
    plot_stat(np.array(imbalance[2:]), None, ax=axes[2], i=2, sizes=sizes[2:])


    axes[0].set_xlabel('Clade overlap')
    axes[1].set_xlabel('Diversity-space dependence')
    axes[2].set_xlabel('Tree imbalance')
    for i, ax in enumerate(axes):
        ax.set_ylim(hist_range[i])
        print(ax.get_xlim())
        ax.set_xlim(-0.45, len(SCENARIOS) - 2 - 0.55)
        print(ax.get_xlim())
        ax.set_xticks(list(range(len(SCENARIOS) - 2)))
        ax.set_xticklabels([SCREEN_NAMES[s] for s in SCENARIOS[2:]])

    # axes[0].set_yticks(np.linspace(0.2, 0.5, 4))
    axes[0].set_yticks(np.linspace(0., 0.6, 4))
    axes[1].set_yticks(np.linspace(-1, 1, 5))

    # labels = ['Directed random walks', 'Constrained expansions', 'Bantu, IE, Austro']
    # legend_elements = [Patch(facecolor=COLORS[i], label=labels[i]) for i in range(len(labels))]
    # axes[2].legend(handles=legend_elements, loc=4, prop={'size': 16})

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.028, right=0.985, top=0.988, bottom=0.13, wspace=0.18, hspace=0.0)
    plt.show()
