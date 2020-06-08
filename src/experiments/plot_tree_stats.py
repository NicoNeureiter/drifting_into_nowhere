#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


CE = 'constrained_expansion'
RW = 'random_walk'
BANTU = 'bantu'
SCENARIOS = [RW, CE, BANTU]
COLORS = ['#a02590', '#cc8800', 'teal']
LABELS = {
    'random_walk': 'Migration',
    'constrained_expansion': 'Expansion',
    'bantu': 'Bantu',
}
X_RANGE = [(0.0, 0.68), (-1, 1.), (0.2, 1)]
X_WIDTH = [maxi - mini for mini, maxi in X_RANGE]
Y_RANGE = [(0., 8. / w) for w in X_WIDTH]  # y scaling such that the area stays constant
X_TICKS = [
    [0., 0.2, 0.4, 0.6],
    [-1, -.5, 0, .5, 1],
    [.2, .4, .6, .8, 1]
]


def load_experiment_results(scenario):
    working_dir = os.path.join('experiments', scenario, 'tree_statistics')
    results_csv_path = os.path.join(working_dir, 'results.csv')
    return pd.read_csv(results_csv_path)


def get_ds_dependence(scenario):
    """Load the diversity-space dependence scores from the results csv."""
    results = load_experiment_results(scenario)
    return results['space_div_dependence']


def get_imbalance(scenario):
    """Load the tree-imbalance scores from the results csv."""
    results = load_experiment_results(scenario)
    return results['deep_imbalance']


def get_clade_overlap(scenario):
    """Load the clade-overlap scores from the results csv."""
    results = load_experiment_results(scenario)
    return results['clade_overlap']
    #return 1 - 2*results['clade_overlap']


if __name__ == '__main__':
    ds_dependence = []
    imbalance = []
    clade_overlap = []
    df_dict = dict(scenario=[], metric=[], value=[])
    METRICS = ['Clade overlap', 'Diversity-space dependence', 'Tree imbalance']
    for scenario in SCENARIOS:
        xs = get_clade_overlap(scenario)
        clade_overlap.append(xs)
        df_dict['scenario'] += [scenario]*len(xs)
        df_dict['metric'] += [METRICS[0]] * len(xs)
        df_dict['value'] += list(xs)

        xs = get_ds_dependence(scenario)
        ds_dependence.append(xs)
        df_dict['scenario'] += [scenario]*len(xs)
        df_dict['metric'] += [METRICS[1]] * len(xs)
        df_dict['value'] += list(xs)

        xs = get_imbalance(scenario)
        imbalance.append(xs)
        df_dict['scenario'] += [scenario]*len(xs)
        df_dict['metric'] += [METRICS[2]] * len(xs)
        df_dict['value'] += list(xs)

    df = pd.DataFrame(df_dict)

    print('Space-diversification dependence')
    for i, scen in enumerate(SCENARIOS):
        score = np.nanmean(ds_dependence[i])
        score_std = np.nanstd(ds_dependence[i])
        low, high = np.nanquantile(ds_dependence[i], [0.025, 0.975])
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

    palette = sns.cubehelix_palette(5, rot=-.25, light=.5, hue=0.2)
    grid = sns.FacetGrid(df, row='scenario', col='metric', hue='scenario', aspect=15, height=.5,
                         palette=palette, sharex=False, sharey=False)

    FACECOLOR = 'darkgray'

    # Draw the densities in a few steps
    grid.map(sns.kdeplot, 'value', clip_on=False, shade=True, alpha=1, lw=.5, bw='scott')
    grid.map(sns.kdeplot, 'value', clip_on=False, color='w', lw=1, bw='scott')
    grid.map(plt.axhline, y=0, lw=1, clip_on=False)

    # Set the subplots to overlap
    grid.fig.subplots_adjust(hspace=0.1, wspace=0.16, left=0.075, bottom=0.07, top=0.93)

    # Remove axes details that don't play well with overlap
    grid.set(yticks=[])
    grid.set_xlabels('')
    grid.set_titles('')
    for i in range(3):
        LABELPAD_Y = 0.
        grid.axes[0, i].set_title(METRICS[i], size='large', fontweight='bold')
        grid.axes[i, 0].set_ylabel(LABELS[SCENARIOS[i]], rotation=0, size='large', labelpad=20, fontweight='bold')

        grid.axes[0, i].set_xticks([])
        grid.axes[1, i].set_xticks([])
        grid.axes[2, i].set_xticks(X_TICKS[i])

        for j in range(3):
            grid.axes[j, i].set_xlim(*X_RANGE[i])
            grid.axes[j, i].set_ylim(*Y_RANGE[i])

    grid.despine(bottom=True, left=True)

    # Manually clip the graphs outside the valid x-ranges.
    h = 0.75
    b = -0.05
    for i in range(3):
        for j in range(3):
            w = .1 * X_WIDTH[j]
            eps = 0.002 * X_WIDTH[j]
            left_rect = Rectangle((X_RANGE[j][0] - w - eps, b), w, h, fill=True, color='w', clip_on=False, lw=0, zorder=100)
            right_rect = Rectangle((X_RANGE[j][1] + eps, b), w, h, fill=True, color='w', clip_on=False, lw=0, zorder=100)
            grid.axes[i, j].add_patch(left_rect)
            grid.axes[i, j].add_patch(right_rect)

    plt.show()
