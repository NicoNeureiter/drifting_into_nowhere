#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from src.beast_interface import run_treeannotator, load_trees
from src.tree import tree_imbalance, node_imbalance
from src.util import dist, delaunay_join_count

LOGGER = logging.getLogger('experiment')


def eval_hpd_hit(root, p_hpd, burnin, working_dir):
    tree_mcc = run_treeannotator(p_hpd, burnin, working_dir=working_dir)
    return tree_mcc.root_in_hpd(root, p_hpd)


def eval_mean_offset(root, trees):
    root_samples = [t.location for t in trees]
    mean_estimate = np.mean(root_samples, axis=0)
    return mean_estimate - root


def eval_bias(root, trees):
    root_samples = [t.location for t in trees]
    mean_estimate = np.mean(root_samples, axis=0)
    return dist(root, mean_estimate)


def eval_stdev(root, trees):
    root_samples = [t.location for t in trees]
    std = np.std(root_samples, axis=0)
    return np.linalg.norm(std)


def eval_rmse(root, trees):
    errors = [dist(root, t.location)**2. for t in trees]
    return np.mean(errors)**0.5


def tree_statistics(tree):
    stats = {}

    # The number of fossils (non contemporary leafs) in the tree.
    stats['n_fossils'] = tree.n_fossils()
    stats['observed_drift'] = observed_drift(tree)
    stats['observed_drift_x'], stats['observed_drift_y'] = mean_offset(tree)

    # Global inbalance stats
    stats['imbalance'] = tree_imbalance(tree)
    stats['deep_imbalance'] = tree_imbalance(tree, max_depth=0.5 * tree.height())

    # Raw size stats
    stats['size'] = tree.n_leafs()

    DROP_FOSSILS = True
    if DROP_FOSSILS:
        tree.drop_fossils()

    CLADE_HEIGHT = tree.height() / 2.
    clades = tree.get_clades_at_height(CLADE_HEIGHT)
    clades = [t for t in clades if t.n_leafs() > 2]

    log_div_rate = [log_diversification_rate(t, height=CLADE_HEIGHT) for t in clades]
    migr_rate = [diffusion_rate(t) for t in clades]

    if len(migr_rate) >= 2 and len(log_div_rate) >= 2:
        stats['space_div_dependence'] = pearsonr(migr_rate, log_div_rate)[0]
    else:
        stats['space_div_dependence'] = np.nan

    stats['clade_overlap'] = mean_clade_overlap(clades)

    return stats


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def evaluate(working_dir, burnin, hpd_values, true_root):
    results = {}
    for hpd in hpd_values:
        # Summarize tree using tree-annotator
        tree = run_treeannotator(hpd, burnin, working_dir=working_dir)

        # Compute HPD coverage
        hit = tree.root_in_hpd(true_root, hpd)
        results['hpd_%i' % hpd] = hit
        LOGGER.info('\t\tRoot in %i%% HPD: %s' % (hpd, hit))

    # Load posterior trees for other metrics
    trees = load_trees(working_dir + 'nowhere.trees')

    # Compute and log RMSE
    rmse = eval_rmse(true_root, trees)
    results['rmse'] = rmse
    LOGGER.info('\t\tRMSE: %.2f' % rmse)

    # Compute and log mean offset
    offset = eval_mean_offset(true_root, trees)
    results['bias_x'] = offset[0]
    results['bias_y'] = offset[1]
    LOGGER.info('\t\tMean offset: (%.2f, %.2f)' % tuple(offset))

    # Compute and log bias
    bias = eval_bias(true_root, trees)
    results['bias_norm'] = bias
    LOGGER.info('\t\tBias: %.2f' % bias)

    # Compute and log standard deviation
    stdev = eval_stdev(true_root, trees)
    results['stdev'] = stdev
    LOGGER.info('\t\tStdev: %.2f' % stdev)

    return results


def diffusion_rate(tree):
    locs = tree.get_leaf_locations()
    std = np.std(locs, axis=0)
    # print(std)
    return np.linalg.norm(std)


def migration_rate(tree):
    if tree.is_leaf():
        return np.nan
    locs = tree.get_leaf_locations()
    diffs = locs - tree.location
    dists = np.linalg.norm(diffs, axis=-1)
    rates = dists / tree.height()
    return np.mean(rates)


def log_diversification_rate(tree):
    if tree.is_leaf():
        return np.nan
    # return (tree.n_leafs() ** (1 / tree.height()) - 1) * 100.
    return np.log(tree.n_leafs()) / tree.height()


def _diffusion_rate(tree, height=None):
    if height is None:
        height = tree.height()

    locs = tree.get_leaf_locations()
    std = np.std(locs, axis=0, ddof=1)
    return np.linalg.norm(std) / height**0.5


def diffusion_rate(tree):
    geo_dists = tree.get_loc_dist_mat()
    phylo_dists = tree.get_phylo_dist_mat()
    geo_rates = geo_dists / (phylo_dists ** 0.5)
    return np.nanmean(geo_rates)


def log_diversification_rate(tree, height=None):
    if tree.is_leaf():
        return np.nan

    if height is None:
        height = tree.height()

    return np.log(tree.n_leafs()) / height


def log_div_rate(tree):
    intern_lens = [t.length for t in tree.iter_descendants() if not t.is_leaf()]
    return 1 / np.mean(intern_lens)


def mean_clade_overlap(clades):
    other_clades = set(clades)

    scores = []
    for clade in clades:
        other_clades.remove(clade)

        locations_clade = [node.location for node in clade.iter_leafs()]
        locations_other = [node.location for other in other_clades for node in other.iter_leafs()]
        locations = locations_clade + locations_other
        locations = np.array(locations)

        labels = [1] * len(locations_clade) + [0] * len(locations_other)
        join_count = delaunay_join_count(locations, labels)
        scores.append(join_count)

    return np.mean(scores)


def mean_offset(tree):
    if tree.is_leaf():
        return np.nan
    mean_loc = np.mean(tree.get_leaf_locations(), axis=0)
    return mean_loc - tree.location


def observed_drift(tree):
    return np.linalg.norm(mean_offset(tree))


def drift_rate(tree):
    return observed_drift(tree) / tree.height()
