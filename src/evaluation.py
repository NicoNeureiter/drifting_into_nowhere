#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging

import numpy as np

from src.beast_interface import run_treeannotator, load_trees
from src.tree import tree_imbalance, node_imbalance
from src.util import dist

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

    t0 = tree.small_child()
    t1 = tree.big_child()
    t10 = t1.small_child()
    t11 = t1.big_child()
    t110 = t11.small_child()
    t111 = t11.big_child()

    # Global inbalance stats
    stats['imbalance'] = tree_imbalance(tree)
    stats['deep_imbalance'] = tree_imbalance(tree, max_depth=0.5 * tree.height())

    # Raw size stats
    stats['size'] = tree.n_leafs()
    stats['size_0_small'] = t0.n_leafs()
    stats['size_0_big'] = t1.n_leafs()
    stats['size_1_small'] = t10.n_leafs()
    stats['size_1_big'] = t11.n_leafs()
    stats['size_2_small'] = t110.n_leafs()
    stats['size_2_big'] = t111.n_leafs()

    # Single node imbalance stats
    stats['imbalance_0'] = node_imbalance(tree, ret_weight=False)
    stats['imbalance_1'] = node_imbalance(t1, ret_weight=False)
    stats['imbalance_2'] = node_imbalance(t11, ret_weight=False)
    stats['imbalance_3'] = node_imbalance(t111, ret_weight=False)

    # Migration, drift and diversification rates:
    stats['migr_rate_0'] = migration_rate(tree)
    stats['migr_rate_0_small'] = migration_rate(t0)
    stats['migr_rate_0_big'] = migration_rate(t1)
    stats['migr_rate_1_small'] = migration_rate(t10)
    stats['migr_rate_1_big'] = migration_rate(t11)
    stats['migr_rate_2_small'] = migration_rate(t110)
    stats['migr_rate_2_big'] = migration_rate(t111)

    stats['diffusion_rate_0'] = diffusion_rate(tree)
    stats['diffusion_rate_0_small'] = diffusion_rate(t0)
    stats['diffusion_rate_0_big'] = diffusion_rate(t1)
    stats['diffusion_rate_1_small'] = diffusion_rate(t10)
    stats['diffusion_rate_1_big'] = diffusion_rate(t11)
    stats['diffusion_rate_2_small'] = diffusion_rate(t110)
    stats['diffusion_rate_2_big'] = diffusion_rate(t111)

    stats['drift_rate_0'] = drift_rate(tree)
    stats['drift_rate_0_small'] = drift_rate(t0)
    stats['drift_rate_0_big'] = drift_rate(t1)
    stats['drift_rate_1_small'] = drift_rate(t10)
    stats['drift_rate_1_big'] = drift_rate(t11)
    stats['drift_rate_2_small'] = drift_rate(t110)
    stats['drift_rate_2_big'] = drift_rate(t111)

    stats['log_div_rate_0'] = log_diversification_rate(tree)
    stats['log_div_rate_0_small'] = log_diversification_rate(t0)
    stats['log_div_rate_0_big'] = log_diversification_rate(t1)
    stats['log_div_rate_1_small'] = log_diversification_rate(t10)
    stats['log_div_rate_1_big'] = log_diversification_rate(t11)
    stats['log_div_rate_2_small'] = log_diversification_rate(t110)
    stats['log_div_rate_2_big'] = log_diversification_rate(t111)

    # for k, v in stats.items():
    #     print(('%s:'.ljust(20) + '%.3f') % (k, v))
    return stats


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


def mean_offset(tree):
    if tree.is_leaf():
        return np.nan
    mean_loc = np.mean(tree.get_leaf_locations(), axis=0)
    return mean_loc - tree.location


def observed_drift(tree):
    return np.linalg.norm(mean_offset(tree))


def drift_rate(tree):
    return observed_drift(tree) / tree.height()
