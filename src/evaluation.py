#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

from src.beast_interface import run_treeannotator
from src.util import dist


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