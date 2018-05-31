#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import attr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
from numpy.random import multivariate_normal as gaussian

from src.util import bounding_box


class GeoState(object):

    def __init__(self, location, step_mean, step_cov, location_history=None):
        self.location = np.asarray(location)
        self.step_mean = np.asarray(step_mean)
        self.step_cov = np.asarray(step_cov)

        if location_history is None:
            self.location_history = [self.location]
        else:
            self.location_history = location_history

    def step(self):
        self.location = self.location + gaussian(self.step_mean, self.step_cov)
        self.location_history.append(self.location)

    def __copy__(self):
        return GeoState(self.location.copy(), self.step_mean, self.step_cov,
                        self.location_history.copy())

    def copy(self):
        return self.__copy__()


class FeatureState(object):

    """
    Attributes:
        features (np.array): Binary feature vector
        rate (float): Probability of
    """

    def __init__(self, features, rate, feature_history=None):
        self.features = np.asarray(features)
        self.rate = rate

        self.n_features = len(self.features)

        if feature_history is None:
            self.feature_history = [self.features]
        else:
            self.feature_history = feature_history

    def step(self):
        step = np.random.binomial(1, p=self.rate, size=self.n_features)
        self.features = np.logical_xor(self.features, step)
        self.feature_history.append(self.features)

    def __copy__(self):
        return FeatureState(self.features.copy(), self.rate, self.feature_history.copy())

    def copy(self):
        return self.__copy__()

class State(object):

    def __init__(self, features, location, rate, geo_step_mean, geo_step_cov,
                 location_history=None, feature_history=None,
                 parent=None, children=None):
        self.geoState = GeoState(location, geo_step_mean, geo_step_cov, location_history)
        self.featureState = FeatureState(features, rate, feature_history)

        self.parent = parent
        self.children = children or []

    @property
    def location_history(self):
        return self.geoState.location_history

    @property
    def feature_history(self):
        return self.featureState.feature_history

    def step(self):
        self.featureState.step()
        self.geoState.step()

    def create_child(self):
        fs = self.featureState.copy()
        gs = self.geoState.copy()

        child = State(fs.features, gs.location, fs.rate, gs.step_mean, gs.step_cov,
                      location_history=gs.location_history, feature_history=fs.feature_history,
                      parent=self)

        self.children.append(child)

        return child


class Simulation(object):

    def __init__(self, n_features, rate, step_mean, step_variance, p_split):

        self.step_mean = np.asarray([10.0, .0])
        self.step_cov = step_variance * np.eye(2)
        self.p_split = p_split

        start_features = np.zeros(n_features, dtype=bool)
        start_location = np.zeros(2)

        self.root = State(start_features, start_location, rate, step_mean, self.step_cov)
        self.history = [self.root]
        self.sites = [self.root.create_child()]

    @property
    def n_sites(self):
        return len(self.sites)

    @property
    def n_internal(self):
        return len(self.history)

    def step(self):
        for i, site in enumerate(self.sites):
            site.step()

        if np.random.random() < self.p_split:
            i = np.random.randint(0, self.n_sites)
            self.split(i)

    def split(self, i):
        geo_state: State = self.sites[i]

        self.history.append(geo_state)

        c1 = geo_state.create_child()
        c2 = geo_state.create_child()

        self.sites[i] = c1
        self.sites.append(c2)

    def get_tree(self):
        vertices = []
        for i, s in enumerate(self.sites + self.history):
            vertices.append(s)
            s.id = i

        edges = []
        todo = [self.root]
        while todo:
            s: State = todo.pop()

            for c in s.children:
                edges.append((s.id, c.id))
                todo.append(c)

        return vertices, edges

    def get_newick_tree(self):
        for i, s in enumerate(self.sites + self.history):
            # TODO
            s.id = i

    def get_features(self):
        return [s.featureState.features for s in self.sites]

    def get_locations(self):
        return [s.geoState.location for s in self.sites]

    def get_history(self):
        location_history = self.get_location_history()
        feature_history = self.get_feature_history()
        return History(location_history, feature_history)

    def get_location_history(self):
        return np.array([s.location_history for s in self.sites]).swapaxes(0, 1)

    def get_feature_history(self):
        return np.array([s.feature_history for s in self.sites]).swapaxes(0, 1)

@attr.s
class History(object):

    geo_history: np.array = attr.ib(converter=np.asarray)
    feature_history: np.array  = attr.ib(converter=np.asarray)

    @property
    def n_steps(self):
        return self.geo_history.shape[0]

    @property
    def n_sites(self):
        return self.geo_history.shape[1]

    @property
    def n_features(self):
        return self.feature_history.shape[1]

    def animate_geo_history(self):
        hist = self.geo_history

        x_min, y_min, x_max, y_max = bounding_box(hist.reshape(-1, 2), margin=0.1)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x_min, x_max), ax.set_xticks([])
        ax.set_ylim(y_min, y_max), ax.set_yticks([])

        paths = []
        for i in range(self.n_sites):
            # w = signal.hann(10)**8.
            # hist[:, i, 0] = signal.convolve(hist[:, i, 0], w, mode='same') / sum(w)
            # hist[:, i, 1] = signal.convolve(hist[:, i, 1], w, mode='same') / sum(w)
            p, = ax.plot([], [], color='grey', alpha=0.1)
            paths.append(p)

        scat = ax.scatter(*hist[0, :, :].T, color='darkred', lw=0.)

        def update(i_frame):
            scat.set_offsets(hist[i_frame, :, :])
            for i, p in enumerate(paths):
                p.set_data(hist[:i_frame, i, 0], hist[:i_frame, i, 1])

        _ = animation.FuncAnimation(fig, update, frames=self.n_steps-1,
                                    interval=80, repeat_delay=1000.)

        plt.show()

    def plot_geo_history(self):
        hist = self.geo_history

        for i in range(self.n_sites):
            plt.plot(*hist[:, i].T, color='grey', lw=0.2)

        plt.scatter(*hist[0, 0], color='teal')
        plt.scatter(*hist[-1, :, :].T, color='darkred')
        plt.show()

    def write_nexus(self):
        pass


def simulate_evolution(n_steps, n_features):

    rate = 0.1
    step_mean = [0.1, 0.]
    step_variance = 1.
    p_split = 0.2

    simulation = Simulation(n_features, rate, step_mean, step_variance, p_split)


    for i in range(n_steps):
        simulation.step()

    return simulation
