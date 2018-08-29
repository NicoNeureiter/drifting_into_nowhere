#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from numpy.random import multivariate_normal as gaussian
import matplotlib.pyplot as plt

from src.tree import Node
from src.util import newick_tree, bernoulli, norm, normalize


class GeoState(object):

    def __init__(self, location, step_mean, step_cov, location_history=None,
                 drift_frequency=1.):
        self.location = np.asarray(location)
        self.step_mean = np.asarray(step_mean)
        self.step_cov = np.asarray(step_cov)
        self.drift_frequency = drift_frequency

        if location_history is None:
            self.location_history = [self.location]
        else:
            self.location_history = location_history

    def step(self, step_mean=None, step_cov=None):
        if step_mean is None:
            if bernoulli(self.drift_frequency):
                step_mean = self.step_mean
            else:
                step_mean = np.zeros(2)

        if step_cov is None:
            step_cov = self.step_cov
        else:
            # TODO do this elegantly
            step_cov = step_cov * np.eye(2)

        self.location = self.location + gaussian(step_mean, step_cov)
        self.location_history.append(self.location)

    def __copy__(self):
        return GeoState(self.location.copy(), self.step_mean.copy(), self.step_cov.copy(),
                        self.location_history.copy(), drift_frequency=self.drift_frequency)

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
        step = bernoulli(p=self.rate, size=self.n_features)
        self.features = np.logical_xor(self.features, step)
        self.feature_history.append(self.features)

    def __copy__(self):
        return FeatureState(self.features.copy(), self.rate, self.feature_history.copy())

    def copy(self):
        return self.__copy__()


class State(Node):

    """This class captures the state for one society in the simulation. It
    is composed of the feature state and the geographic state and provides
    interfaces to the societies (features and geo) history (previous states).

    Attributes:
        geoState (GeoState): Describes the location of the society.
        featureState (FeatureState): Describes the (e.g. language) features of
            the society.
        parent (State): The state of the parent society (historical predecessor).
        children (List[State]): The successor societies.
        _name (str): A name code, implicitly representing the history of the state.
        length (float): The age of this specific society (since the last split).
    """

    def __init__(self, features, location, rate, geo_step_mean, geo_step_cov,
                 drift_frequency=1.,
                 location_history=None, feature_history=None,
                 parent=None, children=None, name='', length=0):
        self.geoState = GeoState(location, geo_step_mean, geo_step_cov, location_history,
                                 drift_frequency=drift_frequency)
        self.featureState = FeatureState(features, rate, feature_history)

        self.parent = parent
        self.children = children or []
        self._name = name
        self.length = length

    @property
    def location(self):
        return self.geoState.location

    @property
    def alignment(self):
        return self.featureState.features

    @property
    def height(self):
        if self.parent is None:
            return 0.
        else:
            return self.parent.height + self.parent.length

    @property
    def name(self):
        """Appends a prefix to the state code, depending on whether it is a
        historical or a present society.

        Returns:
            str: The name of the state
        """
        if self.children:
            return 'fossil_' + self._name
        else:
            return 'society_' + self._name

    @property
    def location_history(self):
        return np.asarray(self.geoState.location_history)

    @property
    def feature_history(self):
        return np.asarray(self.featureState.feature_history)

    def step(self, step_mean=None, step_cov=None):
        self.featureState.step()
        self.geoState.step(step_mean, step_cov)
        self.length += 1

    def create_child(self):
        fs = self.featureState.copy()
        gs = self.geoState.copy()

        i = str(len(self.children))
        child = State(fs.features, gs.location, fs.rate, gs.step_mean, gs.step_cov,
                      location_history=gs.location_history, feature_history=fs.feature_history,
                      parent=self, name=self._name+i, drift_frequency=gs.drift_frequency)

        self.children.append(child)

        return child

    def plot_walk(self):
        if self.parent is not None:
            x, y = self.location_history[-self.length-1:].T
            print(x, y)
            plt.plot(x, y, c='grey', lw=0.5)

        for c in self.children:
            c.plot_walk()


    def __repr__(self):
        return self.name


class Simulation(object):

    def __init__(self, n_features, rate, step_mean, step_variance, p_split,
                 drift_frequency=1., repulsive_force=0.):
        self.n_features = n_features

        self.step_mean = np.asarray([10.0, .0])
        self.step_variance = step_variance
        self.step_cov = step_variance * np.eye(2)
        self.p_split = p_split
        self.drift_frequency = drift_frequency
        self.repulsive_force = repulsive_force

        start_features = np.zeros(n_features, dtype=bool)
        start_location = np.zeros(2)

        self.root = State(start_features, start_location, rate, step_mean, self.step_cov, drift_frequency=drift_frequency)
        self.history = []
        self.societies = []

    @property
    def n_sites(self):
        return len(self.societies)

    @property
    def n_internal(self):
        return len(self.history)

    def run(self, n_steps):
        self.history = [self.root]
        self.societies = [self.root.create_child(),
                          self.root.create_child()]

        for i in range(n_steps):
            self.step()

    def step(self):
        if bernoulli(self.p_split):
            i = np.random.randint(0, self.n_sites)
            self.split(i)

        for i, society in enumerate(self.societies):
            society.step()

        if self.repulsive_force > 0.:
            P = self.get_locations()
            deltas = P - P[:, None]
            dists = np.hypot(*deltas.T)
            directions = deltas / dists[:, :, None]
            # dists = np.maximum(dists, 1.)
            # repel = self.repulsive_force * deltas / (dists ** 2.)[:, :, None]
            repel = self.repulsive_force * directions * np.exp(-0.5 * dists ** 2. / self.step_variance)[:, :, None]
            np.fill_diagonal(repel[:, :, 0], 0.)
            np.fill_diagonal(repel[:, :, 1], 0.)
            repel_sum = np.sum(repel, axis=0)
            # import matplotlib.pyplot as plt
            # plt.scatter(*P.T)
            for i_s, s in enumerate(self.societies):
                # print(repel_sum)
                # plt.arrow(*s.geoState.location, *repel_sum[i_s], width=0.01, length_includes_head=True, lw=0, color='k')
                s.geoState.location += repel_sum[i_s]
            # plt.axis('equal')
            # plt.margins(1.)
            # plt.show()


    def split(self, i):
        society = self.societies[i]

        self.history.append(society)

        c1 = society.create_child()
        c2 = society.create_child()

        self.societies[i] = c1
        self.societies.append(c2)

    def get_tree(self):
        vertices = []
        for i, s in enumerate(self.societies + self.history):
            vertices.append(s)
            s.id = i

        edges = []
        todo = [self.root]
        while todo:
            s = todo.pop()

            for c in s.children:
                edges.append((s.id, c.id))
                todo.append(c)

        return vertices, edges

    def get_newick_tree(self):
        return newick_tree(self.root)

    def get_features(self):
        return np.array([s.featureState.features for s in self.societies])

    def get_locations(self):
        return np.array([s.geoState.location for s in self.societies])

    def get_location_history(self):
        return np.array([s.location_history for s in self.societies]).swapaxes(0, 1)

    def get_feature_history(self):
        return np.array([s.feature_history for s in self.societies]).swapaxes(0, 1)


class SimulationBackbone(Simulation):

    def run(self, n_steps):
        self.history = [self.root]
        self.societies = [self.root.create_child(),
                          self.root.create_child()]

        for i in range(n_steps):
            self.step()
