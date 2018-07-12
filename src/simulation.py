#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from numpy.random import multivariate_normal as gaussian

from src.util import newick_tree


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
        age (float): The age of this specific society (since the last split).
    """

    def __init__(self, features, location, rate, geo_step_mean, geo_step_cov,
                 location_history=None, feature_history=None,
                 parent=None, children=None, name='', age=0.):
        self.geoState = GeoState(location, geo_step_mean, geo_step_cov, location_history)
        self.featureState = FeatureState(features, rate, feature_history)

        self.parent = parent
        self.children = children or []
        self._name = name
        self.age = age

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
        return self.geoState.location_history

    @property
    def feature_history(self):
        return self.featureState.feature_history

    def step(self):
        self.featureState.step()
        self.geoState.step()
        self.age += 1

    def create_child(self):
        fs = self.featureState.copy()
        gs = self.geoState.copy()

        i = str(len(self.children))
        child = State(fs.features, gs.location, fs.rate, gs.step_mean, gs.step_cov,
                      location_history=gs.location_history, feature_history=fs.feature_history,
                      parent=self, name=self._name+i)

        self.children.append(child)

        return child


class Simulation(object):

    def __init__(self, n_features, rate, step_mean, step_variance, p_split):
        self.n_features = n_features

        self.step_mean = np.asarray([10.0, .0])
        self.step_cov = step_variance * np.eye(2)
        self.p_split = p_split

        start_features = np.zeros(n_features, dtype=bool)
        start_location = np.zeros(2)

        self.root = State(start_features, start_location, rate, step_mean, self.step_cov)
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
        if np.random.random() < self.p_split:
            i = np.random.randint(0, self.n_sites)
            self.split(i)

        for i, society in enumerate(self.societies):
            society.step()

    def split(self, i):
        society = self.societies[i]

        self.history.append(society)

        c1 = society.create_child()
        c2 = society.create_child()

        self.societies[i] = c1
        self.societies.append(c2)

        print(len(self.societies))

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
        return [s.featureState.features for s in self.societies]

    def get_locations(self):
        return [s.geoState.location for s in self.societies]

    def get_location_history(self):
        return np.array([s.location_history for s in self.societies]).swapaxes(0, 1)

    def get_feature_history(self):
        return np.array([s.feature_history for s in self.societies]).swapaxes(0, 1)

