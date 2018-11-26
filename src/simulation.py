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
        self.v = np.array([0., 0.])

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
            step_cov = step_cov * np.eye(2)

        step = gaussian(step_mean, step_cov)
        # step = 0.9 * step + 0.1 * self.v
        # self.v = step
        self.location = self.location + step
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
        self.alignment = np.asarray(features)
        self.rate = rate

        self.n_features = len(self.alignment)

        if feature_history is None:
            self.feature_history = [self.alignment]
        else:
            self.feature_history = feature_history

    def step(self):
        step = bernoulli(p=self.rate, size=self.n_features)
        self.alignment = np.logical_xor(self.alignment, step)
        self.feature_history.append(self.alignment)

    def __copy__(self):
        return FeatureState(self.alignment.copy(), self.rate, self.feature_history.copy())

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
                 drift_frequency=1., location_history=None, feature_history=None,
                 parent=None, children=None, name='', length=0):
        self.geoState = GeoState(location, geo_step_mean, geo_step_cov, location_history,
                                 drift_frequency=drift_frequency)
        self.featureState = FeatureState(features, rate, feature_history)
        super(State, self).__init__(length=length, name=name, children=children,
                                    parent=parent, location=location)

    @property
    def location(self):
        return self.geoState.location

    @location.setter
    def location(self, location):
        self.geoState.location = location

    @property
    def alignment(self):
        return self.featureState.alignment

    @alignment.setter
    def alignment(self, alignment):
        self.featureState.alignment = np.asarray(alignment)


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

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def location_history(self):
        return np.asarray(self.geoState.location_history)

    @property
    def feature_history(self):
        return np.asarray(self.featureState.feature_history)

    def step(self, step_mean=None, step_cov=None):
        # self.featureState.step()
        self.geoState.step(step_mean, step_cov)
        self.length += 1

    def create_child(self):
        fs = self.featureState.copy()
        gs = self.geoState.copy()

        i = str(len(self.children))
        child = State(fs.alignment, gs.location, fs.rate, gs.step_mean, gs.step_cov,
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

        self.step_mean = step_mean
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

        self.i_step = 0

    @property
    def n_sites(self):
        return len(self.societies)

    @property
    def n_internal(self):
        return len(self.history)

    def run(self, n_steps):
        self.history = []
        self.societies = [self.root]
        self.split(0)
        # self.societies = [self.root.create_child(),
        #                   self.root.create_child()]

        for self.i_step in range(n_steps):
            self.step()

    def step(self):
        if bernoulli(self.p_split):
            i = np.random.randint(0, self.n_sites)
            self.split(i)

        for i, society in enumerate(self.societies):
            society.step()

        if self.repulsive_force > 0.:
            self.repel()

    def repel(self):
        P = self.get_locations()
        deltas = P - P[:, None]
        dists = np.hypot(*deltas.T)
        directions = deltas / dists[:, :, None]

        # Compute repulsive force between every pair of particles
        repel = self.repulsive_force * self.step_variance * directions / (1 + 1. * dists)[:, :, None]
        # repel = self.repulsive_force * directions * np.exp(-0.5 * dists ** 2. / self.step_variance)[:, :, None]

        # Don't repel yourself
        np.fill_diagonal(repel[:, :, 0], 0.)
        np.fill_diagonal(repel[:, :, 1], 0.)

        # Sum up repelling forces to get total force on every particle
        repel_sum = np.sum(repel, axis=0)

        # Apply repelling force to positions
        for i_s, s in enumerate(self.societies):
            s.geoState.location += repel_sum[i_s]

    def contact(self, contact_dist=0.5, p=0.1):
        # Compute distances (for strength of contact)
        P = self.get_locations()
        deltas = P - P[:, None]
        dists = np.hypot(*deltas.T)

        # contact_possible = (dists < contact_dist)
        p_contact = p ** dists
        contact = np.random.random(dists.shape) < p_contact
        X = self.get_features()
        # X[contact] = X[contact.T]

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

    def __init__(self, *args, backbone_steps=np.inf, **kwargs):
        super(SimulationBackbone, self).__init__(*args, **kwargs)
        self.backbone_steps = backbone_steps

    def split(self, i):
        society = self.societies[i]

        self.history.append(society)

        c1 = society.create_child()
        c2 = society.create_child()
        if self.i_step > self.backbone_steps:
            c1.geoState.step_mean = np.zeros(2)
            c2.geoState.step_mean = np.zeros(2)
        elif bernoulli(0.15):
            c1.geoState.step_mean = np.zeros(2)

        self.societies[i] = c1
        self.societies.append(c2)
