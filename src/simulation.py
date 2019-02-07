#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import collections
import random

import numpy as np
from numpy.random import multivariate_normal as gaussian
import matplotlib.pyplot as plt
from scipy.special import softmax

from src.tree import Node
from src.util import newick_tree, bernoulli, norm, normalize


class GeoState(object):

    def __init__(self, world, location, step_mean, step_cov, location_history=None,
                 drift_frequency=1., min_distance=.1):
        self.world = world
        self.location = np.asarray(location)
        self.step_mean = np.asarray(step_mean)
        self.step_cov = np.asarray(step_cov)
        self.drift_frequency = drift_frequency
        self.v = np.array([0., 0.])
        self.min_distance = min_distance

        if location_history is None:
            self.location_history = [self.location]
        else:
            self.location_history = location_history

    def step(self, step_mean=None, step_cov=None, n_tries=5):
        if step_mean is None:
            if bernoulli(self.drift_frequency):
                step_mean = self.step_mean
            else:
                step_mean = np.zeros(2)

        if step_cov is None:
            step_cov = self.step_cov
        else:
            step_cov = step_cov * np.eye(2)

        for _ in range(n_tries):
            step = gaussian(step_mean, step_cov)
            # step = 0.9 * step + 0.1 * self.v
            # self.v = step
            location_candidate = self.location + step
            if True:  # self.try_step(location_candidate):
                self.location = location_candidate
                break

        self.location_history.append(self.location)

    def try_step(self, x):
        dists = self.world.distances(x)
        return np.any(np.logical_and(0 < dists, dists < self.min_distance))

    def __copy__(self):
        return GeoState(self.world, self.location.copy(), self.step_mean.copy(),
                        self.step_cov.copy(),
                        location_history=self.location_history.copy(),
                        drift_frequency=self.drift_frequency)

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
    interfaces to the sites (features and geo) history (previous states).

    Attributes:
        geoState (GeoState): Describes the location of the society.
        featureState (FeatureState): Describes the (e.g. language) features of
            the society.
        parent (State): The state of the parent society (historical predecessor).
        children (List[State]): The successor sites.
        _name (str): A name code, implicitly representing the history of the state.
        length (float): The age of this specific society (since the last split).
    """

    def __init__(self, world, features, location, rate, geo_step_mean, geo_step_cov,
                 drift_frequency=1., location_history=None, feature_history=None,
                 parent=None, children=None, name='', length=0, age=0):
        self.geoState = GeoState(world, location, geo_step_mean, geo_step_cov, location_history,
                                 drift_frequency=drift_frequency)
        self.featureState = FeatureState(features, rate, feature_history)
        super(State, self).__init__(length=length, name=name, children=children,
                                    parent=parent, location=location)
        self.age = age

        self.world = world

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
        self.age += 1

        if bernoulli(self.split_probability()):
            self.split()

    def split_probability(self):
        return 0.

    def split(self):
        world = self.world
        i = world.sites.index(self)

        c1 = self.create_child()
        c2 = self.create_child()

        world.sites[i] = c1
        world.sites.append(c2)

    def create_child(self):
        fs = self.featureState.copy()
        gs = self.geoState.copy()

        i = str(len(self.children))
        child = State(self.world, fs.alignment, gs.location, fs.rate, gs.step_mean, gs.step_cov,
                      location_history=gs.location_history, feature_history=fs.feature_history,
                      parent=self, name=self._name+i, drift_frequency=gs.drift_frequency,
                      age=self.age)

        self.children.append(child)

        return child

    def plot_walk(self):
        if self.parent is not None:
            x, y = self.location_history[-self.length-1:].T
            plt.plot(x, y, c='grey', lw=0.5)

        for c in self.children:
            c.plot_walk()

    def __repr__(self):
        return self.name


class World(object):

    """Object describing the state of the simulated world at a given point in
    time. It contains a list of all societies and potentially further
    information about the simulated environment.

    Attributes:
        sites (list): A list of all currently existing sites (horizontal slice
            view of the simulated phylogeny).

    """

    def __init__(self):
        self.sites = []

    @property
    def n_site(self):
        return len(self.sites)

    def get_locations(self):
        return np.array([s.geoState.location for s in self.sites])

    def deltas(self, x):
        P = self.get_locations()
        return P - x

    def distances(self, x):
        return np.hypot(*self.deltas(x).T)

    def all_distances(self):
        P = self.get_locations()
        deltas = P - P[:, None]
        return np.hypot(*deltas.T)

    def all_min_distances(self):
        dists = self.all_distances()
        np.fill_diagonal(dists, np.inf)
        return np.min(dists, axis=1)


class Simulation(object):

    def __init__(self, n_features, rate, step_mean, step_variance, p_split,
                 p_settle=0., drift_frequency=1., repulsive_force=0.):
        self.n_features = n_features

        self.step_mean = step_mean
        self.step_variance = step_variance
        self.step_cov = step_variance * np.eye(2)
        self.p_split = p_split
        self.p_settle = p_settle
        self.drift_frequency = drift_frequency
        self.repulsive_force = repulsive_force

        start_features = np.zeros(n_features, dtype=bool)
        start_location = np.zeros(2)

        self.world = World()
        self.root = State(self.world, start_features, start_location, rate,
                          step_mean, self.step_cov,
                          drift_frequency=drift_frequency)
        self.history = []

        self.i_step = 0

    @property
    def sites(self):
        return self.world.sites

    @property
    def n_sites(self):
        return len(self.sites)

    @property
    def n_internal(self):
        return len(self.history)

    def run(self, n_steps):
        self.history = []
        self.world.sites = [self.root]
        self.split(0)

        for self.i_step in range(n_steps):
            self.step()

    def step(self):
        # if bernoulli(self.p_split):
        #     p = self.world.all_min_distances()
        #     # p /= np.sum(p)
        #     p = softmax(10*p)
        #     # print(p)
        #     i = np.random.choice(self.n_sites, p=p)
        #     # i = np.random.randint(0, self.n_sites)
        #     self.split(i)

        for i, society in enumerate(self.sites):
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
        for i_s, s in enumerate(self.sites):
            s.geoState.location += repel_sum[i_s]

    def split(self, i):
        society = self.sites[i]

        self.history.append(society)

        c1 = society.create_child()
        c2 = society.create_child()

        if bernoulli(self.p_settle):
            c1.geoState.step_mean = np.zeros(2)

        self.sites[i] = c1
        self.sites.append(c2)

    def get_newick_tree(self):
        return newick_tree(self.root)

    def get_features(self):
        return np.array([s.featureState.features for s in self.sites])

    def get_locations(self):
        return np.array([s.geoState.location for s in self.sites])

    def get_location_history(self):
        return np.array([s.location_history for s in self.sites]).swapaxes(0, 1)

    def get_feature_history(self):
        return np.array([s.feature_history for s in self.sites]).swapaxes(0, 1)


class SimulationBackbone(Simulation):

    def __init__(self, *args, backbone_steps=np.inf, **kwargs):
        super(SimulationBackbone, self).__init__(*args, **kwargs)
        self.backbone_steps = backbone_steps

    def run(self, n_steps):
        self.history = []
        self.world.sites = [self.root]

        # backbone = []
        for _ in range(min(self.backbone_steps, n_steps)):
            self.split(0)

            self.sites[0].step()
            self.sites[0].step()
            self.sites[0].step()
            self.sites[0].step()
            # backbone.append(self.sites[0].geoState.location)

        bb_sites = list(self.sites)
        n_clades = len(bb_sites)

        for i_child, child in enumerate(bb_sites):
            clade = {i_child: child}
            for self.i_step in range(child.age, n_steps):
                if bernoulli(self.p_split / n_clades):
                    j1, s = random.choice(list(clade.items()))
                    self.split(j1)
                    clade[j1] = self.sites[j1]
                    j2 = self.n_sites-1
                    clade[j2] = self.sites[j2]

                for s in clade.values():
                    # s.step(step_mean=self.step_mean, step_cov=self.step_cov)
                    s.step(step_mean=(0, 0), step_cov=self.step_cov)

        # backbone = np.array(backbone)


    # def split(self, i):
    #     society = self.sites[i]
    #
    #     self.history.append(society)
    #
    #     c1 = society.create_child()
    #     c2 = society.create_child()
    #     if self.i_step > self.backbone_steps:
    #         c1.geoState.step_mean = np.zeros(2)
    #         c2.geoState.step_mean = np.zeros(2)
    #     elif bernoulli(0.15):
    #         c1.geoState.step_mean = np.zeros(2)
    #
    #     self.sites[i] = c1
    #     self.sites.append(c2)

if __name__ == '__main__':
    from src.plotting import plot_root
    from src.config import COLOR_ROOT_TRUE, COLOR_SCATTER
    from src.util import grey


    rnd_seed = np.random.randint(0,4000000000)
    print(rnd_seed)
    rnd_seed = 3073416186
    np.random.seed(rnd_seed)
    print('Random Seed:', rnd_seed)

    p0 = np.zeros(2)
    mean = np.zeros(2)
    var = np.eye(2)
    world = World()
    particle = GeoState(world, p0, mean, var)

    walk = [particle.location.copy()]
    for _ in range(7000):
        particle.step()
        walk.append(particle.location.copy())

    x, y = np.array(walk).T

    plt.plot(x, y, c=grey(0.3), lw=0.5, zorder=0)
    plot_root((0,0), s=800, color=COLOR_ROOT_TRUE)
    plt.scatter(x[-1], y[-1], c=COLOR_SCATTER, lw=0, s=300, zorder=1)

    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()