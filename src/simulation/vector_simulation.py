#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import collections
import random

import numpy as np
from numpy.random import multivariate_normal as _gaussian
import matplotlib.pyplot as plt
from scipy.special import softmax

from src.simulation.simulation import State, World
from src.util import newick_tree, bernoulli, norm, normalize
from src.tree import get_edge_heights

YULE = 'yule'
SATURATION = 'saturation'
LINEAR = 'linear'
TREE_MODELS = [YULE, SATURATION, LINEAR]
TREE_MODEL = TREE_MODELS[0]


gauss_samples = collections.deque(list(_gaussian(np.zeros(2), np.eye(2,2), size=1000)))
def gaussian(mean: np.array, var):
    # return _gaussian(mean, var)
    if len(gauss_samples) == 0:
        gauss_samples.extend(list(_gaussian(np.zeros(2), np.eye(2,2), size=10000)))
    return mean + (var**0.5).dot(gauss_samples.pop())


class VectorState(State):

    """This class represents a model for the geographic state of a society
    represented by a point location (vector) and implements the logic for a
    step of this location in the simulation.

    Attributes:
        world (VectorWorld): A world (environment) in which the state is
            embedded, providing an interface to other states in the simulation.
        location (np.array): The current location of the society in space.
        step_mean (np.array): The mean of a gaussian step (bias/offset).
        step_cov (np.array): The covariance matrix, describing the diffusion
            properties of the states movement.
        drift_frequency (float [0,1]): Frequency at which step_mean is applied.
        parent (VectorState): The state of the parent society (historical predecessor).
        children (List[VectorState]): The successor sites.
        _name (str): A name code, implicitly representing the history of the state.
        length (float): The age of this specific society (since the last split).
    """

    def __init__(self, world, location, step_mean, step_cov, clock_rate,
                 birth_rate, drift_frequency=1., location_history=None,
                 parent=None, children=None, name='', length=0, age=0.,
                 v=(0,0), death_rate=0.):
        # Ensure that we are working with numpy arrays
        location = np.asarray(location)
        self.clock_rate = clock_rate
        self.birth_rate = birth_rate
        self.step_mean = np.asarray(step_mean)
        self.step_cov = np.asarray(step_cov)
        if len(self.step_cov.shape) < 2:
            self.step_cov = self.step_cov * np.eye(2)
        self.drift_frequency = drift_frequency
        self.drift = bernoulli(self.drift_frequency)
        self._death_rate = death_rate

        self.v = np.asarray(v)

        super(VectorState, self).__init__(world, parent=parent, children=children,
                                          name=name, length=length, age=age,
                                          location=location)

        if location_history is None:
            self.location_history = [self.location]
        else:
            self.location_history = location_history

    @property
    def name(self):
        """Appends a prefix to the state code, depending on whether it is a
        historical or a present society.

        Returns:
            str: The name of the state
        """
        if self.children:
            return 'internal_' + self._name
        else:
            return 'leaf_' + self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def death_rate(self):
        # if self.age < 180:
        #     return 0
        # else:
        return self._death_rate

    def step(self, step_mean=None, step_cov=None, last_step=False):
        """Perform a simulation step: Gaussian distribution relative to current
        location. Parameterized by mean, covariance, frequency. super(..) will
        handle random splits and bookkeeping.

        Kwargs:
            step_mean (np.array or None): Optional step-specific offset.
            step_cov (np.array or None): Optional step-specific covariance.
        """

        # Get step_mean from argument or (if arg is None) from attribute
        step_mean = step_mean or self.step_mean
        step_cov = step_cov or self.step_cov
        s = step_cov / self.clock_rate

        # Draw the step according to a gaussian and apply it to current location
        if self.drift:
            step = gaussian(step_mean / self.clock_rate, s)
        else:
            step = gaussian([0, 0], s)
        self.location = self.location + step

        # Add the new location to the history
        self.location_history.append(self.location)

        super(VectorState, self).step(last_step=last_step)

    def split_probability(self):
        # splits_per_step = splits_per_year / steps_per_year
        birth_rate_step = self.birth_rate / self.clock_rate

        if TREE_MODEL == YULE:
            return birth_rate_step
        elif TREE_MODEL == LINEAR:
            return birth_rate_step / self.world.n_sites
        elif TREE_MODEL == SATURATION:
            saturation = self.world.n_sites / self.world.capacity
            return birth_rate_step * (1 - saturation)

    def create_child(self):
        i = str(len(self.children))
        child_name = self._name + i
        child = VectorState(self.world, self.location.copy(), self.step_mean.copy(),
                            self.step_cov.copy(), self.clock_rate, self.birth_rate,
                            drift_frequency=self.drift_frequency, parent=self,
                            # location_history=self.location_history.copy(),
                            name=child_name, age=self.age, death_rate=self._death_rate
                            # v=self.v.copy()
                            )

        return child

    def __repr__(self):
        return self.name


class VectorWorld(World):

    """Object describing the state of the simulated world at a given point in
    time. It contains a list of all societies and potentially further
    information about the simulated environment.

    Attributes:
        sites (list): A list of all currently existing sites (horizontal slice
            view of the simulated phylogeny).

    """

    def __init__(self, capacity=np.inf):
        super(VectorWorld, self).__init__()
        self.capacity = capacity

    def get_locations(self):
        return np.array([s.location for s in self.sites])

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


class BackboneState(VectorState):

    def __init__(self, *args, is_backbone=True, bb_stop=np.inf, **kwargs):
        super(BackboneState, self).__init__(*args, **kwargs)
        self.bb_stop = bb_stop
        self.is_backbone = is_backbone

    def create_child(self):
        i = str(len(self.children))
        child_name = self._name + i

        birth_rate = self.birth_rate
        if (i == '0') or (self.age > self.bb_stop):
            is_backbone = False
            step_mean = np.zeros(2)
            if self.is_backbone:
                birth_rate = self.birth_rate / 6

        else:
            is_backbone = self.is_backbone
            step_mean = self.step_mean.copy()
            birth_rate = self.birth_rate

        child = BackboneState(self.world, self.location.copy(), step_mean,
                            self.step_cov.copy(), self.clock_rate, birth_rate,
                            drift_frequency=self.drift_frequency,
                            location_history=self.location_history.copy(), parent=self,
                            name=child_name, age=self.age, is_backbone=is_backbone,
                            bb_stop=self.bb_stop, death_rate=self._death_rate)

        return child


if __name__ == '__main__':
    import random
    from src.simulation.simulation import run_simulation
    from src.plotting import plot_root, plot_tree, plot_tree_topology
    from src.config import COLOR_ROOT_TRUE, COLOR_SCATTER
    from src.util import grey


    rnd_seed = np.random.randint(0,4000000000)
    # print(rnd_seed)
    # rnd_seed = 3073416186
    # rnd_seed = 3547844550
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    print('Random Seed:', rnd_seed)

    p0 = np.zeros(2)
    mean = 0.5 * np.ones(2)
    var = np.eye(2)
    clock_rate = 1.
    birth_rate = .022
    drift_frequency = 0.5

    # Keep total mean fixed
    mean /= drift_frequency

    world = VectorWorld()
    root = VectorState(world, p0, mean, var, clock_rate, birth_rate,
                       drift_frequency=drift_frequency)

    run_simulation(80, root, world)

    n = root.n_leafs()

    # plot_tree(root, lw=1., color_fun=get_edge_heights)
    # plt.scatter(*root.get_leaf_locations().T, c='k', s=5.)
    # plot_root((0,0), s=800, color='k')

    plot_tree_topology(root)
    plt.scatter(range(n), np.zeros(n), color='k', s=10.)

    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.show()
