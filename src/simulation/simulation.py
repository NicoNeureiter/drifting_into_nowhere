#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

from src.tree import Tree
from src.util import newick_tree, bernoulli


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
        self.root = None

    def set_root(self, root):
        """Define the root / first site of this World."""
        self.root = root
        self.sites = [root]

    @property
    def n_sites(self):
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

    def register_split(self, parent, child_1, child_2):
        i = self.sites.index(parent)
        self.sites[i] = child_1
        self.sites.append(child_2)

    def register_death(self, node):
        self.sites.remove(node)

    def get_newick_tree(self):
        return newick_tree(self.root)

    def stop_condition(self):
        return False


class State(Tree):
    """This class captures the state for one society in the simulation. It
    is composed of the feature state and the geographic state and provides
    interfaces to the sites (features and geo) history (previous states).

    Attributes:
        world (World): The simulated world containing all sites.
        parent (State): The state of the parent society (historical predecessor).
        children (List[State]): The successor sites.
        _name (str): A name code, implicitly representing the history of the state.
        length (float): The age of this specific society (since the last split).
    """

    def __init__(self, world, clock_rate=1., parent=None, children=None, name='',
                 length=0, age=0, location=None):
        super(State, self).__init__(length=length, name=name, children=children,
                                    parent=parent, location=location)
        self.world = world
        self.age = age
        self.clock_rate = clock_rate

    @property
    def name(self):
        """Appends a prefix to the state code, depending on whether it is a
        historical or a present society.

        Returns:
            str: The name of the state
        """
        return 'n' + self._name
        # if self.children:
        #     return 'fossil_' + self._name
        # else:
        #     return 'society_' + self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def death_rate(self):
        """The death rate of the taxon. Implemented as a property, since it
        could depend on time, diversity, environmental factors,..."""
        raise NotImplementedError

    def step(self, last_step=False):
        # Update length and age
        self.length += 1
        self.age += 1

        if bernoulli(self.death_rate * self.clock_rate):
            self.die()
        elif bernoulli(self.split_probability()):
            if not last_step:
                self.split()

    def split_probability(self):
        raise NotImplementedError

    def split(self):

        c1 = self.create_child()
        self.children = [c1]
        c2 = self.create_child()
        self.children.append(c2)

        self.world.register_split(self, c1, c2)

    def die(self):
        self.world.register_death(self)

    def create_child(self):
        """Create a child of the current state. Most parameters will be copied
        (carefull with references), some may be changed. """
        raise NotImplementedError


class BirthDeathState(State):

    def __init__(self, world, birth_rate, death_rate, **kwargs):
        super(BirthDeathState, self).__init__(world, **kwargs)
        self.birth_rate = birth_rate
        self._death_rate = death_rate

    @property
    def death_rate(self):
        return self._death_rate

    def split_probability(self):
        return self.birth_rate * self.clock_rate

    def create_child(self):
        i = str(len(self.children))
        child_name = self._name + i
        child = BirthDeathState(world=self.world,
                                birth_rate=self.birth_rate, death_rate=self._death_rate,
                                clock_rate=self.clock_rate,
                                parent=self, name=child_name, age=self.age)

        return child


def run_simulation(n_steps, root, world, condition_on_root=False):
    """Run a simulation for n_steps. The starting state is defined by ´root´,
    the environment is defined by ´world´.

    Args:
        n_steps (int): Length of the simulation run in steps.
        root (State): The initial state of the simulation.
        world (World): The environment of the simulation, keeping track of all
            states, providing some utility methods, relating to global properties.
    """
    root_init = deepcopy(root)
    world_init = deepcopy(world)

    # Add the root to the world and perform first split.
    # (a phylogeny always starts with the first split)
    world.set_root(root)
    root.split()

    for i_step in range(n_steps):
        last_step = (i_step == n_steps-1)
        for state in list(world.sites):
            state.step(last_step=last_step)

        if condition_on_root:
            if world.n_sites <= 1:
                root = root_init
                world = world_init
                root.world = world
                return run_simulation(n_steps, root_init, world_init,
                                      condition_on_root=condition_on_root)

        if world.stop_condition():
            break

    return root, world


def run_backbone_simulation(n_steps, root, world, backbone_steps=None):
    """

    Args:
        n_steps (int):
        root (State):
        world (World):
        backbone_steps (int):

    Returns:

    """
    if backbone_steps is None:
        backbone_steps = n_steps

    world.set_root(root)
    bb_node = root
    bb_node.split()
    for i in range(backbone_steps):
        # Update bb_node after splits
        if bb_node.children:
            bb_node = bb_node.children[0]

        # All nodes step
        for state in list(world.sites):
            state.step()

        # Split bb_node on every third step (if it didn't happend already)
        if (i % 3 == 0) and not bb_node.children:
            bb_node.split()

    # Finish the remaining steps after backbone is simulated
    for _ in range(n_steps - backbone_steps):
        for state in list(world.sites):
            state.step()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.plotting import plot_tree_topology
    from src.tree import tree_imbalance

    imbalance_scores = []
    for _ in range(100):
        world = World()
        root = BirthDeathState(world=world, birth_rate=1., death_rate=0., clock_rate=0.05)
        print(root.name)
        tree, world = run_simulation(100, root, world, condition_on_root=True)
        tree.drop_fossils()

        print(tree.n_leafs())
        imb = tree_imbalance(tree, weight_by_age=True)
        print(imb)

        if not np.isfinite(imb):
            plot_tree_topology(tree)
            plt.show()
        else:
            imbalance_scores.append(imb)

    print()
    print(np.mean(imbalance_scores))
