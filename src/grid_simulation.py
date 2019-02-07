#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import random
from copy import deepcopy

import numpy as np
from scipy.ndimage.morphology import binary_dilation

from src.simulation import World
from src.util import bernoulli
from src.tree import Node


def neighbourhood(cells):
    """Compute the neighbourhood of the area in cells, i.e. any adjacent cells
    that are not in the area itself."""
    # return binary_dilation(cells, structure=np.ones((3, 3))) ^ cells
    return binary_dilation(cells) ^ cells


def grid_to_index_tuples(cells):
    """Transform a binary grid into a list of cell indices."""
    return list(zip(*np.nonzero(cells)))


def project_grid(cells, d):
    d = np.asarray(d)

    xy = np.array(np.nonzero(cells))
    projected = d.dot(xy)

    m = np.median(projected)
    is_zone_1 = (projected < m)
    return xy.T[is_zone_1].T


def max_var_projected_grid(cells, n_proj=10):
    xy = np.array(np.nonzero(cells))

    # Compute projections n ´n_proj´ different directions
    d0 = np.random.uniform(0, 2*np.pi/n_proj)
    directions = np.linspace(0, 2*np.pi, n_proj, endpoint=False)
    p = np.array([np.cos(directions),
                  np.sin(directions)]).T
    all_projected = p.dot(xy)

    # Compute variance for every projection and take the one with max variance.
    variances = np.var(all_projected, axis=1)
    i_max_var = np.argmax(variances)
    projected = all_projected[i_max_var]

    # Cut the projected grid points at the median
    m = np.median(projected)
    is_zone_1 = (projected < m)
    return xy.T[is_zone_1].T

class SuperState(Node):
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

    def __init__(self, world, parent=None, children=None, name='',
                 length=0, age=0, location=None):
        super(SuperState, self).__init__(length=length, name=name, children=children,
                                    parent=parent, location=location)
        self.world = world
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

    @name.setter
    def name(self, name):
        self._name = name

    def step(self):
        # Update length and age
        self.length += 1
        self.age += 1

        if bernoulli(self.split_probability()):
            self.split()

    def split_probability(self):
        raise NotImplementedError

    def split(self):
        world = self.world
        i = world.sites.index(self)

        c1 = self.create_child()
        c2 = self.create_child()
        self.children = [c1, c2]

        world.sites[i] = c1
        world.sites.append(c2)

    def create_child(self):
        raise NotImplementedError


class GridState(SuperState):

    """The geographical state of a site (language area) represented by cells on
    a grid.

    Attributes:
        world (GridWorld): The simulated world containing sites including their
            ´GridState´s.
        cells (np.array[bool]): Binary array of the whole gridworld. 1 indicates
            that the state covers the corresponding cell.
        p_grow (float): Probability of growing by one cell at each step.
        split_size_range (tuple[int, int]): Minimum area at which a split could
            happen and maximum size when a split will certainly happen. Splitting
            probability is linearly interpolated in between.
    """

    def __init__(self, world, start_cells, p_grow, split_area_range,
                 parent=None, children=None, name='', length=0, age=0):
        super(GridState, self).__init__(world, parent, children, name, length, age)
        self.cells = start_cells
        self.p_grow = p_grow
        self.split_area_range = split_area_range
        self.stuck = False

    @property
    def area(self):
        return np.count_nonzero(self.cells)

    @property
    def location(self):
        cell_locations = grid_to_index_tuples(self.cells)
        return np.mean(cell_locations, axis=0)[::-1]

    def split_probability(self):
        area = self.area
        lower, upper = self.split_area_range

        if lower <= area:
            p_split = (area - lower + 1) / (upper - lower + 1)
            p_split = 1 - (1 - p_split) ** self.p_grow
        else:
            p_split = 0

        return p_split

    def step(self):
        if not self.stuck:
            if bernoulli(self.p_grow):
                self.grow()
                self.grow() # TODO remove

        super(GridState, self).step()

    def grow(self):
        """Extend the current area by a random adjacent cell."""

        # Find free neighbouring cells as candidates for expansion
        neighbour_cells = neighbourhood(self.cells)
        neighbour_cells &= self.world.free_space()
        candidates = grid_to_index_tuples(neighbour_cells)

        # In case there is no space to grow: don't even attempt to in the future
        if len(candidates) == 0:
            print('No space to grow...')
            # self.p_grow = 0
            self.stuck = True
            return

        # Randomly add one of the candidate cells to the site
        i, j = random.choice(candidates)
        self.cells[i, j] = True
        self.world.occupancy_grid[i,j] = True
        assert np.sum([s.cells for s in world.sites], axis=0)[i, j] == 1
        assert np.max(np.sum([s.cells for s in world.sites], axis=0)) <= 1

    def split_area(self):
        # Create a new array for the cells that split of into a new state
        cells_1 = np.array(self.cells)
        cells_2 = np.zeros_like(self.cells)

        # d_project = np.random.random(2)
        # rows, cols = project_grid(self.cells, d_project)
        rows, cols = max_var_projected_grid(self.cells)

        # Set first rows in new cells and remove them from the old cells
        cells_1[rows, cols] = False
        cells_2[rows, cols] = True
        return cells_1, cells_2

        # # Find the cut-off row, where the area is split
        # i, j = np.nonzero(self.cells)
        # i_split = int(np.median(i))
        #
        # # Create and return the new (split-off) state
        # new_state = GridState(self.world, cells_new, self.p_grow, self.split_area_range)
        # return new_state

    def split(self):
        super(GridState, self).split()
        assert len(self.children) == 2, self.children

        child_1, child_2 = self.children
        child_1.cells, child_2.cells = self.split_area()

    def create_child(self):
        i = str(len(self.children))
        child_name = self._name + i

        child = GridState(self.world, self.cells, self.p_grow, self.split_area_range,
                          parent=self, name=child_name, age=self.age)

        return child


class GridWorld(World):

    """
    Attributes:
        grid_size (tuple): The size of the simulated grid.
        sites (list): The list of sites in the simulated world.
        occupancy_grid (np.array): A grid indicating whether a fixed field is
            already occupied by any site.
    """

    def __init__(self, grid_size):
        super(GridWorld, self).__init__()
        self.grid_size = grid_size
        # self.occupancy_grid = np.zeros(grid_size, dtype=bool)

    @property
    def occupancy_grid(self):
        all_grids = [s.cells for s in self.sites]
        return np.any(all_grids, axis=0)

    def free_space(self):
        return ~self.occupancy_grid


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from src.config import COLORS_RGB
    from src.plotting import plot_tree, plot_height, plot_tree_topology, plot_edge

    n = 100
    a = np.zeros((n, n), dtype=bool)
    i, j = np.random.randint(0, n, size=2)
    a[i, j] = True
    world = GridWorld((n, n))
    s0 = GridState(world, a, 1., (60, 100))
    world.sites = [s0]

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    img = np.ones((n, n, 3)).astype(int) * 255
    img_plt = plt.imshow(img, alpha=0.7)
    edges_plotted = []

    def update(i_frame):
        global states, img, ax

        for i, s in enumerate(list(world.sites)):
            img[s.cells] = COLORS_RGB[i]

            s.step()

        assert np.max(np.sum([s.cells for s in world.sites], axis=0)) <= 1

        img_plt.set_array(img)
        print(i_frame)
        if i_frame % 10 == 0:
            # plot_tree(s0, ax=ax)
            for parent, child in s0.iter_edges():
                # if (parent, child) in edges_plotted:
                #     continue
                if parent.stuck:
                    plot_edge(parent, child, ax=ax, no_arrow=False, lw=0, width=0.1, color='k')
                    # edges_plotted.append((parent, child))

        return img_plt,


    # writer = animation.FFMpegFileWriter(fps=20)
    # anim = animation.FuncAnimation(fig, update, frames=50, interval=20, blit=False)
    anim = animation.FuncAnimation(fig, update, frames=201, interval=20)
    plt.axis("off")
    plt.tight_layout(pad=0.)
    anim.save('results/grid_simulation.mp4', writer='imagemagick', dpi=100)
    # anim.save('results/grid_simulation.mp4', fps=20, writer="avconv", codec="libx264")

    #
    # img_plots = []
    # for _ in range(500):
    #     for s in list(world.sites):
    #         s.step()
    #
    #     for i, s in enumerate(list(world.sites)):
    #         img[s.cells] = COLORS_RGB[i]
    #
    #     im = plt.imshow(img, alpha=0.75)
    #     img_plots.append([im])
    #
    # anim = animation.ArtistAnimation(fig, img_plots, interval=10, blit=True)

    # anim.save('results/grid_simulation.mov')  #, writer=writer)
    # plt.axis("off")
    # plt.tight_layout(pad=0.)
    # plt.show()

    # plot_tree_topology(s0)
    #
    # plt.show()