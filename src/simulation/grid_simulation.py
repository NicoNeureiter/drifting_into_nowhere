#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import random

import numpy as np
import scipy as sp
from scipy.ndimage.morphology import binary_dilation
from scipy.stats import beta

from src.simulation.simulation import World, State
from src.util import bernoulli, experiment_preperations

# Plotting imports...
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.config import COLORS_RGB, gamma_transform, GREY_TONES
from src.plotting import plot_tree_topology, plot_edge


DEBUG = False


def neighbourhood(cells):
    """Compute the neighbourhood of the area in cells, i.e. any adjacent cells
    that are not in the area itself."""
    # return binary_dilation(cells, structure=np.ones((3, 3))) ^ cells
    return binary_dilation(cells) ^ cells


def get_neighbours(i, j):
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


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
    """Find the direction with maximum variance and split the cells into two
    equal areas along this direction. Return the indices of one of the two areas.

    Args:
        cells (np.array): Boolean array representing the initial area.
            shape: (grid_height, grid_width)

    Kwargs:
        n_proj (int): Number equidistant projections to check for max variacne.

    Returns:
        np.array: Indices of zone_1 (one of the two almost equal areas).
            shape: (2, size(zone_1))

    TODO: 1st principle component instead of random directions?
    """
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


class GridArea(object):

    def __init__(self, cells):
        self.cells = cells
        self.neighbourhood = neighbourhood(cells)

    @property
    def shape(self):
        return self.cells.shape

    def add_cell(self, i, j):
        # TODO Add sparse adjacency matrix/list to world
        self.cells[i, j] = True
        self.neighbourhood[i, j] = False
        for i2, j2 in get_neighbours(i, j):
            if (0 <= i2 < self.shape[0]) and (0 <= j2 < self.shape[1]):
                if self.cells[i2, j2]:
                    self.neighbourhood[i2, j2] = True


class GridState(State):

    """The geographical state of a site (language area) represented by cells on
    a grid.

    Attributes:
        world (GridWorld): The simulated world containing sites including their
            ´GridState´s.
        cells (np.array[bool]): Binary array of the whole gridworld. 1 indicates
            that the state covers the corresponding cell.
        p_grow (float): Probability of growing by one cell at each step.
        split_size_range (tuple[int, int]): Minimum area at which a split could
            happen and maximum size when a split will certainly happen. The
            actual split_size is selected uniformly at random from this interval
            on initialization.
    """

    def __init__(self, world, start_cells, p_grow_distr, split_size_range,
                 p_conflict=0.,
                 parent=None, children=None, name='', length=0, age=0):
        super(GridState, self).__init__(world, parent=parent, children=children,
                                        name=name, length=length, age=age)
        self.cells = start_cells
        self.p_grow_distr = p_grow_distr
        self.p_grow = p_grow_distr()
        self.p_conflict = p_conflict
        self.split_area_range = split_size_range

        self.split_size_range = split_size_range
        self.split_size = np.random.randint(*split_size_range)
        self.stuck = False

        self.clock_rate = 1.

    @property
    def area(self):
        return np.count_nonzero(self.cells)

    @property
    def location(self):
        cell_indices = grid_to_index_tuples(self.cells)
        mean_index = np.mean(cell_indices, axis=0)[::-1]
        return self.world.km_per_cell * mean_index

    @property
    def grid_size(self):
        return self.cells.shape

    def split_probability(self):
        return float(self.area >= self.split_size)

    @property
    def death_rate(self):
        return 0.  # TODO implement death

    def step(self, last_step=False):
        if not self.stuck:
            if bernoulli(self.p_grow):
                self.grow()

        super(GridState, self).step(last_step=last_step)

    def grow(self):
        """Extend the current area by a random adjacent cell."""
        world = self.world

        # Find free neighbouring cells as candidates for expansion
        neighbour_cells = neighbourhood(self.cells)
        # if bernoulli(1 - self.p_conflict):
        neighbour_cells &= world.free_space()
        # else:
        #     print('CONFLICT!!!')
        candidates = grid_to_index_tuples(neighbour_cells)

        # In case there is no space to grow: don't even attempt to in the future
        if len(candidates) == 0:
            # print('No space to grow...')
            self.stuck = True
            return

        # Randomly add one of the candidate cells to the site
        # n_samples = min(3, len(candidates))
        n_samples = 1
        for i, j in random.sample(candidates, n_samples):
            self.cells[i, j] = True
            world.occupancy_grid[i,j] = True

        # These would be sensible assertions for debuggin (very costly though):
        if DEBUG:
            assert np.sum([s.cells for s in world.sites], axis=0)[i, j] == 1
            assert np.max(np.sum([s.cells for s in world.sites], axis=0)) <= 1

    def split_area(self):
        # Create a new array for the cells that split of into a new state
        cells_1 = np.array(self.cells)
        cells_2 = np.zeros_like(self.cells)

        # Find the row/col indices of the
        rows, cols = max_var_projected_grid(self.cells)

        # Set first rows in new cells and remove them from the old cells
        cells_1[rows, cols] = False
        cells_2[rows, cols] = True

        # print('SPLIT! New sizes: (%i, %i)' % (np.count_nonzero(cells_1), np.count_nonzero(cells_2)))
        return cells_1, cells_2

    def split(self):
        super(GridState, self).split()
        assert len(self.children) == 2, self.children

        child_1, child_2 = self.children
        child_1.cells, child_2.cells = self.split_area()

    def create_child(self):
        i = str(len(self.children))
        child_name = self._name + i

        child = GridState(self.world, self.cells, self.p_grow_distr, self.split_area_range,
                          p_conflict=self.p_conflict,
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

    def __init__(self, grid_size, occupancy_grid=None, km_per_cell=1.):
        super(GridWorld, self).__init__()
        self.grid_size = grid_size
        if occupancy_grid is None:
            self.occupancy_grid = np.zeros(grid_size, dtype=bool)
        else:
            assert occupancy_grid.shape == grid_size
            self.occupancy_grid = occupancy_grid

        self.km_per_cell = km_per_cell

        # self.site_grid

    @property
    def shape(self):
        return self.grid_size

    def set_root(self, root):
        super(GridWorld, self).set_root(root)
        self.occupancy_grid |= root.cells

    # @property
    # def occupancy_grid(self):
    #     all_grids = [s.cells for s in self.sites]
    #     return np.any(all_grids, axis=0)

    def free_space(self):
        return ~self.occupancy_grid

    def stop_condition(self):
        if np.random.random() < 0.2:
            return np.all(self.occupancy_grid)


def plot_gridtree(tree, colors, img=None):
    h, w = tree.grid_size

    if img is None:
        img = 255 * np.ones((h, w, 3), dtype=int)

    for i, state in enumerate(tree.iter_descendants()):
        img[state.cells] = colors[i]

    plt.imshow(img, origin='lower')

    for parent, child in tree.iter_edges():
        plot_edge(parent, child, no_arrow=False, lw=0.2, color='k')

    return img


def init_bantu_simulation():
    P_GROW_DISTR = beta(1., 1.).rvs

    init_grid_path = 'data/africa_bantu_grid.png'
    img = sp.misc.imread(init_grid_path, flatten=True)
    img_color = sp.misc.imread(init_grid_path)
    img = img[:, 200:450]
    img_color = img_color[:, 200:450]
    occupancy_grid = (img < 255)

    world = GridWorld(img.shape, occupancy_grid=occupancy_grid)

    i, j = 85, 50
    a = np.zeros_like(img, dtype=bool)
    a[i, j] = True
    s0 = GridState(world, a, P_GROW_DISTR, (400, 800))
    world.set_root(s0)

    return world, s0, img_color[:, :, :3]


def init_empty_simulation(grid_size, p_grow_distr, min_margin = 50,
                          split_size_range=(45,50), km_per_cell=100.):

    # Init world
    world = GridWorld(grid_size)

    # Choose a random grid point and set as start-state
    a = np.zeros(grid_size, dtype=bool)
    w, h = grid_size
    i = np.random.randint(min_margin, w - min_margin)
    j = np.random.randint(min_margin, h - min_margin)
    a[i, j] = True
    s0 = GridState(world, a, p_grow_distr, split_size_range, km_per_cell=km_per_cell)
    world.set_root(s0)

    # Grow zone to initial size
    initial_size = np.random.uniform(*split_size_range)
    for _ in range(initial_size-1):
        s0.grow()

    img = np.ones((*world.grid_size, 3)).astype(int) * 255

    return world, s0, img


def filter_angles(x, y, min_angle=0, max_angle=2*np.pi):
    angles = np.arctan2(y, x) % (2*np.pi)
    return np.logical_and(min_angle <= angles, angles <= max_angle)

def filter_norm(x, y, max_norm):
    norms = np.hypot(x, y)
    return norms <= max_norm


def init_cone_simulation(grid_size, p_grow_distr, cone_angle=5.5,
                         split_size_range=(45, 50), km_per_cell=100.):
    H, W = grid_size
    cx = (W-1) / 2
    cy = (H-1) / 2

    # Init world
    world = GridWorld(grid_size, km_per_cell=km_per_cell)
    y, x = np.mgrid[:H, :W]
    x = x - cx
    y = y - cy
    world.occupancy_grid = ~filter_angles(x, y, min_angle=0, max_angle=cone_angle)
    world.occupancy_grid |= ~filter_norm(x, y, min(cx, cy))
    img = np.array([1 - world.occupancy_grid] * 3).transpose((1, 2, 0)) * 255

    # Choose a random grid point and set as start-state
    a = np.zeros(grid_size, dtype=bool)
    w, h = grid_size
    i = int(np.ceil(w / 2))
    j = int(np.ceil(h / 2))
    a[i, j] = True
    s0 = GridState(world, a, p_grow_distr, split_size_range=split_size_range)
    world.set_root(s0)

    # Grow zone to initial size
    initial_size = np.random.randint(*split_size_range)
    for _ in range(initial_size-1):
        s0.grow()

    return world, s0, img


def animate_grid_world():
    P_GROW_DISTR = beta(1., 1.).rvs
    world, s0, img = init_cone_simulation((90, 160), p_grow_distr=P_GROW_DISTR,
                                          cone_angle=np.random.random()*np.pi*2,
                                          split_size_range=(45,50))

    fig, ax = plt.subplots()
    # fig.set_size_inches(*world.grid_size)
    img_plt = plt.imshow(img)
    edges_plotted = []

    def update(i_frame):
        global states, img, ax

        for _ in range(5):
            for i, s in enumerate(list(world.sites)):
                color = COLORS_RGB[i]
                img[s.cells] = color
                # if s.name.startswith('society_00'):
                #     img[s.cells] = color
                # else:
                #     img[s.cells] = gamma_transform(color, .1)

                s.step()

        img_plt.set_array(img)
        if i_frame % 10 == 0:
            for parent, child in s0.iter_edges():
                if (parent, child) in edges_plotted:
                    continue
                if len(child.children) or child.stuck:
                    plot_edge(parent, child, ax=ax, no_arrow=False, lw=0.2, color='k')
                    edges_plotted.append((parent, child))

        return img_plt,

    writer = animation.FFMpegFileWriter(fps=15, bitrate=5000)
    anim = animation.FuncAnimation(fig, update, frames=2000, interval=20, repeat=False)
    plt.axis("off")
    plt.tight_layout(pad=0.)
    # anim.save('results/grid_simulation.mp4', writer=writer, dpi=6)
    plt.show()
