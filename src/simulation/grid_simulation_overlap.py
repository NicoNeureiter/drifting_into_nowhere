#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import numpy as np
# import scipy as sp
from scipy.ndimage.morphology import binary_dilation
from scipy.stats import beta

from src.simulation.simulation import World, State
from src.util import bernoulli, experiment_preperations

# Plotting imports...
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from src.config import COLORS_RGB
from src.plotting import plot_tree_topology, plot_edge


DEBUG = True


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
                 p_conflict=0., death_rate=0.,
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
        self._death_rate = death_rate
        self.area = np.count_nonzero(self.cells)

        self.is_neighbour = neighbourhood(self.cells)
        # self.neighbours = grid_to_index_tuples(self.is_neighbour)

    # @property
    # def area(self):
    #     return np.count_nonzero(self.cells)

    def valid_index(self, i, j):
        return (0 <= i < self.cells.shape[0]) and (0 <= j < self.cells.shape[1])

    @property
    def location(self):
        cell_indices = grid_to_index_tuples(self.cells)
        mean_index = np.mean(cell_indices, axis=0)[::-1]
        world_center = self.world.center[::-1]
        # print(mean_index, self.world.center, mean_index - world_center)
        return self.world.km_per_cell * (mean_index - world_center)

    @property
    def grid_size(self):
        return self.cells.shape

    def split_probability(self):
        return float(self.area >= self.split_size)

    @property
    def death_rate(self):
        if self.age < 20:
            return 0.
        else:
            return self._death_rate

    def step(self, last_step=False):
        # if not self.stuck:
        if bernoulli(self.p_grow):
            self.grow()

        super(GridState, self).step(last_step=last_step)

    def add_cell(self, i, j):
        self.cells[i, j] = True
        self.world.occupancy_grid[i, j] += 1
        self.area += 1

        self.is_neighbour[i, j] = False
        for i2, j2 in get_neighbours(i, j):
            if self.valid_index(i2, j2):
                if not self.cells[i2, j2]:
                    self.is_neighbour[i2, j2] = True
                    # self.neighbours.append((i2, j2))

    def remove_cell(self, i, j):
        assert self.cells[i, j]
        self.cells[i, j] = False
        self.world.occupancy_grid[i, j] -= 1
        self.area -= 1
        raise Exception('Shrinking areas not implemented yet!')

    def grow(self):
        """ Extend the current area by a random adjacent cell."""

        # Find free neighbouring cells as candidates for expansion
        # neighbour_cells = neighbourhood(self.cells)
        neighbour_cells = self.is_neighbour.copy()

        if bernoulli(1 - self.p_conflict):
            neighbour_cells &= self.world.free_space()
        else:
            neighbour_cells &= self.world.not_full_space()

        candidates = grid_to_index_tuples(neighbour_cells)
        if len(candidates) == 0:
            return

        # Randomly add one of the candidate cells to the site
        n_samples = min(2, len(candidates))
        # n_samples = 1
        for i, j in random.sample(candidates, n_samples):
            self.add_cell(i, j)

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
        child_1.is_neighbour = neighbourhood(child_1.cells)
        child_2.is_neighbour = neighbourhood(child_2.cells)

    def create_child(self):
        i = str(len(self.children))
        child_name = self._name + i

        child = GridState(self.world, self.cells, self.p_grow_distr, self.split_area_range,
                          p_conflict=self.p_conflict, death_rate=self._death_rate,
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

    def __init__(self, grid_size, occupancy_grid=None, max_density=3, km_per_cell=1.):
        super(GridWorld, self).__init__()
        self.grid_size = grid_size
        if occupancy_grid is None:
            self.occupancy_grid = np.zeros(grid_size, dtype=int)
            self.support = np.ones(grid_size, dtype=bool)
        else:
            assert occupancy_grid.shape == grid_size
            self.occupancy_grid = occupancy_grid.copy().astype(int)
            self.support = (occupancy_grid == 0)

        self.max_density = max_density
        self.km_per_cell = km_per_cell

        self.center = np.zeros(2)
        # self.site_grid

    @property
    def shape(self):
        return self.grid_size

    def set_root(self, root):
        super(GridWorld, self).set_root(root)
        self.occupancy_grid |= root.cells

    def set_center(self, center):
        assert len(center) == 2
        self.center = np.asarray(center)

    def recompute_occupancy_grid(self):
        all_grids = [s.cells for s in self.sites]
        self.occupancy_grid = np.sum(all_grids, axis=0)
        self.occupancy_grid[~self.support] = self.max_density

    def free_space(self):
        return self.occupancy_grid == 0

    def not_full_space(self):
        return self.occupancy_grid < self.max_density

    def register_death(self, node):
        super(GridWorld, self).register_death(node)
        self.recompute_occupancy_grid()

    def stop_condition(self):
        # return False
        if np.random.random() < 0.2:

            if np.all(self.occupancy_grid >= self.max_density):
                print('Stop age:', self.sites[0].age)
                return True
            else:
                return False


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


def filter_angles(x, y, min_angle=0, max_angle=2*np.pi):
    angles = np.arctan2(y, x) % (2*np.pi)
    return np.logical_and(min_angle <= angles, angles <= max_angle)


def filter_max_norm(x, y, max_norm):
    norms = np.hypot(x, y)
    return norms <= max_norm


def filter_min_norm(x, y, min_norm):
    norms = np.hypot(x, y)
    return norms >= min_norm



def init_cone_simulation(grid_size, p_grow_distr, cone_angle=5.5,
                         split_size_range=(45, 50), km_per_cell=100., p_conflict=0.,
                         corner_cutoff=5, death_rate=0.0):
    H, W = grid_size
    cx = (W-1) / 2
    cy = (H-1) / 2

    # Init world
    world = GridWorld(grid_size, km_per_cell=km_per_cell)
    y, x = np.mgrid[:H, :W]
    x = x - cx
    y = y - cy
    cone_grid = filter_angles(x, y, min_angle=0, max_angle=cone_angle)
    cone_grid &= filter_max_norm(x, y, min(cx, cy))
    # cone_grid &= filter_min_norm(x, y, corner_cutoff)

    non_empty_cols = np.any(cone_grid, axis=0)
    left = np.where(non_empty_cols)[0][0]
    non_empty_rows = np.any(cone_grid, axis=1)
    bottom = np.where(non_empty_rows)[0][0]
    top = np.where(non_empty_rows)[0][-1]

    cone_grid = cone_grid[bottom:top, left:]
    grid_size = cone_grid.shape

    max_density = 3
    world.occupancy_grid = (max_density * ~cone_grid)
    world.support = cone_grid
    img = np.array([255*world.support] * 3).transpose((1, 2, 0))

    # Choose a random grid point and set as start-state
    a = np.zeros(grid_size, dtype=bool)
    i = int(np.ceil(cx)) - bottom
    j = int(np.ceil(cy)) - left
    a[i, j] = True
    s0 = GridState(world, a, p_grow_distr, split_size_range=split_size_range, p_conflict=p_conflict, death_rate=death_rate)
    world.set_root(s0)
    world.set_center([i, j])

    # Grow zone to initial size
    min_init_size = 20
    a = cone_angle / (2*np.pi)
    initial_size = int(min_init_size + a*split_size_range[0])
    for _ in range(initial_size-1):
        s0.grow()

    # img = np.array([1-world.occupancy_grid]*3).transpose((1,2,0)) * 255

    return world, s0, img


def sample_random_subtree(tree, n_leaves):
    leaves = tree.get_leafs()
    leaves_drop = random.sample(leaves, tree.n_leafs() - n_leaves)
    tree.remove_nodes(leaves_drop)
    assert tree.n_leafs() == n_leaves


if __name__ == '__main__':
    from src.simulation.simulation import run_simulation
    from src.beast_interface import run_beast, run_treeannotator, load_tree_from_nexus
    from src.plotting import plot_tree, plot_hpd, plot_root
    from src.config import COLOR_ROOT_TRUE, COLOR_ROOT_EST
    from src.evaluation import tree_statistics

    WORKING_DIR = 'data/gridworld/'
    XML_PATH = WORKING_DIR + 'nowhere.xml'
    TREE_PATH = WORKING_DIR + 'nowhere.tree'
    exp_dir = experiment_preperations(WORKING_DIR)
    # exp_dir = experiment_preperations(WORKING_DIR, seed=369388681)

    # Analysis Parameters
    CHAIN_LENGTH = 150000
    BURNIN = 20000
    HPD = 80

    # SIMULATION PARAMETER
    N = 350
    KM_PER_CELL = 100.
    P_GROW = 0.9
    P_GROW_N = 80
    p_grow_distr = beta(P_GROW_N * P_GROW, P_GROW_N * (1 - P_GROW)).rvs

    # SPLIT = {.0: 18, .2: 30, .4: 40, .5: 45, .6: 50, .8: 60, 1.: 70}
    # N_STEPS = {.0: 90, .2: 126, .4: 146, .5: 150, .6: 153, .8: 160, 1.: 165}

    SPLIT = {.0:   18,
             .5:   46,
             1.:   70}
    N_STEPS = {.0:  105,
               .5:  166,
               1.:  180,}

    DEATH_RATE = 0

    TREE_SIZE = 150


    for P_CONFLICT in [.0, .5, 1.]:
    # for P_CONFLICT in [.0, .2, .4, .6, .8, 1.]:

        print('P_CONFLICT =', P_CONFLICT)
        split_range = (SPLIT[P_CONFLICT], SPLIT[P_CONFLICT] + 5)

        CONE_ANGLE = 2. * np.pi
        n = int(N / (CONE_ANGLE ** .5))

        sizes = []
        offsets = []
        overlaps = []
        imbalances = []
        sd_deps = []

        for _ in range(3):
            world, root, img = init_cone_simulation(grid_size=(n, n), p_grow_distr=p_grow_distr,
                                                        cone_angle=CONE_ANGLE,
                                                        split_size_range=split_range,
                                                        km_per_cell=KM_PER_CELL, p_conflict=P_CONFLICT,
                                                        death_rate=DEATH_RATE)

            run_simulation(N_STEPS[P_CONFLICT], root, world, condition_on_root=True)
            root.drop_fossils()

            tree = root
            leafs_mean = np.mean(tree.get_leaf_locations(), axis=0)
            leafs_mean_offset = leafs_mean - root.location
            offset = np.hypot(*leafs_mean_offset)

            print('\tn_leaves:', tree.n_leafs())
            print('\tobserved_drift:', leafs_mean_offset)
            print('\tobserved_drift_norm: %.2f' % offset)
            sizes.append(root.n_leafs())
            offsets.append(offset)



            tree_stats = tree_statistics(tree)
            print('\ttree stats:', tree_stats)
            overlaps.append(tree_stats['clade_overlap'])
            imbalances.append(tree_stats['imbalance'])
            sd_deps.append(tree_stats['space_div_dependence'])

        # print()
        print('Mean size = %.2f' % np.mean(sizes))
        print('Mean drift = %.2f' % np.mean(offsets))
        print('Mean overlap = %.2f' % np.mean(overlaps))
        print('Mean imbalance = %.2f' % np.mean(imbalances))
        print('Mean S-D dep. = %.2f' % np.mean(sd_deps))
        print()

    # Show original tree
    # plot_gridtree(tree, colors=COLORS_RGB)
    plot_tree(tree, color='k', lw=0.2)
    plot_root(tree.location, color=COLOR_ROOT_TRUE)

    # Create an XML file as input for the BEAST analysis
    tree.write_beast_xml(output_path=XML_PATH, chain_length=CHAIN_LENGTH, movement_model='rrw')

    # Run BEAST analysis
    run_beast(working_dir=WORKING_DIR)
    run_treeannotator(HPD, BURNIN, working_dir=WORKING_DIR)

    # # Show reconstructed tree
    reconstructed = load_tree_from_nexus(tree_path=TREE_PATH)
    plot_root(reconstructed.location, color=COLOR_ROOT_EST)
    plot_hpd(reconstructed, HPD, alpha=0.5)


    img_gray = np.mean(img, axis=-1)
    plt.imshow(img_gray, origin='lower', cmap='gray', zorder=0)

    plt.axis('off')
    plt.tight_layout(pad=0.)

    plt.show()
