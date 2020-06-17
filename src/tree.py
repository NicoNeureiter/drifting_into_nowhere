#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import logging
from copy import copy

import numpy as np

from src.util import (remove_whitespace, find, read_locations_file,
                      read_alignment_file, StringTemplate, str_concat_array, norm)
from src.beast_xml_templates import *
from shapely.geometry import Point, Polygon


class Tree(object):

    """Tree node class with children, to recursively define a tree, and a set of attributes.

    Attributes:
        length (float): The branch length of the edge leading to the treee node.
        name (str): The name of the current node (often empty for internal nodes).
        children (List[Tree]): Children of this node.
        parent (Tree): Paren node (None for the root of a tree).
        attributes (dict): A dictionary of additional custom attributes.
        _location (np.array or iterable or None): The private attribute for the
            geo-location (accessed via property `Tree.location`).
    """

    def __init__(self, length, name='', children=None, parent=None,
                 attributes=None, location=None, alignment=None):
        self.length = length
        self.name = name
        self.children = children or []

        self.parent = parent
        self.attributes = attributes or {}
        self._location = location
        self.alignment = alignment or [0]

        for child in self.children:
            child.parent = self

        self.color = None

    def get_subtree(self, subtree_path: list):
        """Compute the subtree defined the indices in subtree path.

        Args:
            subtree_path (list[int]): A list of indices, which specify the child
                at each node along the path to the requested subtree.
                E.g.: [0,1,1] would be the second child of the second child of
                the first child of this tree.

        Returns:
            Tree: The specified subtree.
        """
        if len(subtree_path) == 0:
            return self
        else:
            c_idx, *c_subtree_path = subtree_path
            return self.children[c_idx].get_subtree(c_subtree_path )

    @property
    def location(self):
        if self._location is not None:
            return self._location
        else:
            return self.get_location_from_attributes()

    @location.setter
    def location(self, location):
        self._location = location

    @property
    def alignment(self):
        return self._alignment

    @alignment.setter
    def alignment(self, alignment):
        # self._alignment = np.asarray(alignment)
        self._alignment = alignment

    # TODO Refactor: Should not be a property (it's not O(1))
    @property
    def depth(self):
        """The depth is defined as the length of the path up from the current
        node to the root (sum of lengths).

        Returns:

        """
        if self.parent is None:
            return self.length
        else:
            return self.parent.depth + self.length

    def height(self):
        """The height is defined as the longest path down from the current node
        to a leaf (sum of lengths).

        Returns:
            float: The height of the current node.
        """
        if self.is_leaf():
            return 0
        else:
            return max(c.height() + c.length for c in self.children)

    def tree_size(self):
        """
        Returns:
            int: The number of nodes (internal + leafs) of the current tree.
        """
        size = 1
        for c in self.children:
            size += c.tree_size()
        return size

    def n_leafs(self):
        """
        Returns:
            int: The number of leafs in the current tree.
        """
        size = 1 if self.is_leaf() else 0
        for c in self.children:
            size += c.n_leafs()
        return size

    def is_leaf(self):
        """
        Returns:
            bool: Is this node a leaf?
        """
        return len(self.children) == 0

    def is_root(self):
        """

        Returns:
            bool: IS this node the root?
        """
        return self.parent is None

    def root(self):
        """Get the root of the current tree (iterate parent of the parent of...)
        Returns:
              Tree: Root of the current node.
        """
        if self.is_root():
            return self
        else:
            return self.parent.root()

    def n_fossils(self):
        """Count the fossils in the current tree. A fossil is a leaf that is not
        at the maximal depth (i.e. `contemporary`).

        Returns:
            int: Number of fossils.
        """
        max_depth = self.root().height()
        return len([n for n in self.iter_leafs() if n.depth < max_depth])

    def add_child(self, child):
        """Add a child to the current node."""
        self.children.append(child)
        child.parent = self

    def get_descendant_locations(self):
        """Get the locations of all descendant of the current tree as a numpy
        array.

        Returns:
            np.array: Locations of all descendants.
                shape: (tree_size, 2)
        """
        return np.array([node.location for node in self.iter_descendants()])

    def get_leaf_locations(self):
        """Get the locations of all leaf nodes in the current tree as a numpy
        array.

        Returns:
            np.array: Locations of all leafs.
                shape: (n_leafs, 2)
        """
        return np.array([node.location for node in self.iter_leafs()])

    def small_child(self):
        """Get the smallest child in the list (measured by Tree.tree_size).

        Returns:
            Tree: The smallest child.
        """
        return min(self.children, key=self.__class__.tree_size)

    def big_child(self):
        """Get the biggest child in the list (measured by Tree.tree_size).

        Returns:
            Tree: The biggest child.
        """
        return max(self.children, key=self.__class__.tree_size)

    def set_attribute_type(self, key, Type):
        self.attributes[key] = Type(self.attributes[key])
        for child in self.children:
            child.set_attribute_type(key, Type)

    def set_location_attribute(self, location_attribute):
        self.location_attribute = location_attribute
        for child in self.children:
            child.set_location_attribute(location_attribute)

    def rescale_by(self, factor):
        """Rescale the height of the tree by the given factor (equally on each
        branch).

        Args:
            factor (float): Scaling factor.
        """
        for node in self.iter_descendants():
            node.length *= factor

    def rescale_to(self, target_height):
        """Rescale the height of the tree to the given ´target_height´ (equally
        on each branch).

        Args:
            target_height (float): The target height.
        """
        factor = target_height / self.height()
        self.rescale_by(factor)

    def drop_fossils(self, max_age=0.):
        """Remove all fossils older than ´max_age´."""
        if max_age == np.inf:
            return

        leafs = list(self.iter_leafs())
        t_final = self.root().height()

        too_old = []
        for node in leafs:
            if node.depth < t_final - max_age:
                too_old.append(node)

        self.remove_nodes(too_old)


    @staticmethod
    def from_newick(newick, location_key='location', swap_xy=False, with_attributes=True,
                    translate=None):
        """Create a tree from the Newick representation.

        Args:
            newick (str): Newick tree.

        Returns:
            Tree: The parsed tree.
        """
        if translate is None:
            translate = lambda x: x
        if isinstance(translate, dict):
            translate_dict = translate
            translate = lambda x: translate_dict.get(x, x)

        newick = remove_whitespace(newick)

        hpd_pattern = r'%hpd_1={'
        p_hpd = None
        newick_lower = newick.lower()
        if hpd_pattern in newick_lower:
            before_pattern, _, _ = newick_lower.partition(hpd_pattern)
            _, _, p_hpd_str = before_pattern.rpartition('_')
            assert len(p_hpd_str) in [1, 2]
            p_hpd = int(p_hpd_str)

        tree, _ = parse_tree(newick, location_key=location_key, swap_xy=swap_xy,
                             with_attributes=with_attributes, name_mapping=translate)
        if p_hpd is not None:
            tree.p_hpd = p_hpd

        return tree

    def get_location_from_attributes(self, location_key='location'):
        """Extract the location from the attributes dict (if present) and return
        it as a np.array (or None)

        Kwargs:
            location_key (str): the suffix of the dictionary key for the location
                in the attributes dictionary.
        Returns:
            np.array or None: The extracted location of the node.
        """

        if location_key in self.attributes:
            location_str = self.attributes[location_key]
            location_str = location_str.replace('{', '(') \
                                       .replace('}', ')')
            x, y = eval(location_str)

        else:
            location_key += '%i'
            location_median_key = location_key + '_median'

            if (location_key % 1) in self.attributes:
                x = self[location_key % 1]
                y = self[location_key % 2]
            elif (location_median_key % 1) in self.attributes:
                x = self[location_median_key % 1]
                y = self[location_median_key % 2]
            else:
                return None
        return np.array([x, y])

    def get_hpd(self, p_hpd, location_key='location'):
        """Extract the HPD from the attributes dict."""
        attr_keys = list(self.attributes.keys())
        hpd_key_template = '{location_key}{i_axis}_{p_hpd}%HPD_{i_polygon}'
        hpd_key_template = hpd_key_template.format(location_key=location_key,
                                                   p_hpd=p_hpd,
                                                   i_axis='{i_axis}',
                                                   i_polygon='{i_polygon}')

        i = 1
        hpd_key_x = hpd_key_template.format(i_axis=1, i_polygon=i)
        hpd_key_y = hpd_key_template.format(i_axis=2, i_polygon=i)
        polygons = []
        while hpd_key_x in attr_keys:
            hpd_x_str = self.attributes[hpd_key_x][1:-1]
            hpd_y_str = self.attributes[hpd_key_y][1:-1]
            hpd_x = map(float, hpd_x_str.split(','))
            hpd_y = map(float, hpd_y_str.split(','))

            poly = Polygon(zip(hpd_x, hpd_y))
            polygons.append(poly)

            i += 1
            hpd_key_x = hpd_key_template.format(i_axis=1, i_polygon=i)
            hpd_key_y = hpd_key_template.format(i_axis=2, i_polygon=i)

        if len(polygons) == 0:
            print(attr_keys)
            logging.warning('No HPD polygon found!')

        return polygons

    def root_in_hpd(self, root, p_hpd, location_key='location'):
        """Check whether the given root location is covered by the HPD in the
        node attributes."""

        # Ensure that root is a Point object
        if not isinstance(root, Point):
            assert len(root) == 2
            root = Point(root[0], root[1])

        # Check whether any of the HPD polygons () cover
        for polygon in self.get_hpd(p_hpd, location_key=location_key):
            if polygon.contains(root):
                return True

        return False

    def iter_edges(self):
        """Iterate over all edges in the tree."""
        for c in self.children:
            yield self, c
            yield from c.iter_edges()

    def remove_nodes_by_name(self, names):
        """Remove nodes with the given names from the tree, preserving a valid
        tree topology and branch lengths."""
        remove_list = []
        for c in self.iter_descendants():
            if c.name in names:
                remove_list.append(c)

        self.remove_nodes(remove_list)

    def remove_nodes(self, remove_list):
        """Remove nodes specified in ´remove_list´ from the tree, preserving a
        valid tree topology and branch lengths."""
        # TODO should we allow to remove internal nodes without dropping the whole subtree?

        # for node in self.iter_leafs():
        #     if node in remove_list:
        #         p = node.parent
        #         p.children.remove(node)

        was_leaf = (len(self.children) == 0)

        # print(remove_list)

        # Recursively remove nodes
        for c in self.children.copy():
            if c in remove_list:
                # print(':' + c.name)
                self.children.remove(c)
            else:
                c.remove_nodes(remove_list)

        if was_leaf:
            return

        if len(self.children) == 0:
            # Clean up nodes that became leafs
            # Explanation: Internal that became leaves usually are not intended to
            # (e.g. they don't have a location)
            self.parent.children.remove(self)

        elif len(self.children) == 1:
            # Clean up single-child nodes
            p = self.parent
            c = self.children[0]

            if p is None:
                # self is root -> single child becomes the root
                # prev_len = self.length
                self.copy_other_node(c)
                # self.length += prev_len
                self.length = 0
                self.parent = None
            else:
                # Replace self by c in parent.children
                # (self is skipped -> length must be adapted)
                c_prev_height = c.depth

                self_idx = p.children.index(self)
                p.children[self_idx] = c
                c.parent = p
                c.length += self.length

                assert np.isclose(c.depth, c_prev_height)

    def to_newick(self, write_attributes=True, translate=None):
        """Compute a Newick string representation of the tree."""
        if translate is None:
            translate = lambda x: x
        if isinstance(translate, dict):
            translate_dict = translate
            translate = lambda x: translate_dict.get(x, x)

        if self.children:
            child_newicks = []
            for c in self.children:
                c_newick = c.to_newick(write_attributes=write_attributes,
                                             translate=translate)
                child_newicks.append(c_newick)
            core = '(' + ','.join(child_newicks) + ')'
            core += translate(self.name)
        else:
            core = translate(self.name)

        attr_str = ''
        if self.attributes and write_attributes:
            attr_str = ','.join('%s=%s' % kv for kv in self.attributes.items())
            attr_str = '[&%s]' % attr_str

        return '{core}{attrs}:{len}'.format(core=core, attrs=attr_str, len=self.length)

    def to_nexus(self, fname, write_attributes=True):
        taxa = []
        for n in self.iter_descendants():
            if n.name == '':
                assert not n.is_leaf()
            else:
                taxa.append(n.name)

        translate = dict(map(reversed, enumerate(taxa)))
        print(translate)
        taxa_lines = ['\t\t%s %s' % (i, name) for i, name in enumerate(taxa)]
        tree_line = '\ttree TREE = %s;' % self.to_newick(translate=translate,
                                                         write_attributes=write_attributes)

        lines = [
            '#NEXUS', '',
            # 'begin taxa;',
            # '\tdimension ntax=%i;' % len(taxa),
            # '\ttaxlabels',
            # *taxa_lines,
            # ';', 'end;', '',
            'begin trees;',
            '\ttranslate',
            ',\n'.join(taxa_lines) + ';',
            tree_line,
            'end;']

        with open(fname, 'w') as nexus:
            nexus.write('\n'.join(lines))

    def copy_other_node(self, other):
        # TODO Iterate over attrs?
        self.name = other.name
        self.length = other.length
        self.parent = other.parent
        self.attributes = other.attributes
        self.alignment = other.alignment
        self._location = other._location

        self.children = []
        for c in other.children:
            self.add_child(c)

    def copy(self):
        other = Tree(length=self.length, name=self.name,
                     attributes=copy(self.attributes),
                     location=copy(self.location),
                     alignment=copy(self.alignment))

        for c in self.children:
            other.add_child(c.copy())

        return other

    def _format_location(self):
        # print(self.name)
        x, y = self.location
        return LOCATION_TEMPLATE.format(id=self.name, x=x, y=y, age=self.depth)

    def _format_alignment(self):
        alignment_str = str_concat_array(self.alignment)
        return FEATURES_TEMPLATE.format(id=self.name, features=alignment_str)

    def _format_tree_locations(self):
        return ''.join(map(Tree._format_location, self.iter_leafs()))

    def _format_tree_alignments(self):
        return ''.join(map(Tree._format_alignment, self.iter_leafs()))

    def write_beast_xml(self, output_path, chain_length, root=None,
                        movement_model='rrw', diffusion_on_a_sphere=False,
                        jitter=0.01, adapt_height=False, adapt_tree=False,
                        drift_prior_std=0.1):
        if movement_model == 'rrw':
            template_path = RRW_XML_TEMPLATE_PATH
        elif movement_model == 'rdrw':
            template_path = RDRW_XML_TEMPLATE_PATH
        elif movement_model == 'cdrw':
            template_path = CDRW_XML_TEMPLATE_PATH
        elif movement_model == 'brownian':
            template_path = BROWNIAN_XML_TEMPLATE_PATH
        else:
            raise ValueError('Unknown movement_model `%s`' % movement_model)

        with open(template_path, 'r') as xml_template_file:
            xml_template = StringTemplate(xml_template_file.read())

        # Write newick tree
        newick_str = self.to_newick(write_attributes=False)
        xml_template.tree = newick_str

        # Set locations and features
        xml_template.locations = self._format_tree_locations()
        xml_template.features = self._format_tree_alignments()

        # Fix root / don't fix root by setting set steep / flat prior
        if root is None:
            root = [0., 0.]
            root_precision = 1e-6
        else:
            root_precision = 1e6

        xml_template.root_x = root[0]
        xml_template.root_y = root[1]
        xml_template.root_precision = root_precision

        # Set parameters
        xml_template.chain_length = chain_length
        xml_template.ntax = self.n_leafs()
        xml_template.nchar = len(self.alignment)
        xml_template.jitter = jitter
        xml_template.spherical = SPHERICAL if diffusion_on_a_sphere else ''
        xml_template.tree_operators = TREE_OPERATORS if adapt_tree else ''
        xml_template.height_operators = HEIGHT_OPERATORS if adapt_height else ''
        # xml_template.n_dim_loc = 2 * self.tree_size()
        if movement_model in ('rdrw', 'cdrw'):
            xml_template.drift_prior_std = drift_prior_std

        with open(output_path, 'w') as beast_xml_file:
            beast_xml_file.write(
                xml_template.fill()
            )

    def load_locations_from_csv(self, csv_path, swap_xy=False):
        locations, _ = read_locations_file(csv_path, swap_xy=swap_xy)
        locations = {k.lower():v for k,v in locations.items()}
        for node in self.iter_descendants():
            if node.name in locations:
                node._location = locations[node.name]
            # else:
            #     logging.warning('No location found for node "%s"' % node.name)

    def load_alignment_from_csv(self, csv_path):
        alignments = read_alignment_file(csv_path)
        for node in self.iter_descendants():
            if node.name in alignments:
                node.alignment = alignments[node.name]
            else:
                logging.warning('No alignment found for node "%s"' % node.name)

    def binarize(self):
        # Ensure that self has at most 2 children
        if len(self.children) > 2:
            new_grandchildren = self.children[1:]
            new_child = Tree(0, children=new_grandchildren, parent=self,
                             attributes=self.attributes, location=self.location,
                             alignment=self.alignment)
            self.children = [self.children[0], new_child]

        for c in self.children:
            c.binarize()

    def iter_descendants(self):
        """Iterate over all nodes in the tree.

        Yields:
            Tree: Each node of the tree in depth-first order.
        """
        yield self
        for c in self.children:
            yield from c.iter_descendants()

    def get_descendants(self):
        """Get all nodes in the tree as a list.

        Returns:
            list[Tree]: List of all nodes in depth-first order.
        """
        return list(self.iter_descendants())

    def iter_leafs(self):
        """Iterate over all leaf nodes in the tree.

        Yields:
            Tree: Each leaf node of the tree.
        """
        if self.is_leaf():
            yield self
        for c in self.children:
            yield from c.iter_leafs()

    def get_leafs(self):
        """Get all leaf nodes in the tree as a list.

        Returns:
            list[Tree]: List of all leaf nodes.
        """
        return list(self.iter_leafs())

    def iter_clades(self, max_size):
        """Iterate over all maximal subtrees with n_leafs <= max_size.

        Args:
            max_size (int): The maximum number of leafs in a clade.

        Yields:
            Tree: A clade with n_leafs <= max_size
        """
        if self.n_leafs() <= max_size:
            yield self
        for child in self.children:
            yield from child.iter_clades(max_size=max_size)

    def get_clades(self, max_size, min_size=1):
        """Get all maximal subtrees with n_leafs <= max_size as a list.

        Args:
            max_size (int):  The maximum number of leafs in a clade.

        Returns:
            list[Tree]: The list of all clades with n_leafs <= max_size.
        """
        clades = list(self.iter_clades(max_size=max_size))
        if min_size > 1:
            return [t for t in clades if t.n_leafs() >= min_size]

    def iter_clades_at_height(self, height):
        h = self.height()
        l = self.length
        if h + l < height:
            return
        elif h < height:
            yield self
        else:
            for c in self.children:
                yield from c.iter_clades_at_height(height)

    def get_clades_at_height(self, height):
        return list(self.iter_clades_at_height(height))

    def get_phylo_dist_mat(self):
        if self.is_leaf():
            return np.zeros((1, 1))

        left, right = self.children
        n = self.n_leafs()
        n_left = left.n_leafs()

        X_left = left.get_phylo_dist_mat()
        X_right = right.get_phylo_dist_mat()

        X = np.ones((n, n)) * (2 * self.height())
        X[:n_left, :n_left] = X_left
        X[n_left:, n_left:] = X_right

        return X

    def get_loc_dist_mat(self):
        locs = self.get_leaf_locations()
        diffs = locs[:, np.newaxis, :] - locs[np.newaxis, :, :]
        return np.linalg.norm(diffs, axis=-1)

    def __getitem__(self, key):
        return self.attributes[key]

    def __repr__(self):
        return self.name


def node_imbalance(node: Tree, ret_weight=True):
    assert len(node.children) <= 2, node.children

    size = node.n_leafs()

    if (len(node.children) < 2) or (size < 4):
        I = np.nan
    else:
        c1, c2 = node.children

        bigger = max(c1.n_leafs(), c2.n_leafs())
        m = np.ceil(size / 2)

        I = (bigger - m) / (size - m - 1)

    if not ret_weight:
        return I
    else:
        if size % 2 == 1:   # odd
            w = 1
        else:               # even
            w = 1 - 1 / size
            if I == 0:
                w *= 2
            else:
                assert (I is np.nan) or (I > 0), I
        return I, w


def tree_imbalance(root, max_depth=None, weight_by_age=False):
    """A tree imbalance score according to:
    "Purvis and Agapow: Phylogeny imbalance and taxonomic level"
    
    Args:
        root: 
        max_depth: 

    Returns:

    """
    nodes = list(root.iter_descendants())
    if max_depth is not None:
        nodes = list(filter(lambda n: n.depth < max_depth, nodes))
    assert len(nodes) > 0  # At least the root itself should be in the list

    node_imbalances, weights = np.array([node_imbalance(n) for n in nodes]).T

    if weight_by_age:
        node_heights = np.array([n.height() for n in nodes])
        weights *= node_heights / root.height()

    not_na = np.isfinite(node_imbalances)
    return np.nansum(weights*node_imbalances) / np.sum(weights[not_na])


def parse_tree(s, location_key='location', swap_xy=False, with_attributes=True,
               name_mapping=None):
    """Parse a string in Newick format into a Tree object. The Newick string
    might be partially parsed already, i.e. a suffix of a full Newick tree.

    Args:
        s (str): The (partial) Newick string to be parsed.

    Returns:
        Tree: The parsed Tree object.
        str: The remaining (unparsed) Newick string.
    """
    if name_mapping is None:
        name_mapping = lambda x: x

    name = ''
    children = []

    if s.startswith('('):
        """Parse internal node"""
        # Parse children
        while s.startswith('(') or s.startswith(','):
            s = s[1:]
            child, s = parse_tree(s, location_key=location_key, swap_xy=swap_xy,
                                  with_attributes=with_attributes, name_mapping=name_mapping)
            children.append(child)

        assert s.startswith(')'), '"%s"' % s
        s = s[1:]

        if not s[0] in ':[);':
            if with_attributes:
                name_stop = min(find(s, '['), find(s, ':'))
            else:
                name_stop = find(s, ':')

            name = name_mapping(s[:name_stop])
            s = s[name_stop:]


    else:
        """Parse leaf node"""
        if with_attributes:
            name_stop = min(find(s, '['), find(s, ':'))
        else:
            name_stop = find(s, ':')

        name = name_mapping(s[:name_stop])
        s = s[name_stop:]

    if with_attributes:
        attributes, s = parse_attributes(s)
    else:
        attributes = None

    length, s = parse_length(s)
    tree = Tree(length, name=name, children=children, attributes=attributes)

    tree._location = tree.get_location_from_attributes(location_key)
    if swap_xy:
        tree._location = tree._location[::-1]

    return tree, s


def parse_attributes(s):
    if not s.startswith('['):
        return {}, s
    s = s[1:]
    if s.startswith('&'):
        s = s[1:]

    attrs = {}
    k1, _, s = s.partition('=')
    while find(s, '=') < find(s, ']'):
        v1_k2, _, s = s.partition('=')
        v1, _, k2 = v1_k2.rpartition(',')

        attrs[k1] = parse_value(v1)

        k1 = k2

    v1, _, s = s.partition(']')
    attrs[k1] = parse_value(v1)

    return attrs, s


def parse_length(s):
    if s.startswith(';'):
        return 0., s

    assert s.startswith(':'), (len(s), s)
    s = s[1:]

    end = min(find(s, ','), find(s, ')'))

    slen = s[:end]
    if '@' in slen:
        slen = slen[:find(s, '@')]
    if ';' in slen:
        slen = slen[:find(s, ';')]

    try:
        length = float(slen)
    except Exception as e:
        print(s[:end])
        print(s)
        raise
    s = s[end:]
    return length, s


def parse_value(s):
    try:
        return float(s)
    # except (NameError, SyntaxError):
    except ValueError:
        return s


""" TESTING """


def test_parse_length():
    s = ':3.14)[&a=1,b=2];'

    l, s = parse_length(s)

    assert l == 3.14
    assert s == ')[&a=1,b=2];'

    print('Test SUCCESSFUL: parse_length')


def test_parse_attributes():
    s = '[&a=1,b=2];'

    attrs, s = parse_attributes(s)

    assert len(attrs) == 2
    assert attrs['a'] == 1
    assert attrs['b'] == 2
    assert s == ';'

    print('Test SUCCESSFUL: parse_attributes')


def test_newick():
    s = '\t(A[&name = a]:1.2, (B[&name=b]:3.4,C[&tmp=x, name=c]:5.6)[&name=internal]:7.8)[&name=root];\n'

    root = Tree.from_newick(s)

    assert len(root.children) == 2
    a, internal = root.children

    assert len(internal.children) == 2
    b, c = internal.children

    assert root['name'] == 'root'
    assert internal['name'] == 'internal'
    assert a['name'] == 'a'
    assert b['name'] == 'b'
    assert c['name'] == 'c'

    assert root.name == ''
    assert internal.name == ''
    assert a.name == 'A'
    assert b.name == 'B'
    assert c.name == 'C'

    assert root.length == 0.
    assert internal.length == 7.8
    assert a.length == 1.2
    assert b.length == 3.4
    assert c.length == 5.6

    print('Test SUCCESSFUL: parse_newick')


def test_tree_imbalance():
    pass


# if __name__ == '__main__':
    # test_parse_length()
    # test_parse_attributes()
    # test_newick()
    # import matplotlib
    # print(matplotlib.matplotlib_fname())


def get_edge_heights(parent, child):
    return (parent.height() + child.height()) / 2.


def get_old_edges(parent, child, threshold=250.):
    return (parent.depth + child.depth) / 2. > threshold


def angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def get_edge_diff_rate(parent, child):
    step = child.location - parent.location
    diff_rate = norm(step) / child.length
    return diff_rate


def rename_nodes(tree, name_mapping):
    for node in tree.iter_descendants():
        assert node.name in name_mapping or node.name == '', node.name
        if node.name in name_mapping:
            node.name = name_mapping[node.name]


def assign_lcations(tree, loc_mapping):
    for node in tree.iter_descendants():
        if node.name in loc_mapping:
            node.location = loc_mapping[node.name]


def naive_location_reconstruction(tree):
    if tree.is_leaf():
        assert tree.location is not None
        return

    if tree.location is None:
        c_locs = []
        for c in tree.children:
            naive_location_reconstruction(c)
            c_locs.append(c.location)
        tree.location = np.mean(c_locs, axis=0)
