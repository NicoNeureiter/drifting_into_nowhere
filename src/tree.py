#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

from src.util import remove_whitespace, find


class Node(object):

    """Node class with children, to recursively define a tree, and a set of attributes.

    Attributes:
        length (float): The branch length of the edge leading to the node.
        name (str): The name of the node (often empty for internal nodes).
        children (List[Node]): Children of this node.
        parent (Node): Paren node (None for the root of a tree).
        attributes (dict): A dictionary of additional custom attributes.
        location_attribute (str): Defines the ´attributes´ dict-key for the location attribute.
    """



    def __init__(self, length, name='', children=None, parent=None, attributes=None,
                 location_attribute='location'):
        self.length = length
        self.name = name
        self.children = [] or children
        self.parent = parent
        self.attributes = {} or attributes
        self.location_attribute = location_attribute

        for child in children:
            child.parent = self

    @property
    def tree_size(self):
        size = 1
        for c in self.children:
            size += c.tree_size
        return size

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_attribute_type(self, key, Type):
        self.attributes[key] = Type(self.attributes[key])
        for child in self.children:
            child.set_attribute_type(key, Type)

    def set_location_attribute(self, location_attribute):
        self.location_attribute = location_attribute
        for child in self.children:
            child.set_location_attribute(location_attribute)

    @staticmethod
    def from_newick(newick):
        """Create a tree from the Newick representation.

        Args:
            newick (str): Newick tree.

        Returns:
            Node: The parsed tree.
        """
        newick = remove_whitespace(newick)
        node, _ = parse_node(newick)
        return node

    def get_location(self, location_key='location'):
        location_key += '%i'
        location_median_key = location_key + '_median'

        if (location_key % 1) in self.attributes:
            x = self[location_key % 2]
            y = self[location_key % 1]
        elif (location_median_key % 1) in self.attributes:
            x = self[location_median_key % 2]
            y = self[location_median_key % 1]
        else:
            print(location_key % 1)
            print(self.attributes)
            raise ValueError

        return np.array([x, y])

    @property
    def location(self):
        return self.get_location(self.location_attribute)

    def get_edges(self):
        edges = []
        for c in self.children:
            edges.append((self, c))
            edges += c.get_edges()

        return edges

    def remove_nodes_by_name(self, names):
        was_leaf = (len(self.children) == 0)

        # Recursively remove nodes
        for c in self.children.copy():
            if c.name in names:
                self.children.remove(c)
                print('Removed:', c.name)
            else:
                c.remove_nodes_by_name(names)

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
                self.copy_other_node(c)
            else:
                self_idx = p.children.index(self)
                p.children[self_idx] = c

            c.length += self.length

    def to_newick(self):
        if self.children:
            children = ','.join([c.to_newick() for c in self.children])
            core = '(%s)' % children
        else:
            core = self.name

        attr_str = ''
        if self.attributes:
            attr_str = ','.join('%s=%s' % kv for kv in self.attributes.items())
            attr_str = '[&%s]' % attr_str

        return '{core}{attrs}:{len}'.format(core=core, attrs=attr_str, len=self.length)

    def copy_other_node(self, other):
        self.name = other.name
        self.length = other.length
        self.parent = other.parent
        self.attributes = other.attributes
        self.location_attribute = other.location_attribute

        self.children = []
        for c in other.children:
            self.add_child(c)

    def __getitem__(self, key):
        return self.attributes[key]

    def __repr__(self):
        return 'node_' + self.name


def parse_node(s: str):
    """Parse a string in Newick format into a Node object. The Newick string
    might be partially parsed already, i.e. a suffix of a full Newick tree.

    Args:
        s (str): The (partial) Newick string to be parsed.

    Returns:
        Node: The parsed Node object.
        str: The remaining (unparsed) Newick string.
    """
    name = ''
    children = []

    if s.startswith('('):
        """Parse internal node"""
        # Parse children
        while s.startswith('(') or s.startswith(','):
            s = s[1:]
            child, s = parse_node(s)
            children.append(child)

        assert s.startswith(')'), '"%s"' % s
        s = s[1:]

    else:
        """Parse leaf node"""
        name_stop = min(find(s, '['), find(s, ':'))
        name = s[:name_stop]
        s = s[name_stop:]

    attributes, s = parse_attributes(s)

    length, s = parse_length(s)

    node = Node(length, name=name, children=children, attributes=attributes)
    return node, s


def parse_attributes(s):
    if not s.startswith('[&'):
        return {}, s

    s = s[2:]

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

    assert s.startswith(':')
    s = s[1:]

    end = min(find(s, ','), find(s, ')'))
    length = float(s[:end])
    s = s[end:]

    return length, s


def parse_value(s):
    # TODO do in a clean way
    try:
        return eval(s)
    except (NameError, SyntaxError):
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

    root = Node.from_newick(s)

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


if __name__ == '__main__':
    test_parse_length()
    test_parse_attributes()
    test_newick()
