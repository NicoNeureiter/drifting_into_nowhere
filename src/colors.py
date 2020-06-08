#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
from webcolors import hex_to_rgb, rgb_to_hex
import numpy as np

from src.util import grey


# COLORS
PINK = (0.95, 0.15, 0.8)
TURQUOISE = (0, 0.8, 0.9)
COLOR_ROOT_EST = (0.75, 0.2, 0.0)
COLOR_ROOT_TRUE = (0., 0.5, 0.6)
COLOR_PATH = grey(0.5)
COLOR_SCATTER = 'orange'

class Cycler(object):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        n = len(self.data)
        return self.data[i % n]


_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
           '#ffe999', '#802a66', '#058090']

# COLORS = Cycler(_COLORS)
# COLORS = [plt.get_cmap('viridis')(i) for i in range(0,250,10)]
COLORS = [plt.get_cmap('tab20b')(i) for i in range(20)]
COLORS_RGB = Cycler([hex_to_rgb(c) for c in _COLORS])
GREY_TONES = Cycler([(int(255*x),)*3 for x in np.linspace(0.7,0.95, 10)])

def gamma_transform(rgb, gamma):
    r,g,b = rgb
    rgb = tuple(int(255*(v/255)**gamma) for v in (r,g,b))
    return rgb
