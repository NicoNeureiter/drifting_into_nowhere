#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

from colorsys import rgb_to_hls, hls_to_rgb
from webcolors import hex_to_rgb, rgb_to_hex
import numpy as np

from src.util import grey


# COLORS
PINK = (0.95, 0.15, 0.8)
TURQUOISE = (0, 0.8, 0.9)
COLOR_ROOT_EST = (0.6, 0.0, 0.3)
COLOR_ROOT_TRUE = (0., 0.5, 0.6)
COLOR_PATH = grey(0.6)
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
COLORS = Cycler(_COLORS)
COLORS_RGB = Cycler([hex_to_rgb(c) for c in _COLORS])
GREY_TONES = Cycler([(int(255*x),)*3 for x in np.linspace(0.7,0.95, 10)])

def gamma_transform(rgb, gamma):
    r,g,b = rgb
    rgb = tuple(int(255*(v/255)**gamma) for v in (r,g,b))
    return rgb


# def to_hls(c):
#     rgb = hex_to_rgb(c)
#     return colorsys.rgb_to_hls(*rgb)
#
#
# def hls_to_hex(h, l, s):
#     r, g, b = colorsys.hls_to_rgb(h, l, s)
#     rgb = int(255 * r), int(255 * g), int(255 * b)
#     return rgb_to_hex(rgb)
#
#
# def set_lightness(color, l):
#     h, _, s = to_hls(color)
#     return hls_to_hex(h, l, s)
#
# def set_saturation(color, s):
#     h, l, _ = to_hls(color)
#     return hls_to_hex(h, l, s)
#
# colors = []
# for c in COLORS:
#     h, l, s = to_hls(c)
#     colors.append(hls_to_hex(h, 0.3, 0.8))
# for c in COLORS:
#     h, l, s = to_hls(c)
#     colors.append(hls_to_hex(h, 0.5, 0.4))
# for c in COLORS:
#     h, l, s = to_hls(c)
#     colors.append(hls_to_hex(h, 0.6, 0.6))
#
# COLORS = colors*3