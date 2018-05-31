#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perlin noise implementation.
https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1
Licensed under ISC
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import matplotlib.pyplot as plt


def perlin(x,y):
    # permutation table
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) # FIX1: I was using n10 instead of n01
    return lerp(x1,x2,v) # FIX2: I also had to reverse x1 and x2 here

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def sigmoid(x, mu=0., s=1.):
    return 1 / (1 + np.exp(-(x - mu) / s))


class Landscape(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    from scipy import signal
    from skimage.filters import gaussian as gauss_filter
    import random
    seed = random.randint(0, 1e9)
    np.random.seed(seed)
    print(seed)

    res = 5000

    # img = np.zeros((res, res))
    b = sigmoid(signal.hann(res), mu=0., s=0.1)
    img = 0.2 * np.minimum(b[None, :], b[:, None]) - 0.1
    # plt.imshow(img, origin='upper')
    # plt.show()

    b = signal.hann(res) ** 3.
    img += 0.2 * (b[None, :] + b[:, None])
    # plt.imshow(img, origin='upper')
    # plt.show()


    for i in range(1, 7):

        lin = np.linspace(0, 2*i, res, endpoint=False)
        x,y = np.meshgrid(lin, lin) # FIX3: I thought I had to invert x and y here but it was a
        # mistake

        img += (3. / (2 + i)) * perlin(x, y)

    # img = gauss_filter(img, sigma=10.)

    # img = sigmoid(img, mu=-0.08, s=0.1)
    # img += 0.4 * ((img > 0) - 0.3)
    img = 0.1 + 0.8 * sigmoid(img, s=0.1)**2.
    print(np.min(img), np.max(img))


    plt.imshow(img, origin='upper', cmap=plt.get_cmap('gist_earth'), vmin=0, vmax=1, alpha=.8)
    # plt.imshow(img, origin='upper', cmap=plt.get_cmap('terrain'))  #, alpha=.8)
    plt.show()

    # plt.hist(img.flatten())
    # plt.show()
