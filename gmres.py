#!/usr/bin/env python

# -*- coding: utf8 -*-

import numpy as np
from numpy import linalg as LA

def gmres(A, b, eps=1e-16, iteration=None, x_0=None):
    if x_0 is None:
        x_0 = np.zeros(len(b))
    if iteration is None:
        iteration = len(b)
    r = A.dot(x_0) - b
    beta = LA.norm(r, 2)
    v = [r / beta]
    h = np.zeros([], dtype=np.float32) # TODO: initialization
    w = []
    k = 0
    while k < iteration:
        w.append(A.dot(v))
        for j in range(k):
            h[j][k] = v[j].transpose().dot(w)
            w -= h[j][k] * v[j]
        h[k + 1][k] = LA.norm(w)
        v.append(w / h[k + 1][k])
    # TODO: finish it
