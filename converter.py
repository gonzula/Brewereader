#!/usr/bin/env python3
# -*- coding: utf8 -*-

import cv2
import numpy as np

import os

dimension = 42 * 28

dirname = 'cells'
files = [fname for fname in os.listdir(dirname) if fname.endswith('.png')]
files = sorted(files, key=lambda f: int(f.split('_')[0]))

X = np.empty((len(files), dimension))
Y = np.zeros((len(files), 11)).astype(int)

for idx, fname in enumerate(files):
    img = cv2.imread(os.path.join(dirname, fname))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape(dimension).astype('float64')
    img /= 255
    y = int(fname.split('_')[0])

    X[idx] = img
    Y[idx, y] = 1

np.save('dataset_X.npy', X)
np.save('dataset_Y.npy', Y)
