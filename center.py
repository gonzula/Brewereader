#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import os
from pprint import pprint

import matplotlib.pyplot as plt

dstdirname = 'centered'
dirname = 'cells'
files = [fname for fname in os.listdir(dirname) if fname.endswith('.png')]
files = sorted(files, key=lambda f: int(f.split('_')[0]))

for fname in files:
    img = cv2.imread(os.path.join(dirname, fname))

    height,width = img.shape[:2]
    threshold = 150
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.dilate(img, element,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(255,255,255))

    # fig, ax = plt.subplots()
    # fig.canvas.set_window_title(fname)
    # ax.imshow(img)
    # ax.axis('off')
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()

    # fig, ax = plt.subplots()
    # fig.canvas.set_window_title(fname)
    # ax.imshow(bw)
    # ax.axis('off')
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()

    # continue

    bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)[1]
    min_point = (np.inf, np.inf)
    max_point = (-np.inf, -np.inf)

    black_pix = False
    for y, row in enumerate(bw):
        for x, pix in enumerate(row):
            if not pix:
                min_point = (
                        min(min_point[0], x),
                        min(min_point[1], y),
                        )
                max_point = (
                        max(max_point[0], x),
                        max(max_point[1], y),
                        )
                black_pix = True
    if not black_pix:
        continue

    min_point = np.asarray(min_point)
    max_point = np.asarray(max_point)
    center = (min_point + max_point) / 2
    center = np.rint(center)
    print(fname, center)
    image_center = np.asarray([width, height]) / 2
    center -= image_center

    M = [
            [1, 0, -center[0]],
            [0, 1, -center[1]],
    ]
    M = np.matrix(M)
    img = cv2.warpAffine(
            img, M, (width, height),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
            )
    cv2.imwrite(os.path.join(dstdirname, fname), img)
