#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

import image_utils as iu

import os

if __name__ == '__main__':
    img = cv2.imread('images_set/pestana.png')

    horizontal, vertical = iu.find_lines(img)

    horizontal = iu.merge_lines(horizontal, img.shape)
    vertical = iu.merge_lines(vertical, img.shape)

    top = min(horizontal, key=lambda line: iu.mid_point(line, img.shape)[1])
    bottom = max(horizontal, key=lambda line: iu.mid_point(line, img.shape)[1])
    left = min(vertical, key=lambda line: iu.mid_point(line, img.shape)[0])
    right = max(vertical, key=lambda line: iu.mid_point(line, img.shape)[0])

    img = iu.unwarp(img, top, bottom, left, right)

    horizontal, vertical = iu.find_lines(img)

    horizontal = iu.merge_lines(horizontal, img.shape)
    vertical = iu.merge_lines(vertical, img.shape)

    #  Check if the walls exists
    if min(rho for rho, theta in horizontal) > 60:
        horizontal = np.append(horizontal, [[0, np.pi/2]], axis=0)
    if abs(img.shape[0] - max(rho for rho, theta in horizontal)) > 60:
        horizontal = np.append(horizontal, [[img.shape[0], np.pi/2]], axis=0)
    if min(rho for rho, theta in vertical) > 60:
        vertical = np.append(vertical, [[0, 0]], axis=0)
    if abs(img.shape[1] - max(rho for rho, theta in vertical)) > 60:
        vertical = np.append(vertical, [[img.shape[1], 0]], axis=0)

    horizontal = sorted(
            horizontal,
            key=lambda line: iu.mid_point(line, img.shape)[1])
    vertical = sorted(
            vertical,
            key=lambda line: iu.mid_point(line, img.shape)[0])
    # for line in horizontal + vertical:
    #     rho, theta = line
    #     pt1, pt2 = iu.cvt_line(rho, theta, img.shape)
    #     cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_8)

    pprint(horizontal)
    pprint(vertical)

    print(len(horizontal))
    print(len(vertical))

    # for i in range(1, len(horizontal) - 1):
    #     for j in range(4, len(vertical) - 2):
    #         cell = iu.unwarp(
    #                 img,
    #                 horizontal[i],
    #                 horizontal[i + 1],
    #                 vertical[j],
    #                 vertical[j + 1],
    #                 )
    #         cv2.imwrite(f'cells/{i}_{j-4}.png', cell)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')  # clear x- and y-axes
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
