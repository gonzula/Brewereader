#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import image_utils as iu

import os

if __name__ == '__main__':
    img = cv2.imread('example.png')

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
    if min(rho for rho, theta in horizontal) > 5:
        horizontal.append((0, np.pi/2))
    if abs(img.shape[0] - max(rho for rho, theta in horizontal)) > 5:
        horizontal.append((img.shape[0], np.pi/2))
    if min(rho for rho, theta in vertical) > 5:
        vertical.append((0, 0))
    if abs(img.shape[1] - max(rho for rho, theta in vertical)) > 5:
        vertical.append((img.shape[1], 0))

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

    coord = (6, 6)

    cell = iu.unwarp(
            img,
            horizontal[coord[0]],
            horizontal[coord[0] + 1],
            vertical[coord[1]],
            vertical[coord[1] + 1],
            )

    output = cell
    cv2.imwrite('cell.png', output)
