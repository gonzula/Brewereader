#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

import image_utils as iu

import os

if __name__ == '__main__':
    dirname = 'images_set'
    for fname in ['natalia.png', ]:  # 'pestana2.png']:  # os.listdir(dirname):
        if not fname.endswith('.png'):
            continue
        fname = os.path.join(dirname, fname)
        img = cv2.imread(fname)
        print(fname, img.shape)

        horizontal, vertical = iu.find_lines(img)

        horizontal = iu.group_lines(horizontal, img.shape, 'horizontal')
        vertical = iu.group_lines(vertical, img.shape, 'vertical')
        horizontal = iu.merge_lines(horizontal)
        vertical = iu.merge_lines(vertical)

        top = min(horizontal, key=lambda l: iu.mid_point(l, img.shape)[1])
        bottom = max(horizontal, key=lambda l: iu.mid_point(l, img.shape)[1])
        left = min(vertical, key=lambda l: iu.mid_point(l, img.shape)[0])
        right = max(vertical, key=lambda l: iu.mid_point(l, img.shape)[0])

        tl, tr, br, bl = iu.intersections(top, bottom, left, right, img.shape)
        img = iu.unwarp(img, tl, tr, br, bl, offset=-5)

        horizontal, vertical = iu.find_lines(img)

        horizontal = iu.group_lines(horizontal, img.shape, 'horizontal')
        vertical = iu.group_lines(vertical, img.shape, 'vertical')

        # Check if the walls exists
        if min(rho for rho, theta in horizontal[0]) > 20:
            horizontal.insert(
                    0,
                    [np.array([0, np.pi/2])]
                    )
        if abs(img.shape[0] - max(rho for rho, theta in horizontal[-1])) > 20:
            horizontal.append(
                    [np.array([img.shape[0], np.pi/2])]
                    )
        if min(rho for rho, theta in vertical[0]) > 20:
            vertical.insert(
                    0,
                    [np.array([0, 0])]
                    )
        if abs(img.shape[1] - max(rho for rho, theta in vertical[-1])) > 20:
            vertical.append(
                    [np.array([img.shape[1], 0])]
                    )

        horizontal = sorted(
                horizontal,
                key=lambda bin: iu.mid_point(bin[0], img.shape)[1])
        vertical = sorted(
                vertical,
                key=lambda bin: iu.mid_point(bin[0], img.shape)[0])
        # horizontal = sorted(
        #         horizontal,
        #         key=lambda bin: iu.mid_point(bin[0], img.shape)[1])
        # vertical = sorted(
        #         vertical,
        #         key=lambda bin: iu.mid_point(bin[0], img.shape)[0])
        # for bin in horizontal + vertical:
        #     for rho, theta in bin:
        #         pt1, pt2 = iu.cvt_line(rho, theta, img.shape)
        #         cv2.line(img, pt1, pt2, (0, 255, 0), 5, cv2.LINE_AA)


        print('horizontal')
        pprint(horizontal)
        print('vertical')
        pprint(vertical)

        print(len(horizontal))
        print(len(vertical))

        # v = vertical

        # for l in v:
        #     r, t = l
        #     print(iu.cvt_line(r, t, img.shape))
        #     print('\t', iu.mid_point(l, img.shape))
        # print(iu.line_dist(v[-2], v[-3], img.shape))
        # print(iu.line_dist(v[-4], v[-5], img.shape))

        for i in range(1, len(horizontal) - 1):
            for j in range(4, len(vertical) - 2):
                tl, tr, br, bl = iu.intersections(
                        horizontal[i],
                        horizontal[i + 1],
                        vertical[j],
                        vertical[j + 1],
                        img.shape,
                        group=True,
                        )
                cell = iu.unwarp(img, tl, tr, br, bl, offset=-5)

                bw = cell
                if len(bw.shape) == 3 and bw.shape[2] == 3:
                    bw = cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)
                threshold = 127
                bw = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)[1]
                total = cell.shape[0] * cell.shape[1]
                zero = total - cv2.countNonZero(bw)
                if zero/total > 0.5:
                    print(f'skipping {i}_{j-4}')
                    continue

                blur = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                blur = cv2.adaptiveThreshold(
                        blur,
                        255,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,
                        5,
                        2)
                blur = cv2.GaussianBlur(blur, (11, 11), 0)
                cell = blur

                # blur = cv2.GaussianBlur(cell, (11, 11), 0)
                # cv2.imwrite(f'cells/{i}_{j-4}_1.png', cell)
                canny = cv2.Canny(cell, 50, 200)
                # cv2.imwrite(f'cells/{i}_{j-4}_2.png', canny)

                h, v = iu.find_lines(canny, 0.4)
                # cell = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
                wl = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

                if not h:
                    h = [(0, np.pi/2), (cell.shape[0]-1, np.pi/2)]
                if not v:
                    h = [(0, 0), (cell.shape[1]-1, 0)]

                for rho, theta in np.append(h, v, axis=0):
                    pt1, pt2 = iu.cvt_line(rho, theta, cell.shape)
                    cv2.line(wl, pt1, pt2, (0, 255, 0), 1, cv2.LINE_4)
                cv2.imwrite(f'cells/{i}_{j-4}_3.png', wl)

                h = iu.group_lines(
                        h,
                        cell.shape,
                        'horizontal',
                        merge_distance=20)
                v = iu.group_lines(
                        v,
                        cell.shape,
                        'vertical',
                        merge_distance=20)

                tl, tr, br, bl = iu.intersections(
                        h[0],
                        h[-1],
                        v[0],
                        v[-1],
                        cell.shape,
                        group=True,
                        maximize=False,
                        )
                cell = iu.unwarp(
                        cell,
                        tl,
                        tr,
                        br,
                        bl,
                        dest_size=(42, 28),
                        )

                cell = cv2.adaptiveThreshold(
                        cell,
                        255,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,
                        5,
                        2)
                cv2.imwrite(f'cells/{i}_{j-4}_4.png', cell)

                # h = iu.merge_lines(h)
                # v = iu.merge_lines(v)

                # for rho, theta in np.append(h, v, axis=0):
                #     pt1, pt2 = iu.cvt_line(rho, theta, cell.shape)
                #     cv2.line(cell, pt1, pt2, (0, 255, 0), 1, cv2.LINE_4)
                # cv2.imwrite(f'cells/{i}_{j-4}_3.png', cell)

        # output = img
        # cv2.imwrite('img.png', output)
        # os.system('open img.png')
