#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sys import float_info

import time


def clip(x, up, low):
    if up < low:
        up, low = low, up
    return max(low, min(up, x))


def cvt_line(rho, theta, shape):
    r, t = rho, theta
    h, w = shape[:2]
    h -= 1
    w -= 1
    sin = np.sin
    cos = np.cos
    if abs(cos(t)) > cos(np.radians(45)):
        x1 = r/cos(t)
        x2 = (rho - h * sin(t))/cos(t)
        x1 = clip(x1, 0, w)
        x2 = clip(x2, 0, w)

        if abs(sin(t)) > float_info.epsilon:  # 0
            y1 = (r - x1 * cos(t))/sin(t)
            y2 = (r - x2 * cos(t))/sin(t)
        else:
            y1 = 0
            y2 = h
        y1 = clip(y1, 0, h)
        y2 = clip(y2, 0, h)
    else:
        y1 = r/sin(t)
        y2 = (r - w * cos(t))/sin(t)
        y1 = clip(y1, 0, h)
        y2 = clip(y2, 0, h)

        if abs(cos(t)) > float_info.epsilon:  # 0
            x1 = (rho - y1 * sin(t))/cos(t)
            x2 = (rho - y2 * sin(t))/cos(t)
        else:
            x1 = 0
            x2 = w
        x1 = clip(x1, 0, w)
        x2 = clip(x2, 0, w)

    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))

    return (x1, y1), (x2, y2)


def mid_point(line, shape):
    rho, theta = line
    pt1, pt2 = cvt_line(rho, theta, shape)

    x1, y1 = pt1
    x2, y2 = pt2

    return (x1 + x2)/2, (y1 + y2)/2


def line_dist(l1, l2, shape):
    m1 = mid_point(l1, shape)
    m2 = mid_point(l2, shape)
    return np.sqrt(
            np.power(m1[0] - m2[0], 2) +
            np.power(m1[1] - m2[1], 2)
            )


def intersections(topEdge, bottomEdge, leftEdge, rightEdge, shape):
    height = shape[0]
    width = shape[1]

    if abs(leftEdge[1]) > float_info.epsilon:
        left1 = 0, leftEdge[0]/np.sin(leftEdge[1])
        left2 = width, -width/np.tan(leftEdge[1]) + left1[1]
    else:
        left1 = leftEdge[0]/np.cos(leftEdge[1]), 0
        left2 = left1[0] - height * np.tan(leftEdge[1]), height

    if abs(rightEdge[1]) > float_info.epsilon:
        right1 = 0, rightEdge[0]/np.sin(rightEdge[1])
        right2 = width, -width/np.sin(rightEdge[1]) + right1[1]
    else:
        right1 = rightEdge[0]/np.cos(rightEdge[1]), 0
        right2 = right1[0] - height * np.tan(rightEdge[1]), height

    bottom1 = 0, bottomEdge[0]/np.sin(bottomEdge[1])
    bottom2 = width, -width/np.tan(bottomEdge[1]) + bottom1[1]

    top1 = 0, topEdge[0]/np.sin(topEdge[1])
    top2 = width, -width/np.tan(topEdge[1]) + top1[1]

    leftA = left2[1] - left1[1]
    leftB = left1[0] - left2[0]
    leftC = leftA * left1[0] + leftB * left1[1]

    rightA = right2[1] - right1[1]
    rightB = right1[0] - right2[0]
    rightC = rightA * right1[0] + rightB * right1[1]

    topA = top2[1] - top1[1]
    topB = top1[0] - top2[0]
    topC = topA * top1[0] + topB * top1[1]

    bottomA = bottom2[1] - bottom1[1]
    bottomB = bottom1[0] - bottom2[0]
    bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

    detTopLeft = leftA * topB - leftB * topA
    ptTopLeft = (
            (topB * leftC - leftB * topC)/detTopLeft,
            (leftA * topC - topA * leftC)/detTopLeft,
            )

    detTopRight = rightA * topB - rightB * topA
    ptTopRight = (
            (topB * rightC - rightB * topC)/detTopRight,
            (rightA * topC - topA * rightC)/detTopRight,
            )

    detBottomRight = rightA * bottomB - rightB * bottomA
    ptBottomRight = (
            (bottomB * rightC - rightB * bottomC)/detBottomRight,
            (rightA * bottomC - bottomA * rightC)/detBottomRight,
            )

    detBottomLeft = leftA * bottomB - leftB * bottomA
    ptBottomLeft = (
            (bottomB * leftC - leftB * bottomC)/detBottomLeft,
            (leftA * bottomC - bottomA * leftC)/detBottomLeft,
            )

    return ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft


def unwarp(
        img,
        topEdge,
        bottomEdge,
        leftEdge,
        rightEdge,
        group=False,
        maximize=True,
        dest_size=None,
        ):

    shape = img.shape

    if group:
        start = time.time()
        if maximize:
            ptTopLeft = (np.inf, np.inf)
            ptTopRight  = (-np.inf, np.inf)
            ptBottomLeft = (np.inf, -np.inf)
            ptBottomRight = (-np.inf, -np.inf)
        else:
            ptTopLeft = (-np.inf, -np.inf)
            ptTopRight  = (np.inf, -np.inf)
            ptBottomLeft = (-np.inf, np.inf)
            ptBottomRight = (np.inf, np.inf)
        for t in topEdge:
            for b in bottomEdge:
                for l in leftEdge:
                    for r in rightEdge:
                        tl, tr, br, bl = intersections(
                                t,
                                b,
                                l,
                                r,
                                shape)
                        if maximize:
                            ptTopLeft = (
                                    min(ptTopLeft[0], tl[0]),
                                    min(ptTopLeft[1], tl[1]),
                                    )
                            ptTopRight = (
                                    max(ptTopRight[0], tr[0]),
                                    min(ptTopRight[1], tr[1]),
                                    )
                            ptBottomLeft = (
                                    min(ptBottomLeft[0], bl[0]),
                                    max(ptBottomLeft[1], bl[1]),
                                    )
                            ptBottomRight = (
                                    max(ptBottomRight[0], br[0]),
                                    max(ptBottomRight[1], br[1]),
                                    )
                        else:
                            ptTopLeft = (
                                    max(ptTopLeft[0], tl[0]),
                                    max(ptTopLeft[1], tl[1]),
                                    )
                            ptTopRight = (
                                    min(ptTopRight[0], tr[0]),
                                    max(ptTopRight[1], tr[1]),
                                    )
                            ptBottomLeft = (
                                    max(ptBottomLeft[0], bl[0]),
                                    min(ptBottomLeft[1], bl[1]),
                                    )
                            ptBottomRight = (
                                    min(ptBottomRight[0], br[0]),
                                    min(ptBottomRight[1], br[1]),
                                    )

        end = time.time() - start
        print(f'took {end} secods to find intersections')
    else:
        ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft = intersections(
                topEdge,
                bottomEdge,
                leftEdge,
                rightEdge,
                shape)

    if dest_size is None:
        def dist(p1, p2):
            return np.sqrt(
                np.power(p1[0] - p2[0], 2) +
                np.power(p1[1] - p2[1], 2)
                )

        width = max(
                dist(ptBottomLeft, ptBottomRight),
                dist(ptTopLeft, ptTopRight))
        height = max(
                dist(ptTopLeft, ptBottomLeft),
                dist(ptTopRight, ptBottomRight))
        width = int(width)
        height = int(height)
    else:
        width, height = dest_size

    src = [ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft]
    dst = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
            ]

    src = np.asarray(src).astype('float32')
    dst = np.asarray(dst).astype('float32')

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (width, height))


def find_lines(img, acc_threshold=0.25):
    if len(img.shape) == 3 and img.shape[2] == 3:  # if it's color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            5,
            2)

    img = cv2.bitwise_not(img)

    # thresh = 127
    # edges = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    # edges = cv2.Canny(blur, 500, 500, apertureSize=3)

    should_erode = True
    if should_erode:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        img = cv2.erode(img, element)

    theta = np.pi/2000
    angle_threshold = 2
    horizontal = cv2.HoughLines(
            img,
            1,
            theta,
            int(acc_threshold * img.shape[1]),
            min_theta=np.radians(90 - angle_threshold),
            max_theta=np.radians(90 + angle_threshold))
    vertical = cv2.HoughLines(
            img,
            1,
            theta,
            int(acc_threshold * img.shape[0]),
            min_theta=np.radians(-angle_threshold),
            max_theta=np.radians(angle_threshold),
            )

    horizontal = list(horizontal) if horizontal is not None else []
    vertical = list(vertical) if vertical is not None else []

    horizontal = [line[0] for line in horizontal]
    vertical = [line[0] for line in vertical]

    horizontal = np.asarray(horizontal)
    vertical = np.asarray(vertical)

    return horizontal, vertical


def group_lines(lines, shape, direction, merge_distance=40):

    def dist(l1, l2):  # auxiliary function
        if direction == 'horizontal':
            return abs(mid_point(l1, shape)[1] - mid_point(l2, shape)[1])
        if direction == 'vertical':
            return abs(mid_point(l1, shape)[0] - mid_point(l2, shape)[0])
        if direction == 'both':
            return line_dist(l1, l2, shape)

    bins = []
    for line in lines:
        m = mid_point(line, shape)
        nearest_bin = None
        for bin in bins:
            nearest_line_dist = min(dist(line, l) for l in bin)
            if nearest_line_dist < merge_distance:
                nearest_bin = bin
                break
        else:
            nearest_bin = []
            bins.append(nearest_bin)
        nearest_bin.append(line)

    bins = sorted(
            bins,
            key=lambda bin: np.average([rho for rho, theta in bin])
            )

    return bins


def merge_lines(bins):
    lines = []
    for bin in bins:
        lines.append((
            np.median([rho for rho, _ in bin]),
            np.median([theta for _, theta in bin]),
            ))

    return lines
