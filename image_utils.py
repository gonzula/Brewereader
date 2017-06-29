#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def cvt_line(rho, theta, shape):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(round(x0 + shape[1] * -b))
    y1 = int(round(y0 + shape[0] * a))

    x2 = int(round(x0 - shape[1] * -b))
    y2 = int(round(y0 - shape[0] * a))

    pt1 = x1, y1
    pt2 = x2, y2

    return pt1, pt2


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


def unwarp(img, topEdge, bottomEdge, leftEdge, rightEdge):
    shape = img.shape
    height = shape[0]
    width = shape[1]

    if leftEdge[1] != 0:
        left1 = 0, leftEdge[0]/np.sin(leftEdge[1])
        left2 = width, -width/np.tan(leftEdge[1]) + left1[1]
    else:
        left1 = leftEdge[0]/np.cos(leftEdge[1]), 0
        left2 = left1[0] - height * np.tan(leftEdge[1]), height

    if rightEdge[1] != 0:
        right1 = 0, rightEdge[0]/np.sin(rightEdge[1])
        right2 = width, -width/np.sin(rightEdge[1]) + right1[1]
    else:
        right1 = rightEdge[0]/np.cos(rightEdge[1]), 0
        right2 = right1.x - height * np.tan(rightEdge[1]), height

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


def find_lines(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # if it's color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        acc_threshold = 800
    else:
        acc_threshold = 1200

    theta = np.pi/2000
    angle_threshold = 2
    horizontal = cv2.HoughLines(
            img,
            1,
            theta,
            acc_threshold,
            min_theta=np.radians(90 - angle_threshold),
            max_theta=np.radians(90 + angle_threshold))
    vertical = cv2.HoughLines(
            img,
            1,
            theta,
            acc_threshold,
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


def merge_lines(lines, shape, merge_distance=40):
    bins = []
    for line in lines:
        m = mid_point(line, shape)
        nearest_bin = None
        for bin in bins:
            nearest_line_dist = min(line_dist(line, l, shape) for l in bin)
            if nearest_line_dist < merge_distance:
                nearest_bin = bin
                break
        else:
            nearest_bin = []
            bins.append(nearest_bin)
        nearest_bin.append(line)

    lines = []
    for bin in bins:
        lines.append((
            np.median([rho for rho, _ in bin]),
            np.median([theta for _, theta in bin]),
            ))

    return lines
