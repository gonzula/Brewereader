#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def pca(A, B=None, numpc=None):
    A = (A - np.mean(A.T, axis=1)).T
    latent, coeff = np.linalg.eig(np.cov(A))
    idx = np.argsort(latent)
    idx = idx[::-1]

    coeff = coeff[:, idx]
    latent = latent[idx]
    coeff = coeff[:, range(numpc)]
    A = (coeff.T @ A).T

    if B is not None:
        B = (B - np.mean(B.T, axis=1)).T
        B = (coeff.T @ B).T

        return np.real(A), np.real(B)

    return np.real(A)
