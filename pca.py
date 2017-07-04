#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os


class PCA:
    def __init__(self):
        pass

    def fit(self, A, numpc):
        A = (A - np.mean(A.T, axis=1)).T
        latent, coeff = np.linalg.eig(np.cov(A))
        idx = np.argsort(latent)
        idx = idx[::-1]

        coeff = coeff[:, idx]
        latent = latent[idx]
        coeff = coeff[:, range(numpc)]

        self.latent = latent
        self.coeff = coeff

    def transform(self, A):
        A = (A - np.mean(A.T, axis=1)).T
        return np.real((self.coeff.T @ A).T)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, 'coeff.npy'), self.coeff)
        np.save(os.path.join(dirname, 'latent.npy'), self.latent)

    def load(self, dirname):
        self.coeff = np.load(os.path.join(dirname, 'coeff.npy'))
        self.latent = np.load(os.path.join(dirname, 'latent.npy'))
