#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import mlp
import pca


X = np.load('dataset_X.npy')
Y = np.load('dataset_Y.npy')

X = pca.pca(X, numpc=300)

train_size = int(0.8 * X.shape[0])

shuffle_ids = np.arange(X.shape[0])
np.random.shuffle(shuffle_ids)
X = X[shuffle_ids]
Y = Y[shuffle_ids]

train_X = X[:train_size]
train_Y = Y[:train_size]
test_X = X[train_size:]
test_Y = Y[train_size:]

model = mlp.MLP([train_X.shape[1], 200, train_Y.shape[1]])
model.train(train_X, train_Y)
print('finished training')

correct = 0
for idx, x, expected in zip(range(test_X.shape[0]), test_X, test_Y):
    obtained = model.solve(x)
    obtained = np.argmax(obtained)
    expected = np.argmax(expected)
    if obtained == expected:
        correct += 1

    if idx % 50 == 0:
        print('%6.2f%%' % (idx/len(test_X) * 100))

model.save('%f.mlp' % (correct/test_X.shape[0]))
print()
print('Result: ', correct/len(test_X) * 100, '%', sep='')
