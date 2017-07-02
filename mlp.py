#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Carlos Eduardo Ayoub Fialho  # 7563703

import numpy as np
from sys import stderr
import os


class MLP:
    def __init__(self, sizes):
        self.layers = []

        if sizes:
            prev_size = sizes[0]
            for size in sizes[1:]:
                layer = np.random.rand(size, prev_size + 1) - 0.5
                self.layers.append(layer)
                prev_size = size

    def solve(self, x_p, only_result=True):  # forward mlp
        f_nets = []
        df_dnets = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                fnet1 = np.append(x_p, 1)  # append 1 (bias, Ɵ) to match θ
            else:
                fnet1 = np.append(f_nets[-1], 1)  # get last f_net calculated

            f_net = np.empty(len(layer))
            df_dnet = np.empty(len(layer))
            for j, neuron in enumerate(layer):
                net = fnet1 @ neuron

                f_net[j] = self.f(net)
                df_dnet[j] = self.df(net)

            f_nets.append(f_net)
            df_dnets.append(df_dnet)

        if only_result:
            return f_nets[-1]
        return f_nets, df_dnets

    def train(self, X, Y, eta=0.1, threshold=1e-2):

        squared_error = 2 * threshold  # initial value to enter loop
        try:
            count = 0
            min_squared_error = 1000.0
            while squared_error > threshold:
                squared_error = 0.0

                for x_p, y_p in zip(X, Y):
                    f_nets, df_dnets = self.solve(x_p, only_result=False)

                    o_p = f_nets[-1]  # get last layer's (output layer) result
                    #                 # which is mlp answer to x_p

                    delta_p = y_p - o_p  # get the difference from the expected
                    #                    # answer

                    squared_error += sum(np.power(delta_p, 2))  # add to error

                    layers = list(zip(
                        range(len(self.layers)),
                        self.layers, f_nets, df_dnets))
                    deltas = [None] * len(layers)

                    # Calculating layers delta
                    for idx, layer, f_net, df_dnet in reversed(list(layers)):
                        if idx == len(self.layers) - 1:  # is output layer
                            deltas[idx] = delta_p * df_dnet
                        else:
                            # delete last column to match dimension
                            _lay = np.delete(self.layers[idx + 1], -1, 1)
                            deltas[idx] = df_dnet * (deltas[idx + 1] @ _lay)

                    # Updating layers
                    for idx, layer, f_net, df_net in reversed(layers):
                        _delta = deltas[idx][np.newaxis].T
                        if idx > 0:
                            _f_net = np.append(f_nets[idx - 1], 1)[np.newaxis]
                        else:  # is first hidden layer
                            _f_net = np.append(x_p, 1)[np.newaxis]
                        self.layers[idx] += eta * (_delta @ _f_net)

                squared_error /= len(X)

                count += 1
                if squared_error < min_squared_error:
                    min_squared_error = squared_error
                    print(squared_error, 'new min', file=stderr)
                elif count % 1 == 0:
                    print(squared_error, file=stderr)

        except KeyboardInterrupt as e:
            pass

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for idx, layer in enumerate(self.layers, start=1):
            fname = os.path.join(dirname, '%d.layer.npy' % idx)
            np.save(fname, layer)

    @staticmethod
    def load(dirname):
        fnames = [f for f in os.listdir(dirname) if f.endswith('.layer.npy')]
        fnames = sorted(fnames)
        layers = [np.load(os.path.join(dirname, f)) for f in fnames]
        model = MLP([])
        model.layers = layers
        return model

    @staticmethod
    def f(net):
        return 1 / (1 + np.exp(-net))

    @staticmethod
    def df(net):
        fnet = 1 / (1 + np.exp(-net))
        return fnet * (1 - fnet)
