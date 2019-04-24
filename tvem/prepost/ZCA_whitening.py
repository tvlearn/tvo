# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
#
# Author: Georgios Exarchakis

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import array2d, as_float_array
from scipy.linalg import eigh
import numpy as np


class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=0.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, var=0.95, y=None):
        X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        eigs, eigv = eigh(np.dot(X.T, X) / n_samples + self.bias * np.identity(n_features))
        inds = np.argsort(eigs)[::-1]

        eigs = eigs[inds]
        eigv = eigv[:, inds]
        neigs = eigs / np.sum(eigs)
        nc = np.arange(eigs.shape[0])[np.cumsum(neigs) >= var][0]
        eigs = eigs[:nc]
        eigv = eigv[:, :nc]

        components = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)
        self.components_ = components

        # Order the explained variance from greatest to least
        self.explained_variance_ = eigs  # [inds]
        return self

    def transform(self, X):
        X = array2d(X)
        if self.mean_ is not None:
            X -= self.mean_
        X_transformed = np.dot(X, self.components_)
        return X_transformed


if __name__ == "__main__":
    # Example usage: ZCA-whitening of natural image patches

    # Initialize variables
    N = 200000  # number of data points
    var = 95  # variance kept in percentile
    p = 16  # patch size
    ifname = "natims_conv_1700.npy"  # input file
    ims = np.load(ifname)

    # crop patches
    data = np.zeros((N, p ** 2), dtype="float64")
    indw = np.random.randint(0, ims.shape[2] - p, N)
    indh = np.random.randint(0, ims.shape[1] - p, N)
    indi = np.random.randint(0, ims.shape[0], N)
    for i, ind in enumerate(indi):
        data[i] = ims[ind, indh[i] : indh[i] + p, indw[i] : indw[i] + p].reshape(p ** 2)

    # apply ZCA-whitening
    zca = ZCA()
    zca.fit(data, var=var / 100.0)
    wdata = zca.transform(data)
