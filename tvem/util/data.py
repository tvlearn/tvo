# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch.utils.data import TensorDataset, DataLoader


class TVEMDataLoader(DataLoader):
    def __init__(self, data: to.Tensor, *args, **kwargs):
        """TVEM DataLoader class. Derived from torch.utils.data.DataLoader.

        TVEMDataLoader is constructed exactly the same way as pytorch's DataLoader,
        but it restricts possible input datasets to a single to.Tensor.
        All other arguments are forwarded to pytorch's DataLoader.
        When iterated over, TVEMDataLoader yields a tuple containing the indeces of
        the datapoints in each batch as well as the actual datapoints for each
        tensor in the input Tensor.

        :param data: Tensor containing the input dataset. Must have exactly two dimensions (N,D).
        :param args: forwarded to pytorch's DataLoader.
        :param kwargs: forwarded to pytorch's DataLoader.
        """
        N = data.shape[0]
        if data.dtype is not to.uint8:
            self.precision = data.dtype
        super().__init__(TensorDataset(to.arange(N), data), *args, **kwargs)
