# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch.utils.data import TensorDataset, DataLoader


class TVEMDataLoader(DataLoader):
    def __init__(self, data: TensorDataset, *args, **kwargs):
        """General TVEM DataLoader class. Derived from torch.utils.data.DataLoader.

        TVEMDataLoader is constructed exactly the same way as pytorch's DataLoader,
        but it restricts possible input datasets to TensorDataset, and when looped
        over it yields a tuple containing the indeces of the datapoints in each batch
        as well as the actual datapoints for each tensor in the input TensorDataset.

        :param data: a TensorDataset from which to load the data.
        :param args: forwarded to pytorch's DataLoader.
        :param kwargs: forwarded to pytorch's DataLoader.
        """
        N = data.tensors[0].shape[0]
        super().__init__(TensorDataset(to.arange(N), *data.tensors), *args, **kwargs)
